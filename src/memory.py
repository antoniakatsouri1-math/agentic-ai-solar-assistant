"""
Conversation Memory & Persistence (Feature 5).

ALL database reads/writes for conversation history live here - agent nodes
never touch SQL directly (they only read `state["history"]`, which this
module populates).

5a. In-session memory: get_recent_turns() returns the last N turns so they
    can be prepended to the prompt sent to whichever agent handles the turn.
5b. Database persistence: SQLite-backed conversations/messages tables so a
    conversation survives an application restart and can be resumed by
    conversation_id.
"""
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv

from src.state import Turn

load_dotenv()

DB_PATH = os.environ.get("DATABASE_PATH", "data/conversations.db")
DEFAULT_MEMORY_WINDOW = int(os.environ.get("MEMORY_WINDOW_TURNS", "6"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content         TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the conversations/messages tables if they don't already
    exist. Safe to call on every app startup - it does not touch existing
    data."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)


def conversation_exists(conversation_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return row is not None


def create_conversation(conversation_id: Optional[str] = None) -> str:
    """Create a new conversation row and return its id. Generates a fresh
    uuid4 if none is supplied."""
    conversation_id = conversation_id or str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, created_at, updated_at) VALUES (?, ?, ?)",
            (conversation_id, now, now),
        )
    return conversation_id


def save_message(conversation_id: str, role: str, content: str) -> None:
    """Store one message and bump the parent conversation's updated_at.
    This is step 1 (user message) and step 5 (agent response) of the
    required workflow."""
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (now, conversation_id),
        )


def get_full_history(conversation_id: str) -> List[Turn]:
    """All messages for a conversation, oldest first. Used when resuming a
    conversation after a restart."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY message_id ASC",
            (conversation_id,),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in rows]


def get_recent_turns(conversation_id: str, n_turns: int = DEFAULT_MEMORY_WINDOW) -> List[Turn]:
    """The last n_turns exchanges (1 turn = 1 user message + 1 assistant
    response = up to 2 messages), oldest-first, ready to prepend to the
    next prompt. This is what step 2-3 of the required workflow
    ('retrieve history -> inject into state') actually returns."""
    limit_messages = max(n_turns * 2, 1)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "ORDER BY message_id DESC LIMIT ?",
            (conversation_id, limit_messages),
        ).fetchall()
    rows.reverse()  # back to chronological order
    return [{"role": r, "content": c} for r, c in rows]
