"""
Shared state schema for the LangGraph agent.

Every node in the graph reads from and writes to this single state object.
Keeping it in one place (rather than redefining per-node) is what lets the
router, the agents, and the memory layer all agree on field names.
"""
from typing import List, Optional, TypedDict


class Turn(TypedDict):
    role: str  # "user" | "assistant" | "tool"
    content: str


class Source(TypedDict, total=False):
    title: str
    url: str
    snippet: str
    date: str


class RagChunk(TypedDict, total=False):
    text: str
    source: str
    chunk_index: int
    distance: float


class AgentState(TypedDict, total=False):
    # --- identity / session ---
    conversation_id: str

    # --- current turn ---
    user_input: str          # the message just received
    route: str                # set by the router: weather|rag|sql|general
    response: str              # final natural-language answer for this turn

    # --- memory (Feature 5) ---
    history: List[Turn]        # last N turns, injected before the agent runs
    memory_window: int          # how many turns to keep (default 6)

    # --- grounding / traceability ---
    sources: List[Source]      # for the search agent
    rag_chunks: List[RagChunk]  # retrieved chunks for the rag agent
    sql_query: Optional[str]    # for the sql agent, for transparency
    error: Optional[str]        # set on graceful-failure paths
