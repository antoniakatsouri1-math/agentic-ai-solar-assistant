import os
from typing import List, Optional

from groq import Groq
from dotenv import load_dotenv

from src.state import Turn

load_dotenv()

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


def history_to_messages(history: Optional[List[Turn]]) -> List[dict]:
    if not history:
        return []
    messages = []
    for turn in history:
        role = "assistant" if turn["role"] in ("assistant", "tool") else "user"
        messages.append({"role": role, "content": turn["content"]})
    return messages


def call_llm(
    system: str,
    user_input: str,
    history: Optional[List[Turn]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Call a Groq-hosted model with an optional rolling history prepended."""
    messages = [{"role": "system", "content": system}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": user_input})

    resp = _client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
        include_reasoning=False,
    )
    return resp.choices[0].message.content
