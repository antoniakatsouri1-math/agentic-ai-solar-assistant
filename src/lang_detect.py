"""
Tiny shared helper for picking which hardcoded fallback message to return
(e.g. "no results found") without needing an LLM call - used by agent
nodes that short-circuit before calling the LLM (empty search results,
blocked SQL, etc.), where the response must still match the question's
language.
"""


def looks_greek(text: str) -> bool:
    """True if the text contains Greek script characters."""
    return any("\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff" for ch in text)
