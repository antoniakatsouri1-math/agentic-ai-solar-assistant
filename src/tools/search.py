"""
Web search tool (Feature 1) - wraps Tavily's search API.

Returns structured source dicts (title, url, snippet, date) for the search
agent to ground its answer in and cite. Filtering out low-relevance results
here is what lets the agent fall back gracefully instead of hallucinating
when Tavily returns nothing useful.

In this project, search is mainly used for time-sensitive information that
changes faster than the RAG knowledge base can be re-indexed: current
energy prices, new subsidy program announcements, market news.
"""
import os
from typing import List

from tavily import TavilyClient

from src.state import Source

# Tavily relevance score floor (0-1). Results below this are treated as
# noise, not as usable sources.
RELEVANCE_THRESHOLD = 0.3

_client = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str, max_results: int = 5) -> List[Source]:
    """Run a Tavily search and return only results that clear the relevance
    threshold. An empty list is the signal to the caller that there's
    nothing useful to ground an answer in."""
    client = _get_client()
    raw = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
    )

    sources: List[Source] = []
    for item in raw.get("results", []):
        if item.get("score", 0) < RELEVANCE_THRESHOLD:
            continue
        sources.append(
            {
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "snippet": (item.get("content") or "")[:400],
                "date": item.get("published_date") or "n/a",
            }
        )
    return sources
