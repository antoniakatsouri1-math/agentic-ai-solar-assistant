"""
Web Search Agent (Feature 1).

Retrieves up-to-date external information via Tavily (e.g. current energy
prices, new subsidy announcements, market news), then asks the LLM to
write a grounded answer with inline [n] citations referencing a numbered
source list appended at the end. Falls back gracefully - no hallucination -
when search comes back empty or with nothing sufficiently relevant.
"""
from typing import List

from src.llm import call_llm
from src.state import AgentState, Source
from src.lang_detect import looks_greek
from src.tools.search import web_search

SYSTEM = (
    "You are a research assistant for a solar/renewable-energy equipment "
    "business. You will be given a user question and a numbered list of "
    "web sources (title, url, snippet, date). Write a concise answer using "
    "ONLY information present in the sources. Cite every factual claim "
    "with the matching bracketed number, e.g. [1] or [1][3] for a claim "
    "drawn from multiple sources. Do not invent facts that aren't in the "
    "provided sources, and do not add a sources list yourself - that will "
    "be appended separately. IMPORTANT: respond in the same language the "
    "user asked their question in (Greek question -> Greek answer, "
    "English question -> English answer)."
)

NO_RESULTS_MESSAGE_EL = (
    "Έψαξα στο διαδίκτυο για αυτό αλλά δεν βρήκα αρκετά σχετικά ή "
    "επίκαιρα αποτελέσματα. Δεν θέλω να μαντέψω κάτι που χρειάζεται "
    "τρέχουσα πληροφόρηση - θα μπορούσες να αναδιατυπώσεις την ερώτηση, ή "
    "θες να δοκιμάσω ευρύτερη αναζήτηση;"
)
NO_RESULTS_MESSAGE_EN = (
    "I searched the web for this but didn't find any sufficiently relevant "
    "or up-to-date results. I don't want to guess at something that needs "
    "current information - could you rephrase the query, or would you like "
    "me to try a broader search?"
)


def _format_sources_block(sources: List[Source]) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] {s['title']} ({s['date']}) - {s['url']}\n    {s['snippet']}")
    return "\n".join(lines)


def _format_sources_footer(sources: List[Source]) -> str:
    lines = ["", "Πηγές / Sources:"]
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] {s['title']} - {s['url']}")
    return "\n".join(lines)


def search_node(state: AgentState) -> AgentState:
    query = state["user_input"]
    sources = web_search(query)

    if not sources:
        no_results = NO_RESULTS_MESSAGE_EL if looks_greek(query) else NO_RESULTS_MESSAGE_EN
        return {**state, "response": no_results, "sources": []}

    prompt = f"User question: {query}\n\nSources:\n{_format_sources_block(sources)}"
    answer = call_llm(
        system=SYSTEM,
        user_input=prompt,
        history=state.get("history"),
        max_tokens=600,
    )
    full_response = answer.rstrip() + "\n" + _format_sources_footer(sources)

    return {**state, "response": full_response, "sources": sources}
