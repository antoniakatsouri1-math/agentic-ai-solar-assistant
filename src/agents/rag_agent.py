"""
RAG Retrieval Agent (Feature 2).

Retrieves relevant chunks from the persisted ChromaDB knowledge base
(regulatory documents on net-metering, PV licensing, subsidies, VAT, and
energy communities in Greece), then asks the LLM to answer using ONLY
those chunks. Like the search agent, it falls back gracefully instead of
hallucinating when nothing relevant is found.
"""
from typing import List

from src.llm import call_llm
from src.state import AgentState, RagChunk
from src.lang_detect import looks_greek
from src.tools.rag import retrieve

SYSTEM = (
    "You are a regulatory assistant for a solar/renewable-energy equipment "
    "business, specializing in Greek net-metering, PV licensing, subsidy "
    "programs, and VAT rules. You will be given a user question and "
    "numbered excerpts retrieved from internal regulatory documents. "
    "Answer using ONLY information present in the excerpts. Cite the "
    "source document for each claim using the excerpt number, e.g. [1]. "
    "If the excerpts don't actually contain the answer, say so plainly "
    "instead of guessing - do not use outside knowledge. IMPORTANT: "
    "respond in the same language the user asked their question in (Greek "
    "question -> Greek answer, English question -> English answer)."
)

NO_RESULTS_MESSAGE_EN = (
    "I couldn't find anything in the knowledge base relevant to that "
    "question. It may not be covered by the indexed regulatory documents - "
    "could you rephrase, or is this something outside the document set?"
)
NO_RESULTS_MESSAGE_EL = (
    "Δεν βρήκα κάτι σχετικό με αυτή την ερώτηση στη βάση γνώσης. Μπορεί να "
    "μην καλύπτεται από τα ευρετηριασμένα κανονιστικά έγγραφα - θα "
    "μπορούσες να την αναδιατυπώσεις, ή είναι κάτι έξω από το σύνολο "
    "εγγράφων;"
)

# Below this similarity threshold (Chroma returns L2 distance - lower is
# more similar) we treat a chunk as noise rather than a real match.
DISTANCE_THRESHOLD = 30


def _format_chunks_block(chunks: List[RagChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {c['source']}, chunk {c['chunk_index']})\n{c['text']}")
    return "\n\n".join(lines)


def _format_sources_footer(chunks: List[RagChunk]) -> str:
    seen = []
    lines = ["", "Πηγές / Sources:"]
    for c in chunks:
        label = f"{c['source']} (chunk {c['chunk_index']})"
        if label not in seen:
            seen.append(label)
    for i, label in enumerate(seen, start=1):
        lines.append(f"[{i}] {label}")
    return "\n".join(lines)


def rag_node(state: AgentState) -> AgentState:
    query = state["user_input"]
    chunks = retrieve(query, k=4)
    relevant = [c for c in chunks if c.get("distance", 0) <= DISTANCE_THRESHOLD]

    if not relevant:
        no_results = NO_RESULTS_MESSAGE_EL if looks_greek(query) else NO_RESULTS_MESSAGE_EN
        return {**state, "response": no_results, "rag_chunks": []}

    prompt = f"User question: {query}\n\nRetrieved excerpts:\n{_format_chunks_block(relevant)}"
    answer = call_llm(
        system=SYSTEM,
        user_input=prompt,
        history=state.get("history"),
        max_tokens=500,
    )
    full_response = answer.rstrip() + "\n" + _format_sources_footer(relevant)

    return {**state, "response": full_response, "rag_chunks": relevant}
