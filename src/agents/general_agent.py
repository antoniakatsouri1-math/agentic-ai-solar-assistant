"""Baseline general-purpose LLM node (catch-all / fallback route)."""
from src.llm import call_llm
from src.state import AgentState

SYSTEM = (
    "You are a helpful general-purpose assistant for a solar/renewable-energy "
    "equipment business. Answer the user's question directly and concisely. "
    "Use the conversation history for context when the user refers back to "
    "something said earlier. IMPORTANT: respond in the same language the "
    "user asked their question in (Greek question -> Greek answer, English "
    "question -> English answer)."
)


def general_node(state: AgentState) -> AgentState:
    answer = call_llm(
        system=SYSTEM,
        user_input=state["user_input"],
        history=state.get("history"),
    )
    return {**state, "response": answer}
