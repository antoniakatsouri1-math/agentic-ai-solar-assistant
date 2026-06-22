"""
LangGraph graph definition.

START -> router (LLM intent classification) -> {weather|search|rag|sql|general} -> END

Domain: solar/renewable-energy equipment business assistant. The router
(src/router.py) classifies each message into one of the five routes via a
single LLM call.
"""
from langgraph.graph import END, START, StateGraph

from src.agents.general_agent import general_node
from src.agents.rag_agent import rag_node
from src.agents.search_agent import search_node
from src.agents.sql_agent import sql_node
from src.agents.weather_agent import weather_node
from src.router import router_node
from src.state import AgentState


def _route_from_state(state: AgentState) -> str:
    """Reads the route the router_node already classified and stored in
    state - this is the conditional edge function LangGraph calls to
    decide which node runs next."""
    return state.get("route", "general")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("weather", weather_node)
    graph.add_node("general", general_node)
    graph.add_node("search", search_node)
    graph.add_node("rag", rag_node)
    graph.add_node("sql", sql_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_from_state,
        {
            "weather": "weather",
            "search": "search",
            "rag": "rag",
            "sql": "sql",
            "general": "general",
        },
    )
    graph.add_edge("weather", END)
    graph.add_edge("general", END)
    graph.add_edge("search", END)
    graph.add_edge("rag", END)
    graph.add_edge("sql", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"user_input": "What's the weather like in Athens?"})
    print(result["response"])
