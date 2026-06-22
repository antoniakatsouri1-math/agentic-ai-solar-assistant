import re

from src.llm import call_llm
from src.state import AgentState
from src.tools.weather import get_weather

SYSTEM = (
    "You are a weather assistant for a solar/renewable-energy equipment "
    "business. You will be given a raw weather lookup result that includes "
    "cloud cover and sunshine duration. Turn it into a short, friendly, "
    "natural-language answer to the user's question. If relevant, briefly "
    "note what the cloud cover / sunshine duration implies for solar "
    "production that day (e.g. low cloud cover -> good production "
    "conditions). Do not invent any data not present in the lookup result. "
    "IMPORTANT: respond in the same language the user asked their question "
    "in (Greek question -> Greek answer, English question -> English "
    "answer)."
)


def _extract_city(text: str) -> str:
    match = re.search(r"\bin\s+([A-Za-z\s]+?)(?:[\?\.,]|$)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\bστ[ηο]\w?\s+([Α-ΩΆ-Ώ][α-ωά-ώ]*(?:\s+[Α-ΩΆ-Ώ][α-ωά-ώ]*)*)", text)
    if match:
        return match.group(1).strip()
    return text.strip()

def weather_node(state: AgentState) -> AgentState:
    city = _extract_city(state["user_input"])
    raw_result = get_weather(city)

    answer = call_llm(
        system=SYSTEM,
        user_input=f"User asked: '{state['user_input']}'\nLookup result: {raw_result}",
        history=state.get("history"),
        max_tokens=350,
    )
    return {**state, "response": answer}
