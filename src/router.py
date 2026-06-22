import re

from src.llm import call_llm
from src.state import AgentState

ROUTES = {"weather", "search", "rag", "sql", "general"}

SYSTEM = """You are an intent classifier for a solar/renewable-energy \
equipment business assistant. Classify the user's message into exactly \
ONE of these five labels:

- weather: current weather, cloud cover, sunshine, forecasts for a \
location - relevant for estimating solar production
- search: needs up-to-date external/web information - current energy \
prices, new subsidy program announcements, market news, things that \
change day to day
- rag: questions about Greek regulations for solar/PV systems - net-metering, \
net-billing, licensing, grid connection (ΔΕΔΔΗΕ), the "Φωτοβολταϊκά στη \
Στέγη" subsidy program, VAT/tax treatment, or energy communities - \
answerable from the internal regulatory knowledge base
- sql: questions about sales, products (panels, inverters, batteries, \
accessories), revenue, inventory, or regions - data answerable from the \
internal sales/products database
- general: anything else - general knowledge, explanations of concepts, \
small talk, or anything ambiguous/unclear

Respond with ONLY the label - no punctuation, no explanation, lowercase.

Examples:
"What is the weather in Athens tomorrow?" -> weather
"Θα έχει ηλιοφάνεια στην Κρήτη αυτή την εβδομάδα;" -> weather
"What's the cloud cover forecast for Thessaloniki?" -> weather
"What are current electricity prices in Greece?" -> search
"Υπάρχουν νέα προγράμματα επιδότησης φωτοβολταϊκών αυτή την περίοδο;" -> search
"Any recent news on PV equipment prices?" -> search
"What is net-metering and how does it work?" -> rag
"Ποια είναι τα όρια ισχύος για αυτοπαραγωγούς;" -> rag
"How do I get a Φωτοβολταϊκά στη Στέγη subsidy?" -> rag
"What's the VAT rate on solar panels in Greece?" -> rag
"What were total sales last month?" -> sql
"Ποια κατηγορία προϊόντων είχε τα μεγαλύτερα έσοδα;" -> sql
"How many 10kWh batteries have we sold?" -> sql
"Explain what LangGraph is." -> general
"Hello, how are you?" -> general
"""


def classify_intent(user_input: str) -> str:
    raw = call_llm(system=SYSTEM, user_input=user_input, max_tokens=500, temperature=0)
    cleaned = re.sub(r"[^a-z]", "", raw.strip().lower())

    if cleaned in ROUTES:
        return cleaned

    lowered = raw.lower()
    for route in ROUTES:
        if route in lowered:
            return route

    return "general"


def router_node(state: AgentState) -> AgentState:
    route = classify_intent(state["user_input"])
    return {**state, "route": route}
