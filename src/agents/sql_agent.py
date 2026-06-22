"""
Text-to-SQL Agent (Feature 3).

Pipeline: User question -> Generate SQL -> Validate SQL -> Execute query ->
Generate natural-language answer. Each stage fails gracefully: invalid/
unsafe SQL never reaches execution, and execution errors never reach the
user as a raw traceback.
"""
from src.llm import call_llm
from src.state import AgentState
from src.lang_detect import looks_greek
from src.tools.sql_tool import (
    SQLExecutionError,
    clean_sql,
    execute_query,
    get_schema_description,
    validate_sql,
)

GEN_SYSTEM = (
    "You translate natural-language questions into a single SQLite SELECT "
    "query. You will be given the database schema. Output ONLY the raw SQL "
    "query - no markdown code fences, no explanation, no semicolon. The "
    "query must be read-only (SELECT or WITH ... SELECT only). If the "
    "question cannot plausibly be answered using only the given tables, "
    "still produce your best-effort SELECT query against the closest "
    "matching columns rather than refusing - the validation/execution "
    "stages will catch genuine failures. The question may be in Greek or "
    "English - the table/column names and string values in the database "
    "are in Greek for product/category/region names, so match Greek "
    "string literals exactly where relevant (e.g. category = 'Πάνελ').\n\n"
    f"Schema:\n{get_schema_description()}"
)

EXPLAIN_SYSTEM = (
    "You explain SQL query results in plain, concise natural language. "
    "You will be given the original question, the SQL query that was run, "
    "and the resulting rows. Answer the question directly using only the "
    "given results - do not speculate beyond the data shown. IMPORTANT: "
    "respond in the same language the user asked their question in (Greek "
    "question -> Greek answer, English question -> English answer)."
)

EXECUTION_FAILURE_MESSAGE_EN = (
    "I generated a SQL query for that, but it failed to run against the "
    "database ({reason}). This question may not be answerable with the "
    "available data (products and sales tables) - could you rephrase it?"
)
EXECUTION_FAILURE_MESSAGE_EL = (
    "Δημιούργησα ένα ερώτημα SQL για αυτό, αλλά απέτυχε στην εκτέλεσή του "
    "({reason}). Αυτή η ερώτηση μπορεί να μην απαντιέται με τα διαθέσιμα "
    "δεδομένα (πίνακες products και sales) - θα μπορούσες να την "
    "αναδιατυπώσεις;"
)

VALIDATION_FAILURE_MESSAGE_EN = "I can't run that query: {reason}"
VALIDATION_FAILURE_MESSAGE_EL = "Δεν μπορώ να εκτελέσω αυτό το ερώτημα: {reason}"

NO_ROWS_MESSAGE_EN = "The query ran successfully but returned no matching rows."
NO_ROWS_MESSAGE_EL = "Το ερώτημα εκτελέστηκε επιτυχώς αλλά δεν επέστρεψε αποτελέσματα."


def generate_sql(question: str, history=None) -> str:
    raw = call_llm(system=GEN_SYSTEM, user_input=question, history=history, max_tokens=300, temperature=0)
    return clean_sql(raw)


def sql_node(state: AgentState) -> AgentState:
    question = state["user_input"]
    is_greek = looks_greek(question)
    sql = generate_sql(question, state.get("history"))

    is_valid, reason = validate_sql(sql)
    if not is_valid:
        template = VALIDATION_FAILURE_MESSAGE_EL if is_greek else VALIDATION_FAILURE_MESSAGE_EN
        return {
            **state,
            "response": template.format(reason=reason),
            "sql_query": sql,
            "error": reason,
        }

    try:
        columns, rows = execute_query(sql)
    except SQLExecutionError as e:
        template = EXECUTION_FAILURE_MESSAGE_EL if is_greek else EXECUTION_FAILURE_MESSAGE_EN
        return {
            **state,
            "response": template.format(reason=str(e)),
            "sql_query": sql,
            "error": str(e),
        }

    if not rows:
        answer = NO_ROWS_MESSAGE_EL if is_greek else NO_ROWS_MESSAGE_EN
    else:
        prompt = (
            f"Question: {question}\nSQL: {sql}\nColumns: {columns}\n"
            f"Rows (up to 50 shown): {rows[:50]}"
        )
        answer = call_llm(system=EXPLAIN_SYSTEM, user_input=prompt, history=state.get("history"), max_tokens=300)

    full_response = f"{answer}\n\n(SQL used: {sql})"
    return {**state, "response": full_response, "sql_query": sql}
