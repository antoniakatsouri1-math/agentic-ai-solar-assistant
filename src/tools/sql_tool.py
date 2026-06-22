"""
Text-to-SQL tool (Feature 3).

Provides:
  - get_schema_description(): textual schema the LLM uses to generate SQL
  - validate_sql(): rejects anything that isn't a read-only SELECT/CTE query
  - execute_query(): runs validated SQL against a read-only DB connection
"""
import re
import sqlite3
from typing import List, Optional, Tuple

DB_PATH = "data/database.db"

# Unconditionally blocked regardless of context - matches whole words only
# so e.g. a column literally named "updates" wouldn't false-positive, but
# the keyword UPDATE on its own always will.
BLOCKED_KEYWORDS = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"}

SCHEMA_DESCRIPTION = """\
Table: products
  product_id      INTEGER PRIMARY KEY
  product_name    TEXT        -- e.g. 'Πάνελ Μονοκρυσταλλικό 450W', 'Inverter Τριφασικό 10kW'
  category        TEXT        -- one of 'Πάνελ', 'Inverter', 'Μπαταρία', 'Αξεσουάρ'
  unit_price      REAL
  stock_quantity  INTEGER

Table: sales
  sale_id         INTEGER PRIMARY KEY
  product_id      INTEGER     -- foreign key -> products.product_id
  sale_date       TEXT        -- ISO format 'YYYY-MM-DD'
  quantity        INTEGER
  region          TEXT        -- one of 'Κρήτη', 'Αττική', 'Πελοπόννησος', 'Θεσσαλία', 'Κεντρική Μακεδονία'
  total_amount    REAL        -- quantity * unit_price at time of sale
"""


class SQLValidationError(Exception):
    """Raised when generated SQL fails the read-only safety check."""


class SQLExecutionError(Exception):
    """Raised when validated SQL fails to execute (syntax error, unknown
    column, etc.) - caught by the agent to produce a clean user-facing
    message instead of a raw traceback."""


def get_schema_description() -> str:
    return SCHEMA_DESCRIPTION


def clean_sql(raw: str) -> str:
    """Strip markdown code fences / language tags the LLM sometimes adds,
    and surrounding whitespace."""
    text = raw.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text)
    return text.strip().rstrip(";").strip()


def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
    """Returns (is_valid, reason). Rejects:
      - any blocked keyword (DROP/DELETE/UPDATE/INSERT/ALTER/TRUNCATE), as
        a whole word, anywhere in the query
      - queries that don't start with SELECT or WITH (i.e. anything that
        isn't a read-only query)
      - multiple statements stacked via semicolons (injection guard)
    """
    if not sql.strip():
        return False, "The generated query was empty."

    if ";" in sql:
        return False, "Multiple statements are not allowed."

    tokens = set(re.findall(r"[A-Za-z_]+", sql.upper()))
    blocked_found = tokens & BLOCKED_KEYWORDS
    if blocked_found:
        return False, f"Query contains a blocked operation: {', '.join(sorted(blocked_found))}."

    first_word = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
    if first_word not in ("SELECT", "WITH"):
        return False, "Only read-only SELECT queries are allowed."

    return True, None


def execute_query(sql: str) -> Tuple[List[str], List[tuple]]:
    """Execute a validated SQL string against a read-only connection.
    Raises SQLExecutionError on any sqlite3 failure (syntax error, unknown
    column/table, etc.) so the caller never sees a raw traceback."""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        return columns, rows
    except sqlite3.Error as e:
        raise SQLExecutionError(str(e)) from e
