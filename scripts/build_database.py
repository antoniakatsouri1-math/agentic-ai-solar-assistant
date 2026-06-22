"""
Build data/database.db from data/schema.sql and seed it with realistic
synthetic sample data: ~20 solar/PV equipment products (panels, inverters,
batteries, mounting accessories) and ~180 sales rows spread across 2025
and five Greek regions.

This is intentionally SYNTHETIC sample data, not a real company's actual
sales records - the assignment explicitly asks for "realistic sample
data" for this feature, and no real public dataset matches this exact
shape (products + sales transactions for a PV equipment business).

Run once: python scripts/build_database.py
(Re-running drops and recreates both tables, so it's idempotent.)
"""
import random
import sqlite3
from datetime import date, timedelta

DB_PATH = "data/database.db"
SCHEMA_PATH = "data/schema.sql"

random.seed(42)  # deterministic sample data across reruns

PRODUCTS = [
    ("Πάνελ Μονοκρυσταλλικό 450W",      "Πάνελ",     145.00, 300),
    ("Πάνελ Μονοκρυσταλλικό 550W",      "Πάνελ",     175.00, 260),
    ("Πάνελ Bifacial 600W",             "Πάνελ",     210.00, 150),
    ("Πάνελ Πολυκρυσταλλικό 400W",      "Πάνελ",     120.00, 220),
    ("Πάνελ Half-Cut 540W",             "Πάνελ",     168.00, 190),
    ("Inverter Μονοφασικό 5kW",         "Inverter",  650.00,  80),
    ("Inverter Μονοφασικό 8kW",         "Inverter",  890.00,  60),
    ("Inverter Τριφασικό 10kW",         "Inverter", 1250.00,  45),
    ("Hybrid Inverter 8kW με Μπαταρία", "Inverter", 1680.00,  35),
    ("Inverter Τριφασικό 15kW",         "Inverter", 1790.00,  25),
    ("Μπαταρία Λιθίου 5kWh",            "Μπαταρία",  2200.00,  50),
    ("Μπαταρία Λιθίου 10kWh",           "Μπαταρία",  4100.00,  35),
    ("Μπαταρία Λιθίου 15kWh",           "Μπαταρία",  5950.00,  20),
    ("Βάσεις Στήριξης Κεραμοσκεπής",    "Αξεσουάρ",   45.00, 400),
    ("Βάσεις Στήριξης Δώματος",         "Αξεσουάρ",   65.00, 280),
    ("Καλωδίωση DC Σετ 50m",            "Αξεσουάρ",   38.00, 350),
    ("Διακόπτης Ασφαλείας DC/AC",       "Αξεσουάρ",   55.00, 300),
    ("Σύστημα Παρακολούθησης (Monitoring)", "Αξεσουάρ", 120.00, 150),
    ("Optimizer Ισχύος ανά Πάνελ",      "Αξεσουάρ",   42.00, 320),
    ("Αντικεραυνική Προστασία SPD",     "Αξεσουάρ",   75.00, 200),
]

# Greek regions with meaningfully different solar irradiance profiles -
# ties the SQL data thematically to the weather/production angle.
REGIONS = ["Κρήτη", "Αττική", "Πελοπόννησος", "Θεσσαλία", "Κεντρική Μακεδονία"]

START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)
DAY_SPAN = (END_DATE - START_DATE).days


def random_date():
    return START_DATE + timedelta(days=random.randint(0, DAY_SPAN))


def build_database():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("DROP TABLE IF EXISTS sales; DROP TABLE IF EXISTS products;")
    cur.executescript(schema_sql)

    cur.executemany(
        "INSERT INTO products (product_name, category, unit_price, stock_quantity) "
        "VALUES (?, ?, ?, ?)",
        PRODUCTS,
    )

    n_products = len(PRODUCTS)
    sales_rows = []
    # ~180 sales rows, biased so some products sell more than others (realistic)
    for _ in range(180):
        product_id = random.choices(
            range(1, n_products + 1),
            weights=[random.uniform(0.5, 2.0) for _ in range(n_products)],
            k=1,
        )[0]
        unit_price = PRODUCTS[product_id - 1][2]
        quantity = random.randint(1, 12)
        total_amount = round(unit_price * quantity, 2)
        sales_rows.append(
            (product_id, random_date().isoformat(), quantity, random.choice(REGIONS), total_amount)
        )

    cur.executemany(
        "INSERT INTO sales (product_id, sale_date, quantity, region, total_amount) "
        "VALUES (?, ?, ?, ?, ?)",
        sales_rows,
    )

    conn.commit()

    n_sales = cur.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    n_prods = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    conn.close()
    return n_prods, n_sales


if __name__ == "__main__":
    n_prods, n_sales = build_database()
    print(f"Built {DB_PATH}: {n_prods} products, {n_sales} sales rows")
