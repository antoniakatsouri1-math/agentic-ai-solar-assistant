CREATE TABLE IF NOT EXISTS products (
    product_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name    TEXT NOT NULL,
    category        TEXT NOT NULL,
    unit_price      REAL NOT NULL,
    stock_quantity  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sales (
    sale_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id      INTEGER NOT NULL,
    sale_date       TEXT NOT NULL,      -- ISO format: YYYY-MM-DD
    quantity        INTEGER NOT NULL,
    region          TEXT NOT NULL,      -- Greek region (relevant to solar irradiance)
    total_amount    REAL NOT NULL,      -- quantity * unit_price at time of sale
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE INDEX IF NOT EXISTS idx_sales_product_id ON sales(product_id);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);
