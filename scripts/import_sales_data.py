import pandas as pd
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# Read the CSV
csv_path = "data_export/sales_data_chunks/sales_data_part_001.csv"
df = pd.read_csv(csv_path)

# Rename 'dt' to 'sale_date'
df = df.rename(columns={"dt": "sale_date"})

# Map 'activity_flag' to 'promo_flag'
if "activity_flag" in df.columns:
    df["promo_flag"] = df["activity_flag"]
else:
    df["promo_flag"] = False

# Ensure required columns for sales_data table
required_cols = [
    "store_id",
    "product_id",
    "sale_date",
    "sale_amount",
    "promo_flag",
    "stock_hour6_22_cnt",
    "sale_qty",
    "discount",
    "original_price",
    "stock_hour6_14_cnt",
    "stock_hour14_22_cnt",
    "holiday_flag",
    "created_at",
]

# Add missing columns with defaults
if "sale_qty" not in df.columns:
    df["sale_qty"] = 1
if "original_price" not in df.columns:
    df["original_price"] = df["sale_amount"]
if "stock_hour6_14_cnt" not in df.columns:
    df["stock_hour6_14_cnt"] = 0
if "stock_hour14_22_cnt" not in df.columns:
    df["stock_hour14_22_cnt"] = 0
if "created_at" not in df.columns:
    df["created_at"] = datetime.now()

# Only keep required columns
df = df[required_cols]

# Connect to the database
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")

conn = psycopg2.connect(
    dbname=dbname,
    user=user,
    password=password,
    host=host,
    port=port,
)
cur = conn.cursor()

# Insert data row by row (for large files, use COPY or batch insert)
rows_imported = 0
for _, row in df.iterrows():
    cur.execute(
        """
        INSERT INTO sales_data (
            store_id, product_id, sale_date, sale_amount, sale_qty, discount, original_price,
            stock_hour6_22_cnt, stock_hour6_14_cnt, stock_hour14_22_cnt, holiday_flag, promo_flag, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            int(row["store_id"]),
            int(row["product_id"]),
            row["sale_date"],
            float(row["sale_amount"]),
            int(row["sale_qty"]),
            float(row["discount"]),
            float(row["original_price"]),
            int(row["stock_hour6_22_cnt"]),
            int(row["stock_hour6_14_cnt"]),
            int(row["stock_hour14_22_cnt"]),
            bool(row["holiday_flag"]),
            bool(row["promo_flag"]),
            row["created_at"],
        ),
    )
    rows_imported += 1
    if rows_imported % 1000 == 0:
        conn.commit()
conn.commit()
print(f"Imported {rows_imported} rows from {csv_path} into sales_data table.")
cur.close()
conn.close()
