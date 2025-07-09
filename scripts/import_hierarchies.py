import pandas as pd
import psycopg2

# Read the CSV
csv_path = 'data_export/sales_data_csv.csv'
df = pd.read_csv(csv_path)

unique_stores = df['store_id'].unique()
unique_products = df['product_id'].unique()

conn = psycopg2.connect(dbname='freshretail', user='postgres', password='boolmind', host='localhost', port=5432)
cur = conn.cursor()

# Insert stores with default city_id=0
for store_id in unique_stores:
    cur.execute(
        "INSERT INTO store_hierarchy (store_id, city_id) VALUES (%s, %s) ON CONFLICT (store_id) DO NOTHING",
        (int(store_id), 0)
    )

# Insert products with default category/group values
for product_id in unique_products:
    cur.execute(
        """
        INSERT INTO product_hierarchy
        (product_id, management_group_id, first_category_id, second_category_id, third_category_id)
        VALUES (%s, 0, 0, 0, 0) ON CONFLICT (product_id) DO NOTHING
        """,
        (int(product_id),)
    )

conn.commit()
cur.close()
conn.close()
print('Store and product hierarchies imported!') 