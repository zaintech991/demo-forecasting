"""
Script to test the mappings with sample data from the database.
"""

import os
import json
import pandas as pd
from sqlalchemy import create_engine
from mappings import (
    CITY_MAPPINGS,
    MANAGEMENT_GROUP_MAPPINGS,
    FIRST_CATEGORY_MAPPINGS,
    SALES_MULTIPLIERS,
    CITY_PRICE_ADJUSTMENTS,
    get_store_type,
    get_store_name,
    get_product_name,
    decode_sales_amount,
    encode_sales_amount,
    decode_hourly_sales,
    encode_hourly_sales,
)


def get_db_connection():
    """Create database connection using environment variables."""
    db_params = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "freshretail"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "boolmind"),
    }

    connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    return create_engine(connection_string)


def test_sales_transformations():
    """Test the sales amount and hourly sales transformations."""
    print("\n=== Testing Sales Transformations ===")

    # Test case 1: Basic transformation roundtrip
    encoded_amount = 1.5
    city_id = 0  # Shanghai
    mgmt_group = 2  # Meat & Seafood

    real_amount = decode_sales_amount(encoded_amount, city_id, mgmt_group)
    encoded_back = encode_sales_amount(real_amount, city_id, mgmt_group)

    print("\nTest Case 1: Basic Transformation")
    print(f"Original Encoded: {encoded_amount}")
    print(f"Decoded to Real: ¥{real_amount:.2f}")
    print(f"Encoded Back: {encoded_back:.2f}")
    print(f"Roundtrip Success: {abs(encoded_amount - encoded_back) < 0.0001}")

    # Test case 2: Different cities and categories
    test_cases = [
        (1.0, 0, 0, "Shanghai Fresh Produce"),  # Shanghai Fresh Produce
        (1.0, 3, 2, "Beijing Meat & Seafood"),  # Beijing Meat & Seafood
        (1.0, 8, 4, "Urumqi Beverages"),  # Urumqi Beverages
    ]

    print("\nTest Case 2: City and Category Variations")
    for encoded, city, group, desc in test_cases:
        real = decode_sales_amount(encoded, city, group)
        print(f"\n{desc}:")
        print(f"Encoded Amount: {encoded}")
        print(f"Real Amount: ¥{real:.2f}")
        print(f"City Adjustment: {CITY_PRICE_ADJUSTMENTS[city]}")
        print(f"Category Multiplier: {SALES_MULTIPLIERS[group]}")

    # Test case 3: Hourly sales transformation
    print("\nTest Case 3: Hourly Sales")
    sample_hours = [1.2, 0.8, 1.5, 0.9, 1.1]
    hours_json = json.dumps(sample_hours)

    print("\nTesting hourly sales for Shanghai Fresh Produce:")
    real_hours = decode_hourly_sales(hours_json, 0, 0)
    encoded_hours = encode_hourly_sales(real_hours, 0, 0)

    print("Original Hours:", sample_hours)
    print("Real Values:", [f"¥{x:.2f}" for x in real_hours])
    print("Encoded Back:", [f"{x:.2f}" for x in encoded_hours])
    print(
        "Roundtrip Success:",
        all(abs(a - b) < 0.0001 for a, b in zip(sample_hours, encoded_hours)),
    )


def test_mappings():
    """Test the mappings with sample data."""
    engine = get_db_connection()

    # First run sales transformation tests
    test_sales_transformations()

    # Test city mappings with store performance
    print("\n=== Sample Store Names and Performance ===")
    store_query = """
    SELECT 
        s.city_id,
        s.store_id,
        ROUND(AVG(sd.sale_amount), 2) as avg_sale_amount,
        COUNT(DISTINCT sd.product_id) as unique_products,
        SUM(CASE WHEN sd.stock_hour6_22_cnt > 0 THEN 1 ELSE 0 END) as stockout_days,
        MAX(sd.hours_sale) as sample_hour_sale
    FROM store_hierarchy s
    JOIN sales_data sd ON s.store_id = sd.store_id
    GROUP BY s.city_id, s.store_id
    ORDER BY avg_sale_amount DESC
    LIMIT 10;
    """
    with engine.connect() as conn:
        stores = pd.read_sql_query(store_query, conn)

    for _, store in stores.iterrows():
        store_name = get_store_name(
            store["city_id"], store["store_id"], store["avg_sale_amount"]
        )
        real_avg_sale = decode_sales_amount(
            store["avg_sale_amount"], store["city_id"], 0
        )  # Using 0 as default management group
        print(f"\nStore: {store_name}")
        print(f"Performance:")
        print(f"- Encoded Average Sale: {store['avg_sale_amount']:.2f}")
        print(f"- Real Average Sale: ¥{real_avg_sale:.2f}")
        print(f"- Unique Products: {store['unique_products']}")
        print(f"- Stockout Days: {store['stockout_days']}")

        # Display hourly sales if available
        if store["sample_hour_sale"]:
            real_hourly = decode_hourly_sales(
                store["sample_hour_sale"], store["city_id"], 0
            )
            if real_hourly:
                print(
                    "- Sample Hourly Sales (Real):",
                    [f"¥{x:.2f}" for x in real_hourly[:5]] + ["..."],
                )

    # Test product mappings with sales data
    print("\n\n=== Sample Product Categories and Performance ===")
    product_query = """
    SELECT 
        p.management_group_id,
        p.first_category_id,
        p.product_id,
        ROUND(AVG(s.sale_amount), 2) as avg_sale_amount,
        ROUND(AVG(s.discount), 2) as avg_discount,
        COUNT(DISTINCT s.store_id) as num_stores_selling,
        SUM(CASE WHEN s.stock_hour6_22_cnt > 0 THEN 1 ELSE 0 END) as total_stockout_days
    FROM product_hierarchy p
    JOIN sales_data s ON p.product_id = s.product_id
    GROUP BY p.management_group_id, p.first_category_id, p.product_id
    ORDER BY avg_sale_amount DESC
    LIMIT 10;
    """
    products = pd.read_sql_query(product_query, engine)

    for _, product in products.iterrows():
        product_name = get_product_name(
            product["management_group_id"],
            product["first_category_id"],
            product["product_id"],
        )
        group = MANAGEMENT_GROUP_MAPPINGS[product["management_group_id"]]
        real_avg_sale = decode_sales_amount(
            product["avg_sale_amount"],
            0,  # Using Shanghai as reference city
            product["management_group_id"],
        )
        print(f"\nProduct: {product_name}")
        print(f"Group: {group}")
        print(f"Performance:")
        print(f"- Encoded Average Sale: {product['avg_sale_amount']:.2f}")
        print(f"- Real Average Sale: ¥{real_avg_sale:.2f}")
        print(f"- Average Discount: {product['avg_discount']:.0%}")
        print(f"- Stores Selling: {product['num_stores_selling']}")
        print(f"- Total Stockout Days: {product['total_stockout_days']}")

    # Test weather patterns by city
    print("\n\n=== City Weather Patterns ===")
    weather_query = """
    SELECT 
        city_id,
        ROUND(AVG(avg_temperature), 1) as avg_temp,
        ROUND(AVG(avg_humidity), 1) as avg_humidity,
        ROUND(AVG(precpt), 1) as avg_precipitation,
        COUNT(DISTINCT sale_date) as days_with_data
    FROM sales_data
    GROUP BY city_id
    ORDER BY avg_temp DESC;
    """
    weather = pd.read_sql_query(weather_query, engine)

    for _, city in weather.iterrows():
        city_name = CITY_MAPPINGS[city["city_id"]]
        print(f"\nCity: {city_name}")
        print(f"Weather Patterns:")
        print(f"- Average Temperature: {city['avg_temp']}°C")
        print(f"- Average Humidity: {city['avg_humidity']}%")
        print(f"- Average Precipitation: {city['avg_precipitation']} mm")
        print(f"- Days with Data: {city['days_with_data']}")
        print(
            f"- Price Adjustment Factor: {CITY_PRICE_ADJUSTMENTS[city['city_id']]:.2f}"
        )


if __name__ == "__main__":
    test_mappings()
