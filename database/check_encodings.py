"""
Script to analyze encoded values in the FreshRetailNet-50K dataset.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def get_db_connection():
    """Create database connection using environment variables."""
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'freshretail'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'boolmind')
    }
    
    connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    return create_engine(connection_string)

def analyze_encodings():
    """Analyze encoded values in the dataset."""
    engine = get_db_connection()
    
    # Check city distribution
    print("\n=== City Analysis ===")
    city_query = """
    SELECT 
        city_id,
        COUNT(DISTINCT store_id) as num_stores,
        COUNT(*) as num_records,
        ROUND(AVG(sale_amount), 2) as avg_sale_amount,
        ROUND(AVG(avg_temperature), 2) as avg_temp,
        ROUND(AVG(avg_humidity), 2) as avg_humidity
    FROM sales_data
    GROUP BY city_id
    ORDER BY city_id;
    """
    print(pd.read_sql_query(city_query, engine))
    
    # Check store distribution within cities
    print("\n=== Store Distribution ===")
    store_query = """
    SELECT 
        city_id,
        COUNT(DISTINCT store_id) as num_stores,
        MIN(store_id) as min_store_id,
        MAX(store_id) as max_store_id
    FROM store_hierarchy
    GROUP BY city_id
    ORDER BY city_id;
    """
    print(pd.read_sql_query(store_query, engine))
    
    # Check product category hierarchy
    print("\n=== Product Category Hierarchy ===")
    category_query = """
    SELECT 
        management_group_id,
        first_category_id,
        second_category_id,
        third_category_id,
        COUNT(*) as num_products,
        COUNT(DISTINCT product_id) as unique_products
    FROM product_hierarchy
    GROUP BY 
        management_group_id,
        first_category_id,
        second_category_id,
        third_category_id
    ORDER BY 
        management_group_id,
        first_category_id,
        second_category_id,
        third_category_id;
    """
    print(pd.read_sql_query(category_query, engine))
    
    # Check product sales patterns
    print("\n=== Product Sales Patterns ===")
    product_query = """
    SELECT 
        p.management_group_id,
        p.first_category_id,
        COUNT(DISTINCT p.product_id) as num_products,
        ROUND(AVG(s.sale_amount), 2) as avg_sale_amount,
        ROUND(AVG(s.discount), 2) as avg_discount,
        COUNT(DISTINCT s.store_id) as num_stores_selling
    FROM product_hierarchy p
    JOIN sales_data s ON p.product_id = s.product_id
    GROUP BY 
        p.management_group_id,
        p.first_category_id
    ORDER BY 
        p.management_group_id,
        p.first_category_id;
    """
    print(pd.read_sql_query(product_query, engine))

if __name__ == "__main__":
    analyze_encodings() 