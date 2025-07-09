"""
Script to run SQL queries and explore the FreshRetailNet-50K dataset.
"""
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def run_query(query, engine=None):
    """Run a SQL query and return results as a pandas DataFrame."""
    if engine is None:
        engine = get_db_connection()
    
    print("\nExecuting query:")
    print("-" * 80)
    print(query)
    print("-" * 80)
    
    result = pd.read_sql_query(query, engine)
    
    print("\nResults:")
    print("-" * 80)
    print(result)
    print(f"\nTotal rows: {len(result)}")
    
    return result

def main():
    """Run example queries to explore the data."""
    engine = get_db_connection()
    
    # Example queries
    queries = {
        "Basic Sales Stats": """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT store_id) as unique_stores,
                COUNT(DISTINCT product_id) as unique_products,
                MIN(sale_date) as start_date,
                MAX(sale_date) as end_date,
                SUM(sale_amount) as total_sales,
                AVG(sale_amount) as avg_sale_amount
            FROM sales_data;
        """,
        
        "Top 10 Products by Sales": """
            SELECT 
                p.product_id,
                p.first_category_id,
                SUM(s.sale_amount) as total_sales,
                COUNT(DISTINCT s.store_id) as stores_sold_in,
                AVG(s.discount) as avg_discount
            FROM sales_data s
            JOIN product_hierarchy p ON s.product_id = p.product_id
            GROUP BY p.product_id, p.first_category_id
            ORDER BY total_sales DESC
            LIMIT 10;
        """,
        
        "Store Performance": """
            SELECT 
                s.store_id,
                s.city_id,
                COUNT(DISTINCT sd.product_id) as unique_products,
                AVG(sd.sale_amount) as avg_daily_sales,
                SUM(CASE WHEN sd.stock_hour6_22_cnt > 0 THEN 1 ELSE 0 END) as stockout_days
            FROM store_hierarchy s
            LEFT JOIN sales_data sd ON s.store_id = sd.store_id
            GROUP BY s.store_id, s.city_id
            ORDER BY avg_daily_sales DESC
            LIMIT 10;
        """,
        
        "Weather Impact": """
            SELECT 
                CASE 
                    WHEN avg_temperature < 10 THEN 'Cold (< 10°C)'
                    WHEN avg_temperature < 20 THEN 'Mild (10-20°C)'
                    WHEN avg_temperature < 30 THEN 'Warm (20-30°C)'
                    ELSE 'Hot (> 30°C)'
                END as temperature_range,
                ROUND(AVG(sale_amount), 2) as avg_sales,
                COUNT(*) as total_records
            FROM sales_data
            GROUP BY 
                CASE 
                    WHEN avg_temperature < 10 THEN 'Cold (< 10°C)'
                    WHEN avg_temperature < 20 THEN 'Mild (10-20°C)'
                    WHEN avg_temperature < 30 THEN 'Warm (20-30°C)'
                    ELSE 'Hot (> 30°C)'
                END
            ORDER BY avg_sales DESC;
        """,
        
        "Holiday vs Non-Holiday Sales": """
            SELECT 
                holiday_flag,
                COUNT(*) as total_records,
                ROUND(AVG(sale_amount), 2) as avg_sales,
                ROUND(AVG(discount), 2) as avg_discount,
                COUNT(DISTINCT store_id) as stores_with_sales
            FROM sales_data
            GROUP BY holiday_flag;
        """
    }
    
    # Run each query
    for title, query in queries.items():
        print(f"\n{'=' * 80}")
        print(f"Query: {title}")
        print(f"{'=' * 80}")
        run_query(query, engine)

if __name__ == "__main__":
    main() 