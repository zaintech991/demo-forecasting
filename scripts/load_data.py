"""
Script to load FreshRetailNet-50K data from Hugging Face into Supabase database.
"""
import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
import requests

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import custom modules
from utils.mapping_data import CITY_MAPPING, STORE_MAPPING

# Load environment variables
load_dotenv()

class SupabaseDirectConnector:
    """Simple connector class to execute SQL directly via Supabase REST API"""
    
    def __init__(self):
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        self.service_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY in .env file.")
    
    def execute_query(self, query):
        """Execute SQL query directly using REST API"""
        headers = {
            'apikey': self.key,
            'Authorization': f'Bearer {self.service_key if self.service_key else self.key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }
        
        # For SELECT queries, use the data API
        if query.strip().lower().startswith('select'):
            endpoint = f"{self.url}/rest/v1/"
            # This is a simplified approach - for complex queries, you'd need to build the proper URL
            # For demo, we're only supporting simple INSERT/CREATE/DROP operations
            print("SELECT queries not supported directly. Use the Supabase client instead.")
            return False
        else:
            # For other queries (INSERT, CREATE, etc.), use the SQL endpoint
            endpoint = f"{self.url}/rest/v1/sql"
            payload = {"query": query}
            
        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            
            if response.status_code < 300:
                return True
            else:
                print(f"Query error: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"Error executing query: {e}")
            return False

def get_create_schema_sql():
    """
    Generate the SQL schema creation statements
    
    Returns:
        str: Complete SQL schema as a string
    """
    # Enable the UUID extension if not already enabled
    enable_uuid = """
    -- Enable UUID extension
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    """
    
    # Create sales_data table
    create_sales_table = """
    -- Create sales_data table
    CREATE TABLE IF NOT EXISTS sales_data (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        city_id INTEGER NOT NULL,
        store_id INTEGER NOT NULL,
        management_group_id INTEGER NOT NULL,
        first_category_id INTEGER NOT NULL,
        second_category_id INTEGER NOT NULL,
        third_category_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        dt DATE NOT NULL,
        sale_amount FLOAT NOT NULL,
        stock_hour6_22_cnt INTEGER,
        discount FLOAT,
        holiday_flag INTEGER,
        activity_flag INTEGER,
        precpt FLOAT,
        avg_temperature FLOAT,
        avg_humidity FLOAT,
        avg_wind_level FLOAT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    # Create indexes
    create_indexes = """
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_sales_store_product ON sales_data (store_id, product_id);
    CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_data (dt);
    CREATE INDEX IF NOT EXISTS idx_sales_city ON sales_data (city_id);
    CREATE INDEX IF NOT EXISTS idx_sales_category ON sales_data (first_category_id, second_category_id, third_category_id);
    """
    
    # Create promotions table
    create_promotions_table = """
    -- Create promotions table
    CREATE TABLE IF NOT EXISTS promotions (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        store_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        promotion_type VARCHAR(50) NOT NULL,
        discount FLOAT NOT NULL,
        location VARCHAR(50),
        campaign_id VARCHAR(50),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    # Create holidays table
    create_holidays_table = """
    -- Create holidays table
    CREATE TABLE IF NOT EXISTS holidays (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        date DATE NOT NULL,
        name VARCHAR(100) NOT NULL,
        country VARCHAR(50) DEFAULT 'USA',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (date, name)
    );
    """
    
    # Combine all SQL statements
    full_schema = enable_uuid + "\n" + create_sales_table + "\n" + create_indexes + "\n" + create_promotions_table + "\n" + create_holidays_table
    
    return full_schema

def create_tables(db_conn):
    """
    Create database schema in Supabase
    
    Args:
        db_conn: SupabaseDirectConnector instance
    """
    print("Creating database schema...")
    
    # Get the schema SQL
    schema_sql = get_create_schema_sql()
    
    # Split into individual statements
    statements = schema_sql.split(';')
    
    # Execute each statement
    for statement in statements:
        if statement.strip():
            try:
                db_conn.execute_query(statement)
                # Extract the first few words to identify the statement type
                statement_type = ' '.join(statement.strip().split()[:3])
                print(f"Executed: {statement_type}...")
            except Exception as e:
                print(f"Error executing statement: {e}")
    
    print("Database schema creation completed!")

def load_huggingface_data(limit=None):
    """
    Load FreshRetailNet-50K data from Hugging Face
    
    Args:
        limit: Optional limit on number of records to load (for testing)
        
    Returns:
        DataFrame with sales data
    """
    print("Loading data from Hugging Face...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        
        # Convert to DataFrame - for large datasets, process in batches
        print("Converting to DataFrame...")
        
        # Get first batch to determine schema
        first_batch = next(iter(dataset["train"].iter(batch_size=10)))
        df_train = pd.DataFrame(first_batch)
        
        # If limit is set and we already have enough data, return
        if limit is not None and len(df_train) >= limit:
            print(f"Limited dataset to {limit} records for testing")
            return df_train.head(limit)
            
        # Otherwise, process more batches until we reach the limit
        if limit is not None:
            batch_size = min(1000, limit)  # Use a reasonable batch size
            batches_needed = (limit + batch_size - 1) // batch_size
            
            # Start from the second batch since we already processed the first one
            batch_iter = dataset["train"].iter(batch_size=batch_size)
            next(batch_iter)  # Skip first batch as we already processed it
            
            for i, batch in enumerate(batch_iter):
                if i >= batches_needed - 1:  # -1 because we already processed one batch
                    break
                df_batch = pd.DataFrame(batch)
                df_train = pd.concat([df_train, df_batch], ignore_index=True)
                if len(df_train) >= limit:
                    df_train = df_train.head(limit)
                    break
                    
            print(f"Limited dataset to {len(df_train)} records for testing")
        else:
            print("Loading full dataset - this may take a while and consume significant memory...")
            # For full dataset, process in larger batches
            batch_size = 10000
            df_list = [df_train]  # Start with our first batch
            
            batch_iter = dataset["train"].iter(batch_size=batch_size)
            next(batch_iter)  # Skip first batch as we already processed it
            
            for i, batch in enumerate(batch_iter):
                if i % 10 == 0:
                    print(f"Processed {i*batch_size + len(df_train)} records...")
                df_batch = pd.DataFrame(batch)
                df_list.append(df_batch)
            
            df_train = pd.concat(df_list, ignore_index=True)
            print(f"Loaded {len(df_train)} records total")
        
        return df_train
    except Exception as e:
        print(f"Error loading data from Hugging Face: {e}")
        sys.exit(1)

def process_data(df):
    """
    Process and transform the data for Supabase
    
    Args:
        df: Raw DataFrame from Hugging Face
        
    Returns:
        Processed DataFrame ready for database import
    """
    print("Processing data...")
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert hours_sale sequence to JSON string to store in database
    if 'hours_sale' in processed_df.columns:
        processed_df['hours_sale_json'] = processed_df['hours_sale'].apply(lambda x: str(list(x)))
        processed_df.drop('hours_sale', axis=1, inplace=True)
    
    # Convert hours_stock_status sequence to JSON string to store in database
    if 'hours_stock_status' in processed_df.columns:
        processed_df['hours_stock_status_json'] = processed_df['hours_stock_status'].apply(lambda x: str(list(x)))
        processed_df.drop('hours_stock_status', axis=1, inplace=True)
    
    # Generate fake dates starting from 2022-01-01
    # The dataset doesn't have real dates, so we'll create them
    unique_dates = sorted(processed_df['dt'].unique())
    date_mapping = {
        old_date: pd.to_datetime('2022-01-01') + pd.Timedelta(days=i)
        for i, old_date in enumerate(unique_dates)
    }
    
    processed_df['dt'] = processed_df['dt'].map(date_mapping)
    
    # Generate holiday names for holiday flag=1
    # For demo purposes, we'll use generic holiday names
    holidays = ["New Year's Day", "Martin Luther King Jr. Day", "Presidents' Day", 
                "Memorial Day", "Independence Day", "Labor Day", 
                "Columbus Day", "Veterans Day", "Thanksgiving", "Christmas"]
    
    holiday_rows = processed_df[processed_df['holiday_flag'] == 1]
    unique_holiday_dates = sorted(holiday_rows['dt'].unique())
    
    # Assign holiday names (cycle through the list if there are more holiday dates than names)
    holiday_name_mapping = {
        date: holidays[i % len(holidays)]
        for i, date in enumerate(unique_holiday_dates)
    }
    
    # Add holiday_name column
    processed_df['holiday_name'] = None
    for date, name in holiday_name_mapping.items():
        processed_df.loc[processed_df['dt'] == date, 'holiday_name'] = name
    
    return processed_df

def generate_insert_sql(df, batch_size=20):
    """
    Generate SQL INSERT statements for the data
    
    Args:
        df: Processed DataFrame
        batch_size: Number of records in each INSERT statement
        
    Returns:
        list: List of SQL INSERT statements
    """
    print("Generating INSERT statements...")
    
    # Calculate number of batches
    n_batches = (len(df) + batch_size - 1) // batch_size
    insert_statements = []
    
    # Process in batches
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        
        batch = df.iloc[start_idx:end_idx].copy()
        
        # Convert dates to strings
        batch['dt'] = batch['dt'].dt.strftime('%Y-%m-%d')
        
        # Handle any NaN values
        for col in batch.columns:
            if batch[col].dtype == 'float64':
                batch[col] = batch[col].fillna(0.0)
            elif batch[col].dtype == 'int64':
                batch[col] = batch[col].fillna(0)
            elif batch[col].dtype == 'object':
                batch[col] = batch[col].fillna('')
        
        # Build VALUES part of the query
        values_list = []
        for _, row in batch.iterrows():
            values = f"({row['city_id']}, {row['store_id']}, {row['management_group_id']}, "
            values += f"{row['first_category_id']}, {row['second_category_id']}, {row['third_category_id']}, "
            values += f"{row['product_id']}, '{row['dt']}', {row['sale_amount']}, "
            values += f"{row['stock_hour6_22_cnt']}, {row['discount']}, {row['holiday_flag']}, "
            values += f"{row['activity_flag']}, {row['precpt']}, {row['avg_temperature']}, "
            values += f"{row['avg_humidity']}, {row['avg_wind_level']})"
            values_list.append(values)
        
        # Join all values
        if values_list:
            insert_query = f"""
            -- Insert batch {i+1}/{n_batches}
            INSERT INTO sales_data (
                city_id, store_id, management_group_id, 
                first_category_id, second_category_id, third_category_id,
                product_id, dt, sale_amount, stock_hour6_22_cnt, 
                discount, holiday_flag, activity_flag, precpt, 
                avg_temperature, avg_humidity, avg_wind_level
            ) 
            VALUES {', '.join(values_list)};
            """
            
            insert_statements.append(insert_query)
    
    return insert_statements

def generate_holiday_insert_sql(df):
    """
    Generate SQL INSERT statements for holiday data
    
    Args:
        df: Processed DataFrame with holiday_name column
        
    Returns:
        list: List of SQL INSERT statements for holidays
    """
    print("Generating holiday INSERT statements...")
    
    # Extract holiday data
    holiday_df = df[df['holiday_flag'] == 1].dropna(subset=['holiday_name'])
    
    # Convert dates to strings if they're datetime objects
    if pd.api.types.is_datetime64_any_dtype(holiday_df['dt']):
        holiday_df['dt'] = holiday_df['dt'].dt.strftime('%Y-%m-%d')
    
    # Get unique combinations of date and holiday name
    holiday_data = holiday_df[['dt', 'holiday_name']].drop_duplicates()
    
    # Generate INSERT statements
    insert_statements = []
    for _, row in holiday_data.iterrows():
        insert_query = f"""
        -- Insert holiday {row['holiday_name']} on {row['dt']}
        INSERT INTO holidays (date, name, country)
        VALUES ('{row['dt']}', '{row['holiday_name']}', 'USA')
        ON CONFLICT (date, name) DO NOTHING;
        """
        
        insert_statements.append(insert_query)
    
    return insert_statements

def insert_data_to_supabase(db_conn, df, batch_size=20):
    """
    Insert data into Supabase database using the REST API
    
    Args:
        db_conn: SupabaseDirectConnector instance
        df: Processed DataFrame
        batch_size: Number of records to insert in each batch
    """
    print("Inserting data into Supabase...")
    
    # Generate INSERT statements
    insert_statements = generate_insert_sql(df, batch_size)
    
    # Execute each statement
    success_count = 0
    for i, statement in enumerate(tqdm(insert_statements)):
        try:
            if db_conn.execute_query(statement):
                # Roughly estimate the number of rows in this batch
                if i < len(insert_statements) - 1:
                    success_count += batch_size
                else:
                    success_count += len(df) - (len(insert_statements) - 1) * batch_size
            
            # Add a small delay between batches to avoid overloading the API
            time.sleep(0.2)
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Stopping data insertion.")
            print(f"Inserted approximately {success_count} records before interruption.")
            return
        except Exception as e:
            print(f"Error inserting batch {i+1}/{len(insert_statements)}: {e}")
            continue
    
    print(f"Data insertion completed! Inserted approximately {success_count} records.")

def insert_holiday_data(db_conn, df):
    """
    Insert holiday data into holidays table
    
    Args:
        db_conn: SupabaseDirectConnector instance
        df: Processed DataFrame with holiday_name column
    """
    print("Inserting holiday data...")
    
    # Generate INSERT statements
    insert_statements = generate_holiday_insert_sql(df)
    
    success_count = 0
    # Execute each statement
    for i, statement in enumerate(insert_statements):
        try:
            if db_conn.execute_query(statement):
                success_count += 1
            
        except Exception as e:
            print(f"Error inserting holiday {i+1}/{len(insert_statements)}: {e}")
    
    print(f"Inserted {success_count} of {len(insert_statements)} holidays")

def main():
    """Main function to load data from Hugging Face to Supabase"""
    # Parse command line arguments for testing with small dataset
    import argparse
    parser = argparse.ArgumentParser(description='Load FreshRetailNet-50K data into Supabase')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records for testing')
    parser.add_argument('--generate-sql-only', action='store_true', help='Generate SQL statements only, without executing')
    args = parser.parse_args()
    
    # If we're just generating SQL
    if args.generate_sql_only:
        # Output the schema SQL
        print(get_create_schema_sql())
        
        # For sample data, load a small set
        if args.limit:
            # Load data from Hugging Face
            raw_df = load_huggingface_data(limit=args.limit)
            
            # Process the data
            processed_df = process_data(raw_df)
            
            # Generate and print INSERT statements
            insert_statements = generate_insert_sql(processed_df)
            for statement in insert_statements:
                print(statement)
            
            # Generate and print holiday INSERT statements
            holiday_statements = generate_holiday_insert_sql(processed_df)
            for statement in holiday_statements:
                print(statement)
                
        return
    
    # Connect to Supabase
    try:
        db_conn = SupabaseDirectConnector()
        print("Connected to Supabase!")
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        sys.exit(1)
    
    # Create tables if they don't exist
    create_tables(db_conn)
    
    # Load data from Hugging Face
    raw_df = load_huggingface_data(limit=args.limit)
    
    # Process the data
    processed_df = process_data(raw_df)
    
    # Insert data into Supabase
    insert_data_to_supabase(db_conn, processed_df)
    
    # Insert holiday data
    insert_holiday_data(db_conn, processed_df)
    
    print("Data loading completed successfully!")

if __name__ == "__main__":
    main() 