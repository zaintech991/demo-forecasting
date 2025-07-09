"""
Script to load FreshRetailNet-50K dataset from Hugging Face and ingest it into Supabase.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datasets import load_dataset
from tqdm import tqdm
import holidays
import random

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import get_db_engine, get_supabase_client

def load_dataset_from_huggingface():
    """Load the dataset from Hugging Face."""
    print("Loading dataset from Hugging Face...")
    try:
        ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        print(f"Dataset loaded successfully. Available splits: {ds.keys()}")
        
        # Convert to pandas DataFrame
        if 'train' in ds:
            df = ds['train'].to_pandas()
            print(f"Dataset shape: {df.shape}")
            return df
        else:
            print("No 'train' split found in the dataset.")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("You may need to log in to Hugging Face using 'huggingface-cli login'")
        return None

def create_tables(engine):
    """Create database tables using schema.sql."""
    print("Creating database tables...")
    schema_path = Path(__file__).parent / 'schema.sql'
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
        
    with engine.connect() as conn:
        conn.execute(schema_sql)
        conn.commit()
    
    print("Database tables created successfully.")

def preprocess_data(df):
    """Preprocess the dataset for database ingestion."""
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert date columns if needed
    if 'sale_date' in processed_df.columns and not pd.api.types.is_datetime64_any_dtype(processed_df['sale_date']):
        processed_df['sale_date'] = pd.to_datetime(processed_df['sale_date'])
    
    # Extract hierarchies
    store_hierarchy = extract_store_hierarchy(processed_df)
    product_hierarchy = extract_product_hierarchy(processed_df)
    
    # Generate supplementary data
    weather_data = generate_weather_data(processed_df)
    holiday_data = generate_holiday_data(processed_df)
    promotion_data = generate_promotion_data(processed_df)
    
    # Transform main sales data
    sales_data = transform_sales_data(processed_df)
    
    return {
        'store_hierarchy': store_hierarchy,
        'product_hierarchy': product_hierarchy,
        'sales_data': sales_data,
        'weather_data': weather_data,
        'holiday_data': holiday_data,
        'promotion_data': promotion_data
    }

def extract_store_hierarchy(df):
    """Extract store hierarchy data from the dataset."""
    print("Extracting store hierarchy...")
    
    # Get unique store and city IDs
    if 'store_id' in df.columns and 'city_id' in df.columns:
        store_ids = df[['store_id', 'city_id']].drop_duplicates()
        
        # Generate additional store information
        store_hierarchy = pd.DataFrame({
            'store_id': store_ids['store_id'],
            'city_id': store_ids['city_id'],
            'store_type': [random.choice(['Supermarket', 'Hypermarket', 'Express', 'Convenience']) for _ in range(len(store_ids))],
            'store_name': [f"Store_{store_id}" for store_id in store_ids['store_id']],
            'region': [f"Region_{city_id % 5 + 1}" for city_id in store_ids['city_id']],
            'latitude': np.random.uniform(25.0, 45.0, size=len(store_ids)),
            'longitude': np.random.uniform(110.0, 130.0, size=len(store_ids)),
            'opening_date': [(datetime(2018, 1, 1) + timedelta(days=np.random.randint(0, 365*3))).date() for _ in range(len(store_ids))],
            'created_at': datetime.now()
        })
        
        print(f"Created store hierarchy with {len(store_hierarchy)} records")
        return store_hierarchy
    else:
        print("Required columns for store hierarchy not found.")
        return pd.DataFrame()

def extract_product_hierarchy(df):
    """Extract product hierarchy data from the dataset."""
    print("Extracting product hierarchy...")
    
    # Get unique product IDs and hierarchies
    product_cols = ['product_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']
    if all(col in df.columns for col in product_cols):
        product_ids = df[product_cols].drop_duplicates()
        
        # Generate additional product information
        product_hierarchy = pd.DataFrame({
            'product_id': product_ids['product_id'],
            'management_group_id': product_ids['management_group_id'],
            'first_category_id': product_ids['first_category_id'],
            'second_category_id': product_ids['second_category_id'],
            'third_category_id': product_ids['third_category_id'],
            'product_name': [f"Product_{product_id}" for product_id in product_ids['product_id']],
            'is_fresh': [product_id % 2 == 0 for product_id in product_ids['product_id']],
            'shelf_life_days': [np.random.randint(1, 30) if product_id % 2 == 0 else np.random.randint(30, 365) 
                               for product_id in product_ids['product_id']],
            'unit_cost': np.random.uniform(1.0, 50.0, size=len(product_ids)),
            'unit_price': np.random.uniform(2.0, 100.0, size=len(product_ids)),
            'created_at': datetime.now()
        })
        
        print(f"Created product hierarchy with {len(product_hierarchy)} records")
        return product_hierarchy
    else:
        print("Required columns for product hierarchy not found.")
        return pd.DataFrame()

def transform_sales_data(df):
    """Transform sales data for database ingestion."""
    print("Transforming sales data...")
    
    # Check required columns
    required_cols = ['store_id', 'product_id', 'sale_date', 'sale_amount']
    if not all(col in df.columns for col in required_cols):
        print("Required columns for sales data not found.")
        return pd.DataFrame()
    
    # Select and rename columns for the sales_data table
    sales_data = df[required_cols].copy()
    
    # Add additional columns if they exist
    if 'discount' in df.columns:
        sales_data['discount'] = df['discount']
    else:
        sales_data['discount'] = np.random.uniform(0.0, 0.3, size=len(df))
    
    # Generate sale quantity
    sales_data['sale_qty'] = np.ceil(sales_data['sale_amount'] / np.random.uniform(0.5, 10.0, size=len(sales_data)))
    
    # Add original price
    sales_data['original_price'] = sales_data['sale_amount'] / (1 - sales_data['discount'])
    
    # Add stock information if available
    if 'stock_hour6_22_cnt' in df.columns:
        sales_data['stock_hour6_22_cnt'] = df['stock_hour6_22_cnt']
    else:
        sales_data['stock_hour6_22_cnt'] = np.random.randint(0, 50, size=len(df))
    
    # Split stock counts by time period
    sales_data['stock_hour6_14_cnt'] = np.ceil(sales_data['stock_hour6_22_cnt'] * np.random.uniform(0.3, 0.7, size=len(sales_data)))
    sales_data['stock_hour14_22_cnt'] = sales_data['stock_hour6_22_cnt'] - sales_data['stock_hour6_14_cnt']
    sales_data['stock_hour14_22_cnt'] = sales_data['stock_hour14_22_cnt'].clip(0)
    
    # Add holiday flag
    if 'holiday_flag' in df.columns:
        sales_data['holiday_flag'] = df['holiday_flag']
    else:
        # Use the date to determine if it's a holiday
        us_holidays = holidays.US()
        sales_data['holiday_flag'] = sales_data['sale_date'].apply(lambda x: x in us_holidays)
    
    # Add promotion flag
    if 'promo_flag' in df.columns:
        sales_data['promo_flag'] = df['promo_flag']
    else:
        sales_data['promo_flag'] = np.random.choice([True, False], size=len(df), p=[0.2, 0.8])
    
    # Add timestamp
    sales_data['created_at'] = datetime.now()
    
    print(f"Transformed sales data with {len(sales_data)} records")
    return sales_data

def generate_weather_data(df):
    """Generate weather data based on unique date and city combinations."""
    print("Generating weather data...")
    
    # Check if we have the required columns
    if not all(col in df.columns for col in ['sale_date', 'city_id']):
        print("Required columns for weather data not found.")
        return pd.DataFrame()
    
    # Get unique date-city combinations
    date_city = df[['sale_date', 'city_id']].drop_duplicates()
    
    # Generate weather data
    weather_data = pd.DataFrame({
        'city_id': date_city['city_id'],
        'date': date_city['sale_date'],
        'temp_min': np.random.uniform(-5, 30, size=len(date_city)),
        'temp_max': None,  # Will fill later
        'temp_avg': None,  # Will fill later
        'precipitation': np.random.exponential(0.5, size=len(date_city)),
        'humidity': np.random.uniform(20, 95, size=len(date_city)),
        'wind_speed': np.random.exponential(2, size=len(date_city)),
        'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow', 'Thunderstorm'], 
                                             size=len(date_city), p=[0.4, 0.3, 0.2, 0.05, 0.05]),
        'created_at': datetime.now()
    })
    
    # Ensure max temp is higher than min temp
    temp_diff = np.random.uniform(2, 15, size=len(weather_data))
    weather_data['temp_max'] = weather_data['temp_min'] + temp_diff
    
    # Calculate average temperature
    weather_data['temp_avg'] = (weather_data['temp_min'] + weather_data['temp_max']) / 2
    
    print(f"Generated weather data with {len(weather_data)} records")
    return weather_data

def generate_holiday_data(df):
    """Generate holiday calendar data."""
    print("Generating holiday calendar...")
    
    # Get unique dates from the dataset
    if 'sale_date' not in df.columns:
        print("Required column 'sale_date' not found.")
        return pd.DataFrame()
    
    # Get min and max dates
    min_date = pd.to_datetime(df['sale_date']).min()
    max_date = pd.to_datetime(df['sale_date']).max()
    
    # Use US holidays
    us_holidays = holidays.US(years=range(min_date.year, max_date.year + 1))
    
    # Create holiday DataFrame
    holiday_data = []
    for date, name in us_holidays.items():
        if min_date.date() <= date <= max_date.date():
            holiday_data.append({
                'date': date,
                'holiday_name': name,
                'holiday_type': 'National' if 'day' in name.lower() else 'Religious' if any(r in name.lower() for r in ['christmas', 'easter', 'thanksgiving']) else 'Public',
                'country': 'US',
                'region': '',
                'significance': np.random.randint(1, 6),
                'created_at': datetime.now()
            })
    
    holiday_df = pd.DataFrame(holiday_data)
    print(f"Generated holiday calendar with {len(holiday_df)} records")
    return holiday_df

def generate_promotion_data(df):
    """Generate promotion events data."""
    print("Generating promotion events...")
    
    # Check if we have the required columns
    if not all(col in df.columns for col in ['store_id', 'product_id', 'sale_date']):
        print("Required columns for promotion data not found.")
        return pd.DataFrame()
    
    # Get unique store-product combinations
    store_products = df[['store_id', 'product_id']].drop_duplicates().sample(frac=0.3)
    
    # Get min and max dates
    min_date = pd.to_datetime(df['sale_date']).min()
    max_date = pd.to_datetime(df['sale_date']).max()
    
    # Generate promotion data
    promotion_data = []
    for _, row in store_products.iterrows():
        # Generate 1-3 promotions per store-product
        num_promos = np.random.randint(1, 4)
        for _ in range(num_promos):
            # Random start date within the date range
            start_date = min_date + timedelta(days=np.random.randint(0, (max_date - min_date).days))
            # Promotion length between 1-14 days
            promo_length = np.random.randint(1, 15)
            end_date = start_date + timedelta(days=promo_length)
            
            promotion_data.append({
                'store_id': row['store_id'],
                'product_id': row['product_id'],
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'promotion_type': np.random.choice(['Discount', 'BOGO', '2-for-1', 'Bundle', 'Special Display']),
                'discount_percentage': np.random.uniform(0.05, 0.5),
                'display_location': np.random.choice(['Entrance', 'End Cap', 'Checkout', 'Aisle', 'Special Area']),
                'campaign_id': f"CAMP_{np.random.randint(1000, 10000)}",
                'created_at': datetime.now()
            })
    
    promotion_df = pd.DataFrame(promotion_data)
    print(f"Generated promotion events with {len(promotion_df)} records")
    return promotion_df

def upload_to_supabase(data_dict, batch_size=1000):
    """Upload all datasets to Supabase."""
    print("Uploading data to Supabase...")
    
    # Get engine for direct SQL execution
    engine = get_db_engine()
    
    # Upload store hierarchy
    if 'store_hierarchy' in data_dict and not data_dict['store_hierarchy'].empty:
        print("Uploading store hierarchy...")
        data_dict['store_hierarchy'].to_sql('store_hierarchy', engine, if_exists='append', index=False, 
                                           method='multi', chunksize=batch_size)
    
    # Upload product hierarchy
    if 'product_hierarchy' in data_dict and not data_dict['product_hierarchy'].empty:
        print("Uploading product hierarchy...")
        data_dict['product_hierarchy'].to_sql('product_hierarchy', engine, if_exists='append', index=False, 
                                            method='multi', chunksize=batch_size)
    
    # Upload weather data
    if 'weather_data' in data_dict and not data_dict['weather_data'].empty:
        print("Uploading weather data...")
        data_dict['weather_data'].to_sql('weather_data', engine, if_exists='append', index=False, 
                                        method='multi', chunksize=batch_size)
    
    # Upload holiday data
    if 'holiday_data' in data_dict and not data_dict['holiday_data'].empty:
        print("Uploading holiday data...")
        data_dict['holiday_data'].to_sql('holiday_calendar', engine, if_exists='append', index=False, 
                                        method='multi', chunksize=batch_size)
    
    # Upload promotion data
    if 'promotion_data' in data_dict and not data_dict['promotion_data'].empty:
        print("Uploading promotion data...")
        data_dict['promotion_data'].to_sql('promotion_events', engine, if_exists='append', index=False, 
                                          method='multi', chunksize=batch_size)
    
    # Upload sales data (may be large, so we'll upload in batches)
    if 'sales_data' in data_dict and not data_dict['sales_data'].empty:
        print(f"Uploading sales data in batches of {batch_size}...")
        data = data_dict['sales_data']
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.iloc[i:i+batch_size]
            batch.to_sql('sales_data', engine, if_exists='append', index=False)
    
    print("Data upload complete!")

def refresh_materialized_views(engine):
    """Refresh all materialized views."""
    print("Refreshing materialized views...")
    
    with engine.connect() as conn:
        conn.execute("SELECT refresh_all_mv()")
        conn.commit()
    
    print("Materialized views refreshed successfully.")

def main():
    """Main function to load and process the dataset."""
    try:
        # Load dataset from Hugging Face
        df = load_dataset_from_huggingface()
        if df is None:
            print("Dataset loading failed. Exiting.")
            return
        
        # Get database engine
        engine = get_db_engine()
        
        # Create tables
        create_tables(engine)
        
        # Preprocess data
        processed_data = preprocess_data(df)
        
        # Upload to Supabase
        upload_to_supabase(processed_data)
        
        # Refresh materialized views
        refresh_materialized_views(engine)
        
        print("Dataset loading and processing completed successfully!")
        
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 