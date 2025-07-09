"""
Script to load the FreshRetailNet-50K dataset from Hugging Face to Supabase tables.
This version uses the Supabase REST API directly instead of SQLAlchemy to avoid connection issues.
It focuses on loading data to three target tables: sales_data, promotions, and holidays.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datasets import load_dataset
import json
from tqdm import tqdm
import holidays
import random
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import get_supabase_client

def load_dataset_from_huggingface(dataset_name="Dingdong-Inc/FreshRetailNet-50K", split="train"):
    """
    Load a dataset from Hugging Face.
    
    Args:
        dataset_name (str): The name of the dataset on Hugging Face
        split (str): The dataset split to use (train, test, etc.)
        
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame
    """
    print(f"Loading dataset '{dataset_name}' from Hugging Face...")
    try:
        # Check if Hugging Face token is available
        token = os.getenv('HUGGINGFACE_TOKEN')
        
        # Load dataset with token if available
        if token:
            ds = load_dataset(dataset_name, use_auth_token=token)
        else:
            ds = load_dataset(dataset_name)
            
        print(f"Dataset loaded successfully. Available splits: {ds.keys()}")
        
        # Convert to pandas DataFrame
        if split in ds:
            df = ds[split].to_pandas()
            print(f"Dataset shape: {df.shape}")
            print(f"Dataset columns: {df.columns.tolist()}")
            return df
        else:
            print(f"Split '{split}' not found in the dataset. Available splits: {ds.keys()}")
            if len(ds.keys()) > 0:
                default_split = list(ds.keys())[0]
                print(f"Using default split '{default_split}' instead.")
                df = ds[default_split].to_pandas()
                print(f"Dataset shape: {df.shape}")
                return df
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("If this is a private dataset, you may need to log in to Hugging Face using 'huggingface-cli login'")
        print("or set the HUGGINGFACE_TOKEN environment variable.")
        return None

def preprocess_sales_data(df):
    """
    Transform sales data for Supabase ingestion.
    
    Args:
        df (pd.DataFrame): The raw dataset
        
    Returns:
        pd.DataFrame: Processed sales data
    """
    print("Processing sales data...")
    
    # Rename columns to match the expected schema
    # Map Hugging Face dataset columns to our schema
    column_mapping = {
        'store_id': 'store_id',
        'product_id': 'product_id',
        'dt': 'sale_date',
        'sale_amount': 'sale_amount',
        'discount': 'discount',
        'stock_hour6_22_cnt': 'stock_hour6_22_cnt',
        'holiday_flag': 'holiday_flag',
        'activity_flag': 'promo_flag'  # Using activity_flag as promo_flag
    }
    
    # Check if required columns exist in the dataset
    missing_cols = [col for col in ['store_id', 'product_id', 'dt', 'sale_amount'] 
                    if col not in df.columns]
    if missing_cols:
        print(f"Required columns for sales data not found: {missing_cols}")
        return pd.DataFrame()
    
    # Create a new DataFrame with renamed columns
    sales_data = pd.DataFrame()
    for target_col, source_col in column_mapping.items():
        if source_col in df.columns:
            sales_data[target_col] = df[source_col]
    
    # Ensure data types and fix column names
    # Make sure 'dt' is converted to 'sale_date'
    if 'dt' in df.columns and 'sale_date' not in sales_data.columns:
        sales_data['sale_date'] = df['dt']
    
    # Ensure sale_date is in ISO format string for JSON serialization
    if 'sale_date' in sales_data.columns:
        sales_data['sale_date'] = pd.to_datetime(sales_data['sale_date']).dt.strftime('%Y-%m-%d')
    
    # Add discount if available, otherwise generate it
    if 'discount' not in sales_data.columns and 'discount' in df.columns:
        sales_data['discount'] = df['discount']
    elif 'discount' not in sales_data.columns:
        sales_data['discount'] = np.random.uniform(0.0, 0.3, size=len(df))
    
    # Generate sale quantity
    sales_data['sale_qty'] = np.ceil(sales_data['sale_amount'] / np.random.uniform(0.5, 10.0, size=len(sales_data))).astype(int)
    
    # Add original price
    sales_data['original_price'] = sales_data['sale_amount'] / (1 - sales_data['discount'])
    
    # Add stock information if available
    if 'stock_hour6_22_cnt' not in sales_data.columns and 'stock_hour6_22_cnt' in df.columns:
        sales_data['stock_hour6_22_cnt'] = df['stock_hour6_22_cnt']
    elif 'stock_hour6_22_cnt' not in sales_data.columns:
        sales_data['stock_hour6_22_cnt'] = np.random.randint(0, 50, size=len(df))
    
    # Split stock counts by time period
    sales_data['stock_hour6_14_cnt'] = np.ceil(sales_data['stock_hour6_22_cnt'] * np.random.uniform(0.3, 0.7, size=len(sales_data))).astype(int)
    sales_data['stock_hour14_22_cnt'] = sales_data['stock_hour6_22_cnt'] - sales_data['stock_hour6_14_cnt']
    sales_data['stock_hour14_22_cnt'] = sales_data['stock_hour14_22_cnt'].clip(0).astype(int)
    
    # Add holiday flag if available
    if 'holiday_flag' not in sales_data.columns and 'holiday_flag' in df.columns:
        sales_data['holiday_flag'] = df['holiday_flag'].astype(bool)
    elif 'holiday_flag' not in sales_data.columns:
        # Use the date to determine if it's a holiday
        us_holidays = holidays.US()
        sales_data['holiday_flag'] = pd.to_datetime(sales_data['sale_date']).apply(lambda x: x in us_holidays)
    
    # Add promotion flag if available
    if 'promo_flag' not in sales_data.columns and 'activity_flag' in df.columns:
        sales_data['promo_flag'] = df['activity_flag'].astype(bool)
    elif 'promo_flag' not in sales_data.columns:
        sales_data['promo_flag'] = np.random.choice([True, False], size=len(df), p=[0.2, 0.8])
    
    # Sample the data if it's too large (for testing)
    if len(sales_data) > 10000:
        print(f"Sampling sales data from {len(sales_data)} to 10000 records for testing")
        sales_data = sales_data.sample(10000, random_state=42)
    
    print(f"Processed sales data with {len(sales_data)} records")
    return sales_data

def generate_holiday_data(df):
    """
    Generate holiday calendar data.
    
    Args:
        df (pd.DataFrame): The raw dataset
        
    Returns:
        pd.DataFrame: Holiday data
    """
    print("Generating holiday calendar...")
    
    # Get unique dates from the dataset
    if 'dt' not in df.columns:
        print("Required column 'dt' not found.")
        return pd.DataFrame()
    
    # Get min and max dates
    min_date = pd.to_datetime(df['dt']).min()
    max_date = pd.to_datetime(df['dt']).max()
    
    # Use US holidays
    us_holidays = holidays.US(years=range(min_date.year, max_date.year + 1))
    
    # Create holiday DataFrame
    holiday_data = []
    for date, name in us_holidays.items():
        if min_date.date() <= date <= max_date.date():
            holiday_data.append({
                'date': date.strftime('%Y-%m-%d'),  # Convert to string for JSON serialization
                'holiday_name': name,
                'holiday_type': 'National' if 'day' in name.lower() else 'Religious' if any(r in name.lower() for r in ['christmas', 'easter', 'thanksgiving']) else 'Public',
                'country': 'US',
                'region': '',
                'significance': np.random.randint(1, 6)
            })
    
    holiday_df = pd.DataFrame(holiday_data)
    print(f"Generated holiday calendar with {len(holiday_df)} records")
    return holiday_df

def generate_promotion_data(df):
    """
    Generate promotion events data based on activity_flag.
    
    Args:
        df (pd.DataFrame): The raw dataset
        
    Returns:
        pd.DataFrame: Promotion data
    """
    print("Generating promotion events...")
    
    # Check if we have the required columns
    if not all(col in df.columns for col in ['store_id', 'product_id', 'dt']):
        print("Required columns for promotion data not found.")
        return pd.DataFrame()
    
    # Filter to only include rows with activity_flag=True if it exists
    if 'activity_flag' in df.columns:
        promo_df = df[df['activity_flag'] == True].copy()
    else:
        # Otherwise, randomly sample from the dataset
        promo_df = df.sample(frac=0.1, random_state=42)
    
    # Group by store, product, and date to find continuous promotion periods
    promo_df['dt'] = pd.to_datetime(promo_df['dt'])
    promo_df = promo_df.sort_values(['store_id', 'product_id', 'dt'])
    
    # Get unique store-product combinations
    store_products = promo_df[['store_id', 'product_id']].drop_duplicates().sample(min(1000, len(promo_df[['store_id', 'product_id']].drop_duplicates())))  # Limit to 1000 for testing
    
    # Generate promotion data
    promotion_data = []
    for _, row in store_products.iterrows():
        # Filter data for this store-product combination
        sp_data = promo_df[(promo_df['store_id'] == row['store_id']) & 
                           (promo_df['product_id'] == row['product_id'])]
        
        if len(sp_data) > 0:
            # Generate 1-3 promotions per store-product
            num_promos = min(3, len(sp_data) // 3 + 1)
            for _ in range(num_promos):
                # Pick a random start date from this store-product's data
                start_idx = np.random.randint(0, len(sp_data))
                start_date = sp_data.iloc[start_idx]['dt']
                
                # Promotion length between 1-14 days
                promo_length = np.random.randint(1, 15)
                end_date = start_date + timedelta(days=promo_length)
                
                promotion_data.append({
                    'store_id': int(row['store_id']),
                    'product_id': int(row['product_id']),
                    'start_date': start_date.strftime('%Y-%m-%d'),  # Convert to string for JSON serialization
                    'end_date': end_date.strftime('%Y-%m-%d'),  # Convert to string for JSON serialization
                    'promotion_type': np.random.choice(['Discount', 'BOGO', '2-for-1', 'Bundle', 'Special Display']),
                    'discount_percentage': float(np.random.uniform(0.05, 0.5)),
                    'display_location': np.random.choice(['Entrance', 'End Cap', 'Checkout', 'Aisle', 'Special Area']),
                    'campaign_id': f"CAMP_{np.random.randint(1000, 10000)}"
                })
    
    promotion_df = pd.DataFrame(promotion_data)
    print(f"Generated promotion events with {len(promotion_df)} records")
    return promotion_df

def upload_to_supabase(table_name, data_df, batch_size=1000):
    """
    Upload data to Supabase using the REST API.
    
    Args:
        table_name (str): The name of the table
        data_df (pd.DataFrame): The data to upload
        batch_size (int): Batch size for uploads
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Get Supabase client
    supabase = get_supabase_client()
    if supabase is None:
        print("Failed to get Supabase client.")
        return False
    
    if data_df.empty:
        print(f"No data to upload for {table_name}")
        return True
    
    print(f"Uploading {len(data_df)} records to {table_name} in batches of {batch_size}...")
    
    # Debug the Supabase client
    print(f"Supabase client URL: {supabase.supabase_url}")
    
    # Try a simple query first to check connection
    try:
        response = supabase.table(table_name).select("count").limit(1).execute()
        print(f"Test query response: {response}")
    except Exception as e:
        print(f"Test query failed: {str(e)}")
    
    # Reduce batch size for initial testing
    test_batch_size = min(batch_size, 5)
    if len(data_df) > test_batch_size:
        test_batch = data_df.iloc[:test_batch_size]
        print(f"Testing with small batch of {test_batch_size} records first")
    else:
        test_batch = data_df
    
    # Convert test batch to list of dictionaries
    test_records = test_batch.to_dict(orient='records')
    print(f"Sample record: {test_records[0]}")
    
    # Try with table method
    success = True
    try:
        response = supabase.table(table_name).insert(test_records).execute()
        print(f"Test insert response: {response}")
        if hasattr(response, 'error') and response.error:
            print(f"Error in test upload: {response.error}")
            success = False
    except Exception as e:
        print(f"Exception during test upload: {str(e)}")
        success = False
    
    if not success:
        print("Test upload failed. Not proceeding with full upload.")
        return False
    
    # Proceed with full upload if test was successful
    for i in tqdm(range(0, len(data_df), batch_size)):
        batch = data_df.iloc[i:i+batch_size]
        
        # Convert DataFrame to list of dictionaries
        records = batch.to_dict(orient='records')
        
        try:
            # Use Supabase REST API to insert data
            response = supabase.table(table_name).insert(records).execute()
            
            # Check for errors
            if hasattr(response, 'error') and response.error:
                print(f"Error uploading batch {i//batch_size + 1}: {response.error}")
                success = False
        except Exception as e:
            print(f"Exception during batch {i//batch_size + 1} upload: {str(e)}")
            success = False
    
    if success:
        print(f"Successfully uploaded data to {table_name}")
    else:
        print(f"Errors occurred during upload to {table_name}")
    
    return success

def main():
    """Main function to load and process the dataset."""
    # Load environment variables
    load_dotenv()
    
    try:
        # Load dataset from Hugging Face
        df = load_dataset_from_huggingface()
        if df is None:
            print("Dataset loading failed. Exiting.")
            return
        
        # Process data for each table
        sales_data = preprocess_sales_data(df)
        holiday_data = generate_holiday_data(df)
        promotion_data = generate_promotion_data(df)
        
        # Upload data to Supabase
        # Starting with smaller tables first
        if not holiday_data.empty:
            upload_to_supabase('holiday_calendar', holiday_data, batch_size=100)
        
        if not promotion_data.empty:
            upload_to_supabase('promotion_events', promotion_data, batch_size=500)
        
        if not sales_data.empty:
            upload_to_supabase('sales_data', sales_data, batch_size=1000)
        
        print("Dataset loading and processing completed!")
        
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 