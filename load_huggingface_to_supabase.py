"""
Script to load the FreshRetailNet-50K dataset from Hugging Face directly into Supabase tables.
This script is focused on populating the 'sales_data', 'promotions', and 'holidays' tables.
"""
import os
import sys
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime, timedelta, date
import holidays
import random
import json

# Add project root to Python path
script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_path)
sys.path.append(project_root)
from database.connection import get_supabase_client, get_db_engine
from dotenv import load_dotenv

# Custom JSON encoder to handle date objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

def load_huggingface_dataset(dataset_name="Dingdong-Inc/FreshRetailNet-50K"):
    """
    Load the dataset from Hugging Face.
    
    Args:
        dataset_name (str): The name of the dataset on Hugging Face
        
    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame
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
            
        print(f"Dataset loaded successfully. Available splits: {list(ds.keys())}")
        
        # Use the first available split
        default_split = list(ds.keys())[0]
        df = ds[default_split].to_pandas()
        print(f"Dataset shape: {df.shape}")
        print(f"Dataset columns: {df.columns.tolist()}")
        return df
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def prepare_sales_data(df):
    """
    Prepare the sales data for the 'sales_data' table in Supabase.
    
    Args:
        df (pd.DataFrame): The original dataset
        
    Returns:
        pd.DataFrame: Processed dataframe ready for Supabase sales_data table
    """
    print("Preparing sales data...")
    
    # Create a copy to avoid modifying the original
    sales_df = df.copy()
    
    # Make sure date column is properly formatted
    if 'dt' in sales_df.columns:
        sales_df['dt'] = pd.to_datetime(sales_df['dt']).dt.strftime('%Y-%m-%d')
    elif 'sale_date' in sales_df.columns:
        sales_df['dt'] = pd.to_datetime(sales_df['sale_date']).dt.strftime('%Y-%m-%d')
        sales_df.drop('sale_date', axis=1, inplace=True)
    
    # Map columns to match Supabase schema
    sales_columns = {
        'city_id': 'city_id',
        'store_id': 'store_id',
        'management_group_id': 'management_group_id',
        'first_category_id': 'first_category_id',
        'second_category_id': 'second_category_id',
        'third_category_id': 'third_category_id',
        'product_id': 'product_id',
        'dt': 'dt',
        'sale_amount': 'sale_amount',
        'stock_hour6_22_cnt': 'stock_hour6_22_cnt',
        'discount': 'discount'
    }
    
    # Select only columns that exist in the dataframe
    available_columns = {k: v for k, v in sales_columns.items() if k in sales_df.columns}
    
    # Create the final dataframe with only the columns we need
    final_df = sales_df[list(available_columns.keys())].rename(columns=available_columns)
    
    # Add holiday flag if it exists
    if 'holiday_flag' in sales_df.columns:
        final_df['holiday_flag'] = sales_df['holiday_flag']
    
    # Add activity flag if it exists
    if 'activity_flag' in sales_df.columns:
        final_df['activity_flag'] = sales_df['activity_flag']
    
    # Add weather data if it exists
    weather_columns = ['precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    for col in weather_columns:
        if col in sales_df.columns:
            final_df[col] = sales_df[col]
    
    # Add creation timestamp
    final_df['created_at'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    
    print(f"Prepared sales data with {len(final_df)} records and columns: {final_df.columns.tolist()}")
    return final_df

def prepare_promotions_data(df):
    """
    Generate promotion data based on the dataset.
    
    Args:
        df (pd.DataFrame): The original dataset
        
    Returns:
        pd.DataFrame: Processed dataframe ready for Supabase promotions table
    """
    print("Preparing promotions data...")
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure date column is properly formatted
    if 'dt' in df_copy.columns:
        df_copy['dt'] = pd.to_datetime(df_copy['dt'])
    elif 'sale_date' in df_copy.columns:
        df_copy['dt'] = pd.to_datetime(df_copy['sale_date'])
    
    # Check if dataset already has promotion information
    if 'promotion_type' in df_copy.columns and 'discount' in df_copy.columns:
        promo_data = df_copy[df_copy['discount'] > 0].copy()
        
        # Create a unique set of store-product-date combinations with promotions
        promo_combos = promo_data[['store_id', 'product_id', 'dt']].drop_duplicates()
        
        # Group dates into promotion periods
        promo_periods = []
        for store_id in promo_combos['store_id'].unique():
            store_promos = promo_combos[promo_combos['store_id'] == store_id]
            
            for product_id in store_promos['product_id'].unique():
                product_promos = store_promos[store_promos['product_id'] == product_id]
                dates = sorted(pd.to_datetime(product_promos['dt']))
                
                # Group consecutive dates
                if len(dates) > 0:
                    start_date = dates[0]
                    current_date = start_date
                    
                    for next_date in dates[1:]:
                        if (next_date - current_date).days <= 1:
                            # Continue the current promotion period
                            current_date = next_date
                        else:
                            # End the current period and start a new one
                            promo_periods.append({
                                'store_id': int(store_id),
                                'product_id': int(product_id),
                                'start_date': start_date.strftime('%Y-%m-%d'),
                                'end_date': current_date.strftime('%Y-%m-%d'),
                                'promotion_type': 'Discount',
                                'discount': float(df_copy[(df_copy['store_id'] == store_id) & 
                                              (df_copy['product_id'] == product_id) & 
                                              (pd.to_datetime(df_copy['dt']) >= start_date) & 
                                              (pd.to_datetime(df_copy['dt']) <= current_date)]['discount'].mean()),
                                'location': 'Store',
                                'campaign_id': f"CAMP_{np.random.randint(1000, 10000)}",
                                'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                            })
                            
                            # Start a new period
                            start_date = next_date
                            current_date = next_date
                    
                    # Add the last period
                    promo_periods.append({
                        'store_id': int(store_id),
                        'product_id': int(product_id),
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': current_date.strftime('%Y-%m-%d'),
                        'promotion_type': 'Discount',
                        'discount': float(df_copy[(df_copy['store_id'] == store_id) & 
                                      (df_copy['product_id'] == product_id) & 
                                      (pd.to_datetime(df_copy['dt']) >= start_date) & 
                                      (pd.to_datetime(df_copy['dt']) <= current_date)]['discount'].mean()),
                        'location': 'Store',
                        'campaign_id': f"CAMP_{np.random.randint(1000, 10000)}",
                        'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                    })
    
    else:
        # Generate synthetic promotion data
        # Sample 30% of store-product combinations
        store_products = df_copy[['store_id', 'product_id']].drop_duplicates().sample(frac=0.3)
        
        # Get min and max dates
        min_date = pd.to_datetime(df_copy['dt'] if 'dt' in df_copy.columns else df_copy['sale_date']).min()
        max_date = pd.to_datetime(df_copy['dt'] if 'dt' in df_copy.columns else df_copy['sale_date']).max()
        
        # Generate promotion data
        promo_periods = []
        for _, row in store_products.iterrows():
            # Generate 1-3 promotions per store-product
            num_promos = np.random.randint(1, 4)
            for _ in range(num_promos):
                # Random start date within the date range
                start_date = min_date + timedelta(days=np.random.randint(0, (max_date - min_date).days))
                # Promotion length between 1-14 days
                promo_length = np.random.randint(1, 15)
                end_date = start_date + timedelta(days=promo_length)
                
                promo_periods.append({
                    'store_id': int(row['store_id']),
                    'product_id': int(row['product_id']),
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'promotion_type': np.random.choice(['Discount', 'BOGO', '2-for-1', 'Bundle']),
                    'discount': float(np.random.uniform(0.05, 0.5)),
                    'location': np.random.choice(['Entrance', 'End Cap', 'Checkout', 'Aisle']),
                    'campaign_id': f"CAMP_{np.random.randint(1000, 10000)}",
                    'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                })
    
    promotions_df = pd.DataFrame(promo_periods)
    print(f"Prepared promotions data with {len(promotions_df)} records")
    return promotions_df

def prepare_holidays_data(df):
    """
    Generate holiday data based on the dataset.
    
    Args:
        df (pd.DataFrame): The original dataset
        
    Returns:
        pd.DataFrame: Processed dataframe ready for Supabase holidays table
    """
    print("Preparing holidays data...")
    
    # Get min and max dates from the dataset
    date_col = 'dt' if 'dt' in df.columns else 'sale_date'
    min_date = pd.to_datetime(df[date_col]).min()
    max_date = pd.to_datetime(df[date_col]).max()
    
    # Use US holidays for the date range
    us_holidays = holidays.US(years=range(min_date.year, max_date.year + 1))
    
    # Create holiday DataFrame
    holiday_data = []
    for holiday_date, name in us_holidays.items():
        if min_date.date() <= holiday_date <= max_date.date():
            holiday_data.append({
                'date': holiday_date.strftime('%Y-%m-%d'),
                'name': name,
                'country': 'USA',
                'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            })
    
    holidays_df = pd.DataFrame(holiday_data)
    print(f"Prepared holidays data with {len(holidays_df)} records")
    return holidays_df

def upload_to_supabase(sales_df, promotions_df, holidays_df, batch_size=100):
    """
    Upload the prepared data to Supabase tables using REST API.
    
    Args:
        sales_df (pd.DataFrame): Prepared sales data
        promotions_df (pd.DataFrame): Prepared promotions data
        holidays_df (pd.DataFrame): Prepared holidays data
        batch_size (int): Batch size for uploads
    """
    print("Uploading data to Supabase via REST API...")
    
    # Load environment variables
    load_dotenv()
    
    # Get supabase client
    supabase = get_supabase_client()
    
    if supabase is None:
        print("Failed to connect to Supabase. Exiting.")
        return False
    
    # Set the schema to 'public'
    supabase_schema = 'public'
    
    # Upload holidays data
    if not holidays_df.empty:
        print(f"Uploading {len(holidays_df)} holiday records...")
        try:
            # Convert to records
            holiday_records = holidays_df.to_dict(orient='records')
            
            # Upload in batches
            for i in range(0, len(holiday_records), batch_size):
                batch = holiday_records[i:i+batch_size]
                print(f"Uploading holidays batch {i//batch_size + 1}/{(len(holiday_records) + batch_size - 1)//batch_size}")
                response = supabase.schema(supabase_schema).table('holidays').insert(batch).execute()
                if hasattr(response, 'error') and response.error:
                    print(f"Error uploading holiday batch: {response.error}")
                    
                # Print response for debugging
                data = response.data if hasattr(response, 'data') else None
                print(f"Response: {data}")
            
            print("Holiday data upload complete")
        except Exception as e:
            print(f"Error uploading holiday data: {str(e)}")
    
    # Upload promotions data
    if not promotions_df.empty:
        print(f"Uploading {len(promotions_df)} promotion records...")
        try:
            # Convert to records
            promo_records = promotions_df.to_dict(orient='records')
            
            # Upload in batches
            for i in range(0, len(promo_records), batch_size):
                batch = promo_records[i:i+batch_size]
                print(f"Uploading promotions batch {i//batch_size + 1}/{(len(promo_records) + batch_size - 1)//batch_size}")
                response = supabase.schema(supabase_schema).table('promotions').insert(batch).execute()
                if hasattr(response, 'error') and response.error:
                    print(f"Error uploading promotion batch: {response.error}")
                
                # Print first response for debugging
                if i == 0:
                    data = response.data if hasattr(response, 'data') else None
                    print(f"First promotions response: {data}")
            
            print("Promotions data upload complete")
        except Exception as e:
            print(f"Error uploading promotions data: {str(e)}")
    
    # Upload sales data (may be large, so use smaller batches)
    if not sales_df.empty:
        print(f"Uploading {len(sales_df)} sales records in batches of {batch_size}...")
        try:
            # Sample a smaller subset for testing if the dataset is large
            if len(sales_df) > 1000:
                print("Data is large. Using a sample of 1,000 records for initial upload.")
                upload_df = sales_df.sample(1000)
            else:
                upload_df = sales_df
            
            # Convert to records
            sales_records = upload_df.to_dict(orient='records')
            
            # Upload in batches with progress bar
            for i in tqdm(range(0, len(sales_records), batch_size)):
                batch = sales_records[i:i+batch_size]
                response = supabase.schema(supabase_schema).table('sales_data').insert(batch).execute()
                if hasattr(response, 'error') and response.error:
                    print(f"Error uploading sales batch: {response.error}")
                    # Print response for debugging
                    data = response.data if hasattr(response, 'data') else None
                    print(f"Response: {data}")
                    break
                
                # Print first response for debugging
                if i == 0:
                    data = response.data if hasattr(response, 'data') else None
                    print(f"First sales response: {data}")
            
            print("Sales data sample upload complete")
        except Exception as e:
            print(f"Error uploading sales data: {str(e)}")
    
    print("Data upload to Supabase complete!")
    return True

def main():
    """Main function to load data from Hugging Face and upload to Supabase."""
    try:
        # Load dataset from Hugging Face
        df = load_huggingface_dataset()
        if df is None:
            print("Failed to load dataset. Exiting.")
            return
        
        # Prepare data for each table
        sales_df = prepare_sales_data(df)
        promotions_df = prepare_promotions_data(df)
        holidays_df = prepare_holidays_data(df)
        
        # Upload data to Supabase
        upload_to_supabase(sales_df, promotions_df, holidays_df)
        
        print("Data loading completed successfully!")
        
    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 