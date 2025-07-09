"""
Script to export FreshRetailNet-50K data with exactly matching table columns.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def load_data(limit=None, offset=0):
    """Load data from Hugging Face dataset"""
    print("Loading data from Hugging Face...")
    
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    
    # Get first batch to determine schema
    batch_size = min(1000, limit if limit else 1000)
    first_batch = next(iter(dataset["train"].iter(batch_size=10)))
    df = pd.DataFrame(first_batch)
    
    if offset > 0:
        skip_batches = offset // batch_size
        batch_iter = dataset["train"].iter(batch_size=batch_size)
        # Skip batches
        for _ in range(skip_batches):
            next(batch_iter)
        
        # Get the batch at the offset
        batch = next(batch_iter)
        df = pd.DataFrame(batch)
        
        # Further adjust for offset that isn't exactly at batch boundary
        remaining_offset = offset % batch_size
        if remaining_offset > 0:
            df = df.iloc[remaining_offset:].reset_index(drop=True)
    
    # Apply limit if needed
    if limit and len(df) > limit:
        df = df.head(limit)
    elif limit and len(df) < limit:
        # Load more batches until we reach the limit
        needed = limit - len(df)
        batch_iter = dataset["train"].iter(batch_size=batch_size)
        next(batch_iter)  # Skip first batch
        
        while needed > 0:
            try:
                batch = next(batch_iter)
                batch_df = pd.DataFrame(batch)
                batch_size = min(needed, len(batch_df))
                df = pd.concat([df, batch_df.head(batch_size)], ignore_index=True)
                needed -= batch_size
            except StopIteration:
                break
    
    print(f"Loaded {len(df)} records")
    return df

def process_data_for_sales(df):
    """Process data specifically for sales_data table"""
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Convert string dates to datetime and then to formatted string
    if 'dt' in processed_df.columns:
        # FreshRetailNet uses numeric dt, convert to proper dates
        unique_dates = sorted(processed_df['dt'].unique())
        date_mapping = {
            old_date: pd.to_datetime('2022-01-01') + pd.Timedelta(days=i)
            for i, old_date in enumerate(unique_dates)
        }
        processed_df['dt'] = processed_df['dt'].map(date_mapping)
        processed_df['dt'] = processed_df['dt'].dt.strftime('%Y-%m-%d')
    
    # Handle other columns - keep only what's in the table schema
    cols_to_keep = [
        'city_id', 'store_id', 'management_group_id', 
        'first_category_id', 'second_category_id', 'third_category_id',
        'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt', 
        'discount', 'holiday_flag', 'activity_flag', 'precpt', 
        'avg_temperature', 'avg_humidity', 'avg_wind_level'
    ]
    
    # Only keep columns that exist in the dataframe
    cols_to_keep = [col for col in cols_to_keep if col in processed_df.columns]
    
    # Select only the columns we need
    processed_df = processed_df[cols_to_keep]
    
    return processed_df

def process_data_for_holidays(df):
    """Extract holiday data from sales data"""
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Convert string dates to datetime
    if 'dt' in processed_df.columns:
        # FreshRetailNet uses numeric dt, convert to proper dates
        unique_dates = sorted(processed_df['dt'].unique())
        date_mapping = {
            old_date: pd.to_datetime('2022-01-01') + pd.Timedelta(days=i)
            for i, old_date in enumerate(unique_dates)
        }
        processed_df['dt'] = processed_df['dt'].map(date_mapping)
    
    # Filter only holiday records
    holiday_df = processed_df[processed_df['holiday_flag'] == 1]
    
    # Generate holiday names for holiday flag=1
    holidays = ["New Year's Day", "Martin Luther King Jr. Day", "Presidents' Day", 
                "Memorial Day", "Independence Day", "Labor Day", 
                "Columbus Day", "Veterans Day", "Thanksgiving", "Christmas"]
    
    # Get unique holiday dates
    unique_holiday_dates = sorted(holiday_df['dt'].unique())
    
    # Create holiday DataFrame
    holiday_data = []
    for i, date in enumerate(unique_holiday_dates):
        holiday_name = holidays[i % len(holidays)]
        holiday_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'name': holiday_name,
            'country': 'USA'
        })
    
    # Create DataFrame
    holiday_df = pd.DataFrame(holiday_data)
    
    return holiday_df

def generate_promotions(num_promotions=100):
    """Generate sample promotion data"""
    print(f"Generating {num_promotions} sample promotions...")
    
    # Random product and store IDs
    store_ids = np.random.randint(0, 20, size=num_promotions)
    product_ids = np.random.randint(1, 100, size=num_promotions)
    
    # Random dates in 2022
    start_date = pd.to_datetime('2022-01-01')
    start_days = np.random.randint(0, 330, size=num_promotions)
    durations = np.random.randint(3, 30, size=num_promotions)
    
    start_dates = [
        (start_date + pd.Timedelta(days=int(days))).strftime('%Y-%m-%d') 
        for days in start_days
    ]
    end_dates = [
        (start_date + pd.Timedelta(days=int(days + dur))).strftime('%Y-%m-%d')
        for days, dur in zip(start_days, durations)
    ]
    
    # Promotion types
    promo_types = np.random.choice(
        ["Discount", "BOGO", "Bundle", "Clearance", "New Product"],
        size=num_promotions
    )
    
    # Discounts (0.5 to 0.95, representing 50% to 5% off)
    discounts = np.round(np.random.uniform(0.5, 0.95, size=num_promotions), 2)
    
    # Locations
    locations = np.random.choice(
        ["End Cap", "Front", "Main Aisle", "Register", "Window"], 
        size=num_promotions
    )
    
    # Campaign IDs
    campaign_ids = [f"CAMP{i:04d}" for i in range(1, num_promotions + 1)]
    
    # Create DataFrame
    promo_df = pd.DataFrame({
        'store_id': store_ids,
        'product_id': product_ids,
        'start_date': start_dates,
        'end_date': end_dates,
        'promotion_type': promo_types,
        'discount': discounts,
        'location': locations,
        'campaign_id': campaign_ids
    })
    
    return promo_df

def export_csv(df, output_path):
    """Export DataFrame to CSV"""
    print(f"Exporting {len(df)} records to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Export complete: {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Export data with matching table columns')
    parser.add_argument('--table', type=str, required=True, choices=['sales_data', 'holidays', 'promotions'], 
                        help='Target table to export data for')
    parser.add_argument('--limit', type=int, default=1000, help='Number of records to export')
    parser.add_argument('--offset', type=int, default=0, help='Offset to start from')
    parser.add_argument('--output', type=str, help='Output file path (default: [table_name].csv)')
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        args.output = f"{args.table}.csv"
    
    if args.table == 'sales_data':
        # Load and process sales data
        raw_data = load_data(limit=args.limit, offset=args.offset)
        processed_data = process_data_for_sales(raw_data)
        export_csv(processed_data, args.output)
        
    elif args.table == 'holidays':
        # Generate holiday data
        raw_data = load_data(limit=10000)  # Need a large sample to get enough holidays
        processed_data = process_data_for_holidays(raw_data)
        # Apply limit if specified
        if len(processed_data) > args.limit:
            processed_data = processed_data.head(args.limit)
        export_csv(processed_data, args.output)
        
    elif args.table == 'promotions':
        # Generate promotion data
        processed_data = generate_promotions(num_promotions=args.limit)
        export_csv(processed_data, args.output)

if __name__ == "__main__":
    main() 