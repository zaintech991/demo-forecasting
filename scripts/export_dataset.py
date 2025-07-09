"""
Script to export FreshRetailNet-50K data from Hugging Face to various formats.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import custom modules (if available)
try:
    from utils.mapping_data import CITY_MAPPING, STORE_MAPPING
    HAS_MAPPINGS = True
except ImportError:
    HAS_MAPPINGS = False
    print("Warning: Mapping data not found. City and store IDs won't be mapped to real names.")

def load_huggingface_data(limit=None, offset=0):
    """
    Load FreshRetailNet-50K data from Hugging Face
    
    Args:
        limit: Optional limit on number of records to load
        offset: Optional offset to start loading from
        
    Returns:
        DataFrame with sales data
    """
    print("Loading data from Hugging Face...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        
        # Convert to DataFrame - for large datasets, process in batches
        print("Converting to DataFrame...")
        
        # Calculate how many batches to skip for the offset
        batch_size = 1000
        skip_batches = offset // batch_size
        records_to_skip_in_first_batch = offset % batch_size
        
        # Initialize batch iterator
        batch_iter = dataset["train"].iter(batch_size=batch_size)
        
        # Skip batches for offset
        for _ in range(skip_batches):
            next(batch_iter)
        
        # Get first batch and skip records if needed
        first_batch = next(iter(batch_iter))
        df_train = pd.DataFrame(first_batch)
        
        if records_to_skip_in_first_batch > 0:
            df_train = df_train.iloc[records_to_skip_in_first_batch:].reset_index(drop=True)
        
        # If limit is set and we already have enough data, return
        if limit is not None and len(df_train) >= limit:
            print(f"Limited dataset to {limit} records for testing")
            return df_train.head(limit)
        
        # Otherwise, process more batches until we reach the limit
        if limit is not None:
            needed_records = limit - len(df_train)
            
            while needed_records > 0:
                try:
                    batch = next(batch_iter)
                    df_batch = pd.DataFrame(batch)
                    
                    if len(df_batch) <= needed_records:
                        df_train = pd.concat([df_train, df_batch], ignore_index=True)
                        needed_records -= len(df_batch)
                    else:
                        df_train = pd.concat([df_train, df_batch.head(needed_records)], ignore_index=True)
                        needed_records = 0
                        
                except StopIteration:
                    print("Reached end of dataset")
                    break
            
            print(f"Loaded {len(df_train)} records (offset {offset}, limit {limit})")
        else:
            print("Loading full dataset - this may take a while and consume significant memory...")
            # For full dataset, process in larger batches
            for batch in tqdm(batch_iter):
                df_batch = pd.DataFrame(batch)
                df_train = pd.concat([df_train, df_batch], ignore_index=True)
            
            print(f"Loaded {len(df_train)} records total")
        
        return df_train
    except Exception as e:
        print(f"Error loading data from Hugging Face: {e}")
        sys.exit(1)

def process_data(df):
    """
    Process and transform the data
    
    Args:
        df: Raw DataFrame from Hugging Face
        
    Returns:
        Processed DataFrame
    """
    print("Processing data...")
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert hours_sale sequence to JSON string
    if 'hours_sale' in processed_df.columns:
        processed_df['hours_sale_json'] = processed_df['hours_sale'].apply(lambda x: str(list(x)))
        processed_df.drop('hours_sale', axis=1, inplace=True)
    
    # Convert hours_stock_status sequence to JSON string
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
    
    # Map city IDs to real names if mappings are available
    if HAS_MAPPINGS and 'city_id' in processed_df.columns:
        processed_df['city_name'] = processed_df['city_id'].map(lambda x: CITY_MAPPING.get(x, f"City {x}"))
        
    # Map store IDs to real names if mappings are available
    if HAS_MAPPINGS and 'store_id' in processed_df.columns:
        processed_df['store_name'] = processed_df['store_id'].map(lambda x: STORE_MAPPING.get(x, f"Store {x}"))
    
    return processed_df

def export_data(df, output_path, format='csv'):
    """
    Export processed data to a file
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the output file
        format: Export format (csv, json, parquet)
    """
    print(f"Exporting {len(df)} records to {output_path}...")
    
    if format.lower() == 'csv':
        df.to_csv(output_path, index=False)
    elif format.lower() == 'json':
        df.to_json(output_path, orient='records', lines=True)
    elif format.lower() == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        print(f"Unsupported format: {format}")
        sys.exit(1)
    
    print(f"Data exported successfully to {output_path}")

def extract_holidays(df, output_path):
    """
    Extract and export holiday data
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the holidays file
    """
    print("Extracting holiday data...")
    
    # Extract holiday data
    holiday_df = df[df['holiday_flag'] == 1].dropna(subset=['holiday_name'])
    
    # Get unique combinations of date and holiday name
    holiday_data = holiday_df[['dt', 'holiday_name']].drop_duplicates()
    holiday_data = holiday_data.rename(columns={'dt': 'date', 'holiday_name': 'name'})
    
    # Add country column
    holiday_data['country'] = 'USA'
    
    # Save to CSV
    holiday_data.to_csv(output_path, index=False)
    print(f"Holiday data exported to {output_path}")

def main():
    """Main function to export data"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Export FreshRetailNet-50K data to various formats')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records')
    parser.add_argument('--offset', type=int, default=0, help='Offset to start from')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json', 'parquet'], help='Export format')
    parser.add_argument('--output', type=str, default='data_export.csv', help='Output file path')
    parser.add_argument('--holidays', action='store_true', help='Export holidays to separate file')
    args = parser.parse_args()
    
    # Load data from Hugging Face
    raw_df = load_huggingface_data(limit=args.limit, offset=args.offset)
    
    # Process the data
    processed_df = process_data(raw_df)
    
    # Export the data
    export_data(processed_df, args.output, args.format)
    
    # Export holidays if requested
    if args.holidays:
        holidays_output = os.path.splitext(args.output)[0] + '_holidays.csv'
        extract_holidays(processed_df, holidays_output)
    
    print("Data export completed successfully!")

if __name__ == "__main__":
    main() 