"""
Script to load the complete FreshRetailNet-50K dataset from Hugging Face and ingest it into Supabase.
This script ensures the entire dataset is loaded with all features and associations.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import get_db_engine, get_supabase_client
from database.load_dataset import (
    load_dataset_from_huggingface,
    create_tables,
    preprocess_data,
    upload_to_supabase,
    refresh_materialized_views
)

def load_full_dataset():
    """
    Load the complete FreshRetailNet-50K dataset into Supabase.
    This function ensures all data is properly loaded, transformed, and associated.
    """
    print("Starting full dataset load process...")
    
    try:
        # Load dataset from Hugging Face
        print("Loading dataset: Dingdong-Inc/FreshRetailNet-50K")
        df = load_dataset_from_huggingface()
        if df is None:
            print("Dataset loading failed. Exiting.")
            return
        
        print(f"Dataset loaded with {len(df)} records.")
        print(f"Dataset columns: {df.columns.tolist()}")
        
        # Get database engine
        engine = get_db_engine()
        
        # Create database tables (will only create if they don't exist)
        create_tables(engine)
        
        # Preprocess data for all tables
        print("Preprocessing dataset for Supabase ingestion...")
        processed_data = preprocess_data(df)
        
        # Show statistics for each table
        for table_name, data in processed_data.items():
            if data is not None and not data.empty:
                print(f"Prepared {table_name} with {len(data)} records")
        
        # Upload all data to Supabase (with larger batch size for efficiency)
        print("Uploading complete dataset to Supabase...")
        upload_to_supabase(processed_data, batch_size=5000)
        
        # Refresh materialized views to update aggregated data
        refresh_materialized_views(engine)
        
        print("Full dataset loading completed successfully!")
        
    except Exception as e:
        print(f"Error during full dataset loading: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_full_dataset() 