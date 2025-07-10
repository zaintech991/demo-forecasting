"""
Upload split CSV files to Supabase using the REST API.
This script allows uploading large datasets in chunks that were created with split_csv.py.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import get_supabase_client


def upload_csv_to_supabase(csv_file, table_name, batch_size=1000):
    """
    Upload a CSV file to Supabase in batches.

    Args:
        csv_file (str): Path to the CSV file
        table_name (str): Name of the target table
        batch_size (int): Number of rows to upload in each batch

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Uploading {csv_file} to {table_name}...")

    # Get Supabase client
    supabase = get_supabase_client()
    if supabase is None:
        print("Failed to get Supabase client.")
        return False

    # Get file size
    file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Load CSV in chunks to avoid memory issues
    success = True
    total_rows = 0
    uploaded_rows = 0

    # First count total rows for progress reporting
    for chunk in pd.read_csv(csv_file, chunksize=batch_size):
        total_rows += len(chunk)

    # Reset and read again for actual upload
    with tqdm(total=total_rows, desc=f"Uploading {os.path.basename(csv_file)}") as pbar:
        for chunk in pd.read_csv(csv_file, chunksize=batch_size):
            # Process each batch
            for i in range(0, len(chunk), batch_size):
                batch = chunk.iloc[i : i + batch_size]

                # Convert DataFrame to list of dictionaries
                records = batch.to_dict(orient="records")

                try:
                    # Use Supabase REST API to insert data
                    response = supabase.table(table_name).insert(records).execute()

                    # Check for errors
                    if hasattr(response, "error") and response.error:
                        print(f"Error uploading batch: {response.error}")
                        success = False
                    else:
                        uploaded_rows += len(batch)
                        pbar.update(len(batch))

                except Exception as e:
                    print(f"Exception during batch upload: {str(e)}")
                    success = False

                # Add a small delay to avoid rate limiting
                time.sleep(0.1)

    if success:
        print(f"Successfully uploaded {uploaded_rows} rows from {csv_file}")
    else:
        print(f"Errors occurred during upload from {csv_file}")
        print(f"Successfully uploaded {uploaded_rows} out of {total_rows} rows")

    return success


def upload_all_chunks(
    chunks_dir, table_name="sales_data", file_pattern="sales_data_part_*.csv"
):
    """
    Upload all chunk files from a directory.

    Args:
        chunks_dir (str): Directory containing the chunks
        table_name (str): Name of the target table
        file_pattern (str): Pattern to match chunk files

    Returns:
        bool: True if all successful, False otherwise
    """
    import glob

    # Get all matching files
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, file_pattern)))

    if not chunk_files:
        print(f"No files matching pattern {file_pattern} found in {chunks_dir}")
        return False

    print(f"Found {len(chunk_files)} files to upload")

    # Upload each file
    all_success = True
    for i, file in enumerate(chunk_files):
        print(f"\nUploading file {i+1}/{len(chunk_files)}: {os.path.basename(file)}")
        success = upload_csv_to_supabase(file, table_name)
        if not success:
            all_success = False
            print(f"Failed to upload {file} completely")

            # Ask if we should continue
            response = input("Continue with next file? (y/n): ")
            if response.lower() != "y":
                print("Upload aborted.")
                return False

    return all_success


def main():
    """Main function to upload all chunks."""
    # Load environment variables
    load_dotenv()

    # Define chunks directory
    chunks_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_export",
        "sales_data_chunks",
    )

    # Check if directory exists
    if not os.path.exists(chunks_dir):
        print(f"Error: Chunks directory {chunks_dir} not found.")
        return

    # Ask which table to use
    table_name = (
        input("Enter target table name (default: sales_data): ").strip() or "sales_data"
    )

    # Ask for batch size
    try:
        batch_size = int(
            input("Enter batch size for upload (default: 1000): ").strip() or "1000"
        )
    except ValueError:
        batch_size = 1000
        print("Invalid batch size, using default of 1000")

    # Ask which files to upload
    file_pattern = (
        input("Enter file pattern to match (default: sales_data_part_*.csv): ").strip()
        or "sales_data_part_*.csv"
    )

    # Start upload
    success = upload_all_chunks(chunks_dir, table_name, file_pattern)

    if success:
        print("\nAll files uploaded successfully!")
    else:
        print("\nSome files failed to upload completely. Check the logs for details.")


if __name__ == "__main__":
    main()
