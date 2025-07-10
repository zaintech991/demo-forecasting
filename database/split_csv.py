"""
Split large CSV files into smaller chunks for easier uploading to Supabase.
This script specifically targets the sales_data_csv.csv file which is too large
for the Supabase dashboard's 100MB upload limit.
"""

import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm


def split_csv_file(
    input_file, output_dir, max_rows_per_file=500000, prefix="sales_data_part"
):
    """
    Split a large CSV file into smaller chunks.

    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory to save the output files
        max_rows_per_file (int): Maximum number of rows per output file
        prefix (str): Prefix for output filenames

    Returns:
        list: List of created file paths
    """
    print(f"Splitting {input_file} into smaller chunks...")

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get total number of rows to calculate number of chunks
    total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract 1 for header
    num_chunks = math.ceil(total_rows / max_rows_per_file)

    print(f"Total rows: {total_rows}")
    print(
        f"Will create {num_chunks} files with approximately {max_rows_per_file} rows each"
    )

    # Read and split the CSV in chunks to avoid loading the entire file into memory
    output_files = []

    # Use pandas' chunksize parameter to read in chunks
    for i, chunk in enumerate(
        tqdm(
            pd.read_csv(input_file, chunksize=max_rows_per_file),
            total=num_chunks,
            desc="Creating CSV chunks",
        )
    ):
        # Generate output filename
        output_file = os.path.join(output_dir, f"{prefix}_{i+1:03d}.csv")

        # Write chunk to CSV
        chunk.to_csv(output_file, index=False)
        output_files.append(output_file)

        # Print file size
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Created {output_file} ({file_size_mb:.2f} MB)")

    print(f"Successfully split {input_file} into {len(output_files)} files")
    return output_files


def create_merge_script(output_files, output_dir, table_name="sales_data"):
    """
    Create a SQL script that can merge all the data into the main table.

    Args:
        output_files (list): List of CSV file paths
        output_dir (str): Directory to save the SQL script
        table_name (str): Target table name
    """
    sql_file = os.path.join(output_dir, "merge_csv_files.sql")

    with open(sql_file, "w") as f:
        f.write("-- SQL script to load all CSV chunks into the sales_data table\n\n")

        for csv_file in output_files:
            filename = os.path.basename(csv_file)
            f.write(f"-- Load data from {filename}\n")
            f.write(f"COPY {table_name} (\n")
            f.write("    city_id, store_id, management_group_id,\n")
            f.write("    first_category_id, second_category_id, third_category_id,\n")
            f.write("    product_id, dt, sale_amount, stock_hour6_22_cnt,\n")
            f.write("    discount, holiday_flag, activity_flag, precpt,\n")
            f.write("    avg_temperature, avg_humidity, avg_wind_level\n")
            f.write(")\n")
            f.write(f"FROM '/path/to/{filename}'\n")
            f.write("DELIMITER ','\n")
            f.write("CSV HEADER;\n\n")

    print(f"Created SQL merge script: {sql_file}")


def main():
    """Main function to split the sales data CSV file."""
    # Define input and output paths
    input_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_export",
        "sales_data_csv.csv",
    )
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_export",
        "sales_data_chunks",
    )

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    # Get file size
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Input file size: {file_size_mb:.2f} MB")

    # Calculate optimal chunk size (aim for ~80MB per file to stay safely under 100MB)
    total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract 1 for header
    mb_per_row = file_size_mb / total_rows
    optimal_rows = int(80 / mb_per_row)

    print(f"Calculated optimal chunk size: ~{optimal_rows} rows per file")

    # Split the file
    output_files = split_csv_file(
        input_file, output_dir, max_rows_per_file=optimal_rows
    )

    # Create merge script
    create_merge_script(output_files, output_dir)

    print("CSV splitting complete!")
    print(f"You can now upload each file in {output_dir} to Supabase")
    print(f"Each file should be under 100MB to meet the dashboard upload limit")


if __name__ == "__main__":
    main()
