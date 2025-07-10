"""
Export data from the Hugging Face dataset to CSV files for direct import to Supabase.
Creates three files: sales_data_csv.csv, promotion_csv.csv, and holidays_csv.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datasets import load_dataset
import holidays
import random
import csv
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dataset_from_huggingface(
    dataset_name="Dingdong-Inc/FreshRetailNet-50K", split="train", limit=None
):
    """
    Load a dataset from Hugging Face.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face
        split (str): The dataset split to use (train, test, etc.)
        limit (int, optional): Limit the number of records (for testing)

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame
    """
    print(f"Loading dataset '{dataset_name}' from Hugging Face...")
    try:
        # Check if Hugging Face token is available
        token = os.getenv("HUGGINGFACE_TOKEN")

        # Load dataset with token if available
        if token:
            ds = load_dataset(dataset_name, use_auth_token=token)
        else:
            ds = load_dataset(dataset_name)

        print(f"Dataset loaded successfully. Available splits: {list(ds.keys())}")

        # Convert to pandas DataFrame
        if split in ds:
            df = ds[split].to_pandas()

            # Limit dataset size if specified
            if limit and limit > 0 and limit < len(df):
                df = df.sample(limit, random_state=42)
                print(f"Limited dataset to {limit} records for testing")

            print(f"Dataset shape: {df.shape}")
            return df
        else:
            print(
                f"Split '{split}' not found in the dataset. Available splits: {list(ds.keys())}"
            )
            if len(ds.keys()) > 0:
                default_split = list(ds.keys())[0]
                print(f"Using default split '{default_split}' instead.")
                df = ds[default_split].to_pandas()

                # Limit dataset size if specified
                if limit and limit > 0 and limit < len(df):
                    df = df.sample(limit, random_state=42)
                    print(f"Limited dataset to {limit} records for testing")

                print(f"Dataset shape: {df.shape}")
                return df
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(
            "If this is a private dataset, you may need to log in to Hugging Face using 'huggingface-cli login'"
        )
        print("or set the HUGGINGFACE_TOKEN environment variable.")
        return None


def create_sales_data_csv(df, output_file="sales_data_csv.csv"):
    """
    Create CSV file for sales_data table based on Supabase schema.

    Args:
        df (pd.DataFrame): The Hugging Face dataset
        output_file (str): Output CSV file path
    """
    print(f"Creating sales data CSV file: {output_file}")

    # Create DataFrame with structure matching the Supabase schema
    sales_data = pd.DataFrame()

    # Required columns based on schema
    sales_data["city_id"] = df["city_id"]
    sales_data["store_id"] = df["store_id"]
    sales_data["management_group_id"] = df["management_group_id"]
    sales_data["first_category_id"] = df["first_category_id"]
    sales_data["second_category_id"] = df["second_category_id"]
    sales_data["third_category_id"] = df["third_category_id"]
    sales_data["product_id"] = df["product_id"]
    sales_data["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
    sales_data["sale_amount"] = df["sale_amount"]
    sales_data["stock_hour6_22_cnt"] = df["stock_hour6_22_cnt"]
    sales_data["discount"] = df["discount"]
    sales_data["holiday_flag"] = df["holiday_flag"]
    sales_data["activity_flag"] = (
        df["activity_flag"] if "activity_flag" in df.columns else 0
    )
    sales_data["precpt"] = df["precpt"]
    sales_data["avg_temperature"] = df["avg_temperature"]
    sales_data["avg_humidity"] = df["avg_humidity"]
    sales_data["avg_wind_level"] = df["avg_wind_level"]

    # Save to CSV
    sales_data.to_csv(output_file, index=False)
    print(f"Created sales data CSV with {len(sales_data)} records")
    return sales_data


def create_promotions_csv(df, output_file="promotion_csv.csv", sample_frac=0.05):
    """
    Create CSV file for promotions table based on Supabase schema.

    Args:
        df (pd.DataFrame): The Hugging Face dataset
        output_file (str): Output CSV file path
        sample_frac (float): Fraction of store-product combinations to generate promotions for
    """
    print(f"Creating promotions CSV file: {output_file}")

    # Generate promotion data
    # Filter to only include rows with activity_flag=True if it exists
    if "activity_flag" in df.columns:
        promo_df = df[df["activity_flag"] == 1].copy()
    else:
        # Otherwise, randomly sample from the dataset
        promo_df = df.sample(frac=0.1, random_state=42)

    # Group by store, product to find unique combinations
    promo_df["dt"] = pd.to_datetime(promo_df["dt"])
    promo_df = promo_df.sort_values(["store_id", "product_id", "dt"])

    # Get unique store-product combinations (sample for efficiency)
    store_products = (
        promo_df[["store_id", "product_id"]]
        .drop_duplicates()
        .sample(frac=sample_frac, random_state=42)
    )

    # Get min and max dates
    min_date = promo_df["dt"].min()
    max_date = promo_df["dt"].max()

    # Generate promotion data
    promotion_data = []
    for _, row in tqdm(
        store_products.iterrows(),
        total=len(store_products),
        desc="Generating promotions",
    ):
        # Filter data for this store-product combination
        sp_data = promo_df[
            (promo_df["store_id"] == row["store_id"])
            & (promo_df["product_id"] == row["product_id"])
        ]

        if len(sp_data) > 0:
            # Generate 1-3 promotions per store-product
            num_promos = min(3, max(1, len(sp_data) // 100))
            for _ in range(num_promos):
                # Pick a random start date from this store-product's data
                if len(sp_data) > 0:
                    start_idx = np.random.randint(0, len(sp_data))
                    start_date = sp_data.iloc[start_idx]["dt"]
                else:
                    start_date = min_date + timedelta(
                        days=np.random.randint(0, (max_date - min_date).days)
                    )

                # Promotion length between 1-14 days
                promo_length = np.random.randint(1, 15)
                end_date = start_date + timedelta(days=promo_length)

                promotion_data.append(
                    {
                        "store_id": int(row["store_id"]),
                        "product_id": int(row["product_id"]),
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "promotion_type": np.random.choice(
                            ["Discount", "BOGO", "2-for-1", "Bundle", "Special Display"]
                        ),
                        "discount": float(np.random.uniform(0.05, 0.5)),
                        "location": np.random.choice(
                            ["Entrance", "End Cap", "Checkout", "Aisle", "Special Area"]
                        ),
                        "campaign_id": f"CAMP_{np.random.randint(1000, 10000)}",
                    }
                )

    # Create DataFrame and save to CSV
    promotion_df = pd.DataFrame(promotion_data)
    promotion_df.to_csv(output_file, index=False)
    print(f"Created promotions CSV with {len(promotion_df)} records")
    return promotion_df


def create_holidays_csv(df, output_file="holidays_csv.csv"):
    """
    Create CSV file for holidays table based on Supabase schema.

    Args:
        df (pd.DataFrame): The Hugging Face dataset
        output_file (str): Output CSV file path
    """
    print(f"Creating holidays CSV file: {output_file}")

    # Get min and max dates from the dataset
    min_date = pd.to_datetime(df["dt"]).min()
    max_date = pd.to_datetime(df["dt"]).max()

    # Use US holidays
    us_holidays = holidays.US(years=range(min_date.year, max_date.year + 1))

    # Create holiday DataFrame
    holiday_data = []
    for date, name in us_holidays.items():
        if min_date.date() <= date <= max_date.date():
            holiday_data.append(
                {"date": date.strftime("%Y-%m-%d"), "name": name, "country": "USA"}
            )

    # Create DataFrame and save to CSV
    holiday_df = pd.DataFrame(holiday_data)
    holiday_df.to_csv(output_file, index=False)
    print(f"Created holidays CSV with {len(holiday_df)} records")
    return holiday_df


def main():
    """Main function to load dataset and create CSV files."""
    # Load environment variables
    load_dotenv()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_export"
    )
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load dataset from Hugging Face (use limit=None for full dataset)
        df = load_dataset_from_huggingface(limit=None)
        if df is None:
            print("Dataset loading failed. Exiting.")
            return

        # Create CSV files in the output directory
        sales_output_file = os.path.join(output_dir, "sales_data_csv.csv")
        promotions_output_file = os.path.join(output_dir, "promotion_csv.csv")
        holidays_output_file = os.path.join(output_dir, "holidays_csv.csv")

        # Create all three CSV files
        create_sales_data_csv(df, sales_output_file)
        create_promotions_csv(df, promotions_output_file)
        create_holidays_csv(df, holidays_output_file)

        print(f"CSV files created successfully in directory: {output_dir}")

    except Exception as e:
        print(f"Error during CSV export: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
