"""
Script to load the FreshRetailNet-50K dataset from Hugging Face to any PostgreSQL database.
This is a more flexible version that can work with any PostgreSQL connection.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.load_dataset import preprocess_data, upload_to_supabase


def load_dataset_from_huggingface(
    dataset_name="Dingdong-Inc/FreshRetailNet-50K", split="train"
):
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
        token = os.getenv("HUGGINGFACE_TOKEN")

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
            print(
                f"Split '{split}' not found in the dataset. Available splits: {ds.keys()}"
            )
            if len(ds.keys()) > 0:
                default_split = list(ds.keys())[0]
                print(f"Using default split '{default_split}' instead.")
                df = ds[default_split].to_pandas()
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


def create_db_engine(connection_string=None):
    """
    Create a SQLAlchemy engine for the database connection.

    Args:
        connection_string (str, optional): Database connection string.
                                          If None, will try to build from environment variables.

    Returns:
        Engine: SQLAlchemy engine
    """
    if connection_string is None:
        # Build connection string from environment variables
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD")

        if not all([db_host, db_password]):
            print("Error: DB_HOST and DB_PASSWORD environment variables must be set.")
            print("Please set these in your .env file or environment.")
            return None

        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

    try:
        # Create engine with appropriate timeout
        engine = create_engine(
            connection_string,
            connect_args={"connect_timeout": 15},
            pool_pre_ping=True,
            pool_recycle=300,
        )

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        print(
            f"Successfully connected to database at {connection_string.split('@')[1].split('/')[0]}"
        )
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def create_tables(engine, schema_file="schema.sql"):
    """
    Create database tables using schema.sql.

    Args:
        engine: SQLAlchemy engine
        schema_file (str): Path to schema file, relative to the database directory
    """
    if engine is None:
        print("Cannot create tables: Database engine is None")
        return False

    print("Creating database tables...")
    schema_path = Path(__file__).parent / schema_file

    if not schema_path.exists():
        print(f"Schema file not found at {schema_path}")
        return False

    try:
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        with engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()

        print("Database tables created successfully.")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False


def refresh_materialized_views(engine):
    """
    Refresh all materialized views.

    Args:
        engine: SQLAlchemy engine
    """
    if engine is None:
        print("Cannot refresh views: Database engine is None")
        return False

    print("Refreshing materialized views...")

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT refresh_all_mv()"))
            conn.commit()

        print("Materialized views refreshed successfully.")
        return True
    except Exception as e:
        print(f"Error refreshing materialized views: {e}")
        return False


def main():
    """Main function to load and process the dataset."""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Load FreshRetailNet-50K dataset to PostgreSQL database"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dingdong-Inc/FreshRetailNet-50K",
        help="HuggingFace dataset name (default: Dingdong-Inc/FreshRetailNet-50K)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--connection",
        type=str,
        help="Database connection string (if not provided, will use environment variables)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for database uploads (default: 5000)",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip creating database schema (use if tables already exist)",
    )

    args = parser.parse_args()

    try:
        # Load dataset from Hugging Face
        df = load_dataset_from_huggingface(args.dataset, args.split)
        if df is None:
            print("Dataset loading failed. Exiting.")
            return

        # Get database engine
        engine = create_db_engine(args.connection)
        if engine is None:
            print("Database connection failed. Exiting.")
            return

        # Create tables if needed
        if not args.skip_schema:
            if not create_tables(engine):
                print("Table creation failed. Exiting.")
                return

        # Preprocess data for all tables
        print("Preprocessing dataset for database ingestion...")
        processed_data = preprocess_data(df)

        # Show statistics for each table
        for table_name, data in processed_data.items():
            if data is not None and not data.empty:
                print(f"Prepared {table_name} with {len(data)} records")

        # Upload all data to database
        print(
            f"Uploading complete dataset to database in batches of {args.batch_size}..."
        )
        upload_to_supabase(processed_data, batch_size=args.batch_size)

        # Refresh materialized views to update aggregated data
        refresh_materialized_views(engine)

        print("Dataset loading completed successfully!")

    except Exception as e:
        print(f"Error during dataset loading: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
