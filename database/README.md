# Database Loading Scripts

This directory contains scripts for loading the FreshRetailNet-50K dataset from Hugging Face into a PostgreSQL database (such as Supabase).

## Setup

1. Make sure you have the required Python packages installed:
   ```bash
   pip install datasets pandas sqlalchemy psycopg2-binary python-dotenv tqdm holidays
   ```

2. Create a `.env` file in the project root with your database credentials (see `env_example.txt` for a template):
   ```
   # Copy the contents from env_example.txt and replace with your actual credentials
   ```

## Loading the Dataset

### Option 1: Using the general-purpose loader script

The `load_to_database.py` script provides a flexible way to load the dataset into any PostgreSQL database:

```bash
python database/load_to_database.py --dataset "Dingdong-Inc/FreshRetailNet-50K" --batch-size 5000
```

Optional arguments:
- `--dataset`: HuggingFace dataset name (default: "Dingdong-Inc/FreshRetailNet-50K")
- `--split`: Dataset split to use (default: "train")
- `--connection`: Database connection string (if not provided, will use environment variables)
- `--batch-size`: Batch size for database uploads (default: 5000)
- `--skip-schema`: Skip creating database schema (use if tables already exist)

### Option 2: Using the Supabase-specific loader

If you're specifically using Supabase, you can use the `load_full_dataset.py` script:

```bash
python database/load_full_dataset.py
```

## Troubleshooting

- **Connection Issues**: Make sure your database credentials are correct and that your database is accessible from your current network.
- **Dataset Access**: If the dataset is private, you may need to log in to Hugging Face using `huggingface-cli login` or set the `HUGGINGFACE_TOKEN` environment variable.
- **Memory Issues**: If you encounter memory problems when processing large datasets, try reducing the batch size using the `--batch-size` parameter.

## Database Structure

The dataset will be loaded into the following tables:

1. `store_hierarchy`: Store information and geographic data
2. `product_hierarchy`: Product categorization and attributes
3. `sales_data`: Daily sales records with quantities and amounts
4. `weather_data`: Weather conditions by city and date
5. `holiday_calendar`: Holiday information
6. `promotion_events`: Promotional activities

The schema also includes materialized views for efficient querying of aggregated data. 