# Retail Sales Forecasting Project

This project uses machine learning and time series forecasting techniques to predict retail sales, analyze promotional impact, and identify stockout patterns using the FreshRetailNet-50K dataset from Hugging Face.

## Setup Instructions

### Prerequisites

- Python 3.9+
- PostgreSQL or Supabase account
- Hugging Face account (for dataset access)

### Environment Setup

1. Clone the repository
```bash
git clone <repository-url>
cd Forecasting
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Copy the environment template and configure
```bash
cp env.template .env
```

5. Edit the `.env` file with your Supabase credentials:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
```

### Supabase Setup

1. Create a new Supabase project
2. Obtain your API URL and keys from the project settings
3. Enable the SQL Editor on your project

## Data Loading Options

This project provides multiple ways to load the FreshRetailNet-50K dataset into your Supabase database:

### Option 1: Direct API Loading (Recommended for Small Datasets)

Test with a small dataset first:
```bash
python scripts/load_data.py --limit 10
```

If successful, gradually increase the limit:
```bash
python scripts/load_data.py --limit 100
python scripts/load_data.py --limit 1000
# etc.
```

### Option 2: SQL Generation and Manual Execution

Generate SQL to execute in the Supabase SQL Editor:
```bash
python scripts/load_data.py --generate-sql-only --limit 50 > schema_with_sample_data.sql
```

Then follow the instructions in `SUPABASE_INSTRUCTIONS.md` to execute the SQL using the Supabase SQL Editor.

### Option 3: CSV Export and Import

Export data to CSV for manual import via Supabase UI:
```bash
python scripts/export_dataset.py --limit 1000 --output sales_data.csv
```

Additional export options:
```bash
# Export holidays to a separate file
python scripts/export_dataset.py --limit 1000 --output sales_data.csv --holidays

# Export in different formats
python scripts/export_dataset.py --limit 1000 --format json --output sales_data.json
python scripts/export_dataset.py --limit 1000 --format parquet --output sales_data.parquet

# Skip the first 1000 records
python scripts/export_dataset.py --limit 1000 --offset 1000 --output sales_data_batch2.csv
```

## Running the API

Start the API server:
```bash
python main.py
```

The API will be available at `http://127.0.0.1:8000` (or the host/port configured in your .env file).

## Available Endpoints

- `/api/forecast/{city_id}/{store_id}/{product_id}` - Get sales forecast
- `/api/promotions/impact/{store_id}/{product_id}` - Analyze promotional impact
- `/api/stockout/risk/{store_id}/{product_id}` - Get stockout risk analysis

## Project Structure

- `api/` - API endpoints
- `data/` - Sample data files
- `database/` - Database utilities
- `models/` - ML models and prediction algorithms
- `notebooks/` - Jupyter notebooks for analysis
- `scripts/` - Data loading and utility scripts
- `services/` - Business logic services
- `static/` - Frontend UI files
- `utils/` - Utility functions

## Troubleshooting

If you encounter issues with the Supabase API:

1. Check that your environment variables are set correctly
2. Try using the SQL generation approach (Option 2 above)
3. If loading fails due to rate limits, reduce batch sizes or add delays
4. For database schema errors, check `SUPABASE_INSTRUCTIONS.md` for alternative setup

## Development

- Explore data in the notebooks directory
- Check the API implementation in api/ directory
- Review the model implementations in models/ directory
