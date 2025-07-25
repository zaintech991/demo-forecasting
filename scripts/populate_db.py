
import asyncio
import asyncpg
import pandas as pd
from datetime import datetime, timedelta
from datasets import load_dataset
import os
from dotenv import load_dotenv
import logging
import numpy as np

# Load environment variables for database configuration
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_db_connection_params():
    """Returns database connection parameters from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "your_database"),
        "user": os.getenv("DB_USER", "your_user"),
        "password": os.getenv("DB_PASSWORD", "your_password"),
    }

async def create_connection():
    """Creates and returns a single database connection."""
    params = await get_db_connection_params()
    return await asyncpg.connect(**params)

async def insert_data(conn, table_name, df, conflict_cols=None):
    """
    Inserts DataFrame into a specified table.
    Handles ON CONFLICT DO UPDATE if conflict_cols are provided.
    """
    if df.empty:
        logger.info(f"DataFrame for {table_name} is empty, skipping insertion.")
        return

    columns = df.columns.tolist()
    values_placeholder = ', '.join([f'${i+1}' for i in range(len(columns))])
    
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({values_placeholder})"

    if conflict_cols:
        update_clauses = [f"{col} = EXCLUDED.{col}" for col in columns if col not in conflict_cols]
        if update_clauses:
            query += f" ON CONFLICT ({', '.join(conflict_cols)}) DO UPDATE SET {', '.join(update_clauses)}"
        else:
            query += f" ON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING"
    
    records = df.where(pd.notnull(df), None).values.tolist() # Replace NaN with None for SQL NULL

    logger.info(f"Inserting {len(records)} records into {table_name}...")
    try:
        await conn.executemany(query, records)
        logger.info(f"Successfully inserted {len(records)} records into {table_name}.")
    except Exception as e:
        logger.error(f"Error inserting into {table_name}: {e}")
        logger.error(f"Failed query: {query}")
        # Log first few records that caused the error for debugging
        # if records:
        #     logger.error(f"First few records: {records[:5]}")
        raise # Re-raise to stop if critical insertion fails

async def populate_freshretailnet_data():
    """Main function to load data from Hugging Face and populate Supabase."""
    conn = None
    try:
        conn = await create_connection()
        logger.info("Database connection established.")

        # Load the dataset
        logger.info("Loading FreshRetailNet-50K dataset from Hugging Face...")
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        logger.info("Dataset loaded. Processing splits.")

        all_df = []
        for split_name in dataset.keys():
            df_split = dataset[split_name].to_pandas()
            all_df.append(df_split)
        
        df_raw = pd.concat(all_df, ignore_index=True)
        logger.info(f"Combined dataset has {len(df_raw)} rows.")

        # --- Populate Hierarchy Tables First ---
        # City Hierarchy
        city_hierarchy_df = df_raw[['city_id']].drop_duplicates().copy()
        city_hierarchy_df['city_name'] = 'City ' + city_hierarchy_df['city_id'].astype(str) # Dummy name
        city_hierarchy_df['climate_zone'] = 'Temperate' # Dummy value
        await insert_data(conn, 'city_hierarchy', city_hierarchy_df, conflict_cols=['city_id'])

        # Store Hierarchy (requires city_id)
        store_hierarchy_df = df_raw[['store_id', 'city_id']].drop_duplicates().copy()
        store_hierarchy_df['store_name'] = 'Store ' + store_hierarchy_df['store_id'].astype(str) # Dummy name
        store_hierarchy_df['store_type'] = 'Supermarket' # Dummy value
        store_hierarchy_df['region'] = 'Region ' + store_hierarchy_df['city_id'].astype(str) # Dummy based on city
        # For latitude/longitude, you'd need external data or generate dummy
        store_hierarchy_df['latitude'] = np.random.uniform(25.0, 50.0, len(store_hierarchy_df)) 
        store_hierarchy_df['longitude'] = np.random.uniform(-125.0, -70.0, len(store_hierarchy_df))
        store_hierarchy_df['opening_date'] = datetime(2020, 1, 1).date() # Dummy date
        await insert_data(conn, 'store_hierarchy', store_hierarchy_df, conflict_cols=['store_id'])

        # Product Hierarchy
        product_hierarchy_df = df_raw[['product_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']].drop_duplicates().copy()
        product_hierarchy_df['product_name'] = 'Product ' + product_hierarchy_df['product_id'].astype(str) # Dummy name
        product_hierarchy_df['is_fresh'] = True # Assume all are fresh for this dataset
        product_hierarchy_df['shelf_life_days'] = 7 # Dummy value
        product_hierarchy_df['unit_cost'] = 3.0 # Dummy value
        product_hierarchy_df['unit_price'] = 5.0 # Dummy value
        product_hierarchy_df['promotion_elasticity'] = np.random.uniform(0.1, 0.5, len(product_hierarchy_df)) # Dummy
        await insert_data(conn, 'product_hierarchy', product_hierarchy_df, conflict_cols=['product_id'])

        logger.info("Hierarchy tables populated.")

        # --- Process and Insert Main Data Tables ---
        # Sales Data
        sales_data_df = df_raw.copy()
        sales_data_df['sale_date'] = pd.to_datetime(sales_data_df['dt']).dt.date
        sales_data_df['sale_qty'] = (sales_data_df['sale_amount'] / 5.0).round().astype(int) # Assuming $5 unit price
        sales_data_df['original_price'] = 5.0 # Assuming constant original price for simplicity
        sales_data_df['holiday_flag'] = sales_data_df['holiday_flag'].astype(bool)
        sales_data_df['promo_flag'] = sales_data_df['discount'] < 1.0 # True if there's a discount
        sales_data_df['stock_hour6_22_cnt'] = sales_data_df['stock_hour6_22_cnt'].fillna(0).astype(int)
        sales_data_df['stock_hour6_14_cnt'] = (sales_data_df['stock_hour6_22_cnt'] / 2).astype(int) # Dummy split
        sales_data_df['stock_hour14_22_cnt'] = (sales_data_df['stock_hour6_22_cnt'] / 2).astype(int) # Dummy split

        # Add dummy peak_hour and demand_volatility for sales_data
        sales_data_df['peak_hour'] = np.random.randint(8, 20, len(sales_data_df))
        sales_data_df['demand_volatility'] = np.random.uniform(0.05, 0.3, len(sales_data_df))

        # Filter for sales_data columns from your schema (and add weather related ones if they aren't explicit yet)
        sales_data_cols = [
            'store_id', 'product_id', 'sale_date', 'sale_amount', 'sale_qty', 
            'discount', 'original_price', 'stock_hour6_22_cnt', 'stock_hour6_14_cnt',
            'stock_hour14_22_cnt', 'holiday_flag', 'promo_flag'
        ]
        # Adding weather columns directly to sales_data as per your DatabaseManager.get_sales_data
        sales_data_df['avg_temperature'] = sales_data_df['avg_temperature']
        sales_data_df['avg_humidity'] = sales_data_df['avg_humidity']
        sales_data_df['precpt'] = sales_data_df['precpt']
        sales_data_df['avg_wind_level'] = sales_data_df['avg_wind_level']
        sales_data_cols.extend(['avg_temperature', 'avg_humidity', 'precpt', 'avg_wind_level', 'peak_hour', 'demand_volatility'])


        await insert_data(conn, 'sales_data', sales_data_df[sales_data_cols], conflict_cols=['store_id', 'product_id', 'sale_date'])
        
        # Hourly Sales Data (if you intend to use it, currently not explicitly used in main backend logic heavily)
        # This table is huge if you process all hourly data, consider sampling or aggregating
        # For now, I'll skip detailed hourly_sales_data insertion to reduce initial load.
        # If needed, we can add this later by parsing 'hours_sale' and 'hours_stock_status' sequences.

        # Weather Data
        weather_data_df = df_raw[['city_id', 'dt', 'avg_temperature', 'precpt', 'avg_humidity', 'avg_wind_level']].drop_duplicates().copy()
        weather_data_df['date'] = pd.to_datetime(weather_data_df['dt']).dt.date
        weather_data_df['temp_min'] = weather_data_df['avg_temperature'] - np.random.uniform(2, 5, len(weather_data_df))
        weather_data_df['temp_max'] = weather_data_df['avg_temperature'] + np.random.uniform(2, 5, len(weather_data_df))
        weather_data_df['temp_avg'] = weather_data_df['avg_temperature']
        weather_data_df['precipitation'] = weather_data_df['precpt']
        weather_data_df['humidity'] = weather_data_df['avg_humidity']
        weather_data_df['wind_speed'] = weather_data_df['avg_wind_level']
        weather_data_df['weather_condition'] = 'Clear' # Dummy, can be derived from precpt/temp
        
        weather_data_cols = [
            'city_id', 'date', 'temp_min', 'temp_max', 'temp_avg', 
            'precipitation', 'humidity', 'wind_speed', 'weather_condition'
        ]
        await insert_data(conn, 'weather_data', weather_data_df[weather_data_cols], conflict_cols=['city_id', 'date'])

        # Holiday Calendar
        holiday_calendar_df = df_raw[df_raw['holiday_flag'] == 1][['dt', 'holiday_flag']].drop_duplicates().copy()
        if not holiday_calendar_df.empty:
            holiday_calendar_df['date'] = pd.to_datetime(holiday_calendar_df['dt']).dt.date
            holiday_calendar_df['holiday_name'] = 'Holiday ' + holiday_calendar_df['date'].astype(str) # Dummy name
            holiday_calendar_df['holiday_type'] = 'National'
            holiday_calendar_df['country'] = 'US'
            holiday_calendar_df['region'] = 'All'
            holiday_calendar_df['significance'] = 3 # Medium significance
            
            holiday_calendar_cols = [
                'date', 'holiday_name', 'holiday_type', 'country', 'region', 'significance'
            ]
            await insert_data(conn, 'holiday_calendar', holiday_calendar_df[holiday_calendar_cols], conflict_cols=['date'])

        # Promotion Events (from discount column)
        promotion_events_df = df_raw[df_raw['discount'] < 1.0].copy()
        if not promotion_events_df.empty:
            promotion_events_df['start_date'] = pd.to_datetime(promotion_events_df['dt']).dt.date
            promotion_events_df['end_date'] = promotion_events_df['start_date'] + pd.to_timedelta(7, unit='D') # Assume 7-day promo
            promotion_events_df['promotion_type'] = 'Discount'
            promotion_events_df['discount_percentage'] = promotion_events_df['discount']
            promotion_events_df['display_location'] = 'Online/In-store'
            promotion_events_df['campaign_id'] = 'CAMPAIGN_' + (promotion_events_df.index).astype(str)
            
            promo_events_cols = [
                'store_id', 'product_id', 'start_date', 'end_date', 'promotion_type', 
                'discount_percentage', 'display_location', 'campaign_id'
            ]
            # Ensure no duplicates on (store, product, start_date) for ON CONFLICT
            promotion_events_df = promotion_events_df.drop_duplicates(subset=['store_id', 'product_id', 'start_date'])
            await insert_data(conn, 'promotion_events', promotion_events_df[promo_events_cols], conflict_cols=['store_id', 'product_id', 'start_date'])

        logger.info("Raw data tables populated successfully.")

    except Exception as e:
        logger.error(f"An error occurred during data population: {e}", exc_info=True)
    finally:
        if conn:
            await conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    asyncio.run(populate_freshretailnet_data()) 