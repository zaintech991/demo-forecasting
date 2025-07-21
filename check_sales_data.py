import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path to import DatabaseManager
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.connection import DatabaseManager


async def check_sales_data():
    db_manager = DatabaseManager()
    await db_manager.initialize()

    try:
        print("Checking sales_data table for data distribution...")

        # 1. Get overall date range and total distinct stores
        overall_query = """
        SELECT 
            COUNT(DISTINCT store_id) as total_distinct_stores,
            MIN(dt) as min_sale_date,
            MAX(dt) as max_sale_date
        FROM sales_data;
        """
        overall_result = await db_manager.execute_dataframe_query(overall_query)
        if not overall_result.empty:
            print("\nOverall Sales Data Summary:")
            print(overall_result.to_string(index=False))
        else:
            print("\nNo data found in sales_data table.")
            await db_manager.close()
            return

        # 2. Get distinct active days per store
        store_activity_query = """
        SELECT 
            store_id,
            COUNT(DISTINCT dt) as distinct_active_days,
            MIN(dt) as first_sale_date,
            MAX(dt) as last_sale_date
        FROM sales_data
        GROUP BY store_id
        ORDER BY distinct_active_days DESC;
        """
        store_activity_df = await db_manager.execute_dataframe_query(
            store_activity_query
        )

        if not store_activity_df.empty:
            print("\nDistinct Active Days per Store (Top 10):")
            print(store_activity_df.head(10).to_string(index=False))
            print(f"\nTotal stores with sales data: {len(store_activity_df)}")
            print(
                f"Stores with >= 7 active days: {len(store_activity_df[store_activity_df['distinct_active_days'] >= 7])}"
            )
            print(
                f"Stores with >= 30 active days: {len(store_activity_df[store_activity_df['distinct_active_days'] >= 30])}"
            )
        else:
            print("\nNo store activity data found.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(check_sales_data())
