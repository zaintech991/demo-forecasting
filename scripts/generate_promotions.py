"""
Script to generate sample promotion data for the Supabase database.
"""
import pandas as pd
import numpy as np
import datetime
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import mapping data if available
try:
    from utils.mapping_data import STORE_MAPPING
    HAS_MAPPINGS = True
except ImportError:
    HAS_MAPPINGS = False
    print("Warning: Store mapping not found. Using generic store IDs.")

def generate_promotion_data(num_promotions=100, start_date='2022-01-01', end_date='2022-12-31'):
    """
    Generate sample promotion data.
    
    Args:
        num_promotions: Number of promotions to generate
        start_date: Earliest possible promotion start date
        end_date: Latest possible promotion end date
        
    Returns:
        DataFrame with promotion data
    """
    print(f"Generating {num_promotions} sample promotions...")
    
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Available store IDs
    if HAS_MAPPINGS:
        store_ids = list(STORE_MAPPING.keys())
    else:
        store_ids = list(range(20))  # Assuming 20 stores
    
    # Generate random product IDs
    product_ids = np.random.randint(1, 100, size=50)  # 50 unique products
    
    # Promotion types
    promo_types = [
        "Discount", "BOGO", "Bundle Deal", "Loyalty Reward", 
        "Flash Sale", "Holiday Special", "Clearance", "New Product"
    ]
    
    # Locations
    locations = ["End Cap", "Front of Store", "Main Aisle", "Back of Store", "Register", "Window Display"]
    
    # Generate data
    data = []
    for i in range(num_promotions):
        # Random store and product
        store_id = np.random.choice(store_ids)
        product_id = np.random.choice(product_ids)
        
        # Random dates
        duration_days = np.random.randint(3, 30)  # Promotions last 3-30 days
        days_from_start = np.random.randint(0, (end_dt - start_dt).days - duration_days)
        promo_start = start_dt + datetime.timedelta(days=days_from_start)
        promo_end = promo_start + datetime.timedelta(days=duration_days)
        
        # Random discount (between 5% and 50% off)
        discount = round(1.0 - (np.random.randint(5, 51) / 100), 2)
        
        # Other attributes
        promo_type = np.random.choice(promo_types)
        location = np.random.choice(locations)
        campaign_id = f"CAMP{np.random.randint(1000, 10000)}"
        
        data.append({
            'store_id': store_id,
            'product_id': product_id,
            'start_date': promo_start.strftime('%Y-%m-%d'),
            'end_date': promo_end.strftime('%Y-%m-%d'),
            'promotion_type': promo_type,
            'discount': discount,
            'location': location,
            'campaign_id': campaign_id
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def export_csv(df, output_file):
    """Export DataFrame to CSV"""
    df.to_csv(output_file, index=False)
    print(f"Data exported to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate sample promotion data')
    parser.add_argument('--count', type=int, default=100, help='Number of promotions to generate')
    parser.add_argument('--output', type=str, default='promotions.csv', help='Output CSV file')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='Promotion start date range (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2022-12-31', help='Promotion end date range (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Generate data
    promotions_df = generate_promotion_data(
        num_promotions=args.count, 
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Export to CSV
    export_csv(promotions_df, args.output)

if __name__ == "__main__":
    main() 