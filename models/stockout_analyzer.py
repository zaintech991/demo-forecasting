"""
Stockout analysis module for estimating demand during stockout periods.
"""
import pandas as pd
import numpy as np

def estimate_demand_during_stockout(df, product_id=None, store_id=None):
    """
    Estimate demand during stockout periods using historical patterns
    
    Args:
        df: DataFrame with sales data
        product_id: Optional filter by product
        store_id: Optional filter by store
        
    Returns:
        DataFrame with actual sales and estimated demand
    """
    # Filter data if needed
    filtered_df = df.copy()
    if product_id is not None:
        filtered_df = filtered_df[filtered_df['product_id'] == product_id]
    if store_id is not None:
        filtered_df = filtered_df[filtered_df['store_id'] == store_id]
    
    # Make a copy to avoid modifying the original
    stock_df = filtered_df.copy()
    
    # Convert date column to datetime if needed
    if 'dt' in stock_df.columns and stock_df['dt'].dtype == 'object':
        stock_df['dt'] = pd.to_datetime(stock_df['dt'])
    
    # Create stockout flag for easier filtering
    if 'stock_hour6_22_cnt' in stock_df.columns:
        stock_df['is_stockout'] = stock_df['stock_hour6_22_cnt'] > 0
    else:
        # If no stockout data available, assume no stockouts
        stock_df['is_stockout'] = False
        stock_df['stock_hour6_22_cnt'] = 0
    
    # Calculate daily patterns for the product/store
    if 'dt' in stock_df.columns:
        daily_patterns = stock_df[~stock_df['is_stockout']].groupby(
            stock_df['dt'].dt.dayofweek
        )['sale_amount'].median()
    else:
        # If no date column, use overall median
        daily_patterns = pd.Series([stock_df[~stock_df['is_stockout']]['sale_amount'].median()])
    
    # Create a new column for estimated demand
    stock_df['estimated_demand'] = stock_df['sale_amount']
    
    # For stockout days, estimate demand based on day of week pattern
    for idx, row in stock_df[stock_df['is_stockout']].iterrows():
        if 'dt' in row:
            day_of_week = row['dt'].dayofweek
        else:
            day_of_week = 0  # Default to Monday if no date
            
        pattern_value = None
        if day_of_week in daily_patterns:
            pattern_value = daily_patterns[day_of_week]
        elif len(daily_patterns) > 0:
            # Use mean if specific day not available
            pattern_value = daily_patterns.mean()
            
        if pattern_value is not None:
            # Calculate adjustment factor based on hours out of stock
            hours_out = row['stock_hour6_22_cnt']
            total_hours = 16  # 6:00-22:00 = 16 hours
            adjustment = total_hours / (total_hours - hours_out) if hours_out < total_hours else 2
            
            # Update estimated demand
            stock_df.at[idx, 'estimated_demand'] = row['sale_amount'] * adjustment
            
    # Calculate lost sales
    stock_df['lost_sales'] = stock_df['estimated_demand'] - stock_df['sale_amount']
    stock_df['lost_sales'] = stock_df['lost_sales'].clip(lower=0)  # No negative lost sales
    
    return stock_df

def analyze_stockouts(df):
    """
    Summarize stockout impact
    
    Args:
        df: DataFrame with stockout analysis results
        
    Returns:
        Dictionary with stockout metrics
    """
    # Check if required columns exist
    if 'is_stockout' not in df.columns or 'lost_sales' not in df.columns:
        return {
            'total_days': len(df),
            'stockout_days': 0,
            'stockout_rate': 0,
            'total_lost_sales': 0,
            'avg_lost_sales': 0
        }
    
    # Calculate key metrics
    total_days = len(df)
    stockout_days = df['is_stockout'].sum()
    stockout_rate = stockout_days / total_days if total_days > 0 else 0
    
    total_lost_sales = df['lost_sales'].sum()
    avg_lost_sales = df[df['is_stockout']]['lost_sales'].mean() if stockout_days > 0 else 0
    
    # Create summary
    summary = {
        'total_days': int(total_days),
        'stockout_days': int(stockout_days),
        'stockout_rate': float(stockout_rate),
        'total_lost_sales': float(total_lost_sales),
        'avg_lost_sales': float(avg_lost_sales)
    }
    
    return summary 