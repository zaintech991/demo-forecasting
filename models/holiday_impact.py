"""
Holiday impact analysis module for evaluating sales lift during holidays.
"""
import pandas as pd
import numpy as np

def analyze_holiday_impact(df):
    """
    Analyze sales lift during holidays
    
    Args:
        df: DataFrame with sales data including holiday_flag
        
    Returns:
        DataFrame with holiday impact metrics
    """
    # Make a copy to avoid modifying the original
    analysis_df = df.copy()
    
    # Check if required columns exist
    if 'holiday_flag' not in analysis_df.columns:
        # If holiday flag missing, return empty DataFrame
        return pd.DataFrame(columns=[
            'product_id', 'avg_holiday_sales', 'avg_non_holiday_sales',
            'absolute_lift', 'percentage_lift', 'holiday_count'
        ])
    
    # Make sure date is a datetime
    if 'dt' in analysis_df.columns and analysis_df['dt'].dtype == 'object':
        analysis_df['dt'] = pd.to_datetime(analysis_df['dt'])
    
    # Separate holiday and non-holiday sales
    holiday_sales = analysis_df[analysis_df['holiday_flag'] == 1]
    non_holiday_sales = analysis_df[analysis_df['holiday_flag'] == 0]
    
    if len(holiday_sales) == 0 or len(non_holiday_sales) == 0:
        # If no holiday data, return empty DataFrame
        return pd.DataFrame(columns=[
            'product_id', 'avg_holiday_sales', 'avg_non_holiday_sales',
            'absolute_lift', 'percentage_lift', 'holiday_count'
        ])
    
    # Calculate metrics per product
    product_holiday_impact = []
    
    for product in analysis_df['product_id'].unique():
        product_holiday = holiday_sales[holiday_sales['product_id'] == product]
        product_non_holiday = non_holiday_sales[non_holiday_sales['product_id'] == product]
        
        if len(product_holiday) == 0 or len(product_non_holiday) == 0:
            continue
        
        # Calculate metrics
        avg_holiday_sales = product_holiday['sale_amount'].mean()
        avg_non_holiday_sales = product_non_holiday['sale_amount'].mean()
        
        # Calculate lift
        absolute_lift = avg_holiday_sales - avg_non_holiday_sales
        percentage_lift = (absolute_lift / avg_non_holiday_sales) * 100 if avg_non_holiday_sales > 0 else 0
        
        # Store results
        product_holiday_impact.append({
            'product_id': product,
            'avg_holiday_sales': avg_holiday_sales,
            'avg_non_holiday_sales': avg_non_holiday_sales,
            'absolute_lift': absolute_lift,
            'percentage_lift': percentage_lift,
            'holiday_count': len(product_holiday)
        })
    
    # Convert to DataFrame
    impact_df = pd.DataFrame(product_holiday_impact)
    
    # Sort by percentage lift
    if not impact_df.empty:
        impact_df = impact_df.sort_values('percentage_lift', ascending=False)
    
    return impact_df

def get_holiday_dates_by_name(df, holiday_name):
    """
    Get dates for a specific holiday from sales data
    
    Args:
        df: DataFrame with sales data
        holiday_name: Name of the holiday to extract
        
    Returns:
        List of holiday dates
    """
    # Check if required columns exist
    if 'holiday_flag' not in df.columns or 'holiday_name' not in df.columns or 'dt' not in df.columns:
        return []
    
    # Filter by holiday name
    holiday_data = df[(df['holiday_flag'] == 1) & (df['holiday_name'] == holiday_name)]
    
    # Extract unique dates
    holiday_dates = pd.to_datetime(holiday_data['dt'].unique())
    
    return holiday_dates.tolist()

def get_holiday_names(df):
    """
    Get list of all holiday names in the dataset
    
    Args:
        df: DataFrame with sales data
        
    Returns:
        List of unique holiday names
    """
    # Check if required columns exist
    if 'holiday_flag' not in df.columns or 'holiday_name' not in df.columns:
        return []
    
    # Filter by holiday flag
    holiday_data = df[df['holiday_flag'] == 1]
    
    # Extract unique holiday names
    holiday_names = holiday_data['holiday_name'].dropna().unique().tolist()
    
    return holiday_names 