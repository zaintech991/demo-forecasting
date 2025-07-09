"""
FastAPI endpoints for sales forecasting and demand prediction.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException, Depends
from pydantic import BaseModel, Field
import logging
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.prophet_forecaster import ProphetForecaster
from models.promo_uplift_model import PromoUpliftModel
from database.connection import get_db_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FreshRetail Forecasting API",
    description="API for sales forecasting, promotion impact, and inventory optimization",
    version="1.0.0"
)

# Data models for API requests and responses
class ForecastRequest(BaseModel):
    """Request model for forecasting."""
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: str
    periods: int = 30
    freq: str = "D"
    include_weather: bool = False
    include_holidays: bool = True
    include_promotions: bool = False
    return_components: bool = False

class ForecastResponse(BaseModel):
    """Response model for forecasting."""
    forecast: List[Dict[str, Any]]
    metrics: Optional[Dict[str, float]] = None
    components: Optional[Dict[str, List[Dict[str, Any]]]] = None

class PromotionAnalysisRequest(BaseModel):
    """Request model for promotion analysis."""
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: str
    end_date: str
    promotion_type: Optional[str] = None
    discount_min: Optional[float] = None
    discount_max: Optional[float] = None

class StockoutAnalysisRequest(BaseModel):
    """Request model for stockout impact analysis."""
    store_id: int
    product_id: int
    start_date: str
    end_date: str
    
class HolidayImpactRequest(BaseModel):
    """Request model for holiday impact analysis."""
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    holiday_name: Optional[str] = None
    start_date: str
    end_date: str

# Global model instances (lazy-loaded)
_forecast_model = None
_promo_model = None

def get_forecast_model():
    """Get or initialize the forecast model."""
    global _forecast_model
    if _forecast_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'models/saved/forecast_model.json')
        _forecast_model = ProphetForecaster(
            model_name="api_forecast_model",
            include_holiday=True,
            include_weather=True,
            include_promotions=True
        )
        # Check if a saved model exists
        if os.path.exists(model_path):
            logger.info(f"Loading forecast model from {model_path}")
            _forecast_model.load_model(model_path)
    return _forecast_model

def get_promo_model():
    """Get or initialize the promotion uplift model."""
    global _promo_model
    if _promo_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'models/saved/promo_uplift_model.joblib')
        _promo_model = PromoUpliftModel(model_type="gradient_boost")
        # Check if a saved model exists
        if os.path.exists(model_path):
            logger.info(f"Loading promo uplift model from {model_path}")
            _promo_model.load_model(model_path)
    return _promo_model

def fetch_historical_data(store_id=None, product_id=None, category_id=None, city_id=None, 
                         start_date=None, end_date=None):
    """
    Fetch historical sales data from the database.
    
    Returns:
        pd.DataFrame: Historical sales data
    """
    engine = get_db_engine()
    
    query = """
    SELECT 
        sd.sale_date,
        sd.store_id,
        sd.product_id,
        ph.first_category_id,
        sh.city_id,
        sd.sale_amount,
        sd.sale_qty,
        sd.discount,
        sd.original_price,
        sd.stock_hour6_22_cnt,
        sd.holiday_flag,
        sd.promo_flag
    FROM sales_data sd
    JOIN store_hierarchy sh ON sd.store_id = sh.store_id
    JOIN product_hierarchy ph ON sd.product_id = ph.product_id
    WHERE 1=1
    """
    
    params = {}
    
    if store_id is not None:
        query += " AND sd.store_id = :store_id"
        params['store_id'] = store_id
        
    if product_id is not None:
        query += " AND sd.product_id = :product_id"
        params['product_id'] = product_id
        
    if category_id is not None:
        query += " AND ph.first_category_id = :category_id"
        params['category_id'] = category_id
        
    if city_id is not None:
        query += " AND sh.city_id = :city_id"
        params['city_id'] = city_id
        
    if start_date is not None:
        query += " AND sd.sale_date >= :start_date"
        params['start_date'] = start_date
        
    if end_date is not None:
        query += " AND sd.sale_date <= :end_date"
        params['end_date'] = end_date
    
    query += " ORDER BY sd.sale_date"
    
    try:
        df = pd.read_sql(query, engine, params=params)
        logger.info(f"Fetched {len(df)} records from database")
        return df
    except Exception as e:
        logger.error(f"Database query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def fetch_weather_data(city_id=None, start_date=None, end_date=None, future_periods=0):
    """
    Fetch weather data from the database, including future forecasts if available.
    
    Args:
        city_id: City ID to filter by
        start_date: Start date
        end_date: End date
        future_periods: Number of future periods to include
        
    Returns:
        pd.DataFrame: Weather data
    """
    engine = get_db_engine()
    
    # First try to get historical weather data
    query = """
    SELECT 
        city_id,
        date,
        temp_avg,
        humidity,
        precipitation,
        wind_speed,
        weather_condition
    FROM weather_data
    WHERE 1=1
    """
    
    params = {}
    
    if city_id is not None:
        query += " AND city_id = :city_id"
        params['city_id'] = city_id
        
    if start_date is not None:
        query += " AND date >= :start_date"
        params['start_date'] = start_date
        
    if end_date is not None:
        query += " AND date <= :end_date"
        params['end_date'] = end_date
    
    query += " ORDER BY date"
    
    try:
        df = pd.read_sql(query, engine, params=params)
        
        # If future periods are requested, generate synthetic weather data
        if future_periods > 0 and len(df) > 0:
            last_date = pd.to_datetime(df['date'].max())
            
            # Create future dates
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_periods)
            
            # Sample with replacement from historical data for future weather
            future_weather = []
            for date in future_dates:
                # Sample from the same day of the week historically
                same_dow = df[pd.to_datetime(df['date']).dt.dayofweek == date.dayofweek]
                if len(same_dow) > 0:
                    sample = same_dow.sample(1).iloc[0].to_dict()
                    sample['date'] = date
                    future_weather.append(sample)
                else:
                    # If no matching day of week, just sample randomly
                    sample = df.sample(1).iloc[0].to_dict()
                    sample['date'] = date
                    future_weather.append(sample)
            
            # Combine with historical data
            if future_weather:
                future_df = pd.DataFrame(future_weather)
                df = pd.concat([df, future_df], ignore_index=True)
        
        logger.info(f"Fetched {len(df)} weather records")
        return df
    except Exception as e:
        logger.error(f"Weather data query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Weather data error: {str(e)}")

def fetch_promotion_data(store_id=None, product_id=None, category_id=None, 
                       start_date=None, end_date=None, future_periods=0):
    """
    Fetch promotion data from the database, including future planned promotions.
    
    Returns:
        pd.DataFrame: Promotion data
    """
    engine = get_db_engine()
    
    query = """
    SELECT 
        pe.store_id,
        pe.product_id,
        ph.first_category_id,
        pe.start_date,
        pe.end_date,
        pe.promotion_type,
        pe.discount_percentage
    FROM promotion_events pe
    JOIN product_hierarchy ph ON pe.product_id = ph.product_id
    WHERE 1=1
    """
    
    params = {}
    
    if store_id is not None:
        query += " AND pe.store_id = :store_id"
        params['store_id'] = store_id
        
    if product_id is not None:
        query += " AND pe.product_id = :product_id"
        params['product_id'] = product_id
        
    if category_id is not None:
        query += " AND ph.first_category_id = :category_id"
        params['category_id'] = category_id
    
    # Handle date range for promotions
    if start_date is not None and end_date is not None:
        query += """
        AND (
            (pe.start_date BETWEEN :start_date AND :end_date) OR
            (pe.end_date BETWEEN :start_date AND :end_date) OR
            (pe.start_date <= :start_date AND pe.end_date >= :end_date)
        )
        """
        params['start_date'] = start_date
        params['end_date'] = end_date
    elif start_date is not None:
        query += " AND pe.end_date >= :start_date"
        params['start_date'] = start_date
    elif end_date is not None:
        query += " AND pe.start_date <= :end_date"
        params['end_date'] = end_date
    
    try:
        df = pd.read_sql(query, engine, params=params)
        logger.info(f"Fetched {len(df)} promotion records")
        return df
    except Exception as e:
        logger.error(f"Promotion data query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Promotion data error: {str(e)}")

def fetch_holiday_data(start_date=None, end_date=None, future_periods=0):
    """
    Fetch holiday data from the database.
    
    Returns:
        pd.DataFrame: Holiday data
    """
    engine = get_db_engine()
    
    query = """
    SELECT 
        date,
        holiday_name,
        holiday_type,
        significance
    FROM holiday_calendar
    WHERE 1=1
    """
    
    params = {}
    
    if start_date is not None:
        query += " AND date >= :start_date"
        params['start_date'] = start_date
        
    if end_date is not None:
        query += " AND date <= :end_date"
        params['end_date'] = end_date
    
    query += " ORDER BY date"
    
    try:
        df = pd.read_sql(query, engine, params=params)
        logger.info(f"Fetched {len(df)} holiday records")
        return df
    except Exception as e:
        logger.error(f"Holiday data query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Holiday data error: {str(e)}")

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the FreshRetail Forecasting API"}

@app.post("/forecast/", response_model=ForecastResponse)
def create_forecast(request: ForecastRequest, model: ProphetForecaster = Depends(get_forecast_model)):
    """
    Generate sales forecast for specified parameters.
    
    Parameters:
    - store_id: Optional store ID to filter by
    - product_id: Optional product ID to filter by
    - category_id: Optional category ID to filter by
    - city_id: Optional city ID to filter by
    - start_date: Start date for forecast
    - periods: Number of periods to forecast (default: 30)
    - freq: Frequency of forecast (default: "D" for daily)
    - include_weather: Whether to include weather in the forecast
    - include_holidays: Whether to include holidays in the forecast
    - include_promotions: Whether to include promotions in the forecast
    - return_components: Whether to return forecast components
    
    Returns:
    - Forecast data
    - Metrics (if available)
    - Components (if requested)
    """
    try:
        # Parse start date
        start_date = pd.to_datetime(request.start_date)
        
        # Calculate end date based on periods and frequency
        if request.freq == "D":
            end_date = start_date + timedelta(days=request.periods)
        elif request.freq == "W":
            end_date = start_date + timedelta(weeks=request.periods)
        elif request.freq == "M":
            end_date = start_date + pd.DateOffset(months=request.periods)
        else:
            end_date = start_date + timedelta(days=request.periods)
        
        # Get historical data for training
        hist_start_date = start_date - timedelta(days=365)  # Use 1 year of history for training
        df = fetch_historical_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=request.city_id,
            start_date=hist_start_date,
            end_date=start_date - timedelta(days=1)
        )
        
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Not enough historical data for forecasting")
        
        # Configure the model
        model.include_holiday = request.include_holidays
        model.include_weather = request.include_weather
        model.include_promotions = request.include_promotions
        model.forecast_periods = request.periods
        model.forecast_freq = request.freq
        
        # Train the model if needed
        if not model.trained:
            logger.info("Training forecast model")
            metrics = model.train(df)
        
        # Prepare future dataframe with additional features if needed
        future_features = {}
        
        # Add weather data if requested
        if request.include_weather and request.city_id is not None:
            weather_data = fetch_weather_data(
                city_id=request.city_id,
                start_date=start_date,
                end_date=end_date,
                future_periods=request.periods
            )
            if not weather_data.empty:
                weather_data = weather_data.rename(columns={'date': 'ds'})
                future_features.update({
                    'temp_avg': weather_data[['ds', 'temp_avg']],
                    'humidity': weather_data[['ds', 'humidity']],
                    'precipitation': weather_data[['ds', 'precipitation']],
                    'wind_speed': weather_data[['ds', 'wind_speed']]
                })
        
        # Add promotion data if requested
        if request.include_promotions:
            promo_data = fetch_promotion_data(
                store_id=request.store_id,
                product_id=request.product_id,
                category_id=request.category_id,
                start_date=start_date,
                end_date=end_date
            )
            if not promo_data.empty:
                # Create daily flags for promotions
                date_range = pd.date_range(start=start_date, end=end_date)
                promo_flags = pd.DataFrame({'ds': date_range})
                promo_flags['promo_flag'] = False
                
                # For each promotion, mark the dates it's active
                for _, promo in promo_data.iterrows():
                    promo_start = max(pd.to_datetime(promo['start_date']), start_date)
                    promo_end = min(pd.to_datetime(promo['end_date']), end_date)
                    mask = (promo_flags['ds'] >= promo_start) & (promo_flags['ds'] <= promo_end)
                    promo_flags.loc[mask, 'promo_flag'] = True
                    
                # Add discount information if available
                if 'discount_percentage' in promo_data.columns:
                    promo_flags['discount'] = 0.0
                    for _, promo in promo_data.iterrows():
                        promo_start = max(pd.to_datetime(promo['start_date']), start_date)
                        promo_end = min(pd.to_datetime(promo['end_date']), end_date)
                        mask = (promo_flags['ds'] >= promo_start) & (promo_flags['ds'] <= promo_end)
                        promo_flags.loc[mask, 'discount'] = promo['discount_percentage']
                
                future_features.update({
                    'promo_flag': promo_flags[['ds', 'promo_flag']]
                })
                if 'discount' in promo_flags.columns:
                    future_features.update({
                        'discount': promo_flags[['ds', 'discount']]
                    })
        
        # Prepare future dataframe
        future_df = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=request.periods, freq=request.freq)})
        
        # Add future features if available
        for feature, feature_df in future_features.items():
            if not feature_df.empty:
                future_df = pd.merge(future_df, feature_df, on='ds', how='left')
        
        # Generate forecast
        forecast = model.forecast_future(future_df=future_df)
        
        # Prepare response
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={'ds': 'date', 'yhat': 'forecast', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}
        )
        
        response = {
            "forecast": forecast_data.to_dict(orient='records'),
            "metrics": model.metrics.to_dict() if model.metrics is not None else None,
        }
        
        # Add components if requested
        if request.return_components:
            components = model.get_forecast_components()
            component_dict = {}
            
            for component_name, component_data in components.items():
                if component_data is not None:
                    component_dict[component_name] = component_data.rename(
                        columns={'ds': 'date'}
                    ).to_dict(orient='records')
            
            response["components"] = component_dict
        
        return response
    
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

@app.post("/promotion/analysis/")
def analyze_promotion(request: PromotionAnalysisRequest, model: PromoUpliftModel = Depends(get_promo_model)):
    """
    Analyze promotion effectiveness for specified parameters.
    
    Parameters:
    - store_id: Optional store ID to filter by
    - product_id: Optional product ID to filter by
    - category_id: Optional category ID to filter by
    - city_id: Optional city ID to filter by
    - start_date: Start date for analysis
    - end_date: End date for analysis
    - promotion_type: Optional promotion type to filter by
    - discount_min: Minimum discount percentage
    - discount_max: Maximum discount percentage
    
    Returns:
    - Promotion effectiveness analysis
    """
    try:
        # Parse dates
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        # Get historical data
        df = fetch_historical_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=request.city_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data available for the specified parameters")
        
        # Get promotion data
        promo_data = fetch_promotion_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Merge promotion details if available
        if not promo_data.empty:
            # For each day in the sales data, check if it falls within a promotion period
            for idx, row in df.iterrows():
                sale_date = pd.to_datetime(row['sale_date'])
                
                # Check if this date falls within any promotion periods
                for _, promo in promo_data.iterrows():
                    promo_start = pd.to_datetime(promo['start_date'])
                    promo_end = pd.to_datetime(promo['end_date'])
                    
                    if promo_start <= sale_date <= promo_end:
                        if row['store_id'] == promo['store_id'] or pd.isna(promo['store_id']):
                            if row['product_id'] == promo['product_id'] or pd.isna(promo['product_id']):
                                # Update promotion type if it exists
                                if 'promotion_type' in promo:
                                    df.at[idx, 'promotion_type'] = promo['promotion_type']
                                
                                # Update discount percentage if it exists
                                if 'discount_percentage' in promo:
                                    df.at[idx, 'discount_percentage'] = promo['discount_percentage']
        
        # Filter by promotion type if specified
        if request.promotion_type is not None and 'promotion_type' in df.columns:
            df = df[df['promotion_type'] == request.promotion_type]
        
        # Filter by discount range if specified
        if request.discount_min is not None and 'discount_percentage' in df.columns:
            df = df[df['discount_percentage'] >= request.discount_min]
            
        if request.discount_max is not None and 'discount_percentage' in df.columns:
            df = df[df['discount_percentage'] <= request.discount_max]
        
        # If model is not trained, train it on the available data
        if not model.trained:
            logger.info("Training promotion uplift model")
            model.train(df)
        
        # Analyze promotion effectiveness
        analysis = model.analyze_promotion_effectiveness(df)
        
        return {
            "analysis": analysis.to_dict(orient='records'),
            "summary": {
                "average_uplift": analysis["uplift_mean"].mean(),
                "median_uplift": analysis["uplift_mean"].median(),
                "total_records": len(df),
                "total_promotions": df['promo_flag'].sum() if 'promo_flag' in df.columns else None,
                "date_range": f"{start_date.date()} to {end_date.date()}"
            }
        }
    
    except Exception as e:
        logger.error(f"Promotion analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Promotion analysis error: {str(e)}")

@app.post("/stockout/analysis/")
def analyze_stockout(request: StockoutAnalysisRequest):
    """
    Analyze the impact of stockouts and estimate demand during stockout periods.
    
    Parameters:
    - store_id: Store ID
    - product_id: Product ID
    - start_date: Start date for analysis
    - end_date: End date for analysis
    
    Returns:
    - Stockout analysis with estimated demand during stockout periods
    """
    try:
        # Parse dates
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        # Get historical data
        df = fetch_historical_data(
            store_id=request.store_id,
            product_id=request.product_id,
            start_date=start_date - timedelta(days=30),  # Get some extra history for better estimation
            end_date=end_date
        )
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data available for the specified parameters")
        
        # Identify stockout periods
        df['is_stockout'] = df['stock_hour6_22_cnt'] == 0
        
        # Calculate average sales on non-stockout days by day of week
        avg_sales_by_dow = df[~df['is_stockout']].groupby(
            pd.to_datetime(df[~df['is_stockout']]['sale_date']).dt.dayofweek
        )['sale_amount'].mean().to_dict()
        
        # Estimate demand for stockout days
        stockout_days = df[df['is_stockout']].copy()
        if not stockout_days.empty:
            stockout_days['dow'] = pd.to_datetime(stockout_days['sale_date']).dt.dayofweek
            stockout_days['estimated_demand'] = stockout_days['dow'].map(avg_sales_by_dow)
            stockout_days['lost_sales'] = stockout_days['estimated_demand'] - stockout_days['sale_amount']
            stockout_days['lost_sales'] = stockout_days['lost_sales'].clip(lower=0)
        else:
            stockout_days = pd.DataFrame(columns=['sale_date', 'estimated_demand', 'lost_sales', 'sale_amount'])
        
        # Calculate summary metrics
        if not stockout_days.empty:
            total_lost_sales = stockout_days['lost_sales'].sum()
            stockout_days_count = len(stockout_days)
            avg_daily_lost_sales = stockout_days['lost_sales'].mean() if stockout_days_count > 0 else 0
            stockout_rate = len(stockout_days) / len(df) if len(df) > 0 else 0
        else:
            total_lost_sales = 0
            stockout_days_count = 0
            avg_daily_lost_sales = 0
            stockout_rate = 0
        
        # Prepare response
        return {
            "stockout_analysis": stockout_days[[
                'sale_date', 'sale_amount', 'estimated_demand', 'lost_sales'
            ]].to_dict(orient='records'),
            "summary": {
                "total_lost_sales": float(total_lost_sales),
                "stockout_days_count": stockout_days_count,
                "avg_daily_lost_sales": float(avg_daily_lost_sales),
                "stockout_rate": float(stockout_rate),
                "date_range": f"{start_date.date()} to {end_date.date()}"
            }
        }
    
    except Exception as e:
        logger.error(f"Stockout analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stockout analysis error: {str(e)}")

@app.post("/holiday/impact/")
def analyze_holiday_impact(request: HolidayImpactRequest):
    """
    Analyze the impact of holidays on sales.
    
    Parameters:
    - product_id: Optional product ID to filter by
    - category_id: Optional category ID to filter by
    - holiday_name: Optional specific holiday to analyze
    - start_date: Start date for analysis
    - end_date: End date for analysis
    
    Returns:
    - Holiday impact analysis
    """
    try:
        # Parse dates
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        # Get historical data
        df = fetch_historical_data(
            product_id=request.product_id,
            category_id=request.category_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data available for the specified parameters")
        
        # Get holiday data
        holidays_df = fetch_holiday_data(start_date=start_date, end_date=end_date)
        
        # Filter by holiday name if specified
        if request.holiday_name is not None:
            holidays_df = holidays_df[holidays_df['holiday_name'].str.contains(request.holiday_name, case=False)]
        
        # If no holidays in the selected period, return empty response
        if holidays_df.empty:
            return {"holiday_impact": [], "summary": {"message": "No holidays found in the specified date range"}}
        
        # Merge holiday data with sales data
        sales_data = df.copy()
        sales_data['sale_date'] = pd.to_datetime(sales_data['sale_date'])
        
        holidays_df['date'] = pd.to_datetime(holidays_df['date'])
        
        # For each holiday, analyze sales impact
        holiday_impacts = []
        
        for _, holiday in holidays_df.iterrows():
            holiday_date = holiday['date']
            
            # Get sales on the holiday
            holiday_sales = sales_data[sales_data['sale_date'] == holiday_date]
            
            if holiday_sales.empty:
                continue
            
            # Get average sales for the same day of week in non-holiday periods
            day_of_week = holiday_date.dayofweek
            
            # 4 weeks before and 4 weeks after
            comparison_start = holiday_date - timedelta(days=28)
            comparison_end = holiday_date + timedelta(days=28)
            
            comparison_sales = sales_data[
                (sales_data['sale_date'] != holiday_date) & 
                (sales_data['sale_date'] >= comparison_start) & 
                (sales_data['sale_date'] <= comparison_end) &
                (sales_data['sale_date'].dt.dayofweek == day_of_week)
            ]
            
            if comparison_sales.empty:
                continue
            
            # Calculate metrics
            holiday_total_sales = holiday_sales['sale_amount'].sum()
            comparison_avg_sales = comparison_sales.groupby('sale_date')['sale_amount'].sum().mean()
            
            sales_lift = holiday_total_sales - comparison_avg_sales
            sales_lift_pct = (sales_lift / comparison_avg_sales) * 100 if comparison_avg_sales > 0 else 0
            
            holiday_impacts.append({
                "holiday_date": holiday_date.strftime("%Y-%m-%d"),
                "holiday_name": holiday['holiday_name'],
                "holiday_type": holiday['holiday_type'] if 'holiday_type' in holiday else None,
                "holiday_sales": float(holiday_total_sales),
                "normal_day_avg_sales": float(comparison_avg_sales),
                "sales_lift": float(sales_lift),
                "sales_lift_pct": float(sales_lift_pct)
            })
        
        # Calculate overall metrics
        if holiday_impacts:
            avg_lift_pct = sum(impact['sales_lift_pct'] for impact in holiday_impacts) / len(holiday_impacts)
            total_lift = sum(impact['sales_lift'] for impact in holiday_impacts)
        else:
            avg_lift_pct = 0
            total_lift = 0
        
        # Prepare response
        return {
            "holiday_impact": holiday_impacts,
            "summary": {
                "total_holidays_analyzed": len(holiday_impacts),
                "average_sales_lift_pct": avg_lift_pct,
                "total_sales_lift": total_lift
            }
        }
    
    except Exception as e:
        logger.error(f"Holiday impact analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Holiday impact analysis error: {str(e)}")

if __name__ == "__main__":
    # This is used when running locally
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
