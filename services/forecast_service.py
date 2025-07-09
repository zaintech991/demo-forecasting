import asyncpg
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

# Forecasting and analysis logic moved from api/forecast.py

async def generate_forecast(
    model,
    request,
    fetch_historical_data_fn=None,
    fetch_weather_data_fn=None,
    fetch_promotion_data_fn=None
) -> Dict[str, Any]:
    """
    Generate sales forecast for specified parameters.
    """
    print("=" * 50)
    print("DEBUG: Starting generate_forecast function")
    print(f"DEBUG: Request parameters: {request}")
    print(f"DEBUG: Model type: {type(model)}")
    print(f"DEBUG: Model trained status: {getattr(model, 'trained', 'Unknown')}")
    
    # Await the data fetches
    print("DEBUG: Fetching historical data...")
    df = await fetch_historical_data_fn() if fetch_historical_data_fn else None
    print("DEBUG: Historical data fetched")
    
    print("DEBUG: Fetching weather data...")
    weather_df = await fetch_weather_data_fn() if fetch_weather_data_fn else None
    print("DEBUG: Weather data fetched")
    
    print("DEBUG: Fetching promotion data...")
    promo_df = await fetch_promotion_data_fn() if fetch_promotion_data_fn else None
    print("DEBUG: Promotion data fetched")

    print("DEBUG: DataFrame shape:", df.shape if df is not None else "None")
    print("DEBUG: DataFrame columns:", df.columns.tolist() if df is not None else "None")
    print("DEBUG: DataFrame head:", df.head() if df is not None else "None")
    
    if df is not None:
        print("DEBUG: DataFrame info:")
        print(df.info())
        print("DEBUG: DataFrame describe:")
        print(df.describe())

    # Now proceed as before
    print("DEBUG: Processing request parameters...")
    start_date = pd.to_datetime(request.start_date)
    print(f"DEBUG: Start date: {start_date}")
    
    if request.freq == "D":
        end_date = start_date + timedelta(days=request.periods)
    elif request.freq == "W":
        end_date = start_date + timedelta(weeks=request.periods)
    elif request.freq == "M":
        end_date = start_date + pd.DateOffset(months=request.periods)
    else:
        end_date = start_date + timedelta(days=request.periods)
    
    print(f"DEBUG: End date: {end_date}")
    print(f"DEBUG: Frequency: {request.freq}")
    print(f"DEBUG: Periods: {request.periods}")
    
    hist_start_date = start_date - timedelta(days=365)
    print(f"DEBUG: Historical start date: {hist_start_date}")
    
    # Reduce minimum data requirement for dev/testing
    if df is not None and len(df) < 5:
        print("DEBUG: ERROR - Not enough historical data for forecasting.")
        raise ValueError("Not enough historical data for forecasting.")
    
    print("DEBUG: Configuring model...")
    # Configure the model
    model.include_holiday = request.include_holidays
    model.include_weather = request.include_weather
    model.include_promotions = request.include_promotions
    model.forecast_periods = request.periods
    model.forecast_freq = request.freq
    
    print(f"DEBUG: Model configuration:")
    print(f"  - include_holiday: {getattr(model, 'include_holiday', 'Not set')}")
    print(f"  - include_weather: {getattr(model, 'include_weather', 'Not set')}")
    print(f"  - include_promotions: {getattr(model, 'include_promotions', 'Not set')}")
    print(f"  - forecast_periods: {getattr(model, 'forecast_periods', 'Not set')}")
    print(f"  - forecast_freq: {getattr(model, 'forecast_freq', 'Not set')}")
    
    # Train the model if needed
    if not model.trained:
        print("DEBUG: Training forecast model...")
        try:
            metrics = model.train(df)
            print(f"DEBUG: Model training completed. Metrics: {metrics}")
        except Exception as e:
            print(f"DEBUG: ERROR during model training: {str(e)}")
            raise
    else:
        print("DEBUG: Model already trained, skipping training")
    
    # Prepare future dataframe with additional features if needed
    print("DEBUG: Preparing future dataframe...")
    future_features = {}
    
    if request.include_weather and request.city_id is not None and weather_df is not None:
        print("DEBUG: Adding weather features to future dataframe...")
        weather_data = weather_df.rename(columns={'date': 'ds'})
        future_features.update({
            'temp_avg': weather_data[['ds', 'temp_avg']],
            'humidity': weather_data[['ds', 'humidity']],
            'precipitation': weather_data[['ds', 'precipitation']],
            'wind_speed': weather_data[['ds', 'wind_speed']]
        })
        print(f"DEBUG: Weather features added: {list(future_features.keys())}")
    
    if request.include_promotions and promo_df is not None:
        print("DEBUG: Adding promotion features to future dataframe...")
        date_range = pd.date_range(start=start_date, end=end_date)
        promo_flags = pd.DataFrame({'ds': date_range})
        promo_flags['promo_flag'] = False
        for _, promo in promo_df.iterrows():
            promo_start = max(pd.to_datetime(promo['start_date']), start_date)
            promo_end = min(pd.to_datetime(promo['end_date']), end_date)
            mask = (promo_flags['ds'] >= promo_start) & (promo_flags['ds'] <= promo_end)
            promo_flags.loc[mask, 'promo_flag'] = True
        if 'discount_percentage' in promo_df.columns:
            promo_flags['discount'] = 0.0
            for _, promo in promo_df.iterrows():
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
        print(f"DEBUG: Promotion features added: {list(future_features.keys())}")
    
    print("DEBUG: Creating base future dataframe...")
    future_df = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=request.periods, freq=request.freq)})
    print(f"DEBUG: Base future dataframe shape: {future_df.shape}")
    print(f"DEBUG: Base future dataframe columns: {future_df.columns.tolist()}")
    
    for feature, feature_df in future_features.items():
        if not feature_df.empty:
            print(f"DEBUG: Merging feature '{feature}' into future dataframe...")
            print(f"DEBUG: Feature dataframe shape: {feature_df.shape}")
            print(f"DEBUG: Feature dataframe columns: {feature_df.columns.tolist()}")
            future_df = pd.merge(future_df, feature_df, on='ds', how='left')
            print(f"DEBUG: Future dataframe shape after merge: {future_df.shape}")
    
    print("DEBUG: Final future dataframe:")
    print(f"  - Shape: {future_df.shape}")
    print(f"  - Columns: {future_df.columns.tolist()}")
    print(f"  - Head: {future_df.head()}")
    
    print("DEBUG: Calling model.forecast_future...")
    try:
        forecast = model.forecast_future(future_df=future_df)
        print(f"DEBUG: Forecast completed. Shape: {forecast.shape if hasattr(forecast, 'shape') else 'No shape'}")
        print(f"DEBUG: Forecast columns: {forecast.columns.tolist() if hasattr(forecast, 'columns') else 'No columns'}")
        print(f"DEBUG: Forecast head: {forecast.head() if hasattr(forecast, 'head') else forecast}")
    except Exception as e:
        print(f"DEBUG: ERROR during forecasting: {str(e)}")
        raise
    
    print("DEBUG: Processing forecast results...")
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'date', 'yhat': 'forecast', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}
    )
    print(f"DEBUG: Processed forecast data shape: {forecast_data.shape}")
    print(f"DEBUG: Processed forecast data columns: {forecast_data.columns.tolist()}")
    
    response = {
        "forecast": forecast_data.to_dict(orient='records'),
        "metrics": model.metrics if model.metrics is not None else None,
    }
    
    print(f"DEBUG: Response created with {len(response['forecast'])} forecast records")
    
    if getattr(request, 'return_components', False):
        print("DEBUG: Adding forecast components to response...")
        components = model.get_forecast_components()
        component_dict = {}
        for component_name, component_data in components.items():
            if component_data is not None:
                component_dict[component_name] = component_data.rename(
                    columns={'ds': 'date'}
                ).to_dict(orient='records')
        response["components"] = component_dict
        print(f"DEBUG: Added {len(component_dict)} components to response")
    
    print("DEBUG: generate_forecast function completed successfully")
    print("=" * 50)
    return response

# Promotion analysis, stockout analysis, holiday impact analysis, and fetch_*_data functions
# should be similarly moved here, following the same pattern as above. 

async def fetch_historical_data(store_id=None, product_id=None, category_id=None, city_id=None, start_date=None, end_date=None, conn=None) -> pd.DataFrame:
    """
    Async fetch historical sales data from the database using asyncpg.
    Ensures all required columns are present in the DataFrame, including store_id, product_id, sale_date, category_id, city_id.
    """
    # Convert date strings to datetime.date objects for asyncpg
    if start_date is not None and isinstance(start_date, str):
        from datetime import datetime
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is not None and isinstance(end_date, str):
        from datetime import datetime
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    query = """
    SELECT 
        sd.sale_date,
        sd.store_id,
        sd.product_id,
        ph.first_category_id AS category_id,
        sh.city_id,
        sd.sale_amount,
        sd.sale_qty,
        sd.discount,
        sd.original_price,
        sd.stock_hour6_22_cnt,
        sd.stock_hour6_14_cnt,
        sd.stock_hour14_22_cnt,
        sd.holiday_flag,
        sd.promo_flag,
        sd.created_at
    FROM sales_data sd
    JOIN store_hierarchy sh ON sd.store_id = sh.store_id
    JOIN product_hierarchy ph ON sd.product_id = ph.product_id
    WHERE 1=1
    """
    params = []
    idx = 1
    if store_id is not None:
        query += f" AND sd.store_id = ${idx}"
        params.append(store_id)
        idx += 1
    if product_id is not None:
        query += f" AND sd.product_id = ${idx}"
        params.append(product_id)
        idx += 1
    if category_id is not None:
        query += f" AND ph.first_category_id = ${idx}"
        params.append(category_id)
        idx += 1
    if city_id is not None:
        query += f" AND sh.city_id = ${idx}"
        params.append(city_id)
        idx += 1
    if start_date is not None:
        query += f" AND sd.sale_date >= ${idx}"
        params.append(start_date)
        idx += 1
    if end_date is not None:
        query += f" AND sd.sale_date <= ${idx}"
        params.append(end_date)
        idx += 1
    query += " ORDER BY sd.sale_date"
    records = await conn.fetch(query, *params)
    df = pd.DataFrame(records, columns=[k for k in records[0].keys()]) if records else pd.DataFrame()
    # Ensure all required columns are present
    required_cols = [
        'store_id', 'product_id', 'sale_date', 'sale_amount', 'promo_flag', 'stock_hour6_22_cnt', 'sale_qty', 'discount', 'original_price',
        'stock_hour6_14_cnt', 'stock_hour14_22_cnt', 'holiday_flag', 'created_at', 'category_id', 'city_id'
    ]
    for col in required_cols:
        if col not in df.columns:
            if col == 'promo_flag':
                df[col] = False
            elif col in ['sale_amount', 'original_price', 'discount']:
                df[col] = 0.0
            elif col == 'created_at':
                from datetime import datetime
                df[col] = datetime.now()
            elif col in ['sale_date', 'category_id', 'city_id']:
                df[col] = None
            else:
                df[col] = 0
    print('fetch_historical_data columns:', df.columns.tolist())
    return df

async def fetch_weather_data(city_id=None, start_date=None, end_date=None, future_periods=0, conn=None) -> pd.DataFrame:
    """
    Async fetch weather data from the database using asyncpg.
    """
    # Convert date strings to datetime.date objects for asyncpg
    if start_date is not None and isinstance(start_date, str):
        from datetime import datetime
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is not None and isinstance(end_date, str):
        from datetime import datetime
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
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
    params = []
    idx = 1
    if city_id is not None:
        query += f" AND city_id = ${idx}"
        params.append(city_id)
        idx += 1
    if start_date is not None:
        query += f" AND date >= ${idx}"
        params.append(start_date)
        idx += 1
    if end_date is not None:
        query += f" AND date <= ${idx}"
        params.append(end_date)
        idx += 1
    query += " ORDER BY date"
    records = await conn.fetch(query, *params)
    df = pd.DataFrame(records, columns=[k for k in records[0].keys()]) if records else pd.DataFrame()
    return df

async def fetch_promotion_data(store_id=None, product_id=None, category_id=None, start_date=None, end_date=None, future_periods=0, conn=None) -> pd.DataFrame:
    """
    Async fetch promotion data from the database using asyncpg.
    Ensures all required columns are present in the DataFrame, including store_id, product_id, start_date, end_date.
    """
    # Convert date strings to datetime.date objects for asyncpg
    if start_date is not None and isinstance(start_date, str):
        from datetime import datetime
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is not None and isinstance(end_date, str):
        from datetime import datetime
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
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
    params = []
    idx = 1
    if store_id is not None:
        query += f" AND pe.store_id = ${idx}"
        params.append(store_id)
        idx += 1
    if product_id is not None:
        query += f" AND pe.product_id = ${idx}"
        params.append(product_id)
        idx += 1
    if category_id is not None:
        query += f" AND ph.first_category_id = ${idx}"
        params.append(category_id)
        idx += 1
    if start_date is not None:
        query += f" AND pe.end_date >= ${idx}"
        params.append(start_date)
        idx += 1
    if end_date is not None:
        query += f" AND pe.start_date <= ${idx}"
        params.append(end_date)
        idx += 1
    query += " ORDER BY pe.start_date"
    records = await conn.fetch(query, *params)
    df = pd.DataFrame(records, columns=[k for k in records[0].keys()]) if records else pd.DataFrame()
    # Ensure all required columns are present
    required_cols = [
        'store_id', 'product_id', 'start_date', 'end_date', 'promotion_type', 'discount_percentage', 'first_category_id', 'promo_flag'
    ]
    for col in required_cols:
        if col not in df.columns:
            if col == 'promo_flag':
                df[col] = False
            elif col in ['discount_percentage']:
                df[col] = 0.0
            elif col in ['start_date', 'end_date', 'promotion_type', 'first_category_id']:
                df[col] = None
            else:
                df[col] = 0
    print('fetch_promotion_data columns:', df.columns.tolist())
    return df

async def fetch_holiday_data(start_date=None, end_date=None, future_periods=0, conn=None) -> pd.DataFrame:
    """
    Async fetch holiday data from the database using asyncpg.
    """
    # Convert date strings to datetime.date objects for asyncpg
    if start_date is not None and isinstance(start_date, str):
        from datetime import datetime
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is not None and isinstance(end_date, str):
        from datetime import datetime
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    query = """
    SELECT 
        date,
        holiday_name,
        holiday_type,
        significance
    FROM holiday_calendar
    WHERE 1=1
    """
    params = []
    idx = 1
    if start_date is not None:
        query += f" AND date >= ${idx}"
        params.append(start_date)
        idx += 1
    if end_date is not None:
        query += f" AND date <= ${idx}"
        params.append(end_date)
        idx += 1
    query += " ORDER BY date"
    records = await conn.fetch(query, *params)
    df = pd.DataFrame(records, columns=[k for k in records[0].keys()]) if records else pd.DataFrame()
    return df

# The analyze_* functions can be made async if they need to await DB or other async calls, otherwise they can remain sync if only using pandas.

async def analyze_promotion_effectiveness(df: pd.DataFrame, promo_data: pd.DataFrame, model, request) -> Dict[str, Any]:
    """
    Analyze promotion effectiveness for specified parameters.
    Ensures required columns are present.
    """
    # Merge promotion details if available
    if not promo_data.empty:
        for idx, row in df.iterrows():
            sale_date = pd.to_datetime(row['sale_date'])
            for _, promo in promo_data.iterrows():
                promo_start = pd.to_datetime(promo['start_date'])
                promo_end = pd.to_datetime(promo['end_date'])
                # Use .item() for scalar comparison if Series
                store_id_match = False
                product_id_match = False
                if 'store_id' in promo:
                    if isinstance(row['store_id'], pd.Series):
                        store_id_match = (row['store_id'] == promo['store_id']).any() or pd.isna(promo['store_id'])
                    else:
                        store_id_match = (row['store_id'] == promo['store_id']) or pd.isna(promo['store_id'])
                if 'product_id' in promo:
                    if isinstance(row['product_id'], pd.Series):
                        product_id_match = (row['product_id'] == promo['product_id']).any() or pd.isna(promo['product_id'])
                    else:
                        product_id_match = (row['product_id'] == promo['product_id']) or pd.isna(promo['product_id'])
                if promo_start <= sale_date <= promo_end and bool(store_id_match) and bool(product_id_match):
                    if 'promotion_type' in promo:
                        df.at[idx, 'promotion_type'] = promo['promotion_type']
                    if 'discount_percentage' in promo:
                        df.at[idx, 'discount_percentage'] = promo['discount_percentage']
    # Ensure required columns
    for col in ['sale_amount', 'promo_flag']:
        if col not in df.columns:
            if col == 'promo_flag':
                df[col] = False
            else:
                df[col] = 0.0
    # Filter by promotion type if specified
    if hasattr(request, 'promotion_type') and request.promotion_type is not None and 'promotion_type' in df.columns:
        if isinstance(df['promotion_type'], pd.Series):
            df = df[df['promotion_type'] == request.promotion_type]
    if hasattr(request, 'discount_min') and request.discount_min is not None and 'discount_percentage' in df.columns:
        if isinstance(df['discount_percentage'], pd.Series):
            df = df[df['discount_percentage'] >= request.discount_min]
    if hasattr(request, 'discount_max') and request.discount_max is not None and 'discount_percentage' in df.columns:
        if isinstance(df['discount_percentage'], pd.Series):
            df = df[df['discount_percentage'] <= request.discount_max]
    if not model.trained:
        logger.info("Training promotion uplift model")
        model.train(df)
    analysis = model.analyze_promotion_effectiveness(df)
    return {
        "analysis": analysis,
        "summary": {
            "average_uplift": analysis["uplift_mean"].mean() if "uplift_mean" in analysis else None,
            "median_uplift": analysis["uplift_mean"].median() if "uplift_mean" in analysis else None,
            "total_records": len(df),
            "total_promotions": df['promo_flag'].sum() if 'promo_flag' in df.columns else None,
        }
    }

async def analyze_stockout(df: pd.DataFrame, request) -> Dict[str, Any]:
    """
    Analyze the impact of stockouts and estimate demand during stockout periods.
    Ensures required columns are present.
    """
    if 'stock_hour6_22_cnt' not in df.columns:
        df['stock_hour6_22_cnt'] = 0
    if 'sale_amount' not in df.columns:
        df['sale_amount'] = 0.0
    df['is_stockout'] = df['stock_hour6_22_cnt'] == 0
    # Only proceed if DataFrame is not empty
    if not df.empty:
        # Ensure sale_dates is a pandas Series
        sale_dates = pd.to_datetime(df.loc[~df['is_stockout'], 'sale_date'])
        if isinstance(sale_dates, pd.Series) and not sale_dates.empty:
            avg_sales_by_dow = df.loc[~df['is_stockout']].groupby(sale_dates.dt.dayofweek)['sale_amount'].mean().to_dict()
        else:
            avg_sales_by_dow = {}
        stockout_days = df[df['is_stockout']].copy()
        if not stockout_days.empty:
            stockout_sale_dates = pd.to_datetime(stockout_days['sale_date'])
            if isinstance(stockout_sale_dates, pd.Series):
                stockout_days['dow'] = stockout_sale_dates.dt.dayofweek
            else:
                stockout_days['dow'] = 0
            # Use pd.Series for .map
            stockout_days['estimated_demand'] = pd.Series(stockout_days['dow']).map(avg_sales_by_dow)
            stockout_days['lost_sales'] = stockout_days['estimated_demand'] - stockout_days['sale_amount']
            stockout_days['lost_sales'] = stockout_days['lost_sales'].clip(lower=0)
        else:
            stockout_days = pd.DataFrame(columns=['sale_date', 'estimated_demand', 'lost_sales', 'sale_amount'])
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
        # Ensure the result is a DataFrame before calling to_dict
        result_df = stockout_days.loc[:, ['sale_date', 'sale_amount', 'estimated_demand', 'lost_sales']]
        if not isinstance(result_df, pd.DataFrame):
            result_df = pd.DataFrame(result_df)
        return {
            "stockout_analysis": result_df.to_dict(orient='records'),
            "summary": {
                "total_lost_sales": float(total_lost_sales),
                "stockout_days_count": stockout_days_count,
                "avg_daily_lost_sales": float(avg_daily_lost_sales),
                "stockout_rate": float(stockout_rate),
            }
        }
    else:
        return {
            "stockout_analysis": [],
            "summary": {
                "total_lost_sales": 0,
                "stockout_days_count": 0,
                "avg_daily_lost_sales": 0,
                "stockout_rate": 0,
            }
        }

async def analyze_holiday_impact(df: pd.DataFrame, holidays_df: pd.DataFrame, request) -> Dict[str, Any]:
    """
    Analyze the impact of holidays on sales.
    """
    sales_data = df.copy()
    sales_data['sale_date'] = pd.to_datetime(sales_data['sale_date'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    holiday_impacts = []
    for _, holiday in holidays_df.iterrows():
        holiday_date = holiday['date']
        holiday_sales = sales_data[sales_data['sale_date'] == holiday_date]
        if holiday_sales.empty:
            continue
        # Ensure day_of_week is an int, not ndarray or Series
        if isinstance(holiday_date, pd.Timestamp):
            day_of_week = holiday_date.dayofweek
        elif isinstance(holiday_date, np.ndarray):
            day_of_week = pd.Timestamp(holiday_date[0]).dayofweek
            holiday_date = holiday_date[0]
        elif isinstance(holiday_date, pd.Series):
            # If holiday_date is a Series, use the first value
            day_of_week = pd.Timestamp(holiday_date.iloc[0]).dayofweek
            holiday_date = holiday_date.iloc[0]
        elif hasattr(holiday_date, 'iloc'):
            # Fallback for other objects with iloc
            day_of_week = pd.Timestamp(holiday_date.iloc[0]).dayofweek
            holiday_date = holiday_date.iloc[0]
        else:
            day_of_week = pd.Timestamp(holiday_date).dayofweek
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
        holiday_total_sales = holiday_sales['sale_amount'].sum()
        comparison_avg_sales = comparison_sales.groupby('sale_date')['sale_amount'].sum().mean()
        sales_lift = holiday_total_sales - comparison_avg_sales
        sales_lift_pct = (sales_lift / comparison_avg_sales) * 100 if comparison_avg_sales > 0 else 0
        # Ensure holiday_date is a string
        if isinstance(holiday_date, pd.Timestamp):
            holiday_date_str = holiday_date.strftime("%Y-%m-%d")
        else:
            holiday_date_str = str(holiday_date)
        holiday_impacts.append({
            "holiday_date": holiday_date_str,
            "holiday_name": holiday['holiday_name'],
            "holiday_type": holiday['holiday_type'] if 'holiday_type' in holiday else None,
            "holiday_sales": float(holiday_total_sales),
            "normal_day_avg_sales": float(comparison_avg_sales),
            "sales_lift": float(sales_lift),
            "sales_lift_pct": float(sales_lift_pct)
        })
    if holiday_impacts:
        avg_lift_pct = sum(impact['sales_lift_pct'] for impact in holiday_impacts) / len(holiday_impacts)
        total_lift = sum(impact['sales_lift'] for impact in holiday_impacts)
    else:
        avg_lift_pct = 0
        total_lift = 0
    return {
        "holiday_impact": holiday_impacts,
        "summary": {
            "total_holidays_analyzed": len(holiday_impacts),
            "average_sales_lift_pct": avg_lift_pct,
            "total_sales_lift": total_lift
        }
    } 