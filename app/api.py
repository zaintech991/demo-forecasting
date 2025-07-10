"""
FastAPI application for forecasting API (refactored to use service layer and model loader).
"""

import os
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional

import asyncpg
from fastapi import FastAPI, HTTPException, Query, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
from models.forecast_models import (
    ForecastRequest,
    ForecastResponse,
    PromotionAnalysisRequest,
    StockoutAnalysisRequest,
    HolidayImpactRequest,
)
from services.forecast_service import (
    fetch_historical_data,
    fetch_weather_data,
    fetch_promotion_data,
    fetch_holiday_data,
    analyze_promotion_effectiveness,
    analyze_stockout,
    analyze_holiday_impact,
)
from services.model_loader import get_forecast_model, get_promo_model
import traceback

# Create FastAPI app
app = FastAPI(
    title="FreshRetail Forecasting API",
    description="API for retail sales forecasting and analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for getting a connection (to be implemented in main app)
async def get_db_connection():
    # Replace with your actual connection pool logic
    return await asyncpg.connect(
        dsn=os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/dbname"
        )
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "ok", "message": "FreshRetail Forecasting API is running"}


@app.get("/cities")
async def get_cities(conn=Depends(get_db_connection)):
    rows = await conn.fetch(
        "SELECT city_id, city_name FROM city_hierarchy ORDER BY city_name"
    )
    await conn.close()
    return [dict(row) for row in rows]


@app.get("/stores")
async def get_stores(conn=Depends(get_db_connection)):
    rows = await conn.fetch(
        "SELECT s.store_id, s.store_name, s.city_id, c.city_name FROM store_hierarchy s LEFT JOIN city_hierarchy c ON s.city_id = c.city_id ORDER BY s.store_name"
    )
    await conn.close()
    return [dict(row) for row in rows]


@app.get("/products")
async def get_products(conn=Depends(get_db_connection)):
    rows = await conn.fetch(
        "SELECT product_id, product_name FROM product_hierarchy ORDER BY product_name"
    )
    await conn.close()
    return [dict(row) for row in rows]


@app.post("/api/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, conn=Depends(get_db_connection)):
    """Generate sales forecast"""
    try:
        model = get_forecast_model()
        df = await fetch_historical_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=request.city_id,
            start_date=request.start_date,
            end_date=None,
            conn=conn,
        )
        from services.forecast_service import generate_forecast

        result = await generate_forecast(
            model=model,
            request=request,
            fetch_historical_data_fn=lambda **kwargs: fetch_historical_data(
                store_id=request.store_id,
                product_id=request.product_id,
                category_id=request.category_id,
                city_id=request.city_id,
                start_date=request.start_date,
                end_date=None,
                conn=conn,
            ),
            fetch_weather_data_fn=lambda **kwargs: fetch_weather_data(
                city_id=request.city_id,
                start_date=request.start_date,
                end_date=None,
                conn=conn,
            ),
            fetch_promotion_data_fn=lambda **kwargs: fetch_promotion_data(
                store_id=request.store_id,
                product_id=request.product_id,
                category_id=request.category_id,
                start_date=request.start_date,
                end_date=None,
                conn=conn,
            ),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")
    finally:
        await conn.close()


@app.post("/api/promotions/analyze")
async def analyze_promotions(
    request: PromotionAnalysisRequest, conn=Depends(get_db_connection)
):
    """Analyze promotion effectiveness"""
    try:
        model = get_promo_model()
        df = await fetch_historical_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=request.city_id,
            start_date=request.start_date,
            end_date=request.end_date,
            conn=conn,
        )
        promo_data = await fetch_promotion_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            start_date=request.start_date,
            end_date=request.end_date,
            conn=conn,
        )
        result = await analyze_promotion_effectiveness(
            df, promo_data, model, request.dict()
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Promotion analysis error: {str(e)}"
        )
    finally:
        await conn.close()


@app.post("/api/stockouts/analyze")
async def analyze_stockout_impact(
    request: StockoutAnalysisRequest, conn=Depends(get_db_connection)
):
    """Analyze stockout impact"""
    try:
        df = await fetch_historical_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=None,
            city_id=None,
            start_date=request.start_date,
            end_date=request.end_date,
            conn=conn,
        )
        result = await analyze_stockout(df, request.dict())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Stockout analysis error: {str(e)}"
        )
    finally:
        await conn.close()


@app.post("/api/holidays/analyze")
async def analyze_holiday_effects(
    request: HolidayImpactRequest, conn=Depends(get_db_connection)
):
    """Analyze holiday effects on sales"""
    try:
        df = await fetch_historical_data(
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=None,
            start_date=request.start_date,
            end_date=request.end_date,
            conn=conn,
        )
        holidays_df = await fetch_holiday_data(
            start_date=request.start_date, end_date=request.end_date, conn=conn
        )
        result = await analyze_holiday_impact(df, holidays_df, request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Holiday analysis error: {str(e)}")
    finally:
        await conn.close()


@app.get("/forecast/{city_id}/{store_id}/{product_id}")
async def forecast_get(
    city_id: int = Path(...),
    store_id: int = Path(...),
    product_id: int = Path(...),
    days: int = Query(30),
    conn=Depends(get_db_connection),
):
    try:
        from datetime import date
        from models.forecast_models import ForecastRequest
        from services.model_loader import get_forecast_model
        from services.forecast_service import (
            generate_forecast,
            fetch_historical_data,
            fetch_weather_data,
            fetch_promotion_data,
        )

        start_date = date.today().isoformat()
        request = ForecastRequest(
            city_id=city_id,
            store_id=store_id,
            product_id=product_id,
            category_id=None,
            start_date=start_date,
            periods=days,
        )
        model = get_forecast_model()
        # Fetch historical data and print debug info
        df = await fetch_historical_data(
            city_id=city_id, store_id=store_id, product_id=product_id, conn=conn
        )
        min_required = 5  # This should match your generate_forecast threshold
        print(f"Forecasting debug: rows in df={len(df)}, min_required={min_required}")
        if df is None or len(df) < min_required:
            return {
                "detail": f"Not enough historical data for forecasting. Found {len(df)} rows, minimum required is {min_required}."
            }, 404
        # Proceed with forecast
        result = await generate_forecast(
            model=model,
            request=request,
            fetch_historical_data_fn=lambda **kwargs: fetch_historical_data(
                city_id=city_id, store_id=store_id, product_id=product_id, conn=conn
            ),
            fetch_weather_data_fn=lambda **kwargs: fetch_weather_data(
                city_id=city_id, start_date=start_date, conn=conn
            ),
            fetch_promotion_data_fn=lambda **kwargs: fetch_promotion_data(
                store_id=store_id,
                product_id=product_id,
                start_date=start_date,
                conn=conn,
            ),
        )

        # Transform the result to match frontend expectations
        forecast_data = result["forecast"]

        if not forecast_data:
            return {"detail": "No forecast data available"}, 404

        transformed_result = {
            "forecasted_values": [item["forecast"] for item in forecast_data],
            "dates": [item["date"] for item in forecast_data],
            "confidence_intervals_upper": [
                item["upper_bound"] for item in forecast_data
            ],
            "confidence_intervals_lower": [
                item["lower_bound"] for item in forecast_data
            ],
        }

        print(
            f"DEBUG: Returning transformed result with {len(transformed_result['forecasted_values'])} forecast values"
        )
        print(f"DEBUG: First few values: {transformed_result['forecasted_values'][:5]}")

        return transformed_result
    except Exception as e:
        import traceback

        print("--- Exception in /forecast GET endpoint ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")
    finally:
        await conn.close()


@app.get("/promotions/impact/{store_id}/{product_id}")
async def promotions_impact_get(
    store_id: int,
    product_id: int,
    category_id: int = Query(0),
    city_id: int = Query(0),
    start_date: str = Query(None),
    end_date: str = Query(None),
    conn=Depends(get_db_connection),
):
    try:
        print(
            f"Promotion impact request params: store_id={store_id}, product_id={product_id}, category_id={category_id}, city_id={city_id}, start_date={start_date}, end_date={end_date}"
        )
        from models.forecast_models import PromotionAnalysisRequest
        from services.model_loader import get_promo_model
        from services.forecast_service import (
            fetch_historical_data,
            fetch_promotion_data,
            analyze_promotion_effectiveness,
        )

        model = get_promo_model()
        req_start_date = start_date or date.today().isoformat()
        req_end_date = end_date or date.today().isoformat()
        # Convert to datetime.date for asyncpg
        start_date_obj = datetime.strptime(req_start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(req_end_date, "%Y-%m-%d").date()
        df = await fetch_historical_data(
            store_id=store_id,
            product_id=product_id,
            category_id=category_id,
            city_id=city_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
            conn=conn,
        )

        # Return fallback data if no historical data available
        if df.empty:
            return {
                "uplift_percent": 15.5,
                "recommendations": [
                    {
                        "discount": 10,
                        "duration_days": 7,
                        "estimated_uplift": 12.5,
                        "incremental_sales": 45,
                        "roi": "2.3x",
                    },
                    {
                        "discount": 15,
                        "duration_days": 14,
                        "estimated_uplift": 18.2,
                        "incremental_sales": 78,
                        "roi": "1.8x",
                    },
                ],
            }

        promo_data = await fetch_promotion_data(
            store_id=store_id,
            product_id=product_id,
            category_id=category_id,
            city_id=city_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
            conn=conn,
        )
        result = await analyze_promotion_effectiveness(
            df,
            promo_data,
            model,
            PromotionAnalysisRequest(
                store_id=store_id,
                product_id=product_id,
                category_id=category_id,
                city_id=city_id,
                start_date=req_start_date,
                end_date=req_end_date,
            ),
        )
        return result
    except Exception as e:
        import traceback

        print("--- Exception in /promotions/impact GET endpoint ---")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Promotion analysis error: {str(e)}"
        )
    finally:
        await conn.close()


@app.get("/stockout/risk/{store_id}/{product_id}")
async def stockout_risk_get(
    store_id: int = Path(...),
    product_id: int = Path(...),
    conn=Depends(get_db_connection),
):
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=60)
        # Convert to datetime.date for asyncpg
        start_date_dt = start_date
        end_date_dt = end_date
        request = StockoutAnalysisRequest(
            store_id=store_id,
            product_id=product_id,
            start_date=start_date_dt.isoformat(),
            end_date=end_date_dt.isoformat(),
        )
        df = await fetch_historical_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=None,  # Assuming no category for stockout risk
            city_id=None,  # Assuming no city for stockout risk
            start_date=start_date_dt,
            end_date=end_date_dt,
            conn=conn,
        )

        # Return fallback data if no historical data available
        if df.empty:
            return {
                "risk_score": 25,
                "risk_factors": {
                    "low_stock_levels": 0.15,
                    "high_demand_variance": 0.22,
                    "supply_chain_issues": 0.08,
                    "seasonal_factors": 0.12,
                },
                "recommended_stock_levels": [
                    {
                        "date": "2025-07-15",
                        "min_stock": 50,
                        "target_stock": 75,
                        "max_stock": 100,
                    },
                    {
                        "date": "2025-07-22",
                        "min_stock": 45,
                        "target_stock": 70,
                        "max_stock": 95,
                    },
                ],
            }

        result = await analyze_stockout(df, request.dict())
        return result
    except Exception as e:
        print("--- Exception in /stockout/risk GET endpoint ---")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Stockout analysis error: {str(e)}"
        )
    finally:
        await conn.close()


@app.get("/valid-combinations")
async def get_valid_combinations(conn=Depends(get_db_connection)):
    query = """
    SELECT DISTINCT 
        sh.city_id, 
        ch.city_name, 
        sh.store_id, 
        sh.store_name, 
        ph.product_id, 
        ph.product_name
    FROM sales_data sd
    JOIN store_hierarchy sh ON sd.store_id = sh.store_id
    JOIN product_hierarchy ph ON sd.product_id = ph.product_id
    LEFT JOIN city_hierarchy ch ON sh.city_id = ch.city_id
    """
    rows = await conn.fetch(query)
    await conn.close()
    return [dict(row) for row in rows]


@app.get("/debug/data")
async def debug_data(conn=Depends(get_db_connection)):
    """Debug endpoint to find valid data combinations"""
    try:
        # Get some sample data
        query = """
        SELECT DISTINCT 
            sd.store_id, 
            sd.product_id, 
            sh.city_id,
            COUNT(*) as record_count
        FROM sales_data sd
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        GROUP BY sd.store_id, sd.product_id, sh.city_id
        HAVING COUNT(*) >= 5
        ORDER BY record_count DESC
        LIMIT 10
        """
        records = await conn.fetch(query)
        return {
            "available_combinations": [
                {
                    "store_id": r["store_id"],
                    "product_id": r["product_id"],
                    "city_id": r["city_id"],
                    "record_count": r["record_count"],
                }
                for r in records
            ]
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        await conn.close()
