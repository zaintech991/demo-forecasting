"""
Main application file for the Retail Sales Forecasting API.
"""
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Retail Sales Forecasting API",
    description="API for forecasting retail sales, analyzing promotions, and predicting stockout risk",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define response models
class ForecastResponse(BaseModel):
    store_id: int
    product_id: int
    dates: List[str]
    forecasted_values: List[float]
    confidence_intervals_lower: List[float]
    confidence_intervals_upper: List[float]

class PromotionAnalysisResponse(BaseModel):
    store_id: int
    product_id: int
    uplift_percent: float
    recommendations: List[Dict[str, Any]]

class StockoutRiskResponse(BaseModel):
    store_id: int
    product_id: int
    risk_score: float
    risk_factors: Dict[str, float]
    recommended_stock_levels: List[Dict[str, Any]]

# Mock data generation for demo purposes
def generate_mock_forecast(city_id: int, store_id: int, product_id: int, days: int = 30):
    """Generate mock forecast data for demo purposes"""
    np.random.seed(store_id + product_id)
    
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    
    # Generate base forecast with some seasonality
    base = 10 + (store_id % 5) * 2 + (product_id % 10)
    trend = np.linspace(0, 2, days)
    seasonality = 2 * np.sin(np.arange(days) * 2 * np.pi / 7)  # Weekly seasonality
    
    forecasted_values = base + trend + seasonality + np.random.normal(0, 1, days)
    forecasted_values = np.maximum(0, forecasted_values)  # No negative sales
    
    # Generate confidence intervals
    uncertainty = 1.0 + (np.arange(days) * 0.05)  # Increasing uncertainty over time
    confidence_lower = forecasted_values - uncertainty
    confidence_upper = forecasted_values + uncertainty
    
    # Ensure no negative values
    confidence_lower = np.maximum(0, confidence_lower)
    
    # Convert all NumPy arrays to Python lists with native Python types
    forecasted_values_list = [float(x) for x in forecasted_values.tolist()]
    confidence_lower_list = [float(x) for x in confidence_lower.tolist()]
    confidence_upper_list = [float(x) for x in confidence_upper.tolist()]
    
    return {
        "city_id": int(city_id),
        "store_id": int(store_id),
        "product_id": int(product_id),
        "dates": dates,
        "forecasted_values": forecasted_values_list,
        "confidence_intervals_lower": confidence_lower_list,
        "confidence_intervals_upper": confidence_upper_list
    }

def analyze_promotion_impact(store_id: int, product_id: int):
    """Generate mock promotion analysis for demo purposes"""
    np.random.seed(store_id + product_id)
    
    # Calculate uplift percentage (5-25%)
    uplift_percent = 5 + np.random.random() * 20
    
    # Generate recommendations
    discount_options = [10, 15, 20, 25, 30]
    durations = [7, 14, 21, 30]
    
    recommendations = []
    for _ in range(3):
        discount = int(np.random.choice(discount_options))  # Convert to Python int
        duration = int(np.random.choice(durations))  # Convert to Python int
        incremental_sales = round((discount * duration) / 10, 2)
        roi = round(incremental_sales * (100 - discount) / 100, 2)
        
        recommendations.append({
            "discount": discount,
            "duration_days": duration,
            "estimated_uplift": round(uplift_percent * (discount / 20), 2),
            "incremental_sales": incremental_sales,
            "roi": roi
        })
    
    return {
        "store_id": int(store_id),  # Ensure Python int
        "product_id": int(product_id),  # Ensure Python int
        "uplift_percent": round(uplift_percent, 2),
        "recommendations": recommendations
    }

def assess_stockout_risk(store_id: int, product_id: int):
    """Generate mock stockout risk assessment for demo purposes"""
    np.random.seed(store_id + product_id)
    
    # Generate risk score (0-100)
    risk_score = int(np.random.randint(0, 101))  # Convert to Python int
    
    # Generate risk factors
    factors = {
        "historical_stockouts": float(np.random.random() * 0.3),  # Convert to Python float
        "demand_volatility": float(np.random.random() * 0.25),  # Convert to Python float
        "forecast_uncertainty": float(np.random.random() * 0.2),  # Convert to Python float
        "supply_chain_delays": float(np.random.random() * 0.15),  # Convert to Python float
        "seasonality": float(np.random.random() * 0.1)  # Convert to Python float
    }
    
    # Normalize to make factors sum to 1
    factor_sum = sum(factors.values())
    factors = {k: round(v / factor_sum, 2) for k, v in factors.items()}
    
    # Generate recommended stock levels
    today = datetime.now()
    recommended_levels = []
    
    for i in range(7):  # Next 7 days
        date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
        base_level = 10 + (store_id % 5) * 2 + (product_id % 10)
        
        # Add some randomness and weekday effect
        weekday = (today.weekday() + i) % 7
        weekday_factor = 1.0 if weekday < 5 else 1.2  # Higher for weekends
        
        min_stock = int(round(base_level * 0.8 * weekday_factor))  # Convert to Python int
        target_stock = int(round(base_level * weekday_factor))  # Convert to Python int
        max_stock = int(round(base_level * 1.2 * weekday_factor))  # Convert to Python int
        
        recommended_levels.append({
            "date": date,
            "min_stock": min_stock,
            "target_stock": target_stock,
            "max_stock": max_stock
        })
    
    return {
        "store_id": int(store_id),  # Ensure Python int
        "product_id": int(product_id),  # Ensure Python int
        "risk_score": risk_score,
        "risk_factors": factors,
        "recommended_stock_levels": recommended_levels
    }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Retail Sales Forecasting API", 
        "docs": "/docs",
        "ui": "/static/index.html"
    }

@app.get("/api/forecast/{city_id}/{store_id}/{product_id}", response_model=ForecastResponse)
async def get_forecast(
    city_id: int, 
    store_id: int, 
    product_id: int, 
    days: int = Query(30, description="Number of days to forecast")
):
    """Get sales forecast for a specific product in a specific store"""
    # In a real application, you would fetch data from your database and use your forecast model
    forecast_data = generate_mock_forecast(city_id, store_id, product_id, days)
    return forecast_data

@app.get("/api/promotions/impact/{store_id}/{product_id}", response_model=PromotionAnalysisResponse)
async def get_promotion_impact(store_id: int, product_id: int):
    """Analyze the impact of promotions on a specific product in a specific store"""
    # In a real application, you would fetch data from your database and use your promotion model
    promotion_data = analyze_promotion_impact(store_id, product_id)
    return promotion_data

@app.get("/api/stockout/risk/{store_id}/{product_id}", response_model=StockoutRiskResponse)
async def get_stockout_risk(store_id: int, product_id: int):
    """Assess the risk of stockout for a specific product in a specific store"""
    # In a real application, you would fetch data from your database and use your stockout model
    stockout_data = assess_stockout_risk(store_id, product_id)
    return stockout_data

# Run the application
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8080))
    uvicorn.run("main:app", host=host, port=port, reload=True)
