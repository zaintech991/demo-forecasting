"""
API endpoints for weather-sensitive demand modeling.
Provides endpoints for weather impact analysis, forecasting, and sensitivity analysis.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from fastapi import FastAPI, Query, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import logging
import asyncio

# Import services
from services.weather_demand_service import WeatherDemandService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Weather-Sensitive Demand API",
    description="API for weather-sensitive demand modeling and forecasting",
    version="1.0.0",
)

# Initialize service
weather_service = WeatherDemandService()


# Pydantic models for API requests and responses
class WeatherSensitivityRequest(BaseModel):
    """Request model for weather sensitivity analysis."""

    store_id: Optional[int] = None
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")

    @validator("start_date", "end_date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class WeatherForecastRequest(BaseModel):
    """Request model for weather-sensitive demand forecasting."""

    store_id: Optional[int] = None
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: str = Field(..., description="Start date for forecast (YYYY-MM-DD)")
    periods: int = Field(
        default=30, ge=1, le=365, description="Number of days to forecast"
    )
    weather_scenarios: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Weather scenarios to test (list of dicts with weather variables)",
    )

    @validator("start_date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class WeatherImpactRequest(BaseModel):
    """Request model for weather impact analysis."""

    weather_variable: str = Field(..., description="Weather variable to analyze")
    impact_range_min: float = Field(..., description="Minimum value for impact range")
    impact_range_max: float = Field(..., description="Maximum value for impact range")
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    city_id: Optional[int] = None
    reference_date: Optional[str] = Field(
        default=None, description="Reference date (YYYY-MM-DD)"
    )

    @validator("weather_variable")
    def validate_weather_variable(cls, v):
        valid_variables = [
            "avg_temperature",
            "avg_humidity",
            "precpt",
            "avg_wind_level",
        ]
        if v not in valid_variables:
            raise ValueError(f"Weather variable must be one of: {valid_variables}")
        return v

    @validator("reference_date")
    def validate_date_format(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class ModelTrainingRequest(BaseModel):
    """Request model for training weather demand models."""

    model_type: str = Field(
        default="gradient_boost", description="Type of model to train"
    )
    store_id: Optional[int] = None
    category_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: Optional[str] = Field(
        default=None, description="Start date for training data"
    )
    end_date: Optional[str] = Field(
        default=None, description="End date for training data"
    )

    @validator("model_type")
    def validate_model_type(cls, v):
        valid_types = ["gradient_boost", "random_forest", "elastic_net", "ensemble"]
        if v not in valid_types:
            raise ValueError(f"Model type must be one of: {valid_types}")
        return v


class WeatherCorrelationRequest(BaseModel):
    """Request model for weather correlation analysis."""

    city_id: Optional[int] = None
    category_id: Optional[int] = None
    start_date: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=100, description="Page size")


# API Endpoints
@app.get("/")
async def read_root():
    """Root endpoint providing API information."""
    return {
        "service": "Weather-Sensitive Demand API",
        "version": "1.0.0",
        "description": "API for weather-sensitive demand modeling and forecasting",
        "endpoints": {
            "analysis": "/weather/analyze/",
            "forecast": "/weather/forecast/",
            "impact": "/weather/impact/",
            "train": "/weather/train/",
            "correlations": "/weather/correlations/",
        },
    }


@app.post("/weather/analyze/")
async def analyze_weather_sensitivity(request: WeatherSensitivityRequest):
    """
    Analyze weather sensitivity for demand patterns.

    This endpoint analyzes how weather conditions affect demand for specified
    products, categories, stores, or cities over a given time period.
    """
    try:
        logger.info(f"Weather sensitivity analysis request: {request.dict()}")

        result = await weather_service.analyze_weather_sensitivity(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=request.city_id,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "analysis_results": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Weather sensitivity analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/weather/forecast/")
async def forecast_weather_demand(request: WeatherForecastRequest):
    """
    Generate weather-sensitive demand forecasts.

    This endpoint generates demand forecasts that account for weather conditions
    and can test multiple weather scenarios.
    """
    try:
        logger.info(f"Weather demand forecast request: {request.dict()}")

        result = await weather_service.forecast_weather_demand(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=request.city_id,
            start_date=request.start_date,
            periods=request.periods,
            weather_scenarios=request.weather_scenarios,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "forecast_results": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Weather demand forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/weather/impact/")
async def analyze_weather_impact(request: WeatherImpactRequest):
    """
    Analyze the impact of weather variable changes on demand.

    This endpoint analyzes how changes in specific weather variables
    (temperature, humidity, precipitation, wind) affect demand levels.
    """
    try:
        logger.info(f"Weather impact analysis request: {request.dict()}")

        impact_range = (request.impact_range_min, request.impact_range_max)

        result = await weather_service.analyze_weather_impact(
            weather_variable=request.weather_variable,
            impact_range=impact_range,
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=request.city_id,
            reference_date=request.reference_date,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "impact_analysis": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Weather impact analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Impact analysis failed: {str(e)}")


@app.post("/weather/train/")
async def train_weather_model(
    request: ModelTrainingRequest, background_tasks: BackgroundTasks
):
    """
    Train a weather-sensitive demand model.

    This endpoint trains a machine learning model to predict demand based on
    weather conditions. Training can be run as a background task for large datasets.
    """
    try:
        logger.info(f"Weather model training request: {request.dict()}")

        # For small datasets, train synchronously
        # For large datasets, you might want to run this as a background task
        result = await weather_service.train_weather_model(
            model_type=request.model_type,
            store_id=request.store_id,
            category_id=request.category_id,
            city_id=request.city_id,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "training_results": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Weather model training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/weather/correlations/")
async def get_weather_correlations(request: WeatherCorrelationRequest):
    """
    Get weather-sales correlations for different segments.

    This endpoint returns correlation coefficients between weather variables
    and sales for different cities and product categories.
    """
    try:
        logger.info(f"Weather correlations request: {request.dict()}")

        result = await weather_service.get_weather_correlations(
            city_id=request.city_id,
            category_id=request.category_id,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.page_size,
            offset=(request.page - 1) * request.page_size,
        )

        return {
            "status": "success",
            "correlations": result.get("data", []),
            "pagination": {
                "page": result.get("page", request.page),
                "page_size": result.get("page_size", request.page_size),
                "has_more": result.get("has_more", False),
            },
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Weather correlations error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Correlations analysis failed: {str(e)}"
        )


# Additional utility endpoints
@app.get("/weather/models/")
async def get_available_models():
    """Get information about available weather demand models."""
    return {
        "available_models": {
            "gradient_boost": {
                "description": "Gradient Boosting Regressor - Good for complex non-linear patterns",
                "pros": [
                    "High accuracy",
                    "Handles non-linearity well",
                    "Feature importance",
                ],
                "cons": ["Can overfit", "Slower training"],
            },
            "random_forest": {
                "description": "Random Forest Regressor - Robust ensemble method",
                "pros": [
                    "Robust to outliers",
                    "Good generalization",
                    "Fast prediction",
                ],
                "cons": ["Less interpretable", "Memory intensive"],
            },
            "elastic_net": {
                "description": "Elastic Net Regression - Linear model with regularization",
                "pros": ["Interpretable", "Fast training", "Handles multicollinearity"],
                "cons": ["Assumes linearity", "May underfit complex patterns"],
            },
            "ensemble": {
                "description": "Ensemble of multiple models - Best overall performance",
                "pros": [
                    "Highest accuracy",
                    "Robust predictions",
                    "Combines strengths",
                ],
                "cons": ["Slower prediction", "More complex"],
            },
        },
        "weather_variables": {
            "avg_temperature": {
                "description": "Average temperature in Celsius",
                "typical_range": "0-40°C",
                "impact": "Non-linear relationship with demand",
            },
            "avg_humidity": {
                "description": "Average humidity percentage",
                "typical_range": "20-100%",
                "impact": "Affects comfort and shopping behavior",
            },
            "precpt": {
                "description": "Precipitation in millimeters",
                "typical_range": "0-50mm",
                "impact": "Light rain may increase indoor shopping, heavy rain decreases it",
            },
            "avg_wind_level": {
                "description": "Average wind level",
                "typical_range": "0-20",
                "impact": "High wind generally decreases shopping activity",
            },
        },
    }


@app.get("/weather/health/")
async def health_check():
    """Health check endpoint for weather demand service."""
    try:
        # Check if service can access the model
        service_health = {
            "service_status": "healthy",
            "model_loaded": weather_service.model is not None,
            "model_trained": (
                weather_service.model.is_fitted if weather_service.model else False
            ),
        }

        if weather_service.model:
            service_health["model_type"] = weather_service.model.model_type

        return service_health

    except Exception as e:
        return {"service_status": "unhealthy", "error": str(e)}


# Example usage endpoints
@app.get("/weather/examples/")
async def get_usage_examples():
    """Get example API requests for weather demand modeling."""
    return {
        "examples": {
            "analyze_temperature_sensitivity": {
                "endpoint": "/weather/analyze/",
                "method": "POST",
                "payload": {
                    "city_id": 1,
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                },
                "description": "Analyze how temperature affects demand in city 1",
            },
            "forecast_with_weather_scenarios": {
                "endpoint": "/weather/forecast/",
                "method": "POST",
                "payload": {
                    "store_id": 1,
                    "start_date": "2024-01-01",
                    "periods": 30,
                    "weather_scenarios": [
                        {"avg_temperature": 25, "avg_humidity": 60},
                        {"avg_temperature": 15, "precpt": 10},
                    ],
                },
                "description": "Forecast demand for store 1 with different weather scenarios",
            },
            "analyze_temperature_impact": {
                "endpoint": "/weather/impact/",
                "method": "POST",
                "payload": {
                    "weather_variable": "avg_temperature",
                    "impact_range_min": 0,
                    "impact_range_max": 40,
                    "category_id": 5,
                },
                "description": "Analyze temperature impact on category 5 demand",
            },
            "train_ensemble_model": {
                "endpoint": "/weather/train/",
                "method": "POST",
                "payload": {
                    "model_type": "ensemble",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                },
                "description": "Train an ensemble model with one year of data",
            },
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
