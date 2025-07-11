"""
API endpoints for category-level demand forecasting.
Provides endpoints for category analysis, forecasting, and performance insights.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from fastapi import FastAPI, Query, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import logging
import asyncio

# Import services
from services.category_forecast_service import CategoryForecastService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Category-Level Demand Forecasting API",
    description="API for category-level demand forecasting and analysis",
    version="1.0.0",
)

# Initialize service
category_service = CategoryForecastService()


# Pydantic models for API requests and responses
class CategorySeasonalityRequest(BaseModel):
    """Request model for category seasonality analysis."""

    category_id: Optional[int] = None
    store_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    aggregation_level: str = Field(default="category", description="Aggregation level")

    @validator("aggregation_level")
    def validate_aggregation_level(cls, v):
        valid_levels = ["category", "subcategory", "brand"]
        if v not in valid_levels:
            raise ValueError(f"Aggregation level must be one of: {valid_levels}")
        return v

    @validator("start_date", "end_date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class CategoryForecastRequest(BaseModel):
    """Request model for category-level demand forecasting."""

    categories: List[int] = Field(..., description="List of category IDs to forecast")
    stores: List[int] = Field(..., description="List of store IDs to forecast")
    start_date: str = Field(..., description="Start date for forecast (YYYY-MM-DD)")
    periods: int = Field(
        default=30, ge=1, le=365, description="Number of days to forecast"
    )
    aggregation_level: str = Field(default="category", description="Aggregation level")
    model_type: str = Field(default="gradient_boost", description="Model type")
    include_confidence: bool = Field(
        default=True, description="Include confidence intervals"
    )

    @validator("categories")
    def validate_categories(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one category ID must be provided")
        if len(v) > 50:
            raise ValueError("Maximum 50 categories allowed per request")
        return v

    @validator("stores")
    def validate_stores(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one store ID must be provided")
        if len(v) > 100:
            raise ValueError("Maximum 100 stores allowed per request")
        return v

    @validator("aggregation_level")
    def validate_aggregation_level(cls, v):
        valid_levels = ["category", "subcategory", "brand"]
        if v not in valid_levels:
            raise ValueError(f"Aggregation level must be one of: {valid_levels}")
        return v

    @validator("model_type")
    def validate_model_type(cls, v):
        valid_types = ["gradient_boost", "random_forest", "linear"]
        if v not in valid_types:
            raise ValueError(f"Model type must be one of: {valid_types}")
        return v

    @validator("start_date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class CategoryPerformanceRequest(BaseModel):
    """Request model for category performance analysis."""

    category_id: Optional[int] = None
    store_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    aggregation_level: str = Field(default="category", description="Aggregation level")

    @validator("start_date", "end_date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class CategoryModelTrainingRequest(BaseModel):
    """Request model for training category forecasting models."""

    aggregation_level: str = Field(default="category", description="Aggregation level")
    model_type: str = Field(default="gradient_boost", description="Model type")
    category_id: Optional[int] = None
    store_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: Optional[str] = Field(
        default=None, description="Start date for training data"
    )
    end_date: Optional[str] = Field(
        default=None, description="End date for training data"
    )

    @validator("aggregation_level")
    def validate_aggregation_level(cls, v):
        valid_levels = ["category", "subcategory", "brand"]
        if v not in valid_levels:
            raise ValueError(f"Aggregation level must be one of: {valid_levels}")
        return v

    @validator("model_type")
    def validate_model_type(cls, v):
        valid_types = ["gradient_boost", "random_forest", "linear"]
        if v not in valid_types:
            raise ValueError(f"Model type must be one of: {valid_types}")
        return v


class CategoryHierarchyRequest(BaseModel):
    """Request model for category hierarchy information."""

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=100, description="Page size")


# API Endpoints
@app.get("/")
async def read_root():
    """Root endpoint providing API information."""
    return {
        "service": "Category-Level Demand Forecasting API",
        "version": "1.0.0",
        "description": "API for category-level demand forecasting and analysis",
        "endpoints": {
            "seasonality": "/category/seasonality/",
            "forecast": "/category/forecast/",
            "performance": "/category/performance/",
            "train": "/category/train/",
            "hierarchy": "/category/hierarchy/",
        },
    }


@app.post("/category/seasonality/")
async def analyze_category_seasonality(request: CategorySeasonalityRequest):
    """
    Analyze seasonality patterns for categories.

    This endpoint analyzes seasonal patterns, trends, and holiday impacts
    for product categories over a specified time period.
    """
    try:
        logger.info(f"Category seasonality analysis request: {request.dict()}")

        result = await category_service.analyze_category_seasonality(
            category_id=request.category_id,
            store_id=request.store_id,
            city_id=request.city_id,
            start_date=request.start_date,
            end_date=request.end_date,
            aggregation_level=request.aggregation_level,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "seasonality_analysis": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Category seasonality analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/category/forecast/")
async def forecast_category_demand(request: CategoryForecastRequest):
    """
    Generate category-level demand forecasts.

    This endpoint generates demand forecasts for specified categories and stores
    using hierarchical forecasting models.
    """
    try:
        logger.info(f"Category demand forecast request: {request.dict()}")

        result = await category_service.forecast_category_demand(
            categories=request.categories,
            stores=request.stores,
            start_date=request.start_date,
            periods=request.periods,
            aggregation_level=request.aggregation_level,
            model_type=request.model_type,
            include_confidence=request.include_confidence,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "forecast_results": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Category demand forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/category/performance/")
async def analyze_category_performance(request: CategoryPerformanceRequest):
    """
    Analyze category performance metrics and trends.

    This endpoint provides comprehensive performance analysis including
    sales trends, market share, volatility, and promotion effectiveness.
    """
    try:
        logger.info(f"Category performance analysis request: {request.dict()}")

        result = await category_service.analyze_category_performance(
            category_id=request.category_id,
            store_id=request.store_id,
            city_id=request.city_id,
            start_date=request.start_date,
            end_date=request.end_date,
            aggregation_level=request.aggregation_level,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "performance_analysis": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Category performance analysis error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Performance analysis failed: {str(e)}"
        )


@app.post("/category/train/")
async def train_category_model(
    request: CategoryModelTrainingRequest, background_tasks: BackgroundTasks
):
    """
    Train a category-level forecasting model.

    This endpoint trains machine learning models for category-level demand
    forecasting with support for different aggregation levels and model types.
    """
    try:
        logger.info(f"Category model training request: {request.dict()}")

        result = await category_service.train_category_model(
            aggregation_level=request.aggregation_level,
            model_type=request.model_type,
            category_id=request.category_id,
            store_id=request.store_id,
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
        logger.error(f"Category model training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/category/hierarchy/")
async def get_category_hierarchies(request: CategoryHierarchyRequest):
    """
    Get category hierarchy information.

    This endpoint returns information about the product category hierarchy
    including subcategories, brands, and product counts.
    """
    try:
        logger.info(f"Category hierarchy request: {request.dict()}")

        result = await category_service.get_category_hierarchies(
            limit=request.page_size, offset=(request.page - 1) * request.page_size
        )

        return {
            "status": "success",
            "hierarchies": result.get("data", []),
            "pagination": {
                "page": result.get("page", request.page),
                "page_size": result.get("page_size", request.page_size),
                "has_more": result.get("has_more", False),
            },
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Category hierarchy error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Hierarchy retrieval failed: {str(e)}"
        )


# Bulk operations endpoints
@app.post("/category/bulk-forecast/")
async def bulk_category_forecast(
    categories: List[int] = Query(..., description="Category IDs"),
    stores: List[int] = Query(..., description="Store IDs"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    periods: int = Query(30, description="Number of periods"),
    aggregation_level: str = Query("category", description="Aggregation level"),
    model_type: str = Query("gradient_boost", description="Model type"),
):
    """
    Bulk forecast for multiple categories and stores.

    This endpoint allows for efficient bulk forecasting operations
    for large numbers of categories and stores.
    """
    try:
        # Validate input sizes
        if len(categories) > 100:
            raise HTTPException(
                status_code=400, detail="Maximum 100 categories allowed"
            )
        if len(stores) > 200:
            raise HTTPException(status_code=400, detail="Maximum 200 stores allowed")

        result = await category_service.forecast_category_demand(
            categories=categories,
            stores=stores,
            start_date=start_date,
            periods=periods,
            aggregation_level=aggregation_level,
            model_type=model_type,
            include_confidence=False,  # Disable for bulk operations
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Simplified response for bulk operations
        return {
            "status": "success",
            "bulk_forecasts": result["forecasts"],
            "summary": result["forecast_summary"],
            "processing_info": {
                "categories_processed": len(categories),
                "stores_processed": len(stores),
                "total_forecasts": len(result["forecasts"]),
            },
        }

    except Exception as e:
        logger.error(f"Bulk category forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk forecast failed: {str(e)}")


# Analytics and insights endpoints
@app.get("/category/insights/")
async def get_category_insights(
    aggregation_level: str = Query("category", description="Aggregation level"),
    model_type: str = Query("gradient_boost", description="Model type"),
):
    """
    Get category insights and model performance metrics.

    This endpoint provides insights about category patterns, model performance,
    and forecasting accuracy.
    """
    try:
        # Get model
        model = category_service.get_or_create_model(aggregation_level, model_type)

        if not model.is_fitted:
            return {
                "status": "warning",
                "message": "Model not trained yet",
                "available_insights": [],
                "recommendation": "Train the model first using /category/train/ endpoint",
            }

        # Get model insights
        insights = model.get_category_insights()

        return {
            "status": "success",
            "model_insights": insights,
            "model_info": {
                "aggregation_level": aggregation_level,
                "model_type": model_type,
                "is_trained": model.is_fitted,
            },
        }

    except Exception as e:
        logger.error(f"Category insights error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Insights retrieval failed: {str(e)}"
        )


@app.get("/category/models/")
async def get_available_models():
    """Get information about available category forecasting models."""
    return {
        "aggregation_levels": {
            "category": {
                "description": "First-level category aggregation",
                "use_case": "High-level strategic planning and broad market analysis",
                "data_points": "Aggregated across all products in category",
            },
            "subcategory": {
                "description": "Second-level category aggregation",
                "use_case": "Mid-level planning and category management",
                "data_points": "Aggregated across products in subcategory",
            },
            "brand": {
                "description": "Brand-level aggregation",
                "use_case": "Brand performance analysis and competitive insights",
                "data_points": "Aggregated across all products of the brand",
            },
        },
        "model_types": {
            "gradient_boost": {
                "description": "Gradient Boosting - Best for complex patterns",
                "pros": [
                    "High accuracy",
                    "Handles non-linearities",
                    "Feature importance",
                ],
                "cons": ["Longer training time", "Risk of overfitting"],
                "recommended_for": "Large datasets with complex patterns",
            },
            "random_forest": {
                "description": "Random Forest - Robust and fast",
                "pros": [
                    "Good generalization",
                    "Fast prediction",
                    "Robust to outliers",
                ],
                "cons": ["Less interpretable", "Memory intensive"],
                "recommended_for": "Medium datasets, quick deployment",
            },
            "linear": {
                "description": "Linear Regression - Simple and interpretable",
                "pros": ["Fast training", "Interpretable", "Good baseline"],
                "cons": ["Assumes linearity", "Limited complexity"],
                "recommended_for": "Simple patterns, proof of concept",
            },
        },
        "features": {
            "seasonality_analysis": "Automatic detection of seasonal patterns",
            "hierarchical_forecasting": "Multi-level aggregation support",
            "cross_category_effects": "Category interaction modeling",
            "promotion_impact": "Promotion effectiveness analysis",
            "trend_decomposition": "Trend and seasonality separation",
        },
    }


@app.get("/category/health/")
async def health_check():
    """Health check endpoint for category forecasting service."""
    try:
        service_health = {
            "service_status": "healthy",
            "available_models": {},
            "cache_status": "active",
        }

        # Check available models
        for agg_level in ["category", "subcategory"]:
            for model_type in ["gradient_boost", "random_forest"]:
                model_key = f"{agg_level}_{model_type}"
                if model_key in category_service.models:
                    model = category_service.models[model_key]
                    service_health["available_models"][model_key] = {
                        "loaded": True,
                        "trained": model.is_fitted,
                        "categories_count": len(model.models) if model.is_fitted else 0,
                    }

        return service_health

    except Exception as e:
        return {"service_status": "unhealthy", "error": str(e)}


@app.get("/category/examples/")
async def get_usage_examples():
    """Get example API requests for category forecasting."""
    return {
        "examples": {
            "analyze_category_seasonality": {
                "endpoint": "/category/seasonality/",
                "method": "POST",
                "payload": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "aggregation_level": "category",
                },
                "description": "Analyze seasonal patterns across all categories",
            },
            "forecast_specific_categories": {
                "endpoint": "/category/forecast/",
                "method": "POST",
                "payload": {
                    "categories": [1, 2, 3],
                    "stores": [10, 20, 30],
                    "start_date": "2024-01-01",
                    "periods": 30,
                    "aggregation_level": "category",
                    "model_type": "gradient_boost",
                },
                "description": "Forecast demand for specific categories and stores",
            },
            "analyze_category_performance": {
                "endpoint": "/category/performance/",
                "method": "POST",
                "payload": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "aggregation_level": "category",
                },
                "description": "Analyze performance metrics for all categories",
            },
            "train_category_model": {
                "endpoint": "/category/train/",
                "method": "POST",
                "payload": {
                    "aggregation_level": "category",
                    "model_type": "gradient_boost",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                },
                "description": "Train a category forecasting model",
            },
            "bulk_category_forecast": {
                "endpoint": "/category/bulk-forecast/",
                "method": "POST",
                "query_params": {
                    "categories": "1,2,3,4,5",
                    "stores": "10,20,30,40,50",
                    "start_date": "2024-01-01",
                    "periods": 30,
                },
                "description": "Bulk forecast for multiple categories",
            },
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
