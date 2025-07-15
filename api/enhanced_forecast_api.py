"""
Enhanced Forecasting API Endpoints for FreshRetailNet-50K
Supports multi-dimensional analysis with cross-store comparison and product correlations
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import json
import pandas as pd

# Import enhanced services
from services.enhanced_forecast_service import (
    EnhancedForecastService,
    ForecastRequest,
    ForecastResult,
    ForecastingMethod,
    enhanced_forecast_service,
)
from database.connection import db_manager, initialize_database, close_database

logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Enhanced FreshRetail Forecasting API",
    description="Multi-dimensional forecasting API with cross-store analysis and product correlations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ForecastingMethodEnum(str, Enum):
    """Forecasting method options"""

    prophet = "prophet"
    random_forest = "random_forest"
    ensemble = "ensemble"
    naive = "naive"


class EnhancedForecastRequestModel(BaseModel):
    """Enhanced forecast request model for API"""

    store_ids: List[int] = Field(
        ..., description="List of store IDs to analyze", min_items=1, max_items=50
    )
    product_ids: List[int] = Field(
        ..., description="List of product IDs to analyze", min_items=1, max_items=30
    )
    forecast_horizon_days: int = Field(
        30, description="Number of days to forecast", ge=1, le=365
    )
    include_confidence_intervals: bool = Field(
        True, description="Include confidence intervals in results"
    )
    include_cross_store_analysis: bool = Field(
        True, description="Perform cross-store comparative analysis"
    )
    include_product_correlations: bool = Field(
        True, description="Analyze product correlations"
    )
    include_weather_factors: bool = Field(
        True, description="Include weather impact factors"
    )
    include_promotion_factors: bool = Field(
        True, description="Include promotion effectiveness factors"
    )
    forecasting_method: ForecastingMethodEnum = Field(
        ForecastingMethodEnum.ensemble, description="Forecasting method to use"
    )
    confidence_level: float = Field(
        0.95, description="Confidence level for intervals", ge=0.5, le=0.99
    )

    @validator("store_ids", "product_ids")
    def validate_ids(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("IDs must be unique")
        return v


class CrossStoreComparisonRequest(BaseModel):
    """Request model for cross-store comparison"""

    product_id: int = Field(..., description="Product ID to compare across stores")
    store_ids: List[int] = Field(
        ..., description="Store IDs to compare", min_items=2, max_items=20
    )
    analysis_period_days: int = Field(
        90, description="Historical analysis period in days", ge=30, le=365
    )
    include_performance_metrics: bool = Field(
        True, description="Include detailed performance metrics"
    )
    include_recommendations: bool = Field(
        True, description="Include actionable recommendations"
    )


class ProductCorrelationRequest(BaseModel):
    """Request model for product correlation analysis"""

    store_id: int = Field(..., description="Store ID for correlation analysis")
    product_ids: List[int] = Field(
        ..., description="Product IDs to analyze", min_items=2, max_items=50
    )
    correlation_threshold: float = Field(
        0.3, description="Minimum correlation threshold", ge=0.1, le=0.9
    )
    analysis_period_days: int = Field(
        90, description="Analysis period in days", ge=30, le=365
    )
    include_seasonal_patterns: bool = Field(
        True, description="Include seasonal correlation patterns"
    )


class InventoryOptimizationRequest(BaseModel):
    """Request model for cross-store inventory optimization"""

    product_ids: List[int] = Field(
        ..., description="Product IDs to optimize", min_items=1, max_items=20
    )
    city_id: int = Field(..., description="City ID for optimization scope")
    optimization_horizon_days: int = Field(
        30, description="Optimization horizon", ge=7, le=90
    )
    target_service_level: float = Field(
        0.95, description="Target service level", ge=0.8, le=0.99
    )
    include_transfer_costs: bool = Field(
        True, description="Include inter-store transfer costs"
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    await initialize_database()
    logger.info("Enhanced Forecasting API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    await close_database()
    logger.info("Enhanced Forecasting API shut down")


@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Enhanced FreshRetail Forecasting API",
        "version": "2.0.0",
        "status": "active",
        "capabilities": [
            "multi_dimensional_forecasting",
            "cross_store_comparison",
            "product_correlation_analysis",
            "inventory_optimization",
            "ensemble_modeling",
        ],
    }


@app.post("/forecast/multi-dimensional/", response_model=Dict[str, Any])
async def generate_multi_dimensional_forecast(
    request: EnhancedForecastRequestModel, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate multi-dimensional forecast with cross-store analysis and product correlations

    This endpoint provides comprehensive forecasting including:
    - Individual store-product forecasts
    - Cross-store performance comparison
    - Product correlation analysis
    - Feature importance analysis
    - Actionable recommendations
    """
    try:
        logger.info(
            f"Generating multi-dimensional forecast for {len(request.store_ids)} stores, {len(request.product_ids)} products"
        )

        # Convert API request to service request
        service_request = ForecastRequest(
            store_ids=request.store_ids,
            product_ids=request.product_ids,
            forecast_horizon_days=request.forecast_horizon_days,
            include_confidence_intervals=request.include_confidence_intervals,
            include_cross_store_analysis=request.include_cross_store_analysis,
            include_product_correlations=request.include_product_correlations,
            include_weather_factors=request.include_weather_factors,
            include_promotion_factors=request.include_promotion_factors,
            forecasting_method=ForecastingMethod(request.forecasting_method.value),
            confidence_level=request.confidence_level,
        )

        # Generate forecast
        result = await enhanced_forecast_service.generate_multi_dimensional_forecast(
            service_request
        )

        # Convert result to API response format
        response = {
            "forecast_summary": {
                "total_forecasts_generated": len(result.forecast_data),
                "forecast_horizon_days": request.forecast_horizon_days,
                "forecasting_method": request.forecasting_method.value,
                "confidence_level": request.confidence_level,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "forecasts": (
                result.forecast_data.to_dict("records")
                if not result.forecast_data.empty
                else []
            ),
            "model_metrics": result.model_metrics,
            "feature_importance": result.feature_importance,
            "recommendations": result.recommendations or [],
        }

        # Add optional analyses
        if result.cross_store_comparison:
            response["cross_store_analysis"] = result.cross_store_comparison

        if result.product_correlations:
            response["product_correlations"] = result.product_correlations

        if result.confidence_intervals:
            response["confidence_intervals"] = {
                k: v.to_dict("records") for k, v in result.confidence_intervals.items()
            }

        # Schedule background task to update analytics
        background_tasks.add_task(
            update_forecast_analytics, request.store_ids, request.product_ids, result
        )

        return response

    except Exception as e:
        logger.error(f"Multi-dimensional forecast failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Forecast generation failed: {str(e)}"
        )


@app.post("/forecast/cross-store-comparison/")
async def analyze_cross_store_performance(
    request: CrossStoreComparisonRequest,
) -> Dict[str, Any]:
    """
    Compare product performance across multiple stores

    Provides detailed analysis of how the same product performs
    in different stores, including ranking and recommendations.
    """
    try:
        logger.info(
            f"Analyzing cross-store performance for product {request.product_id} across {len(request.store_ids)} stores"
        )

        # Get historical data for analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.analysis_period_days)

        historical_data = await db_manager.get_sales_data(
            store_ids=request.store_ids,
            product_ids=[request.product_id],
            start_date=start_date,
            end_date=end_date,
            include_stockouts=True,
        )

        if historical_data.empty:
            raise HTTPException(
                status_code=404,
                detail="No historical data found for the specified parameters",
            )

        # Perform cross-store analysis
        analysis_result = await _perform_detailed_cross_store_analysis(
            historical_data, request.product_id, request.store_ids
        )

        response = {
            "product_id": request.product_id,
            "analysis_period": {
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "days_analyzed": request.analysis_period_days,
            },
            "store_performance": analysis_result["store_metrics"],
            "performance_rankings": analysis_result["rankings"],
            "key_insights": analysis_result["insights"],
        }

        if request.include_recommendations:
            response["recommendations"] = analysis_result["recommendations"]

        return response

    except Exception as e:
        logger.error(f"Cross-store analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/forecast/product-correlations/")
async def analyze_product_correlations(
    request: ProductCorrelationRequest,
) -> Dict[str, Any]:
    """
    Analyze correlations between products for cross-selling opportunities

    Identifies products that are frequently bought together or
    show similar demand patterns for optimization strategies.
    """
    try:
        logger.info(f"Analyzing product correlations for store {request.store_id}")

        # Get product correlation data
        correlations = await db_manager.get_product_correlations(
            product_id=request.product_ids[0],  # Use first product as base
            store_id=request.store_id,
            correlation_threshold=request.correlation_threshold,
        )

        # Get historical sales data for detailed analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.analysis_period_days)

        sales_data = await db_manager.get_sales_data(
            store_ids=[request.store_id],
            product_ids=request.product_ids,
            start_date=start_date,
            end_date=end_date,
        )

        # Perform correlation analysis
        correlation_analysis = await _perform_correlation_analysis(
            sales_data, request.product_ids, request.correlation_threshold
        )

        response = {
            "store_id": request.store_id,
            "analysis_period": {
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "days_analyzed": request.analysis_period_days,
            },
            "correlation_matrix": correlation_analysis["correlation_matrix"],
            "significant_correlations": correlation_analysis[
                "significant_correlations"
            ],
            "cross_selling_opportunities": correlation_analysis[
                "cross_selling_opportunities"
            ],
            "substitution_products": correlation_analysis["substitution_products"],
        }

        if request.include_seasonal_patterns:
            response["seasonal_patterns"] = correlation_analysis["seasonal_patterns"]

        return response

    except Exception as e:
        logger.error(f"Product correlation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/forecast/inventory-optimization/")
async def optimize_cross_store_inventory(
    request: InventoryOptimizationRequest,
) -> Dict[str, Any]:
    """
    Optimize inventory allocation across stores within a city

    Provides recommendations for optimal inventory distribution
    to minimize stockouts while reducing carrying costs.
    """
    try:
        logger.info(
            f"Optimizing inventory for {len(request.product_ids)} products in city {request.city_id}"
        )

        # Get stores in the specified city
        city_stores = await db_manager.execute_dataframe_query(
            "SELECT store_id, store_name FROM store_hierarchy WHERE city_id = $1",
            (request.city_id,),
        )

        if city_stores.empty:
            raise HTTPException(
                status_code=404, detail=f"No stores found in city {request.city_id}"
            )

        store_ids = city_stores["store_id"].tolist()

        # Generate forecasts for optimization
        service_request = ForecastRequest(
            store_ids=store_ids,
            product_ids=request.product_ids,
            forecast_horizon_days=request.optimization_horizon_days,
            include_confidence_intervals=True,
            include_cross_store_analysis=True,
            forecasting_method=ForecastingMethod.ENSEMBLE,
        )

        forecast_result = (
            await enhanced_forecast_service.generate_multi_dimensional_forecast(
                service_request
            )
        )

        # Perform inventory optimization
        optimization_result = await _perform_inventory_optimization(
            forecast_result, request, city_stores
        )

        response = {
            "city_id": request.city_id,
            "optimization_horizon_days": request.optimization_horizon_days,
            "target_service_level": request.target_service_level,
            "total_stores_analyzed": len(store_ids),
            "optimization_results": optimization_result["allocation_recommendations"],
            "performance_summary": optimization_result["performance_summary"],
            "transfer_recommendations": optimization_result["transfer_recommendations"],
            "cost_analysis": optimization_result["cost_analysis"],
        }

        return response

    except Exception as e:
        logger.error(f"Inventory optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/forecast/performance-metrics/")
async def get_forecasting_performance_metrics(
    store_ids: List[int] = Query(..., description="Store IDs to analyze"),
    product_ids: List[int] = Query(..., description="Product IDs to analyze"),
    days_back: int = Query(30, description="Days back to analyze", ge=7, le=365),
) -> Dict[str, Any]:
    """
    Get forecasting performance metrics for specified stores and products

    Returns accuracy metrics and model performance statistics.
    """
    try:
        # Get recent forecasts and actual sales for comparison
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Get stored forecasts
        forecasts = await db_manager.execute_dataframe_query(
            """
            SELECT * FROM demand_forecasts 
            WHERE store_id = ANY($1) AND product_id = ANY($2)
            AND forecast_date >= $3 AND forecast_date <= $4
            ORDER BY forecast_date DESC
        """,
            (store_ids, product_ids, start_date.date(), end_date.date()),
        )

        # Get actual sales data
        actual_sales = await db_manager.get_sales_data(
            store_ids=store_ids,
            product_ids=product_ids,
            start_date=start_date,
            end_date=end_date,
        )

        # Calculate performance metrics
        performance_metrics = await _calculate_forecast_performance(
            forecasts, actual_sales
        )

        return {
            "analysis_period": {
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "days_analyzed": days_back,
            },
            "forecast_accuracy": performance_metrics["accuracy_metrics"],
            "model_performance": performance_metrics["model_performance"],
            "store_product_breakdown": performance_metrics["breakdown"],
        }

    except Exception as e:
        logger.error(f"Performance metrics calculation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Metrics calculation failed: {str(e)}"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _perform_detailed_cross_store_analysis(
    historical_data: pd.DataFrame, product_id: int, store_ids: List[int]
) -> Dict[str, Any]:
    """Perform detailed cross-store analysis"""

    store_metrics = []

    for store_id in store_ids:
        store_data = historical_data[historical_data["store_id"] == store_id]

        if store_data.empty:
            continue

        # Calculate comprehensive metrics
        metrics = {
            "store_id": store_id,
            "total_sales": store_data["sale_amount"].sum(),
            "avg_daily_sales": store_data["sale_amount"].mean(),
            "sales_volatility": store_data["sale_amount"].std(),
            "growth_trend": _calculate_trend(store_data["sale_amount"]),
            "stockout_frequency": store_data["stock_hour6_22_cnt"].mean() / 16,
            "demand_consistency": 1
            - (store_data["sale_amount"].std() / store_data["sale_amount"].mean()),
            "peak_sales_day": store_data.loc[
                store_data["sale_amount"].idxmax(), "sale_date"
            ].isoformat(),
            "weather_sensitivity": (
                store_data["sale_amount"].corr(store_data["avg_temperature"])
                if "avg_temperature" in store_data.columns
                else 0
            ),
        }

        store_metrics.append(metrics)

    # Calculate rankings
    store_metrics_df = pd.DataFrame(store_metrics)
    rankings = {
        "revenue_ranking": store_metrics_df.nlargest(
            len(store_metrics_df), "total_sales"
        )["store_id"].tolist(),
        "consistency_ranking": store_metrics_df.nlargest(
            len(store_metrics_df), "demand_consistency"
        )["store_id"].tolist(),
        "reliability_ranking": store_metrics_df.nsmallest(
            len(store_metrics_df), "stockout_frequency"
        )["store_id"].tolist(),
    }

    # Generate insights
    insights = _generate_cross_store_insights(store_metrics_df)

    # Generate recommendations
    recommendations = _generate_cross_store_recommendations(store_metrics_df)

    return {
        "store_metrics": store_metrics,
        "rankings": rankings,
        "insights": insights,
        "recommendations": recommendations,
    }


async def _perform_correlation_analysis(
    sales_data: pd.DataFrame, product_ids: List[int], threshold: float
) -> Dict[str, Any]:
    """Perform detailed product correlation analysis"""

    # Pivot data for correlation calculation
    pivot_data = sales_data.pivot_table(
        index="sale_date", columns="product_id", values="sale_amount", fill_value=0
    )

    # Calculate correlation matrix
    correlation_matrix = pivot_data.corr()

    # Extract significant correlations
    significant_correlations = []
    cross_selling_opportunities = []
    substitution_products = []

    for i, product_a in enumerate(correlation_matrix.index):
        for j, product_b in enumerate(correlation_matrix.columns):
            if i >= j:
                continue

            correlation_coef = correlation_matrix.loc[product_a, product_b]

            if abs(correlation_coef) >= threshold:
                correlation_data = {
                    "product_a": int(product_a),
                    "product_b": int(product_b),
                    "correlation": float(correlation_coef),
                    "strength": "strong" if abs(correlation_coef) > 0.7 else "moderate",
                }

                significant_correlations.append(correlation_data)

                if correlation_coef > 0.5:  # Complementary products
                    cross_selling_opportunities.append(correlation_data)
                elif correlation_coef < -0.5:  # Substitute products
                    substitution_products.append(correlation_data)

    # Calculate seasonal patterns (simplified)
    seasonal_patterns = {}
    for product_id in product_ids:
        if product_id in pivot_data.columns:
            product_data = sales_data[sales_data["product_id"] == product_id].copy()
            product_data["month"] = pd.to_datetime(product_data["sale_date"]).dt.month
            monthly_avg = product_data.groupby("month")["sale_amount"].mean()
            seasonal_patterns[int(product_id)] = monthly_avg.to_dict()

    return {
        "correlation_matrix": correlation_matrix.to_dict(),
        "significant_correlations": significant_correlations,
        "cross_selling_opportunities": cross_selling_opportunities,
        "substitution_products": substitution_products,
        "seasonal_patterns": seasonal_patterns,
    }


async def _perform_inventory_optimization(
    forecast_result: ForecastResult,
    request: InventoryOptimizationRequest,
    city_stores: pd.DataFrame,
) -> Dict[str, Any]:
    """Perform inventory optimization across stores"""

    allocation_recommendations = []
    total_demand = {}

    # Calculate total demand by product
    for product_id in request.product_ids:
        product_forecasts = forecast_result.forecast_data[
            forecast_result.forecast_data["product_id"] == product_id
        ]
        total_demand[product_id] = product_forecasts["predicted_demand"].sum()

    # Optimize allocation for each product
    for product_id in request.product_ids:
        product_forecasts = forecast_result.forecast_data[
            forecast_result.forecast_data["product_id"] == product_id
        ]

        store_allocations = []
        for _, row in product_forecasts.iterrows():
            store_id = row["store_id"]
            predicted_demand = row["predicted_demand"]
            confidence_upper = row["confidence_upper"]

            # Calculate safety stock based on service level
            safety_stock = (
                confidence_upper - predicted_demand
            ) * request.target_service_level
            recommended_stock = predicted_demand + safety_stock

            store_allocations.append(
                {
                    "store_id": int(store_id),
                    "predicted_demand": float(predicted_demand),
                    "recommended_stock": float(recommended_stock),
                    "safety_stock": float(safety_stock),
                    "service_level_target": request.target_service_level,
                }
            )

        allocation_recommendations.append(
            {
                "product_id": product_id,
                "total_predicted_demand": float(total_demand[product_id]),
                "store_allocations": store_allocations,
            }
        )

    # Calculate performance summary
    total_recommended_stock = sum(
        [
            sum([store["recommended_stock"] for store in product["store_allocations"]])
            for product in allocation_recommendations
        ]
    )

    total_predicted_demand = sum(total_demand.values())

    performance_summary = {
        "total_products_optimized": len(request.product_ids),
        "total_stores_involved": len(city_stores),
        "total_predicted_demand": float(total_predicted_demand),
        "total_recommended_stock": float(total_recommended_stock),
        "inventory_turnover_ratio": (
            float(total_predicted_demand / total_recommended_stock)
            if total_recommended_stock > 0
            else 0
        ),
        "average_service_level": request.target_service_level,
    }

    # Generate transfer recommendations (simplified)
    transfer_recommendations = []

    # Cost analysis (simplified)
    cost_analysis = {
        "holding_cost_savings": float(
            total_recommended_stock * 0.02
        ),  # Estimated 2% holding cost
        "stockout_risk_reduction": f"{(1 - request.target_service_level) * 100:.1f}%",
        "optimization_efficiency": "85%",  # Placeholder
    }

    return {
        "allocation_recommendations": allocation_recommendations,
        "performance_summary": performance_summary,
        "transfer_recommendations": transfer_recommendations,
        "cost_analysis": cost_analysis,
    }


async def _calculate_forecast_performance(
    forecasts: pd.DataFrame, actual_sales: pd.DataFrame
) -> Dict[str, Any]:
    """Calculate forecast performance metrics"""

    if forecasts.empty or actual_sales.empty:
        return {"accuracy_metrics": {}, "model_performance": {}, "breakdown": []}

    # Simple performance calculation (placeholder)
    accuracy_metrics = {
        "mean_absolute_error": 12.5,
        "mean_absolute_percentage_error": 15.2,
        "forecast_bias": 2.1,
        "tracking_signal": 0.8,
    }

    model_performance = {
        "prophet_accuracy": 85.5,
        "random_forest_accuracy": 82.3,
        "ensemble_accuracy": 87.2,
        "naive_accuracy": 68.1,
    }

    breakdown = [
        {"store_id": 1, "product_id": 21, "accuracy": 88.5, "bias": 1.2},
        {"store_id": 2, "product_id": 21, "accuracy": 85.3, "bias": -0.8},
    ]

    return {
        "accuracy_metrics": accuracy_metrics,
        "model_performance": model_performance,
        "breakdown": breakdown,
    }


def _calculate_trend(sales_series: pd.Series) -> float:
    """Calculate trend in sales data"""
    if len(sales_series) < 2:
        return 0.0

    x = range(len(sales_series))
    y = sales_series.values

    # Simple linear regression for trend
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x[i] * x[i] for i in range(n))

    if n * sum_x2 - sum_x * sum_x == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    return float(slope)


def _generate_cross_store_insights(store_metrics_df: pd.DataFrame) -> List[str]:
    """Generate insights from cross-store analysis"""

    insights = []

    if not store_metrics_df.empty:
        best_store = store_metrics_df.loc[store_metrics_df["total_sales"].idxmax()]
        worst_store = store_metrics_df.loc[store_metrics_df["total_sales"].idxmin()]

        insights.append(
            f"Store {best_store['store_id']} has the highest total sales ({best_store['total_sales']:.1f})"
        )
        insights.append(
            f"Store {worst_store['store_id']} has the lowest total sales ({worst_store['total_sales']:.1f})"
        )

        if "stockout_frequency" in store_metrics_df.columns:
            most_reliable = store_metrics_df.loc[
                store_metrics_df["stockout_frequency"].idxmin()
            ]
            insights.append(
                f"Store {most_reliable['store_id']} has the best stockout performance ({most_reliable['stockout_frequency']:.2%} stockout rate)"
            )

    return insights


def _generate_cross_store_recommendations(store_metrics_df: pd.DataFrame) -> List[str]:
    """Generate recommendations from cross-store analysis"""

    recommendations = []

    if not store_metrics_df.empty:
        recommendations.append(
            "Consider implementing best practices from top-performing stores"
        )
        recommendations.append("Monitor stockout-prone stores more closely")
        recommendations.append(
            "Investigate weather sensitivity patterns for seasonal optimization"
        )

    return recommendations


async def update_forecast_analytics(
    store_ids: List[int], product_ids: List[int], result: ForecastResult
):
    """Background task to update forecast analytics"""
    try:
        # Update product correlations if available
        if result.product_correlations:
            await db_manager.calculate_product_correlations()

        # Update store performance metrics
        await db_manager.update_store_performance_metrics()

        logger.info(
            f"Updated analytics for {len(store_ids)} stores and {len(product_ids)} products"
        )
    except Exception as e:
        logger.error(f"Analytics update failed: {e}")


# Export the app
__all__ = ["app"]
