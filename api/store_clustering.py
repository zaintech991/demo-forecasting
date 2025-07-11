"""
API endpoints for store clustering and behavior segmentation.
Provides endpoints for store clustering analysis, insights, and recommendations.
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
from services.store_clustering_service import StoreClusteringService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Store Clustering & Behavior Segmentation API",
    description="API for store clustering and behavior segmentation analysis",
    version="1.0.0",
)

# Initialize service
clustering_service = StoreClusteringService()


# Pydantic models for API requests and responses
class StoreClusteringRequest(BaseModel):
    """Request model for store clustering analysis."""

    clustering_method: str = Field(default="kmeans", description="Clustering algorithm")
    n_clusters: int = Field(default=5, ge=2, le=20, description="Number of clusters")
    store_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    auto_optimize: bool = Field(default=True, description="Auto-optimize cluster count")

    @validator("clustering_method")
    def validate_clustering_method(cls, v):
        valid_methods = ["kmeans", "dbscan", "hierarchical"]
        if v not in valid_methods:
            raise ValueError(f"Clustering method must be one of: {valid_methods}")
        return v

    @validator("start_date", "end_date")
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class StorePredictionRequest(BaseModel):
    """Request model for store cluster prediction."""

    store_ids: List[int] = Field(..., description="List of store IDs to predict")
    clustering_method: str = Field(default="kmeans", description="Clustering algorithm")
    n_clusters: int = Field(default=5, ge=2, le=20, description="Number of clusters")

    @validator("store_ids")
    def validate_store_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one store ID must be provided")
        if len(v) > 100:
            raise ValueError("Maximum 100 stores allowed per request")
        return v

    @validator("clustering_method")
    def validate_clustering_method(cls, v):
        valid_methods = ["kmeans", "dbscan", "hierarchical"]
        if v not in valid_methods:
            raise ValueError(f"Clustering method must be one of: {valid_methods}")
        return v


class StoreBehaviorInsightsRequest(BaseModel):
    """Request model for store behavior insights."""

    store_id: Optional[int] = None
    cluster_id: Optional[int] = None
    city_id: Optional[int] = None
    clustering_method: str = Field(default="kmeans", description="Clustering algorithm")
    n_clusters: int = Field(default=5, ge=2, le=20, description="Number of clusters")

    @validator("clustering_method")
    def validate_clustering_method(cls, v):
        valid_methods = ["kmeans", "dbscan", "hierarchical"]
        if v not in valid_methods:
            raise ValueError(f"Clustering method must be one of: {valid_methods}")
        return v


class ClusteringTrainingRequest(BaseModel):
    """Request model for training clustering models."""

    clustering_method: str = Field(default="kmeans", description="Clustering algorithm")
    n_clusters: int = Field(default=5, ge=2, le=20, description="Number of clusters")
    store_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: Optional[str] = Field(
        default=None, description="Start date for training data"
    )
    end_date: Optional[str] = Field(
        default=None, description="End date for training data"
    )
    auto_optimize: bool = Field(default=True, description="Auto-optimize cluster count")

    @validator("clustering_method")
    def validate_clustering_method(cls, v):
        valid_methods = ["kmeans", "dbscan", "hierarchical"]
        if v not in valid_methods:
            raise ValueError(f"Clustering method must be one of: {valid_methods}")
        return v


class ClusterComparisonRequest(BaseModel):
    """Request model for clustering comparison."""

    clustering_method: str = Field(default="kmeans", description="Clustering algorithm")
    n_clusters_list: List[int] = Field(
        default=[3, 4, 5, 6, 7], description="List of cluster counts to compare"
    )
    start_date: Optional[str] = Field(default=None, description="Start date")
    end_date: Optional[str] = Field(default=None, description="End date")

    @validator("n_clusters_list")
    def validate_cluster_list(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one cluster count must be provided")
        if any(n < 2 or n > 20 for n in v):
            raise ValueError("All cluster counts must be between 2 and 20")
        return v


class StoreSegmentsRequest(BaseModel):
    """Request model for store segments."""

    city_id: Optional[int] = None
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=100, description="Page size")


# API Endpoints
@app.get("/")
async def read_root():
    """Root endpoint providing API information."""
    return {
        "service": "Store Clustering & Behavior Segmentation API",
        "version": "1.0.0",
        "description": "API for store clustering and behavior segmentation analysis",
        "endpoints": {
            "clustering": "/stores/cluster/",
            "prediction": "/stores/predict/",
            "insights": "/stores/insights/",
            "training": "/stores/train/",
            "comparison": "/stores/compare/",
            "segments": "/stores/segments/",
        },
    }


@app.post("/stores/cluster/")
async def analyze_store_clustering(request: StoreClusteringRequest):
    """
    Perform store clustering analysis.

    This endpoint analyzes store behavior patterns and groups stores
    into clusters based on their operational and performance characteristics.
    """
    try:
        logger.info(f"Store clustering analysis request: {request.dict()}")

        result = await clustering_service.analyze_store_clustering(
            clustering_method=request.clustering_method,
            n_clusters=request.n_clusters,
            store_id=request.store_id,
            city_id=request.city_id,
            start_date=request.start_date,
            end_date=request.end_date,
            auto_optimize=request.auto_optimize,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "clustering_analysis": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Store clustering analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/stores/predict/")
async def predict_store_clusters(request: StorePredictionRequest):
    """
    Predict cluster assignments for stores.

    This endpoint predicts which cluster stores belong to based on
    a trained clustering model.
    """
    try:
        logger.info(f"Store cluster prediction request: {request.dict()}")

        result = await clustering_service.predict_store_clusters(
            store_ids=request.store_ids,
            clustering_method=request.clustering_method,
            n_clusters=request.n_clusters,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "prediction_results": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Store cluster prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/stores/insights/")
async def get_store_behavior_insights(request: StoreBehaviorInsightsRequest):
    """
    Get behavioral insights for stores or clusters.

    This endpoint provides detailed behavioral analysis, performance metrics,
    and actionable recommendations for stores or store clusters.
    """
    try:
        logger.info(f"Store behavior insights request: {request.dict()}")

        result = await clustering_service.get_store_behavior_insights(
            store_id=request.store_id,
            cluster_id=request.cluster_id,
            city_id=request.city_id,
            clustering_method=request.clustering_method,
            n_clusters=request.n_clusters,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "behavior_insights": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Store behavior insights error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Insights generation failed: {str(e)}"
        )


@app.post("/stores/train/")
async def train_clustering_model(
    request: ClusteringTrainingRequest, background_tasks: BackgroundTasks
):
    """
    Train a store clustering model.

    This endpoint trains machine learning models for store clustering
    using historical sales and operational data.
    """
    try:
        logger.info(f"Store clustering training request: {request.dict()}")

        result = await clustering_service.train_clustering_model(
            clustering_method=request.clustering_method,
            n_clusters=request.n_clusters,
            store_id=request.store_id,
            city_id=request.city_id,
            start_date=request.start_date,
            end_date=request.end_date,
            auto_optimize=request.auto_optimize,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "training_results": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Store clustering training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/stores/compare/")
async def compare_clustering_configurations(request: ClusterComparisonRequest):
    """
    Compare different clustering configurations.

    This endpoint compares multiple clustering configurations to help
    determine the optimal number of clusters and clustering approach.
    """
    try:
        logger.info(f"Clustering comparison request: {request.dict()}")

        result = await clustering_service.compare_store_clusters(
            clustering_method=request.clustering_method,
            n_clusters_list=request.n_clusters_list,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "comparison_results": result,
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Clustering comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.post("/stores/segments/")
async def get_store_segments(request: StoreSegmentsRequest):
    """
    Get store segmentation information.

    This endpoint returns basic store segmentation data including
    performance metrics and operational characteristics.
    """
    try:
        logger.info(f"Store segments request: {request.dict()}")

        result = await clustering_service.get_store_segments(
            city_id=request.city_id,
            limit=request.page_size,
            offset=(request.page - 1) * request.page_size,
        )

        return {
            "status": "success",
            "store_segments": result.get("data", []),
            "pagination": {
                "page": result.get("page", request.page),
                "page_size": result.get("page_size", request.page_size),
                "has_more": result.get("has_more", False),
            },
            "request_parameters": request.dict(),
        }

    except Exception as e:
        logger.error(f"Store segments error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Segments retrieval failed: {str(e)}"
        )


# Quick analysis endpoints
@app.get("/stores/quick-cluster/")
async def quick_store_clustering(
    city_id: Optional[int] = Query(None, description="City ID filter"),
    clustering_method: str = Query("kmeans", description="Clustering method"),
    n_clusters: int = Query(5, description="Number of clusters"),
    auto_optimize: bool = Query(True, description="Auto-optimize clusters"),
):
    """
    Quick store clustering analysis with default parameters.

    This endpoint provides a simplified interface for quick clustering analysis
    using the last 6 months of data.
    """
    try:
        # Use last 6 months of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        result = await clustering_service.analyze_store_clustering(
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            city_id=city_id,
            start_date=start_date,
            end_date=end_date,
            auto_optimize=auto_optimize,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Simplified response for quick analysis
        return {
            "status": "success",
            "quick_analysis": {
                "clusters_identified": result["clustering_results"]["n_clusters"],
                "stores_analyzed": result["analysis_summary"]["total_stores_analyzed"],
                "cluster_summary": result["cluster_insights"]["cluster_summary"],
                "top_recommendations": _extract_top_recommendations(
                    result["cluster_insights"]
                ),
            },
            "analysis_period": {"start_date": start_date, "end_date": end_date},
        }

    except Exception as e:
        logger.error(f"Quick clustering error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")


@app.get("/stores/cluster-summary/")
async def get_cluster_summary(
    clustering_method: str = Query("kmeans", description="Clustering method"),
    n_clusters: int = Query(5, description="Number of clusters"),
):
    """
    Get a summary of existing cluster analysis.

    This endpoint returns a summary of already computed clustering results
    without re-running the analysis.
    """
    try:
        # Get model
        model = clustering_service.get_or_create_model(clustering_method, n_clusters)

        if not model.is_fitted:
            return {
                "status": "warning",
                "message": "No clustering model found",
                "recommendation": "Run clustering analysis first using /stores/cluster/ endpoint",
            }

        # Get cluster insights
        insights = model.get_cluster_insights()

        return {
            "status": "success",
            "cluster_summary": insights["cluster_summary"],
            "business_insights": insights["business_insights"],
            "total_clusters": len(insights["cluster_profiles"]),
            "model_info": {
                "clustering_method": clustering_method,
                "n_clusters": n_clusters,
            },
        }

    except Exception as e:
        logger.error(f"Cluster summary error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Summary retrieval failed: {str(e)}"
        )


# Information and help endpoints
@app.get("/stores/methods/")
async def get_clustering_methods():
    """Get information about available clustering methods."""
    return {
        "clustering_methods": {
            "kmeans": {
                "description": "K-Means clustering - Partitions stores into k clusters",
                "pros": ["Fast", "Scalable", "Good for spherical clusters"],
                "cons": ["Requires pre-specified k", "Sensitive to initialization"],
                "best_for": "Well-separated, roughly equal-sized clusters",
                "parameters": ["n_clusters"],
            },
            "dbscan": {
                "description": "DBSCAN - Density-based clustering",
                "pros": [
                    "No need to specify k",
                    "Finds arbitrary shapes",
                    "Handles noise",
                ],
                "cons": ["Sensitive to parameters", "Varying densities problematic"],
                "best_for": "Irregularly shaped clusters with noise",
                "parameters": ["eps", "min_samples"],
            },
            "hierarchical": {
                "description": "Agglomerative Hierarchical clustering",
                "pros": [
                    "No need to specify k upfront",
                    "Hierarchical structure",
                    "Deterministic",
                ],
                "cons": ["Computationally expensive", "Sensitive to noise"],
                "best_for": "Understanding cluster hierarchy and relationships",
                "parameters": ["n_clusters", "linkage"],
            },
        },
        "feature_categories": {
            "sales_performance": [
                "Total sales",
                "Average daily sales",
                "Sales consistency",
            ],
            "customer_behavior": [
                "Weekend preference",
                "Peak hours",
                "Customer loyalty",
            ],
            "operational_efficiency": [
                "Stock management",
                "Inventory turnover",
                "Promotion effectiveness",
            ],
            "product_mix": [
                "Product diversity",
                "Price positioning",
                "Discount strategy",
            ],
            "temporal_patterns": ["Seasonal variation", "Peak day analysis"],
            "geographic_factors": [
                "City characteristics",
                "Market share",
                "Competition",
            ],
        },
    }


@app.get("/stores/health/")
async def health_check():
    """Health check endpoint for store clustering service."""
    try:
        service_health = {
            "service_status": "healthy",
            "available_models": {},
            "cache_status": "active",
        }

        # Check available models
        for method in ["kmeans", "dbscan", "hierarchical"]:
            for n_clusters in [3, 4, 5, 6, 7]:
                model_key = f"{method}_{n_clusters}"
                if model_key in clustering_service.models:
                    model = clustering_service.models[model_key]
                    service_health["available_models"][model_key] = {
                        "loaded": True,
                        "trained": model.is_fitted,
                        "stores_count": (
                            len(model.store_features) if model.is_fitted else 0
                        ),
                    }

        return service_health

    except Exception as e:
        return {"service_status": "unhealthy", "error": str(e)}


@app.get("/stores/examples/")
async def get_usage_examples():
    """Get example API requests for store clustering."""
    return {
        "examples": {
            "basic_clustering": {
                "endpoint": "/stores/cluster/",
                "method": "POST",
                "payload": {
                    "clustering_method": "kmeans",
                    "n_clusters": 5,
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "auto_optimize": True,
                },
                "description": "Basic store clustering analysis",
            },
            "predict_store_clusters": {
                "endpoint": "/stores/predict/",
                "method": "POST",
                "payload": {
                    "store_ids": [1, 2, 3, 4, 5],
                    "clustering_method": "kmeans",
                    "n_clusters": 5,
                },
                "description": "Predict clusters for specific stores",
            },
            "get_cluster_insights": {
                "endpoint": "/stores/insights/",
                "method": "POST",
                "payload": {
                    "cluster_id": 1,
                    "clustering_method": "kmeans",
                    "n_clusters": 5,
                },
                "description": "Get insights for a specific cluster",
            },
            "compare_configurations": {
                "endpoint": "/stores/compare/",
                "method": "POST",
                "payload": {
                    "clustering_method": "kmeans",
                    "n_clusters_list": [3, 4, 5, 6, 7],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                },
                "description": "Compare different cluster configurations",
            },
            "quick_analysis": {
                "endpoint": "/stores/quick-cluster/",
                "method": "GET",
                "query_params": {
                    "clustering_method": "kmeans",
                    "n_clusters": 5,
                    "auto_optimize": "true",
                },
                "description": "Quick clustering analysis with defaults",
            },
        }
    }


# Helper functions
def _extract_top_recommendations(
    cluster_insights: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract top recommendations from cluster insights."""
    recommendations = []

    if "actionable_recommendations" in cluster_insights:
        for cluster_id, cluster_recs in cluster_insights[
            "actionable_recommendations"
        ].items():
            if cluster_recs:  # If there are recommendations for this cluster
                recommendations.append(
                    {
                        "cluster_id": int(cluster_id),
                        "top_recommendation": (
                            cluster_recs[0]
                            if cluster_recs
                            else "No specific recommendations"
                        ),
                        "total_recommendations": len(cluster_recs),
                    }
                )

    return recommendations


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
