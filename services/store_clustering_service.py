"""
Store Clustering Service for store segmentation and behavior analysis.
Provides comprehensive store clustering and business insights.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import json

# Import custom modules
from models.store_clustering_model import StoreClustering
from database.connection import get_pool, cached, paginate
from services.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class StoreClusteringService:
    """Service for store clustering and behavior segmentation."""

    def __init__(self):
        """Initialize the store clustering service."""
        self.models = {}  # Multiple models for different clustering approaches
        self.preprocessor = DataPreprocessor()
        self.cache_duration = 7200  # 2 hours cache for clustering results

    def get_or_create_model(
        self, clustering_method: str = "kmeans", n_clusters: int = 5
    ) -> StoreClustering:
        """Get or create a store clustering model."""
        model_key = f"{clustering_method}_{n_clusters}"

        if model_key not in self.models:
            self.models[model_key] = StoreClustering(
                clustering_method=clustering_method, n_clusters=n_clusters
            )

            # Try to load existing model
            try:
                model_filename = (
                    f"store_clustering_{clustering_method}_{n_clusters}.joblib"
                )
                self.models[model_key].load_model(model_filename)
                logger.info(f"Loaded existing store clustering model: {model_key}")
            except (ValueError, FileNotFoundError):
                logger.info(
                    f"No existing model found for {model_key}, will need to train"
                )

        return self.models[model_key]

    @cached("store_sales_data")
    async def fetch_store_sales_data(
        self,
        store_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch comprehensive sales data for store clustering analysis.

        Args:
            store_id: Filter by store ID
            city_id: Filter by city ID
            start_date: Start date for data
            end_date: End date for data
            limit: Limit number of results
            offset: Offset for pagination

        Returns:
            List of dictionaries with sales data
        """
        pool = await get_pool()

        query = """
        SELECT 
            sd.sale_date,
            sd.store_id,
            sd.product_id,
            ph.first_category_id,
            ph.second_category_id,
            ph.brand_id,
            sh.city_id,
            sd.sale_amount,
            sd.sale_qty,
            sd.discount,
            sd.original_price,
            sd.stock_hour6_22_cnt,
            sd.holiday_flag,
            sd.promo_flag,
            sd.hours_sale,
            sd.hours_stock_status,
            wd.avg_temperature,
            wd.avg_humidity,
            wd.precpt,
            wd.avg_wind_level
        FROM sales_data sd
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        JOIN product_hierarchy ph ON sd.product_id = ph.product_id
        LEFT JOIN weather_data wd ON sd.sale_date = wd.date AND sh.city_id = wd.city_id
        WHERE 1=1
        """

        params = []
        param_count = 0

        if store_id is not None:
            param_count += 1
            query += f" AND sd.store_id = ${param_count}"
            params.append(store_id)

        if city_id is not None:
            param_count += 1
            query += f" AND sh.city_id = ${param_count}"
            params.append(city_id)

        if start_date is not None:
            param_count += 1
            query += f" AND sd.sale_date >= ${param_count}"
            params.append(start_date)

        if end_date is not None:
            param_count += 1
            query += f" AND sd.sale_date <= ${param_count}"
            params.append(end_date)

        query += " ORDER BY sd.sale_date, sd.store_id"

        if limit is not None:
            param_count += 1
            query += f" LIMIT ${param_count}"
            params.append(limit)

        if offset is not None:
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(offset)

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            return [dict(row) for row in rows]

    async def analyze_store_clustering(
        self,
        clustering_method: str = "kmeans",
        n_clusters: int = 5,
        store_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        auto_optimize: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform store clustering analysis.

        Args:
            clustering_method: Clustering algorithm to use
            n_clusters: Number of clusters
            store_id: Store ID filter
            city_id: City ID filter
            start_date: Start date for analysis
            end_date: End date for analysis
            auto_optimize: Whether to automatically optimize cluster count

        Returns:
            Store clustering analysis results
        """
        logger.info(f"Starting store clustering analysis with {clustering_method}...")

        try:
            # Fetch data
            data = await self.fetch_store_sales_data(
                store_id=store_id,
                city_id=city_id,
                start_date=start_date,
                end_date=end_date,
                limit=200000,  # Large limit for comprehensive analysis
            )

            if not data:
                return {"error": "No data found for the specified criteria"}

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Preprocess data
            df = self.preprocessor.handle_missing_values(df)
            df = self.preprocessor.add_time_features(df)

            # Get or create model
            model = StoreClustering(
                clustering_method=clustering_method, n_clusters=n_clusters
            )

            # Perform clustering
            clustering_results = model.fit(df, auto_optimize=auto_optimize)

            # Get cluster insights
            cluster_insights = model.get_cluster_insights()

            # Add data summary
            analysis_summary = {
                "total_stores_analyzed": len(model.store_features),
                "date_range": {
                    "start": df["sale_date"].min().isoformat(),
                    "end": df["sale_date"].max().isoformat(),
                },
                "data_summary": {
                    "total_records": len(df),
                    "unique_products": df["product_id"].nunique(),
                    "unique_categories": df["first_category_id"].nunique(),
                    "total_sales_value": float(df["sale_amount"].sum()),
                },
            }

            # Store the model for future use
            model_key = f"{clustering_method}_{model.n_clusters}"
            self.models[model_key] = model

            logger.info("Store clustering analysis completed successfully")

            return {
                "status": "success",
                "clustering_results": clustering_results,
                "cluster_insights": cluster_insights,
                "analysis_summary": analysis_summary,
                "model_info": {
                    "clustering_method": clustering_method,
                    "n_clusters": model.n_clusters,
                    "auto_optimized": auto_optimize,
                },
            }

        except Exception as e:
            logger.error(f"Error in store clustering analysis: {str(e)}")
            return {"error": f"Clustering analysis failed: {str(e)}"}

    async def train_clustering_model(
        self,
        clustering_method: str = "kmeans",
        n_clusters: int = 5,
        store_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        auto_optimize: bool = True,
    ) -> Dict[str, Any]:
        """
        Train a store clustering model.

        Args:
            clustering_method: Clustering algorithm to use
            n_clusters: Number of clusters
            store_id: Store ID filter for training data
            city_id: City ID filter for training data
            start_date: Start date for training data
            end_date: End date for training data
            auto_optimize: Whether to automatically optimize cluster count

        Returns:
            Training results and metrics
        """
        logger.info(f"Training store clustering model: {clustering_method}")

        try:
            # Fetch training data
            data = await self.fetch_store_sales_data(
                store_id=store_id,
                city_id=city_id,
                start_date=start_date,
                end_date=end_date,
                limit=500000,  # Large limit for training
            )

            if not data:
                return {"error": "No training data found"}

            if len(data) < 1000:
                return {
                    "error": "Insufficient training data (minimum 1000 records required)"
                }

            # Convert to DataFrame and preprocess
            df = pd.DataFrame(data)
            df = self.preprocessor.handle_missing_values(df)
            df = self.preprocessor.add_time_features(df)

            # Create and train model
            model = StoreClustering(
                clustering_method=clustering_method, n_clusters=n_clusters
            )

            training_results = model.fit(df, auto_optimize=auto_optimize)

            # Save model
            model.save_model()

            # Update service model
            model_key = f"{clustering_method}_{model.n_clusters}"
            self.models[model_key] = model

            logger.info("Store clustering model training completed successfully")

            return {
                "status": "success",
                "clustering_method": clustering_method,
                "n_clusters": model.n_clusters,
                "training_data_size": len(df),
                "stores_clustered": len(model.store_features),
                "training_metrics": training_results,
            }

        except Exception as e:
            logger.error(f"Error training clustering model: {str(e)}")
            return {"error": f"Training failed: {str(e)}"}

    async def predict_store_clusters(
        self,
        store_ids: List[int],
        clustering_method: str = "kmeans",
        n_clusters: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict cluster assignments for stores.

        Args:
            store_ids: List of store IDs
            clustering_method: Clustering method used
            n_clusters: Number of clusters

        Returns:
            Store cluster predictions
        """
        logger.info(f"Predicting clusters for {len(store_ids)} stores")

        try:
            # Get model
            model = self.get_or_create_model(
                clustering_method=clustering_method, n_clusters=n_clusters
            )

            if not model.is_fitted:
                return {"error": "Model is not trained. Please train the model first."}

            # Get cluster predictions
            cluster_assignments = model.predict_store_cluster(store_ids)

            # Add cluster profiles for assigned clusters
            assigned_clusters = list(set(cluster_assignments.values()))
            cluster_profiles = {}

            for cluster_id in assigned_clusters:
                if str(cluster_id) in model.cluster_profiles:
                    cluster_profiles[str(cluster_id)] = model.cluster_profiles[
                        str(cluster_id)
                    ]

            # Create response
            store_predictions = []
            for store_id, cluster_id in cluster_assignments.items():
                store_predictions.append(
                    {
                        "store_id": store_id,
                        "predicted_cluster": cluster_id,
                        "cluster_profile": cluster_profiles.get(str(cluster_id), {}),
                    }
                )

            logger.info("Store cluster prediction completed successfully")

            return {
                "status": "success",
                "store_predictions": store_predictions,
                "cluster_profiles": cluster_profiles,
                "prediction_summary": {
                    "stores_predicted": len(store_ids),
                    "unique_clusters": len(assigned_clusters),
                    "model_info": {
                        "clustering_method": clustering_method,
                        "n_clusters": n_clusters,
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error in store cluster prediction: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}

    async def get_store_behavior_insights(
        self,
        store_id: Optional[int] = None,
        cluster_id: Optional[int] = None,
        city_id: Optional[int] = None,
        clustering_method: str = "kmeans",
        n_clusters: int = 5,
    ) -> Dict[str, Any]:
        """
        Get behavioral insights for stores or clusters.

        Args:
            store_id: Specific store ID
            cluster_id: Specific cluster ID
            city_id: City ID filter
            clustering_method: Clustering method
            n_clusters: Number of clusters

        Returns:
            Store behavior insights
        """
        logger.info("Generating store behavior insights...")

        try:
            # Get model
            model = self.get_or_create_model(
                clustering_method=clustering_method, n_clusters=n_clusters
            )

            if not model.is_fitted:
                return {"error": "Model is not trained. Please train the model first."}

            # Get comprehensive insights
            cluster_insights = model.get_cluster_insights()

            # Filter results based on request
            if store_id is not None:
                # Get specific store insights
                store_data = model.store_features[
                    model.store_features["store_id"] == store_id
                ]
                if store_data.empty:
                    return {
                        "error": f"Store {store_id} not found in clustering results"
                    }

                store_cluster = store_data["cluster"].iloc[0]

                return {
                    "status": "success",
                    "store_insights": {
                        "store_id": store_id,
                        "assigned_cluster": int(store_cluster),
                        "cluster_profile": cluster_insights["cluster_profiles"].get(
                            str(store_cluster), {}
                        ),
                        "store_characteristics": self._extract_store_characteristics(
                            store_data.iloc[0]
                        ),
                        "recommendations": cluster_insights[
                            "actionable_recommendations"
                        ].get(str(store_cluster), []),
                    },
                }

            elif cluster_id is not None:
                # Get specific cluster insights
                if str(cluster_id) not in cluster_insights["cluster_profiles"]:
                    return {"error": f"Cluster {cluster_id} not found"}

                cluster_stores = model.store_features[
                    model.store_features["cluster"] == cluster_id
                ]["store_id"].tolist()

                return {
                    "status": "success",
                    "cluster_insights": {
                        "cluster_id": cluster_id,
                        "cluster_profile": cluster_insights["cluster_profiles"][
                            str(cluster_id)
                        ],
                        "stores_in_cluster": cluster_stores,
                        "business_insights": cluster_insights["business_insights"],
                        "recommendations": cluster_insights[
                            "actionable_recommendations"
                        ].get(str(cluster_id), []),
                    },
                }

            else:
                # Return overall insights
                return {
                    "status": "success",
                    "overall_insights": cluster_insights,
                    "model_info": {
                        "clustering_method": clustering_method,
                        "n_clusters": n_clusters,
                        "total_stores": len(model.store_features),
                    },
                }

        except Exception as e:
            logger.error(f"Error generating behavior insights: {str(e)}")
            return {"error": f"Insights generation failed: {str(e)}"}

    @paginate()
    async def get_store_segments(
        self,
        city_id: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get store segmentation information.

        Args:
            city_id: City ID filter
            limit: Limit for pagination
            offset: Offset for pagination

        Returns:
            Store segmentation data
        """
        pool = await get_pool()

        query = """
        SELECT 
            sh.store_id,
            sh.city_id,
            COUNT(DISTINCT sd.product_id) as product_count,
            COUNT(DISTINCT ph.first_category_id) as category_count,
            SUM(sd.sale_amount) as total_sales,
            AVG(sd.sale_amount) as avg_sale_amount,
            COUNT(*) as transaction_count,
            AVG(sd.discount) as avg_discount,
            SUM(CASE WHEN sd.promo_flag > 0 THEN 1 ELSE 0 END)::float / COUNT(*) as promo_frequency
        FROM store_hierarchy sh
        LEFT JOIN sales_data sd ON sh.store_id = sd.store_id
        LEFT JOIN product_hierarchy ph ON sd.product_id = ph.product_id
        WHERE 1=1
        """

        params = []
        param_count = 0

        if city_id is not None:
            param_count += 1
            query += f" AND sh.city_id = ${param_count}"
            params.append(city_id)

        query += """
        GROUP BY sh.store_id, sh.city_id
        HAVING COUNT(*) > 0
        ORDER BY total_sales DESC
        """

        if limit is not None:
            param_count += 1
            query += f" LIMIT ${param_count}"
            params.append(limit)

        if offset is not None:
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(offset)

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            return [dict(row) for row in rows]

    async def compare_store_clusters(
        self,
        clustering_method: str = "kmeans",
        n_clusters_list: List[int] = [3, 4, 5, 6, 7],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare different clustering configurations.

        Args:
            clustering_method: Clustering method to use
            n_clusters_list: List of cluster numbers to compare
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Clustering comparison results
        """
        logger.info(f"Comparing clustering configurations: {n_clusters_list}")

        try:
            # Fetch data once for all comparisons
            data = await self.fetch_store_sales_data(
                start_date=start_date, end_date=end_date, limit=200000
            )

            if not data:
                return {"error": "No data found for comparison"}

            df = pd.DataFrame(data)
            df = self.preprocessor.handle_missing_values(df)
            df = self.preprocessor.add_time_features(df)

            comparison_results = {}

            for n_clusters in n_clusters_list:
                try:
                    # Create and fit model
                    model = StoreClustering(
                        clustering_method=clustering_method, n_clusters=n_clusters
                    )

                    clustering_results = model.fit(df, auto_optimize=False)

                    comparison_results[str(n_clusters)] = {
                        "n_clusters": n_clusters,
                        "clustering_metrics": clustering_results["clustering_metrics"],
                        "cluster_sizes": clustering_results["cluster_sizes"],
                        "silhouette_score": clustering_results[
                            "clustering_metrics"
                        ].get("silhouette_score", 0),
                        "calinski_harabasz_score": clustering_results[
                            "clustering_metrics"
                        ].get("calinski_harabasz_score", 0),
                    }

                except Exception as e:
                    logger.warning(f"Failed to cluster with {n_clusters} clusters: {e}")
                    comparison_results[str(n_clusters)] = {
                        "n_clusters": n_clusters,
                        "error": str(e),
                    }

            # Determine best configuration
            valid_results = {
                k: v for k, v in comparison_results.items() if "error" not in v
            }

            if valid_results:
                best_config = max(
                    valid_results.items(), key=lambda x: x[1].get("silhouette_score", 0)
                )

                return {
                    "status": "success",
                    "comparison_results": comparison_results,
                    "recommended_configuration": {
                        "n_clusters": best_config[1]["n_clusters"],
                        "silhouette_score": best_config[1]["silhouette_score"],
                        "reason": "Highest silhouette score",
                    },
                    "clustering_method": clustering_method,
                }
            else:
                return {"error": "All clustering configurations failed"}

        except Exception as e:
            logger.error(f"Error in clustering comparison: {str(e)}")
            return {"error": f"Comparison failed: {str(e)}"}

    # Private helper methods
    def _extract_store_characteristics(self, store_row: pd.Series) -> Dict[str, Any]:
        """Extract key characteristics for a specific store."""
        characteristics = {
            "sales_performance": {
                "total_sales": float(store_row.get("sale_amount_sum", 0)),
                "avg_daily_sales": float(store_row.get("sale_amount_mean", 0)),
                "sales_consistency": float(store_row.get("customer_loyalty", 0.5)),
            },
            "operational_metrics": {
                "stock_efficiency": float(store_row.get("stock_efficiency", 0)),
                "inventory_turnover": float(store_row.get("inventory_turnover", 0)),
                "promo_effectiveness": float(store_row.get("promo_effectiveness", 0)),
            },
            "customer_behavior": {
                "weekend_preference": float(store_row.get("weekend_preference", 1)),
                "peak_hour": int(store_row.get("peak_hour", 12)),
                "customer_loyalty": float(store_row.get("customer_loyalty", 0.5)),
            },
            "product_mix": {
                "product_diversity": float(store_row.get("product_diversity", 1)),
                "discount_frequency": float(store_row.get("discount_frequency", 0)),
                "price_range": float(store_row.get("price_range", 0)),
            },
        }

        return characteristics
