"""
Store Clustering & Behavior Segmentation Model.
Analyzes store patterns, customer behavior, and operational characteristics for segmentation.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.metrics import silhouette_score, calinski_harabasz_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import joblib  # type: ignore
from pathlib import Path
import warnings
from scipy.spatial.distance import pdist, squareform  # type: ignore
from scipy.cluster.hierarchy import dendrogram, linkage  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class StoreClustering:
    """Store clustering and behavior segmentation model."""

    def __init__(
        self,
        clustering_method: str = "kmeans",
        n_clusters: int = 5,
        save_path: str = "models/saved",
    ):
        """
        Initialize store clustering model.

        Args:
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (for kmeans and hierarchical)
            save_path: Path to save models
        """
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.clustering_model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.store_features = {}
        self.cluster_profiles = {}
        self.feature_importance = {}
        self.is_fitted = False

        # Initialize clustering model
        self._initialize_clustering_model()

    def _initialize_clustering_model(self):
        """Initialize the clustering model based on method."""
        if self.clustering_method == "kmeans":
            self.clustering_model = KMeans(
                n_clusters=self.n_clusters, random_state=42, n_init=10
            )
        elif self.clustering_method == "dbscan":
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        elif self.clustering_method == "hierarchical":
            self.clustering_model = AgglomerativeClustering(
                n_clusters=self.n_clusters, linkage="ward"
            )
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")

    def extract_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive store-level features for clustering.

        Args:
            df: Sales data at product level

        Returns:
            DataFrame with store-level features
        """
        logger.info("Extracting store-level features for clustering...")

        # Basic store statistics
        store_features = (
            df.groupby("store_id")
            .agg(
                {
                    "sale_amount": ["sum", "mean", "std", "count"],
                    "sale_qty": ["sum", "mean", "std"],
                    "discount": ["mean", "max", "count"],
                    "original_price": ["mean", "std"],
                    "stock_hour6_22_cnt": ["mean", "std"],
                    "holiday_flag": ["sum", "mean"],
                    "promo_flag": ["sum", "mean"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        store_features.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in store_features.columns
        ]

        # Sales performance features
        store_features["sales_efficiency"] = (
            store_features["sale_amount_sum"] / store_features["sale_amount_count"]
        )
        store_features["sales_consistency"] = 1 / (
            1 + store_features["sale_amount_std"] / store_features["sale_amount_mean"]
        )
        store_features["revenue_per_transaction"] = (
            store_features["sale_amount_sum"] / store_features["sale_amount_count"]
        )

        # Customer behavior indicators
        store_features = self._add_customer_behavior_features(store_features, df)

        # Operational efficiency features
        store_features = self._add_operational_features(store_features, df)

        # Product mix and variety features
        store_features = self._add_product_mix_features(store_features, df)

        # Temporal patterns
        store_features = self._add_temporal_patterns(store_features, df)

        # Geographic and demographic features
        store_features = self._add_geographic_features(store_features, df)

        # Promotion and marketing features
        store_features = self._add_promotion_features(store_features, df)

        logger.info(
            f"Extracted {len(store_features.columns)} features for {len(store_features)} stores"
        )
        return store_features

    def _add_customer_behavior_features(
        self, store_features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add customer behavior indicators."""
        # Shopping pattern analysis
        if "hours_sale" in df.columns:

            def analyze_peak_hours(hours_str):
                try:
                    import json

                    hours = json.loads(hours_str) if isinstance(hours_str, str) else []
                    if not hours:
                        return {"peak_hour": 12, "peak_intensity": 0, "hours_active": 0}

                    peak_hour = np.argmax(hours)
                    peak_intensity = (
                        max(hours) / (sum(hours) / len(hours)) if sum(hours) > 0 else 0
                    )
                    hours_active = sum([1 for h in hours if h > 0])

                    return {
                        "peak_hour": peak_hour,
                        "peak_intensity": peak_intensity,
                        "hours_active": hours_active,
                    }
                except:
                    return {"peak_hour": 12, "peak_intensity": 0, "hours_active": 0}

            shopping_patterns = (
                df.groupby("store_id")["hours_sale"]
                .apply(lambda x: x.iloc[0] if len(x) > 0 else "[]")
                .apply(analyze_peak_hours)
            )

            pattern_df = pd.DataFrame(
                shopping_patterns.tolist(), index=shopping_patterns.index
            )
            pattern_df["store_id"] = pattern_df.index

            store_features = store_features.merge(pattern_df, on="store_id", how="left")

        # Weekend vs weekday preference
        df["is_weekend"] = df["sale_date"].dt.dayofweek.isin([5, 6])
        weekend_sales = df[df["is_weekend"]].groupby("store_id")["sale_amount"].mean()
        weekday_sales = df[~df["is_weekend"]].groupby("store_id")["sale_amount"].mean()
        weekend_preference = (weekend_sales / weekday_sales).fillna(1)

        store_features["weekend_preference"] = store_features["store_id"].map(
            weekend_preference
        )

        # Customer loyalty indicators (sales consistency)
        daily_sales = (
            df.groupby(["store_id", "sale_date"])["sale_amount"].sum().reset_index()
        )
        loyalty_metrics = daily_sales.groupby("store_id")["sale_amount"].agg(
            ["std", "mean"]
        )
        loyalty_score = 1 / (1 + loyalty_metrics["std"] / loyalty_metrics["mean"])

        store_features["customer_loyalty"] = store_features["store_id"].map(
            loyalty_score.fillna(0.5)
        )

        return store_features

    def _add_operational_features(
        self, store_features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add operational efficiency features."""
        # Stock management efficiency
        stock_metrics = df.groupby("store_id").agg(
            {"stock_hour6_22_cnt": ["mean", "std", "min"], "sale_qty": "sum"}
        )

        stock_efficiency = (
            stock_metrics[("stock_hour6_22_cnt", "mean")]
            / stock_metrics[("sale_qty", "sum")]
        )
        stock_stability = 1 / (1 + stock_metrics[("stock_hour6_22_cnt", "std")])

        store_features["stock_efficiency"] = store_features["store_id"].map(
            stock_efficiency.fillna(0)
        )
        store_features["stock_stability"] = store_features["store_id"].map(
            stock_stability.fillna(0.5)
        )

        # Inventory turnover approximation
        avg_stock = df.groupby("store_id")["stock_hour6_22_cnt"].mean()
        total_sales_qty = df.groupby("store_id")["sale_qty"].sum()
        days_in_period = (df["sale_date"].max() - df["sale_date"].min()).days

        inventory_turnover = (total_sales_qty * 365) / (avg_stock * days_in_period)
        store_features["inventory_turnover"] = store_features["store_id"].map(
            inventory_turnover.fillna(0)
        )

        return store_features

    def _add_product_mix_features(
        self, store_features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add product mix and variety features."""
        # Product diversity
        product_diversity = df.groupby("store_id").agg(
            {"product_id": "nunique", "first_category_id": "nunique"}
        )

        store_features["product_count"] = store_features["store_id"].map(
            product_diversity["product_id"]
        )
        store_features["category_count"] = store_features["store_id"].map(
            product_diversity["first_category_id"]
        )
        store_features["product_diversity"] = (
            store_features["product_count"] / store_features["category_count"]
        )

        # Price positioning
        price_percentiles = (
            df.groupby("store_id")["original_price"]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
        )
        store_features["price_range"] = store_features["store_id"].map(
            price_percentiles[0.75] - price_percentiles[0.25]
        )
        store_features["median_price"] = store_features["store_id"].map(
            price_percentiles[0.5]
        )

        # Premium vs discount orientation
        if "discount" in df.columns:
            discount_frequency = df.groupby("store_id")["discount"].apply(
                lambda x: (x > 0).mean()
            )
            avg_discount = df[df["discount"] > 0].groupby("store_id")["discount"].mean()

            store_features["discount_frequency"] = store_features["store_id"].map(
                discount_frequency.fillna(0)
            )
            store_features["avg_discount"] = store_features["store_id"].map(
                avg_discount.fillna(0)
            )

        return store_features

    def _add_temporal_patterns(
        self, store_features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add temporal pattern features."""
        # Daily patterns
        df["hour"] = pd.to_datetime(df["sale_date"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["sale_date"]).dt.dayofweek
        df["month"] = pd.to_datetime(df["sale_date"]).dt.month

        # Peak day analysis
        daily_sales = (
            df.groupby(["store_id", "day_of_week"])["sale_amount"]
            .mean()
            .unstack(fill_value=0)
        )
        peak_day = daily_sales.idxmax(axis=1)
        peak_day_intensity = daily_sales.max(axis=1) / daily_sales.mean(axis=1)

        store_features["peak_day"] = store_features["store_id"].map(peak_day)
        store_features["peak_day_intensity"] = store_features["store_id"].map(
            peak_day_intensity.fillna(1)
        )

        # Seasonal patterns
        monthly_sales = (
            df.groupby(["store_id", "month"])["sale_amount"]
            .mean()
            .unstack(fill_value=0)
        )
        seasonal_variation = monthly_sales.std(axis=1) / monthly_sales.mean(axis=1)

        store_features["seasonal_variation"] = store_features["store_id"].map(
            seasonal_variation.fillna(0)
        )

        return store_features

    def _add_geographic_features(
        self, store_features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add geographic and demographic features."""
        # City-level features
        if "city_id" in df.columns:
            city_mapping = df.groupby("store_id")["city_id"].first()
            store_features["city_id"] = store_features["store_id"].map(city_mapping)

            # City market characteristics
            city_sales = df.groupby("city_id")["sale_amount"].agg(
                ["mean", "std", "count"]
            )
            city_competition = df.groupby("city_id")["store_id"].nunique()

            store_features["city_avg_sales"] = store_features["city_id"].map(
                city_sales["mean"]
            )
            store_features["city_market_size"] = store_features["city_id"].map(
                city_sales["count"]
            )
            store_features["city_competition"] = store_features["city_id"].map(
                city_competition
            )

            # Store's market share in city
            store_city_sales = df.groupby(["city_id", "store_id"])["sale_amount"].sum()
            city_total_sales = df.groupby("city_id")["sale_amount"].sum()

            market_share = store_city_sales / store_city_sales.groupby("city_id").sum()
            store_features["city_market_share"] = store_features.apply(
                lambda row: market_share.get((row["city_id"], row["store_id"]), 0),
                axis=1,
            )

        # Weather sensitivity (if weather data available)
        if "avg_temperature" in df.columns:
            weather_correlation = df.groupby("store_id").apply(
                lambda x: (
                    x["sale_amount"].corr(x["avg_temperature"]) if len(x) > 10 else 0
                )
            )
            store_features["weather_sensitivity"] = store_features["store_id"].map(
                weather_correlation.fillna(0)
            )

        return store_features

    def _add_promotion_features(
        self, store_features: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add promotion and marketing features."""
        if "promo_flag" in df.columns:
            # Promotion effectiveness
            promo_sales = (
                df[df["promo_flag"] > 0].groupby("store_id")["sale_amount"].mean()
            )
            regular_sales = (
                df[df["promo_flag"] == 0].groupby("store_id")["sale_amount"].mean()
            )

            promo_uplift = ((promo_sales - regular_sales) / regular_sales * 100).fillna(
                0
            )
            store_features["promo_effectiveness"] = store_features["store_id"].map(
                promo_uplift
            )

            # Promotion frequency
            promo_frequency = df.groupby("store_id")["promo_flag"].mean()
            store_features["promo_frequency"] = store_features["store_id"].map(
                promo_frequency
            )

        # Holiday performance
        if "holiday_flag" in df.columns:
            holiday_sales = (
                df[df["holiday_flag"] > 0].groupby("store_id")["sale_amount"].mean()
            )
            regular_sales = (
                df[df["holiday_flag"] == 0].groupby("store_id")["sale_amount"].mean()
            )

            holiday_boost = (
                (holiday_sales - regular_sales) / regular_sales * 100
            ).fillna(0)
            store_features["holiday_boost"] = store_features["store_id"].map(
                holiday_boost
            )

        return store_features

    def prepare_clustering_features(self, store_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and select features for clustering.

        Args:
            store_features: Store-level features DataFrame

        Returns:
            Processed features ready for clustering
        """
        # Select numerical features for clustering
        numerical_cols = store_features.select_dtypes(include=[np.number]).columns
        clustering_features = store_features[numerical_cols].copy()

        # Remove identifier columns
        id_cols = ["store_id", "city_id"]
        clustering_features = clustering_features.drop(
            columns=[col for col in id_cols if col in clustering_features.columns]
        )

        # Handle missing values
        clustering_features = clustering_features.fillna(clustering_features.median())

        # Remove constant columns
        constant_cols = clustering_features.columns[clustering_features.var() == 0]
        clustering_features = clustering_features.drop(columns=constant_cols)

        # Remove highly correlated features
        correlation_matrix = clustering_features.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        high_corr_features = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > 0.95)
        ]
        clustering_features = clustering_features.drop(columns=high_corr_features)

        logger.info(
            f"Selected {len(clustering_features.columns)} features for clustering"
        )
        return clustering_features

    def determine_optimal_clusters(
        self, X: pd.DataFrame, max_clusters: int = 10
    ) -> Dict[str, Any]:
        """
        Determine optimal number of clusters using various metrics.

        Args:
            X: Features for clustering
            max_clusters: Maximum number of clusters to test

        Returns:
            Clustering evaluation results
        """
        logger.info("Determining optimal number of clusters...")

        if self.clustering_method != "kmeans":
            logger.warning("Optimal cluster determination only supported for kmeans")
            return {"optimal_clusters": self.n_clusters}

        evaluation_results = {
            "cluster_range": list(range(2, max_clusters + 1)),
            "inertia": [],
            "silhouette_scores": [],
            "calinski_harabasz_scores": [],
            "optimal_clusters": self.n_clusters,
        }

        X_scaled = self.scaler.fit_transform(X)

        for n_clusters in range(2, max_clusters + 1):
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Calculate metrics
            evaluation_results["inertia"].append(kmeans.inertia_)
            evaluation_results["silhouette_scores"].append(
                silhouette_score(X_scaled, cluster_labels)
            )
            evaluation_results["calinski_harabasz_scores"].append(
                calinski_harabasz_score(X_scaled, cluster_labels)
            )

        # Determine optimal clusters based on silhouette score
        optimal_idx = np.argmax(evaluation_results["silhouette_scores"])
        evaluation_results["optimal_clusters"] = evaluation_results["cluster_range"][
            optimal_idx
        ]

        # Elbow method analysis
        inertias = evaluation_results["inertia"]
        elbow_scores = []
        for i in range(1, len(inertias) - 1):
            elbow_score = abs(inertias[i - 1] - 2 * inertias[i] + inertias[i + 1])
            elbow_scores.append(elbow_score)

        if elbow_scores:
            elbow_optimal_idx = (
                np.argmax(elbow_scores) + 1
            )  # +1 because we started from index 1
            evaluation_results["elbow_optimal_clusters"] = evaluation_results[
                "cluster_range"
            ][elbow_optimal_idx]

        logger.info(
            f"Optimal clusters by silhouette score: {evaluation_results['optimal_clusters']}"
        )
        return evaluation_results

    def fit(self, df: pd.DataFrame, auto_optimize: bool = True) -> Dict[str, Any]:
        """
        Fit the store clustering model.

        Args:
            df: Sales data at product level
            auto_optimize: Whether to automatically optimize cluster count

        Returns:
            Clustering results and metrics
        """
        logger.info("Fitting store clustering model...")

        # Extract store features
        store_features = self.extract_store_features(df)

        # Prepare clustering features
        clustering_features = self.prepare_clustering_features(store_features)

        # Store feature names
        self.feature_columns = clustering_features.columns.tolist()

        # Optimize cluster count if requested
        if auto_optimize and self.clustering_method == "kmeans":
            optimization_results = self.determine_optimal_clusters(clustering_features)
            self.n_clusters = optimization_results["optimal_clusters"]
            self.clustering_model = KMeans(
                n_clusters=self.n_clusters, random_state=42, n_init=10
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(clustering_features)

        # Apply PCA for dimensionality reduction (optional)
        if X_scaled.shape[1] > 20:  # Only if many features
            self.pca = PCA(n_components=min(20, X_scaled.shape[1]), random_state=42)
            X_scaled = self.pca.fit_transform(X_scaled)

        # Fit clustering model
        cluster_labels = self.clustering_model.fit_predict(X_scaled)

        # Store results
        store_features["cluster"] = cluster_labels
        self.store_features = store_features

        # Generate cluster profiles
        self.cluster_profiles = self._generate_cluster_profiles(
            store_features, clustering_features
        )

        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance(
            clustering_features, cluster_labels
        )

        # Calculate clustering metrics
        clustering_metrics = self._calculate_clustering_metrics(
            X_scaled, cluster_labels
        )

        self.is_fitted = True

        logger.info(
            f"Store clustering completed with {len(np.unique(cluster_labels))} clusters"
        )

        return {
            "n_clusters": len(np.unique(cluster_labels)),
            "cluster_sizes": dict(zip(*np.unique(cluster_labels, return_counts=True))),
            "clustering_metrics": clustering_metrics,
            "feature_importance": self.feature_importance,
        }

    def predict_store_cluster(self, store_ids: List[int]) -> Dict[int, int]:
        """
        Predict cluster assignments for new stores.

        Args:
            store_ids: List of store IDs

        Returns:
            Dictionary mapping store IDs to cluster assignments
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # For existing stores, return stored assignments
        cluster_assignments = {}
        for store_id in store_ids:
            if store_id in self.store_features["store_id"].values:
                cluster = self.store_features[
                    self.store_features["store_id"] == store_id
                ]["cluster"].iloc[0]
                cluster_assignments[store_id] = int(cluster)
            else:
                # For new stores, would need to extract features and predict
                # For now, assign to most common cluster
                most_common_cluster = self.store_features["cluster"].mode().iloc[0]
                cluster_assignments[store_id] = int(most_common_cluster)

        return cluster_assignments

    def get_cluster_insights(self) -> Dict[str, Any]:
        """Get detailed insights about store clusters."""
        if not self.is_fitted:
            return {"error": "Model not fitted"}

        insights = {
            "cluster_profiles": self.cluster_profiles,
            "cluster_summary": {},
            "business_insights": {},
            "actionable_recommendations": {},
        }

        # Cluster summary statistics
        for cluster_id in self.store_features["cluster"].unique():
            cluster_data = self.store_features[
                self.store_features["cluster"] == cluster_id
            ]

            insights["cluster_summary"][str(cluster_id)] = {
                "store_count": len(cluster_data),
                "avg_total_sales": float(cluster_data["sale_amount_sum"].mean()),
                "avg_daily_sales": float(cluster_data["sale_amount_mean"].mean()),
                "sales_consistency": float(
                    cluster_data.get("customer_loyalty", pd.Series([0])).mean()
                ),
                "geographic_spread": (
                    len(cluster_data["city_id"].unique())
                    if "city_id" in cluster_data
                    else 1
                ),
            }

        # Business insights
        insights["business_insights"] = self._generate_business_insights()

        # Actionable recommendations
        insights["actionable_recommendations"] = self._generate_recommendations()

        return insights

    def save_model(self, filename: Optional[str] = None):
        """Save the trained clustering model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        if filename is None:
            filename = (
                f"store_clustering_{self.clustering_method}_{self.n_clusters}.joblib"
            )

        model_path = self.save_path / filename

        model_data = {
            "clustering_model": self.clustering_model,
            "scaler": self.scaler,
            "pca": self.pca,
            "clustering_method": self.clustering_method,
            "n_clusters": self.n_clusters,
            "feature_columns": self.feature_columns,
            "store_features": self.store_features,
            "cluster_profiles": self.cluster_profiles,
            "feature_importance": self.feature_importance,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Store clustering model saved to {model_path}")

    def load_model(self, filename: str):
        """Load a trained clustering model."""
        model_path = self.save_path / filename

        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)

        self.clustering_model = model_data["clustering_model"]
        self.scaler = model_data["scaler"]
        self.pca = model_data.get("pca")
        self.clustering_method = model_data["clustering_method"]
        self.n_clusters = model_data["n_clusters"]
        self.feature_columns = model_data["feature_columns"]
        self.store_features = model_data["store_features"]
        self.cluster_profiles = model_data["cluster_profiles"]
        self.feature_importance = model_data["feature_importance"]
        self.is_fitted = model_data["is_fitted"]

        logger.info(f"Store clustering model loaded from {model_path}")

    # Private helper methods
    def _generate_cluster_profiles(
        self, store_features: pd.DataFrame, clustering_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive cluster profiles."""
        profiles = {}

        for cluster_id in store_features["cluster"].unique():
            cluster_data = store_features[store_features["cluster"] == cluster_id]
            cluster_features = clustering_features.iloc[cluster_data.index]

            profiles[str(cluster_id)] = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(store_features) * 100,
                "characteristics": {
                    "sales_performance": {
                        "avg_total_sales": float(
                            cluster_data["sale_amount_sum"].mean()
                        ),
                        "avg_daily_sales": float(
                            cluster_data["sale_amount_mean"].mean()
                        ),
                        "sales_efficiency": float(
                            cluster_data.get("sales_efficiency", pd.Series([0])).mean()
                        ),
                    },
                    "customer_behavior": {
                        "weekend_preference": float(
                            cluster_data.get(
                                "weekend_preference", pd.Series([1])
                            ).mean()
                        ),
                        "customer_loyalty": float(
                            cluster_data.get(
                                "customer_loyalty", pd.Series([0.5])
                            ).mean()
                        ),
                        "peak_hour": float(
                            cluster_data.get("peak_hour", pd.Series([12])).mean()
                        ),
                    },
                    "operational_efficiency": {
                        "stock_efficiency": float(
                            cluster_data.get("stock_efficiency", pd.Series([0])).mean()
                        ),
                        "inventory_turnover": float(
                            cluster_data.get(
                                "inventory_turnover", pd.Series([0])
                            ).mean()
                        ),
                        "promo_effectiveness": float(
                            cluster_data.get(
                                "promo_effectiveness", pd.Series([0])
                            ).mean()
                        ),
                    },
                    "product_mix": {
                        "product_diversity": float(
                            cluster_data.get("product_diversity", pd.Series([1])).mean()
                        ),
                        "avg_price_range": float(
                            cluster_data.get("price_range", pd.Series([0])).mean()
                        ),
                        "discount_frequency": float(
                            cluster_data.get(
                                "discount_frequency", pd.Series([0])
                            ).mean()
                        ),
                    },
                },
                "feature_means": cluster_features.mean().to_dict(),
                "top_features": self._get_cluster_distinguishing_features(
                    cluster_id, clustering_features, store_features
                ),
            }

        return profiles

    def _get_cluster_distinguishing_features(
        self,
        cluster_id: int,
        clustering_features: pd.DataFrame,
        store_features: pd.DataFrame,
    ) -> Dict[str, float]:
        """Get features that distinguish this cluster from others."""
        cluster_mask = store_features["cluster"] == cluster_id
        cluster_features = clustering_features[cluster_mask]
        other_features = clustering_features[~cluster_mask]

        # Calculate z-scores for cluster vs overall mean
        overall_mean = clustering_features.mean()
        overall_std = clustering_features.std()
        cluster_mean = cluster_features.mean()

        z_scores = (cluster_mean - overall_mean) / overall_std

        # Return top distinguishing features (highest absolute z-scores)
        distinguishing_features = z_scores.abs().nlargest(10)

        return {
            feature: float(z_scores[feature])
            for feature in distinguishing_features.index
        }

    def _calculate_feature_importance(
        self, clustering_features: pd.DataFrame, cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance for clustering."""
        # Use variance ratio as a proxy for feature importance
        total_variance = clustering_features.var()

        importance_scores = {}
        for feature in clustering_features.columns:
            feature_values = clustering_features[feature]

            # Calculate between-cluster variance vs within-cluster variance
            cluster_means = (
                pd.Series(cluster_labels)
                .groupby(cluster_labels)
                .apply(lambda x: feature_values.iloc[x.index].mean())
            )

            overall_mean = feature_values.mean()
            between_variance = ((cluster_means - overall_mean) ** 2).mean()
            within_variance = total_variance[feature] - between_variance

            # Importance as ratio of between to within cluster variance
            if within_variance > 0:
                importance_scores[feature] = float(between_variance / within_variance)
            else:
                importance_scores[feature] = 0.0

        # Normalize scores
        max_importance = (
            max(importance_scores.values()) if importance_scores.values() else 1
        )
        importance_scores = {
            k: v / max_importance for k, v in importance_scores.items()
        }

        # Sort by importance
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

    def _calculate_clustering_metrics(
        self, X_scaled: np.ndarray, cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}

        try:
            if len(np.unique(cluster_labels)) > 1:
                metrics["silhouette_score"] = float(
                    silhouette_score(X_scaled, cluster_labels)
                )
                metrics["calinski_harabasz_score"] = float(
                    calinski_harabasz_score(X_scaled, cluster_labels)
                )
            else:
                metrics["silhouette_score"] = 0.0
                metrics["calinski_harabasz_score"] = 0.0
        except:
            metrics["silhouette_score"] = 0.0
            metrics["calinski_harabasz_score"] = 0.0

        if hasattr(self.clustering_model, "inertia_"):
            metrics["inertia"] = float(self.clustering_model.inertia_)

        return metrics

    def _generate_business_insights(self) -> Dict[str, Any]:
        """Generate business insights from clustering results."""
        insights = {
            "performance_segments": {},
            "geographic_patterns": {},
            "operational_insights": {},
        }

        # Performance-based segmentation
        performance_ranking = []
        for cluster_id, profile in self.cluster_profiles.items():
            avg_sales = profile["characteristics"]["sales_performance"][
                "avg_total_sales"
            ]
            performance_ranking.append((cluster_id, avg_sales))

        performance_ranking.sort(key=lambda x: x[1], reverse=True)

        insights["performance_segments"] = {
            "high_performers": performance_ranking[:2],
            "medium_performers": (
                performance_ranking[2:4] if len(performance_ranking) > 3 else []
            ),
            "improvement_needed": (
                performance_ranking[4:] if len(performance_ranking) > 4 else []
            ),
        }

        # Geographic insights (if available)
        if "city_id" in self.store_features.columns:
            city_cluster_dist = (
                self.store_features.groupby(["city_id", "cluster"])
                .size()
                .unstack(fill_value=0)
            )
            dominant_clusters = city_cluster_dist.idxmax(axis=1)
            insights["geographic_patterns"] = dominant_clusters.to_dict()

        # Operational insights
        efficiency_scores = []
        for cluster_id, profile in self.cluster_profiles.items():
            efficiency = profile["characteristics"]["operational_efficiency"][
                "stock_efficiency"
            ]
            efficiency_scores.append((cluster_id, efficiency))

        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        insights["operational_insights"] = {
            "most_efficient_clusters": efficiency_scores[:2],
            "efficiency_improvement_clusters": efficiency_scores[2:],
        }

        return insights

    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate actionable recommendations for each cluster."""
        recommendations = {}

        for cluster_id, profile in self.cluster_profiles.items():
            cluster_recommendations = []

            chars = profile["characteristics"]

            # Sales performance recommendations
            if (
                chars["sales_performance"]["avg_total_sales"] < 10000
            ):  # Low sales threshold
                cluster_recommendations.append(
                    "Focus on increasing foot traffic through targeted marketing"
                )
                cluster_recommendations.append(
                    "Review product mix and pricing strategy"
                )

            # Customer behavior recommendations
            if chars["customer_behavior"]["customer_loyalty"] < 0.4:
                cluster_recommendations.append("Implement customer loyalty programs")
                cluster_recommendations.append(
                    "Improve customer experience and service quality"
                )

            # Operational efficiency recommendations
            if chars["operational_efficiency"]["stock_efficiency"] < 0.5:
                cluster_recommendations.append(
                    "Optimize inventory management processes"
                )
                cluster_recommendations.append(
                    "Implement demand forecasting for better stock planning"
                )

            if (
                chars["operational_efficiency"]["promo_effectiveness"] < 5
            ):  # Low promotion impact
                cluster_recommendations.append(
                    "Review and optimize promotion strategies"
                )
                cluster_recommendations.append(
                    "Test different promotion types and timing"
                )

            # Product mix recommendations
            if chars["product_mix"]["product_diversity"] < 2:
                cluster_recommendations.append(
                    "Expand product variety to attract more customers"
                )

            if chars["product_mix"]["discount_frequency"] > 0.3:
                cluster_recommendations.append(
                    "Reduce dependency on discounts, focus on value proposition"
                )

            recommendations[cluster_id] = cluster_recommendations

        return recommendations
