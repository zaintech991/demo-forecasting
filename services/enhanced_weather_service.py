# type: ignore
"""
Enhanced weather-sensitive demand modeling service with advanced analytics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Import database manager
from database.connection import db_manager

logger = logging.getLogger(__name__)


class WeatherSensitivityLevel(Enum):
    """Weather sensitivity levels"""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class WeatherFactor(Enum):
    """Weather factors for analysis"""

    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRECIPITATION = "precipitation"
    WIND = "wind"
    ALL = "all"


@dataclass
class WeatherImpactRequest:
    """Request model for weather impact analysis"""

    product_ids: List[int]
    city_ids: Optional[List[int]] = None
    store_ids: Optional[List[int]] = None
    analysis_period_days: int = 180
    weather_factors: Optional[List[WeatherFactor]] = None
    include_seasonal_analysis: bool = True
    include_product_clustering: bool = True
    include_forecasting: bool = True
    forecast_horizon_days: int = 14


@dataclass
class WeatherImpactResult:
    """Result model for weather impact analysis"""

    weather_correlations: Dict[str, Any]
    product_clustering: Optional[Dict[str, Any]] = None
    seasonal_patterns: Optional[Dict[str, Any]] = None
    weather_forecasts: Optional[Dict[str, Any]] = None
    optimization_recommendations: Optional[List[str]] = None
    elasticity_analysis: Optional[Dict[str, Any]] = None


class EnhancedWeatherService:
    """
    Enhanced weather-sensitive demand modeling service
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.weather_models: Dict[str, Any] = {}

    async def analyze_weather_impact(
        self, request: WeatherImpactRequest
    ) -> WeatherImpactResult:
        """
        Comprehensive weather impact analysis

        Args:
            request: Weather impact analysis request

        Returns:
            WeatherImpactResult with comprehensive analysis
        """
        logger.info(f"Analyzing weather impact for {len(request.product_ids)} products")

        # 1. Get comprehensive data
        historical_data = await self._get_weather_sales_data(request)

        if historical_data.empty:
            raise ValueError("No historical data available for weather analysis")

        # 2. Calculate weather correlations
        weather_correlations = await self._calculate_weather_correlations(
            historical_data, request.weather_factors or [WeatherFactor.ALL]
        )

        # 3. Perform product clustering by weather sensitivity
        product_clustering = None
        if request.include_product_clustering:
            product_clustering = await self._cluster_products_by_weather_sensitivity(
                historical_data, request.product_ids
            )

        # 4. Analyze seasonal patterns
        seasonal_patterns = None
        if request.include_seasonal_analysis:
            seasonal_patterns = await self._analyze_seasonal_weather_patterns(
                historical_data, request.product_ids
            )

        # 5. Generate weather-based forecasts
        weather_forecasts = None
        if request.include_forecasting:
            weather_forecasts = await self._generate_weather_based_forecasts(
                historical_data, request.forecast_horizon_days
            )

        # 6. Calculate elasticity analysis
        elasticity_analysis = await self._calculate_weather_elasticity(
            historical_data, request.product_ids
        )

        # 7. Generate optimization recommendations
        recommendations = self._generate_weather_optimization_recommendations(
            weather_correlations,
            product_clustering,
            seasonal_patterns,
            elasticity_analysis,
        )

        return WeatherImpactResult(
            weather_correlations=weather_correlations,
            product_clustering=product_clustering,
            seasonal_patterns=seasonal_patterns,
            weather_forecasts=weather_forecasts,
            elasticity_analysis=elasticity_analysis,
            optimization_recommendations=recommendations,
        )

    async def _get_weather_sales_data(
        self, request: WeatherImpactRequest
    ) -> pd.DataFrame:
        """Get comprehensive weather and sales data"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.analysis_period_days)

        # Get sales data with weather information
        sales_data = await db_manager.get_sales_data(
            store_ids=request.store_ids,
            product_ids=request.product_ids,
            city_ids=request.city_ids,
            start_date=start_date,
            end_date=end_date,
            include_hourly=False,
            include_stockouts=True,
        )

        if sales_data.empty:
            return sales_data

        # Add temporal features
        sales_data["sale_date"] = pd.to_datetime(sales_data["sale_date"])
        sales_data["day_of_year"] = sales_data["sale_date"].dt.dayofyear
        sales_data["month"] = sales_data["sale_date"].dt.month
        sales_data["quarter"] = sales_data["sale_date"].dt.quarter
        sales_data["season"] = sales_data["month"].apply(self._get_season)
        sales_data["is_weekend"] = (
            sales_data["sale_date"].dt.dayofweek.isin([5, 6]).astype(int)
        )

        # Add weather interaction features
        if (
            "avg_temperature" in sales_data.columns
            and "avg_humidity" in sales_data.columns
        ):
            sales_data["temp_humidity_interaction"] = (
                sales_data["avg_temperature"] * sales_data["avg_humidity"]
            )

        if "avg_temperature" in sales_data.columns:
            sales_data["temp_squared"] = sales_data["avg_temperature"] ** 2
            sales_data["temp_comfort"] = self._calculate_temperature_comfort(
                sales_data["avg_temperature"]
            )

        if "precpt" in sales_data.columns:
            sales_data["is_rainy"] = (sales_data["precpt"] > 0.1).astype(int)
            sales_data["heavy_rain"] = (sales_data["precpt"] > 10.0).astype(int)

        return sales_data

    async def _calculate_weather_correlations(
        self, data: pd.DataFrame, weather_factors: List[WeatherFactor]
    ) -> Dict[str, Any]:
        """Calculate weather-sales correlations"""

        correlations = {}

        weather_columns = {
            WeatherFactor.TEMPERATURE: [
                "avg_temperature",
                "temp_squared",
                "temp_comfort",
            ],
            WeatherFactor.HUMIDITY: ["avg_humidity"],
            WeatherFactor.PRECIPITATION: ["precpt", "is_rainy", "heavy_rain"],
            WeatherFactor.WIND: ["avg_wind_level"],
        }

        # Process each product
        for product_id in data["product_id"].unique():
            product_data = data[data["product_id"] == product_id].copy()

            if len(product_data) < 30:  # Need sufficient data
                continue

            product_correlations: Dict[str, Any] = {
                "product_id": int(product_id),
                "total_observations": len(product_data),
                "weather_correlations": {},
            }

            # Calculate correlations for each weather factor
            for factor in weather_factors:
                if factor == WeatherFactor.ALL:
                    # Calculate for all factors
                    all_weather_cols = []
                    for cols in weather_columns.values():
                        all_weather_cols.extend(
                            [col for col in cols if col in product_data.columns]
                        )
                    factor_cols = all_weather_cols
                else:
                    factor_cols = [
                        col
                        for col in weather_columns[factor]
                        if col in product_data.columns
                    ]

                factor_correlations: Dict[str, Any] = {}
                for col in factor_cols:
                    try:
                        correlation = float(product_data["sale_amount"].corr(product_data[col]))  # type: ignore
                        if not pd.isna(correlation):
                            factor_correlations[col] = {
                                "correlation": correlation,
                                "strength": self._classify_correlation_strength(
                                    abs(correlation)
                                ),
                                "direction": (
                                    "positive" if correlation > 0 else "negative"
                                ),
                            }
                    except (TypeError, ValueError):
                        continue

                if factor == WeatherFactor.ALL:
                    product_correlations["weather_correlations"] = factor_correlations
                else:
                    product_correlations["weather_correlations"][
                        factor.value
                    ] = factor_correlations

            # Calculate overall weather sensitivity score
            all_correlations = []
            for factor_data in product_correlations["weather_correlations"].values():
                if isinstance(factor_data, dict):
                    for corr_data in factor_data.values():
                        if isinstance(corr_data, dict) and "correlation" in corr_data:
                            all_correlations.append(abs(corr_data["correlation"]))
                    else:
                        if (
                            isinstance(factor_data, dict)
                            and "correlation" in factor_data
                        ):
                            all_correlations.append(abs(factor_data["correlation"]))

            if all_correlations:
                avg_correlation = np.mean(all_correlations)
                max_correlation = np.max(all_correlations)

                product_correlations["overall_weather_sensitivity"] = {
                    "average_correlation": float(avg_correlation),
                    "maximum_correlation": float(max_correlation),
                    "sensitivity_level": self._classify_weather_sensitivity(
                        max_correlation
                    ),
                    "primary_weather_factor": self._identify_primary_weather_factor(
                        product_correlations["weather_correlations"]
                    ),
                }

            correlations[f"product_{product_id}"] = product_correlations

        return correlations

    async def _cluster_products_by_weather_sensitivity(
        self, data: pd.DataFrame, product_ids: List[int]
    ) -> Dict[str, Any]:
        """Cluster products by weather sensitivity patterns"""

        # Prepare features for clustering
        clustering_features = []
        product_mapping = []

        weather_features = [
            "avg_temperature",
            "avg_humidity",
            "precpt",
            "avg_wind_level",
        ]
        available_features = [col for col in weather_features if col in data.columns]

        if len(available_features) < 2:
            return {"error": "Insufficient weather features for clustering"}

        for product_id in product_ids:
            product_data = data[data["product_id"] == product_id]

            if len(product_data) < 30:
                continue

            # Calculate weather sensitivity features
            features = []
            for weather_col in available_features:
                correlation = product_data["sale_amount"].corr(
                    product_data[weather_col]
                )
                features.append(correlation if not pd.isna(correlation) else 0.0)

            # Add seasonal sensitivity
            seasonal_variance = (
                product_data.groupby("season")["sale_amount"].var().var()
            )
            features.append(
                seasonal_variance if not pd.isna(seasonal_variance) else 0.0
            )

            # Add weather volatility impact
            weather_volatility = np.mean(
                [product_data[col].std() for col in available_features]
            )
            sales_volatility = product_data["sale_amount"].std()
            volatility_ratio = (
                sales_volatility / weather_volatility if weather_volatility > 0 else 0.0
            )
            features.append(volatility_ratio)

            clustering_features.append(features)
            product_mapping.append(product_id)

        if len(clustering_features) < 3:
            return {"error": "Insufficient products for meaningful clustering"}

        # Perform clustering
        clustering_features_array = np.array(clustering_features)

        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(clustering_features_array)

        # K-means clustering
        optimal_k = min(5, len(clustering_features) // 2)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_features)

        # DBSCAN clustering for comparison
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(normalized_features)

        # Organize clustering results
        clusters: Dict[str, Any] = {}

        for i, product_id in enumerate(product_mapping):
            kmeans_cluster = int(cluster_labels[i])
            dbscan_cluster = int(dbscan_labels[i])

            if kmeans_cluster not in clusters:
                clusters[kmeans_cluster] = {
                    "products": [],
                    "characteristics": {},
                    "average_features": [],
                }

            clusters[kmeans_cluster]["products"].append(
                {
                    "product_id": product_id,
                    "features": clustering_features[i],
                    "dbscan_cluster": dbscan_cluster,
                }
            )

        # Calculate cluster characteristics
        for cluster_id, cluster_data in clusters.items():
            if cluster_data["products"]:
                # Calculate average features
                all_features = np.array(
                    [p["features"] for p in cluster_data["products"]]
                )
                avg_features = np.mean(all_features, axis=0)

                cluster_data["average_features"] = avg_features.tolist()
                cluster_data["size"] = len(cluster_data["products"])

                # Classify cluster
                temp_sensitivity = abs(avg_features[0]) if len(avg_features) > 0 else 0
                humidity_sensitivity = (
                    abs(avg_features[1]) if len(avg_features) > 1 else 0
                )
                precip_sensitivity = (
                    abs(avg_features[2]) if len(avg_features) > 2 else 0
                )

                max_sensitivity = max(
                    temp_sensitivity, humidity_sensitivity, precip_sensitivity
                )

                cluster_data["characteristics"] = {
                    "weather_sensitivity_level": self._classify_weather_sensitivity(
                        max_sensitivity
                    ),
                    "primary_weather_factor": available_features[
                        np.argmax(
                            [temp_sensitivity, humidity_sensitivity, precip_sensitivity]
                        )
                    ],
                    "seasonal_sensitivity": (
                        avg_features[-2] if len(avg_features) > 4 else 0
                    ),
                    "volatility_ratio": (
                        avg_features[-1] if len(avg_features) > 3 else 0
                    ),
                }

        return {
            "clustering_method": "kmeans",
            "number_of_clusters": optimal_k,
            "clusters": clusters,
            "feature_names": available_features
            + ["seasonal_variance", "volatility_ratio"],
            "clustering_summary": {
                "total_products_clustered": len(product_mapping),
                "largest_cluster_size": (
                    max([c["size"] for c in clusters.values()]) if clusters else 0
                ),
                "most_weather_sensitive_cluster": self._identify_most_sensitive_cluster(
                    clusters
                ),
            },
        }

    async def _analyze_seasonal_weather_patterns(
        self, data: pd.DataFrame, product_ids: List[int]
    ) -> Dict[str, Any]:
        """Analyze seasonal weather-demand patterns"""

        seasonal_patterns = {}

        for product_id in product_ids:
            product_data = data[data["product_id"] == product_id].copy()

            if len(product_data) < 60:  # Need at least 2 months of data
                continue

            # Group by season
            seasonal_analysis = {}

            for season in ["spring", "summer", "autumn", "winter"]:
                season_data = product_data[product_data["season"] == season]

                if len(season_data) < 10:
                    continue

                # Calculate seasonal metrics
                seasonal_metrics = {
                    "average_sales": float(season_data["sale_amount"].mean()),
                    "sales_volatility": float(season_data["sale_amount"].std()),
                    "average_temperature": (
                        float(season_data["avg_temperature"].mean())
                        if "avg_temperature" in season_data.columns
                        else None
                    ),
                    "average_humidity": (
                        float(season_data["avg_humidity"].mean())
                        if "avg_humidity" in season_data.columns
                        else None
                    ),
                    "total_precipitation": (
                        float(season_data["precpt"].sum())
                        if "precpt" in season_data.columns
                        else None
                    ),
                    "stockout_frequency": (
                        float(season_data["stock_hour6_22_cnt"].mean() / 16)
                        if "stock_hour6_22_cnt" in season_data.columns
                        else None
                    ),
                    "data_points": len(season_data),
                }

                # Calculate weather-sales correlations for this season
                weather_correlations = {}
                for weather_col in [
                    "avg_temperature",
                    "avg_humidity",
                    "precpt",
                    "avg_wind_level",
                ]:
                    if weather_col in season_data.columns:
                        correlation = season_data["sale_amount"].corr(
                            season_data[weather_col]
                        )
                        if not pd.isna(correlation):
                            weather_correlations[weather_col] = float(correlation)

                seasonal_metrics["weather_correlations"] = weather_correlations
                seasonal_analysis[season] = seasonal_metrics

            # Calculate year-over-year patterns if data spans multiple years
            if len(product_data) > 365:
                yearly_analysis = self._analyze_yearly_patterns(product_data)
                seasonal_analysis["yearly_trends"] = yearly_analysis

            # Identify optimal weather conditions
            optimal_conditions = self._identify_optimal_weather_conditions(product_data)
            seasonal_analysis["optimal_conditions"] = optimal_conditions

            seasonal_patterns[f"product_{product_id}"] = seasonal_analysis

        return seasonal_patterns

    async def _generate_weather_based_forecasts(
        self, data: pd.DataFrame, horizon_days: int
    ) -> Dict[str, Any]:
        """Generate demand forecasts based on weather patterns"""

        forecasts = {}

        # Prepare weather features
        weather_features = [
            "avg_temperature",
            "avg_humidity",
            "precpt",
            "avg_wind_level",
        ]
        available_features = [col for col in weather_features if col in data.columns]

        if len(available_features) < 2:
            return {"error": "Insufficient weather features for forecasting"}

        for product_id in data["product_id"].unique():
            product_data = data[data["product_id"] == product_id].copy()

            if len(product_data) < 60:  # Need sufficient training data
                continue

            try:
                # Prepare features and target
                X = product_data[
                    available_features + ["day_of_year", "is_weekend"]
                ].fillna(0)
                y = product_data["sale_amount"]

                # Train weather-based forecast model
                rf_model = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )

                rf_model.fit(X, y)

                # Generate future weather scenarios (simplified)
                future_scenarios = self._generate_weather_scenarios(
                    product_data, available_features, horizon_days
                )

                # Generate forecasts for each scenario
                scenario_forecasts = {}

                for scenario_name, scenario_data in future_scenarios.items():
                    scenario_predictions = rf_model.predict(scenario_data)

                    scenario_forecasts[scenario_name] = {
                        "predictions": scenario_predictions.tolist(),
                        "total_demand": float(scenario_predictions.sum()),
                        "average_daily_demand": float(scenario_predictions.mean()),
                        "demand_variance": float(scenario_predictions.var()),
                    }

                # Calculate feature importance
                feature_importance = dict(
                    zip(
                        available_features + ["day_of_year", "is_weekend"],
                        rf_model.feature_importances_,
                    )
                )

                # Model performance on training data
                train_predictions = rf_model.predict(X)
                model_performance = {
                    "r2_score": float(r2_score(y, train_predictions)),
                    "mae": float(mean_absolute_error(y, train_predictions)),
                    "training_samples": len(X),
                }

                forecasts[f"product_{product_id}"] = {
                    "scenario_forecasts": scenario_forecasts,
                    "feature_importance": feature_importance,
                    "model_performance": model_performance,
                    "forecast_horizon_days": horizon_days,
                }

            except Exception as e:
                logger.warning(
                    f"Weather forecasting failed for product {product_id}: {e}"
                )
                continue

        return forecasts

    async def _calculate_weather_elasticity(
        self, data: pd.DataFrame, product_ids: List[int]
    ) -> Dict[str, Any]:
        """Calculate weather elasticity for demand optimization"""

        elasticity_analysis = {}

        for product_id in product_ids:
            product_data = data[data["product_id"] == product_id].copy()

            if len(product_data) < 30:
                continue

            elasticities = {}

            # Temperature elasticity
            if "avg_temperature" in product_data.columns:
                temp_elasticity = self._calculate_elasticity(
                    product_data["avg_temperature"], product_data["sale_amount"]
                )
                elasticities["temperature"] = {
                    "elasticity": float(temp_elasticity),
                    "interpretation": self._interpret_elasticity(
                        temp_elasticity, "temperature"
                    ),
                    "optimal_range": self._find_optimal_temperature_range(product_data),
                }

            # Humidity elasticity
            if "avg_humidity" in product_data.columns:
                humidity_elasticity = self._calculate_elasticity(
                    product_data["avg_humidity"], product_data["sale_amount"]
                )
                elasticities["humidity"] = {
                    "elasticity": float(humidity_elasticity),
                    "interpretation": self._interpret_elasticity(
                        humidity_elasticity, "humidity"
                    ),
                }

            # Precipitation elasticity
            if "precpt" in product_data.columns:
                precip_elasticity = self._calculate_elasticity(
                    product_data["precpt"] + 0.1,  # Add small value to avoid log(0)
                    product_data["sale_amount"],
                )
                elasticities["precipitation"] = {
                    "elasticity": float(precip_elasticity),
                    "interpretation": self._interpret_elasticity(
                        precip_elasticity, "precipitation"
                    ),
                }

            # Calculate demand response to extreme weather
            extreme_weather_impact = self._analyze_extreme_weather_impact(product_data)

            elasticity_analysis[f"product_{product_id}"] = {
                "elasticities": elasticities,
                "extreme_weather_impact": extreme_weather_impact,
                "weather_sensitivity_classification": self._classify_product_weather_sensitivity(
                    elasticities
                ),
            }

        return elasticity_analysis

    def _generate_weather_optimization_recommendations(
        self,
        weather_correlations: Dict[str, Any],
        product_clustering: Optional[Dict[str, Any]],
        seasonal_patterns: Optional[Dict[str, Any]],
        elasticity_analysis: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate weather-based optimization recommendations"""

        recommendations = []

        # Weather correlation recommendations
        if weather_correlations:
            high_sensitivity_products = []
            for product_key, product_data in weather_correlations.items():
                if "overall_weather_sensitivity" in product_data:
                    sensitivity_level = product_data["overall_weather_sensitivity"].get(
                        "sensitivity_level"
                    )
                    if sensitivity_level in ["very_high", "high"]:
                        high_sensitivity_products.append(product_key)

            if high_sensitivity_products:
                recommendations.append(
                    f"Products {', '.join(high_sensitivity_products)} show high weather sensitivity. "
                    "Consider weather-based dynamic pricing and inventory adjustments."
                )

        # Clustering recommendations
        if product_clustering and "clusters" in product_clustering:
            most_sensitive_cluster = product_clustering.get(
                "clustering_summary", {}
            ).get("most_weather_sensitive_cluster")
            if most_sensitive_cluster is not None:
                recommendations.append(
                    f"Cluster {most_sensitive_cluster} contains the most weather-sensitive products. "
                    "Implement coordinated weather response strategies for these products."
                )

        # Seasonal recommendations
        if seasonal_patterns:
            for product_key, seasonal_data in seasonal_patterns.items():
                if "optimal_conditions" in seasonal_data:
                    optimal_temp = seasonal_data["optimal_conditions"].get(
                        "optimal_temperature_range"
                    )
                    if optimal_temp:
                        recommendations.append(
                            f"{product_key}: Optimal sales temperature range is {optimal_temp}. "
                            "Monitor weather forecasts and adjust inventory 2-3 days in advance."
                        )

        # Elasticity recommendations
        if elasticity_analysis:
            for product_key, elasticity_data in elasticity_analysis.items():
                sensitivity_class = elasticity_data.get(
                    "weather_sensitivity_classification"
                )
                if sensitivity_class == "highly_elastic":
                    recommendations.append(
                        f"{product_key} is highly weather-elastic. "
                        "Implement automated inventory adjustments based on weather forecasts."
                    )

        # General recommendations
        recommendations.extend(
            [
                "Integrate real-time weather data into demand forecasting models",
                "Develop weather-contingent promotional strategies",
                "Consider weather insurance for highly weather-sensitive product categories",
                "Implement cross-store inventory balancing based on regional weather patterns",
            ]
        )

        return recommendations[:15]  # Return top 15 recommendations

    # Helper methods

    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def _calculate_temperature_comfort(self, temperatures: pd.Series) -> pd.Series:
        """Calculate temperature comfort index"""
        # Simple comfort index: optimal around 20-25°C
        optimal_temp = 22.5
        return 1 - np.abs(temperatures - optimal_temp) / 30

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        if correlation >= 0.7:
            return "very_strong"
        elif correlation >= 0.5:
            return "strong"
        elif correlation >= 0.3:
            return "moderate"
        elif correlation >= 0.1:
            return "weak"
        else:
            return "negligible"

    def _classify_weather_sensitivity(self, max_correlation: float) -> str:
        """Classify overall weather sensitivity"""
        if max_correlation >= 0.7:
            return WeatherSensitivityLevel.VERY_HIGH.value
        elif max_correlation >= 0.5:
            return WeatherSensitivityLevel.HIGH.value
        elif max_correlation >= 0.3:
            return WeatherSensitivityLevel.MEDIUM.value
        elif max_correlation >= 0.1:
            return WeatherSensitivityLevel.LOW.value
        else:
            return WeatherSensitivityLevel.NONE.value

    def _identify_primary_weather_factor(
        self, weather_correlations: Dict[str, Any]
    ) -> str:
        """Identify the primary weather factor affecting demand"""
        max_correlation = 0
        primary_factor = "temperature"  # default

        for factor, factor_data in weather_correlations.items():
            if isinstance(factor_data, dict):
                for metric, corr_data in factor_data.items():
                    if isinstance(corr_data, dict) and "correlation" in corr_data:
                        if abs(corr_data["correlation"]) > max_correlation:
                            max_correlation = abs(corr_data["correlation"])
                            primary_factor = metric
            elif isinstance(factor_data, dict) and "correlation" in factor_data:
                if abs(factor_data["correlation"]) > max_correlation:
                    max_correlation = abs(factor_data["correlation"])
                    primary_factor = factor

        return primary_factor

    def _identify_most_sensitive_cluster(
        self, clusters: Dict[str, Any]
    ) -> Optional[int]:
        """Identify the most weather-sensitive cluster"""
        max_sensitivity = 0
        most_sensitive_cluster = None

        for cluster_id, cluster_data in clusters.items():
            characteristics = cluster_data.get("characteristics", {})
            sensitivity_level = characteristics.get("weather_sensitivity_level", "none")

            sensitivity_score = {
                "very_high": 4,
                "high": 3,
                "medium": 2,
                "low": 1,
                "none": 0,
            }.get(sensitivity_level, 0)

            if sensitivity_score > max_sensitivity:
                max_sensitivity = sensitivity_score
                most_sensitive_cluster = cluster_id

        return most_sensitive_cluster

    def _analyze_yearly_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze year-over-year weather-demand patterns"""
        data["year"] = data["sale_date"].dt.year
        yearly_patterns = {}

        for year in data["year"].unique():
            year_data = data[data["year"] == year]
            yearly_patterns[str(year)] = {
                "average_sales": float(year_data["sale_amount"].mean()),
                "sales_volatility": float(year_data["sale_amount"].std()),
                "weather_correlation": (
                    float(year_data["sale_amount"].corr(year_data["avg_temperature"]))
                    if "avg_temperature" in year_data.columns
                    else None
                ),
            }

        return yearly_patterns

    def _identify_optimal_weather_conditions(
        self, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Identify optimal weather conditions for maximum sales"""
        optimal_conditions = {}

        if "avg_temperature" in data.columns:
            # Find temperature range with highest average sales
            temp_bins = pd.cut(data["avg_temperature"], bins=10)
            temp_sales = data.groupby(temp_bins)["sale_amount"].mean()
            optimal_temp_bin = temp_sales.idxmax()

            optimal_conditions["optimal_temperature_range"] = (
                f"{optimal_temp_bin.left:.1f}°C - {optimal_temp_bin.right:.1f}°C"
            )
            optimal_conditions["max_sales_at_optimal_temp"] = float(temp_sales.max())

        if "avg_humidity" in data.columns:
            humidity_bins = pd.cut(data["avg_humidity"], bins=10)
            humidity_sales = data.groupby(humidity_bins)["sale_amount"].mean()
            optimal_humidity_bin = humidity_sales.idxmax()

            optimal_conditions["optimal_humidity_range"] = (
                f"{optimal_humidity_bin.left:.1f}% - {optimal_humidity_bin.right:.1f}%"
            )

        return optimal_conditions

    def _generate_weather_scenarios(
        self,
        historical_data: pd.DataFrame,
        weather_features: List[str],
        horizon_days: int,
    ) -> Dict[str, pd.DataFrame]:
        """Generate future weather scenarios for forecasting"""

        scenarios = {}

        # Get recent patterns
        recent_data = historical_data.tail(30)

        # Scenario 1: Historical average
        avg_weather = recent_data[weather_features].mean()
        historical_scenario = pd.DataFrame([avg_weather] * horizon_days)

        # Add temporal features
        future_dates = pd.date_range(
            start=historical_data["sale_date"].max() + timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )

        historical_scenario["day_of_year"] = future_dates.dayofyear
        historical_scenario["is_weekend"] = future_dates.dayofweek.isin([5, 6]).astype(
            int
        )

        scenarios["historical_average"] = historical_scenario

        # Scenario 2: Optimistic weather (based on optimal conditions)
        optimistic_weather = avg_weather.copy()
        if "avg_temperature" in weather_features:
            optimistic_weather["avg_temperature"] = 22.5  # Optimal temperature
        if "avg_humidity" in weather_features:
            optimistic_weather["avg_humidity"] = 50.0  # Moderate humidity
        if "precpt" in weather_features:
            optimistic_weather["precpt"] = 0.0  # No rain

        optimistic_scenario = pd.DataFrame([optimistic_weather] * horizon_days)
        optimistic_scenario["day_of_year"] = future_dates.dayofyear
        optimistic_scenario["is_weekend"] = future_dates.dayofweek.isin([5, 6]).astype(
            int
        )

        scenarios["optimistic"] = optimistic_scenario

        # Scenario 3: Pessimistic weather
        pessimistic_weather = avg_weather.copy()
        if "avg_temperature" in weather_features:
            pessimistic_weather["avg_temperature"] = (
                avg_weather["avg_temperature"]
                + 2 * recent_data["avg_temperature"].std()
            )
        if "precpt" in weather_features:
            pessimistic_weather["precpt"] = recent_data["precpt"].quantile(
                0.9
            )  # High precipitation

        pessimistic_scenario = pd.DataFrame([pessimistic_weather] * horizon_days)
        pessimistic_scenario["day_of_year"] = future_dates.dayofyear
        pessimistic_scenario["is_weekend"] = future_dates.dayofweek.isin([5, 6]).astype(
            int
        )

        scenarios["pessimistic"] = pessimistic_scenario

        return scenarios

    def _calculate_elasticity(
        self, weather_var: pd.Series, demand_var: pd.Series
    ) -> float:
        """Calculate elasticity of demand with respect to weather variable"""
        # Simple elasticity calculation: % change in demand / % change in weather
        weather_pct_change = weather_var.pct_change().dropna()
        demand_pct_change = demand_var.pct_change().dropna()

        # Align series
        min_length = min(len(weather_pct_change), len(demand_pct_change))
        weather_pct_change = weather_pct_change.iloc[:min_length]
        demand_pct_change = demand_pct_change.iloc[:min_length]

        # Calculate elasticity
        if len(weather_pct_change) > 0 and weather_pct_change.std() > 0:
            elasticity = demand_pct_change.corr(weather_pct_change)
            return elasticity if not pd.isna(elasticity) else 0.0
        else:
            return 0.0

    def _interpret_elasticity(self, elasticity: float, weather_factor: str) -> str:
        """Interpret elasticity value"""
        abs_elasticity = abs(elasticity)

        if abs_elasticity > 1:
            sensitivity = "highly elastic"
        elif abs_elasticity > 0.5:
            sensitivity = "moderately elastic"
        elif abs_elasticity > 0.1:
            sensitivity = "slightly elastic"
        else:
            sensitivity = "inelastic"

        direction = "positively" if elasticity > 0 else "negatively"

        return f"Demand is {sensitivity} and {direction} responsive to {weather_factor} changes"

    def _find_optimal_temperature_range(self, data: pd.DataFrame) -> Optional[str]:
        """Find optimal temperature range for sales"""
        if "avg_temperature" not in data.columns:
            return None

        # Group by temperature bins and find optimal range
        temp_bins = pd.cut(data["avg_temperature"], bins=10)
        temp_sales = data.groupby(temp_bins)["sale_amount"].mean()

        if not temp_sales.empty:
            optimal_bin = temp_sales.idxmax()
            return f"{optimal_bin.left:.1f}°C - {optimal_bin.right:.1f}°C"

        return None

    def _analyze_extreme_weather_impact(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact of extreme weather events"""
        extreme_impact = {}

        if "avg_temperature" in data.columns:
            temp_q95 = data["avg_temperature"].quantile(0.95)
            temp_q05 = data["avg_temperature"].quantile(0.05)

            extreme_hot = data[data["avg_temperature"] >= temp_q95]
            extreme_cold = data[data["avg_temperature"] <= temp_q05]
            normal_temp = data[
                (data["avg_temperature"] > temp_q05)
                & (data["avg_temperature"] < temp_q95)
            ]

            if len(extreme_hot) > 0 and len(normal_temp) > 0:
                hot_impact = (
                    extreme_hot["sale_amount"].mean()
                    / normal_temp["sale_amount"].mean()
                    - 1
                ) * 100
                extreme_impact["extreme_heat_impact"] = (
                    f"{hot_impact:.1f}% change in sales"
                )

            if len(extreme_cold) > 0 and len(normal_temp) > 0:
                cold_impact = (
                    extreme_cold["sale_amount"].mean()
                    / normal_temp["sale_amount"].mean()
                    - 1
                ) * 100
                extreme_impact["extreme_cold_impact"] = (
                    f"{cold_impact:.1f}% change in sales"
                )

        if "precpt" in data.columns:
            heavy_rain = data[data["precpt"] > data["precpt"].quantile(0.9)]
            no_rain = data[data["precpt"] == 0]

            if len(heavy_rain) > 0 and len(no_rain) > 0:
                rain_impact = (
                    heavy_rain["sale_amount"].mean() / no_rain["sale_amount"].mean() - 1
                ) * 100
                extreme_impact["heavy_rain_impact"] = (
                    f"{rain_impact:.1f}% change in sales"
                )

        return extreme_impact

    def _classify_product_weather_sensitivity(
        self, elasticities: Dict[str, Any]
    ) -> str:
        """Classify product's overall weather sensitivity"""
        max_elasticity = 0

        for factor_data in elasticities.values():
            if isinstance(factor_data, dict) and "elasticity" in factor_data:
                max_elasticity = max(max_elasticity, abs(factor_data["elasticity"]))

        if max_elasticity > 1:
            return "highly_elastic"
        elif max_elasticity > 0.5:
            return "moderately_elastic"
        elif max_elasticity > 0.1:
            return "slightly_elastic"
        else:
            return "inelastic"


# Export the service
enhanced_weather_service = EnhancedWeatherService()

__all__ = [
    "EnhancedWeatherService",
    "WeatherImpactRequest",
    "WeatherImpactResult",
    "WeatherSensitivityLevel",
    "WeatherFactor",
    "enhanced_weather_service",
]
