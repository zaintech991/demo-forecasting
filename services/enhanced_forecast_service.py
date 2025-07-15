# type: ignore
"""
Enhanced Multi-Dimensional Forecasting Service for FreshRetailNet-50K
Supports cross-store comparison, product correlation, portfolio forecasting, and ensemble models
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

# Import ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Import our database manager
from database.connection import db_manager, get_db_connection
from models.prophet_forecaster import ProphetForecaster

logger = logging.getLogger(__name__)


class ForecastingMethod(Enum):
    """Enumeration of available forecasting methods"""

    PROPHET = "prophet"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"
    NAIVE = "naive"


@dataclass
class ForecastRequest:
    """Enhanced forecast request with multi-dimensional capabilities"""

    store_ids: List[int]
    product_ids: List[int]
    forecast_horizon_days: int = 30
    include_confidence_intervals: bool = True
    include_cross_store_analysis: bool = True
    include_product_correlations: bool = True
    include_weather_factors: bool = True
    include_promotion_factors: bool = True
    forecasting_method: ForecastingMethod = ForecastingMethod.ENSEMBLE
    confidence_level: float = 0.95


@dataclass
class ForecastResult:
    """Enhanced forecast result with comparative analysis"""

    forecast_data: pd.DataFrame
    confidence_intervals: Dict[str, pd.DataFrame]
    model_metrics: Dict[str, float]
    cross_store_comparison: Optional[Dict[str, Any]] = None
    product_correlations: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    recommendations: Optional[List[str]] = None


class EnhancedForecastService:
    """
    Enhanced forecasting service with multi-dimensional analysis capabilities
    """

    def __init__(self):
        self.model_cache: Dict[str, Any] = {}
        self.scaler = StandardScaler()

    async def generate_multi_dimensional_forecast(
        self, request: ForecastRequest
    ) -> ForecastResult:
        """
        Generate multi-dimensional forecast with comparative analysis

        Args:
            request: Enhanced forecast request with multi-dimensional parameters

        Returns:
            ForecastResult with comprehensive analysis
        """
        logger.info(
            f"Generating multi-dimensional forecast for stores: {request.store_ids}, products: {request.product_ids}"
        )

        # 1. Get historical data
        historical_data = await self._get_comprehensive_historical_data(request)

        if historical_data.empty:
            raise ValueError(
                "No historical data available for the specified parameters"
            )

        # 2. Generate base forecasts
        base_forecasts = await self._generate_base_forecasts(historical_data, request)

        # 3. Perform cross-store analysis if requested
        cross_store_analysis = None
        if request.include_cross_store_analysis and len(request.store_ids) > 1:
            cross_store_analysis = await self._perform_cross_store_analysis(
                historical_data, base_forecasts, request
            )

        # 4. Analyze product correlations if requested
        product_correlations = None
        if request.include_product_correlations and len(request.product_ids) > 1:
            product_correlations = await self._analyze_product_correlations(
                historical_data, request
            )

        # 5. Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            base_forecasts, request.confidence_level
        )

        # 6. Calculate model metrics
        model_metrics = self._calculate_model_metrics(historical_data, base_forecasts)

        # 7. Extract feature importance
        feature_importance = self._extract_feature_importance(historical_data)

        # 8. Generate recommendations
        recommendations = self._generate_recommendations(
            base_forecasts, cross_store_analysis, product_correlations, model_metrics
        )

        return ForecastResult(
            forecast_data=base_forecasts,
            confidence_intervals=confidence_intervals,
            model_metrics=model_metrics,
            cross_store_comparison=cross_store_analysis,
            product_correlations=product_correlations,
            feature_importance=feature_importance,
            recommendations=recommendations,
        )

    async def _get_comprehensive_historical_data(
        self, request: ForecastRequest
    ) -> pd.DataFrame:
        """Get comprehensive historical data including all relevant factors"""

        # Get base sales data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of history

        sales_data = await db_manager.get_sales_data(
            store_ids=request.store_ids,
            product_ids=request.product_ids,
            start_date=start_date,
            end_date=end_date,
            include_hourly=True,
            include_stockouts=True,
        )

        if sales_data.empty:
            return sales_data

        # Enrich with additional data
        if request.include_weather_factors:
            weather_data = await db_manager.get_weather_impact_data(
                product_ids=request.product_ids, include_sales_data=False
            )
            if not weather_data.empty:
                sales_data = sales_data.merge(
                    weather_data[
                        [
                            "product_id",
                            "city_id",
                            "temperature_elasticity",
                            "humidity_impact",
                            "precipitation_impact",
                        ]
                    ],
                    on=["product_id", "city_id"],
                    how="left",
                )

        if request.include_promotion_factors:
            promotion_data = await db_manager.get_promotion_effectiveness(
                store_ids=request.store_ids,
                product_ids=request.product_ids,
                start_date=start_date,
                end_date=end_date,
            )
            if not promotion_data.empty:
                # Aggregate promotion data by store/product/date
                promo_agg = (
                    promotion_data.groupby(["store_id", "product_id"])
                    .agg({"uplift_percentage": "mean", "roi": "mean"})
                    .reset_index()
                )

                sales_data = sales_data.merge(
                    promo_agg, on=["store_id", "product_id"], how="left"
                )

        # Add time-based features
        sales_data["sale_date"] = pd.to_datetime(sales_data["sale_date"])
        sales_data["day_of_week"] = sales_data["sale_date"].dt.dayofweek
        sales_data["month"] = sales_data["sale_date"].dt.month
        sales_data["quarter"] = sales_data["sale_date"].dt.quarter
        sales_data["is_weekend"] = sales_data["day_of_week"].isin([5, 6]).astype(int)
        sales_data["days_since_start"] = (
            sales_data["sale_date"] - sales_data["sale_date"].min()
        ).dt.days

        # Add lag features
        for store_id in request.store_ids:
            for product_id in request.product_ids:
                mask = (sales_data["store_id"] == store_id) & (
                    sales_data["product_id"] == product_id
                )
                if mask.sum() > 0:
                    # Sort by date for lag calculations
                    store_product_data = sales_data[mask].sort_values("sale_date")

                    # Add lag features
                    sales_data.loc[mask, "lag_1"] = store_product_data[
                        "sale_amount"
                    ].shift(1)
                    sales_data.loc[mask, "lag_7"] = store_product_data[
                        "sale_amount"
                    ].shift(7)
                    sales_data.loc[mask, "lag_30"] = store_product_data[
                        "sale_amount"
                    ].shift(30)

                    # Add rolling averages
                    sales_data.loc[mask, "rolling_7_mean"] = (
                        store_product_data["sale_amount"].rolling(7).mean()
                    )
                    sales_data.loc[mask, "rolling_30_mean"] = (
                        store_product_data["sale_amount"].rolling(30).mean()
                    )

                    # Add trend features
                    sales_data.loc[mask, "trend_7"] = (
                        store_product_data["sale_amount"].rolling(7).mean()
                        - store_product_data["sale_amount"].rolling(14).mean()
                    )

        return sales_data.fillna(0)

    async def _generate_base_forecasts(
        self, historical_data: pd.DataFrame, request: ForecastRequest
    ) -> pd.DataFrame:
        """Generate base forecasts using the specified method"""

        forecasts = []

        for store_id in request.store_ids:
            for product_id in request.product_ids:
                # Filter data for this store-product combination
                store_product_data = historical_data[
                    (historical_data["store_id"] == store_id)
                    & (historical_data["product_id"] == product_id)
                ].copy()

                if len(store_product_data) < 30:  # Need at least 30 days of data
                    logger.warning(
                        f"Insufficient data for store {store_id}, product {product_id}"
                    )
                    continue

                # Generate forecast based on method
                if request.forecasting_method == ForecastingMethod.PROPHET:
                    forecast = await self._generate_prophet_forecast(
                        store_product_data, request.forecast_horizon_days
                    )
                elif request.forecasting_method == ForecastingMethod.RANDOM_FOREST:
                    forecast = self._generate_rf_forecast(
                        store_product_data, request.forecast_horizon_days
                    )
                elif request.forecasting_method == ForecastingMethod.ENSEMBLE:
                    forecast = await self._generate_ensemble_forecast(
                        store_product_data, request.forecast_horizon_days
                    )
                else:  # NAIVE
                    forecast = self._generate_naive_forecast(
                        store_product_data, request.forecast_horizon_days
                    )

                # Add store and product identifiers
                forecast["store_id"] = store_id
                forecast["product_id"] = product_id

                forecasts.append(forecast)

        if not forecasts:
            return pd.DataFrame()

        return pd.concat(forecasts, ignore_index=True)

    async def _generate_prophet_forecast(
        self, data: pd.DataFrame, horizon_days: int
    ) -> pd.DataFrame:
        """Generate Prophet-based forecast"""

        try:
            # Prepare data for Prophet
            prophet_data = data[["sale_date", "sale_amount"]].copy()
            prophet_data.columns = ["ds", "y"]
            prophet_data = prophet_data.sort_values("ds")

            # Initialize Prophet model
            forecaster = ProphetForecaster(
                include_weather=True, include_holidays=True, include_promotions=True
            )

            # Add regressors if available
            if "avg_temperature" in data.columns:
                prophet_data["avg_temperature"] = data["avg_temperature"].values
            if "avg_humidity" in data.columns:
                prophet_data["avg_humidity"] = data["avg_humidity"].values
            if "discount" in data.columns:
                prophet_data["discount"] = data["discount"].values

            # Train model
            forecaster.train(prophet_data)

            # Generate forecast
            forecast_result = forecaster.predict(periods=horizon_days, freq="D")

            # Convert to standard format
            forecast_df = pd.DataFrame(
                {
                    "forecast_date": forecast_result["ds"],
                    "predicted_demand": forecast_result["yhat"],
                    "confidence_lower": forecast_result["yhat_lower"],
                    "confidence_upper": forecast_result["yhat_upper"],
                    "model_type": "prophet",
                }
            )

            return forecast_df

        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
            # Fall back to naive forecast
            return self._generate_naive_forecast(data, horizon_days)

    def _generate_rf_forecast(
        self, data: pd.DataFrame, horizon_days: int
    ) -> pd.DataFrame:
        """Generate Random Forest-based forecast"""

        try:
            # Prepare features
            feature_columns = [
                "day_of_week",
                "month",
                "quarter",
                "is_weekend",
                "days_since_start",
                "avg_temperature",
                "avg_humidity",
                "precpt",
                "discount",
                "holiday_flag",
                "lag_1",
                "lag_7",
                "lag_30",
                "rolling_7_mean",
                "rolling_30_mean",
                "trend_7",
            ]

            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]

            if len(available_features) < 3:
                return self._generate_naive_forecast(data, horizon_days)

            # Prepare training data
            train_data = data.dropna(subset=available_features + ["sale_amount"])

            if len(train_data) < 30:
                return self._generate_naive_forecast(data, horizon_days)

            X = train_data[available_features]
            y = train_data["sale_amount"]

            # Train Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

            rf_model.fit(X, y)

            # Generate future features
            last_date = data["sale_date"].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1), periods=horizon_days, freq="D"
            )

            future_features = []
            for i, future_date in enumerate(future_dates):
                features = {}
                features["day_of_week"] = future_date.dayofweek
                features["month"] = future_date.month
                features["quarter"] = future_date.quarter
                features["is_weekend"] = int(future_date.dayofweek in [5, 6])
                features["days_since_start"] = (
                    future_date - data["sale_date"].min()
                ).days

                # Use last known values for other features
                last_row = data.iloc[-1]
                for feature in available_features:
                    if feature not in features:
                        features[feature] = last_row.get(feature, 0)

                future_features.append(features)

            future_df = pd.DataFrame(future_features)
            future_X = future_df[available_features]

            # Generate predictions
            predictions = rf_model.predict(future_X)

            # Estimate confidence intervals (using prediction variance)
            predictions_std = np.std(
                [tree.predict(future_X) for tree in rf_model.estimators_], axis=0
            )
            confidence_lower = predictions - 1.96 * predictions_std
            confidence_upper = predictions + 1.96 * predictions_std

            forecast_df = pd.DataFrame(
                {
                    "forecast_date": future_dates,
                    "predicted_demand": predictions,
                    "confidence_lower": confidence_lower,
                    "confidence_upper": confidence_upper,
                    "model_type": "random_forest",
                }
            )

            return forecast_df

        except Exception as e:
            logger.error(f"Random Forest forecast failed: {e}")
            return self._generate_naive_forecast(data, horizon_days)

    async def _generate_ensemble_forecast(
        self, data: pd.DataFrame, horizon_days: int
    ) -> pd.DataFrame:
        """Generate ensemble forecast combining multiple methods"""

        try:
            # Generate forecasts from different methods
            prophet_forecast = await self._generate_prophet_forecast(data, horizon_days)
            rf_forecast = self._generate_rf_forecast(data, horizon_days)
            naive_forecast = self._generate_naive_forecast(data, horizon_days)

            # Combine forecasts with weights
            weights = {"prophet": 0.5, "random_forest": 0.3, "naive": 0.2}

            ensemble_predictions = (
                weights["prophet"] * prophet_forecast["predicted_demand"]
                + weights["random_forest"] * rf_forecast["predicted_demand"]
                + weights["naive"] * naive_forecast["predicted_demand"]
            )

            # Combine confidence intervals
            ensemble_lower = (
                weights["prophet"] * prophet_forecast["confidence_lower"]
                + weights["random_forest"] * rf_forecast["confidence_lower"]
                + weights["naive"] * naive_forecast["confidence_lower"]
            )

            ensemble_upper = (
                weights["prophet"] * prophet_forecast["confidence_upper"]
                + weights["random_forest"] * rf_forecast["confidence_upper"]
                + weights["naive"] * naive_forecast["confidence_upper"]
            )

            forecast_df = pd.DataFrame(
                {
                    "forecast_date": prophet_forecast["forecast_date"],
                    "predicted_demand": ensemble_predictions,
                    "confidence_lower": ensemble_lower,
                    "confidence_upper": ensemble_upper,
                    "model_type": "ensemble",
                }
            )

            return forecast_df

        except Exception as e:
            logger.error(f"Ensemble forecast failed: {e}")
            return self._generate_naive_forecast(data, horizon_days)

    def _generate_naive_forecast(
        self, data: pd.DataFrame, horizon_days: int
    ) -> pd.DataFrame:
        """Generate naive forecast based on recent averages"""

        # Use last 30 days average as forecast
        recent_data = data.tail(30)
        avg_sales = recent_data["sale_amount"].mean()
        std_sales = recent_data["sale_amount"].std()

        # Generate future dates
        last_date = data["sale_date"].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=horizon_days, freq="D"
        )

        # Add some seasonal adjustment based on day of week
        predictions = []
        for future_date in future_dates:
            # Simple day-of-week adjustment
            dow_factor = 1.0
            if future_date.dayofweek in [5, 6]:  # Weekend
                dow_factor = 0.8  # Assume lower weekend sales

            predictions.append(avg_sales * dow_factor)

        forecast_df = pd.DataFrame(
            {
                "forecast_date": future_dates,
                "predicted_demand": predictions,
                "confidence_lower": [max(0, p - 1.96 * std_sales) for p in predictions],
                "confidence_upper": [p + 1.96 * std_sales for p in predictions],
                "model_type": "naive",
            }
        )

        return forecast_df

    async def _perform_cross_store_analysis(
        self,
        historical_data: pd.DataFrame,
        forecasts: pd.DataFrame,
        request: ForecastRequest,
    ) -> Dict[str, Any]:
        """Perform cross-store comparative analysis"""

        cross_store_analysis = {}

        for product_id in request.product_ids:
            product_analysis = {}

            # Get forecasts for this product across all stores
            product_forecasts = forecasts[forecasts["product_id"] == product_id]

            if product_forecasts.empty:
                continue

            # Calculate store performance metrics
            store_metrics = []
            for store_id in request.store_ids:
                store_forecast = product_forecasts[
                    product_forecasts["store_id"] == store_id
                ]
                store_historical = historical_data[
                    (historical_data["store_id"] == store_id)
                    & (historical_data["product_id"] == product_id)
                ]

                if store_forecast.empty or store_historical.empty:
                    continue

                # Calculate metrics
                total_predicted = store_forecast["predicted_demand"].sum()
                avg_historical = store_historical["sale_amount"].mean()
                volatility = (
                    store_historical["sale_amount"].std() / avg_historical
                    if avg_historical > 0
                    else 0
                )
                stockout_rate = (
                    store_historical["stock_hour6_22_cnt"].mean() / 16
                )  # 16 hours (6-22)

                store_metrics.append(
                    {
                        "store_id": store_id,
                        "total_predicted_demand": total_predicted,
                        "avg_historical_demand": avg_historical,
                        "demand_volatility": volatility,
                        "stockout_rate": stockout_rate,
                        "growth_potential": (
                            total_predicted / avg_historical - 1
                            if avg_historical > 0
                            else 0
                        ),
                    }
                )

            if store_metrics:
                store_metrics_df = pd.DataFrame(store_metrics)

                # Rank stores
                store_metrics_df["demand_rank"] = store_metrics_df[
                    "total_predicted_demand"
                ].rank(ascending=False)
                store_metrics_df["reliability_rank"] = store_metrics_df[
                    "stockout_rate"
                ].rank(ascending=True)
                store_metrics_df["growth_rank"] = store_metrics_df[
                    "growth_potential"
                ].rank(ascending=False)

                # Calculate overall score
                store_metrics_df["overall_score"] = (
                    store_metrics_df["demand_rank"] * 0.4
                    + store_metrics_df["reliability_rank"] * 0.3
                    + store_metrics_df["growth_rank"] * 0.3
                )

                product_analysis["store_rankings"] = store_metrics_df.to_dict("records")
                product_analysis["best_performing_store"] = store_metrics_df.loc[
                    store_metrics_df["overall_score"].idxmin(), "store_id"
                ]
                product_analysis["highest_demand_store"] = store_metrics_df.loc[
                    store_metrics_df["total_predicted_demand"].idxmax(), "store_id"
                ]
                product_analysis["most_reliable_store"] = store_metrics_df.loc[
                    store_metrics_df["stockout_rate"].idxmin(), "store_id"
                ]

            cross_store_analysis[f"product_{product_id}"] = product_analysis

        return cross_store_analysis

    async def _analyze_product_correlations(
        self, historical_data: pd.DataFrame, request: ForecastRequest
    ) -> Dict[str, Any]:
        """Analyze product correlations for cross-selling opportunities"""

        correlations = {}

        for store_id in request.store_ids:
            store_data = historical_data[historical_data["store_id"] == store_id]

            if store_data.empty:
                continue

            # Pivot data for correlation analysis
            pivot_data = store_data.pivot_table(
                index="sale_date",
                columns="product_id",
                values="sale_amount",
                fill_value=0,
            )

            if pivot_data.shape[1] < 2:  # Need at least 2 products
                continue

            # Calculate correlation matrix
            correlation_matrix = pivot_data.corr()

            # Extract significant correlations
            significant_correlations = []
            for i, product_a in enumerate(correlation_matrix.index):
                for j, product_b in enumerate(correlation_matrix.columns):
                    if i >= j:  # Avoid duplicates and self-correlations
                        continue

                    correlation_coef = correlation_matrix.loc[product_a, product_b]

                    if (
                        abs(correlation_coef) >= 0.3
                    ):  # Significant correlation threshold
                        correlation_type = (
                            "complementary" if correlation_coef > 0 else "substitute"
                        )

                        significant_correlations.append(
                            {
                                "product_a": int(product_a),
                                "product_b": int(product_b),
                                "correlation": correlation_coef,
                                "correlation_type": correlation_type,
                                "strength": (
                                    "strong"
                                    if abs(correlation_coef) > 0.7
                                    else "moderate"
                                ),
                            }
                        )

            correlations[f"store_{store_id}"] = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "significant_correlations": significant_correlations,
            }

        return correlations

    def _calculate_confidence_intervals(
        self, forecasts: pd.DataFrame, confidence_level: float
    ) -> Dict[str, pd.DataFrame]:
        """Calculate confidence intervals for forecasts"""

        confidence_intervals = {}

        for store_id in forecasts["store_id"].unique():
            for product_id in forecasts["product_id"].unique():
                key = f"store_{store_id}_product_{product_id}"

                store_product_forecast = forecasts[
                    (forecasts["store_id"] == store_id)
                    & (forecasts["product_id"] == product_id)
                ]

                if not store_product_forecast.empty:
                    confidence_intervals[key] = store_product_forecast[
                        ["forecast_date", "confidence_lower", "confidence_upper"]
                    ].copy()

        return confidence_intervals

    def _calculate_model_metrics(
        self, historical_data: pd.DataFrame, forecasts: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate model performance metrics"""

        # For demonstration, calculate simple metrics
        # In practice, you'd use holdout data for validation

        metrics = {
            "forecast_coverage": len(forecasts)
            / (
                len(historical_data["store_id"].unique())
                * len(historical_data["product_id"].unique())
            ),
            "avg_predicted_demand": forecasts["predicted_demand"].mean(),
            "prediction_variance": forecasts["predicted_demand"].var(),
            "confidence_interval_width": (
                forecasts["confidence_upper"] - forecasts["confidence_lower"]
            ).mean(),
        }

        return metrics

    def _extract_feature_importance(
        self, historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract feature importance from historical patterns"""

        # Calculate feature importance based on correlation with sales
        feature_importance = {}

        numeric_columns = historical_data.select_dtypes(include=[np.number]).columns
        target_column = "sale_amount"

        if target_column in numeric_columns:
            for column in numeric_columns:
                if column != target_column and not historical_data[column].isna().all():
                    correlation = historical_data[column].corr(
                        historical_data[target_column]
                    )
                    if not pd.isna(correlation):
                        feature_importance[column] = abs(correlation)

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    def _generate_recommendations(
        self,
        forecasts: pd.DataFrame,
        cross_store_analysis: Optional[Dict[str, Any]],
        product_correlations: Optional[Dict[str, Any]],
        model_metrics: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""

        recommendations = []

        # Forecast-based recommendations
        if not forecasts.empty:
            avg_demand = forecasts["predicted_demand"].mean()
            max_demand = forecasts["predicted_demand"].max()
            min_demand = forecasts["predicted_demand"].min()

            if max_demand > 2 * avg_demand:
                recommendations.append(
                    f"High demand variation detected (max: {max_demand:.1f}, avg: {avg_demand:.1f}). "
                    "Consider dynamic inventory allocation."
                )

            if min_demand < 0.5 * avg_demand:
                recommendations.append(
                    f"Low demand periods identified (min: {min_demand:.1f}). "
                    "Consider promotional activities during these periods."
                )

        # Cross-store recommendations
        if cross_store_analysis:
            for product_key, analysis in cross_store_analysis.items():
                if "best_performing_store" in analysis:
                    recommendations.append(
                        f"Store {analysis['best_performing_store']} shows best overall performance for {product_key}. "
                        "Consider replicating successful practices to other stores."
                    )

        # Product correlation recommendations
        if product_correlations:
            for store_key, corr_data in product_correlations.items():
                strong_correlations = [
                    corr
                    for corr in corr_data.get("significant_correlations", [])
                    if corr["strength"] == "strong"
                    and corr["correlation_type"] == "complementary"
                ]

                if strong_correlations:
                    recommendations.append(
                        f"Strong product complementarity detected in {store_key}. "
                        "Consider bundling strategies for cross-selling opportunities."
                    )

        # Model quality recommendations
        if model_metrics.get("confidence_interval_width", 0) > avg_demand * 0.5:
            recommendations.append(
                "High prediction uncertainty detected. Consider gathering more historical data "
                "or additional features for improved accuracy."
            )

        return recommendations[:10]  # Return top 10 recommendations


# Export the service
enhanced_forecast_service = EnhancedForecastService()

__all__ = [
    "EnhancedForecastService",
    "ForecastRequest",
    "ForecastResult",
    "ForecastingMethod",
    "enhanced_forecast_service",
]
