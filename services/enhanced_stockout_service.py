# type: ignore
"""
Enhanced Stockout Risk Assessment Service for FreshRetailNet-50K
Includes hourly prediction, cross-store optimization, and real-time monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

from database.connection import db_manager

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Stockout risk levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class StockoutAssessmentRequest:
    """Request model for stockout assessment"""

    store_ids: List[int]
    product_ids: List[int]
    assessment_horizon_days: int = 14
    include_hourly_prediction: bool = True
    include_cross_store_optimization: bool = True
    include_seasonal_factors: bool = True
    target_service_level: float = 0.95


@dataclass
class StockoutAssessmentResult:
    """Result model for stockout assessment"""

    risk_assessments: Dict[str, Any]
    hourly_predictions: Optional[Dict[str, Any]] = None
    cross_store_optimization: Optional[Dict[str, Any]] = None
    reorder_recommendations: Optional[List[Dict[str, Any]]] = None
    early_warnings: Optional[List[Dict[str, Any]]] = None


class EnhancedStockoutService:
    """Enhanced stockout risk assessment service"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.risk_models: Dict[str, Any] = {}

    async def assess_stockout_risk(
        self, request: StockoutAssessmentRequest
    ) -> StockoutAssessmentResult:
        """Comprehensive stockout risk assessment"""

        logger.info(f"Assessing stockout risk for {len(request.product_ids)} products")

        # Get comprehensive data
        historical_data = await self._get_stockout_historical_data(request)

        if historical_data.empty:
            raise ValueError("No historical data available")

        # Assess current risk levels
        risk_assessments = await self._assess_current_risk_levels(
            historical_data, request
        )

        # Hourly predictions
        hourly_predictions = None
        if request.include_hourly_prediction:
            hourly_predictions = await self._predict_hourly_stockouts(
                historical_data, request
            )

        # Cross-store optimization
        cross_store_optimization = None
        if request.include_cross_store_optimization:
            cross_store_optimization = await self._optimize_cross_store_inventory(
                historical_data, request
            )

        # Generate reorder recommendations
        reorder_recommendations = self._generate_reorder_recommendations(
            risk_assessments, request
        )

        # Generate early warnings
        early_warnings = self._generate_early_warnings(risk_assessments, request)

        return StockoutAssessmentResult(
            risk_assessments=risk_assessments,
            hourly_predictions=hourly_predictions,
            cross_store_optimization=cross_store_optimization,
            reorder_recommendations=reorder_recommendations,
            early_warnings=early_warnings,
        )

    async def _get_stockout_historical_data(
        self, request: StockoutAssessmentRequest
    ) -> pd.DataFrame:
        """Get historical stockout and sales data"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months of history

        # Get sales data with stockout information
        sales_data = await db_manager.get_sales_data(
            store_ids=request.store_ids,
            product_ids=request.product_ids,
            start_date=start_date,
            end_date=end_date,
            include_hourly=True,
            include_stockouts=True,
        )

        # Get stockout events
        stockout_events = await db_manager.get_stockout_events(
            store_ids=request.store_ids,
            product_ids=request.product_ids,
            start_date=start_date,
            end_date=end_date,
        )

        # Merge stockout event data
        if not stockout_events.empty:
            stockout_events["stockout_date"] = pd.to_datetime(
                stockout_events["stockout_start"]
            ).dt.date
            stockout_summary = (
                stockout_events.groupby(["store_id", "product_id", "stockout_date"])
                .agg({"duration_hours": "sum", "estimated_lost_sales": "sum"})
                .reset_index()
            )

            sales_data = sales_data.merge(
                stockout_summary,
                left_on=["store_id", "product_id", "sale_date"],
                right_on=["store_id", "product_id", "stockout_date"],
                how="left",
            )

        # Add features for risk prediction
        sales_data = self._add_risk_features(sales_data)

        return sales_data

    def _add_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features for stockout risk prediction"""

        # Convert stockout hours to binary flag
        data["had_stockout"] = (data["stock_hour6_22_cnt"] > 0).astype(int)

        # Calculate demand volatility
        for store_id in data["store_id"].unique():
            for product_id in data["product_id"].unique():
                mask = (data["store_id"] == store_id) & (
                    data["product_id"] == product_id
                )
                if mask.sum() > 7:  # Need at least a week of data
                    demand_series = data.loc[mask, "sale_amount"]
                    # 7-day rolling volatility
                    volatility = demand_series.rolling(7).std()
                    data.loc[mask, "demand_volatility"] = volatility

        # Add temporal features
        data["sale_date"] = pd.to_datetime(data["sale_date"])
        data["day_of_week"] = data["sale_date"].dt.dayofweek
        data["month"] = data["sale_date"].dt.month
        data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
        data["is_month_end"] = (data["sale_date"].dt.day > 25).astype(int)

        # Add lag features
        for store_id in data["store_id"].unique():
            for product_id in data["product_id"].unique():
                mask = (data["store_id"] == store_id) & (
                    data["product_id"] == product_id
                )
                if mask.sum() > 0:
                    store_product_data = data[mask].sort_values("sale_date")

                    # Demand lags
                    data.loc[mask, "demand_lag_1"] = store_product_data[
                        "sale_amount"
                    ].shift(1)
                    data.loc[mask, "demand_lag_7"] = store_product_data[
                        "sale_amount"
                    ].shift(7)

                    # Stockout lags
                    data.loc[mask, "stockout_lag_1"] = store_product_data[
                        "had_stockout"
                    ].shift(1)
                    data.loc[mask, "stockout_lag_7"] = store_product_data[
                        "had_stockout"
                    ].shift(7)

                    # Trend features
                    data.loc[mask, "demand_trend_7"] = (
                        store_product_data["sale_amount"].rolling(7).mean().pct_change()
                    )

        return data.fillna(0)

    async def _assess_current_risk_levels(
        self, data: pd.DataFrame, request: StockoutAssessmentRequest
    ) -> Dict[str, Any]:
        """Assess current stockout risk levels"""

        risk_assessments = {}

        for store_id in request.store_ids:
            for product_id in request.product_ids:
                store_product_data = data[
                    (data["store_id"] == store_id) & (data["product_id"] == product_id)
                ].copy()

                if len(store_product_data) < 30:
                    continue

                # Calculate risk metrics
                recent_data = store_product_data.tail(14)  # Last 2 weeks

                # Historical stockout frequency
                stockout_frequency = store_product_data["had_stockout"].mean()

                # Recent demand trend
                recent_trend = recent_data["sale_amount"].pct_change().mean()

                # Demand volatility
                demand_volatility = (
                    recent_data["sale_amount"].std() / recent_data["sale_amount"].mean()
                    if recent_data["sale_amount"].mean() > 0
                    else 0
                )

                # Current stock level proxy (inverse of recent stockout hours)
                current_stock_proxy = 1 - (
                    recent_data["stock_hour6_22_cnt"].mean() / 16
                )

                # Lead time risk (simplified)
                lead_time_risk = self._calculate_lead_time_risk(store_product_data)

                # Composite risk score
                risk_score = (
                    stockout_frequency * 0.3
                    + (1 - current_stock_proxy) * 0.25
                    + demand_volatility * 0.2
                    + max(0, recent_trend) * 0.15  # Positive trend increases risk
                    + lead_time_risk * 0.1
                )

                # Classify risk level
                risk_level = self._classify_risk_level(risk_score)

                # Calculate recommended safety stock
                safety_stock = self._calculate_safety_stock(
                    store_product_data, request.target_service_level
                )

                risk_assessments[f"store_{store_id}_product_{product_id}"] = {
                    "risk_score": float(risk_score),
                    "risk_level": risk_level.value,
                    "stockout_frequency": float(stockout_frequency),
                    "demand_volatility": float(demand_volatility),
                    "recent_trend": float(recent_trend),
                    "current_stock_proxy": float(current_stock_proxy),
                    "lead_time_risk": float(lead_time_risk),
                    "recommended_safety_stock": float(safety_stock),
                    "days_until_predicted_stockout": self._predict_days_until_stockout(
                        store_product_data
                    ),
                }

        return risk_assessments

    async def _predict_hourly_stockouts(
        self, data: pd.DataFrame, request: StockoutAssessmentRequest
    ) -> Dict[str, Any]:
        """Predict hourly stockout probabilities"""

        hourly_predictions = {}

        # Features for hourly prediction
        hourly_features = [
            "hour_of_day",
            "day_of_week",
            "demand_volatility",
            "demand_lag_1",
            "stockout_lag_1",
            "avg_temperature",
            "holiday_flag",
        ]

        available_features = [f for f in hourly_features if f in data.columns]

        if len(available_features) < 3:
            return {"error": "Insufficient features for hourly prediction"}

        for store_id in request.store_ids:
            for product_id in request.product_ids:
                store_product_data = data[
                    (data["store_id"] == store_id) & (data["product_id"] == product_id)
                ].copy()

                if len(store_product_data) < 50:
                    continue

                try:
                    # Prepare training data
                    X = store_product_data[available_features].fillna(0)
                    y = store_product_data["had_stockout"]

                    # Train hourly stockout classifier
                    rf_classifier = RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight="balanced"
                    )

                    rf_classifier.fit(X, y)

                    # Generate hourly predictions for next 7 days
                    future_hours = []
                    base_date = data["sale_date"].max() + timedelta(days=1)

                    for day in range(7):
                        for hour in range(24):
                            future_datetime = base_date + timedelta(
                                days=day, hours=hour
                            )

                            # Create feature vector for this hour
                            hour_features = {
                                "hour_of_day": hour,
                                "day_of_week": future_datetime.weekday(),
                                "holiday_flag": 0,  # Simplified
                            }

                            # Use recent values for other features
                            recent_data = store_product_data.tail(1).iloc[0]
                            for feature in available_features:
                                if feature not in hour_features:
                                    hour_features[feature] = recent_data.get(
                                        feature, 0.0
                                    )

                            # Predict stockout probability
                            feature_vector = [
                                hour_features.get(f, 0) for f in available_features
                            ]
                            stockout_prob = rf_classifier.predict_proba(
                                [feature_vector]
                            )[0][1]

                            future_hours.append(
                                {
                                    "datetime": future_datetime.isoformat(),
                                    "hour": hour,
                                    "day": day + 1,
                                    "stockout_probability": float(stockout_prob),
                                    "risk_level": self._classify_hourly_risk(
                                        stockout_prob
                                    ),
                                }
                            )

                    hourly_predictions[f"store_{store_id}_product_{product_id}"] = {
                        "predictions": future_hours,
                        "model_features": available_features,
                        "feature_importance": dict(
                            zip(available_features, rf_classifier.feature_importances_)
                        ),
                    }

                except Exception as e:
                    logger.warning(
                        f"Hourly prediction failed for store {store_id}, product {product_id}: {e}"
                    )
                    continue

        return hourly_predictions

    async def _optimize_cross_store_inventory(
        self, data: pd.DataFrame, request: StockoutAssessmentRequest
    ) -> Dict[str, Any]:
        """Optimize inventory allocation across stores"""

        optimization = {}

        for product_id in request.product_ids:
            product_data = data[data["product_id"] == product_id]

            if len(product_data) < 30:
                continue

            store_metrics = []

            for store_id in request.store_ids:
                store_product_data = product_data[product_data["store_id"] == store_id]

                if len(store_product_data) < 10:
                    continue

                # Calculate store metrics
                avg_demand = store_product_data["sale_amount"].mean()
                demand_volatility = store_product_data["sale_amount"].std()
                stockout_frequency = store_product_data["had_stockout"].mean()
                recent_trend = (
                    store_product_data.tail(7)["sale_amount"].pct_change().mean()
                )

                store_metrics.append(
                    {
                        "store_id": store_id,
                        "avg_demand": avg_demand,
                        "demand_volatility": demand_volatility,
                        "stockout_frequency": stockout_frequency,
                        "recent_trend": recent_trend,
                        "risk_score": stockout_frequency
                        + (demand_volatility / avg_demand if avg_demand > 0 else 0),
                    }
                )

            if store_metrics:
                store_metrics_df = pd.DataFrame(store_metrics)

                # Calculate optimal allocation
                total_demand = store_metrics_df["avg_demand"].sum()

                for idx, row in store_metrics_df.iterrows():
                    # Adjust allocation based on risk
                    base_allocation = row["avg_demand"] / total_demand
                    risk_adjustment = 1 + (
                        row["risk_score"] * 0.2
                    )  # Increase allocation for high-risk stores
                    adjusted_allocation = base_allocation * risk_adjustment

                    store_metrics_df.loc[idx, "recommended_allocation_percentage"] = (  # type: ignore
                        adjusted_allocation * 100
                    )

                # Normalize allocations to sum to 100%
                total_allocation = store_metrics_df[
                    "recommended_allocation_percentage"
                ].sum()
                store_metrics_df["recommended_allocation_percentage"] *= (
                    100 / total_allocation
                )

                optimization[f"product_{product_id}"] = {
                    "store_allocations": store_metrics_df.to_dict("records"),
                    "total_demand": float(total_demand),
                    "highest_risk_store": int(
                        float(
                            store_metrics_df.loc[
                                store_metrics_df["risk_score"].idxmax(), "store_id"
                            ]
                        )
                    ),
                    "lowest_risk_store": int(
                        float(
                            store_metrics_df.loc[
                                store_metrics_df["risk_score"].idxmin(), "store_id"
                            ]
                        )
                    ),
                }

        return optimization

    def _calculate_lead_time_risk(self, data: pd.DataFrame) -> float:
        """Calculate lead time risk based on historical patterns"""
        # Simplified lead time risk calculation
        # In practice, this would use supplier data and historical lead times

        # Use demand trend as proxy for lead time pressure
        recent_trend = data.tail(14)["sale_amount"].pct_change().mean()
        return max(0, recent_trend) * 2  # Positive trend increases lead time risk

    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk score into risk level"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _classify_hourly_risk(self, probability: float) -> str:
        """Classify hourly stockout probability"""
        if probability >= 0.7:
            return "critical"
        elif probability >= 0.5:
            return "high"
        elif probability >= 0.3:
            return "medium"
        elif probability >= 0.1:
            return "low"
        else:
            return "minimal"

    def _calculate_safety_stock(
        self, data: pd.DataFrame, service_level: float
    ) -> float:
        """Calculate recommended safety stock"""
        # Simplified safety stock calculation
        # safety_stock = z_score * demand_std * sqrt(lead_time)

        demand_std = data["sale_amount"].std()
        avg_demand = data["sale_amount"].mean()

        # Z-score for service level (approximation)
        z_scores = {0.90: 1.28, 0.95: 1.64, 0.99: 2.33}
        z_score = z_scores.get(service_level, 1.64)

        # Assume 3-day lead time
        lead_time = 3

        safety_stock = z_score * demand_std * np.sqrt(lead_time)

        return max(safety_stock, avg_demand * 0.1)  # Minimum 10% of average demand

    def _predict_days_until_stockout(self, data: pd.DataFrame) -> Optional[int]:
        """Predict days until stockout based on current trends"""
        recent_data = data.tail(7)

        if len(recent_data) < 3:
            return None

        avg_demand = recent_data["sale_amount"].mean()
        trend = recent_data["sale_amount"].pct_change().mean()

        if avg_demand <= 0:
            return None

        # Simplified calculation based on trend
        # In practice, this would use current inventory levels
        current_stock_proxy = (
            (1 - recent_data["stock_hour6_22_cnt"].mean() / 16) * avg_demand * 7
        )

        if trend > 0:
            # Accelerating demand
            days_until_stockout = current_stock_proxy / (avg_demand * (1 + trend))
        else:
            # Stable or declining demand
            days_until_stockout = current_stock_proxy / avg_demand

        return max(1, int(days_until_stockout)) if days_until_stockout > 0 else None

    def _generate_reorder_recommendations(
        self, risk_assessments: Dict[str, Any], request: StockoutAssessmentRequest
    ) -> List[Dict[str, Any]]:
        """Generate reorder recommendations"""

        recommendations = []

        for key, assessment in risk_assessments.items():
            risk_level = assessment["risk_level"]

            if risk_level in ["critical", "high"]:
                # Parse store and product IDs from key
                parts = key.split("_")
                store_id = int(parts[1])
                product_id = int(parts[3])

                urgency = "immediate" if risk_level == "critical" else "high"
                reorder_quantity = assessment["recommended_safety_stock"] * 2

                recommendations.append(
                    {
                        "store_id": store_id,
                        "product_id": product_id,
                        "urgency": urgency,
                        "risk_level": risk_level,
                        "recommended_reorder_quantity": float(reorder_quantity),
                        "days_until_stockout": assessment.get(
                            "days_until_predicted_stockout"
                        ),
                        "reason": f"Risk score: {assessment['risk_score']:.2f}, Stockout frequency: {assessment['stockout_frequency']:.2%}",
                    }
                )

        # Sort by urgency and risk score
        recommendations.sort(
            key=lambda x: (
                x["urgency"] == "immediate",
                -risk_assessments[f"store_{x['store_id']}_product_{x['product_id']}"][
                    "risk_score"
                ],
            )
        )

        return recommendations

    def _generate_early_warnings(
        self, risk_assessments: Dict[str, Any], request: StockoutAssessmentRequest
    ) -> List[Dict[str, Any]]:
        """Generate early warning alerts"""

        warnings = []

        for key, assessment in risk_assessments.items():
            # Parse store and product IDs
            parts = key.split("_")
            store_id = int(parts[1])
            product_id = int(parts[3])

            # Critical warnings
            if assessment["risk_level"] == "critical":
                warnings.append(
                    {
                        "alert_type": "critical_stockout_risk",
                        "store_id": store_id,
                        "product_id": product_id,
                        "message": f"Critical stockout risk detected. Risk score: {assessment['risk_score']:.2f}",
                        "recommended_action": "Immediate reorder required",
                        "severity": "critical",
                    }
                )

            # High demand volatility
            if assessment["demand_volatility"] > 1.0:
                warnings.append(
                    {
                        "alert_type": "high_demand_volatility",
                        "store_id": store_id,
                        "product_id": product_id,
                        "message": f"High demand volatility detected: {assessment['demand_volatility']:.2f}",
                        "recommended_action": "Increase safety stock and monitor closely",
                        "severity": "warning",
                    }
                )

            # Rapid trend increase
            if assessment["recent_trend"] > 0.2:
                warnings.append(
                    {
                        "alert_type": "rapid_demand_increase",
                        "store_id": store_id,
                        "product_id": product_id,
                        "message": f"Rapid demand increase: {assessment['recent_trend']:.1%} trend",
                        "recommended_action": "Prepare for increased demand, consider expedited reorder",
                        "severity": "warning",
                    }
                )

        return warnings[:20]  # Return top 20 warnings


# Export the service
enhanced_stockout_service = EnhancedStockoutService()

__all__ = [
    "EnhancedStockoutService",
    "StockoutAssessmentRequest",
    "StockoutAssessmentResult",
    "RiskLevel",
    "enhanced_stockout_service",
]
