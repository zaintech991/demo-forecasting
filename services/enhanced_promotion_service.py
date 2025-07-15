"""
Enhanced Promotion Impact Analysis Service for FreshRetailNet-50K
Includes cross-product effects, competitive pricing models, and ROI optimization
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

from database.connection import db_manager

logger = logging.getLogger(__name__)


class PromotionType(Enum):
    """Promotion types"""

    PERCENTAGE_OFF = "percentage_off"
    FIXED_AMOUNT = "fixed_amount"
    BOGO = "buy_one_get_one"
    BUNDLE = "bundle"
    LOYALTY = "loyalty"


@dataclass
class PromotionAnalysisRequest:
    """Request model for promotion analysis"""

    store_ids: List[int]
    product_ids: List[int]
    analysis_period_days: int = 180
    include_cross_product_effects: bool = True
    include_cannibalization_analysis: bool = True
    include_roi_optimization: bool = True
    discount_range: Tuple[float, float] = (0.05, 0.50)  # 5% to 50%


@dataclass
class PromotionAnalysisResult:
    """Result model for promotion analysis"""

    promotion_effectiveness: Dict[str, Any]
    cross_product_effects: Optional[Dict[str, Any]] = None
    cannibalization_analysis: Optional[Dict[str, Any]] = None
    roi_optimization: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None


class EnhancedPromotionService:
    """Enhanced promotion impact analysis service"""

    def __init__(self):
        self.scaler = StandardScaler()

    async def analyze_promotion_impact(
        self, request: PromotionAnalysisRequest
    ) -> PromotionAnalysisResult:
        """Comprehensive promotion impact analysis"""

        logger.info(
            f"Analyzing promotion impact for {len(request.product_ids)} products"
        )

        # Get historical data
        historical_data = await self._get_promotion_sales_data(request)

        if historical_data.empty:
            raise ValueError("No historical data available")

        # Analyze promotion effectiveness
        effectiveness = await self._analyze_promotion_effectiveness(
            historical_data, request
        )

        # Cross-product effects analysis
        cross_effects = None
        if request.include_cross_product_effects:
            cross_effects = await self._analyze_cross_product_effects(
                historical_data, request
            )

        # Cannibalization analysis
        cannibalization = None
        if request.include_cannibalization_analysis:
            cannibalization = await self._analyze_cannibalization(
                historical_data, request
            )

        # ROI optimization
        roi_optimization = None
        if request.include_roi_optimization:
            roi_optimization = await self._optimize_promotion_roi(
                historical_data, request
            )

        # Generate recommendations
        recommendations = self._generate_promotion_recommendations(
            effectiveness, cross_effects, cannibalization, roi_optimization
        )

        return PromotionAnalysisResult(
            promotion_effectiveness=effectiveness,
            cross_product_effects=cross_effects,
            cannibalization_analysis=cannibalization,
            roi_optimization=roi_optimization,
            recommendations=recommendations,
        )

    async def _get_promotion_sales_data(
        self, request: PromotionAnalysisRequest
    ) -> pd.DataFrame:
        """Get sales data with promotion information"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.analysis_period_days)

        sales_data = await db_manager.get_sales_data(
            store_ids=request.store_ids,
            product_ids=request.product_ids,
            start_date=start_date,
            end_date=end_date,
        )

        # Add promotion flags and metrics
        sales_data["is_promoted"] = (sales_data["discount"] < 1.0).astype(int)
        sales_data["discount_percentage"] = (1 - sales_data["discount"]) * 100
        sales_data["promotion_depth"] = np.where(
            sales_data["discount_percentage"] > 0,
            pd.cut(
                sales_data["discount_percentage"],
                bins=[0, 10, 20, 30, 50, 100],
                labels=["light", "moderate", "deep", "very_deep", "extreme"],
            ),
            "none",
        )

        return sales_data

    async def _analyze_promotion_effectiveness(
        self, data: pd.DataFrame, request: PromotionAnalysisRequest
    ) -> Dict[str, Any]:
        """Analyze promotion effectiveness by product and store"""

        effectiveness = {}

        for product_id in request.product_ids:
            product_data = data[data["product_id"] == product_id]

            if len(product_data) < 30:
                continue

            # Calculate baseline (non-promoted) performance
            baseline_data = product_data[product_data["is_promoted"] == 0]
            promoted_data = product_data[product_data["is_promoted"] == 1]

            if len(baseline_data) == 0 or len(promoted_data) == 0:
                continue

            baseline_avg = baseline_data["sale_amount"].mean()
            promoted_avg = promoted_data["sale_amount"].mean()

            # Calculate uplift
            uplift_percentage = (
                ((promoted_avg - baseline_avg) / baseline_avg) * 100
                if baseline_avg > 0
                else 0
            )

            # Analyze by discount depth
            depth_analysis = {}
            for depth in promoted_data["promotion_depth"].unique():
                if depth != "none":
                    depth_data = promoted_data[
                        promoted_data["promotion_depth"] == depth
                    ]
                    if len(depth_data) > 0:
                        depth_avg = depth_data["sale_amount"].mean()
                        depth_uplift = (
                            ((depth_avg - baseline_avg) / baseline_avg) * 100
                            if baseline_avg > 0
                            else 0
                        )

                        depth_analysis[str(depth)] = {
                            "average_sales": float(depth_avg),
                            "uplift_percentage": float(depth_uplift),
                            "sample_size": len(depth_data),
                            "average_discount": float(
                                depth_data["discount_percentage"].mean()
                            ),
                        }

            # Store-level analysis
            store_analysis = {}
            for store_id in product_data["store_id"].unique():
                store_product_data = product_data[product_data["store_id"] == store_id]
                store_baseline = store_product_data[
                    store_product_data["is_promoted"] == 0
                ]["sale_amount"].mean()
                store_promoted = store_product_data[
                    store_product_data["is_promoted"] == 1
                ]["sale_amount"].mean()

                if store_baseline > 0 and not pd.isna(store_promoted):
                    store_uplift = (
                        (store_promoted - store_baseline) / store_baseline
                    ) * 100
                    store_analysis[int(store_id)] = {
                        "uplift_percentage": float(store_uplift),
                        "baseline_sales": float(store_baseline),
                        "promoted_sales": float(store_promoted),
                    }

            effectiveness[f"product_{product_id}"] = {
                "overall_uplift_percentage": float(uplift_percentage),
                "baseline_average_sales": float(baseline_avg),
                "promoted_average_sales": float(promoted_avg),
                "promotion_frequency": float(len(promoted_data) / len(product_data)),
                "depth_analysis": depth_analysis,
                "store_analysis": store_analysis,
            }

        return effectiveness

    async def _analyze_cross_product_effects(
        self, data: pd.DataFrame, request: PromotionAnalysisRequest
    ) -> Dict[str, Any]:
        """Analyze cross-product promotion effects"""

        cross_effects = {}

        # Get product correlations for cross-selling analysis
        correlations = await db_manager.calculate_product_correlations(
            analysis_period_days=request.analysis_period_days
        )

        for product_id in request.product_ids:
            product_effects = {}

            # Find correlated products
            product_correlations = await db_manager.get_product_correlations(
                product_id=product_id, correlation_threshold=0.3
            )

            if not product_correlations.empty:
                for _, corr_row in product_correlations.iterrows():
                    related_product_id = (
                        corr_row["product_b_id"]
                        if corr_row["product_a_id"] == product_id
                        else corr_row["product_a_id"]
                    )

                    # Analyze impact of promoting product_id on related_product_id
                    impact = self._calculate_cross_product_impact(
                        data, product_id, related_product_id
                    )

                    if impact:
                        product_effects[f"product_{related_product_id}"] = impact

            if product_effects:
                cross_effects[f"product_{product_id}"] = product_effects

        return cross_effects

    async def _analyze_cannibalization(
        self, data: pd.DataFrame, request: PromotionAnalysisRequest
    ) -> Dict[str, Any]:
        """Analyze cannibalization effects of promotions"""

        cannibalization = {}

        for store_id in request.store_ids:
            store_data = data[data["store_id"] == store_id]
            store_cannibalization = {}

            for product_id in request.product_ids:
                product_data = store_data[store_data["product_id"] == product_id]
                other_products_data = store_data[store_data["product_id"] != product_id]

                if len(product_data) < 10 or len(other_products_data) < 10:
                    continue

                # Find periods when this product was promoted
                promotion_periods = product_data[product_data["is_promoted"] == 1][
                    "sale_date"
                ].tolist()

                if not promotion_periods:
                    continue

                # Calculate impact on other products during promotion periods
                cannibalization_effects = []

                for other_product_id in other_products_data["product_id"].unique():
                    other_product_data = other_products_data[
                        other_products_data["product_id"] == other_product_id
                    ]

                    # Sales during promotion periods vs normal periods
                    promotion_sales = other_product_data[
                        other_product_data["sale_date"].isin(promotion_periods)
                    ]["sale_amount"].mean()

                    normal_sales = other_product_data[
                        ~other_product_data["sale_date"].isin(promotion_periods)
                    ]["sale_amount"].mean()

                    if normal_sales > 0 and not pd.isna(promotion_sales):
                        cannibalization_effect = (
                            (promotion_sales - normal_sales) / normal_sales
                        ) * 100

                        cannibalization_effects.append(
                            {
                                "affected_product_id": int(other_product_id),
                                "cannibalization_percentage": float(
                                    cannibalization_effect
                                ),
                                "severity": (
                                    "high"
                                    if cannibalization_effect < -20
                                    else (
                                        "moderate"
                                        if cannibalization_effect < -10
                                        else "low"
                                    )
                                ),
                            }
                        )

                if cannibalization_effects:
                    store_cannibalization[f"product_{product_id}"] = (
                        cannibalization_effects
                    )

            if store_cannibalization:
                cannibalization[f"store_{store_id}"] = store_cannibalization

        return cannibalization

    async def _optimize_promotion_roi(
        self, data: pd.DataFrame, request: PromotionAnalysisRequest
    ) -> Dict[str, Any]:
        """Optimize promotion ROI using ML models"""

        optimization = {}

        for product_id in request.product_ids:
            product_data = data[data["product_id"] == product_id].copy()

            if len(product_data) < 50:
                continue

            # Prepare features for ROI optimization
            features = [
                "discount_percentage",
                "avg_temperature",
                "avg_humidity",
                "holiday_flag",
                "activity_flag",
            ]
            available_features = [f for f in features if f in product_data.columns]

            if len(available_features) < 2:
                continue

            # Calculate ROI (simplified as sales uplift per discount unit)
            baseline_sales = product_data[product_data["is_promoted"] == 0][
                "sale_amount"
            ].mean()
            product_data["roi"] = np.where(
                product_data["discount_percentage"] > 0,
                (product_data["sale_amount"] - baseline_sales)
                / product_data["discount_percentage"],
                0,
            )

            # Train ROI prediction model
            training_data = product_data[product_data["discount_percentage"] > 0].copy()

            if len(training_data) < 20:
                continue

            X = training_data[available_features].fillna(0)
            y = training_data["roi"]

            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X, y)

                # Find optimal discount levels
                optimal_discounts = []
                discount_range = np.arange(
                    request.discount_range[0], request.discount_range[1], 0.05
                )

                for discount in discount_range:
                    # Create scenario with this discount
                    scenario_features = X.mean().copy()
                    scenario_features["discount_percentage"] = discount * 100

                    predicted_roi = rf_model.predict([scenario_features])[0]

                    optimal_discounts.append(
                        {
                            "discount_percentage": float(discount * 100),
                            "predicted_roi": float(predicted_roi),
                            "expected_uplift": float(predicted_roi * discount * 100),
                        }
                    )

                # Find best discount level
                best_discount = max(optimal_discounts, key=lambda x: x["predicted_roi"])

                optimization[f"product_{product_id}"] = {
                    "optimal_discount_percentage": best_discount["discount_percentage"],
                    "predicted_roi": best_discount["predicted_roi"],
                    "expected_uplift": best_discount["expected_uplift"],
                    "discount_response_curve": optimal_discounts,
                    "feature_importance": dict(
                        zip(available_features, rf_model.feature_importances_)
                    ),
                }

            except Exception as e:
                logger.warning(f"ROI optimization failed for product {product_id}: {e}")
                continue

        return optimization

    def _calculate_cross_product_impact(
        self, data: pd.DataFrame, promoted_product_id: int, affected_product_id: int
    ) -> Optional[Dict[str, Any]]:
        """Calculate impact of promoting one product on another"""

        promoted_data = data[data["product_id"] == promoted_product_id]
        affected_data = data[data["product_id"] == affected_product_id]

        if len(promoted_data) < 10 or len(affected_data) < 10:
            return None

        # Find promotion periods
        promotion_dates = promoted_data[promoted_data["is_promoted"] == 1][
            "sale_date"
        ].tolist()

        if not promotion_dates:
            return None

        # Calculate impact on affected product
        affected_during_promo = affected_data[
            affected_data["sale_date"].isin(promotion_dates)
        ]["sale_amount"].mean()

        affected_normal = affected_data[
            ~affected_data["sale_date"].isin(promotion_dates)
        ]["sale_amount"].mean()

        if affected_normal > 0 and not pd.isna(affected_during_promo):
            impact_percentage = (
                (affected_during_promo - affected_normal) / affected_normal
            ) * 100

            return {
                "impact_percentage": float(impact_percentage),
                "impact_type": "positive" if impact_percentage > 0 else "negative",
                "significance": (
                    "high"
                    if abs(impact_percentage) > 15
                    else "moderate" if abs(impact_percentage) > 5 else "low"
                ),
                "affected_sales_during_promo": float(affected_during_promo),
                "affected_sales_normal": float(affected_normal),
            }

        return None

    def _generate_promotion_recommendations(
        self,
        effectiveness: Dict[str, Any],
        cross_effects: Optional[Dict[str, Any]],
        cannibalization: Optional[Dict[str, Any]],
        roi_optimization: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate promotion optimization recommendations"""

        recommendations = []

        # Effectiveness recommendations
        if effectiveness:
            high_uplift_products = []
            low_uplift_products = []

            for product_key, product_data in effectiveness.items():
                uplift = product_data.get("overall_uplift_percentage", 0)
                if uplift > 20:
                    high_uplift_products.append(product_key)
                elif uplift < 5:
                    low_uplift_products.append(product_key)

            if high_uplift_products:
                recommendations.append(
                    f"Products {', '.join(high_uplift_products)} show excellent promotion response (>20% uplift). "
                    "Consider increasing promotion frequency."
                )

            if low_uplift_products:
                recommendations.append(
                    f"Products {', '.join(low_uplift_products)} show poor promotion response (<5% uplift). "
                    "Consider alternative strategies or deeper discounts."
                )

        # Cross-effects recommendations
        if cross_effects:
            for product_key, effects in cross_effects.items():
                positive_effects = [
                    k for k, v in effects.items() if v.get("impact_percentage", 0) > 10
                ]
                if positive_effects:
                    recommendations.append(
                        f"Promoting {product_key} positively impacts {', '.join(positive_effects)}. "
                        "Consider bundle promotions."
                    )

        # Cannibalization recommendations
        if cannibalization:
            for store_key, store_effects in cannibalization.items():
                for product_key, effects in store_effects.items():
                    high_cannibalization = [
                        e["affected_product_id"]
                        for e in effects
                        if e["severity"] == "high"
                    ]
                    if high_cannibalization:
                        recommendations.append(
                            f"In {store_key}, promoting {product_key} significantly cannibalizes "
                            f"products {high_cannibalization}. Consider coordinated promotions."
                        )

        # ROI optimization recommendations
        if roi_optimization:
            for product_key, opt_data in roi_optimization.items():
                optimal_discount = opt_data.get("optimal_discount_percentage", 0)
                recommendations.append(
                    f"{product_key}: Optimal discount is {optimal_discount:.1f}% for maximum ROI."
                )

        # General recommendations
        recommendations.extend(
            [
                "Implement dynamic pricing based on real-time demand signals",
                "Use A/B testing to validate promotion strategies",
                "Consider personalized promotions based on customer segments",
                "Monitor competitor pricing and adjust promotion timing accordingly",
            ]
        )

        return recommendations[:12]


# Export the service
enhanced_promotion_service = EnhancedPromotionService()

__all__ = [
    "EnhancedPromotionService",
    "PromotionAnalysisRequest",
    "PromotionAnalysisResult",
    "PromotionType",
    "enhanced_promotion_service",
]
