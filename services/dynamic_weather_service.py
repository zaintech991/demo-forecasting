"""
Dynamic Weather-Sensitive Demand Service
Provides truly dynamic weather analysis based on actual database data.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database.connection import get_pool

logger = logging.getLogger(__name__)


class DynamicWeatherService:
    """Dynamic weather analysis service with product-specific insights."""

    def __init__(self):
        self.cache = {}

    async def get_weather_data(
        self,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        city_id: Optional[int] = None,
        limit: int = 2000,
    ) -> pd.DataFrame:
        """Fetch actual weather and sales data from database."""

        pool = await get_pool()

        query = """
        SELECT 
            sd.dt as date,
            sd.store_id,
            sd.product_id,
            sd.sale_amount,
            sd.avg_temperature,
            sd.avg_humidity,
            sd.precpt as precipitation,
            sd.avg_wind_level as wind_speed,
            sd.discount,
            sd.holiday_flag,
            sd.activity_flag,
            ph.product_name,
            sh.store_name,
            sh.city_id
        FROM sales_data sd
        JOIN product_hierarchy ph ON sd.product_id = ph.product_id
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        WHERE 1=1
        """

        params: List[Any] = []
        if store_id is not None:
            query += f" AND sd.store_id = ${len(params) + 1}"
            params.append(store_id)

        if product_id is not None:
            query += f" AND sd.product_id = ${len(params) + 1}"
            params.append(product_id)

        if city_id is not None:
            query += f" AND sh.city_id = ${len(params) + 1}"
            params.append(city_id)

        query += f" ORDER BY sd.dt DESC LIMIT ${len(params) + 1}"
        params.append(limit)

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df

    async def analyze_weather_sensitivity(
        self,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        city_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze weather sensitivity with dynamic, product-specific insights."""

        try:
            # Get actual data
            df = await self.get_weather_data(store_id, product_id, city_id)

            if df.empty:
                return {"error": "No data found for the specified criteria"}

            # Calculate actual correlations
            weather_correlations = {}
            weather_impacts = []

            # Temperature correlation
            if df["avg_temperature"].notna().sum() > 10:
                temp_corr = df["sale_amount"].corr(df["avg_temperature"])
                weather_correlations["temperature_correlation"] = (
                    round(float(temp_corr), 2) if not pd.isna(temp_corr) else 0.0
                )
                weather_impacts.append(
                    abs(weather_correlations["temperature_correlation"]) * 100
                )
            else:
                weather_correlations["temperature_correlation"] = 0.0
                weather_impacts.append(0.0)

            # Humidity correlation
            if df["avg_humidity"].notna().sum() > 10:
                humidity_corr = df["sale_amount"].corr(df["avg_humidity"])
                weather_correlations["humidity_correlation"] = (
                    round(float(humidity_corr), 2)
                    if not pd.isna(humidity_corr)
                    else 0.0
                )
                weather_impacts.append(
                    abs(weather_correlations["humidity_correlation"]) * 100
                )
            else:
                weather_correlations["humidity_correlation"] = 0.0
                weather_impacts.append(0.0)

            # Precipitation correlation
            if df["precipitation"].notna().sum() > 10:
                precip_corr = df["sale_amount"].corr(df["precipitation"])
                weather_correlations["precipitation_correlation"] = (
                    float(precip_corr) if not pd.isna(precip_corr) else 0.0
                )
                weather_impacts.append(
                    abs(weather_correlations["precipitation_correlation"]) * 100
                )
            else:
                weather_correlations["precipitation_correlation"] = 0.0
                weather_impacts.append(0.0)

            # Wind correlation
            if df["wind_speed"].notna().sum() > 10:
                wind_corr = df["sale_amount"].corr(df["wind_speed"])
                weather_correlations["wind_correlation"] = (
                    float(wind_corr) if not pd.isna(wind_corr) else 0.0
                )
                weather_impacts.append(
                    abs(weather_correlations["wind_correlation"]) * 100
                )
            else:
                weather_correlations["wind_correlation"] = 0.0
                weather_impacts.append(0.0)

            # Generate dynamic, product-specific recommendations
            recommendations = await self.generate_dynamic_recommendations(
                df, weather_correlations
            )

            return {
                "weather_sensitivity": weather_correlations,
                "weather_impacts": weather_impacts,
                "recommendations": recommendations,
                "data_insights": {
                    "total_records": len(df),
                    "date_range": {
                        "start": df["date"].min().strftime("%Y-%m-%d"),
                        "end": df["date"].max().strftime("%Y-%m-%d"),
                    },
                    "avg_sales": float(df["sale_amount"].mean()),
                    "weather_ranges": {
                        "temperature": {
                            "min": (
                                float(df["avg_temperature"].min())
                                if df["avg_temperature"].notna().any()
                                else None
                            ),
                            "max": (
                                float(df["avg_temperature"].max())
                                if df["avg_temperature"].notna().any()
                                else None
                            ),
                            "avg": (
                                float(df["avg_temperature"].mean())
                                if df["avg_temperature"].notna().any()
                                else None
                            ),
                        },
                        "humidity": {
                            "min": (
                                float(df["avg_humidity"].min())
                                if df["avg_humidity"].notna().any()
                                else None
                            ),
                            "max": (
                                float(df["avg_humidity"].max())
                                if df["avg_humidity"].notna().any()
                                else None
                            ),
                            "avg": (
                                float(df["avg_humidity"].mean())
                                if df["avg_humidity"].notna().any()
                                else None
                            ),
                        },
                    },
                },
            }

        except Exception as e:
            logger.error(f"Weather analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    async def generate_dynamic_recommendations(
        self, df: pd.DataFrame, correlations: Dict[str, float]
    ) -> List[str]:
        """Generate dynamic, product-specific recommendations."""

        recommendations = []

        # Get product info
        product_name = df["product_name"].iloc[0] if not df.empty else "this product"
        avg_sales = df["sale_amount"].mean()

        # Temperature recommendations
        temp_corr = correlations.get("temperature_correlation", 0)
        if abs(temp_corr) > 0.3:
            if temp_corr > 0:
                optimal_temp = df.loc[
                    df["sale_amount"] > avg_sales, "avg_temperature"
                ].mean()
                if not pd.isna(optimal_temp):
                    recommendations.append(
                        f"🌡️ {product_name} sales increase with warmer weather. "
                        f"Optimize inventory when temperatures reach {optimal_temp:.1f}°C"
                    )
            else:
                optimal_temp = df.loc[
                    df["sale_amount"] > avg_sales, "avg_temperature"
                ].mean()
                if not pd.isna(optimal_temp):
                    recommendations.append(
                        f"❄️ {product_name} sales peak in cooler weather. "
                        f"Stock up when temperatures drop to {optimal_temp:.1f}°C"
                    )

        # Precipitation recommendations
        precip_corr = correlations.get("precipitation_correlation", 0)
        if abs(precip_corr) > 0.2:
            rainy_sales = (
                df[df["precipitation"] > 0]["sale_amount"].mean()
                if (df["precipitation"] > 0).any()
                else avg_sales
            )
            dry_sales = (
                df[df["precipitation"] == 0]["sale_amount"].mean()
                if (df["precipitation"] == 0).any()
                else avg_sales
            )

            if rainy_sales > dry_sales:
                recommendations.append(
                    f"🌧️ {product_name} demand increases by {((rainy_sales - dry_sales) / dry_sales * 100):.1f}% during rainy weather. "
                    f"Prepare extra stock before forecasted rain"
                )
            else:
                recommendations.append(
                    f"☀️ {product_name} performs better in dry conditions. "
                    f"Consider promotional activities during rainy periods"
                )

        # Humidity recommendations
        humidity_corr = correlations.get("humidity_correlation", 0)
        if abs(humidity_corr) > 0.3:
            if humidity_corr > 0:
                recommendations.append(
                    f"💧 {product_name} sales correlate with humidity. "
                    f"Monitor weather forecasts and adjust inventory for humid conditions"
                )
            else:
                recommendations.append(
                    f"🏜️ {product_name} performs better in dry conditions. "
                    f"Reduce inventory during high humidity periods"
                )

        # Promotion-weather integration
        if not df.empty:
            promo_weather_insight = await self.analyze_promotion_weather_correlation(df)
            if promo_weather_insight:
                recommendations.append(promo_weather_insight)

        # Holiday-weather interaction
        holiday_weather_insight = await self.analyze_holiday_weather_interaction(df)
        if holiday_weather_insight:
            recommendations.append(holiday_weather_insight)

        # Default recommendation if no strong correlations
        if not recommendations:
            recommendations.append(
                f"📊 Weather impact on {product_name} is minimal. "
                f"Focus on other demand drivers like promotions and seasonality"
            )

        return recommendations

    async def analyze_promotion_weather_correlation(
        self, df: pd.DataFrame
    ) -> Optional[str]:
        """Analyze correlation between promotions and weather conditions."""

        if df.empty or "promotion_flag" not in df.columns:
            return None

        promo_df = df[df["promotion_flag"] == True]
        no_promo_df = df[df["promotion_flag"] == False]

        if len(promo_df) < 5 or len(no_promo_df) < 5:
            return None

        # Find weather conditions that work best with promotions
        promo_temp_avg = promo_df["avg_temperature"].mean()
        no_promo_temp_avg = no_promo_df["avg_temperature"].mean()

        promo_sales_avg = promo_df["sale_amount"].mean()
        no_promo_sales_avg = no_promo_df["sale_amount"].mean()

        if promo_sales_avg > no_promo_sales_avg * 1.2:  # Promotions are effective
            if promo_temp_avg > no_promo_temp_avg + 2:
                return f"🎯 Promotions are {((promo_sales_avg / no_promo_sales_avg - 1) * 100):.1f}% more effective in warmer weather (avg {promo_temp_avg:.1f}°C)"
            elif promo_temp_avg < no_promo_temp_avg - 2:
                return f"❄️ Promotions work best in cooler weather. Consider timing promotions with cold fronts"

        return None

    async def analyze_holiday_weather_interaction(
        self, df: pd.DataFrame
    ) -> Optional[str]:
        """Analyze how holidays interact with weather conditions."""

        if df.empty or "holiday_flag" not in df.columns:
            return None

        holiday_df = df[df["holiday_flag"] == True]
        regular_df = df[df["holiday_flag"] == False]

        if len(holiday_df) < 3:
            return None

        holiday_sales = holiday_df["sale_amount"].mean()
        regular_sales = regular_df["sale_amount"].mean()

        if holiday_sales > regular_sales * 1.3:  # Holidays boost sales
            holiday_temp = holiday_df["avg_temperature"].mean()
            if not pd.isna(holiday_temp):
                return f"🎉 Holiday sales are {((holiday_sales / regular_sales - 1) * 100):.1f}% higher. Weather during holidays averaged {holiday_temp:.1f}°C"

        return None

    async def get_weather_forecast_impact(
        self,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        weather_scenario: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """Predict sales impact based on weather forecast."""

        if weather_scenario is None:
            weather_scenario = {
                "temperature": 20.0,
                "humidity": 50.0,
                "precipitation": 0.0,
                "wind_speed": 10.0,
            }

        # Get historical data
        df = await self.get_weather_data(store_id, product_id, limit=500)

        if df.empty:
            return {"error": "No historical data for prediction"}

        # Find similar weather conditions in history
        similar_conditions = df[
            (abs(df["avg_temperature"] - weather_scenario["temperature"]) <= 5)
            & (abs(df["avg_humidity"] - weather_scenario["humidity"]) <= 20)
        ]

        if not similar_conditions.empty:
            predicted_sales = similar_conditions["sale_amount"].mean()
            baseline_sales = df["sale_amount"].mean()

            impact_percent = ((predicted_sales - baseline_sales) / baseline_sales) * 100

            return {
                "predicted_sales": float(predicted_sales),
                "baseline_sales": float(baseline_sales),
                "impact_percent": float(impact_percent),
                "confidence": min(
                    len(similar_conditions) / 20, 1.0
                ),  # Confidence based on sample size
                "recommendation": f"Expected sales change: {impact_percent:+.1f}% under these weather conditions",
            }

        return {
            "predicted_sales": float(df["sale_amount"].mean()),
            "baseline_sales": float(df["sale_amount"].mean()),
            "impact_percent": 0.0,
            "confidence": 0.1,
            "recommendation": "Limited historical data for these weather conditions",
        }
