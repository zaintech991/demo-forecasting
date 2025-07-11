"""
Dynamic Category Forecasting Service
Provides real category-level insights based on actual database data.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database.connection import get_pool

logger = logging.getLogger(__name__)


class DynamicCategoryService:
    """Dynamic category analysis service with real performance insights."""

    def __init__(self):
        self.cache = {}

    async def get_category_data(
        self,
        category_id: Optional[int] = None,
        store_id: Optional[int] = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Fetch actual category sales data from database."""

        pool = await get_pool()

        query = """
        SELECT 
            sd.dt as date,
            sd.store_id,
            sd.product_id,
            sd.city_id,
            sd.sale_amount,
            sd.first_category_id,
            sd.second_category_id,
            sd.third_category_id,
            sd.discount,
            sd.holiday_flag,
            sd.activity_flag as promotion_flag,
            ph.product_name,
            sh.store_name,
            EXTRACT(MONTH FROM sd.dt) as month,
            EXTRACT(YEAR FROM sd.dt) as year,
            EXTRACT(DOW FROM sd.dt) as day_of_week
        FROM sales_data sd
        JOIN product_hierarchy ph ON sd.product_id = ph.product_id
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        WHERE 1=1
        """

        params: List[Any] = []
        if category_id is not None:
            query += f" AND ph.first_category_id = ${len(params) + 1}"
            params.append(category_id)

        if store_id is not None:
            query += f" AND sd.store_id = ${len(params) + 1}"
            params.append(store_id)

        query += f" ORDER BY sd.dt DESC LIMIT ${len(params) + 1}"
        params.append(limit)

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df

    async def analyze_category_performance(
        self, category_id: Optional[int] = None, store_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze category performance with dynamic insights."""

        try:
            # Get actual data
            df = await self.get_category_data(category_id, store_id)

            if df.empty:
                return {"error": "No data found for the specified criteria"}

            # Calculate performance metrics
            performance_metrics = []

            if category_id:
                # Single category analysis
                category_data = df[df["first_category_id"] == category_id]
                total_sales = category_data["sale_amount"].sum()

                # Calculate market share (compared to all categories in same store)
                if store_id:
                    all_store_data = await self.get_category_data(None, store_id)
                    store_total = all_store_data["sale_amount"].sum()
                    market_share = (
                        (total_sales / store_total * 100) if store_total > 0 else 0
                    )
                else:
                    market_share = 100  # If no store specified, assume 100%

                # Calculate growth rate (recent vs older data)
                if len(category_data) > 30:
                    recent_data = category_data.head(15)
                    older_data = category_data.tail(15)
                    recent_avg = recent_data["sale_amount"].mean()
                    older_avg = older_data["sale_amount"].mean()
                    growth_rate = (
                        ((recent_avg - older_avg) / older_avg * 100)
                        if older_avg > 0
                        else 0
                    )
                else:
                    growth_rate = 0

                performance_metrics.append(
                    {
                        "category_id": category_id,
                        "total_sales": float(total_sales),
                        "market_share_percent": float(market_share),
                        "growth_rate_percent": float(growth_rate),
                        "avg_daily_sales": float(category_data["sale_amount"].mean()),
                        "transaction_count": len(category_data),
                    }
                )
            else:
                # Multi-category analysis
                category_groups = df.groupby("first_category_id")
                total_all_sales = df["sale_amount"].sum()

                for cat_id, cat_data in category_groups:
                    cat_total = cat_data["sale_amount"].sum()
                    market_share = (
                        (cat_total / total_all_sales * 100)
                        if total_all_sales > 0
                        else 0
                    )

                    # Growth rate calculation
                    if len(cat_data) > 20:
                        cat_data_sorted = cat_data.sort_values("date")
                        recent = cat_data_sorted.tail(10)["sale_amount"].mean()
                        older = cat_data_sorted.head(10)["sale_amount"].mean()
                        growth_rate = (
                            ((recent - older) / older * 100) if older > 0 else 0
                        )
                    else:
                        growth_rate = 0

                    performance_metrics.append(
                        {
                            "category_id": int(cat_id),
                            "total_sales": float(cat_total),
                            "market_share_percent": float(market_share),
                            "growth_rate_percent": float(growth_rate),
                            "avg_daily_sales": float(cat_data["sale_amount"].mean()),
                            "transaction_count": len(cat_data),
                        }
                    )

                # Sort by market share
                performance_metrics.sort(
                    key=lambda x: x["market_share_percent"], reverse=True
                )

            # Generate monthly sales data
            monthly_data = self.calculate_monthly_trends(df)

            # Generate seasonality insights
            seasonality_insights = self.analyze_seasonality(df)

            return {
                "performance_metrics": performance_metrics,
                "monthly_data": monthly_data,
                "seasonality_insights": seasonality_insights,
                "data_summary": {
                    "total_records": len(df),
                    "date_range": {
                        "start": df["date"].min().strftime("%Y-%m-%d"),
                        "end": df["date"].max().strftime("%Y-%m-%d"),
                    },
                    "total_categories": df["first_category_id"].nunique(),
                    "total_products": df["product_id"].nunique(),
                },
            }

        except Exception as e:
            logger.error(f"Category analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def calculate_monthly_trends(self, df: pd.DataFrame) -> List[float]:
        """Calculate monthly sales trends from actual data."""

        if df.empty:
            return [100, 110, 120, 115, 125, 130, 140, 135, 125, 120, 130, 150]

        monthly_sales = df.groupby("month")["sale_amount"].sum()

        # Normalize to base 100 for the first month
        if len(monthly_sales) > 0:
            base_value = monthly_sales.iloc[0] if monthly_sales.iloc[0] > 0 else 1
            normalized_sales = (monthly_sales / base_value * 100).tolist()

            # Ensure we have 12 months of data
            monthly_data = [0] * 12
            for month, value in monthly_sales.items():
                if 1 <= month <= 12:
                    monthly_data[int(month) - 1] = float(value / base_value * 100)

            # Fill missing months with interpolation
            for i in range(12):
                if monthly_data[i] == 0:
                    if i > 0:
                        monthly_data[i] = monthly_data[i - 1]
                    else:
                        monthly_data[i] = 100

            return monthly_data

        return [100, 110, 120, 115, 125, 130, 140, 135, 125, 120, 130, 150]

    def analyze_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in the data."""

        if df.empty:
            return {
                "peak_months": [12, 1],
                "low_months": [2, 3],
                "pattern": "holiday_driven",
            }

        monthly_avg = df.groupby("month")["sale_amount"].mean()

        if len(monthly_avg) > 0:
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()

            # Determine pattern
            peak_value = monthly_avg.max()
            low_value = monthly_avg.min()
            seasonality_strength = (
                (peak_value - low_value) / low_value if low_value > 0 else 0
            )

            if seasonality_strength > 0.3:
                if peak_month in [11, 12, 1]:  # Nov, Dec, Jan
                    pattern = "holiday_driven"
                elif peak_month in [6, 7, 8]:  # Summer months
                    pattern = "summer_peak"
                elif peak_month in [3, 4, 5]:  # Spring months
                    pattern = "spring_driven"
                else:
                    pattern = "irregular"
            else:
                pattern = "stable_year_round"

            return {
                "peak_months": [int(peak_month)],
                "low_months": [int(low_month)],
                "pattern": pattern,
                "seasonality_strength": float(seasonality_strength),
                "monthly_averages": {int(k): float(v) for k, v in monthly_avg.items()},
            }

        return {"peak_months": [12], "low_months": [2], "pattern": "insufficient_data"}

    async def get_category_recommendations(
        self, category_id: Optional[int] = None, store_id: Optional[int] = None
    ) -> List[str]:
        """Generate dynamic category-specific recommendations."""

        df = await self.get_category_data(category_id, store_id)
        recommendations = []

        if df.empty:
            return ["Insufficient data for category recommendations"]

        # Promotion effectiveness analysis
        if "promotion_flag" in df.columns:
            promo_data = df[df["promotion_flag"] == True]
            non_promo_data = df[df["promotion_flag"] == False]

            if len(promo_data) > 5 and len(non_promo_data) > 5:
                promo_avg = promo_data["sale_amount"].mean()
                regular_avg = non_promo_data["sale_amount"].mean()

                if promo_avg > regular_avg * 1.2:
                    uplift = (promo_avg / regular_avg - 1) * 100
                    recommendations.append(
                        f"🎯 Promotions are highly effective for this category, "
                        f"showing {uplift:.1f}% sales uplift"
                    )
                elif promo_avg < regular_avg * 0.9:
                    recommendations.append(
                        f"⚠️ Promotions may be cannibalizing regular sales. "
                        f"Consider reducing promotion frequency"
                    )

        # Holiday performance analysis
        if "holiday_flag" in df.columns:
            holiday_data = df[df["holiday_flag"] == True]
            regular_data = df[df["holiday_flag"] == False]

            if len(holiday_data) > 3:
                holiday_avg = holiday_data["sale_amount"].mean()
                regular_avg = regular_data["sale_amount"].mean()

                if holiday_avg > regular_avg * 1.5:
                    recommendations.append(
                        f"🎉 Strong holiday performance detected. "
                        f"Increase inventory {int((holiday_avg / regular_avg - 1) * 100)}% before holidays"
                    )

        # Day of week patterns
        if "day_of_week" in df.columns:
            dow_avg = df.groupby("day_of_week")["sale_amount"].mean()
            best_day = dow_avg.idxmax()
            worst_day = dow_avg.idxmin()

            day_names = [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]

            if dow_avg.max() > dow_avg.min() * 1.3:
                recommendations.append(
                    f"📅 Peak sales occur on {day_names[int(best_day)]}. "
                    f"Consider targeted promotions on {day_names[int(worst_day)]} to boost weak days"
                )

        # Growth trend analysis
        if len(df) > 30:
            df_sorted = df.sort_values("date")
            recent_trend = df_sorted.tail(15)["sale_amount"].mean()
            older_trend = df_sorted.head(15)["sale_amount"].mean()

            if recent_trend > older_trend * 1.1:
                recommendations.append(
                    f"📈 Category showing positive growth trend. "
                    f"Consider expanding product range or increasing shelf space"
                )
            elif recent_trend < older_trend * 0.9:
                recommendations.append(
                    f"📉 Declining trend detected. "
                    f"Investigate competitive pressures or changing customer preferences"
                )

        if not recommendations:
            recommendations.append(
                "📊 Category performance is stable. "
                "Monitor regularly for seasonal changes and promotion opportunities"
            )

        return recommendations
