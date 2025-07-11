"""
Advanced feature engineering module for demand forecasting and store analytics.
Builds sophisticated features for weather sensitivity, category forecasting, and store clustering.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class AdvancedFeatureBuilder:
    """Advanced feature engineering for retail demand forecasting."""

    def __init__(self, save_path="models/preprocessor"):
        """Initialize the feature builder."""
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def build_weather_sensitivity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build weather sensitivity features for demand modeling."""
        df = df.copy()

        # Weather impact scores
        df["temperature_demand_impact"] = self._calculate_temperature_impact(df)
        df["humidity_demand_impact"] = self._calculate_humidity_impact(df)
        df["precipitation_demand_impact"] = self._calculate_precipitation_impact(df)
        df["wind_demand_impact"] = self._calculate_wind_impact(df)

        # Weather seasonality patterns
        df["weather_seasonality"] = self._calculate_weather_seasonality(df)

        # Weather comfort index
        df["weather_comfort_index"] = self._calculate_comfort_index(df)

        # Weather volatility (rolling std of weather conditions)
        df = self._add_weather_volatility(df)

        # Product-weather interaction features
        df = self._add_product_weather_interactions(df)

        return df

    def build_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build category-level aggregation and forecasting features."""
        df = df.copy()

        # Category performance metrics
        df = self._add_category_performance_metrics(df)

        # Category seasonality patterns
        df = self._add_category_seasonality(df)

        # Cross-category correlation features
        df = self._add_cross_category_features(df)

        # Category lifecycle features
        df = self._add_category_lifecycle_features(df)

        # Category elasticity features
        df = self._add_category_elasticity_features(df)

        return df

    def build_store_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features for store clustering and behavior segmentation."""
        df = df.copy()

        # Store performance metrics
        df = self._add_store_performance_metrics(df)

        # Store behavior patterns
        df = self._add_store_behavior_patterns(df)

        # Customer behavior indicators
        df = self._add_customer_behavior_indicators(df)

        # Store operational efficiency features
        df = self._add_operational_efficiency_features(df)

        # Store location and demographic features
        df = self._add_location_demographic_features(df)

        return df

    def build_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build advanced temporal features."""
        df = df.copy()

        # Convert sale_date to datetime if needed
        if df["sale_date"].dtype != "datetime64[ns]":
            df["sale_date"] = pd.to_datetime(df["sale_date"])

        # Cyclical time features
        df["day_sin"] = np.sin(2 * np.pi * df["sale_date"].dt.dayofyear / 365.25)
        df["day_cos"] = np.cos(2 * np.pi * df["sale_date"].dt.dayofyear / 365.25)
        df["week_sin"] = np.sin(2 * np.pi * df["sale_date"].dt.isocalendar().week / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["sale_date"].dt.isocalendar().week / 52)
        df["month_sin"] = np.sin(2 * np.pi * df["sale_date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["sale_date"].dt.month / 12)

        # Advanced temporal patterns
        df["quarter"] = df["sale_date"].dt.quarter
        df["is_month_start"] = (df["sale_date"].dt.day <= 5).astype(int)
        df["is_month_end"] = (df["sale_date"].dt.day >= 25).astype(int)
        df["days_since_last_holiday"] = self._calculate_days_since_holiday(df)
        df["days_until_next_holiday"] = self._calculate_days_until_holiday(df)

        return df

    def build_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = "sale_amount",
        group_cols: List[str] = ["store_id", "product_id"],
    ) -> pd.DataFrame:
        """Build lag and rolling window features."""
        df = df.copy()
        df = df.sort_values(["sale_date"] + group_cols)

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(
                lag
            )

        # Rolling window features
        for window in [7, 14, 30]:
            df[f"{target_col}_rolling_mean_{window}"] = (
                df.groupby(group_cols)[target_col]
                .rolling(window)
                .mean()
                .reset_index(level=group_cols, drop=True)
            )
            df[f"{target_col}_rolling_std_{window}"] = (
                df.groupby(group_cols)[target_col]
                .rolling(window)
                .std()
                .reset_index(level=group_cols, drop=True)
            )
            df[f"{target_col}_rolling_min_{window}"] = (
                df.groupby(group_cols)[target_col]
                .rolling(window)
                .min()
                .reset_index(level=group_cols, drop=True)
            )
            df[f"{target_col}_rolling_max_{window}"] = (
                df.groupby(group_cols)[target_col]
                .rolling(window)
                .max()
                .reset_index(level=group_cols, drop=True)
            )

        # Trend features
        df[f"{target_col}_trend_7d"] = (
            df[f"{target_col}_rolling_mean_7"] - df[f"{target_col}_lag_7"]
        ) / df[f"{target_col}_lag_7"].replace(0, np.nan)
        df[f"{target_col}_trend_30d"] = (
            df[f"{target_col}_rolling_mean_30"] - df[f"{target_col}_lag_30"]
        ) / df[f"{target_col}_lag_30"].replace(0, np.nan)

        return df

    # Private helper methods
    def _calculate_temperature_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate temperature impact on demand."""
        # Optimal temperature range for retail (20-25°C)
        optimal_temp = 22.5
        temp_deviation = np.abs(df["avg_temperature"] - optimal_temp)
        return 1 / (1 + temp_deviation / 10)  # Sigmoid-like function

    def _calculate_humidity_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate humidity impact on demand."""
        # Optimal humidity range (40-60%)
        optimal_humidity = 50
        humidity_deviation = np.abs(df["avg_humidity"] - optimal_humidity)
        return 1 / (1 + humidity_deviation / 20)

    def _calculate_precipitation_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate precipitation impact on demand."""
        # Light rain might increase indoor shopping, heavy rain decreases it
        precip_impact = df["precpt"].apply(
            lambda x: (
                1.1
                if 0 < x <= 5  # Light rain boosts indoor shopping
                else (
                    1.0 - min(x / 50, 0.5)
                    if x > 5  # Heavy rain decreases shopping
                    else 1.0
                )
            )  # No rain
        )
        return precip_impact

    def _calculate_wind_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate wind impact on demand."""
        # High wind generally decreases shopping activity
        return 1.0 - (df["avg_wind_level"] / df["avg_wind_level"].max()) * 0.2

    def _calculate_weather_seasonality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate weather-based seasonality."""
        # Combination of temperature and month for seasonal patterns
        temp_normalized = (df["avg_temperature"] - df["avg_temperature"].min()) / (
            df["avg_temperature"].max() - df["avg_temperature"].min()
        )
        month_factor = np.sin(2 * np.pi * df["sale_date"].dt.month / 12)
        return temp_normalized * month_factor

    def _calculate_comfort_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate weather comfort index."""
        # Weighted combination of weather factors
        temp_comfort = self._calculate_temperature_impact(df)
        humidity_comfort = self._calculate_humidity_impact(df)
        precip_comfort = self._calculate_precipitation_impact(df)
        wind_comfort = self._calculate_wind_impact(df)

        return (
            temp_comfort * 0.4
            + humidity_comfort * 0.3
            + precip_comfort * 0.2
            + wind_comfort * 0.1
        )

    def _add_weather_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather volatility features."""
        df = df.sort_values("sale_date")

        # Rolling standard deviation of weather conditions
        df["temp_volatility_7d"] = df["avg_temperature"].rolling(7).std()
        df["humidity_volatility_7d"] = df["avg_humidity"].rolling(7).std()
        df["weather_change_magnitude"] = (
            np.abs(df["avg_temperature"].diff())
            + np.abs(df["avg_humidity"].diff()) / 10
        )

        return df

    def _add_product_weather_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add product-weather interaction features."""
        # Temperature-product interactions (different products react differently to temperature)
        if "first_category_id" in df.columns:
            df["temp_category_interaction"] = (
                df["avg_temperature"] * df["first_category_id"]
            )
            df["humidity_category_interaction"] = (
                df["avg_humidity"] * df["first_category_id"]
            )

        return df

    def _add_category_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add category performance metrics."""
        if "first_category_id" not in df.columns:
            return df

        # Category sales statistics
        category_stats = (
            df.groupby("first_category_id")["sale_amount"]
            .agg(["mean", "std", "min", "max", "count"])
            .add_prefix("category_")
        )

        df = df.merge(
            category_stats, left_on="first_category_id", right_index=True, how="left"
        )

        # Category market share
        total_sales = df.groupby("sale_date")["sale_amount"].sum()
        category_daily_sales = df.groupby(["sale_date", "first_category_id"])[
            "sale_amount"
        ].sum()
        df["category_market_share"] = df.apply(
            lambda row: category_daily_sales.get(
                (row["sale_date"], row["first_category_id"]), 0
            )
            / total_sales.get(row["sale_date"], 1),
            axis=1,
        )

        return df

    def _add_category_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add category seasonality patterns."""
        if "first_category_id" not in df.columns:
            return df

        # Monthly category performance
        monthly_category_avg = df.groupby(
            ["first_category_id", df["sale_date"].dt.month]
        )["sale_amount"].mean()
        df["category_monthly_avg"] = df.apply(
            lambda row: monthly_category_avg.get(
                (row["first_category_id"], row["sale_date"].month), 0
            ),
            axis=1,
        )

        return df

    def _add_cross_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-category correlation features."""
        # This would require more complex analysis of category correlations
        # For now, adding placeholder for category diversity
        if "first_category_id" in df.columns:
            store_category_count = df.groupby("store_id")["first_category_id"].nunique()
            df["store_category_diversity"] = df["store_id"].map(store_category_count)

        return df

    def _add_category_lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add category lifecycle features."""
        if "first_category_id" not in df.columns:
            return df

        # Category introduction date (first appearance)
        category_first_date = df.groupby("first_category_id")["sale_date"].min()
        df["category_days_since_introduction"] = df.apply(
            lambda row: (
                row["sale_date"] - category_first_date[row["first_category_id"]]
            ).days,
            axis=1,
        )

        return df

    def _add_category_elasticity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add category price elasticity features."""
        if "first_category_id" not in df.columns or "discount" not in df.columns:
            return df

        # Category discount sensitivity
        category_discount_response = df.groupby("first_category_id").apply(
            lambda x: x["sale_amount"].corr(x["discount"]) if len(x) > 10 else 0
        )
        df["category_discount_sensitivity"] = df["first_category_id"].map(
            category_discount_response
        )

        return df

    def _add_store_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add store performance metrics."""
        # Store sales statistics
        store_stats = (
            df.groupby("store_id")["sale_amount"]
            .agg(["mean", "std", "min", "max", "count"])
            .add_prefix("store_")
        )

        df = df.merge(store_stats, left_on="store_id", right_index=True, how="left")

        # Store efficiency metrics
        df["store_sales_cv"] = (
            df["store_std"] / df["store_mean"]
        )  # Coefficient of variation

        return df

    def _add_store_behavior_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add store behavior pattern features."""
        # Peak shopping hours analysis
        if "hours_sale" in df.columns:

            def analyze_shopping_pattern(hours_str):
                try:
                    hours = json.loads(hours_str) if isinstance(hours_str, str) else []
                    if not hours:
                        return "unknown"
                    peak_hour = np.argmax(hours)
                    if peak_hour < 10:
                        return "morning"
                    elif peak_hour < 15:
                        return "afternoon"
                    else:
                        return "evening"
                except:
                    return "unknown"

            df["store_shopping_pattern"] = df["hours_sale"].apply(
                analyze_shopping_pattern
            )

        # Weekend vs weekday preference
        weekend_sales = (
            df[df["sale_date"].dt.dayofweek.isin([5, 6])]
            .groupby("store_id")["sale_amount"]
            .mean()
        )
        weekday_sales = (
            df[~df["sale_date"].dt.dayofweek.isin([5, 6])]
            .groupby("store_id")["sale_amount"]
            .mean()
        )
        weekend_preference = weekend_sales / weekday_sales
        df["store_weekend_preference"] = df["store_id"].map(
            weekend_preference.fillna(1)
        )

        return df

    def _add_customer_behavior_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add customer behavior indicators."""
        # Average transaction size
        store_avg_transaction = df.groupby("store_id")["sale_amount"].mean()
        df["store_avg_transaction_size"] = df["store_id"].map(store_avg_transaction)

        # Customer loyalty indicator (consistency in sales)
        store_sales_consistency = 1 / (
            1
            + df.groupby("store_id")["sale_amount"].std()
            / df.groupby("store_id")["sale_amount"].mean()
        )
        df["store_customer_loyalty"] = df["store_id"].map(
            store_sales_consistency.fillna(0.5)
        )

        return df

    def _add_operational_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add operational efficiency features."""
        # Stock management efficiency
        if "stock_hour6_22_cnt" in df.columns:
            store_stock_efficiency = df.groupby("store_id")["stock_hour6_22_cnt"].mean()
            df["store_stock_efficiency"] = df["store_id"].map(store_stock_efficiency)

        # Promotion effectiveness
        if "promo_flag" in df.columns:
            promo_data = df[df["promo_flag"] == 1]
            non_promo_data = df[df["promo_flag"] == 0]

            if len(promo_data) > 0 and len(non_promo_data) > 0:
                promo_sales = promo_data.groupby("store_id")["sale_amount"].mean()
                regular_sales = non_promo_data.groupby("store_id")["sale_amount"].mean()
                promo_effectiveness = promo_sales / regular_sales
                df["store_promo_effectiveness"] = df["store_id"].map(
                    promo_effectiveness.fillna(1)
                )

        return df

    def _add_location_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location and demographic features."""
        if "city_id" in df.columns:
            # City-level sales statistics
            city_stats = (
                df.groupby("city_id")["sale_amount"]
                .agg(["mean", "std"])
                .add_prefix("city_")
            )
            df = df.merge(city_stats, left_on="city_id", right_index=True, how="left")

            # Store density in city
            stores_per_city = df.groupby("city_id")["store_id"].nunique()
            df["city_store_density"] = df["city_id"].map(stores_per_city)

        return df

    def _calculate_days_since_holiday(self, df: pd.DataFrame) -> pd.Series:
        """Calculate days since last holiday."""
        if "holiday_flag" not in df.columns:
            return pd.Series(0, index=df.index)

        holiday_dates = df[df["holiday_flag"] == 1]["sale_date"].unique()

        def days_since_holiday(date):
            past_holidays = holiday_dates[holiday_dates <= date]
            if len(past_holidays) == 0:
                return 365  # No holidays found
            return (date - past_holidays.max()).days

        return df["sale_date"].apply(days_since_holiday)

    def _calculate_days_until_holiday(self, df: pd.DataFrame) -> pd.Series:
        """Calculate days until next holiday."""
        if "holiday_flag" not in df.columns:
            return pd.Series(0, index=df.index)

        holiday_dates = df[df["holiday_flag"] == 1]["sale_date"].unique()

        def days_until_holiday(date):
            future_holidays = holiday_dates[holiday_dates > date]
            if len(future_holidays) == 0:
                return 365  # No holidays found
            return (future_holidays.min() - date).days

        return df["sale_date"].apply(days_until_holiday)
