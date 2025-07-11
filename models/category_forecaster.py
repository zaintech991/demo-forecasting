"""
Category-Level Demand Forecasting Model.
Provides hierarchical forecasting, category aggregation, and category-specific analysis.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.model_selection import train_test_split, TimeSeriesSplit  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
import joblib  # type: ignore
from pathlib import Path
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
from scipy import stats  # type: ignore

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class CategoryLevelForecaster:
    """Category-level demand forecasting with hierarchical aggregation."""

    def __init__(
        self,
        aggregation_level: str = "category",
        model_type: str = "gradient_boost",
        save_path: str = "models/saved",
    ):
        """
        Initialize category-level forecaster.

        Args:
            aggregation_level: Level of aggregation ('category', 'subcategory', 'brand')
            model_type: Type of underlying model
            save_path: Path to save models
        """
        self.aggregation_level = aggregation_level
        self.model_type = model_type
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.models = {}  # One model per category
        self.scalers = {}
        self.category_metadata = {}
        self.seasonality_patterns = {}
        self.trend_patterns = {}
        self.is_fitted = False

        # Initialize base model architecture
        self._initialize_model_architecture()

    def _initialize_model_architecture(self):
        """Initialize the model architecture based on type."""
        if self.model_type == "gradient_boost":
            self.base_model_config = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
            }
        elif self.model_type == "random_forest":
            self.base_model_config = {
                "n_estimators": 100,
                "max_depth": 8,
                "random_state": 42,
                "n_jobs": -1,
            }
        elif self.model_type == "linear":
            self.base_model_config = {"fit_intercept": True}

    def aggregate_to_category_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate product-level data to category level.

        Args:
            df: Product-level sales data

        Returns:
            Category-level aggregated data
        """
        logger.info(f"Aggregating data to {self.aggregation_level} level...")

        # Define aggregation grouping based on level
        if self.aggregation_level == "category":
            group_cols = ["sale_date", "first_category_id", "store_id", "city_id"]
            category_col = "first_category_id"
        elif self.aggregation_level == "subcategory":
            group_cols = [
                "sale_date",
                "first_category_id",
                "second_category_id",
                "store_id",
                "city_id",
            ]
            category_col = "second_category_id"
        else:  # Default to category
            group_cols = ["sale_date", "first_category_id", "store_id", "city_id"]
            category_col = "first_category_id"

        # Aggregate numerical columns
        agg_dict = {
            "sale_amount": ["sum", "mean", "count"],
            "sale_qty": ["sum", "mean"],
            "discount": ["mean", "max"],
            "original_price": ["mean"],
            "stock_hour6_22_cnt": ["mean"],
            "holiday_flag": ["max"],  # If any product had holiday, category has holiday
            "promo_flag": ["mean"],  # Proportion of products on promotion
        }

        # Add weather data if available
        weather_cols = ["avg_temperature", "avg_humidity", "precpt", "avg_wind_level"]
        for col in weather_cols:
            if col in df.columns:
                agg_dict[col] = "mean"

        # Perform aggregation
        category_df = df.groupby(group_cols).agg(agg_dict).reset_index()

        # Flatten column names
        category_df.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in category_df.columns
        ]

        # Rename main target column
        category_df = category_df.rename(columns={"sale_amount_sum": "category_sales"})

        # Add category-specific features
        category_df = self._add_category_features(category_df, category_col)

        logger.info(f"Aggregated to {len(category_df)} category-level records")
        return category_df

    def _add_category_features(
        self, df: pd.DataFrame, category_col: str
    ) -> pd.DataFrame:
        """Add category-specific features."""
        df = df.copy()

        # Category market share
        daily_totals = df.groupby("sale_date")["category_sales"].sum()
        df["category_market_share"] = df.apply(
            lambda row: row["category_sales"] / daily_totals.get(row["sale_date"], 1),
            axis=1,
        )

        # Category velocity (average sales per day)
        category_velocity = df.groupby(category_col)["category_sales"].mean()
        df["category_velocity"] = df[category_col].map(category_velocity)

        # Category volatility (coefficient of variation)
        category_cv = df.groupby(category_col)["category_sales"].agg(
            lambda x: x.std() / x.mean()
        )
        df["category_volatility"] = df[category_col].map(category_cv.fillna(0))

        # Category growth trend (simple linear trend)
        df = df.sort_values(["sale_date", category_col])
        df["days_since_start"] = (df["sale_date"] - df["sale_date"].min()).dt.days

        category_trends = {}
        for cat in df[category_col].unique():
            cat_data = df[df[category_col] == cat].copy()
            if len(cat_data) > 10:  # Need sufficient data for trend
                try:
                    slope, _, _, _, _ = stats.linregress(
                        cat_data["days_since_start"], cat_data["category_sales"]
                    )
                    category_trends[cat] = slope
                except:
                    category_trends[cat] = 0
            else:
                category_trends[cat] = 0

        df["category_trend"] = df[category_col].map(category_trends)

        return df

    def analyze_category_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze seasonality patterns for each category.

        Args:
            df: Category-level data

        Returns:
            Seasonality analysis results
        """
        logger.info("Analyzing category seasonality patterns...")

        seasonality_results = {}
        category_col = (
            "first_category_id"
            if "first_category_id" in df.columns
            else df.select_dtypes(include=[np.number]).columns[1]
        )

        for category in df[category_col].unique():
            if pd.isna(category):
                continue

            cat_data = df[df[category_col] == category].copy()
            cat_data = cat_data.sort_values("sale_date")

            if len(cat_data) < 52:  # Need at least a year of data
                continue

            try:
                # Create time series
                ts_data = cat_data.set_index("sale_date")["category_sales"]
                ts_data = ts_data.asfreq("D", fill_value=0)  # Daily frequency

                # Seasonal decomposition
                if len(ts_data) >= 730:  # 2 years for yearly seasonality
                    decomposition = seasonal_decompose(
                        ts_data, model="additive", period=365
                    )
                elif len(ts_data) >= 365:  # 1 year for monthly seasonality
                    decomposition = seasonal_decompose(
                        ts_data, model="additive", period=30
                    )
                else:
                    decomposition = seasonal_decompose(
                        ts_data, model="additive", period=7
                    )

                # Extract patterns
                seasonality_results[str(category)] = {
                    "seasonal_strength": float(
                        np.std(decomposition.seasonal) / np.std(ts_data)
                    ),
                    "trend_strength": float(
                        np.std(decomposition.trend.dropna()) / np.std(ts_data)
                    ),
                    "monthly_pattern": self._extract_monthly_pattern(cat_data),
                    "weekly_pattern": self._extract_weekly_pattern(cat_data),
                    "holiday_impact": self._calculate_holiday_impact(cat_data),
                }

            except Exception as e:
                logger.warning(
                    f"Could not analyze seasonality for category {category}: {e}"
                )
                seasonality_results[str(category)] = {
                    "seasonal_strength": 0,
                    "trend_strength": 0,
                    "monthly_pattern": {},
                    "weekly_pattern": {},
                    "holiday_impact": 0,
                }

        self.seasonality_patterns = seasonality_results
        return seasonality_results

    def _extract_monthly_pattern(self, df: pd.DataFrame) -> Dict[int, float]:
        """Extract monthly seasonality pattern."""
        df["month"] = df["sale_date"].dt.month
        monthly_avg = df.groupby("month")["category_sales"].mean()
        overall_avg = df["category_sales"].mean()

        return {month: float(avg / overall_avg) for month, avg in monthly_avg.items()}

    def _extract_weekly_pattern(self, df: pd.DataFrame) -> Dict[int, float]:
        """Extract weekly seasonality pattern."""
        df["dayofweek"] = df["sale_date"].dt.dayofweek
        weekly_avg = df.groupby("dayofweek")["category_sales"].mean()
        overall_avg = df["category_sales"].mean()

        return {day: float(avg / overall_avg) for day, avg in weekly_avg.items()}

    def _calculate_holiday_impact(self, df: pd.DataFrame) -> float:
        """Calculate average holiday impact on category sales."""
        if "holiday_flag" not in df.columns:
            return 0.0

        holiday_sales = df[df["holiday_flag"] > 0]["category_sales"].mean()
        regular_sales = df[df["holiday_flag"] == 0]["category_sales"].mean()

        if regular_sales > 0:
            return float((holiday_sales - regular_sales) / regular_sales)
        return 0.0

    def prepare_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for category-level forecasting.

        Args:
            df: Category-level data

        Returns:
            DataFrame with prepared features
        """
        from services.feature_builder import AdvancedFeatureBuilder

        feature_builder = AdvancedFeatureBuilder()

        # Build category-specific features
        df = feature_builder.build_category_features(df)

        # Build temporal features
        df = feature_builder.build_temporal_features(df)

        # Build lag features for category sales
        df = feature_builder.build_lag_features(
            df,
            target_col="category_sales",
            group_cols=(
                ["first_category_id", "store_id"]
                if "first_category_id" in df.columns
                else ["store_id"]
            ),
        )

        # Add seasonality features
        df = self._add_seasonality_features(df)

        # Add cross-category features
        df = self._add_cross_category_features(df)

        return df

    def _add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonality features based on learned patterns."""
        df = df.copy()

        if hasattr(self, "seasonality_patterns") and self.seasonality_patterns:
            category_col = (
                "first_category_id"
                if "first_category_id" in df.columns
                else df.select_dtypes(include=[np.number]).columns[1]
            )

            # Monthly seasonality
            df["month"] = df["sale_date"].dt.month
            df["monthly_seasonality"] = df.apply(
                lambda row: self.seasonality_patterns.get(str(row[category_col]), {})
                .get("monthly_pattern", {})
                .get(row["month"], 1.0),
                axis=1,
            )

            # Weekly seasonality
            df["dayofweek"] = df["sale_date"].dt.dayofweek
            df["weekly_seasonality"] = df.apply(
                lambda row: self.seasonality_patterns.get(str(row[category_col]), {})
                .get("weekly_pattern", {})
                .get(row["dayofweek"], 1.0),
                axis=1,
            )

            # Holiday impact
            if "holiday_flag" in df.columns:
                df["expected_holiday_impact"] = df.apply(
                    lambda row: self.seasonality_patterns.get(
                        str(row[category_col]), {}
                    ).get("holiday_impact", 0.0)
                    * row["holiday_flag"],
                    axis=1,
                )

        return df

    def _add_cross_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-category interaction features."""
        df = df.copy()

        # Category correlation features (simplified)
        if "first_category_id" in df.columns:
            # Add category diversity in store
            store_category_count = df.groupby(["sale_date", "store_id"])[
                "first_category_id"
            ].nunique()
            df["store_category_count"] = df.set_index(
                ["sale_date", "store_id"]
            ).index.map(store_category_count)

            # Add total store sales for the day
            store_daily_sales = df.groupby(["sale_date", "store_id"])[
                "category_sales"
            ].sum()
            df["store_total_sales"] = df.set_index(["sale_date", "store_id"]).index.map(
                store_daily_sales
            )

            # Category's share of store sales
            df["category_store_share"] = df["category_sales"] / df[
                "store_total_sales"
            ].replace(0, np.nan)

        return df

    def fit(
        self, df: pd.DataFrame, target_col: str = "category_sales"
    ) -> Dict[str, Any]:
        """
        Fit category-level forecasting models.

        Args:
            df: Training data (should be category-level aggregated)
            target_col: Target column name

        Returns:
            Training results and metrics
        """
        logger.info("Training category-level forecasting models...")

        # Analyze seasonality first
        seasonality_results = self.analyze_category_seasonality(df)

        # Prepare features
        df_processed = self.prepare_category_features(df)

        # Get feature columns
        feature_cols = self._select_category_features(df_processed)

        category_col = (
            "first_category_id"
            if "first_category_id" in df.columns
            else df.select_dtypes(include=[np.number]).columns[1]
        )
        training_results = {}

        # Train separate model for each category
        for category in df[category_col].unique():
            if pd.isna(category):
                continue

            cat_data = df_processed[df_processed[category_col] == category].copy()

            if len(cat_data) < 50:  # Minimum data requirement
                logger.warning(
                    f"Insufficient data for category {category} ({len(cat_data)} records)"
                )
                continue

            logger.info(
                f"Training model for category {category} with {len(cat_data)} records"
            )

            # Prepare data
            X = cat_data[feature_cols].fillna(0)
            y = cat_data[target_col].fillna(0)

            # Time-based split for time series
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []

            # Initialize model
            if self.model_type == "gradient_boost":
                model = GradientBoostingRegressor(**self.base_model_config)
            elif self.model_type == "random_forest":
                model = RandomForestRegressor(**self.base_model_config)
            elif self.model_type == "linear":
                model = LinearRegression(**self.base_model_config)
            else:
                model = GradientBoostingRegressor(**self.base_model_config)

            # Initialize scaler
            scaler = StandardScaler()

            # Cross-validation
            for train_idx, test_idx in tscv.split(X):
                X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
                y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

                # Scale features
                X_train_scaled = scaler.fit_transform(X_train_cv)
                X_test_scaled = scaler.transform(X_test_cv)

                # Fit and predict
                model.fit(X_train_scaled, y_train_cv)
                y_pred_cv = model.predict(X_test_scaled)

                # Calculate score
                score = r2_score(y_test_cv, y_pred_cv)
                cv_scores.append(score)

            # Final training on all data
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)

            # Store model and scaler
            self.models[str(category)] = model
            self.scalers[str(category)] = scaler

            # Store metadata
            self.category_metadata[str(category)] = {
                "data_size": len(cat_data),
                "feature_count": len(feature_cols),
                "cv_scores": cv_scores,
                "average_cv_score": np.mean(cv_scores),
                "std_cv_score": np.std(cv_scores),
            }

            training_results[str(category)] = {
                "cv_score_mean": np.mean(cv_scores),
                "cv_score_std": np.std(cv_scores),
                "data_size": len(cat_data),
                "feature_importance": self._get_feature_importance(model, feature_cols),
            }

        self.feature_columns = feature_cols
        self.is_fitted = True

        logger.info(f"Trained models for {len(self.models)} categories")
        return training_results

    def predict_category_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict category-level demand.

        Args:
            df: Input data for prediction

        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        logger.info("Generating category-level demand predictions...")

        # Prepare features
        df_processed = self.prepare_category_features(df)

        # Prepare results
        results = df.copy()
        results["predicted_category_sales"] = 0.0
        results["prediction_confidence"] = 0.0

        category_col = (
            "first_category_id"
            if "first_category_id" in df.columns
            else df.select_dtypes(include=[np.number]).columns[1]
        )

        # Predict for each category
        for category in df[category_col].unique():
            if pd.isna(category):
                continue

            category_str = str(category)
            if category_str not in self.models:
                logger.warning(f"No trained model found for category {category}")
                continue

            # Filter data for this category
            cat_mask = df_processed[category_col] == category
            cat_data = df_processed[cat_mask]

            if len(cat_data) == 0:
                continue

            # Prepare features
            X = cat_data[self.feature_columns].fillna(0)

            # Scale and predict
            model = self.models[category_str]
            scaler = self.scalers[category_str]

            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)

            # Store predictions
            results.loc[cat_mask, "predicted_category_sales"] = predictions

            # Calculate confidence based on CV scores
            cv_score = self.category_metadata[category_str]["average_cv_score"]
            results.loc[cat_mask, "prediction_confidence"] = max(0, cv_score)

        return results

    def forecast_category_demand(
        self,
        categories: List[int],
        stores: List[int],
        start_date: str,
        periods: int = 30,
    ) -> pd.DataFrame:
        """
        Generate category demand forecasts.

        Args:
            categories: List of category IDs to forecast
            stores: List of store IDs to forecast
            start_date: Start date for forecast
            periods: Number of periods to forecast

        Returns:
            DataFrame with forecasts
        """
        logger.info(
            f"Generating category forecasts for {len(categories)} categories, {len(stores)} stores, {periods} periods"
        )

        # Create forecast framework
        forecast_dates = pd.date_range(start=start_date, periods=periods, freq="D")

        forecast_data = []
        for date in forecast_dates:
            for category in categories:
                for store in stores:
                    forecast_data.append(
                        {
                            "sale_date": date,
                            "first_category_id": category,
                            "store_id": store,
                            "city_id": 1,  # Would need to map store to city
                            "category_sales": 0,  # Placeholder
                            "holiday_flag": 0,  # Would need holiday calendar
                            "promo_flag": 0,
                        }
                    )

        forecast_df = pd.DataFrame(forecast_data)

        # Generate predictions
        forecasts = self.predict_category_demand(forecast_df)

        return forecasts

    def get_category_insights(self) -> Dict[str, Any]:
        """Get insights about category performance and patterns."""
        if not self.is_fitted:
            return {"error": "Model not fitted"}

        insights = {
            "category_count": len(self.models),
            "seasonality_patterns": self.seasonality_patterns,
            "model_performance": {},
            "category_rankings": {},
        }

        # Model performance summary
        for cat, metadata in self.category_metadata.items():
            insights["model_performance"][cat] = {
                "cv_score": metadata["average_cv_score"],
                "data_size": metadata["data_size"],
                "model_quality": (
                    "good"
                    if metadata["average_cv_score"] > 0.5
                    else "fair" if metadata["average_cv_score"] > 0.2 else "poor"
                ),
            }

        # Category rankings by various metrics
        performance_data = [
            (cat, data["average_cv_score"])
            for cat, data in self.category_metadata.items()
        ]
        performance_data.sort(key=lambda x: x[1], reverse=True)
        insights["category_rankings"]["by_model_performance"] = performance_data[:10]

        return insights

    def save_model(self, filename: Optional[str] = None):
        """Save the trained category models."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        if filename is None:
            filename = (
                f"category_forecaster_{self.aggregation_level}_{self.model_type}.joblib"
            )

        model_path = self.save_path / filename

        model_data = {
            "models": self.models,
            "scalers": self.scalers,
            "aggregation_level": self.aggregation_level,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "category_metadata": self.category_metadata,
            "seasonality_patterns": self.seasonality_patterns,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Category forecaster saved to {model_path}")

    def load_model(self, filename: str):
        """Load a trained category forecaster."""
        model_path = self.save_path / filename

        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)

        self.models = model_data["models"]
        self.scalers = model_data["scalers"]
        self.aggregation_level = model_data["aggregation_level"]
        self.model_type = model_data["model_type"]
        self.feature_columns = model_data["feature_columns"]
        self.category_metadata = model_data["category_metadata"]
        self.seasonality_patterns = model_data.get("seasonality_patterns", {})
        self.is_fitted = model_data["is_fitted"]

        logger.info(f"Category forecaster loaded from {model_path}")

    # Private helper methods
    def _select_category_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for category-level modeling."""
        # Base category features
        category_features = [
            "category_market_share",
            "category_velocity",
            "category_volatility",
            "category_trend",
        ]

        # Temporal features
        temporal_features = [
            "day_sin",
            "day_cos",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
            "quarter",
            "is_month_start",
            "is_month_end",
        ]

        # Lag features
        lag_features = [
            col
            for col in df.columns
            if "lag_" in col or "rolling_" in col or "trend_" in col
        ]

        # Seasonality features
        seasonality_features = [
            "monthly_seasonality",
            "weekly_seasonality",
            "expected_holiday_impact",
        ]

        # Cross-category features
        cross_features = [
            "store_category_count",
            "store_total_sales",
            "category_store_share",
        ]

        # Business features
        business_features = ["holiday_flag", "promo_flag", "discount_mean"]

        # Combine all features
        all_features = (
            category_features
            + temporal_features
            + lag_features
            + seasonality_features
            + cross_features
            + business_features
        )

        # Return only features that exist in the DataFrame
        return [col for col in all_features if col in df.columns]

    def _get_feature_importance(
        self, model, feature_columns: List[str]
    ) -> Dict[str, float]:
        """Extract feature importance from the model."""
        importance_dict = {}

        if hasattr(model, "feature_importances_"):
            # Tree-based models
            for i, feature in enumerate(feature_columns):
                if i < len(model.feature_importances_):
                    importance_dict[feature] = float(model.feature_importances_[i])
        elif hasattr(model, "coef_"):
            # Linear models
            for i, feature in enumerate(feature_columns):
                if i < len(model.coef_):
                    importance_dict[feature] = float(abs(model.coef_[i]))

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
