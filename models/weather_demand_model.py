"""
Weather-Sensitive Demand Modeling for retail forecasting.
Analyzes weather impact on demand patterns and provides weather-based demand forecasting.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # type: ignore
from sklearn.linear_model import ElasticNet  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import train_test_split, cross_val_score  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
import joblib  # type: ignore
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class WeatherSensitiveDemandModel:
    """Weather-sensitive demand forecasting model."""

    def __init__(
        self, model_type: str = "gradient_boost", save_path: str = "models/saved"
    ):
        """
        Initialize the weather-sensitive demand model.

        Args:
            model_type: Type of model ('gradient_boost', 'random_forest', 'elastic_net', 'ensemble')
            save_path: Path to save the model
        """
        self.model_type = model_type
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.scalers = {}
        self.weather_impact_coefficients = {}
        self.is_fitted = False

        # Initialize models based on type
        self._initialize_models()

    def _initialize_models(self):
        """Initialize the underlying models."""
        if self.model_type == "gradient_boost":
            self.models["main"] = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )
        elif self.model_type == "random_forest":
            self.models["main"] = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif self.model_type == "elastic_net":
            self.models["main"] = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
        elif self.model_type == "ensemble":
            self.models["gb"] = GradientBoostingRegressor(
                n_estimators=50, random_state=42
            )
            self.models["rf"] = RandomForestRegressor(n_estimators=50, random_state=42)
            self.models["en"] = ElasticNet(alpha=1.0, random_state=42)

        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()

    def analyze_weather_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze weather sensitivity across different products and categories.

        Args:
            df: DataFrame with sales and weather data

        Returns:
            Dictionary with weather sensitivity analysis
        """
        logger.info("Analyzing weather sensitivity patterns...")

        analysis_results = {
            "temperature_sensitivity": {},
            "humidity_sensitivity": {},
            "precipitation_sensitivity": {},
            "wind_sensitivity": {},
            "weather_elasticity": {},
            "seasonal_weather_patterns": {},
            "category_weather_correlation": {},
        }

        # Ensure required columns exist
        required_cols = [
            "avg_temperature",
            "avg_humidity",
            "precpt",
            "avg_wind_level",
            "sale_amount",
        ]
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for weather analysis")
            return analysis_results

        # Temperature sensitivity analysis
        analysis_results["temperature_sensitivity"] = (
            self._analyze_temperature_sensitivity(df)
        )

        # Humidity sensitivity analysis
        analysis_results["humidity_sensitivity"] = self._analyze_humidity_sensitivity(
            df
        )

        # Precipitation sensitivity analysis
        analysis_results["precipitation_sensitivity"] = (
            self._analyze_precipitation_sensitivity(df)
        )

        # Wind sensitivity analysis
        analysis_results["wind_sensitivity"] = self._analyze_wind_sensitivity(df)

        # Weather elasticity analysis
        analysis_results["weather_elasticity"] = self._calculate_weather_elasticity(df)

        # Seasonal weather patterns
        analysis_results["seasonal_weather_patterns"] = (
            self._analyze_seasonal_weather_patterns(df)
        )

        # Category-weather correlations
        if "first_category_id" in df.columns:
            analysis_results["category_weather_correlation"] = (
                self._analyze_category_weather_correlation(df)
            )

        return analysis_results

    def prepare_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare weather-specific features for modeling.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with weather features
        """
        from services.feature_builder import AdvancedFeatureBuilder

        feature_builder = AdvancedFeatureBuilder()

        # Build weather sensitivity features
        df = feature_builder.build_weather_sensitivity_features(df)

        # Add temporal features
        df = feature_builder.build_temporal_features(df)

        # Add lag features for weather variables
        weather_cols = ["avg_temperature", "avg_humidity", "precpt", "avg_wind_level"]
        for col in weather_cols:
            if col in df.columns:
                df = df.sort_values("sale_date")
                df[f"{col}_lag_1"] = df[col].shift(1)
                df[f"{col}_lag_7"] = df[col].shift(7)
                df[f"{col}_rolling_mean_7"] = df[col].rolling(7).mean()
                df[f"{col}_rolling_std_7"] = df[col].rolling(7).std()

        # Weather interaction features
        if all(col in df.columns for col in ["avg_temperature", "avg_humidity"]):
            df["temp_humidity_interaction"] = df["avg_temperature"] * df["avg_humidity"]

        if all(col in df.columns for col in ["precpt", "avg_wind_level"]):
            df["precip_wind_interaction"] = df["precpt"] * df["avg_wind_level"]

        # Weather deviation from normal
        if "avg_temperature" in df.columns:
            monthly_temp_avg = df.groupby(df["sale_date"].dt.month)[
                "avg_temperature"
            ].transform("mean")
            df["temp_deviation_from_normal"] = df["avg_temperature"] - monthly_temp_avg

        return df

    def fit(self, df: pd.DataFrame, target_col: str = "sale_amount") -> Dict[str, Any]:
        """
        Fit the weather-sensitive demand model.

        Args:
            df: Training data
            target_col: Target column name

        Returns:
            Training metrics and results
        """
        logger.info(
            f"Training weather-sensitive demand model with {len(df)} samples..."
        )

        # Prepare features
        df_processed = self.prepare_weather_features(df)

        # Select features for modeling
        feature_cols = self._select_weather_features(df_processed)

        # Prepare data
        X = df_processed[feature_cols].fillna(0)
        y = df_processed[target_col].fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        training_results = {}

        # Train models
        if self.model_type == "ensemble":
            # Train ensemble models
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name} model...")

                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)

                # Fit model
                model.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = model.predict(X_test_scaled)
                training_results[model_name] = {
                    "mae": mean_absolute_error(y_test, y_pred),
                    "mse": mean_squared_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                }
        else:
            # Train single model
            model = self.models["main"]
            X_train_scaled = self.scalers["main"].fit_transform(X_train)
            X_test_scaled = self.scalers["main"].transform(X_test)

            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            training_results["main"] = {
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
            }

        # Analyze weather impact coefficients
        self.weather_impact_coefficients = self._extract_weather_coefficients(X.columns)

        self.is_fitted = True
        self.feature_columns = feature_cols

        logger.info("Weather-sensitive demand model training completed")
        return training_results

    def predict_weather_demand(
        self, df: pd.DataFrame, weather_scenarios: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Predict demand based on weather conditions.

        Args:
            df: Input data for prediction
            weather_scenarios: Optional weather scenarios to test

        Returns:
            DataFrame with demand predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare features
        df_processed = self.prepare_weather_features(df)

        # Select features
        X = df_processed[self.feature_columns].fillna(0)

        predictions = []

        if self.model_type == "ensemble":
            # Ensemble predictions
            ensemble_preds = {}
            for model_name, model in self.models.items():
                X_scaled = self.scalers[model_name].transform(X)
                ensemble_preds[model_name] = model.predict(X_scaled)

            # Simple averaging ensemble
            final_pred = np.mean(list(ensemble_preds.values()), axis=0)
            predictions = final_pred
        else:
            # Single model prediction
            X_scaled = self.scalers["main"].transform(X)
            predictions = self.models["main"].predict(X_scaled)

        # Create results DataFrame
        results = df.copy()
        results["predicted_demand"] = predictions

        # Add weather scenario predictions if provided
        if weather_scenarios:
            for i, scenario in enumerate(weather_scenarios):
                scenario_df = self._apply_weather_scenario(df, scenario)
                scenario_processed = self.prepare_weather_features(scenario_df)
                scenario_X = scenario_processed[self.feature_columns].fillna(0)

                if self.model_type == "ensemble":
                    scenario_preds = []
                    for model_name, model in self.models.items():
                        X_scaled = self.scalers[model_name].transform(scenario_X)
                        scenario_preds.append(model.predict(X_scaled))
                    scenario_prediction = np.mean(scenario_preds, axis=0)
                else:
                    X_scaled = self.scalers["main"].transform(scenario_X)
                    scenario_prediction = self.models["main"].predict(X_scaled)

                results[f"scenario_{i+1}_demand"] = scenario_prediction

        return results

    def calculate_weather_impact(
        self, df: pd.DataFrame, weather_var: str, impact_range: Tuple[float, float]
    ) -> pd.DataFrame:
        """
        Calculate the impact of weather variable changes on demand.

        Args:
            df: Input data
            weather_var: Weather variable to analyze
            impact_range: Range of values to test (min, max)

        Returns:
            DataFrame with impact analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating impact")

        if weather_var not in df.columns:
            raise ValueError(f"Weather variable '{weather_var}' not found in data")

        # Create scenarios with different weather values
        test_values = np.linspace(impact_range[0], impact_range[1], 20)
        impact_results = []

        # Get baseline prediction
        baseline_pred = self.predict_weather_demand(df)["predicted_demand"].mean()

        for value in test_values:
            # Create scenario
            scenario_df = df.copy()
            scenario_df[weather_var] = value

            # Predict
            scenario_pred = self.predict_weather_demand(scenario_df)[
                "predicted_demand"
            ].mean()

            impact_results.append(
                {
                    weather_var: value,
                    "predicted_demand": scenario_pred,
                    "demand_change": scenario_pred - baseline_pred,
                    "demand_change_pct": (
                        (scenario_pred - baseline_pred) / baseline_pred
                    )
                    * 100,
                }
            )

        return pd.DataFrame(impact_results)

    def save_model(self, filename: Optional[str] = None):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        if filename is None:
            filename = f"weather_demand_model_{self.model_type}.joblib"

        model_path = self.save_path / filename

        model_data = {
            "models": self.models,
            "scalers": self.scalers,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "weather_impact_coefficients": self.weather_impact_coefficients,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Weather demand model saved to {model_path}")

    def load_model(self, filename: str):
        """Load a trained model."""
        model_path = self.save_path / filename

        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)

        self.models = model_data["models"]
        self.scalers = model_data["scalers"]
        self.model_type = model_data["model_type"]
        self.feature_columns = model_data["feature_columns"]
        self.weather_impact_coefficients = model_data.get(
            "weather_impact_coefficients", {}
        )
        self.is_fitted = model_data["is_fitted"]

        logger.info(f"Weather demand model loaded from {model_path}")

    # Private helper methods
    def _select_weather_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for weather modeling."""
        # Base weather features
        weather_features = [
            "avg_temperature",
            "avg_humidity",
            "precpt",
            "avg_wind_level",
            "temperature_demand_impact",
            "humidity_demand_impact",
            "precipitation_demand_impact",
            "wind_demand_impact",
            "weather_comfort_index",
            "weather_seasonality",
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

        # Lag and rolling features
        lag_features = [col for col in df.columns if "lag_" in col or "rolling_" in col]

        # Interaction features
        interaction_features = [col for col in df.columns if "interaction" in col]

        # Combine all feature types
        all_features = (
            weather_features + temporal_features + lag_features + interaction_features
        )

        # Return only features that exist in the DataFrame
        return [col for col in all_features if col in df.columns]

    def _analyze_temperature_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temperature sensitivity patterns."""
        temp_bins = pd.cut(df["avg_temperature"], bins=10, labels=False)
        temp_sensitivity = df.groupby(temp_bins)["sale_amount"].agg(
            ["mean", "std", "count"]
        )

        # Calculate correlation
        temp_correlation = df["avg_temperature"].corr(df["sale_amount"])

        # Find optimal temperature range
        optimal_temp_bin = temp_sensitivity["mean"].idxmax()
        temp_ranges = pd.cut(df["avg_temperature"], bins=10)
        optimal_temp_range = temp_ranges.cat.categories[optimal_temp_bin]

        return {
            "correlation": temp_correlation,
            "sensitivity_by_range": temp_sensitivity.to_dict(),
            "optimal_temperature_range": f"{optimal_temp_range.left:.1f}-{optimal_temp_range.right:.1f}°C",
            "temperature_elasticity": self._calculate_elasticity(
                df, "avg_temperature", "sale_amount"
            ),
        }

    def _analyze_humidity_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze humidity sensitivity patterns."""
        humidity_bins = pd.cut(df["avg_humidity"], bins=10, labels=False)
        humidity_sensitivity = df.groupby(humidity_bins)["sale_amount"].agg(
            ["mean", "std", "count"]
        )

        humidity_correlation = df["avg_humidity"].corr(df["sale_amount"])

        optimal_humidity_bin = humidity_sensitivity["mean"].idxmax()
        humidity_ranges = pd.cut(df["avg_humidity"], bins=10)
        optimal_humidity_range = humidity_ranges.cat.categories[optimal_humidity_bin]

        return {
            "correlation": humidity_correlation,
            "sensitivity_by_range": humidity_sensitivity.to_dict(),
            "optimal_humidity_range": f"{optimal_humidity_range.left:.1f}-{optimal_humidity_range.right:.1f}%",
            "humidity_elasticity": self._calculate_elasticity(
                df, "avg_humidity", "sale_amount"
            ),
        }

    def _analyze_precipitation_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze precipitation sensitivity patterns."""
        # Create precipitation categories
        df_temp = df.copy()
        df_temp["precip_category"] = pd.cut(
            df_temp["precpt"],
            bins=[-0.1, 0, 5, 15, float("inf")],
            labels=["No Rain", "Light Rain", "Moderate Rain", "Heavy Rain"],
        )

        precip_sensitivity = df_temp.groupby("precip_category")["sale_amount"].agg(
            ["mean", "std", "count"]
        )
        precip_correlation = df["precpt"].corr(df["sale_amount"])

        return {
            "correlation": precip_correlation,
            "sensitivity_by_category": precip_sensitivity.to_dict(),
            "precipitation_elasticity": self._calculate_elasticity(
                df, "precpt", "sale_amount"
            ),
        }

    def _analyze_wind_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze wind sensitivity patterns."""
        wind_bins = pd.cut(df["avg_wind_level"], bins=5, labels=False)
        wind_sensitivity = df.groupby(wind_bins)["sale_amount"].agg(
            ["mean", "std", "count"]
        )

        wind_correlation = df["avg_wind_level"].corr(df["sale_amount"])

        return {
            "correlation": wind_correlation,
            "sensitivity_by_range": wind_sensitivity.to_dict(),
            "wind_elasticity": self._calculate_elasticity(
                df, "avg_wind_level", "sale_amount"
            ),
        }

    def _calculate_weather_elasticity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate weather elasticity measures."""
        weather_vars = ["avg_temperature", "avg_humidity", "precpt", "avg_wind_level"]
        elasticity = {}

        for var in weather_vars:
            if var in df.columns:
                elasticity[var] = self._calculate_elasticity(df, var, "sale_amount")

        return elasticity

    def _calculate_elasticity(self, df: pd.DataFrame, x_var: str, y_var: str) -> float:
        """Calculate elasticity between two variables."""
        try:
            # Calculate percentage changes
            x_pct_change = df[x_var].pct_change().dropna()
            y_pct_change = df[y_var].pct_change().dropna()

            # Align the series
            min_len = min(len(x_pct_change), len(y_pct_change))
            x_pct_change = x_pct_change.iloc[:min_len]
            y_pct_change = y_pct_change.iloc[:min_len]

            # Calculate elasticity as correlation of percentage changes
            if len(x_pct_change) > 1 and x_pct_change.std() != 0:
                elasticity = (y_pct_change.std() / x_pct_change.std()) * (
                    x_pct_change.corr(y_pct_change)
                )
                return elasticity if not np.isnan(elasticity) else 0.0
            else:
                return 0.0
        except:
            return 0.0

    def _analyze_seasonal_weather_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal weather patterns and their impact on sales."""
        df_temp = df.copy()
        df_temp["month"] = df_temp["sale_date"].dt.month
        df_temp["season"] = df_temp["month"].map(
            {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Fall",
                10: "Fall",
                11: "Fall",
            }
        )

        seasonal_patterns = {}

        # Analyze by season
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            season_data = df_temp[df_temp["season"] == season]
            if len(season_data) > 0:
                seasonal_patterns[season] = {
                    "avg_temperature": season_data["avg_temperature"].mean(),
                    "avg_humidity": season_data["avg_humidity"].mean(),
                    "avg_precipitation": season_data["precpt"].mean(),
                    "avg_sales": season_data["sale_amount"].mean(),
                    "sales_std": season_data["sale_amount"].std(),
                }

        return seasonal_patterns

    def _analyze_category_weather_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weather correlations by product category."""
        category_correlations = {}

        weather_vars = ["avg_temperature", "avg_humidity", "precpt", "avg_wind_level"]

        for category in df["first_category_id"].unique():
            if pd.isna(category):
                continue

            category_data = df[df["first_category_id"] == category]

            if (
                len(category_data) > 10
            ):  # Minimum data points for meaningful correlation
                category_correlations[str(category)] = {}

                for var in weather_vars:
                    if var in category_data.columns:
                        correlation = category_data[var].corr(
                            category_data["sale_amount"]
                        )
                        category_correlations[str(category)][var] = (
                            correlation if not np.isnan(correlation) else 0.0
                        )

        return category_correlations

    def _extract_weather_coefficients(
        self, feature_columns: List[str]
    ) -> Dict[str, float]:
        """Extract weather impact coefficients from trained models."""
        coefficients = {}

        if self.model_type == "elastic_net":
            # For ElasticNet, we can get coefficients directly
            model = self.models["main"]
            if hasattr(model, "coef_"):
                weather_features = [
                    col
                    for col in feature_columns
                    if any(
                        weather in col.lower()
                        for weather in [
                            "temperature",
                            "humidity",
                            "precpt",
                            "wind",
                            "weather",
                        ]
                    )
                ]

                for i, feature in enumerate(feature_columns):
                    if feature in weather_features and i < len(model.coef_):
                        coefficients[feature] = model.coef_[i]

        elif self.model_type in ["gradient_boost", "random_forest"]:
            # For tree-based models, use feature importance
            model = self.models["main"]
            if hasattr(model, "feature_importances_"):
                weather_features = [
                    col
                    for col in feature_columns
                    if any(
                        weather in col.lower()
                        for weather in [
                            "temperature",
                            "humidity",
                            "precpt",
                            "wind",
                            "weather",
                        ]
                    )
                ]

                for i, feature in enumerate(feature_columns):
                    if feature in weather_features and i < len(
                        model.feature_importances_
                    ):
                        coefficients[feature] = model.feature_importances_[i]

        return coefficients

    def _apply_weather_scenario(
        self, df: pd.DataFrame, scenario: Dict[str, float]
    ) -> pd.DataFrame:
        """Apply a weather scenario to the data."""
        scenario_df = df.copy()

        for weather_var, value in scenario.items():
            if weather_var in scenario_df.columns:
                scenario_df[weather_var] = value

        return scenario_df
