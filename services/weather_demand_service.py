"""
Weather Demand Service for weather-sensitive demand analysis and forecasting.
Provides high-level interface for weather impact analysis and demand prediction.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import json

# Import custom modules
from models.weather_demand_model import WeatherSensitiveDemandModel
from database.connection import get_pool, cached, paginate
from services.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class WeatherDemandService:
    """Service for weather-sensitive demand analysis and forecasting."""

    def __init__(self):
        """Initialize the weather demand service."""
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.cache_duration = 3600  # 1 hour cache

    def get_or_create_model(
        self, model_type: str = "gradient_boost"
    ) -> WeatherSensitiveDemandModel:
        """Get or create a weather demand model."""
        if self.model is None or self.model.model_type != model_type:
            self.model = WeatherSensitiveDemandModel(model_type=model_type)

            # Try to load existing model
            try:
                model_filename = f"weather_demand_model_{model_type}.joblib"
                self.model.load_model(model_filename)
                logger.info(f"Loaded existing weather demand model: {model_type}")
            except (ValueError, FileNotFoundError):
                logger.info(
                    f"No existing model found for {model_type}, will need to train"
                )

        return self.model

    @cached("weather_data")
    async def fetch_weather_sales_data(
        self,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        category_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch sales data with weather information.

        Args:
            store_id: Filter by store ID
            product_id: Filter by product ID
            category_id: Filter by category ID
            city_id: Filter by city ID
            start_date: Start date for data
            end_date: End date for data
            limit: Limit number of results
            offset: Offset for pagination

        Returns:
            List of dictionaries with sales and weather data
        """
        pool = await get_pool()

        query = """
        SELECT 
            sd.dt as sale_date,
            sd.store_id,
            sd.product_id,
            ph.first_category_id,
            sd.city_id,
            sd.sale_amount,
            sd.discount,
            sd.stock_hour6_22_cnt,
            sd.holiday_flag,
            sd.activity_flag as promo_flag,
            sd.avg_temperature,
            sd.avg_humidity,
            sd.precpt,
            sd.avg_wind_level,
            ph.product_name,
            sh.store_name
        FROM sales_data sd
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        JOIN product_hierarchy ph ON sd.product_id = ph.product_id
        WHERE 1=1
        """

        params = []
        param_count = 0

        if store_id is not None:
            param_count += 1
            query += f" AND sd.store_id = ${param_count}"
            params.append(store_id)

        if product_id is not None:
            param_count += 1
            query += f" AND sd.product_id = ${param_count}"
            params.append(product_id)

        if category_id is not None:
            param_count += 1
            query += f" AND ph.first_category_id = ${param_count}"
            params.append(category_id)

        if city_id is not None:
            param_count += 1
            query += f" AND sd.city_id = ${param_count}"
            params.append(city_id)

        if start_date is not None:
            param_count += 1
            query += f" AND sd.dt >= ${param_count}"
            params.append(start_date)

        if end_date is not None:
            param_count += 1
            query += f" AND sd.dt <= ${param_count}"
            params.append(end_date)

        query += " ORDER BY sd.dt, sd.store_id, sd.product_id"

        if limit is not None:
            param_count += 1
            query += f" LIMIT ${param_count}"
            params.append(limit)

        if offset is not None:
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(offset)

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            return [dict(row) for row in rows]

    @cached("weather_analysis")
    async def analyze_weather_sensitivity(
        self,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        category_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze weather sensitivity for specified filters.

        Args:
            store_id: Store ID filter
            product_id: Product ID filter
            category_id: Category ID filter
            city_id: City ID filter
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Weather sensitivity analysis results
        """
        logger.info("Starting weather sensitivity analysis...")

        try:
            # Fetch data
            data = await self.fetch_weather_sales_data(
                store_id=store_id,
                product_id=product_id,
                category_id=category_id,
                city_id=city_id,
                start_date=start_date,
                end_date=end_date,
            )

            if not data:
                return {"error": "No data found for the specified criteria"}

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Preprocess data
            df = self.preprocessor.handle_missing_values(df)
            df = self.preprocessor.add_time_features(df)
            df = self.preprocessor.add_weather_features(df)

            # Get or create model
            model = self.get_or_create_model("gradient_boost")

            # Analyze weather sensitivity
            analysis_results = model.analyze_weather_sensitivity(df)

            # Add summary statistics
            analysis_results["data_summary"] = {
                "total_records": len(df),
                "date_range": {
                    "start": (
                        df["sale_date"].min().isoformat() if not df.empty else None
                    ),
                    "end": df["sale_date"].max().isoformat() if not df.empty else None,
                },
                "average_sales": float(df["sale_amount"].mean()) if not df.empty else 0,
                "weather_ranges": {
                    "temperature": {
                        "min": (
                            float(df["avg_temperature"].min())
                            if "avg_temperature" in df.columns
                            else None
                        ),
                        "max": (
                            float(df["avg_temperature"].max())
                            if "avg_temperature" in df.columns
                            else None
                        ),
                        "avg": (
                            float(df["avg_temperature"].mean())
                            if "avg_temperature" in df.columns
                            else None
                        ),
                    },
                    "humidity": {
                        "min": (
                            float(df["avg_humidity"].min())
                            if "avg_humidity" in df.columns
                            else None
                        ),
                        "max": (
                            float(df["avg_humidity"].max())
                            if "avg_humidity" in df.columns
                            else None
                        ),
                        "avg": (
                            float(df["avg_humidity"].mean())
                            if "avg_humidity" in df.columns
                            else None
                        ),
                    },
                    "precipitation": {
                        "min": (
                            float(df["precpt"].min())
                            if "precpt" in df.columns
                            else None
                        ),
                        "max": (
                            float(df["precpt"].max())
                            if "precpt" in df.columns
                            else None
                        ),
                        "avg": (
                            float(df["precpt"].mean())
                            if "precpt" in df.columns
                            else None
                        ),
                    },
                },
            }

            logger.info("Weather sensitivity analysis completed successfully")
            return analysis_results

        except Exception as e:
            logger.error(f"Error in weather sensitivity analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    async def train_weather_model(
        self,
        model_type: str = "gradient_boost",
        store_id: Optional[int] = None,
        category_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a weather-sensitive demand model.

        Args:
            model_type: Type of model to train
            store_id: Store ID filter for training data
            category_id: Category ID filter for training data
            city_id: City ID filter for training data
            start_date: Start date for training data
            end_date: End date for training data

        Returns:
            Training results and metrics
        """
        logger.info(f"Training weather demand model: {model_type}")

        try:
            # Fetch training data
            data = await self.fetch_weather_sales_data(
                store_id=store_id,
                category_id=category_id,
                city_id=city_id,
                start_date=start_date,
                end_date=end_date,
                limit=50000,  # Limit for training
            )

            if not data:
                return {"error": "No training data found"}

            if len(data) < 100:
                return {
                    "error": "Insufficient training data (minimum 100 records required)"
                }

            # Convert to DataFrame and preprocess
            df = pd.DataFrame(data)
            df = self.preprocessor.handle_missing_values(df)
            df = self.preprocessor.add_time_features(df)
            df = self.preprocessor.add_weather_features(df)
            df = self.preprocessor.handle_outliers(df)

            # Create and train model
            model = WeatherSensitiveDemandModel(model_type=model_type)
            training_results = model.fit(df, target_col="sale_amount")

            # Save model
            model.save_model()

            # Update service model
            self.model = model

            logger.info("Weather demand model training completed successfully")

            return {
                "status": "success",
                "model_type": model_type,
                "training_data_size": len(df),
                "training_metrics": training_results,
                "weather_impact_coefficients": model.weather_impact_coefficients,
            }

        except Exception as e:
            logger.error(f"Error training weather model: {str(e)}")
            return {"error": f"Training failed: {str(e)}"}

    async def forecast_weather_demand(
        self,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        category_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: str = None,
        periods: int = 30,
        weather_scenarios: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate weather-sensitive demand forecasts.

        Args:
            store_id: Store ID for forecasting
            product_id: Product ID for forecasting
            category_id: Category ID for forecasting
            city_id: City ID for forecasting
            start_date: Start date for forecast
            periods: Number of periods to forecast
            weather_scenarios: List of weather scenarios to test

        Returns:
            Forecast results with weather scenarios
        """
        logger.info("Generating weather-sensitive demand forecasts...")

        try:
            # Get model
            model = self.get_or_create_model()

            if not model.is_fitted:
                return {"error": "Model is not trained. Please train the model first."}

            # Fetch recent historical data for context
            historical_end = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(
                days=1
            )
            historical_start = historical_end - timedelta(
                days=90
            )  # 3 months of history

            historical_data = await self.fetch_weather_sales_data(
                store_id=store_id,
                product_id=product_id,
                category_id=category_id,
                city_id=city_id,
                start_date=historical_start.strftime("%Y-%m-%d"),
                end_date=historical_end.strftime("%Y-%m-%d"),
            )

            if not historical_data:
                return {"error": "No historical data found for forecasting context"}

            # Create forecast periods
            forecast_dates = pd.date_range(start=start_date, periods=periods, freq="D")

            # Create base forecast DataFrame
            forecast_df = pd.DataFrame(
                {
                    "sale_date": forecast_dates,
                    "store_id": (
                        store_id if store_id else historical_data[0].get("store_id")
                    ),
                    "product_id": (
                        product_id
                        if product_id
                        else historical_data[0].get("product_id")
                    ),
                    "first_category_id": (
                        category_id
                        if category_id
                        else historical_data[0].get("first_category_id")
                    ),
                    "city_id": (
                        city_id if city_id else historical_data[0].get("city_id")
                    ),
                }
            )

            # Add average weather conditions (you might want to fetch actual weather forecasts)
            historical_df = pd.DataFrame(historical_data)
            if not historical_df.empty:
                # Use historical averages for weather (in real implementation, use weather API)
                avg_weather = {
                    "avg_temperature": historical_df["avg_temperature"].mean(),
                    "avg_humidity": historical_df["avg_humidity"].mean(),
                    "precpt": historical_df["precpt"].mean(),
                    "avg_wind_level": historical_df["avg_wind_level"].mean(),
                }

                for col, value in avg_weather.items():
                    forecast_df[col] = value

            # Add other required columns
            forecast_df["holiday_flag"] = 0  # Would need holiday calendar integration
            forecast_df["promo_flag"] = 0
            forecast_df["discount"] = 0
            forecast_df["stock_hour6_22_cnt"] = (
                historical_df["stock_hour6_22_cnt"].mean()
                if not historical_df.empty
                else 10
            )

            # Generate forecasts
            forecast_results = model.predict_weather_demand(
                forecast_df, weather_scenarios
            )

            # Convert to list format for API response
            forecasts = []
            for _, row in forecast_results.iterrows():
                forecast_item = {
                    "date": row["sale_date"].isoformat(),
                    "predicted_demand": float(row["predicted_demand"]),
                    "weather_conditions": {
                        "temperature": float(row.get("avg_temperature", 0)),
                        "humidity": float(row.get("avg_humidity", 0)),
                        "precipitation": float(row.get("precpt", 0)),
                        "wind_level": float(row.get("avg_wind_level", 0)),
                    },
                }

                # Add scenario predictions if available
                scenario_predictions = {}
                for col in row.index:
                    if col.startswith("scenario_") and col.endswith("_demand"):
                        scenario_num = col.replace("scenario_", "").replace(
                            "_demand", ""
                        )
                        scenario_predictions[f"scenario_{scenario_num}"] = float(
                            row[col]
                        )

                if scenario_predictions:
                    forecast_item["scenario_predictions"] = scenario_predictions

                forecasts.append(forecast_item)

            logger.info("Weather-sensitive demand forecasting completed successfully")

            return {
                "status": "success",
                "forecasts": forecasts,
                "forecast_summary": {
                    "total_periods": periods,
                    "average_predicted_demand": float(
                        forecast_results["predicted_demand"].mean()
                    ),
                    "demand_range": {
                        "min": float(forecast_results["predicted_demand"].min()),
                        "max": float(forecast_results["predicted_demand"].max()),
                    },
                },
                "weather_scenarios_tested": (
                    len(weather_scenarios) if weather_scenarios else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error in weather demand forecasting: {str(e)}")
            return {"error": f"Forecasting failed: {str(e)}"}

    async def analyze_weather_impact(
        self,
        weather_variable: str,
        impact_range: Tuple[float, float],
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        category_id: Optional[int] = None,
        city_id: Optional[int] = None,
        reference_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the impact of weather variable changes on demand.

        Args:
            weather_variable: Weather variable to analyze
            impact_range: Range of values to test
            store_id: Store ID filter
            product_id: Product ID filter
            category_id: Category ID filter
            city_id: City ID filter
            reference_date: Reference date for analysis

        Returns:
            Impact analysis results
        """
        logger.info(f"Analyzing weather impact for {weather_variable}")

        try:
            # Get model
            model = self.get_or_create_model()

            if not model.is_fitted:
                return {"error": "Model is not trained. Please train the model first."}

            # Get reference data
            if reference_date is None:
                reference_date = (datetime.now() - timedelta(days=30)).strftime(
                    "%Y-%m-%d"
                )

            reference_data = await self.fetch_weather_sales_data(
                store_id=store_id,
                product_id=product_id,
                category_id=category_id,
                city_id=city_id,
                start_date=reference_date,
                end_date=reference_date,
                limit=1,
            )

            if not reference_data:
                # Create synthetic reference data
                reference_df = pd.DataFrame(
                    [
                        {
                            "sale_date": pd.to_datetime(reference_date),
                            "store_id": store_id or 1,
                            "product_id": product_id or 1,
                            "first_category_id": category_id or 1,
                            "city_id": city_id or 1,
                            "avg_temperature": 22.0,
                            "avg_humidity": 50.0,
                            "precpt": 0.0,
                            "avg_wind_level": 5.0,
                            "holiday_flag": 0,
                            "promo_flag": 0,
                            "discount": 0,
                            "stock_hour6_22_cnt": 10,
                        }
                    ]
                )
            else:
                reference_df = pd.DataFrame(reference_data)

            # Analyze impact
            impact_results = model.calculate_weather_impact(
                reference_df, weather_variable, impact_range
            )

            # Convert to API response format
            impact_analysis = []
            for _, row in impact_results.iterrows():
                impact_analysis.append(
                    {
                        "weather_value": float(row[weather_variable]),
                        "predicted_demand": float(row["predicted_demand"]),
                        "demand_change": float(row["demand_change"]),
                        "demand_change_percentage": float(row["demand_change_pct"]),
                    }
                )

            # Calculate summary statistics
            max_impact = impact_results.loc[
                impact_results["demand_change_pct"].abs().idxmax()
            ]
            optimal_value = impact_results.loc[
                impact_results["predicted_demand"].idxmax()
            ]

            logger.info("Weather impact analysis completed successfully")

            return {
                "status": "success",
                "weather_variable": weather_variable,
                "impact_range": impact_range,
                "impact_analysis": impact_analysis,
                "summary": {
                    "maximum_impact": {
                        "weather_value": float(max_impact[weather_variable]),
                        "demand_change_percentage": float(
                            max_impact["demand_change_pct"]
                        ),
                    },
                    "optimal_weather_value": {
                        "weather_value": float(optimal_value[weather_variable]),
                        "predicted_demand": float(optimal_value["predicted_demand"]),
                    },
                    "sensitivity_score": float(
                        impact_results["demand_change_pct"].std()
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error in weather impact analysis: {str(e)}")
            return {"error": f"Impact analysis failed: {str(e)}"}

    @paginate()
    async def get_weather_correlations(
        self,
        city_id: Optional[int] = None,
        category_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get weather-sales correlations for different segments.

        Args:
            city_id: City ID filter
            category_id: Category ID filter
            start_date: Start date for analysis
            end_date: End date for analysis
            limit: Limit for pagination
            offset: Offset for pagination

        Returns:
            Weather correlation data
        """
        pool = await get_pool()

        query = """
        SELECT 
            sh.city_id,
            ph.first_category_id,
            CORR(sd.sale_amount, wd.avg_temperature) as temperature_correlation,
            CORR(sd.sale_amount, wd.avg_humidity) as humidity_correlation,
            CORR(sd.sale_amount, wd.precpt) as precipitation_correlation,
            CORR(sd.sale_amount, wd.avg_wind_level) as wind_correlation,
            COUNT(*) as sample_size,
            AVG(sd.sale_amount) as avg_sales,
            AVG(wd.avg_temperature) as avg_temperature,
            AVG(wd.avg_humidity) as avg_humidity
        FROM sales_data sd
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        JOIN product_hierarchy ph ON sd.product_id = ph.product_id
        LEFT JOIN weather_data wd ON sd.sale_date = wd.date AND sh.city_id = wd.city_id
        WHERE wd.date IS NOT NULL
        """

        params = []
        param_count = 0

        if city_id is not None:
            param_count += 1
            query += f" AND sh.city_id = ${param_count}"
            params.append(city_id)

        if category_id is not None:
            param_count += 1
            query += f" AND ph.first_category_id = ${param_count}"
            params.append(category_id)

        if start_date is not None:
            param_count += 1
            query += f" AND sd.sale_date >= ${param_count}"
            params.append(start_date)

        if end_date is not None:
            param_count += 1
            query += f" AND sd.sale_date <= ${param_count}"
            params.append(end_date)

        query += """
        GROUP BY sh.city_id, ph.first_category_id
        HAVING COUNT(*) >= 30
        ORDER BY ABS(CORR(sd.sale_amount, wd.avg_temperature)) DESC
        """

        if limit is not None:
            param_count += 1
            query += f" LIMIT ${param_count}"
            params.append(limit)

        if offset is not None:
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(offset)

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            return [dict(row) for row in rows]
