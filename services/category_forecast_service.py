"""
Category Forecasting Service for category-level demand analysis and forecasting.
Provides hierarchical forecasting and category performance insights.
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
from models.category_forecaster import CategoryLevelForecaster
from database.connection import get_pool, cached, paginate
from services.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class CategoryForecastService:
    """Service for category-level demand forecasting and analysis."""

    def __init__(self):
        """Initialize the category forecast service."""
        self.models = {}  # Multiple models for different aggregation levels
        self.preprocessor = DataPreprocessor()
        self.cache_duration = 3600  # 1 hour cache

    def get_or_create_model(
        self, aggregation_level: str = "category", model_type: str = "gradient_boost"
    ) -> CategoryLevelForecaster:
        """Get or create a category forecasting model."""
        model_key = f"{aggregation_level}_{model_type}"

        if model_key not in self.models:
            self.models[model_key] = CategoryLevelForecaster(
                aggregation_level=aggregation_level, model_type=model_type
            )

            # Try to load existing model
            try:
                model_filename = (
                    f"category_forecaster_{aggregation_level}_{model_type}.joblib"
                )
                self.models[model_key].load_model(model_filename)
                logger.info(f"Loaded existing category forecaster: {model_key}")
            except (ValueError, FileNotFoundError):
                logger.info(
                    f"No existing model found for {model_key}, will need to train"
                )

        return self.models[model_key]

    @cached("category_sales_data")
    async def fetch_category_sales_data(
        self,
        category_id: Optional[int] = None,
        store_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregation_level: str = "category",
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch product-level sales data for category aggregation.

        Args:
            category_id: Filter by category ID
            store_id: Filter by store ID
            city_id: Filter by city ID
            start_date: Start date for data
            end_date: End date for data
            aggregation_level: Level of aggregation
            limit: Limit number of results
            offset: Offset for pagination

        Returns:
            List of dictionaries with sales data
        """
        pool = await get_pool()

        # Base query for product-level data
        query = """
        SELECT 
            sd.sale_date,
            sd.store_id,
            sd.product_id,
            ph.first_category_id,
            ph.second_category_id,
            ph.brand_id,
            sh.city_id,
            sd.sale_amount,
            sd.sale_qty,
            sd.discount,
            sd.original_price,
            sd.stock_hour6_22_cnt,
            sd.holiday_flag,
            sd.promo_flag,
            sd.hours_sale,
            sd.hours_stock_status,
            wd.avg_temperature,
            wd.avg_humidity,
            wd.precpt,
            wd.avg_wind_level
        FROM sales_data sd
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        JOIN product_hierarchy ph ON sd.product_id = ph.product_id
        LEFT JOIN weather_data wd ON sd.sale_date = wd.date AND sh.city_id = wd.city_id
        WHERE 1=1
        """

        params: List[Any] = []
        param_count = 0

        if category_id is not None:
            param_count += 1
            query += f" AND ph.first_category_id = ${param_count}"
            params.append(category_id)

        if store_id is not None:
            param_count += 1
            query += f" AND sd.store_id = ${param_count}"
            params.append(store_id)

        if city_id is not None:
            param_count += 1
            query += f" AND sh.city_id = ${param_count}"
            params.append(city_id)

        if start_date is not None:
            param_count += 1
            query += f" AND sd.sale_date >= ${param_count}"
            params.append(start_date)

        if end_date is not None:
            param_count += 1
            query += f" AND sd.sale_date <= ${param_count}"
            params.append(end_date)

        query += " ORDER BY sd.sale_date, ph.first_category_id, sd.store_id"

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

    async def aggregate_category_data(
        self,
        category_id: Optional[int] = None,
        store_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregation_level: str = "category",
    ) -> pd.DataFrame:
        """
        Fetch and aggregate data to category level.

        Args:
            category_id: Filter by category ID
            store_id: Filter by store ID
            city_id: Filter by city ID
            start_date: Start date for data
            end_date: End date for data
            aggregation_level: Level of aggregation

        Returns:
            Aggregated category-level DataFrame
        """
        # Fetch product-level data
        data = await self.fetch_category_sales_data(
            category_id=category_id,
            store_id=store_id,
            city_id=city_id,
            start_date=start_date,
            end_date=end_date,
            aggregation_level=aggregation_level,
            limit=100000,  # Large limit for aggregation
        )

        if not data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Preprocess data
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.add_time_features(df)

        # Get model for aggregation
        model = self.get_or_create_model(aggregation_level=aggregation_level)

        # Aggregate to category level
        category_df = model.aggregate_to_category_level(df)

        return category_df

    @cached("category_seasonality")
    async def analyze_category_seasonality(
        self,
        category_id: Optional[int] = None,
        store_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregation_level: str = "category",
    ) -> Dict[str, Any]:
        """
        Analyze seasonality patterns for categories.

        Args:
            category_id: Filter by category ID
            store_id: Filter by store ID
            city_id: Filter by city ID
            start_date: Start date for analysis
            end_date: End date for analysis
            aggregation_level: Level of aggregation

        Returns:
            Seasonality analysis results
        """
        logger.info("Analyzing category seasonality patterns...")

        try:
            # Get aggregated data
            category_df = await self.aggregate_category_data(
                category_id=category_id,
                store_id=store_id,
                city_id=city_id,
                start_date=start_date,
                end_date=end_date,
                aggregation_level=aggregation_level,
            )

            if category_df.empty:
                return {"error": "No data found for seasonality analysis"}

            # Get model
            model = self.get_or_create_model(aggregation_level=aggregation_level)

            # Analyze seasonality
            seasonality_results = model.analyze_category_seasonality(category_df)

            # Add summary statistics
            analysis_summary = {
                "total_categories_analyzed": len(seasonality_results),
                "date_range": {
                    "start": category_df["sale_date"].min().isoformat(),
                    "end": category_df["sale_date"].max().isoformat(),
                },
                "data_summary": {
                    "total_records": len(category_df),
                    "average_category_sales": float(
                        category_df["category_sales"].mean()
                    ),
                    "total_category_sales": float(category_df["category_sales"].sum()),
                },
            }

            # Category performance ranking
            category_performance = []
            for cat, patterns in seasonality_results.items():
                avg_sales = category_df[category_df["first_category_id"] == int(cat)][
                    "category_sales"
                ].mean()
                category_performance.append(
                    {
                        "category_id": int(cat),
                        "average_sales": float(avg_sales),
                        "seasonal_strength": patterns["seasonal_strength"],
                        "trend_strength": patterns["trend_strength"],
                        "holiday_impact": patterns["holiday_impact"],
                    }
                )

            # Sort by average sales
            category_performance.sort(key=lambda x: x["average_sales"], reverse=True)

            logger.info("Category seasonality analysis completed successfully")

            return {
                "status": "success",
                "seasonality_patterns": seasonality_results,
                "analysis_summary": analysis_summary,
                "category_performance": category_performance[:20],  # Top 20 categories
            }

        except Exception as e:
            logger.error(f"Error in category seasonality analysis: {str(e)}")
            return {"error": f"Seasonality analysis failed: {str(e)}"}

    async def train_category_model(
        self,
        aggregation_level: str = "category",
        model_type: str = "gradient_boost",
        category_id: Optional[int] = None,
        store_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a category-level forecasting model.

        Args:
            aggregation_level: Level of aggregation
            model_type: Type of model to train
            category_id: Category ID filter for training data
            store_id: Store ID filter for training data
            city_id: City ID filter for training data
            start_date: Start date for training data
            end_date: End date for training data

        Returns:
            Training results and metrics
        """
        logger.info(
            f"Training category forecasting model: {aggregation_level}_{model_type}"
        )

        try:
            # Get aggregated training data
            category_df = await self.aggregate_category_data(
                category_id=category_id,
                store_id=store_id,
                city_id=city_id,
                start_date=start_date,
                end_date=end_date,
                aggregation_level=aggregation_level,
            )

            if category_df.empty:
                return {"error": "No training data found"}

            if len(category_df) < 100:
                return {
                    "error": "Insufficient training data (minimum 100 records required)"
                }

            # Additional preprocessing
            category_df = self.preprocessor.handle_outliers(category_df)

            # Create and train model
            model = CategoryLevelForecaster(
                aggregation_level=aggregation_level, model_type=model_type
            )

            training_results = model.fit(category_df, target_col="category_sales")

            # Save model
            model.save_model()

            # Update service model
            model_key = f"{aggregation_level}_{model_type}"
            self.models[model_key] = model

            logger.info("Category forecasting model training completed successfully")

            return {
                "status": "success",
                "aggregation_level": aggregation_level,
                "model_type": model_type,
                "training_data_size": len(category_df),
                "categories_trained": len(training_results),
                "training_metrics": training_results,
            }

        except Exception as e:
            logger.error(f"Error training category model: {str(e)}")
            return {"error": f"Training failed: {str(e)}"}

    async def forecast_category_demand(
        self,
        categories: List[int],
        stores: List[int],
        start_date: str,
        periods: int = 30,
        aggregation_level: str = "category",
        model_type: str = "gradient_boost",
        include_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate category-level demand forecasts.

        Args:
            categories: List of category IDs to forecast
            stores: List of store IDs to forecast
            start_date: Start date for forecast
            periods: Number of periods to forecast
            aggregation_level: Level of aggregation
            model_type: Type of model to use
            include_confidence: Whether to include confidence intervals

        Returns:
            Forecast results
        """
        logger.info(
            f"Generating category forecasts for {len(categories)} categories, {len(stores)} stores"
        )

        try:
            # Get model
            model = self.get_or_create_model(
                aggregation_level=aggregation_level, model_type=model_type
            )

            if not model.is_fitted:
                return {"error": "Model is not trained. Please train the model first."}

            # Generate forecasts
            forecasts_df = model.forecast_category_demand(
                categories=categories,
                stores=stores,
                start_date=start_date,
                periods=periods,
            )

            # Convert to API response format
            forecasts = []
            for _, row in forecasts_df.iterrows():
                forecast_item = {
                    "date": row["sale_date"].isoformat(),
                    "category_id": int(row["first_category_id"]),
                    "store_id": int(row["store_id"]),
                    "predicted_sales": float(row["predicted_category_sales"]),
                    "confidence_score": float(row.get("prediction_confidence", 0.0)),
                }

                forecasts.append(forecast_item)

            # Calculate summary statistics
            forecast_summary = {
                "total_forecasts": len(forecasts),
                "forecast_period": periods,
                "categories_forecasted": len(categories),
                "stores_forecasted": len(stores),
                "average_predicted_sales": float(
                    forecasts_df["predicted_category_sales"].mean()
                ),
                "total_predicted_sales": float(
                    forecasts_df["predicted_category_sales"].sum()
                ),
                "sales_range": {
                    "min": float(forecasts_df["predicted_category_sales"].min()),
                    "max": float(forecasts_df["predicted_category_sales"].max()),
                },
            }

            # Category-level aggregations
            category_aggregates = (
                forecasts_df.groupby("first_category_id")
                .agg({"predicted_category_sales": ["sum", "mean", "count"]})
                .round(2)
            )

            category_insights = []
            for cat in categories:
                if cat in category_aggregates.index:
                    cat_data = category_aggregates.loc[cat]
                    category_insights.append(
                        {
                            "category_id": cat,
                            "total_predicted_sales": float(
                                cat_data[("predicted_category_sales", "sum")]
                            ),
                            "average_daily_sales": float(
                                cat_data[("predicted_category_sales", "mean")]
                            ),
                            "forecast_days": int(
                                cat_data[("predicted_category_sales", "count")]
                            ),
                        }
                    )

            logger.info("Category demand forecasting completed successfully")

            return {
                "status": "success",
                "forecasts": forecasts,
                "forecast_summary": forecast_summary,
                "category_insights": category_insights,
                "model_info": {
                    "aggregation_level": aggregation_level,
                    "model_type": model_type,
                },
            }

        except Exception as e:
            logger.error(f"Error in category demand forecasting: {str(e)}")
            return {"error": f"Forecasting failed: {str(e)}"}

    async def analyze_category_performance(
        self,
        category_id: Optional[int] = None,
        store_id: Optional[int] = None,
        city_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        aggregation_level: str = "category",
    ) -> Dict[str, Any]:
        """
        Analyze category performance metrics and trends.

        Args:
            category_id: Filter by category ID
            store_id: Filter by store ID
            city_id: Filter by city ID
            start_date: Start date for analysis
            end_date: End date for analysis
            aggregation_level: Level of aggregation

        Returns:
            Category performance analysis
        """
        logger.info("Analyzing category performance...")

        try:
            # Get aggregated data
            category_df = await self.aggregate_category_data(
                category_id=category_id,
                store_id=store_id,
                city_id=city_id,
                start_date=start_date,
                end_date=end_date,
                aggregation_level=aggregation_level,
            )

            if category_df.empty:
                return {"error": "No data found for performance analysis"}

            # Calculate performance metrics by category
            performance_metrics = []

            for cat in category_df["first_category_id"].unique():
                if pd.isna(cat):
                    continue

                cat_data = category_df[category_df["first_category_id"] == cat]

                # Basic metrics
                total_sales = cat_data["category_sales"].sum()
                avg_daily_sales = cat_data["category_sales"].mean()
                sales_volatility = (
                    cat_data["category_sales"].std() / avg_daily_sales
                    if avg_daily_sales > 0
                    else 0
                )

                # Growth trend
                cat_data_sorted = cat_data.sort_values("sale_date")
                if len(cat_data_sorted) > 30:  # Need sufficient data for trend
                    first_month = cat_data_sorted.head(30)["category_sales"].mean()
                    last_month = cat_data_sorted.tail(30)["category_sales"].mean()
                    growth_rate = (
                        ((last_month - first_month) / first_month * 100)
                        if first_month > 0
                        else 0
                    )
                else:
                    growth_rate = 0

                # Market share
                market_share = total_sales / category_df["category_sales"].sum() * 100

                # Promotion effectiveness
                if "promo_flag" in cat_data.columns:
                    promo_sales = cat_data[cat_data["promo_flag"] > 0][
                        "category_sales"
                    ].mean()
                    regular_sales = cat_data[cat_data["promo_flag"] == 0][
                        "category_sales"
                    ].mean()
                    promo_uplift = (
                        ((promo_sales - regular_sales) / regular_sales * 100)
                        if regular_sales > 0
                        else 0
                    )
                else:
                    promo_uplift = 0

                performance_metrics.append(
                    {
                        "category_id": int(cat),
                        "total_sales": float(total_sales),
                        "average_daily_sales": float(avg_daily_sales),
                        "sales_volatility": float(sales_volatility),
                        "growth_rate_percent": float(growth_rate),
                        "market_share_percent": float(market_share),
                        "promotion_uplift_percent": float(promo_uplift),
                        "data_points": len(cat_data),
                    }
                )

            # Sort by total sales
            performance_metrics.sort(key=lambda x: x["total_sales"], reverse=True)

            # Overall insights
            total_market_sales = category_df["category_sales"].sum()
            top_5_categories = performance_metrics[:5]
            top_5_share = sum([cat["market_share_percent"] for cat in top_5_categories])

            insights = {
                "market_concentration": {
                    "top_5_categories_share": float(top_5_share),
                    "market_concentration_index": float(
                        sum(
                            [
                                cat["market_share_percent"] ** 2
                                for cat in performance_metrics
                            ]
                        )
                    ),
                    "total_categories": len(performance_metrics),
                },
                "growth_trends": {
                    "growing_categories": len(
                        [
                            cat
                            for cat in performance_metrics
                            if cat["growth_rate_percent"] > 5
                        ]
                    ),
                    "declining_categories": len(
                        [
                            cat
                            for cat in performance_metrics
                            if cat["growth_rate_percent"] < -5
                        ]
                    ),
                    "stable_categories": len(
                        [
                            cat
                            for cat in performance_metrics
                            if -5 <= cat["growth_rate_percent"] <= 5
                        ]
                    ),
                },
                "volatility_analysis": {
                    "high_volatility_categories": len(
                        [
                            cat
                            for cat in performance_metrics
                            if cat["sales_volatility"] > 1.0
                        ]
                    ),
                    "low_volatility_categories": len(
                        [
                            cat
                            for cat in performance_metrics
                            if cat["sales_volatility"] < 0.3
                        ]
                    ),
                    "average_volatility": float(
                        np.mean(
                            [cat["sales_volatility"] for cat in performance_metrics]
                        )
                    ),
                },
            }

            logger.info("Category performance analysis completed successfully")

            return {
                "status": "success",
                "performance_metrics": performance_metrics,
                "market_insights": insights,
                "analysis_period": {
                    "start_date": category_df["sale_date"].min().isoformat(),
                    "end_date": category_df["sale_date"].max().isoformat(),
                    "total_days": (
                        category_df["sale_date"].max() - category_df["sale_date"].min()
                    ).days,
                },
                "total_market_value": float(total_market_sales),
            }

        except Exception as e:
            logger.error(f"Error in category performance analysis: {str(e)}")
            return {"error": f"Performance analysis failed: {str(e)}"}

    @paginate()
    async def get_category_hierarchies(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get category hierarchy information.

        Args:
            limit: Limit for pagination
            offset: Offset for pagination

        Returns:
            Category hierarchy data
        """
        pool = await get_pool()

        query = """
        SELECT 
            first_category_id,
            second_category_id,
            brand_id,
            COUNT(DISTINCT product_id) as product_count,
            AVG(original_price) as avg_price,
            STRING_AGG(DISTINCT product_name, ', ') as sample_products
        FROM product_hierarchy
        WHERE first_category_id IS NOT NULL
        GROUP BY first_category_id, second_category_id, brand_id
        ORDER BY first_category_id, second_category_id, brand_id
        """

        params = []

        if limit is not None:
            query += f" LIMIT ${len(params) + 1}"
            params.append(limit)

        if offset is not None:
            query += f" OFFSET ${len(params) + 1}"
            params.append(offset)

        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            return [dict(row) for row in rows]
