"""
Enhanced Database Connection Manager for FreshRetailNet-50K
Provides unified access with connection pooling, caching, and advanced analytics capabilities
"""

import asyncio
import asyncpg
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import wraps
import json
import time
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from .config import get_db_config
import hashlib
from fastapi import Request # Import Request for type hinting in decorator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the logger level to DEBUG for this module
logger.setLevel(logging.DEBUG)


class DatabaseManager:
    """
    Unified database connection manager for FreshRetailNet-50K analytics
    Supports connection pooling, query caching, and advanced analytics operations
    """

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.query_cache: Dict[str, Any] = {}
        self.cache_ttl: int = 300  # 5 minutes default TTL
        self.max_cache_size: int = 1000
        logger.debug(f"DatabaseManager initialized. query_cache ID: {id(self.query_cache)}")

        # Database configuration
        self.db_config = get_db_config()

        # Pool configuration from environment
        self.db_config["min_size"] = int(os.getenv("DB_POOL_MIN_SIZE", "10"))
        self.db_config["max_size"] = int(os.getenv("DB_POOL_MAX_SIZE", "20"))
        self.db_config["command_timeout"] = int(os.getenv("DB_COMMAND_TIMEOUT", "60"))

    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                min_size=self.db_config["min_size"],
                max_size=self.db_config["max_size"],
                command_timeout=self.db_config["command_timeout"],
                statement_cache_size=0,  # Disable prepared statements for Supabase
            )
            logger.info("Database connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            # This should ideally be initialized by startup event
            raise RuntimeError("Database pool not initialized.")

        async with self.pool.acquire() as connection:
            yield connection

    def cache_key(self, query: str, params: tuple = ()) -> str:
        """Generate cache key for query and parameters"""
        # Convert any list within params to tuple to make it hashable
        hashable_params = tuple(tuple(p) if isinstance(p, list) else p for p in params)
        key = f"{hash(query)}_{hash(hashable_params)}"
        logger.debug(f"Generated cache key for query: {query[:50]}... with params: {params} -> {key}")
        return key

    def is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached result is still valid"""
        return time.time() - timestamp < self.cache_ttl

    async def execute_cached_query(
        self,
        query: str,
        params: tuple = (),
        cache_enabled: bool = True,
        fetch_mode: str = "all",  # 'all', 'one', 'val'
    ) -> Any:
        """
        Execute query with optional caching

        Args:
            query: SQL query string
            params: Query parameters
            cache_enabled: Whether to use caching
            fetch_mode: How to fetch results ('all', 'one', 'val')
        """
        cache_key_str = self.cache_key(query, params)

        # Check cache first
        if cache_enabled and cache_key_str in self.query_cache:
            cached_result, timestamp = self.query_cache[cache_key_str]
            if self.is_cache_valid(timestamp):
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
            else:
                logger.debug(f"Cache expired for query: {query[:50]}... Cache miss.")
        else:
            logger.debug(f"Cache miss for query: {query[:50]}...")

        # Execute query
        async with self.get_connection() as conn:
            try:
                if fetch_mode == "all":
                    result = await conn.fetch(query, *params)
                elif fetch_mode == "one":
                    result = await conn.fetchrow(query, *params)
                elif fetch_mode == "val":
                    result = await conn.fetchval(query, *params)
                else:
                    raise ValueError(f"Invalid fetch_mode: {fetch_mode}")

                # Cache result if caching is enabled
                if cache_enabled:
                    # Manage cache size
                    if len(self.query_cache) >= self.max_cache_size:
                        # Remove oldest entry
                        oldest_key = min(
                            self.query_cache.keys(),
                            key=lambda k: self.query_cache[k][1],
                        )
                        del self.query_cache[oldest_key]

                    self.query_cache[cache_key_str] = (result, time.time())

                return result

            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Params: {params}")
                raise

    async def execute_dataframe_query(
        self, query: str, params: tuple = (), cache_enabled: bool = True
    ) -> pd.DataFrame:
        """Execute query and return results as pandas DataFrame"""
        result = await self.execute_cached_query(query, params, cache_enabled)

        if not result:
            return pd.DataFrame()

        # Convert asyncpg records to DataFrame
        columns = list(result[0].keys())
        data = [list(record.values()) for record in result]

        return pd.DataFrame(data, columns=columns)

    # =============================================================================
    # FRESHRETAILNET-50K SPECIFIC QUERIES
    # =============================================================================

    async def get_sales_data(
        self,
        store_ids: Optional[List[int]] = None,
        product_ids: Optional[List[int]] = None,
        city_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_hourly: bool = False,
        include_stockouts: bool = False,
    ) -> pd.DataFrame:
        """
        Get sales data with flexible filtering

        Args:
            store_ids: List of store IDs to filter
            product_ids: List of product IDs to filter
            city_ids: List of city IDs to filter
            start_date: Start date for data
            end_date: End date for data
            include_hourly: Whether to include hourly breakdown
            include_stockouts: Whether to include stockout information
        """

        # Build dynamic query
        base_query = """
        SELECT 
            sd.sale_date,
            sd.city_id,
            sd.store_id,
            sd.product_id,
            sd.sale_amount,
            sd.units_sold,
            sd.average_unit_price,
            sd.discount,
            sd.holiday_flag,
            sd.activity_flag,
            sd.avg_temperature,
            sd.avg_humidity,
            sd.precpt,
            sd.avg_wind_level,
            sd.peak_hour,
            sd.demand_volatility
        """

        if include_hourly:
            base_query += ",\n            sd.hours_sale"

        if include_stockouts:
            base_query += """,
            sd.stock_hour6_22_cnt,
            sd.hours_stock_status,
            sd.stockout_duration_minutes
            """

        base_query += """
        FROM sales_data sd
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        JOIN product_hierarchy ph ON sd.product_id = ph.product_id
        JOIN city_hierarchy ch ON sd.city_id = ch.city_id
        WHERE 1=1
        """

        conditions: List[str] = []
        params: List[Any] = []
        param_count = 1

        if store_ids:
            conditions.append("sd.store_id = ANY($" + str(param_count) + ")")
            params.append(store_ids)
            param_count += 1

        if product_ids:
            conditions.append("sd.product_id = ANY($" + str(param_count) + ")")
            params.append(product_ids)
            param_count += 1

        if city_ids:
            conditions.append("sd.city_id = ANY($" + str(param_count) + ")")
            params.append(city_ids)
            param_count += 1

        if start_date:
            conditions.append(f"sd.sale_date >= ${{param_count}}")
            params.append(start_date.date())
            param_count += 1

        if end_date:
            conditions.append(f"sd.sale_date <= ${{param_count}}")
            params.append(end_date.date())
            param_count += 1

        if conditions:
            base_query += " AND " + " AND ".join(conditions)

        base_query += " ORDER BY sd.sale_date, sd.store_id, sd.product_id"

        return await self.execute_dataframe_query(base_query, tuple(params))

    async def get_store_performance_metrics(
        self,
        store_ids: Optional[List[int]] = None,
        metric_date: Optional[datetime] = None,
        date_range_days: int = 30,
    ) -> pd.DataFrame:
        """Get store performance metrics for comparison analysis"""

        query = """
        SELECT 
            spm.*,
            sh.store_name,
            sh.city_id,
            ch.city_name,
            sh.store_format,
            sh.location_type
        FROM store_performance_metrics spm
        JOIN store_hierarchy sh ON spm.store_id = sh.store_id
        JOIN city_hierarchy ch ON sh.city_id = ch.city_id
        WHERE 1=1
        """

        conditions: List[str] = []
        params: List[Any] = []
        param_count = 1

        if store_ids:
            conditions.append("spm.store_id = ANY($" + str(param_count) + ")")
            params.append(store_ids)
            param_count += 1

        if metric_date:
            conditions.append(f"spm.metric_date >= ${{param_count}}")
            params.append((metric_date - timedelta(days=date_range_days)).date())
            param_count += 1

            conditions.append(f"spm.metric_date <= ${{param_count}}")
            params.append(metric_date.date())
            param_count += 1

        if conditions:
            query += " AND " + " AND ".join(conditions)

        query += " ORDER BY spm.metric_date DESC, spm.performance_rank"

        return await self.execute_dataframe_query(query, tuple(params))

    async def get_product_correlations(
        self,
        product_id: int,
        store_id: Optional[int] = None,
        correlation_threshold: float = 0.5,
        correlation_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get product correlation data for cross-selling analysis"""

        query = """
        SELECT 
            pc.*,
            ph_a.product_name as product_a_name,
            ph_b.product_name as product_b_name,
            ph_a.first_category_id as product_a_category,
            ph_b.first_category_id as product_b_category
        FROM product_correlations pc
        JOIN product_hierarchy ph_a ON pc.product_a_id = ph_a.product_id
        JOIN product_hierarchy ph_b ON pc.product_b_id = ph_b.product_id
        WHERE (pc.product_a_id = $1 OR pc.product_b_id = $1)
        AND ABS(pc.correlation_coefficient) >= $2
        """

        params: List[Any] = [product_id, correlation_threshold]
        param_count = 3

        if store_id:
            query += " AND (pc.store_id = $" + str(param_count) + " OR pc.store_id IS NULL)"
            params.append(store_id)
            param_count += 1

        if correlation_types:
            query += " AND pc.correlation_type = ANY($" + str(param_count) + ")"
            params.append(correlation_types)
            param_count += 1

        query += " ORDER BY ABS(pc.correlation_coefficient) DESC"

        return await self.execute_dataframe_query(query, tuple(params))

    async def get_weather_impact_data(
        self,
        product_ids: Optional[List[int]] = None,
        city_ids: Optional[List[int]] = None,
        include_sales_data: bool = True,
    ) -> pd.DataFrame:
        """Get weather impact analysis data"""

        base_query = """
        SELECT 
            wia.*,
            ph.product_name,
            ph.is_perishable,
            ch.city_name,
            ch.climate_zone
        """

        if include_sales_data:
            base_query += """,
            AVG(sd.sale_amount) as avg_sales,
            COUNT(sd.sale_date) as data_points,
            CORR(sd.sale_amount, sd.avg_temperature) as actual_temp_correlation,
            CORR(sd.sale_amount, sd.avg_humidity) as actual_humidity_correlation,
            CORR(sd.sale_amount, sd.precpt) as actual_precip_correlation
            """

        base_query += """
        FROM weather_impact_analysis wia
        JOIN product_hierarchy ph ON wia.product_id = ph.product_id
        JOIN city_hierarchy ch ON wia.city_id = ch.city_id
        """

        if include_sales_data:
            base_query += """
            LEFT JOIN sales_data sd ON wia.product_id = sd.product_id 
                                   AND wia.city_id = sd.city_id
                                   AND sd.sale_date >= CURRENT_DATE - INTERVAL '90 days'
            """

        base_query += " WHERE 1=1"

        conditions: List[str] = []
        params: List[Any] = []
        param_count = 1

        if product_ids:
            conditions.append("wia.product_id = ANY($" + str(param_count) + ")")
            params.append(product_ids)
            param_count += 1

        if city_ids:
            conditions.append("wia.city_id = ANY($" + str(param_count) + ")")
            params.append(city_ids)
            param_count += 1

        if conditions:
            base_query += " AND " + " AND ".join(conditions)

        if include_sales_data:
            base_query += """
            GROUP BY wia.id, wia.product_id, wia.city_id, wia.temperature_elasticity,
                     wia.humidity_impact, wia.precipitation_impact, wia.wind_impact,
                     wia.spring_multiplier, wia.summer_multiplier, wia.autumn_multiplier,
                     wia.winter_multiplier, wia.optimal_temperature_min, wia.optimal_temperature_max,
                     wia.optimal_humidity_range, wia.confidence_score, wia.last_updated,
                     ph.product_name, ph.is_perishable, ch.city_name, ch.climate_zone
            """

        base_query += " ORDER BY wia.confidence_score DESC"

        return await self.execute_dataframe_query(base_query, tuple(params))

    async def get_stockout_events(
        self,
        store_ids: Optional[List[int]] = None,
        product_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_duration_hours: Optional[float] = None,
    ) -> pd.DataFrame:
        """Get stockout events for risk assessment"""

        query = """
        SELECT 
            se.*,
            sh.store_name,
            sh.city_id,
            ch.city_name,
            ph.product_name,
            ph.is_perishable,
            ph.shelf_life_days
        FROM stockout_events se
        JOIN store_hierarchy sh ON se.store_id = sh.store_id
        JOIN city_hierarchy ch ON sh.city_id = ch.city_id
        JOIN product_hierarchy ph ON se.product_id = ph.product_id
        WHERE 1=1
        """

        conditions: List[str] = []
        params: List[Any] = []
        param_count = 1

        if store_ids:
            conditions.append("se.store_id = ANY($" + str(param_count) + ")")
            params.append(store_ids)
            param_count += 1

        if product_ids:
            conditions.append("se.product_id = ANY($" + str(param_count) + ")")
            params.append(product_ids)
            param_count += 1

        if start_date:
            conditions.append(f"se.stockout_start >= ${{param_count}}")
            params.append(start_date)
            param_count += 1

        if end_date:
            conditions.append(f"se.stockout_start <= ${{param_count}}")
            params.append(end_date)
            param_count += 1

        if min_duration_hours:
            conditions.append(f"se.duration_hours >= ${{param_count}}")
            params.append(min_duration_hours)
            param_count += 1

        if conditions:
            query += " AND " + " AND ".join(conditions)

        query += " ORDER BY se.stockout_start DESC"

        return await self.execute_dataframe_query(query, tuple(params))

    async def get_promotion_effectiveness(
        self,
        store_ids: Optional[List[int]] = None,
        product_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_roi: Optional[float] = None,
    ) -> pd.DataFrame:
        """Get promotion effectiveness data"""

        query = """
        SELECT 
            pe.*,
            sh.store_name,
            sh.city_id,
            ch.city_name,
            ph.product_name,
            ph.promotion_elasticity
        FROM promotion_effectiveness pe
        JOIN store_hierarchy sh ON pe.store_id = sh.store_id
        JOIN city_hierarchy ch ON sh.city_id = ch.city_id
        JOIN product_hierarchy ph ON pe.product_id = ph.product_id
        WHERE 1=1
        """

        conditions: List[str] = []
        params: List[Any] = []
        param_count = 1

        if store_ids:
            conditions.append("pe.store_id = ANY($" + str(param_count) + ")")
            params.append(store_ids)
            param_count += 1

        if product_ids:
            conditions.append("pe.product_id = ANY($" + str(param_count) + ")")
            params.append(product_ids)
            param_count += 1

        if start_date:
            conditions.append(f"pe.promotion_start_date >= ${{param_count}}")
            params.append(start_date.date())
            param_count += 1

        if end_date:
            conditions.append(f"pe.promotion_end_date <= ${{param_count}}")
            params.append(end_date.date())
            param_count += 1

        if min_roi:
            conditions.append(f"pe.roi >= ${{param_count}}")
            params.append(min_roi)
            param_count += 1

        if conditions:
            query += " AND " + " AND ".join(conditions)

        query += " ORDER BY pe.roi DESC"

        return await self.execute_dataframe_query(query, tuple(params))

    # =============================================================================
    # ADVANCED ANALYTICS METHODS
    # =============================================================================

    async def calculate_product_correlations(
        self,
        store_id: Optional[int] = None,
        analysis_period_days: int = 90,
        min_correlation: float = 0.3,
    ) -> pd.DataFrame:
        """Calculate and store product correlations"""

        # First, get sales data for correlation analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=analysis_period_days)

        sales_data = await self.get_sales_data(
            store_ids=[store_id] if store_id else None,
            start_date=start_date,
            end_date=end_date,
        )

        if sales_data.empty:
            return pd.DataFrame()

        # Pivot data for correlation calculation
        pivot_data = sales_data.pivot_table(
            index="sale_date", columns="product_id", values="sale_amount", fill_value=0
        )

        # Calculate correlations
        correlation_matrix = pivot_data.corr()

        # Store results in database
        correlations_to_store = []

        for product_a in correlation_matrix.index:
            for product_b in correlation_matrix.columns:
                if product_a >= product_b:  # Avoid duplicates
                    continue

                correlation_coef = correlation_matrix.loc[product_a, product_b]

                if abs(correlation_coef) >= min_correlation:
                    # Determine correlation type
                    if correlation_coef > 0.7:
                        correlation_type = "complementary"
                    elif correlation_coef < -0.5:
                        correlation_type = "substitute"
                    else:
                        correlation_type = "neutral"

                    correlations_to_store.append(
                        {
                            "product_a_id": int(product_a),
                            "product_b_id": int(product_b),
                            "store_id": store_id,
                            "correlation_coefficient": float(correlation_coef),
                            "correlation_type": correlation_type,
                            "confidence_level": 0.95,  # Statistical confidence
                            "analysis_period_start": start_date.date(),
                            "analysis_period_end": end_date.date(),
                        }
                    )

        # Insert correlations into database
        if correlations_to_store:
            await self._batch_insert_correlations(correlations_to_store)

        return pd.DataFrame(correlations_to_store)

    async def _batch_insert_correlations(self, correlations: List[Dict]):
        """Batch insert product correlations"""

        query = """
        INSERT INTO product_correlations 
        (product_a_id, product_b_id, store_id, correlation_coefficient, 
         correlation_type, confidence_level, analysis_period_start, analysis_period_end)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (product_a_id, product_b_id, store_id) 
        DO UPDATE SET 
            correlation_coefficient = EXCLUDED.correlation_coefficient,
            correlation_type = EXCLUDED.correlation_type,
            confidence_level = EXCLUDED.confidence_level,
            analysis_period_start = EXCLUDED.analysis_period_start,
            analysis_period_end = EXCLUDED.analysis_period_end,
            updated_at = NOW()
        """

        async with self.get_connection() as conn:
            await conn.executemany(
                query,
                [
                    (
                        corr["product_a_id"],
                        corr["product_b_id"],
                        corr["store_id"],
                        corr["correlation_coefficient"],
                        corr["correlation_type"],
                        corr["confidence_level"],
                        corr["analysis_period_start"],
                        corr["analysis_period_end"],
                    )
                    for corr in correlations
                ],
            )

    async def update_store_performance_metrics(
        self, metric_date: Optional[datetime] = None
    ) -> None:
        """Update store performance metrics for all stores"""

        if not metric_date:
            metric_date = datetime.now() - timedelta(days=1)  # Yesterday

        query = """
        INSERT INTO store_performance_metrics 
        (store_id, metric_date, total_revenue, total_units_sold, 
         average_transaction_value, stockout_frequency, performance_rank)
        
        WITH daily_store_metrics AS (
            SELECT 
                sd.store_id,
                SUM(sd.sale_amount) as total_revenue,
                SUM(sd.units_sold) as total_units_sold,
                AVG(sd.average_unit_price) as avg_transaction_value,
                AVG(CASE WHEN sd.stock_hour6_22_cnt > 0 THEN 1.0 ELSE 0.0 END) as stockout_frequency
            FROM sales_data sd
            WHERE sd.sale_date = $1
            GROUP BY sd.store_id
        ),
        ranked_metrics AS (
            SELECT 
                *,
                RANK() OVER (ORDER BY total_revenue DESC) as performance_rank
            FROM daily_store_metrics
        )
        SELECT 
            store_id,
            $1 as metric_date,
            total_revenue,
            total_units_sold,
            avg_transaction_value,
            stockout_frequency,
            performance_rank
        FROM ranked_metrics
        
        ON CONFLICT (store_id, metric_date)
        DO UPDATE SET 
            total_revenue = EXCLUDED.total_revenue,
            total_units_sold = EXCLUDED.total_units_sold,
            average_transaction_value = EXCLUDED.average_transaction_value,
            stockout_frequency = EXCLUDED.stockout_frequency,
            performance_rank = EXCLUDED.performance_rank,
            updated_at = NOW()
        """

        async with self.get_connection() as conn:
            await conn.execute(query, metric_date.date())

        logger.info(f"Updated store performance metrics for {metric_date.date()}")

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")

    async def execute_insert(self, table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes an INSERT query and returns the inserted row."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f"${i+1}" for i in range(len(data)))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) RETURNING *"
        values = list(data.values())

        async with self.get_connection() as conn:
            result = await conn.fetchrow(query, *values)
            return dict(result) if result else {}

    async def execute_update(self, table_name: str, data: Dict[str, Any], conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Executes an UPDATE query and returns the updated rows."""
        set_clauses = ", ".join(f"{col} = ${i+1}" for i, col in enumerate(data.keys()))
        where_clauses = " AND ".join(f"{col} = ${i+len(data)+1}" for i, col in enumerate(conditions.keys()))
        query = f"UPDATE {table_name} SET {set_clauses} WHERE {where_clauses} RETURNING *"
        values = list(data.values()) + list(conditions.values())

        async with self.get_connection() as conn:
            results = await conn.fetch(query, *values)
            return [dict(row) for row in results]

    async def execute_delete(self, table_name: str, conditions: Dict[str, Any]) -> int:
        """Executes a DELETE query and returns the number of deleted rows."""
        where_clauses = " AND ".join(f"{col} = ${i+1}" for i, col in enumerate(conditions.keys()))
        query = f"DELETE FROM {table_name} WHERE {where_clauses}"
        values = list(conditions.values())

        async with self.get_connection() as conn:
            status = await conn.execute(query, *values)
            # The status string is typically 'DELETE N' where N is the number of rows.
            return int(status.split(' ')[1]) if ' ' in status else 0

# Remove global db_manager instance, it will be managed by app.state
# _db_manager_instance: Optional[DatabaseManager] = None

# def get_db_manager() -> DatabaseManager:
#     global _db_manager_instance
#     if _db_manager_instance is None:
#         _db_manager_instance = DatabaseManager()
#     return _db_manager_instance

# Remove get_pool and related legacy functions, manage pool directly via app.state.db_manager
# async def get_pool():
#     manager = get_db_manager()
#     if not manager.pool:
#         await manager.initialize()
#     return manager.pool

# def paginate(page: int = 1, page_size: int = 100):
#     """Helper function for pagination"""
#     offset = (page - 1) * page_size
#     return f"LIMIT {page_size} OFFSET {offset}"

# async def get_db_engine():
#     return await get_db_manager()

# async def get_supabase_client():
#     return await get_db_engine()

# async def get_db_connection():
#     async with get_db_manager().get_connection() as conn:
#         yield conn

# async def initialize_database():
#     await get_db_manager().initialize()

# async def close_database():
#     await get_db_manager().close()


def cached(ttl: int = 300):
    """Decorator for caching function results"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object to get db_manager from app.state
            request: Optional[Request] = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if 'request' in kwargs and isinstance(kwargs['request'], Request):
                request = kwargs['request']
            
            # If request is None (e.g., during startup, internal calls), bypass caching and proceed
            if request is None:
                logger.debug(f"Request object is None in @cached for {func.__name__}. Bypassing cache and directly executing function.")
                return await func(*args, **kwargs)

            # Now that we know request is not None, we can safely access its attributes
            # If it's not an HTTP request (e.g., ASGI lifespan, background task), bypass caching
            if request.scope is None or not hasattr(request.scope, 'type') or request.scope['type'] != 'http':
                logger.debug(f"Bypassing cache for non-HTTP request (scope type is not 'http') to {func.__name__}")
                return await func(*args, **kwargs)

            # CRITICAL: Also bypass if db_manager is not yet available in app.state (e.g., during startup before app.on_event runs)
            if not hasattr(request.app.state, 'db_manager') or request.app.state.db_manager is None:
                logger.warning(f"db_manager not found or is None in app.state for {func.__name__}. Bypassing cache for this request.")
                return await func(*args, **kwargs)

            manager = request.app.state.db_manager

            # Convert args and kwargs to a JSON string for consistent hashing
            def default_json_encoder(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if hasattr(obj, 'model_dump'): # Pydantic v2
                    return obj.model_dump()
                if hasattr(obj, 'dict'): # Pydantic v1
                    return obj.dict()
                # Handle Request object by ignoring it in cache key generation
                if isinstance(obj, Request):
                    return "_REQUEST_OBJECT_" # Placeholder, will be consistent
                # Ensure consistent order for lists and dictionary keys for hashing
                if isinstance(obj, list):
                    return sorted(obj, key=lambda x: str(x)) # Sort list elements
                if isinstance(obj, dict):
                    return {k: obj[k] for k in sorted(obj)} # Sort dictionary keys
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            try:
                # Filter out the Request object from args/kwargs for caching purposes
                filtered_args = [arg for arg in args if not isinstance(arg, Request)]
                filtered_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, Request)}

                # Recursively sort lists and dictionary keys within filtered_args and filtered_kwargs
                def deep_sort(item):
                    if isinstance(item, dict):
                        return {k: deep_sort(v) for k, v in sorted(item.items())}
                    if isinstance(item, list):
                        # Attempt to sort list items, handling non-comparable types
                        try:
                            return sorted(item, key=lambda x: str(deep_sort(x)))
                        except TypeError:
                            return item # Cannot sort, return as is
                    return item
                
                cache_input = {
                    'args': deep_sort(filtered_args),
                    'kwargs': deep_sort(filtered_kwargs)
                }

                cache_key_str = json.dumps(cache_input, sort_keys=True, default=default_json_encoder)
                logger.debug(f"Generated raw cache key string: {cache_key_str}")
                # Use SHA256 for consistent hashing across runs
                cache_key = f"{func.__name__}_{hashlib.sha256(cache_key_str.encode()).hexdigest()}"
            except TypeError as e:
                logger.warning(f"Could not serialize function arguments for caching: {e}. Falling back to basic hashing.")
                # Fallback remains, though should be rarely hit now
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"

            logger.debug(f"Generated function cache key for {func.__name__} with args: {args}, kwargs: {kwargs} -> {cache_key}")
            logger.debug(f"Current query_cache size: {len(manager.query_cache)}. query_cache ID: {id(manager.query_cache)}")

            if (
                hasattr(manager, "query_cache")
                and cache_key in manager.query_cache
            ):
                cached_result, timestamp = manager.query_cache[cache_key]
                if (time.time() - timestamp) < ttl:
                    logger.info(f"Cache hit for function {func.__name__} with key: {cache_key}")
                    return cached_result
                else:
                    logger.info(f"Cache expired for function {func.__name__}. Cache miss with key: {cache_key}")
                    logger.debug(f"Cache content for expired key: {manager.query_cache.get(cache_key)}")
            else:
                logger.info(f"Cache miss for function {func.__name__} with key: {cache_key}")

            result = await func(*args, **kwargs)
            if hasattr(manager, "query_cache"):
                # Manage cache size before adding new item
                if len(manager.query_cache) >= manager.max_cache_size:
                    oldest_key = min(
                        manager.query_cache.keys(),
                        key=lambda k: manager.query_cache[k][1]
                    )
                    logger.debug(f"Cache max size reached. Removing oldest entry: {oldest_key}")
                    del manager.query_cache[oldest_key]
                manager.query_cache[cache_key] = (result, time.time())
                logger.debug(f"Added to cache. New cache size: {len(manager.query_cache)}")
            return result

        return wrapper

    return decorator


# Export main components
__all__ = [
    "DatabaseManager",
    "cached",
]
