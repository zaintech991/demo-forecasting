"""
Template repository demonstrating connection pooling, caching, and pagination.
"""

from typing import List, Dict, Any, Optional
from .connection import get_pool, cached, paginate


class BaseRepository:
    """Base repository class with common database operations."""

    @cached(key_prefix="base")
    async def get_by_id(self, table: str, id: int) -> Optional[Dict[str, Any]]:
        """Get a record by ID with caching."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = f"SELECT * FROM {table} WHERE id = $1"
            record = await conn.fetchrow(query, id)
            return dict(record) if record else None

    @paginate()
    async def get_all(
        self, table: str, order_by: str = "id", **kwargs
    ) -> List[Dict[str, Any]]:
        """Get all records with pagination."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = f"SELECT * FROM {table} ORDER BY {order_by} LIMIT $1 OFFSET $2"
            records = await conn.fetch(query, kwargs.get("limit"), kwargs.get("offset"))
            return [dict(record) for record in records]

    async def create(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Generate the INSERT query dynamically
            columns = list(data.keys())
            values = list(data.values())
            placeholders = [f"${i+1}" for i in range(len(values))]

            query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING *
            """

            record = await conn.fetchrow(query, *values)
            return dict(record)

    async def update(
        self, table: str, id: int, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a record by ID."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Generate the SET clause dynamically
            set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(data.keys()))
            values = list(data.values())

            query = f"""
                UPDATE {table}
                SET {set_clause}
                WHERE id = $1
                RETURNING *
            """

            record = await conn.fetchrow(query, id, *values)
            return dict(record) if record else None

    async def delete(self, table: str, id: int) -> bool:
        """Delete a record by ID."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = f"DELETE FROM {table} WHERE id = $1 RETURNING id"
            record = await conn.fetchrow(query, id)
            return record is not None


# Example usage:
class SalesRepository(BaseRepository):
    """Repository for sales-related database operations."""

    TABLE_NAME = "sales_data"

    @cached(key_prefix="sales")
    async def get_sales_by_store(self, store_id: int, **kwargs) -> List[Dict[str, Any]]:
        """Get sales data for a specific store with caching."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT *
                FROM sales_data
                WHERE store_id = $1
                ORDER BY date DESC
                LIMIT $2 OFFSET $3
            """
            records = await conn.fetch(
                query, store_id, kwargs.get("limit", 50), kwargs.get("offset", 0)
            )
            return [dict(record) for record in records]

    @cached(key_prefix="sales_stats")
    async def get_store_statistics(self, store_id: int) -> Optional[Dict[str, Any]]:
        """Get aggregated statistics for a store with caching."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT
                    COUNT(*) as total_sales,
                    SUM(quantity) as total_quantity,
                    AVG(quantity) as avg_quantity,
                    MIN(date) as first_sale,
                    MAX(date) as last_sale
                FROM sales_data
                WHERE store_id = $1
            """
            record = await conn.fetchrow(query, store_id)
            return dict(record) if record else None
