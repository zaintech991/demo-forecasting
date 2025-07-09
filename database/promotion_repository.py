import asyncpg  # type: ignore
from typing import List, Optional, Dict, Any
from datetime import datetime

class PromotionRepository:
    """
    Async repository for promotion CRUD and query operations using asyncpg.
    """
    def __init__(self, dsn: str):
        self.dsn = dsn

    async def get_connection(self):
        return await asyncpg.connect(self.dsn)

    async def create_promotion(self, promotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a new promotion into the database.
        Returns the created promotion record.
        """
        query = """
            INSERT INTO promotion_events (
                store_id, product_id, start_date, end_date, promotion_type, discount_percentage, display_location, campaign_id, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9
            ) RETURNING id, store_id, product_id, start_date, end_date, promotion_type, discount_percentage, display_location, campaign_id, created_at
        """
        async with await self.get_connection() as conn:
            row = await conn.fetchrow(
                query,
                promotion_data.get("store_id"),
                promotion_data.get("product_id"),
                promotion_data.get("start_date"),
                promotion_data.get("end_date"),
                promotion_data.get("promotion_type"),
                promotion_data.get("discount_percentage"),
                promotion_data.get("display_location"),
                promotion_data.get("campaign_id"),
                promotion_data.get("created_at", datetime.now()),
            )
            return dict(row) if row else {}

    async def update_promotion(self, promotion_id: int, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing promotion by ID.
        Returns the updated promotion record, or None if not found.
        """
        if not update_data:
            return None
        set_clauses = []
        values = []
        idx = 1
        for key, value in update_data.items():
            set_clauses.append(f"{key} = ${idx}")
            values.append(value)
            idx += 1
        values.append(promotion_id)
        set_clause = ", ".join(set_clauses)
        query = f"""
            UPDATE promotion_events
            SET {set_clause}
            WHERE id = ${idx}
            RETURNING id, store_id, product_id, start_date, end_date, promotion_type, discount_percentage, display_location, campaign_id, created_at
        """
        async with await self.get_connection() as conn:
            row = await conn.fetchrow(query, *values)
            return dict(row) if row else None

    async def delete_promotion(self, promotion_id: int) -> bool:
        """
        Delete a promotion by ID.
        Returns True if deleted, False if not found.
        """
        query = """
            DELETE FROM promotion_events
            WHERE id = $1
            RETURNING id
        """
        async with await self.get_connection() as conn:
            row = await conn.fetchrow(query, promotion_id)
            return row is not None

    async def get_promotion(self, promotion_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a single promotion by ID.
        Returns the promotion record, or None if not found.
        """
        query = """
            SELECT id, store_id, product_id, start_date, end_date, promotion_type, discount_percentage, display_location, campaign_id, created_at
            FROM promotion_events
            WHERE id = $1
        """
        async with await self.get_connection() as conn:
            row = await conn.fetchrow(query, promotion_id)
            return dict(row) if row else None

    async def list_promotions(
        self,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        category_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        List promotions with optional filters.
        Returns a list of promotion records.
        """
        filters = []
        values = []
        idx = 1
        join_category = False
        if store_id is not None:
            filters.append(f"pe.store_id = ${idx}")
            values.append(store_id)
            idx += 1
        if product_id is not None:
            filters.append(f"pe.product_id = ${idx}")
            values.append(product_id)
            idx += 1
        if category_id is not None:
            join_category = True
            filters.append(f"ph.first_category_id = ${idx}")
            values.append(category_id)
            idx += 1
        if start_date is not None:
            filters.append(f"pe.start_date >= ${idx}")
            values.append(start_date)
            idx += 1
        if end_date is not None:
            filters.append(f"pe.end_date <= ${idx}")
            values.append(end_date)
            idx += 1
        if active:
            filters.append(f"pe.start_date <= ${idx} AND pe.end_date >= ${idx+1}")
            now = datetime.now().date()
            values.append(now)
            values.append(now)
            idx += 2
        base_query = """
            SELECT pe.id, pe.store_id, pe.product_id, pe.start_date, pe.end_date, pe.promotion_type, pe.discount_percentage, pe.display_location, pe.campaign_id, pe.created_at
            FROM promotion_events pe
        """
        if join_category:
            base_query += " JOIN product_hierarchy ph ON pe.product_id = ph.product_id "
        if filters:
            base_query += " WHERE " + " AND ".join(filters)
        base_query += " ORDER BY pe.start_date DESC"
        async with await self.get_connection() as conn:
            rows = await conn.fetch(base_query, *values)
            return [dict(row) for row in rows] 