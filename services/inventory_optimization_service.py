"""
Cross-Store Inventory Optimization Service
Analyzes inventory levels across stores, identifies transfer opportunities,
and provides optimization recommendations for better stock distribution.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal
import asyncio
import json
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InventoryOpportunity:
    source_store_id: int
    target_store_id: int
    product_id: int
    city_id: int
    recommended_quantity: int
    optimization_score: Decimal
    potential_revenue_impact: Decimal
    transfer_cost: Decimal
    urgency_level: int
    implementation_priority: str
    reasoning: str
    expected_benefit: Dict[str, Any]


@dataclass
class StoreInventoryProfile:
    store_id: int
    city_id: int
    total_products: int
    avg_stock_level: float
    stock_turnover_rate: float
    stockout_risk_products: int
    overstock_products: int
    inventory_value: Decimal
    efficiency_score: Decimal


class InventoryOptimizationService:
    """Advanced cross-store inventory optimization and transfer recommendation system."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.optimization_params = {
            "min_transfer_quantity": 5,
            "max_transfer_distance_km": 50,
            "transfer_cost_per_km": 0.5,
            "min_optimization_score": 60,
            "stockout_threshold_days": 3,
            "overstock_threshold_days": 30,
            "safety_stock_multiplier": 1.5,
        }

    async def analyze_cross_store_opportunities(
        self, city_id: Optional[int] = None
    ) -> List[InventoryOpportunity]:
        """Identify cross-store inventory optimization opportunities."""
        try:
            # Get comprehensive inventory analysis
            inventory_data = await self._get_inventory_analysis(city_id)
            store_profiles = await self._generate_store_profiles(city_id)
            transfer_costs = await self._calculate_transfer_costs(city_id)

            opportunities = []

            # Analyze each product across stores
            products = set(item["product_id"] for item in inventory_data)

            for product_id in products:
                product_opportunities = await self._analyze_product_optimization(
                    product_id, inventory_data, store_profiles, transfer_costs
                )
                opportunities.extend(product_opportunities)

            # Prioritize opportunities
            opportunities = self._prioritize_opportunities(opportunities)

            # Store analysis results
            await self._store_optimization_analysis(opportunities)

            return opportunities[:50]  # Return top 50 opportunities

        except Exception as e:
            self.logger.error(f"Error analyzing cross-store opportunities: {e}")
            raise

    async def _get_inventory_analysis(
        self, city_id: Optional[int] = None
    ) -> List[Dict]:
        """Get comprehensive inventory analysis across stores."""
        try:
            async with get_db_connection() as conn:
                query = """
                WITH inventory_metrics AS (
                    SELECT 
                        s.store_id,
                        s.product_id,
                        sh.city_id,
                        sh.latitude,
                        sh.longitude,
                        ph.product_name,
                        ph.first_category_id,
                        AVG(s.units_sold) as avg_daily_sales,
                        STDDEV(s.units_sold) as sales_stddev,
                        MAX(s.stock_level) as current_stock,
                        MIN(s.stock_level) as min_stock_level,
                        COUNT(*) as days_data,
                        AVG(s.stock_level) as avg_stock_level,
                        SUM(s.units_sold * 25) as total_revenue  -- Estimated price
                    FROM sales_data s
                    JOIN store_hierarchy sh ON s.store_id = sh.store_id
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '30 days'
                        AND s.stock_level IS NOT NULL
                """

                params = []
                if city_id is not None:
                    query += " AND sh.city_id = $1"
                    params.append(city_id)

                query += """
                    GROUP BY s.store_id, s.product_id, sh.city_id, sh.latitude, 
                             sh.longitude, ph.product_name, ph.first_category_id
                    HAVING COUNT(*) >= 14
                ),
                demand_forecast AS (
                    SELECT 
                        store_id,
                        product_id,
                        avg_daily_sales,
                        CASE 
                            WHEN sales_stddev > 0 AND avg_daily_sales > 0
                            THEN avg_daily_sales * %s  -- Safety stock multiplier
                            ELSE avg_daily_sales * 2
                        END as recommended_safety_stock
                    FROM inventory_metrics
                )
                SELECT 
                    im.*,
                    df.recommended_safety_stock,
                    CASE 
                        WHEN im.avg_daily_sales > 0 
                        THEN im.current_stock / im.avg_daily_sales
                        ELSE 999
                    END as days_of_inventory,
                    CASE 
                        WHEN im.avg_daily_sales > 0 AND im.current_stock <= df.recommended_safety_stock
                        THEN 'stockout_risk'
                        WHEN im.avg_daily_sales > 0 AND im.current_stock > im.avg_daily_sales * %s
                        THEN 'overstock'
                        ELSE 'normal'
                    END as stock_status
                FROM inventory_metrics im
                JOIN demand_forecast df ON im.store_id = df.store_id AND im.product_id = df.product_id
                ORDER BY im.city_id, im.product_id, im.store_id
                """ % (
                    self.optimization_params["safety_stock_multiplier"],
                    self.optimization_params["overstock_threshold_days"],
                )

                result = await conn.fetch(query, *params)
                return [dict(row) for row in result]

        except Exception as e:
            self.logger.error(f"Error getting inventory analysis: {e}")
            return []

    async def _generate_store_profiles(
        self, city_id: Optional[int] = None
    ) -> Dict[int, StoreInventoryProfile]:
        """Generate comprehensive profiles for each store."""
        try:
            async with get_db_connection() as conn:
                query = """
                WITH store_metrics AS (
                    SELECT 
                        s.store_id,
                        sh.city_id,
                        COUNT(DISTINCT s.product_id) as total_products,
                        AVG(s.stock_level) as avg_stock_level,
                        SUM(s.units_sold) / NULLIF(SUM(s.stock_level), 0) as stock_turnover_rate,
                        COUNT(CASE WHEN s.stock_level <= 5 THEN 1 END) as low_stock_products,
                        COUNT(CASE WHEN s.stock_level > s.units_sold * 30 THEN 1 END) as overstock_products,
                        SUM(s.stock_level * 25) as inventory_value  -- Estimated value
                    FROM sales_data s
                    JOIN store_hierarchy sh ON s.store_id = sh.store_id
                    WHERE s.date >= NOW() - INTERVAL '30 days'
                        AND s.stock_level IS NOT NULL
                """

                params = []
                if city_id is not None:
                    query += " AND sh.city_id = $1"
                    params.append(city_id)

                query += """
                    GROUP BY s.store_id, sh.city_id
                    HAVING COUNT(*) >= 30
                )
                SELECT 
                    *,
                    CASE 
                        WHEN stock_turnover_rate > 0.1 AND low_stock_products < total_products * 0.1
                        THEN 90 + (stock_turnover_rate * 100)
                        WHEN stock_turnover_rate > 0.05
                        THEN 70 + (stock_turnover_rate * 200)
                        ELSE 50
                    END as efficiency_score
                FROM store_metrics
                ORDER BY store_id
                """

                result = await conn.fetch(query, *params)

                profiles = {}
                for row in result:
                    profiles[row["store_id"]] = StoreInventoryProfile(
                        store_id=row["store_id"],
                        city_id=row["city_id"],
                        total_products=row["total_products"],
                        avg_stock_level=row["avg_stock_level"] or 0,
                        stock_turnover_rate=row["stock_turnover_rate"] or 0,
                        stockout_risk_products=row["low_stock_products"],
                        overstock_products=row["overstock_products"],
                        inventory_value=Decimal(str(row["inventory_value"] or 0)),
                        efficiency_score=Decimal(str(row["efficiency_score"])),
                    )

                return profiles

        except Exception as e:
            self.logger.error(f"Error generating store profiles: {e}")
            return {}

    async def _calculate_transfer_costs(
        self, city_id: Optional[int] = None
    ) -> Dict[Tuple[int, int], Decimal]:
        """Calculate transfer costs between stores based on distance."""
        try:
            async with get_db_connection() as conn:
                query = """
                SELECT 
                    store_id,
                    latitude,
                    longitude
                FROM store_hierarchy
                WHERE latitude IS NOT NULL 
                    AND longitude IS NOT NULL
                """

                params = []
                if city_id is not None:
                    query += " AND city_id = $1"
                    params.append(city_id)

                result = await conn.fetch(query, *params)

                stores = {
                    row["store_id"]: (row["latitude"], row["longitude"])
                    for row in result
                }
                transfer_costs = {}

                # Calculate distances between all store pairs
                for store1_id, (lat1, lon1) in stores.items():
                    for store2_id, (lat2, lon2) in stores.items():
                        if store1_id != store2_id:
                            # Haversine distance formula
                            distance_km = self._calculate_distance(
                                lat1, lon1, lat2, lon2
                            )
                            cost = Decimal(
                                str(
                                    distance_km
                                    * self.optimization_params["transfer_cost_per_km"]
                                )
                            )
                            transfer_costs[(store1_id, store2_id)] = cost

                return transfer_costs

        except Exception as e:
            self.logger.error(f"Error calculating transfer costs: {e}")
            return {}

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points using Haversine formula."""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371  # Earth's radius in kilometers

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    async def _analyze_product_optimization(
        self,
        product_id: int,
        inventory_data: List[Dict],
        store_profiles: Dict[int, StoreInventoryProfile],
        transfer_costs: Dict[Tuple[int, int], Decimal],
    ) -> List[InventoryOpportunity]:
        """Analyze optimization opportunities for a specific product."""
        opportunities: List[InventoryOpportunity] = []

        # Filter data for this product
        product_data = [
            item for item in inventory_data if item["product_id"] == product_id
        ]

        if len(product_data) < 2:
            return opportunities

        # Identify stores with excess stock and stores with shortage
        excess_stores = []
        shortage_stores = []

        for item in product_data:
            if (
                item["stock_status"] == "overstock"
                and item["days_of_inventory"]
                > self.optimization_params["overstock_threshold_days"]
            ):
                excess_stores.append(item)
            elif (
                item["stock_status"] == "stockout_risk"
                and item["days_of_inventory"]
                < self.optimization_params["stockout_threshold_days"]
            ):
                shortage_stores.append(item)

        # Calculate transfer opportunities
        for excess_item in excess_stores:
            for shortage_item in shortage_stores:
                if excess_item["store_id"] == shortage_item["store_id"]:
                    continue

                # Check if stores are in reasonable distance
                transfer_key = (excess_item["store_id"], shortage_item["store_id"])
                if transfer_key not in transfer_costs:
                    continue

                transfer_cost = transfer_costs[transfer_key]

                # Skip if too expensive to transfer
                if transfer_cost > 100:  # Max $100 transfer cost
                    continue

                # Calculate optimal transfer quantity
                excess_quantity = max(
                    0,
                    excess_item["current_stock"]
                    - excess_item["recommended_safety_stock"],
                )
                shortage_quantity = max(
                    0,
                    shortage_item["recommended_safety_stock"]
                    - shortage_item["current_stock"],
                )

                recommended_quantity = min(excess_quantity, shortage_quantity)

                if (
                    recommended_quantity
                    < self.optimization_params["min_transfer_quantity"]
                ):
                    continue

                # Calculate optimization score
                optimization_score = self._calculate_optimization_score(
                    excess_item, shortage_item, recommended_quantity, transfer_cost
                )

                if (
                    optimization_score
                    < self.optimization_params["min_optimization_score"]
                ):
                    continue

                # Calculate potential revenue impact
                potential_sales_increase = min(
                    recommended_quantity, shortage_item["avg_daily_sales"] * 7
                )
                revenue_impact = Decimal(
                    str(potential_sales_increase * 25)
                )  # Estimated price

                # Determine urgency and priority
                urgency_level = self._calculate_urgency_level(
                    shortage_item, excess_item
                )
                priority = self._determine_priority(
                    optimization_score, urgency_level, revenue_impact
                )

                opportunity = InventoryOpportunity(
                    source_store_id=excess_item["store_id"],
                    target_store_id=shortage_item["store_id"],
                    product_id=product_id,
                    city_id=excess_item["city_id"],
                    recommended_quantity=int(recommended_quantity),
                    optimization_score=Decimal(str(optimization_score)),
                    potential_revenue_impact=revenue_impact,
                    transfer_cost=transfer_cost,
                    urgency_level=urgency_level,
                    implementation_priority=priority,
                    reasoning=f"Transfer {recommended_quantity} units from overstocked Store {excess_item['store_id']} ({excess_item['days_of_inventory']:.1f} days inventory) to shortage Store {shortage_item['store_id']} ({shortage_item['days_of_inventory']:.1f} days inventory)",
                    expected_benefit={
                        "source_days_reduction": excess_item["days_of_inventory"]
                        - (excess_item["current_stock"] - recommended_quantity)
                        / max(excess_item["avg_daily_sales"], 1),
                        "target_days_increase": (
                            shortage_item["current_stock"] + recommended_quantity
                        )
                        / max(shortage_item["avg_daily_sales"], 1)
                        - shortage_item["days_of_inventory"],
                        "potential_sales_increase": potential_sales_increase,
                        "roi_estimate": float(
                            revenue_impact / max(transfer_cost, Decimal("1"))
                        ),
                    },
                )

                opportunities.append(opportunity)

        return opportunities

    def _calculate_optimization_score(
        self,
        excess_item: Dict,
        shortage_item: Dict,
        quantity: float,
        transfer_cost: Decimal,
    ) -> float:
        """Calculate optimization score for a transfer opportunity."""
        # Base score from urgency
        urgency_score = max(
            0,
            (
                self.optimization_params["stockout_threshold_days"]
                - shortage_item["days_of_inventory"]
            )
            * 10,
        )
        excess_score = max(
            0,
            (
                excess_item["days_of_inventory"]
                - self.optimization_params["overstock_threshold_days"]
            )
            * 2,
        )

        # Revenue potential score
        revenue_potential = quantity * 25  # Estimated revenue
        cost_effectiveness = revenue_potential / max(float(transfer_cost), 1)

        # Distance penalty
        distance_penalty = min(20, float(transfer_cost) * 0.2)

        # Final optimization score
        optimization_score = (
            urgency_score + excess_score + cost_effectiveness * 10 - distance_penalty
        )

        return max(0, min(100, optimization_score))

    def _calculate_urgency_level(self, shortage_item: Dict, excess_item: Dict) -> int:
        """Calculate urgency level (1-5) for the transfer."""
        if shortage_item["days_of_inventory"] <= 1:
            return 5  # Critical
        elif shortage_item["days_of_inventory"] <= 2:
            return 4  # High
        elif shortage_item["days_of_inventory"] <= 3:
            return 3  # Medium
        elif excess_item["days_of_inventory"] > 45:
            return 3  # Medium due to high excess
        else:
            return 2  # Low

    def _determine_priority(
        self, optimization_score: float, urgency_level: int, revenue_impact: Decimal
    ) -> str:
        """Determine implementation priority."""
        if urgency_level >= 4 and optimization_score >= 80:
            return "immediate"
        elif urgency_level >= 3 and optimization_score >= 70:
            return "high"
        elif optimization_score >= 60 or revenue_impact >= 500:
            return "medium"
        else:
            return "low"

    def _prioritize_opportunities(
        self, opportunities: List[InventoryOpportunity]
    ) -> List[InventoryOpportunity]:
        """Prioritize opportunities based on multiple factors."""
        # Sort by implementation priority, then optimization score, then revenue impact
        priority_order = {"immediate": 4, "high": 3, "medium": 2, "low": 1}

        opportunities.sort(
            key=lambda x: (
                priority_order[x.implementation_priority],
                x.optimization_score,
                x.potential_revenue_impact,
            ),
            reverse=True,
        )

        return opportunities

    async def _store_optimization_analysis(
        self, opportunities: List[InventoryOpportunity]
    ) -> None:
        """Store optimization analysis results in database."""
        try:
            async with get_db_connection() as conn:
                # Clear old analysis for today
                await conn.execute(
                    """
                    DELETE FROM cross_store_inventory
                    WHERE analysis_date::date = CURRENT_DATE
                """
                )

                # Insert new analysis
                for opp in opportunities:
                    await conn.execute(
                        """
                        INSERT INTO cross_store_inventory (
                            source_store_id, target_store_id, product_id, city_id,
                            recommended_quantity, transfer_cost, optimization_score,
                            potential_revenue_impact, urgency_level, implementation_priority,
                            transfer_reasoning, expected_benefits, analysis_date
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
                    """,
                        opp.source_store_id,
                        opp.target_store_id,
                        opp.product_id,
                        opp.city_id,
                        opp.recommended_quantity,
                        opp.transfer_cost,
                        opp.optimization_score,
                        opp.potential_revenue_impact,
                        opp.urgency_level,
                        opp.implementation_priority,
                        opp.reasoning,
                        json.dumps(opp.expected_benefit),
                    )

        except Exception as e:
            self.logger.error(f"Error storing optimization analysis: {e}")

    async def get_store_optimization_summary(self, store_id: int) -> Dict[str, Any]:
        """Get optimization summary for a specific store."""
        try:
            async with get_db_connection() as conn:
                # Get opportunities where this store is involved
                query = """
                SELECT 
                    'source' as role,
                    target_store_id as other_store_id,
                    product_id,
                    recommended_quantity,
                    optimization_score,
                    potential_revenue_impact,
                    implementation_priority
                FROM cross_store_inventory
                WHERE source_store_id = $1
                    AND analysis_date::date = CURRENT_DATE
                
                UNION ALL
                
                SELECT 
                    'target' as role,
                    source_store_id as other_store_id,
                    product_id,
                    recommended_quantity,
                    optimization_score,
                    potential_revenue_impact,
                    implementation_priority
                FROM cross_store_inventory
                WHERE target_store_id = $1
                    AND analysis_date::date = CURRENT_DATE
                ORDER BY optimization_score DESC
                """

                result = await conn.fetch(query, store_id)

                summary = {
                    "store_id": store_id,
                    "total_opportunities": len(result),
                    "as_source": len([r for r in result if r["role"] == "source"]),
                    "as_target": len([r for r in result if r["role"] == "target"]),
                    "total_potential_revenue": sum(
                        r["potential_revenue_impact"] or 0 for r in result
                    ),
                    "high_priority_count": len(
                        [
                            r
                            for r in result
                            if r["implementation_priority"] in ["immediate", "high"]
                        ]
                    ),
                    "opportunities": [dict(row) for row in result[:20]],  # Top 20
                }

                return summary

        except Exception as e:
            self.logger.error(f"Error getting store optimization summary: {e}")
            return {}

    async def get_city_optimization_overview(self, city_id: int) -> Dict[str, Any]:
        """Get optimization overview for an entire city."""
        try:
            async with get_db_connection() as conn:
                query = """
                SELECT 
                    COUNT(*) as total_opportunities,
                    COUNT(DISTINCT source_store_id) as stores_with_excess,
                    COUNT(DISTINCT target_store_id) as stores_with_shortage,
                    COUNT(DISTINCT product_id) as products_affected,
                    SUM(recommended_quantity) as total_units_to_transfer,
                    SUM(potential_revenue_impact) as total_revenue_potential,
                    AVG(optimization_score) as avg_optimization_score,
                    COUNT(CASE WHEN implementation_priority = 'immediate' THEN 1 END) as immediate_count,
                    COUNT(CASE WHEN implementation_priority = 'high' THEN 1 END) as high_priority_count
                FROM cross_store_inventory
                WHERE city_id = $1
                    AND analysis_date::date = CURRENT_DATE
                """

                result = await conn.fetchrow(query, city_id)

                if result:
                    return dict(result)
                else:
                    return {}

        except Exception as e:
            self.logger.error(f"Error getting city optimization overview: {e}")
            return {}

    async def execute_transfer_recommendation(
        self, source_store_id: int, target_store_id: int, product_id: int, quantity: int
    ) -> Dict[str, Any]:
        """Simulate execution of a transfer recommendation."""
        try:
            async with get_db_connection() as conn:
                # In a real system, this would integrate with inventory management
                # For now, we'll create a transfer record and update optimization status

                transfer_id = f"TRF-{datetime.now().strftime('%Y%m%d')}-{source_store_id}-{target_store_id}-{product_id}"

                # Log the transfer execution
                execution_result = {
                    "transfer_id": transfer_id,
                    "status": "initiated",
                    "source_store_id": source_store_id,
                    "target_store_id": target_store_id,
                    "product_id": product_id,
                    "quantity": quantity,
                    "initiated_at": datetime.now().isoformat(),
                    "estimated_completion": (
                        datetime.now() + timedelta(hours=2)
                    ).isoformat(),
                }

                # Update the optimization record as executed
                await conn.execute(
                    """
                    UPDATE cross_store_inventory
                    SET implementation_status = 'executed',
                        execution_date = NOW()
                    WHERE source_store_id = $1 
                        AND target_store_id = $2 
                        AND product_id = $3
                        AND analysis_date::date = CURRENT_DATE
                """,
                    source_store_id,
                    target_store_id,
                    product_id,
                )

                return execution_result

        except Exception as e:
            self.logger.error(f"Error executing transfer recommendation: {e}")
            return {"status": "error", "message": str(e)}

    async def get_transfer_history(
        self, store_id: Optional[int] = None, days: int = 30
    ) -> List[Dict]:
        """Get transfer history for analysis."""
        try:
            async with get_db_connection() as conn:
                query = (
                    """
                SELECT 
                    source_store_id,
                    target_store_id,
                    product_id,
                    recommended_quantity,
                    optimization_score,
                    potential_revenue_impact,
                    implementation_priority,
                    implementation_status,
                    execution_date,
                    analysis_date
                FROM cross_store_inventory
                WHERE analysis_date >= NOW() - INTERVAL '%s days'
                """
                    % days
                )

                params = []
                if store_id:
                    query += " AND (source_store_id = $1 OR target_store_id = $1)"
                    params.append(store_id)

                query += " ORDER BY analysis_date DESC, optimization_score DESC"

                result = await conn.fetch(query, *params)
                return [dict(row) for row in result]

        except Exception as e:
            self.logger.error(f"Error getting transfer history: {e}")
            return []
