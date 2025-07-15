"""
Real-Time Alerts Service for Dynamic Business Intelligence
Monitors store operations and generates intelligent alerts with severity levels,
business impact scoring, and actionable recommendations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal
import json
import logging
from dataclasses import dataclass
from enum import Enum

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


class AlertType(Enum):
    STOCKOUT = "stockout"
    DEMAND_SPIKE = "demand_spike"
    WEATHER_IMPACT = "weather_impact"
    PROMOTION_OPPORTUNITY = "promotion_opportunity"
    PERFORMANCE_DECLINE = "performance_decline"
    ANOMALY_DETECTION = "anomaly_detection"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertData:
    alert_type: AlertType
    severity: Severity
    store_id: int
    product_id: Optional[int]
    city_id: Optional[int]
    message: str
    data: Dict[str, Any]
    threshold_value: Optional[Decimal]
    current_value: Optional[Decimal]
    predicted_impact: Optional[Decimal]
    recommended_action: str
    business_impact_score: Decimal
    urgency_level: int
    affected_customers: Optional[int]
    estimated_revenue_impact: Optional[Decimal]


class RealTimeAlertsService:
    """Advanced real-time monitoring and alerting system."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.alert_thresholds = {
            "stockout_threshold": 5,  # Days of inventory
            "demand_spike_threshold": 2.0,  # Standard deviations
            "weather_correlation_threshold": 0.3,  # Correlation coefficient
            "promotion_roi_threshold": 1.2,  # Minimum ROI
            "performance_decline_threshold": 0.15,  # 15% decline
        }

    async def monitor_all_stores(self) -> List[AlertData]:
        """Run comprehensive monitoring across all stores and generate alerts."""
        try:
            alerts = []

            # Get all active stores
            stores = await self._get_active_stores()

            # Run monitoring tasks in parallel
            monitoring_tasks = [
                self._monitor_stockout_risks(stores),
                self._monitor_demand_anomalies(stores),
                self._monitor_weather_impacts(stores),
                self._monitor_promotion_opportunities(stores),
                self._monitor_performance_declines(stores),
            ]

            results = await asyncio.gather(*monitoring_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    alerts.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Monitoring task failed: {result}")

            # Prioritize and deduplicate alerts
            alerts = self._prioritize_alerts(alerts)

            # Store alerts in database
            await self._store_alerts(alerts)

            return alerts

        except Exception as e:
            self.logger.error(f"Error in monitor_all_stores: {e}")
            raise

    async def _monitor_stockout_risks(self, stores: List[Dict]) -> List[AlertData]:
        """Monitor inventory levels and predict stockout risks."""
        alerts = []

        try:
            async with get_db_connection() as conn:
                query = """
                SELECT 
                    s.store_id,
                    s.product_id,
                    sh.city_id,
                    ph.first_category_id,
                    ph.product_name,
                    AVG(s.units_sold) as avg_daily_sales,
                    STDDEV(s.units_sold) as sales_stddev,
                    MAX(s.stock_level) as current_stock,
                    COUNT(*) as days_data
                FROM sales_data s
                JOIN store_hierarchy sh ON s.store_id = sh.store_id
                JOIN product_hierarchy ph ON s.product_id = ph.product_id
                WHERE s.date >= NOW() - INTERVAL '30 days'
                    AND s.stock_level IS NOT NULL
                GROUP BY s.store_id, s.product_id, sh.city_id, 
                         ph.first_category_id, ph.product_name
                HAVING COUNT(*) >= 7
                """

                result = await conn.fetch(query)

                for row in result:
                    if row["current_stock"] and row["avg_daily_sales"]:
                        days_of_inventory = row["current_stock"] / max(
                            row["avg_daily_sales"], 1
                        )

                        if (
                            days_of_inventory
                            <= self.alert_thresholds["stockout_threshold"]
                        ):
                            # Calculate business impact
                            impact_score = min(100, (6 - days_of_inventory) * 20)
                            severity = self._calculate_stockout_severity(
                                days_of_inventory, row["avg_daily_sales"]
                            )

                            estimated_loss = (
                                row["avg_daily_sales"] * 10 * 25
                            )  # Avg price estimate

                            alert = AlertData(
                                alert_type=AlertType.STOCKOUT,
                                severity=severity,
                                store_id=row["store_id"],
                                product_id=row["product_id"],
                                city_id=row["city_id"],
                                message=f"LOW STOCK ALERT: {row['product_name']} at Store {row['store_id']} - Only {days_of_inventory:.1f} days remaining",
                                data={
                                    "days_of_inventory": days_of_inventory,
                                    "current_stock": row["current_stock"],
                                    "avg_daily_sales": row["avg_daily_sales"],
                                    "product_name": row["product_name"],
                                    "category_id": row["first_category_id"],
                                },
                                threshold_value=Decimal(
                                    str(self.alert_thresholds["stockout_threshold"])
                                ),
                                current_value=Decimal(str(days_of_inventory)),
                                predicted_impact=Decimal(str(estimated_loss)),
                                recommended_action=f"URGENT: Reorder {int(row['avg_daily_sales'] * 14)} units immediately. Consider emergency transfer from nearby stores.",
                                business_impact_score=Decimal(str(impact_score)),
                                urgency_level=5 if days_of_inventory <= 2 else 4,
                                affected_customers=int(row["avg_daily_sales"] * 3),
                                estimated_revenue_impact=Decimal(str(estimated_loss)),
                            )
                            alerts.append(alert)

        except Exception as e:
            self.logger.error(f"Error monitoring stockout risks: {e}")

        return alerts

    async def _monitor_demand_anomalies(self, stores: List[Dict]) -> List[AlertData]:
        """Detect unusual demand patterns and spikes."""
        alerts = []

        try:
            async with get_db_connection() as conn:
                query = """
                WITH recent_sales AS (
                    SELECT 
                        store_id,
                        product_id,
                        date,
                        units_sold,
                        LAG(units_sold, 7) OVER (PARTITION BY store_id, product_id ORDER BY date) as units_sold_week_ago,
                        AVG(units_sold) OVER (
                            PARTITION BY store_id, product_id 
                            ORDER BY date 
                            ROWS BETWEEN 21 PRECEDING AND 8 PRECEDING
                        ) as avg_baseline,
                        STDDEV(units_sold) OVER (
                            PARTITION BY store_id, product_id 
                            ORDER BY date 
                            ROWS BETWEEN 21 PRECEDING AND 8 PRECEDING
                        ) as stddev_baseline
                    FROM sales_data
                    WHERE date >= NOW() - INTERVAL '30 days'
                ),
                anomalies AS (
                    SELECT 
                        rs.*,
                        sh.city_id,
                        ph.product_name,
                        CASE 
                            WHEN stddev_baseline > 0 
                            THEN (units_sold - avg_baseline) / stddev_baseline
                            ELSE 0
                        END as z_score
                    FROM recent_sales rs
                    JOIN store_hierarchy sh ON rs.store_id = sh.store_id
                    JOIN product_hierarchy ph ON rs.product_id = ph.product_id
                    WHERE rs.date >= NOW() - INTERVAL '3 days'
                        AND rs.avg_baseline IS NOT NULL
                        AND rs.stddev_baseline > 0
                )
                SELECT *
                FROM anomalies
                WHERE ABS(z_score) >= %s
                ORDER BY ABS(z_score) DESC
                LIMIT 50
                """

                result = await conn.fetch(
                    query, self.alert_thresholds["demand_spike_threshold"]
                )

                for row in result:
                    z_score = abs(row["z_score"])
                    is_spike = row["z_score"] > 0

                    severity = self._calculate_anomaly_severity(z_score)
                    impact_score = min(100, z_score * 25)

                    if is_spike:
                        message = f"DEMAND SPIKE: {row['product_name']} at Store {row['store_id']} - {row['units_sold']} units sold ({z_score:.1f}σ above normal)"
                        action = f"Increase inventory by {int(row['units_sold'] * 0.5)} units. Check for promotion opportunities."
                        alert_type = AlertType.DEMAND_SPIKE
                    else:
                        message = f"DEMAND DROP: {row['product_name']} at Store {row['store_id']} - Only {row['units_sold']} units sold ({z_score:.1f}σ below normal)"
                        action = f"Investigate cause: competitor promotion, quality issue, or market change. Consider promotional response."
                        alert_type = AlertType.ANOMALY_DETECTION

                    estimated_impact = (
                        abs(row["units_sold"] - row["avg_baseline"]) * 25
                    )  # Avg price

                    alert = AlertData(
                        alert_type=alert_type,
                        severity=severity,
                        store_id=row["store_id"],
                        product_id=row["product_id"],
                        city_id=row["city_id"],
                        message=message,
                        data={
                            "z_score": z_score,
                            "current_sales": row["units_sold"],
                            "baseline_sales": row["avg_baseline"],
                            "product_name": row["product_name"],
                            "date": row["date"].isoformat() if row["date"] else None,
                        },
                        threshold_value=Decimal(
                            str(self.alert_thresholds["demand_spike_threshold"])
                        ),
                        current_value=Decimal(str(z_score)),
                        predicted_impact=Decimal(str(estimated_impact)),
                        recommended_action=action,
                        business_impact_score=Decimal(str(impact_score)),
                        urgency_level=min(5, int(z_score)),
                        affected_customers=int(
                            max(row["units_sold"], row["avg_baseline"]) * 2
                        ),
                        estimated_revenue_impact=Decimal(str(estimated_impact)),
                    )
                    alerts.append(alert)

        except Exception as e:
            self.logger.error(f"Error monitoring demand anomalies: {e}")

        return alerts

    async def _monitor_weather_impacts(self, stores: List[Dict]) -> List[AlertData]:
        """Monitor weather conditions and their impact on sales."""
        alerts = []

        try:
            async with get_db_connection() as conn:
                # Get products with strong weather correlations
                query = """
                WITH weather_correlations AS (
                    SELECT 
                        s.store_id,
                        s.product_id,
                        sh.city_id,
                        ph.product_name,
                        CORR(s.units_sold, s.temperature) as temp_corr,
                        CORR(s.units_sold, s.humidity) as humidity_corr,
                        CORR(s.units_sold, s.precipitation) as precip_corr,
                        AVG(s.units_sold) as avg_sales,
                        AVG(s.temperature) as avg_temp,
                        STDDEV(s.temperature) as temp_stddev
                    FROM sales_data s
                    JOIN store_hierarchy sh ON s.store_id = sh.store_id
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '90 days'
                        AND s.temperature IS NOT NULL
                    GROUP BY s.store_id, s.product_id, sh.city_id, ph.product_name
                    HAVING COUNT(*) >= 30
                ),
                recent_weather AS (
                    SELECT 
                        store_id,
                        product_id,
                        temperature,
                        humidity,
                        precipitation,
                        units_sold,
                        date
                    FROM sales_data
                    WHERE date >= NOW() - INTERVAL '3 days'
                        AND temperature IS NOT NULL
                )
                SELECT 
                    wc.*,
                    rw.temperature as current_temp,
                    rw.humidity as current_humidity,
                    rw.precipitation as current_precip,
                    rw.units_sold as recent_sales,
                    rw.date as recent_date
                FROM weather_correlations wc
                JOIN recent_weather rw ON wc.store_id = rw.store_id AND wc.product_id = rw.product_id
                WHERE ABS(wc.temp_corr) >= %s
                    OR ABS(wc.humidity_corr) >= %s
                    OR ABS(wc.precip_corr) >= %s
                """

                threshold = self.alert_thresholds["weather_correlation_threshold"]
                result = await conn.fetch(query, threshold, threshold, threshold)

                for row in result:
                    # Check if current weather is unusual
                    if row["temp_stddev"] and row["temp_stddev"] > 0:
                        temp_z_score = abs(
                            (row["current_temp"] - row["avg_temp"]) / row["temp_stddev"]
                        )

                        if temp_z_score >= 2.0:  # Unusual weather
                            impact_direction = (
                                "increase" if row["temp_corr"] > 0 else "decrease"
                            )
                            if row["temp_corr"] < 0:
                                impact_direction = (
                                    "decrease"
                                    if row["current_temp"] > row["avg_temp"]
                                    else "increase"
                                )

                            predicted_change = (
                                abs(row["temp_corr"])
                                * temp_z_score
                                * row["avg_sales"]
                                * 0.2
                            )
                            severity = (
                                Severity.MEDIUM if temp_z_score >= 3.0 else Severity.LOW
                            )

                            alert = AlertData(
                                alert_type=AlertType.WEATHER_IMPACT,
                                severity=severity,
                                store_id=row["store_id"],
                                product_id=row["product_id"],
                                city_id=row["city_id"],
                                message=f"WEATHER IMPACT: {row['product_name']} at Store {row['store_id']} - Unusual temperature may {impact_direction} demand",
                                data={
                                    "temperature_correlation": row["temp_corr"],
                                    "current_temperature": row["current_temp"],
                                    "average_temperature": row["avg_temp"],
                                    "temperature_z_score": temp_z_score,
                                    "product_name": row["product_name"],
                                },
                                threshold_value=Decimal("2.0"),
                                current_value=Decimal(str(temp_z_score)),
                                predicted_impact=Decimal(str(predicted_change * 25)),
                                recommended_action=f"Adjust inventory: expect {impact_direction} in demand by ~{predicted_change:.0f} units. Monitor closely.",
                                business_impact_score=Decimal(
                                    str(min(100, abs(row["temp_corr"]) * 100))
                                ),
                                urgency_level=3 if temp_z_score >= 3.0 else 2,
                                affected_customers=int(row["avg_sales"] * 2),
                                estimated_revenue_impact=Decimal(
                                    str(predicted_change * 25)
                                ),
                            )
                            alerts.append(alert)

        except Exception as e:
            self.logger.error(f"Error monitoring weather impacts: {e}")

        return alerts

    async def _monitor_promotion_opportunities(
        self, stores: List[Dict]
    ) -> List[AlertData]:
        """Identify promotion opportunities based on inventory and demand patterns."""
        alerts = []

        try:
            async with get_db_connection() as conn:
                query = """
                WITH promotion_analysis AS (
                    SELECT 
                        s.store_id,
                        s.product_id,
                        sh.city_id,
                        ph.product_name,
                        ph.first_category_id,
                        AVG(s.units_sold) as avg_sales,
                        MAX(s.stock_level) as current_stock,
                        AVG(CASE WHEN s.discount_percentage > 0 THEN s.units_sold END) as promo_avg_sales,
                        AVG(CASE WHEN s.discount_percentage = 0 THEN s.units_sold END) as regular_avg_sales,
                        COUNT(CASE WHEN s.discount_percentage > 0 THEN 1 END) as promo_days,
                        AVG(s.discount_percentage) as avg_discount
                    FROM sales_data s
                    JOIN store_hierarchy sh ON s.store_id = sh.store_id
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '90 days'
                    GROUP BY s.store_id, s.product_id, sh.city_id, ph.product_name, ph.first_category_id
                    HAVING COUNT(*) >= 30
                )
                SELECT *,
                    CASE 
                        WHEN promo_avg_sales IS NOT NULL AND regular_avg_sales > 0
                        THEN promo_avg_sales / regular_avg_sales
                        ELSE 1.0
                    END as promotion_lift
                FROM promotion_analysis
                WHERE current_stock > avg_sales * 10  -- High inventory
                    AND promo_days >= 5  -- Enough promotion history
                ORDER BY current_stock / avg_sales DESC
                LIMIT 20
                """

                result = await conn.fetch(query)

                for row in result:
                    if (
                        row["promotion_lift"]
                        >= self.alert_thresholds["promotion_roi_threshold"]
                    ):
                        days_of_inventory = row["current_stock"] / max(
                            row["avg_sales"], 1
                        )

                        if days_of_inventory > 14:  # More than 2 weeks inventory
                            potential_uplift = (
                                (row["promotion_lift"] - 1) * row["avg_sales"] * 7
                            )  # Weekly uplift
                            revenue_opportunity = (
                                potential_uplift * 25
                            )  # Avg price estimate

                            alert = AlertData(
                                alert_type=AlertType.PROMOTION_OPPORTUNITY,
                                severity=Severity.MEDIUM,
                                store_id=row["store_id"],
                                product_id=row["product_id"],
                                city_id=row["city_id"],
                                message=f"PROMOTION OPPORTUNITY: {row['product_name']} at Store {row['store_id']} - High inventory with strong promotion history",
                                data={
                                    "days_of_inventory": days_of_inventory,
                                    "promotion_lift": row["promotion_lift"],
                                    "avg_discount": row["avg_discount"],
                                    "product_name": row["product_name"],
                                    "category_id": row["first_category_id"],
                                },
                                threshold_value=Decimal("14"),
                                current_value=Decimal(str(days_of_inventory)),
                                predicted_impact=Decimal(str(revenue_opportunity)),
                                recommended_action=f"Launch {row['avg_discount']:.0f}% promotion. Expected {row['promotion_lift']:.1f}x sales boost, clearing {potential_uplift:.0f} extra units.",
                                business_impact_score=Decimal(
                                    str(min(100, (row["promotion_lift"] - 1) * 100))
                                ),
                                urgency_level=2,
                                affected_customers=int(potential_uplift * 1.5),
                                estimated_revenue_impact=Decimal(
                                    str(revenue_opportunity)
                                ),
                            )
                            alerts.append(alert)

        except Exception as e:
            self.logger.error(f"Error monitoring promotion opportunities: {e}")

        return alerts

    async def _monitor_performance_declines(
        self, stores: List[Dict]
    ) -> List[AlertData]:
        """Monitor store performance and detect declining trends."""
        alerts = []

        try:
            async with get_db_connection() as conn:
                query = """
                WITH performance_comparison AS (
                    SELECT 
                        store_id,
                        sh.city_id,
                        sh.store_format,
                        AVG(CASE WHEN date >= NOW() - INTERVAL '7 days' THEN units_sold END) as recent_avg,
                        AVG(CASE WHEN date BETWEEN NOW() - INTERVAL '28 days' AND NOW() - INTERVAL '7 days' THEN units_sold END) as baseline_avg,
                        SUM(CASE WHEN date >= NOW() - INTERVAL '7 days' THEN units_sold * 25 END) as recent_revenue,
                        SUM(CASE WHEN date BETWEEN NOW() - INTERVAL '28 days' AND NOW() - INTERVAL '7 days' THEN units_sold * 25 END) as baseline_revenue,
                        COUNT(DISTINCT CASE WHEN date >= NOW() - INTERVAL '7 days' THEN product_id END) as recent_products,
                        COUNT(DISTINCT CASE WHEN date BETWEEN NOW() - INTERVAL '28 days' AND NOW() - INTERVAL '7 days' THEN product_id END) as baseline_products
                    FROM sales_data s
                    JOIN store_hierarchy sh ON s.store_id = sh.store_id
                    WHERE s.date >= NOW() - INTERVAL '28 days'
                    GROUP BY store_id, sh.city_id, sh.store_format
                    HAVING COUNT(*) >= 20
                )
                SELECT *,
                    CASE 
                        WHEN baseline_avg > 0 
                        THEN (recent_avg - baseline_avg) / baseline_avg
                        ELSE 0
                    END as performance_change
                FROM performance_comparison
                WHERE baseline_avg > 0
                """

                result = await conn.fetch(query)

                for row in result:
                    performance_change = row["performance_change"]

                    if (
                        performance_change
                        <= -self.alert_thresholds["performance_decline_threshold"]
                    ):
                        severity = self._calculate_performance_severity(
                            abs(performance_change)
                        )
                        impact_score = min(100, abs(performance_change) * 200)

                        revenue_impact = (
                            row["baseline_revenue"] - row["recent_revenue"]
                            if row["baseline_revenue"] and row["recent_revenue"]
                            else 0
                        )

                        alert = AlertData(
                            alert_type=AlertType.PERFORMANCE_DECLINE,
                            severity=severity,
                            store_id=row["store_id"],
                            product_id=None,
                            city_id=row["city_id"],
                            message=f"PERFORMANCE DECLINE: Store {row['store_id']} sales down {abs(performance_change)*100:.1f}% vs previous 3 weeks",
                            data={
                                "performance_change": performance_change,
                                "recent_avg_sales": row["recent_avg"],
                                "baseline_avg_sales": row["baseline_avg"],
                                "store_format": row["store_format"],
                                "recent_products": row["recent_products"],
                                "baseline_products": row["baseline_products"],
                            },
                            threshold_value=Decimal(
                                str(
                                    self.alert_thresholds[
                                        "performance_decline_threshold"
                                    ]
                                )
                            ),
                            current_value=Decimal(str(abs(performance_change))),
                            predicted_impact=Decimal(str(revenue_impact)),
                            recommended_action=f"Urgent investigation needed: check staffing, inventory, competition, local events. Consider immediate intervention.",
                            business_impact_score=Decimal(str(impact_score)),
                            urgency_level=4 if abs(performance_change) >= 0.25 else 3,
                            affected_customers=int(
                                row["baseline_avg"] * 7 * 2
                            ),  # Weekly customers estimate
                            estimated_revenue_impact=Decimal(str(revenue_impact)),
                        )
                        alerts.append(alert)

        except Exception as e:
            self.logger.error(f"Error monitoring performance declines: {e}")

        return alerts

    def _calculate_stockout_severity(
        self, days_remaining: float, avg_sales: float
    ) -> Severity:
        """Calculate severity based on stockout risk."""
        if days_remaining <= 1:
            return Severity.CRITICAL
        elif days_remaining <= 2:
            return Severity.HIGH
        elif days_remaining <= 3:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _calculate_anomaly_severity(self, z_score: float) -> Severity:
        """Calculate severity based on statistical anomaly level."""
        if z_score >= 4.0:
            return Severity.CRITICAL
        elif z_score >= 3.0:
            return Severity.HIGH
        elif z_score >= 2.5:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _calculate_performance_severity(self, decline_percentage: float) -> Severity:
        """Calculate severity based on performance decline."""
        if decline_percentage >= 0.30:
            return Severity.CRITICAL
        elif decline_percentage >= 0.20:
            return Severity.HIGH
        elif decline_percentage >= 0.15:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _prioritize_alerts(self, alerts: List[AlertData]) -> List[AlertData]:
        """Prioritize and deduplicate alerts."""
        # Sort by business impact score and urgency
        alerts.sort(
            key=lambda x: (x.business_impact_score, x.urgency_level), reverse=True
        )

        # Remove duplicates for same store/product combinations
        seen = set()
        deduplicated = []

        for alert in alerts:
            key = (alert.store_id, alert.product_id, alert.alert_type.value)
            if key not in seen:
                seen.add(key)
                deduplicated.append(alert)

        return deduplicated[:50]  # Limit to top 50 alerts

    async def _get_active_stores(self) -> List[Dict]:
        """Get list of active stores."""
        try:
            async with get_db_connection() as conn:
                query = """
                SELECT DISTINCT store_id, city_id
                FROM store_hierarchy
                WHERE store_id IN (
                    SELECT DISTINCT store_id
                    FROM sales_data
                    WHERE date >= NOW() - INTERVAL '7 days'
                )
                ORDER BY store_id
                """
                result = await conn.fetch(query)
                return [dict(row) for row in result]
        except Exception as e:
            self.logger.error(f"Error getting active stores: {e}")
            return []

    async def _store_alerts(self, alerts: List[AlertData]) -> None:
        """Store alerts in the database."""
        try:
            async with get_db_connection() as conn:
                for alert in alerts:
                    await conn.execute(
                        """
                        INSERT INTO real_time_alerts (
                            alert_type, severity, store_id, product_id, city_id,
                            alert_message, alert_data, threshold_value, current_value,
                            predicted_impact, recommended_action, business_impact_score,
                            urgency_level, affected_customers, estimated_revenue_impact,
                            expires_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """,
                        alert.alert_type.value,
                        alert.severity.value,
                        alert.store_id,
                        alert.product_id,
                        alert.city_id,
                        alert.message,
                        json.dumps(alert.data),
                        alert.threshold_value,
                        alert.current_value,
                        alert.predicted_impact,
                        alert.recommended_action,
                        alert.business_impact_score,
                        alert.urgency_level,
                        alert.affected_customers,
                        alert.estimated_revenue_impact,
                        datetime.now() + timedelta(days=7),  # Expire in 7 days
                    )
        except Exception as e:
            self.logger.error(f"Error storing alerts: {e}")

    async def get_active_alerts(
        self, store_id: Optional[int] = None, severity: Optional[str] = None
    ) -> List[Dict]:
        """Get active alerts with optional filtering."""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT 
                        alert_id,
                        alert_type,
                        severity,
                        store_id,
                        product_id,
                        city_id,
                        alert_message,
                        alert_data,
                        threshold_value,
                        current_value,
                        predicted_impact,
                        recommended_action,
                        business_impact_score,
                        urgency_level,
                        affected_customers,
                        estimated_revenue_impact,
                        created_at,
                        is_acknowledged,
                        resolution_status
                    FROM real_time_alerts
                    WHERE resolution_status = 'open'
                        AND (expires_at IS NULL OR expires_at > NOW())
                """
                params: List[Any] = []

                if store_id:
                    query += " AND store_id = $1"
                    params.append(store_id)

                if severity:
                    param_num = len(params) + 1
                    query += f" AND severity = ${param_num}"
                    params.append(severity)

                query += (
                    " ORDER BY business_impact_score DESC, created_at DESC LIMIT 100"
                )

                result = await conn.fetch(query, *params)
                return [dict(row) for row in result]

        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            async with get_db_connection() as conn:
                result = await conn.execute(
                    """
                    UPDATE real_time_alerts
                    SET is_acknowledged = TRUE,
                        acknowledged_by = $2,
                        acknowledged_at = NOW()
                    WHERE alert_id = $1
                """,
                    alert_id,
                    acknowledged_by,
                )

                return result == "UPDATE 1"

        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False

    async def resolve_alert(self, alert_id: str, resolution_status: str) -> bool:
        """Resolve an alert with a status."""
        try:
            async with get_db_connection() as conn:
                result = await conn.execute(
                    """
                    UPDATE real_time_alerts
                    SET resolution_status = $2
                    WHERE alert_id = $1
                """,
                    alert_id,
                    resolution_status,
                )

                return result == "UPDATE 1"

        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            return False

    async def get_alert_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get alert summary statistics."""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT 
                        alert_type,
                        severity,
                        COUNT(*) as count,
                        AVG(business_impact_score) as avg_impact,
                        SUM(estimated_revenue_impact) as total_revenue_impact,
                        COUNT(CASE WHEN is_acknowledged THEN 1 END) as acknowledged_count
                    FROM real_time_alerts
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY alert_type, severity
                    ORDER BY alert_type, severity
                """

                result = await conn.fetch(query.replace("%s", str(days)))

                summary = {
                    "total_alerts": sum(row["count"] for row in result),
                    "by_type": {},
                    "by_severity": {},
                    "total_revenue_impact": sum(
                        row["total_revenue_impact"] or 0 for row in result
                    ),
                    "acknowledgment_rate": 0,
                }

                total_acknowledged = sum(row["acknowledged_count"] for row in result)
                if summary["total_alerts"] > 0:
                    summary["acknowledgment_rate"] = (
                        total_acknowledged / summary["total_alerts"]
                    )

                for row in result:
                    alert_type = row["alert_type"]
                    severity = row["severity"]

                    if alert_type not in summary["by_type"]:
                        summary["by_type"][alert_type] = 0
                    summary["by_type"][alert_type] += row["count"]

                    if severity not in summary["by_severity"]:
                        summary["by_severity"][severity] = 0
                    summary["by_severity"][severity] += row["count"]

                return summary

        except Exception as e:
            self.logger.error(f"Error getting alert summary: {e}")
            return {}
