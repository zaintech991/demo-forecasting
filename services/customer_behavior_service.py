"""
Customer Behavior Analysis Service
Analyzes shopping patterns, customer segmentation, and provides 
personalized insights for better customer engagement and retention.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal
import asyncio
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CustomerSegment:
    segment_id: int
    segment_name: str
    customer_count: int
    avg_transaction_value: Decimal
    avg_frequency: float
    lifetime_value: Decimal
    churn_probability: float
    preferred_categories: List[str]
    shopping_patterns: Dict[str, Any]
    engagement_strategy: str


@dataclass
class ShoppingPattern:
    pattern_id: str
    pattern_type: str  # 'seasonal', 'promotional', 'behavioral'
    description: str
    frequency: int
    confidence_score: float
    affected_customers: int
    revenue_impact: Decimal
    optimization_opportunity: str


@dataclass
class CustomerInsight:
    insight_type: str
    customer_segment: str
    insight_summary: str
    data_points: Dict[str, Any]
    recommended_actions: List[str]
    impact_potential: str
    implementation_difficulty: str


class CustomerBehaviorService:
    """Advanced customer behavior analysis and segmentation system."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.analysis_params = {
            "min_transactions_per_customer": 5,
            "analysis_period_days": 90,
            "num_customer_segments": 5,
            "churn_threshold_days": 30,
            "high_value_threshold": 1000,
            "frequency_threshold": 0.1,  # Transactions per day
        }

    async def analyze_customer_behavior(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Comprehensive customer behavior analysis."""
        try:
            # Run parallel analysis tasks
            segmentation_task = self._perform_customer_segmentation(store_id, city_id)
            patterns_task = self._identify_shopping_patterns(store_id, city_id)
            lifecycle_task = self._analyze_customer_lifecycle(store_id, city_id)
            preferences_task = self._analyze_product_preferences(store_id, city_id)

            segments, patterns, lifecycle, preferences = await asyncio.gather(
                segmentation_task, patterns_task, lifecycle_task, preferences_task
            )

            # Generate actionable insights
            insights = await self._generate_customer_insights(
                segments, patterns, lifecycle, preferences
            )

            # Store analysis results
            await self._store_behavior_analysis(
                store_id, city_id, segments, patterns, insights
            )

            return {
                "customer_segments": segments,
                "shopping_patterns": patterns[:15],  # Top 15 patterns
                "lifecycle_analysis": lifecycle,
                "product_preferences": preferences,
                "actionable_insights": insights,
                "analysis_summary": self._generate_analysis_summary(
                    segments, patterns, lifecycle
                ),
                "analysis_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error in customer behavior analysis: {e}")
            raise

    async def _perform_customer_segmentation(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[CustomerSegment]:
        """Perform advanced customer segmentation using RFM and behavioral analysis."""
        try:
            async with get_db_connection() as conn:
                # Create synthetic customer data based on transaction patterns
                # In a real system, this would use actual customer IDs
                query = (
                    """
                WITH customer_transactions AS (
                    SELECT 
                        s.store_id,
                        s.product_id,
                        s.date,
                        s.units_sold,
                        s.units_sold * 25 as transaction_value,  -- Estimated value
                        sh.city_id,
                        ph.first_category_id,
                        ph.first_category_name,
                        -- Create synthetic customer ID based on store, product, and date patterns
                        MD5(CONCAT(s.store_id, s.product_id, EXTRACT(DOW FROM s.date)))::uuid as customer_id
                    FROM sales_data s
                    JOIN store_hierarchy sh ON s.store_id = sh.store_id
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '%s days'
                        AND s.units_sold > 0
                """
                    % self.analysis_params["analysis_period_days"]
                )

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND sh.city_id = $1"
                    params.append(city_id)

                query += (
                    """
                ),
                customer_metrics AS (
                    SELECT 
                        customer_id,
                        COUNT(*) as frequency,
                        SUM(transaction_value) as monetary_value,
                        AVG(transaction_value) as avg_transaction_value,
                        MAX(date) as last_purchase_date,
                        MIN(date) as first_purchase_date,
                        EXTRACT(DAYS FROM NOW() - MAX(date)) as days_since_last_purchase,
                        COUNT(DISTINCT first_category_id) as category_diversity,
                        COUNT(DISTINCT store_id) as store_loyalty,
                        STDDEV(transaction_value) as spending_consistency
                    FROM customer_transactions
                    GROUP BY customer_id
                    HAVING COUNT(*) >= %s
                )
                SELECT 
                    *,
                    CASE 
                        WHEN days_since_last_purchase <= 7 THEN 5
                        WHEN days_since_last_purchase <= 14 THEN 4
                        WHEN days_since_last_purchase <= 30 THEN 3
                        WHEN days_since_last_purchase <= 60 THEN 2
                        ELSE 1
                    END as recency_score,
                    CASE 
                        WHEN frequency >= 20 THEN 5
                        WHEN frequency >= 15 THEN 4
                        WHEN frequency >= 10 THEN 3
                        WHEN frequency >= 5 THEN 2
                        ELSE 1
                    END as frequency_score,
                    CASE 
                        WHEN monetary_value >= 2000 THEN 5
                        WHEN monetary_value >= 1000 THEN 4
                        WHEN monetary_value >= 500 THEN 3
                        WHEN monetary_value >= 200 THEN 2
                        ELSE 1
                    END as monetary_score
                FROM customer_metrics
                ORDER BY monetary_value DESC
                """
                    % self.analysis_params["min_transactions_per_customer"]
                )

                result = await conn.fetch(query, *params)

                if not result:
                    return []

                # Convert to DataFrame for clustering
                df = pd.DataFrame([dict(row) for row in result])

                # Prepare features for clustering
                features = [
                    "recency_score",
                    "frequency_score",
                    "monetary_score",
                    "category_diversity",
                    "store_loyalty",
                    "spending_consistency",
                ]

                # Handle missing values
                df[features] = df[features].fillna(df[features].median())

                # Standardize features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(df[features])

                # Perform K-means clustering
                kmeans = KMeans(
                    n_clusters=self.analysis_params["num_customer_segments"],
                    random_state=42,
                    n_init=10,
                )
                df["segment"] = kmeans.fit_predict(scaled_features)

                # Analyze each segment
                segments = []

                for segment_id in range(self.analysis_params["num_customer_segments"]):
                    segment_data = df[df["segment"] == segment_id]

                    if len(segment_data) == 0:
                        continue

                    # Calculate segment characteristics
                    avg_recency = segment_data["recency_score"].mean()
                    avg_frequency = segment_data["frequency_score"].mean()
                    avg_monetary = segment_data["monetary_score"].mean()

                    # Determine segment name based on RFM scores
                    segment_name = self._determine_segment_name(
                        avg_recency, avg_frequency, avg_monetary
                    )

                    # Calculate churn probability
                    churn_probability = len(
                        segment_data[
                            segment_data["days_since_last_purchase"]
                            > self.analysis_params["churn_threshold_days"]
                        ]
                    ) / len(segment_data)

                    # Get preferred categories (simulate based on segment characteristics)
                    if avg_monetary >= 4:
                        preferred_categories = [
                            "Premium Electronics",
                            "Luxury Items",
                            "Organic Foods",
                        ]
                    elif avg_frequency >= 4:
                        preferred_categories = [
                            "Daily Essentials",
                            "Groceries",
                            "Personal Care",
                        ]
                    elif avg_recency >= 4:
                        preferred_categories = [
                            "Trending Items",
                            "New Arrivals",
                            "Seasonal Products",
                        ]
                    else:
                        preferred_categories = [
                            "Value Items",
                            "Discount Products",
                            "Basic Necessities",
                        ]

                    # Determine engagement strategy
                    engagement_strategy = self._determine_engagement_strategy(
                        avg_recency, avg_frequency, avg_monetary, churn_probability
                    )

                    segment = CustomerSegment(
                        segment_id=segment_id,
                        segment_name=segment_name,
                        customer_count=len(segment_data),
                        avg_transaction_value=Decimal(
                            str(segment_data["avg_transaction_value"].mean())
                        ),
                        avg_frequency=float(segment_data["frequency"].mean()),
                        lifetime_value=Decimal(
                            str(segment_data["monetary_value"].mean())
                        ),
                        churn_probability=churn_probability,
                        preferred_categories=preferred_categories,
                        shopping_patterns={
                            "avg_category_diversity": float(
                                segment_data["category_diversity"].mean()
                            ),
                            "store_loyalty_score": float(
                                segment_data["store_loyalty"].mean()
                            ),
                            "spending_consistency": float(
                                segment_data["spending_consistency"].mean()
                            ),
                            "purchase_frequency_days": float(
                                90 / segment_data["frequency"].mean()
                            ),
                            "recency_score": avg_recency,
                            "frequency_score": avg_frequency,
                            "monetary_score": avg_monetary,
                        },
                        engagement_strategy=engagement_strategy,
                    )

                    segments.append(segment)

                # Sort segments by lifetime value
                segments.sort(key=lambda x: x.lifetime_value, reverse=True)

                return segments

        except Exception as e:
            self.logger.error(f"Error performing customer segmentation: {e}")
            return []

    def _determine_segment_name(
        self, recency: float, frequency: float, monetary: float
    ) -> str:
        """Determine customer segment name based on RFM scores."""
        if recency >= 4 and frequency >= 4 and monetary >= 4:
            return "VIP Champions"
        elif recency >= 3 and frequency >= 4 and monetary >= 3:
            return "Loyal Customers"
        elif recency >= 4 and frequency < 3 and monetary >= 4:
            return "Big Spenders"
        elif recency >= 4 and frequency >= 3 and monetary < 3:
            return "Potential Loyalists"
        elif recency < 3 and frequency >= 4 and monetary >= 3:
            return "At Risk Loyal"
        elif recency < 3 and frequency < 3 and monetary >= 4:
            return "Cannot Lose Them"
        elif recency >= 3 and frequency < 3 and monetary < 3:
            return "New Customers"
        elif recency < 3 and frequency < 3 and monetary < 3:
            return "Lost Customers"
        else:
            return "Regular Customers"

    def _determine_engagement_strategy(
        self, recency: float, frequency: float, monetary: float, churn_prob: float
    ) -> str:
        """Determine optimal engagement strategy for customer segment."""
        if churn_prob > 0.5:
            return "retention_focused"
        elif monetary >= 4:
            return "premium_experience"
        elif frequency >= 4:
            return "loyalty_rewards"
        elif recency >= 4:
            return "cross_sell_upsell"
        else:
            return "reactivation_campaign"

    async def _identify_shopping_patterns(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[ShoppingPattern]:
        """Identify key shopping patterns and behaviors."""
        try:
            patterns = []

            # Pattern 1: Seasonal Shopping Patterns
            seasonal_patterns = await self._analyze_seasonal_patterns(store_id, city_id)
            patterns.extend(seasonal_patterns)

            # Pattern 2: Promotional Response Patterns
            promo_patterns = await self._analyze_promotional_patterns(store_id, city_id)
            patterns.extend(promo_patterns)

            # Pattern 3: Day-of-Week Patterns
            dow_patterns = await self._analyze_day_of_week_patterns(store_id, city_id)
            patterns.extend(dow_patterns)

            # Pattern 4: Product Affinity Patterns
            affinity_patterns = await self._analyze_product_affinity_patterns(
                store_id, city_id
            )
            patterns.extend(affinity_patterns)

            # Sort patterns by confidence score and revenue impact
            patterns.sort(
                key=lambda x: (x.confidence_score * float(x.revenue_impact)),
                reverse=True,
            )

            return patterns

        except Exception as e:
            self.logger.error(f"Error identifying shopping patterns: {e}")
            return []

    async def _analyze_seasonal_patterns(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[ShoppingPattern]:
        """Analyze seasonal shopping patterns."""
        patterns = []

        try:
            async with get_db_connection() as conn:
                query = """
                WITH seasonal_sales AS (
                    SELECT 
                        EXTRACT(MONTH FROM s.date) as month,
                        EXTRACT(QUARTER FROM s.date) as quarter,
                        ph.first_category_name,
                        SUM(s.units_sold) as total_units,
                        SUM(s.units_sold * 25) as total_revenue,
                        COUNT(DISTINCT s.date) as active_days,
                        AVG(s.units_sold) as avg_daily_sales
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '365 days'
                """

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY EXTRACT(MONTH FROM s.date), EXTRACT(QUARTER FROM s.date), ph.first_category_name
                    HAVING COUNT(*) >= 10
                ),
                seasonal_variance AS (
                    SELECT 
                        first_category_name,
                        AVG(avg_daily_sales) as overall_avg,
                        STDDEV(avg_daily_sales) as seasonal_variance,
                        MAX(avg_daily_sales) as peak_sales,
                        MIN(avg_daily_sales) as low_sales
                    FROM seasonal_sales
                    GROUP BY first_category_name
                )
                SELECT 
                    sv.first_category_name,
                    sv.seasonal_variance / NULLIF(sv.overall_avg, 0) as variation_coefficient,
                    sv.peak_sales / NULLIF(sv.low_sales, 0) as peak_to_low_ratio,
                    SUM(ss.total_revenue) as annual_revenue
                FROM seasonal_variance sv
                JOIN seasonal_sales ss ON sv.first_category_name = ss.first_category_name
                WHERE sv.seasonal_variance > sv.overall_avg * 0.3  -- High seasonal variation
                GROUP BY sv.first_category_name, sv.seasonal_variance, sv.overall_avg, sv.peak_sales, sv.low_sales
                ORDER BY sv.seasonal_variance / NULLIF(sv.overall_avg, 0) DESC
                LIMIT 10
                """

                result = await conn.fetch(query, *params)

                for row in result:
                    if (
                        row["variation_coefficient"]
                        and row["variation_coefficient"] > 0.3
                    ):
                        pattern = ShoppingPattern(
                            pattern_id=f"SEASONAL-{row['first_category_name'].replace(' ', '')[:10]}",
                            pattern_type="seasonal",
                            description=f"Strong seasonal variation in {row['first_category_name']} category",
                            frequency=int(12),  # Monthly seasonal pattern
                            confidence_score=min(1.0, row["variation_coefficient"]),
                            affected_customers=int(
                                row["annual_revenue"] / 500
                            ),  # Estimate based on revenue
                            revenue_impact=Decimal(str(row["annual_revenue"])),
                            optimization_opportunity=f"Optimize inventory and promotions for {row['first_category_name']} seasonal peaks (up to {row['peak_to_low_ratio']:.1f}x variation)",
                        )
                        patterns.append(pattern)

        except Exception as e:
            self.logger.error(f"Error analyzing seasonal patterns: {e}")

        return patterns

    async def _analyze_promotional_patterns(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[ShoppingPattern]:
        """Analyze customer response to promotions."""
        patterns = []

        try:
            async with get_db_connection() as conn:
                query = """
                WITH promotion_impact AS (
                    SELECT 
                        CASE 
                            WHEN s.discount_percentage > 0 THEN 'promotion'
                            ELSE 'regular'
                        END as sale_type,
                        ph.first_category_name,
                        AVG(s.units_sold) as avg_units,
                        COUNT(*) as transaction_count,
                        SUM(s.units_sold * 25) as total_revenue
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '90 days'
                """

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY 
                        CASE WHEN s.discount_percentage > 0 THEN 'promotion' ELSE 'regular' END,
                        ph.first_category_name
                    HAVING COUNT(*) >= 10
                )
                SELECT 
                    first_category_name,
                    SUM(CASE WHEN sale_type = 'promotion' THEN avg_units ELSE 0 END) as promo_avg,
                    SUM(CASE WHEN sale_type = 'regular' THEN avg_units ELSE 0 END) as regular_avg,
                    SUM(CASE WHEN sale_type = 'promotion' THEN total_revenue ELSE 0 END) as promo_revenue,
                    SUM(CASE WHEN sale_type = 'regular' THEN total_revenue ELSE 0 END) as regular_revenue,
                    SUM(total_revenue) as category_total_revenue
                FROM promotion_impact
                GROUP BY first_category_name
                HAVING SUM(CASE WHEN sale_type = 'promotion' THEN avg_units ELSE 0 END) > 0
                    AND SUM(CASE WHEN sale_type = 'regular' THEN avg_units ELSE 0 END) > 0
                ORDER BY (SUM(CASE WHEN sale_type = 'promotion' THEN avg_units ELSE 0 END) / 
                         NULLIF(SUM(CASE WHEN sale_type = 'regular' THEN avg_units ELSE 0 END), 0)) DESC
                LIMIT 8
                """

                result = await conn.fetch(query, *params)

                for row in result:
                    if row["regular_avg"] > 0:
                        promotion_lift = row["promo_avg"] / row["regular_avg"]

                        if promotion_lift > 1.2:  # At least 20% lift
                            pattern = ShoppingPattern(
                                pattern_id=f"PROMO-{row['first_category_name'].replace(' ', '')[:10]}",
                                pattern_type="promotional",
                                description=f"Strong promotional response in {row['first_category_name']} ({promotion_lift:.1f}x lift)",
                                frequency=int(
                                    row["promo_revenue"] / 1000
                                ),  # Frequency based on promotion activity
                                confidence_score=min(1.0, (promotion_lift - 1) / 2),
                                affected_customers=int(
                                    row["category_total_revenue"] / 300
                                ),
                                revenue_impact=Decimal(str(row["promo_revenue"])),
                                optimization_opportunity=f"Increase promotion frequency for {row['first_category_name']} - {promotion_lift:.1f}x sales boost potential",
                            )
                            patterns.append(pattern)

        except Exception as e:
            self.logger.error(f"Error analyzing promotional patterns: {e}")

        return patterns

    async def _analyze_day_of_week_patterns(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[ShoppingPattern]:
        """Analyze day-of-week shopping patterns."""
        patterns = []

        try:
            async with get_db_connection() as conn:
                query = """
                WITH dow_analysis AS (
                    SELECT 
                        EXTRACT(DOW FROM s.date) as day_of_week,
                        CASE EXTRACT(DOW FROM s.date)
                            WHEN 0 THEN 'Sunday'
                            WHEN 1 THEN 'Monday'
                            WHEN 2 THEN 'Tuesday'
                            WHEN 3 THEN 'Wednesday'
                            WHEN 4 THEN 'Thursday'
                            WHEN 5 THEN 'Friday'
                            WHEN 6 THEN 'Saturday'
                        END as day_name,
                        AVG(s.units_sold) as avg_sales,
                        SUM(s.units_sold * 25) as total_revenue,
                        COUNT(*) as transaction_count
                    FROM sales_data s
                    WHERE s.date >= NOW() - INTERVAL '60 days'
                """

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY EXTRACT(DOW FROM s.date)
                    ORDER BY day_of_week
                )
                SELECT 
                    *,
                    avg_sales / AVG(avg_sales) OVER() as relative_performance
                FROM dow_analysis
                """

                result = await conn.fetch(query, *params)

                if result:
                    # Find peak and low days
                    peak_day = max(result, key=lambda x: x["avg_sales"])
                    low_day = min(result, key=lambda x: x["avg_sales"])

                    if (
                        peak_day["avg_sales"] / low_day["avg_sales"] > 1.3
                    ):  # Significant variation
                        pattern = ShoppingPattern(
                            pattern_id="DOW-VARIATION",
                            pattern_type="behavioral",
                            description=f"Strong day-of-week pattern: Peak on {peak_day['day_name']}, Low on {low_day['day_name']}",
                            frequency=7,  # Weekly pattern
                            confidence_score=min(
                                1.0,
                                (peak_day["avg_sales"] / low_day["avg_sales"] - 1) / 2,
                            ),
                            affected_customers=sum(
                                row["transaction_count"] for row in result
                            )
                            // 7,
                            revenue_impact=Decimal(
                                str(sum(row["total_revenue"] for row in result))
                            ),
                            optimization_opportunity=f"Optimize staffing and inventory for {peak_day['day_name']} peaks and {low_day['day_name']} lows",
                        )
                        patterns.append(pattern)

        except Exception as e:
            self.logger.error(f"Error analyzing day-of-week patterns: {e}")

        return patterns

    async def _analyze_product_affinity_patterns(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[ShoppingPattern]:
        """Analyze product affinity and cross-purchase patterns."""
        patterns = []

        try:
            # This would typically analyze market basket data
            # For now, we'll simulate based on category relationships
            async with get_db_connection() as conn:
                query = """
                WITH category_cooccurrence AS (
                    SELECT 
                        ph1.first_category_name as category_a,
                        ph2.first_category_name as category_b,
                        COUNT(*) as cooccurrence_count,
                        AVG(s1.units_sold + s2.units_sold) as avg_combined_sales,
                        SUM((s1.units_sold + s2.units_sold) * 25) as combined_revenue
                    FROM sales_data s1
                    JOIN sales_data s2 ON s1.store_id = s2.store_id 
                        AND s1.date = s2.date 
                        AND s1.product_id != s2.product_id
                    JOIN product_hierarchy ph1 ON s1.product_id = ph1.product_id
                    JOIN product_hierarchy ph2 ON s2.product_id = ph2.product_id
                    WHERE s1.date >= NOW() - INTERVAL '30 days'
                        AND ph1.first_category_name != ph2.first_category_name
                """

                params = []
                if store_id:
                    query += " AND s1.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND s1.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY ph1.first_category_name, ph2.first_category_name
                    HAVING COUNT(*) >= 10
                    ORDER BY cooccurrence_count DESC
                    LIMIT 10
                """

                result = await conn.fetch(query, *params)

                for row in result:
                    if row["cooccurrence_count"] > 20:  # Strong affinity
                        pattern = ShoppingPattern(
                            pattern_id=f"AFFINITY-{row['category_a'][:5]}-{row['category_b'][:5]}",
                            pattern_type="behavioral",
                            description=f"Strong affinity between {row['category_a']} and {row['category_b']} categories",
                            frequency=int(row["cooccurrence_count"]),
                            confidence_score=min(1.0, row["cooccurrence_count"] / 100),
                            affected_customers=int(row["cooccurrence_count"] * 2),
                            revenue_impact=Decimal(str(row["combined_revenue"])),
                            optimization_opportunity=f"Cross-merchandise {row['category_a']} with {row['category_b']} for increased basket size",
                        )
                        patterns.append(pattern)

        except Exception as e:
            self.logger.error(f"Error analyzing product affinity patterns: {e}")

        return patterns

    async def _analyze_customer_lifecycle(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze customer lifecycle stages and transitions."""
        try:
            async with get_db_connection() as conn:
                # Simulate customer lifecycle analysis
                query = """
                WITH customer_lifecycle AS (
                    SELECT 
                        COUNT(*) as total_customers,
                        COUNT(CASE WHEN days_since_last <= 7 THEN 1 END) as active_customers,
                        COUNT(CASE WHEN days_since_last BETWEEN 8 AND 30 THEN 1 END) as at_risk_customers,
                        COUNT(CASE WHEN days_since_last > 30 THEN 1 END) as churned_customers,
                        AVG(total_value) as avg_customer_value,
                        AVG(purchase_frequency) as avg_frequency
                    FROM (
                        SELECT 
                            MD5(CONCAT(s.store_id, s.product_id, EXTRACT(DOW FROM s.date)))::uuid as customer_id,
                            EXTRACT(DAYS FROM NOW() - MAX(s.date)) as days_since_last,
                            SUM(s.units_sold * 25) as total_value,
                            COUNT(*) as purchase_frequency
                        FROM sales_data s
                        WHERE s.date >= NOW() - INTERVAL '90 days'
                """

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                        GROUP BY MD5(CONCAT(s.store_id, s.product_id, EXTRACT(DOW FROM s.date)))::uuid
                        HAVING COUNT(*) >= 3
                    ) customer_summary
                )
                SELECT * FROM customer_lifecycle
                """

                result = await conn.fetchrow(query, *params)

                if result:
                    total_customers = result["total_customers"]

                    return {
                        "total_customers": total_customers,
                        "customer_distribution": {
                            "active": {
                                "count": result["active_customers"],
                                "percentage": (
                                    result["active_customers"] / total_customers
                                    if total_customers > 0
                                    else 0
                                ),
                            },
                            "at_risk": {
                                "count": result["at_risk_customers"],
                                "percentage": (
                                    result["at_risk_customers"] / total_customers
                                    if total_customers > 0
                                    else 0
                                ),
                            },
                            "churned": {
                                "count": result["churned_customers"],
                                "percentage": (
                                    result["churned_customers"] / total_customers
                                    if total_customers > 0
                                    else 0
                                ),
                            },
                        },
                        "lifecycle_metrics": {
                            "avg_customer_value": float(
                                result["avg_customer_value"] or 0
                            ),
                            "avg_purchase_frequency": float(
                                result["avg_frequency"] or 0
                            ),
                            "churn_rate": (
                                result["churned_customers"] / total_customers
                                if total_customers > 0
                                else 0
                            ),
                            "retention_rate": (
                                (
                                    result["active_customers"]
                                    + result["at_risk_customers"]
                                )
                                / total_customers
                                if total_customers > 0
                                else 0
                            ),
                        },
                        "recommendations": [
                            f"Focus retention efforts on {result['at_risk_customers']} at-risk customers",
                            f"Implement win-back campaigns for {result['churned_customers']} churned customers",
                            f"Develop loyalty programs for {result['active_customers']} active customers",
                        ],
                    }

                return {}

        except Exception as e:
            self.logger.error(f"Error analyzing customer lifecycle: {e}")
            return {}

    async def _analyze_product_preferences(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze product preferences across customer segments."""
        try:
            async with get_db_connection() as conn:
                query = """
                WITH product_preferences AS (
                    SELECT 
                        ph.first_category_name,
                        ph.second_category_name,
                        ph.product_name,
                        SUM(s.units_sold) as total_sales,
                        SUM(s.units_sold * 25) as total_revenue,
                        COUNT(DISTINCT s.store_id) as store_reach,
                        AVG(s.units_sold) as avg_daily_sales,
                        STDDEV(s.units_sold) as sales_consistency
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '60 days'
                """

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY ph.first_category_name, ph.second_category_name, ph.product_name
                    HAVING COUNT(*) >= 10
                    ORDER BY total_revenue DESC
                    LIMIT 30
                )
                SELECT 
                    first_category_name,
                    COUNT(*) as products_in_category,
                    SUM(total_revenue) as category_revenue,
                    AVG(avg_daily_sales) as category_avg_sales,
                    SUM(total_sales) as category_total_units
                FROM product_preferences
                GROUP BY first_category_name
                ORDER BY category_revenue DESC
                """

                result = await conn.fetch(query, *params)

                return {
                    "top_categories": [
                        {
                            "category_name": row["first_category_name"],
                            "products_count": row["products_in_category"],
                            "total_revenue": float(row["category_revenue"]),
                            "avg_daily_sales": float(row["category_avg_sales"]),
                            "total_units": row["category_total_units"],
                        }
                        for row in result[:10]
                    ],
                    "preference_insights": [
                        (
                            f"Top category: {result[0]['first_category_name']} generates ${result[0]['category_revenue']:,.0f} revenue"
                            if result
                            else "No data available"
                        ),
                        f"Category diversity: {len(result)} active categories",
                        (
                            f"Average category performance: ${sum(row['category_revenue'] for row in result) / len(result):,.0f}"
                            if result
                            else "No data"
                        ),
                    ],
                }

        except Exception as e:
            self.logger.error(f"Error analyzing product preferences: {e}")
            return {}

    async def _generate_customer_insights(
        self,
        segments: List[CustomerSegment],
        patterns: List[ShoppingPattern],
        lifecycle: Dict[str, Any],
        preferences: Dict[str, Any],
    ) -> List[CustomerInsight]:
        """Generate actionable customer insights."""
        insights = []

        try:
            # Insight 1: High-Value Customer Retention
            vip_segments = [
                s
                for s in segments
                if "VIP" in s.segment_name or "Champion" in s.segment_name
            ]
            if vip_segments:
                total_vip_value = sum(
                    float(s.lifetime_value) * s.customer_count for s in vip_segments
                )
                vip_churn_risk = sum(
                    s.churn_probability * s.customer_count for s in vip_segments
                ) / sum(s.customer_count for s in vip_segments)

                insights.append(
                    CustomerInsight(
                        insight_type="retention_opportunity",
                        customer_segment="VIP/Champions",
                        insight_summary=f"${total_vip_value:,.0f} revenue at risk from {vip_churn_risk:.1%} VIP churn rate",
                        data_points={
                            "total_vip_customers": sum(
                                s.customer_count for s in vip_segments
                            ),
                            "total_vip_value": total_vip_value,
                            "average_churn_risk": vip_churn_risk,
                            "potential_loss": total_vip_value * vip_churn_risk,
                        },
                        recommended_actions=[
                            "Implement personalized retention campaigns for VIP customers",
                            "Create exclusive experiences and early access programs",
                            "Establish dedicated customer success manager for top accounts",
                            "Develop predictive churn models for early intervention",
                        ],
                        impact_potential="high",
                        implementation_difficulty="medium",
                    )
                )

            # Insight 2: Cross-Selling Opportunities
            strong_patterns = [
                p
                for p in patterns
                if p.confidence_score > 0.6 and p.pattern_type == "behavioral"
            ]
            if strong_patterns:
                cross_sell_revenue = sum(
                    float(p.revenue_impact) for p in strong_patterns[:3]
                )

                insights.append(
                    CustomerInsight(
                        insight_type="cross_selling",
                        customer_segment="All Segments",
                        insight_summary=f"${cross_sell_revenue:,.0f} cross-selling opportunity from behavioral patterns",
                        data_points={
                            "strong_patterns_identified": len(strong_patterns),
                            "total_cross_sell_potential": cross_sell_revenue,
                            "highest_confidence_pattern": (
                                strong_patterns[0].description
                                if strong_patterns
                                else None
                            ),
                        },
                        recommended_actions=[
                            "Implement automated cross-selling recommendations",
                            "Train staff on product affinity patterns",
                            "Create bundled offers based on purchasing patterns",
                            "Develop personalized product recommendation engine",
                        ],
                        impact_potential="high",
                        implementation_difficulty="low",
                    )
                )

            # Insight 3: Customer Lifecycle Optimization
            if lifecycle.get("lifecycle_metrics"):
                churn_rate = lifecycle["lifecycle_metrics"]["churn_rate"]
                at_risk_count = lifecycle["customer_distribution"]["at_risk"]["count"]

                if churn_rate > 0.2:  # High churn rate
                    insights.append(
                        CustomerInsight(
                            insight_type="churn_prevention",
                            customer_segment="At-Risk Customers",
                            insight_summary=f"{churn_rate:.1%} churn rate with {at_risk_count} customers at immediate risk",
                            data_points={
                                "churn_rate": churn_rate,
                                "at_risk_customers": at_risk_count,
                                "potential_revenue_loss": at_risk_count
                                * lifecycle["lifecycle_metrics"]["avg_customer_value"],
                            },
                            recommended_actions=[
                                "Launch immediate retention campaign for at-risk customers",
                                "Implement proactive customer service outreach",
                                "Offer loyalty incentives and exclusive discounts",
                                "Analyze churn reasons and address root causes",
                            ],
                            impact_potential="medium",
                            implementation_difficulty="low",
                        )
                    )

            # Insight 4: Segment-Specific Opportunities
            for segment in segments[:3]:  # Top 3 segments
                if segment.customer_count > 10:  # Significant segment size
                    segment_revenue_potential = (
                        float(segment.lifetime_value)
                        * segment.customer_count
                        * (1 - segment.churn_probability)
                    )

                    insights.append(
                        CustomerInsight(
                            insight_type="segment_optimization",
                            customer_segment=segment.segment_name,
                            insight_summary=f"{segment.segment_name}: {segment.customer_count} customers with ${segment_revenue_potential:,.0f} protected revenue potential",
                            data_points={
                                "segment_size": segment.customer_count,
                                "avg_lifetime_value": float(segment.lifetime_value),
                                "churn_probability": segment.churn_probability,
                                "preferred_categories": segment.preferred_categories,
                            },
                            recommended_actions=[
                                f"Develop {segment.engagement_strategy} strategy for {segment.segment_name}",
                                f'Focus promotions on {", ".join(segment.preferred_categories[:2])}',
                                f"Customize communication frequency based on segment behavior",
                                f"Create targeted offers matching segment preferences",
                            ],
                            impact_potential="medium",
                            implementation_difficulty="medium",
                        )
                    )

            return insights

        except Exception as e:
            self.logger.error(f"Error generating customer insights: {e}")
            return []

    def _generate_analysis_summary(
        self,
        segments: List[CustomerSegment],
        patterns: List[ShoppingPattern],
        lifecycle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate high-level analysis summary."""
        try:
            total_customers = sum(s.customer_count for s in segments)
            total_revenue_potential = sum(
                float(s.lifetime_value) * s.customer_count for s in segments
            )
            avg_churn_rate = (
                sum(s.churn_probability * s.customer_count for s in segments)
                / total_customers
                if total_customers > 0
                else 0
            )

            return {
                "customer_base_size": total_customers,
                "total_revenue_potential": total_revenue_potential,
                "average_churn_rate": avg_churn_rate,
                "identified_segments": len(segments),
                "behavioral_patterns": len(patterns),
                "top_segment": segments[0].segment_name if segments else "None",
                "strongest_pattern": patterns[0].description if patterns else "None",
                "key_metrics": {
                    "retention_rate": lifecycle.get("lifecycle_metrics", {}).get(
                        "retention_rate", 0
                    ),
                    "avg_customer_value": lifecycle.get("lifecycle_metrics", {}).get(
                        "avg_customer_value", 0
                    ),
                    "active_customer_percentage": lifecycle.get(
                        "customer_distribution", {}
                    )
                    .get("active", {})
                    .get("percentage", 0),
                },
                "recommendations_priority": [
                    "Focus on VIP customer retention",
                    "Implement cross-selling based on behavioral patterns",
                    "Address high churn rate segments",
                    "Develop segment-specific engagement strategies",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error generating analysis summary: {e}")
            return {}

    async def _store_behavior_analysis(
        self,
        store_id: Optional[int],
        city_id: Optional[int],
        segments: List[CustomerSegment],
        patterns: List[ShoppingPattern],
        insights: List[CustomerInsight],
    ) -> None:
        """Store customer behavior analysis results."""
        try:
            async with get_db_connection() as conn:
                # Store main behavior analysis record
                await conn.execute(
                    """
                    INSERT INTO customer_behavior_patterns (
                        store_id, city_id, customer_segment, segment_size,
                        avg_transaction_value, purchase_frequency, lifetime_value,
                        churn_probability, preferred_categories, shopping_patterns,
                        engagement_score, analysis_period_start, analysis_period_end
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    store_id,
                    city_id,
                    "Overall Analysis",
                    sum(s.customer_count for s in segments),
                    (
                        sum(
                            float(s.avg_transaction_value) * s.customer_count
                            for s in segments
                        )
                        / sum(s.customer_count for s in segments)
                        if segments
                        else 0
                    ),
                    (
                        sum(s.avg_frequency * s.customer_count for s in segments)
                        / sum(s.customer_count for s in segments)
                        if segments
                        else 0
                    ),
                    (
                        sum(
                            float(s.lifetime_value) * s.customer_count for s in segments
                        )
                        / sum(s.customer_count for s in segments)
                        if segments
                        else 0
                    ),
                    (
                        sum(s.churn_probability * s.customer_count for s in segments)
                        / sum(s.customer_count for s in segments)
                        if segments
                        else 0
                    ),
                    json.dumps([s.preferred_categories for s in segments[:3]]),
                    json.dumps(
                        {
                            "segments": [
                                {
                                    "name": s.segment_name,
                                    "size": s.customer_count,
                                    "value": float(s.lifetime_value),
                                    "churn_risk": s.churn_probability,
                                }
                                for s in segments
                            ],
                            "patterns": [
                                {
                                    "type": p.pattern_type,
                                    "description": p.description,
                                    "confidence": p.confidence_score,
                                }
                                for p in patterns[:10]
                            ],
                        }
                    ),
                    85.0,  # Overall engagement score
                    datetime.now()
                    - timedelta(days=self.analysis_params["analysis_period_days"]),
                    datetime.now(),
                )

        except Exception as e:
            self.logger.error(f"Error storing behavior analysis: {e}")

    async def get_customer_segment_details(self, segment_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific customer segment."""
        try:
            # This would retrieve detailed segment information from database
            # For now, return simulated detailed analysis

            return {
                "segment_name": segment_name,
                "detailed_characteristics": {
                    "demographic_profile": "Mixed age groups, value-conscious shoppers",
                    "shopping_behavior": "Regular purchasers with predictable patterns",
                    "product_preferences": [
                        "Groceries",
                        "Personal Care",
                        "Home Essentials",
                    ],
                    "channel_preferences": ["In-store shopping", "Mobile app"],
                    "communication_preferences": ["Email offers", "SMS alerts"],
                },
                "engagement_opportunities": [
                    "Loyalty program enrollment",
                    "Personalized product recommendations",
                    "Seasonal promotion targeting",
                    "Cross-category bundling",
                ],
                "risk_factors": [
                    "Price sensitivity to competition",
                    "Limited brand loyalty",
                    "Seasonal purchase variations",
                ],
                "optimization_strategies": [
                    "Implement tiered loyalty rewards",
                    "Develop predictive reorder reminders",
                    "Create personalized shopping experiences",
                    "Optimize product placement and inventory",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting segment details: {e}")
            return {}
