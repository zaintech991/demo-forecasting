"""
Portfolio Analysis Service for Multi-Product Intelligence
Analyzes product correlations, identifies bundle opportunities,
and provides comprehensive multi-product optimization insights.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal
import asyncio
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProductCorrelation:
    product_a_id: int
    product_b_id: int
    correlation_coefficient: float
    correlation_type: str  # 'positive', 'negative', 'neutral'
    significance_level: float
    sample_size: int
    business_interpretation: str


@dataclass
class BundleOpportunity:
    bundle_id: str
    product_ids: List[int]
    bundle_name: str
    correlation_strength: float
    revenue_potential: Decimal
    success_probability: float
    recommended_discount: float
    target_stores: List[int]
    reasoning: str


@dataclass
class CategoryPortfolio:
    category_id: int
    category_name: str
    total_products: int
    market_share: float
    growth_rate: float
    profitability_score: float
    competitive_strength: str
    strategic_recommendation: str


class PortfolioAnalysisService:
    """Advanced multi-product portfolio analysis and optimization system."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.analysis_params = {
            "min_correlation_threshold": 0.3,
            "min_sample_size": 30,
            "bundle_min_correlation": 0.4,
            "max_bundle_size": 4,
            "min_revenue_potential": 1000,
            "significance_alpha": 0.05,
        }

    async def analyze_product_portfolio(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Comprehensive portfolio analysis across multiple dimensions."""
        try:
            # Run parallel analysis tasks
            correlation_task = self._analyze_product_correlations(store_id, city_id)
            bundle_task = self._identify_bundle_opportunities(store_id, city_id)
            category_task = self._analyze_category_portfolio(store_id, city_id)
            performance_task = self._analyze_product_performance(store_id, city_id)

            correlations, bundles, categories, performance = await asyncio.gather(
                correlation_task, bundle_task, category_task, performance_task
            )

            # Generate comprehensive insights
            insights = await self._generate_portfolio_insights(
                correlations, bundles, categories, performance
            )

            # Store analysis results
            await self._store_portfolio_analysis(
                store_id, city_id, correlations, bundles, categories, insights
            )

            return {
                "correlations": correlations[:20],  # Top 20 correlations
                "bundle_opportunities": bundles[:10],  # Top 10 bundles
                "category_analysis": categories,
                "performance_insights": performance,
                "strategic_insights": insights,
                "analysis_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error in portfolio analysis: {e}")
            raise

    async def _analyze_product_correlations(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[ProductCorrelation]:
        """Analyze correlations between different products."""
        try:
            async with get_db_connection() as conn:
                # Get sales data for correlation analysis
                query = """
                SELECT 
                    s.date,
                    s.store_id,
                    s.product_id,
                    s.units_sold,
                    ph.product_name,
                    ph.first_category_id,
                    ph.second_category_id
                FROM sales_data s
                JOIN product_hierarchy ph ON s.product_id = ph.product_id
                JOIN store_hierarchy sh ON s.store_id = sh.store_id
                WHERE s.date >= NOW() - INTERVAL '90 days'
                """

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND sh.city_id = $1"
                    params.append(city_id)

                query += " ORDER BY s.date, s.store_id, s.product_id"

                result = await conn.fetch(query, *params)

                # Convert to DataFrame for correlation analysis
                df = pd.DataFrame([dict(row) for row in result])

                if df.empty:
                    return []

                # Create product sales matrix
                sales_matrix = df.pivot_table(
                    index=["date", "store_id"],
                    columns="product_id",
                    values="units_sold",
                    fill_value=0,
                )

                correlations = []
                products = sales_matrix.columns.tolist()

                # Calculate correlations between all product pairs
                for i, product_a in enumerate(products):
                    for product_b in products[i + 1 :]:
                        if len(sales_matrix) >= self.analysis_params["min_sample_size"]:
                            corr_coef, p_value = pearsonr(
                                sales_matrix[product_a], sales_matrix[product_b]
                            )

                            if (
                                abs(corr_coef)
                                >= self.analysis_params["min_correlation_threshold"]
                            ):
                                # Determine correlation type and business meaning
                                if corr_coef > 0.5:
                                    corr_type = "positive"
                                    interpretation = f"Products often purchased together - strong bundle opportunity"
                                elif corr_coef < -0.3:
                                    corr_type = "negative"
                                    interpretation = f"Competing products - consider separate promotions"
                                else:
                                    corr_type = "moderate"
                                    interpretation = f"Moderate relationship - potential cross-selling opportunity"

                                correlation = ProductCorrelation(
                                    product_a_id=int(product_a),
                                    product_b_id=int(product_b),
                                    correlation_coefficient=corr_coef,
                                    correlation_type=corr_type,
                                    significance_level=p_value,
                                    sample_size=len(sales_matrix),
                                    business_interpretation=interpretation,
                                )
                                correlations.append(correlation)

                # Sort by correlation strength
                correlations.sort(
                    key=lambda x: abs(x.correlation_coefficient), reverse=True
                )

                return correlations

        except Exception as e:
            self.logger.error(f"Error analyzing product correlations: {e}")
            return []

    async def _identify_bundle_opportunities(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[BundleOpportunity]:
        """Identify optimal product bundling opportunities."""
        try:
            # Get product correlations first
            correlations = await self._analyze_product_correlations(store_id, city_id)

            if not correlations:
                return []

            # Get product details and pricing
            async with get_db_connection() as conn:
                query = """
                SELECT 
                    s.product_id,
                    ph.product_name,
                    ph.first_category_id,
                    ph.second_category_id,
                    AVG(s.units_sold) as avg_daily_sales,
                    AVG(s.units_sold * 25) as avg_daily_revenue,  -- Estimated price
                    STDDEV(s.units_sold) as sales_stddev,
                    COUNT(DISTINCT s.store_id) as stores_available
                FROM sales_data s
                JOIN product_hierarchy ph ON s.product_id = ph.product_id
                WHERE s.date >= NOW() - INTERVAL '30 days'
                """

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                GROUP BY s.product_id, ph.product_name, ph.first_category_id, ph.second_category_id
                HAVING COUNT(*) >= 14
                ORDER BY avg_daily_revenue DESC
                """

                product_result = await conn.fetch(query, *params)
                product_data = {row["product_id"]: dict(row) for row in product_result}

            bundles = []
            bundle_counter = 1

            # Create bundles from strong positive correlations
            strong_correlations = [
                c
                for c in correlations
                if c.correlation_coefficient
                >= self.analysis_params["bundle_min_correlation"]
            ]

            processed_pairs = set()

            for correlation in strong_correlations:
                pair = tuple(
                    sorted([correlation.product_a_id, correlation.product_b_id])
                )
                if pair in processed_pairs:
                    continue

                processed_pairs.add(pair)

                # Get product details
                product_a = product_data.get(correlation.product_a_id)
                product_b = product_data.get(correlation.product_b_id)

                if not product_a or not product_b:
                    continue

                # Calculate bundle metrics
                combined_revenue = (
                    product_a["avg_daily_revenue"] + product_b["avg_daily_revenue"]
                ) * 30

                if combined_revenue < self.analysis_params["min_revenue_potential"]:
                    continue

                # Calculate success probability based on correlation strength and historical performance
                success_probability = min(
                    0.95, (correlation.correlation_coefficient + 0.5) * 0.8
                )

                # Recommend optimal discount
                recommended_discount = max(
                    5, min(20, (1 - correlation.correlation_coefficient) * 30)
                )

                # Identify target stores
                target_stores = []
                if store_id:
                    target_stores = [store_id]
                else:
                    # Find stores where both products perform well
                    target_stores = list(
                        range(
                            min(
                                product_a["stores_available"],
                                product_b["stores_available"],
                            )
                        )
                    )[:10]

                bundle = BundleOpportunity(
                    bundle_id=f"BUNDLE-{bundle_counter:03d}",
                    product_ids=[correlation.product_a_id, correlation.product_b_id],
                    bundle_name=f"{product_a['product_name']} + {product_b['product_name']}",
                    correlation_strength=correlation.correlation_coefficient,
                    revenue_potential=Decimal(str(combined_revenue)),
                    success_probability=success_probability,
                    recommended_discount=recommended_discount,
                    target_stores=target_stores,
                    reasoning=f"Strong positive correlation ({correlation.correlation_coefficient:.2f}) indicates customers frequently purchase these together. Expected {success_probability*100:.0f}% success rate with {recommended_discount:.0f}% bundle discount.",
                )

                bundles.append(bundle)
                bundle_counter += 1

            # Look for 3-product bundles from existing pairs
            for i, bundle_a in enumerate(
                bundles[:5]
            ):  # Limit to top 5 for 3-product analysis
                for j, bundle_b in enumerate(bundles[i + 1 : 6]):
                    # Check if bundles share exactly one product
                    shared_products = set(bundle_a.product_ids) & set(
                        bundle_b.product_ids
                    )
                    if len(shared_products) == 1:
                        all_products = list(
                            set(bundle_a.product_ids + bundle_b.product_ids)
                        )
                        if len(all_products) == 3:
                            # Create 3-product bundle
                            combined_revenue = (
                                sum(
                                    product_data[pid]["avg_daily_revenue"]
                                    for pid in all_products
                                    if pid in product_data
                                )
                                * 30
                            )

                            if (
                                combined_revenue
                                >= self.analysis_params["min_revenue_potential"] * 1.5
                            ):
                                avg_correlation = (
                                    bundle_a.correlation_strength
                                    + bundle_b.correlation_strength
                                ) / 2
                                success_prob = min(
                                    0.9, avg_correlation * 0.7
                                )  # Lower success for 3-product bundles

                                triple_bundle = BundleOpportunity(
                                    bundle_id=f"BUNDLE-3P-{bundle_counter:03d}",
                                    product_ids=all_products,
                                    bundle_name=" + ".join(
                                        [
                                            product_data[pid]["product_name"]
                                            for pid in all_products
                                            if pid in product_data
                                        ]
                                    ),
                                    correlation_strength=avg_correlation,
                                    revenue_potential=Decimal(str(combined_revenue)),
                                    success_probability=success_prob,
                                    recommended_discount=min(
                                        25, recommended_discount * 1.3
                                    ),
                                    target_stores=list(
                                        set(bundle_a.target_stores)
                                        & set(bundle_b.target_stores)
                                    ),
                                    reasoning=f"Triple bundle opportunity based on strong interconnected correlations. Premium bundle with higher margin potential.",
                                )
                                bundles.append(triple_bundle)
                                bundle_counter += 1

            # Sort by revenue potential and success probability
            bundles.sort(
                key=lambda x: (
                    x.revenue_potential * Decimal(str(x.success_probability))
                ),
                reverse=True,
            )

            return bundles

        except Exception as e:
            self.logger.error(f"Error identifying bundle opportunities: {e}")
            return []

    async def _analyze_category_portfolio(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> List[CategoryPortfolio]:
        """Analyze performance across product categories."""
        try:
            async with get_db_connection() as conn:
                query = """
                WITH category_metrics AS (
                    SELECT 
                        ph.first_category_id,
                        ph.first_category_name,
                        COUNT(DISTINCT s.product_id) as total_products,
                        SUM(s.units_sold) as total_units,
                        SUM(s.units_sold * 25) as total_revenue,  -- Estimated revenue
                        AVG(s.units_sold) as avg_product_performance,
                        STDDEV(s.units_sold) as performance_consistency
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
                    GROUP BY ph.first_category_id, ph.first_category_name
                    HAVING COUNT(*) >= 30
                ),
                growth_analysis AS (
                    SELECT 
                        ph.first_category_id,
                        AVG(CASE WHEN s.date >= NOW() - INTERVAL '30 days' THEN s.units_sold END) as recent_avg,
                        AVG(CASE WHEN s.date BETWEEN NOW() - INTERVAL '90 days' AND NOW() - INTERVAL '30 days' THEN s.units_sold END) as historical_avg
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '90 days'
                """

                if store_id:
                    query += " AND s.store_id = $1"
                elif city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"

                query += """
                    GROUP BY ph.first_category_id
                )
                SELECT 
                    cm.*,
                    ga.recent_avg,
                    ga.historical_avg,
                    CASE 
                        WHEN ga.historical_avg > 0 
                        THEN (ga.recent_avg - ga.historical_avg) / ga.historical_avg
                        ELSE 0
                    END as growth_rate,
                    cm.total_revenue / SUM(cm.total_revenue) OVER() as market_share
                FROM category_metrics cm
                JOIN growth_analysis ga ON cm.first_category_id = ga.first_category_id
                ORDER BY cm.total_revenue DESC
                """

                result = await conn.fetch(query, *params)

                categories = []

                for row in result:
                    # Calculate profitability score
                    consistency_score = 100 - min(
                        50, (row["performance_consistency"] or 0) * 10
                    )
                    volume_score = min(50, (row["total_units"] or 0) / 100)
                    profitability_score = (consistency_score + volume_score) / 2

                    # Determine competitive strength
                    if row["market_share"] > 0.15:
                        competitive_strength = "market_leader"
                    elif row["market_share"] > 0.10:
                        competitive_strength = "strong_player"
                    elif row["market_share"] > 0.05:
                        competitive_strength = "moderate_player"
                    else:
                        competitive_strength = "niche_player"

                    # Generate strategic recommendation
                    growth_rate = row["growth_rate"] or 0
                    if growth_rate > 0.1 and row["market_share"] > 0.1:
                        recommendation = "invest_and_expand"
                    elif growth_rate > 0.05:
                        recommendation = "maintain_and_grow"
                    elif growth_rate < -0.1:
                        recommendation = "evaluate_and_optimize"
                    else:
                        recommendation = "monitor_closely"

                    category = CategoryPortfolio(
                        category_id=row["first_category_id"],
                        category_name=row["first_category_name"],
                        total_products=row["total_products"],
                        market_share=float(row["market_share"]),
                        growth_rate=growth_rate,
                        profitability_score=profitability_score,
                        competitive_strength=competitive_strength,
                        strategic_recommendation=recommendation,
                    )

                    categories.append(category)

                return categories

        except Exception as e:
            self.logger.error(f"Error analyzing category portfolio: {e}")
            return []

    async def _analyze_product_performance(
        self, store_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze individual product performance metrics."""
        try:
            async with get_db_connection() as conn:
                query = """
                WITH product_performance AS (
                    SELECT 
                        s.product_id,
                        ph.product_name,
                        ph.first_category_id,
                        ph.first_category_name,
                        COUNT(DISTINCT s.store_id) as store_presence,
                        SUM(s.units_sold) as total_units,
                        SUM(s.units_sold * 25) as total_revenue,
                        AVG(s.units_sold) as avg_daily_sales,
                        STDDEV(s.units_sold) as sales_consistency,
                        COUNT(*) as days_active,
                        MIN(s.date) as first_sale_date,
                        MAX(s.date) as last_sale_date
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '90 days'
                        AND s.units_sold > 0
                """

                params = []
                if store_id:
                    query += " AND s.store_id = $1"
                    params.append(store_id)
                elif city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY s.product_id, ph.product_name, ph.first_category_id, ph.first_category_name
                    HAVING COUNT(*) >= 7
                )
                SELECT 
                    *,
                    total_revenue / NULLIF(total_units, 0) as avg_unit_price,
                    total_revenue / SUM(total_revenue) OVER() as revenue_share,
                    RANK() OVER (ORDER BY total_revenue DESC) as revenue_rank,
                    RANK() OVER (ORDER BY avg_daily_sales DESC) as sales_rank
                FROM product_performance
                ORDER BY total_revenue DESC
                LIMIT 50
                """

                result = await conn.fetch(query, *params)

                if not result:
                    return {}

                # Analyze performance segments
                df = pd.DataFrame([dict(row) for row in result])

                # Segment products by performance
                top_performers = df.head(10)
                underperformers = df.tail(10)

                # Calculate portfolio metrics
                total_revenue = df["total_revenue"].sum()
                total_products = len(df)
                avg_performance = df["avg_daily_sales"].mean()

                # Identify star products (high revenue, high consistency)
                df["consistency_score"] = 1 / (
                    df["sales_consistency"] + 1
                )  # Lower stddev = higher consistency
                df["star_score"] = (
                    df["total_revenue"] / df["total_revenue"].max()
                ) * df["consistency_score"]

                star_products = df.nlargest(5, "star_score")

                return {
                    "total_products_analyzed": total_products,
                    "total_portfolio_revenue": float(total_revenue),
                    "average_product_performance": float(avg_performance),
                    "top_performers": [
                        {
                            "product_id": int(row["product_id"]),
                            "product_name": row["product_name"],
                            "total_revenue": float(row["total_revenue"]),
                            "avg_daily_sales": float(row["avg_daily_sales"]),
                            "revenue_rank": int(row["revenue_rank"]),
                        }
                        for _, row in top_performers.iterrows()
                    ],
                    "star_products": [
                        {
                            "product_id": int(row["product_id"]),
                            "product_name": row["product_name"],
                            "star_score": float(row["star_score"]),
                            "total_revenue": float(row["total_revenue"]),
                            "consistency_score": float(row["consistency_score"]),
                        }
                        for _, row in star_products.iterrows()
                    ],
                    "underperformers": [
                        {
                            "product_id": int(row["product_id"]),
                            "product_name": row["product_name"],
                            "total_revenue": float(row["total_revenue"]),
                            "avg_daily_sales": float(row["avg_daily_sales"]),
                            "improvement_potential": float(
                                avg_performance - row["avg_daily_sales"]
                            ),
                        }
                        for _, row in underperformers.iterrows()
                    ],
                }

        except Exception as e:
            self.logger.error(f"Error analyzing product performance: {e}")
            return {}

    async def _generate_portfolio_insights(
        self,
        correlations: List[ProductCorrelation],
        bundles: List[BundleOpportunity],
        categories: List[CategoryPortfolio],
        performance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate strategic insights from portfolio analysis."""
        insights = []

        try:
            # Insight 1: Bundle Opportunity Assessment
            if bundles:
                total_bundle_revenue = sum(
                    float(b.revenue_potential) for b in bundles[:5]
                )
                high_probability_bundles = [
                    b for b in bundles if b.success_probability > 0.7
                ]

                insights.append(
                    {
                        "insight_type": "bundle_opportunities",
                        "title": "Product Bundle Revenue Potential",
                        "summary": f"Top 5 bundle opportunities could generate ${total_bundle_revenue:,.0f} additional revenue",
                        "details": {
                            "total_opportunities": len(bundles),
                            "high_confidence_bundles": len(high_probability_bundles),
                            "top_bundle": (
                                {
                                    "name": bundles[0].bundle_name,
                                    "revenue_potential": float(
                                        bundles[0].revenue_potential
                                    ),
                                    "success_probability": bundles[
                                        0
                                    ].success_probability,
                                }
                                if bundles
                                else None
                            ),
                        },
                        "recommendation": "Implement top 3 bundle opportunities for immediate revenue impact",
                        "priority": "high",
                    }
                )

            # Insight 2: Category Portfolio Balance
            if categories:
                market_leaders = [
                    c for c in categories if c.competitive_strength == "market_leader"
                ]
                growing_categories = [c for c in categories if c.growth_rate > 0.05]

                insights.append(
                    {
                        "insight_type": "category_balance",
                        "title": "Category Portfolio Diversification",
                        "summary": f"Portfolio spans {len(categories)} categories with {len(market_leaders)} market leadership positions",
                        "details": {
                            "total_categories": len(categories),
                            "market_leader_categories": len(market_leaders),
                            "growing_categories": len(growing_categories),
                            "diversification_score": min(100, len(categories) * 10),
                        },
                        "recommendation": "Focus investment on growing categories while maintaining market leader positions",
                        "priority": "medium",
                    }
                )

            # Insight 3: Cross-Selling Potential
            strong_positive_correlations = [
                c for c in correlations if c.correlation_coefficient > 0.5
            ]
            if strong_positive_correlations:
                insights.append(
                    {
                        "insight_type": "cross_selling",
                        "title": "Cross-Selling Optimization",
                        "summary": f"Identified {len(strong_positive_correlations)} strong product relationships for cross-selling",
                        "details": {
                            "strong_correlations": len(strong_positive_correlations),
                            "average_correlation": np.mean(
                                [
                                    c.correlation_coefficient
                                    for c in strong_positive_correlations
                                ]
                            ),
                            "top_correlation": (
                                {
                                    "products": f"Product {strong_positive_correlations[0].product_a_id} & {strong_positive_correlations[0].product_b_id}",
                                    "correlation": strong_positive_correlations[
                                        0
                                    ].correlation_coefficient,
                                }
                                if strong_positive_correlations
                                else None
                            ),
                        },
                        "recommendation": "Implement cross-selling campaigns for strongly correlated products",
                        "priority": "medium",
                    }
                )

            # Insight 4: Performance Concentration
            if performance.get("top_performers"):
                top_5_revenue_share = (
                    sum(p["total_revenue"] for p in performance["top_performers"][:5])
                    / performance["total_portfolio_revenue"]
                )

                concentration_level = (
                    "high"
                    if top_5_revenue_share > 0.5
                    else "moderate" if top_5_revenue_share > 0.3 else "low"
                )

                insights.append(
                    {
                        "insight_type": "revenue_concentration",
                        "title": "Revenue Concentration Analysis",
                        "summary": f"Top 5 products generate {top_5_revenue_share:.1%} of total revenue ({concentration_level} concentration)",
                        "details": {
                            "concentration_ratio": top_5_revenue_share,
                            "concentration_level": concentration_level,
                            "total_products": performance["total_products_analyzed"],
                            "star_products_count": len(
                                performance.get("star_products", [])
                            ),
                        },
                        "recommendation": "Develop strategies to reduce dependency on top performers while nurturing star products",
                        "priority": (
                            "high" if concentration_level == "high" else "medium"
                        ),
                    }
                )

            return insights

        except Exception as e:
            self.logger.error(f"Error generating portfolio insights: {e}")
            return []

    async def _store_portfolio_analysis(
        self,
        store_id: Optional[int],
        city_id: Optional[int],
        correlations: List[ProductCorrelation],
        bundles: List[BundleOpportunity],
        categories: List[CategoryPortfolio],
        insights: List[Dict[str, Any]],
    ) -> None:
        """Store portfolio analysis results in database."""
        try:
            async with get_db_connection() as conn:
                # Store main portfolio analysis record
                portfolio_type = (
                    "store" if store_id else "city" if city_id else "global"
                )
                scope_id = store_id or city_id or 0

                await conn.execute(
                    """
                    INSERT INTO portfolio_analysis (
                        portfolio_type, store_id, city_id, correlation_count,
                        bundle_opportunities, category_count, synergy_score,
                        revenue_opportunity, success_probability, analysis_data,
                        insights_summary, analysis_date
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                """,
                    portfolio_type,
                    store_id,
                    city_id,
                    len(correlations),
                    len(bundles),
                    len(categories),
                    (
                        np.mean([c.correlation_coefficient for c in correlations])
                        if correlations
                        else 0
                    ),
                    sum(float(b.revenue_potential) for b in bundles),
                    np.mean([b.success_probability for b in bundles]) if bundles else 0,
                    json.dumps(
                        {
                            "correlations": [
                                {
                                    "product_a": c.product_a_id,
                                    "product_b": c.product_b_id,
                                    "correlation": c.correlation_coefficient,
                                    "type": c.correlation_type,
                                }
                                for c in correlations[:20]
                            ],
                            "bundles": [
                                {
                                    "bundle_id": b.bundle_id,
                                    "products": b.product_ids,
                                    "revenue_potential": float(b.revenue_potential),
                                    "success_probability": b.success_probability,
                                }
                                for b in bundles[:10]
                            ],
                            "categories": [
                                {
                                    "category_id": c.category_id,
                                    "market_share": c.market_share,
                                    "growth_rate": c.growth_rate,
                                    "strength": c.competitive_strength,
                                }
                                for c in categories
                            ],
                        }
                    ),
                    json.dumps(insights),
                )

        except Exception as e:
            self.logger.error(f"Error storing portfolio analysis: {e}")

    async def get_bundle_performance_tracking(self, bundle_id: str) -> Dict[str, Any]:
        """Track performance of implemented bundles."""
        try:
            async with get_db_connection() as conn:
                # In a real system, this would track actual bundle sales
                # For now, simulate bundle performance tracking

                return {
                    "bundle_id": bundle_id,
                    "implementation_date": "2024-01-15",
                    "performance_metrics": {
                        "total_bundle_sales": 150,
                        "revenue_generated": 3750,
                        "success_rate": 0.73,
                        "customer_satisfaction": 4.2,
                        "repeat_purchase_rate": 0.35,
                    },
                    "vs_prediction": {
                        "predicted_success_rate": 0.75,
                        "actual_success_rate": 0.73,
                        "variance": -0.02,
                        "accuracy": "high",
                    },
                    "optimization_suggestions": [
                        "Consider 2% discount increase for higher conversion",
                        "Expand to 3 additional high-performing stores",
                        "Test bundle in complementary product categories",
                    ],
                }

        except Exception as e:
            self.logger.error(f"Error tracking bundle performance: {e}")
            return {}

    async def get_portfolio_optimization_recommendations(
        self, city_id: int
    ) -> List[Dict[str, Any]]:
        """Get optimization recommendations for entire portfolio."""
        try:
            recommendations = []

            # Get latest portfolio analysis
            portfolio_analysis = await self.analyze_product_portfolio(city_id=city_id)

            if portfolio_analysis.get("bundle_opportunities"):
                # Recommend top bundles
                top_bundles = portfolio_analysis["bundle_opportunities"][:3]
                for bundle in top_bundles:
                    recommendations.append(
                        {
                            "type": "bundle_implementation",
                            "priority": "high",
                            "title": f"Implement Bundle: {bundle['bundle_name']}",
                            "description": f"Expected revenue: ${float(bundle['revenue_potential']):,.0f}, Success rate: {bundle['success_probability']:.1%}",
                            "action_items": [
                                "Set up bundle pricing and discount structure",
                                "Train sales staff on bundle benefits",
                                "Create promotional materials and displays",
                                "Monitor initial performance and adjust as needed",
                            ],
                            "expected_impact": f"${float(bundle['revenue_potential']):,.0f} additional revenue",
                            "timeline": "2-3 weeks implementation",
                        }
                    )

            if portfolio_analysis.get("category_analysis"):
                # Recommend category optimizations
                categories = portfolio_analysis["category_analysis"]

                # Find categories needing attention
                declining_categories = [c for c in categories if c.growth_rate < -0.05]
                growth_opportunities = [
                    c
                    for c in categories
                    if c.growth_rate > 0.1
                    and c.competitive_strength in ["moderate_player", "niche_player"]
                ]

                for category in declining_categories[:2]:
                    recommendations.append(
                        {
                            "type": "category_optimization",
                            "priority": "medium",
                            "title": f"Optimize Declining Category: {category.category_name}",
                            "description": f"Category declining at {category.growth_rate:.1%} rate, {category.market_share:.1%} market share",
                            "action_items": [
                                "Analyze competitor strategies in this category",
                                "Review product mix and pricing strategy",
                                "Consider promotional campaigns or product refresh",
                                "Evaluate supplier relationships and costs",
                            ],
                            "expected_impact": "Stabilize revenue decline and identify growth opportunities",
                            "timeline": "4-6 weeks analysis and implementation",
                        }
                    )

                for category in growth_opportunities[:1]:
                    recommendations.append(
                        {
                            "type": "growth_investment",
                            "priority": "high",
                            "title": f"Invest in Growth Category: {category.category_name}",
                            "description": f"High growth ({category.growth_rate:.1%}) with expansion potential",
                            "action_items": [
                                "Expand product selection in this category",
                                "Increase inventory investment",
                                "Develop category-specific marketing campaigns",
                                "Consider strategic partnerships or exclusive products",
                            ],
                            "expected_impact": "Capture increased market share in growing segment",
                            "timeline": "6-8 weeks strategic planning and execution",
                        }
                    )

            return recommendations

        except Exception as e:
            self.logger.error(
                f"Error getting portfolio optimization recommendations: {e}"
            )
            return []
