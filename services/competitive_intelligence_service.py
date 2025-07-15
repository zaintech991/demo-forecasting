"""
Competitive Intelligence Service for Market Analysis
Analyzes market competition, pricing strategies, threat assessment,
and provides strategic recommendations for competitive advantage.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal
import asyncio
from dataclasses import dataclass
import json

from database.connection import get_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CompetitorProfile:
    competitor_name: str
    market_share_estimate: float
    pricing_strategy: str  # 'premium', 'competitive', 'discount', 'value'
    threat_level: str  # 'low', 'medium', 'high', 'critical'
    competitive_advantages: List[str]
    weaknesses: List[str]
    strategic_focus: List[str]


@dataclass
class MarketThreat:
    threat_id: str
    threat_type: str  # 'pricing', 'product', 'service', 'expansion'
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_categories: List[str]
    estimated_impact: Decimal
    response_urgency: int  # 1-5 scale
    recommended_response: str
    threat_timeline: str


@dataclass
class CompetitiveOpportunity:
    opportunity_id: str
    opportunity_type: str
    market_gap: str
    revenue_potential: Decimal
    implementation_difficulty: str
    success_probability: float
    strategic_importance: str
    action_plan: List[str]


class CompetitiveIntelligenceService:
    """Advanced competitive intelligence and market analysis system."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.analysis_params = {
            "market_analysis_period_days": 90,
            "price_sensitivity_threshold": 0.05,
            "threat_assessment_window": 30,
            "min_market_share_concern": 0.02,
            "competitive_response_time_hours": 24,
        }

    async def analyze_competitive_landscape(
        self, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Comprehensive competitive landscape analysis."""
        try:
            # Run parallel competitive analysis tasks
            competitor_analysis_task = self._analyze_competitor_profiles(city_id)
            threat_assessment_task = self._assess_market_threats(city_id)
            opportunity_task = self._identify_competitive_opportunities(city_id)
            pricing_task = self._analyze_competitive_pricing(city_id)

            competitors, threats, opportunities, pricing_analysis = (
                await asyncio.gather(
                    competitor_analysis_task,
                    threat_assessment_task,
                    opportunity_task,
                    pricing_task,
                )
            )

            # Generate strategic recommendations
            strategic_recommendations = await self._generate_strategic_recommendations(
                competitors, threats, opportunities, pricing_analysis
            )

            # Store analysis results
            await self._store_competitive_analysis(
                city_id, competitors, threats, opportunities, strategic_recommendations
            )

            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "market_scope": f"City {city_id}" if city_id else "All Markets",
                "competitor_profiles": competitors,
                "threat_assessment": {
                    "total_threats": len(threats),
                    "critical_threats": len(
                        [t for t in threats if t.severity == "critical"]
                    ),
                    "high_priority_threats": len(
                        [t for t in threats if t.severity == "high"]
                    ),
                    "threats": threats,
                },
                "competitive_opportunities": {
                    "total_opportunities": len(opportunities),
                    "high_value_opportunities": len(
                        [o for o in opportunities if float(o.revenue_potential) > 10000]
                    ),
                    "strategic_opportunities": len(
                        [o for o in opportunities if o.strategic_importance == "high"]
                    ),
                    "opportunities": opportunities,
                },
                "pricing_intelligence": pricing_analysis,
                "strategic_recommendations": strategic_recommendations,
                "competitive_strength_score": self._calculate_competitive_strength(
                    competitors, threats, opportunities
                ),
            }

        except Exception as e:
            self.logger.error(f"Error in competitive landscape analysis: {e}")
            raise

    async def _analyze_competitor_profiles(
        self, city_id: Optional[int] = None
    ) -> List[CompetitorProfile]:
        """Analyze competitor profiles and market positioning."""
        try:
            # Since we don't have actual competitor data, we'll simulate based on market patterns
            async with get_db_connection() as conn:
                # Analyze market patterns to infer competitive landscape
                query = (
                    """
                WITH market_analysis AS (
                    SELECT 
                        ph.first_category_name,
                        AVG(s.units_sold) as avg_sales,
                        STDDEV(s.units_sold) as sales_variance,
                        COUNT(DISTINCT s.store_id) as store_coverage,
                        COUNT(DISTINCT s.product_id) as product_variety,
                        SUM(s.units_sold * 25) as estimated_revenue  -- Estimated pricing
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '%s days'
                """
                    % self.analysis_params["market_analysis_period_days"]
                )

                params = []
                if city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY ph.first_category_name
                    HAVING COUNT(*) >= 30
                    ORDER BY estimated_revenue DESC
                )
                SELECT 
                    first_category_name,
                    avg_sales,
                    sales_variance,
                    store_coverage,
                    product_variety,
                    estimated_revenue,
                    RANK() OVER (ORDER BY estimated_revenue DESC) as revenue_rank
                FROM market_analysis
                LIMIT 10
                """

                result = await conn.fetch(query, *params)

                # Generate competitor profiles based on market analysis
                competitors = []

                # Simulate major competitors based on category performance
                competitor_templates = [
                    {
                        "name": "Premium Market Leader",
                        "strategy": "premium",
                        "advantages": [
                            "Brand Recognition",
                            "Quality Products",
                            "Customer Loyalty",
                        ],
                        "weaknesses": ["Higher Prices", "Limited Value Options"],
                        "focus": ["Premium Segments", "Quality Leadership"],
                    },
                    {
                        "name": "Value Competitor",
                        "strategy": "discount",
                        "advantages": ["Low Prices", "Cost Efficiency", "Volume Sales"],
                        "weaknesses": ["Lower Margins", "Quality Perception"],
                        "focus": ["Price-Sensitive Customers", "Volume Growth"],
                    },
                    {
                        "name": "Category Specialist",
                        "strategy": "competitive",
                        "advantages": [
                            "Category Expertise",
                            "Product Innovation",
                            "Supplier Relationships",
                        ],
                        "weaknesses": ["Limited Scope", "Dependency Risk"],
                        "focus": ["Category Dominance", "Product Development"],
                    },
                    {
                        "name": "Regional Chain",
                        "strategy": "value",
                        "advantages": [
                            "Local Knowledge",
                            "Community Presence",
                            "Flexibility",
                        ],
                        "weaknesses": ["Limited Resources", "Scale Disadvantage"],
                        "focus": ["Local Market", "Customer Service"],
                    },
                ]

                total_market_revenue = sum(row["estimated_revenue"] for row in result)

                for i, template in enumerate(competitor_templates):
                    # Calculate market share based on category performance
                    if i < len(result):
                        category_revenue = result[i]["estimated_revenue"]
                        market_share = (
                            category_revenue / total_market_revenue
                        ) * 0.7  # Assume we have 70% visibility
                    else:
                        market_share = 0.05 + (
                            0.03 * (4 - i)
                        )  # Decreasing market share

                    # Determine threat level
                    if market_share > 0.15:
                        threat_level = "critical"
                    elif market_share > 0.10:
                        threat_level = "high"
                    elif market_share > 0.05:
                        threat_level = "medium"
                    else:
                        threat_level = "low"

                    competitor = CompetitorProfile(
                        competitor_name=str(template["name"]),
                        market_share_estimate=market_share,
                        pricing_strategy=str(template["strategy"]),
                        threat_level=threat_level,
                        competitive_advantages=list(template["advantages"]),
                        weaknesses=list(template["weaknesses"]),
                        strategic_focus=list(template["focus"]),
                    )
                    competitors.append(competitor)

                return competitors

        except Exception as e:
            self.logger.error(f"Error analyzing competitor profiles: {e}")
            return []

    async def _assess_market_threats(
        self, city_id: Optional[int] = None
    ) -> List[MarketThreat]:
        """Assess current and emerging market threats."""
        try:
            threats = []

            async with get_db_connection() as conn:
                # Analyze sales trends to identify potential threats
                query = """
                WITH trend_analysis AS (
                    SELECT 
                        ph.first_category_name,
                        AVG(CASE WHEN s.date >= NOW() - INTERVAL '14 days' THEN s.units_sold END) as recent_avg,
                        AVG(CASE WHEN s.date BETWEEN NOW() - INTERVAL '45 days' AND NOW() - INTERVAL '14 days' THEN s.units_sold END) as baseline_avg,
                        SUM(s.units_sold * 25) as category_revenue,
                        COUNT(DISTINCT s.store_id) as store_presence
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '45 days'
                """

                params = []
                if city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY ph.first_category_name
                    HAVING COUNT(*) >= 20
                ),
                declining_categories AS (
                    SELECT 
                        *,
                        CASE 
                            WHEN baseline_avg > 0 
                            THEN (recent_avg - baseline_avg) / baseline_avg
                            ELSE 0
                        END as trend_change
                    FROM trend_analysis
                    WHERE baseline_avg > 0
                )
                SELECT *
                FROM declining_categories
                WHERE trend_change < -0.10  -- 10% decline indicates potential threat
                ORDER BY trend_change ASC
                LIMIT 8
                """

                result = await conn.fetch(query, *params)

                # Generate threats based on declining trends
                for i, row in enumerate(result):
                    decline_rate = abs(row["trend_change"])

                    # Determine severity
                    if decline_rate > 0.25:
                        severity = "critical"
                    elif decline_rate > 0.15:
                        severity = "high"
                    else:
                        severity = "medium"

                    # Calculate estimated impact
                    estimated_impact = Decimal(
                        str(row["category_revenue"] * decline_rate)
                    )

                    threat = MarketThreat(
                        threat_id=f"THREAT-{i+1:03d}",
                        threat_type="competitive_pressure",
                        severity=severity,
                        affected_categories=[row["first_category_name"]],
                        estimated_impact=estimated_impact,
                        response_urgency=(
                            5
                            if severity == "critical"
                            else 4 if severity == "high" else 3
                        ),
                        recommended_response=f"Investigate {row['first_category_name']} market dynamics. Consider pricing review, promotion campaign, or product refresh.",
                        threat_timeline=(
                            "immediate"
                            if severity in ["critical", "high"]
                            else "short_term"
                        ),
                    )
                    threats.append(threat)

            # Add seasonal and market threats
            current_month = datetime.now().month

            # Seasonal threat analysis
            if current_month in [11, 12]:  # Holiday season
                threats.append(
                    MarketThreat(
                        threat_id="SEASONAL-001",
                        threat_type="seasonal_competition",
                        severity="high",
                        affected_categories=["Electronics", "Gifts", "Holiday Items"],
                        estimated_impact=Decimal("50000"),
                        response_urgency=4,
                        recommended_response="Implement aggressive holiday pricing and promotion strategy. Ensure inventory availability.",
                        threat_timeline="immediate",
                    )
                )
            elif current_month in [6, 7, 8]:  # Summer season
                threats.append(
                    MarketThreat(
                        threat_id="SEASONAL-002",
                        threat_type="seasonal_shift",
                        severity="medium",
                        affected_categories=["Apparel", "Outdoor", "Travel"],
                        estimated_impact=Decimal("25000"),
                        response_urgency=3,
                        recommended_response="Adjust seasonal inventory mix. Focus on summer category promotions.",
                        threat_timeline="short_term",
                    )
                )

            # Economic threat analysis
            threats.append(
                MarketThreat(
                    threat_id="ECONOMIC-001",
                    threat_type="economic_pressure",
                    severity="medium",
                    affected_categories=["Discretionary Spending", "Premium Products"],
                    estimated_impact=Decimal("30000"),
                    response_urgency=3,
                    recommended_response="Develop value-oriented product lines. Enhance customer loyalty programs.",
                    threat_timeline="medium_term",
                )
            )

            return threats

        except Exception as e:
            self.logger.error(f"Error assessing market threats: {e}")
            return []

    async def _identify_competitive_opportunities(
        self, city_id: Optional[int] = None
    ) -> List[CompetitiveOpportunity]:
        """Identify competitive opportunities and market gaps."""
        try:
            opportunities = []

            async with get_db_connection() as conn:
                # Analyze underperforming categories for opportunities
                query = """
                WITH category_performance AS (
                    SELECT 
                        ph.first_category_name,
                        ph.first_category_id,
                        COUNT(DISTINCT s.product_id) as product_count,
                        SUM(s.units_sold) as total_sales,
                        SUM(s.units_sold * 25) as total_revenue,
                        AVG(s.units_sold) as avg_performance,
                        STDDEV(s.units_sold) as performance_consistency
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '60 days'
                """

                params = []
                if city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY ph.first_category_name, ph.first_category_id
                    HAVING COUNT(*) >= 20
                ),
                opportunity_analysis AS (
                    SELECT 
                        *,
                        total_revenue / SUM(total_revenue) OVER() as market_share,
                        RANK() OVER (ORDER BY avg_performance ASC) as performance_rank,
                        RANK() OVER (ORDER BY total_revenue DESC) as revenue_rank
                    FROM category_performance
                )
                SELECT *
                FROM opportunity_analysis
                WHERE market_share < 0.15  -- Categories with less than 15% market share
                    AND product_count >= 5  -- Sufficient product variety
                ORDER BY (performance_rank + revenue_rank) DESC  -- Combined opportunity score
                LIMIT 6
                """

                result = await conn.fetch(query, *params)

                # Generate opportunities based on analysis
                for i, row in enumerate(result):
                    # Calculate opportunity metrics
                    revenue_potential = Decimal(
                        str(row["total_revenue"] * 1.5)
                    )  # 50% growth potential
                    success_probability = min(0.85, 0.4 + (row["product_count"] / 20))

                    # Determine implementation difficulty
                    if row["product_count"] >= 15:
                        difficulty = "low"
                    elif row["product_count"] >= 10:
                        difficulty = "medium"
                    else:
                        difficulty = "high"

                    # Strategic importance based on market share potential
                    if row["market_share"] < 0.05:
                        importance = "high"
                    elif row["market_share"] < 0.10:
                        importance = "medium"
                    else:
                        importance = "low"

                    opportunity = CompetitiveOpportunity(
                        opportunity_id=f"OPP-{i+1:03d}",
                        opportunity_type="market_expansion",
                        market_gap=f"Underperforming category: {row['first_category_name']} with {row['market_share']:.1%} market share",
                        revenue_potential=revenue_potential,
                        implementation_difficulty=difficulty,
                        success_probability=success_probability,
                        strategic_importance=importance,
                        action_plan=[
                            f"Expand product selection in {row['first_category_name']}",
                            "Develop targeted marketing campaigns",
                            "Optimize pricing strategy for category",
                            "Establish supplier partnerships",
                            "Monitor competitive response",
                        ],
                    )
                    opportunities.append(opportunity)

            # Add strategic opportunities
            strategic_opportunities = [
                CompetitiveOpportunity(
                    opportunity_id="STRATEGIC-001",
                    opportunity_type="digital_transformation",
                    market_gap="Enhanced digital shopping experience",
                    revenue_potential=Decimal("75000"),
                    implementation_difficulty="medium",
                    success_probability=0.75,
                    strategic_importance="high",
                    action_plan=[
                        "Develop mobile app with personalized recommendations",
                        "Implement omnichannel customer experience",
                        "Create loyalty program with digital rewards",
                        "Establish social media presence and engagement",
                    ],
                ),
                CompetitiveOpportunity(
                    opportunity_id="STRATEGIC-002",
                    opportunity_type="private_label",
                    market_gap="Private label product development",
                    revenue_potential=Decimal("45000"),
                    implementation_difficulty="high",
                    success_probability=0.65,
                    strategic_importance="high",
                    action_plan=[
                        "Identify high-margin product categories",
                        "Develop private label supplier relationships",
                        "Create brand positioning and marketing strategy",
                        "Ensure quality control and customer satisfaction",
                    ],
                ),
                CompetitiveOpportunity(
                    opportunity_id="STRATEGIC-003",
                    opportunity_type="customer_experience",
                    market_gap="Superior customer service differentiation",
                    revenue_potential=Decimal("35000"),
                    implementation_difficulty="low",
                    success_probability=0.80,
                    strategic_importance="medium",
                    action_plan=[
                        "Implement advanced staff training programs",
                        "Create customer feedback and response system",
                        "Develop personalized shopping assistance",
                        "Establish customer satisfaction metrics and goals",
                    ],
                ),
            ]

            opportunities.extend(strategic_opportunities)

            return opportunities

        except Exception as e:
            self.logger.error(f"Error identifying competitive opportunities: {e}")
            return []

    async def _analyze_competitive_pricing(
        self, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze competitive pricing strategies and positioning."""
        try:
            async with get_db_connection() as conn:
                # Analyze pricing patterns to infer competitive landscape
                query = """
                WITH pricing_analysis AS (
                    SELECT 
                        ph.first_category_name,
                        AVG(s.units_sold * 25) as avg_transaction_value,  -- Estimated pricing
                        STDDEV(s.units_sold * 25) as price_variance,
                        MIN(s.units_sold * 25) as min_price_estimate,
                        MAX(s.units_sold * 25) as max_price_estimate,
                        COUNT(*) as sample_size,
                        SUM(s.units_sold) as total_volume
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '30 days'
                        AND s.units_sold > 0
                """

                params = []
                if city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                query += """
                    GROUP BY ph.first_category_name
                    HAVING COUNT(*) >= 20
                    ORDER BY total_volume DESC
                    LIMIT 15
                )
                SELECT 
                    *,
                    CASE 
                        WHEN avg_transaction_value > 75 THEN 'premium'
                        WHEN avg_transaction_value > 40 THEN 'mid_market'
                        WHEN avg_transaction_value > 20 THEN 'value'
                        ELSE 'budget'
                    END as pricing_tier
                FROM pricing_analysis
                """

                result = await conn.fetch(query, *params)

                # Analyze pricing positioning
                pricing_analysis = {
                    "overall_pricing_position": "mid_market",  # Based on average transaction values
                    "pricing_flexibility": "moderate",
                    "price_competitiveness_score": 75,
                    "category_pricing": [],
                    "pricing_recommendations": [],
                    "competitive_pricing_insights": {},
                }

                total_revenue = sum(
                    row["avg_transaction_value"] * row["total_volume"] for row in result
                )

                for row in result:
                    category_analysis = {
                        "category": row["first_category_name"],
                        "pricing_tier": row["pricing_tier"],
                        "avg_transaction_value": float(row["avg_transaction_value"]),
                        "price_variance": float(row["price_variance"] or 0),
                        "volume_share": (
                            (
                                row["total_volume"]
                                / sum(r["total_volume"] for r in result)
                            )
                            if result
                            else 0
                        ),
                        "pricing_pressure": (
                            "high"
                            if row["price_variance"]
                            > row["avg_transaction_value"] * 0.3
                            else "moderate"
                        ),
                    }
                    pricing_analysis["category_pricing"].append(category_analysis)  # type: ignore

                # Generate pricing recommendations
                premium_categories = [
                    cat
                    for cat in pricing_analysis["category_pricing"]  # type: ignore
                    if cat["pricing_tier"] == "premium"
                ]
                budget_categories = [
                    cat
                    for cat in pricing_analysis["category_pricing"]  # type: ignore
                    if cat["pricing_tier"] == "budget"
                ]

                if len(premium_categories) > 5:
                    pricing_analysis["pricing_recommendations"].append(  # type: ignore
                        {
                            "type": "premium_optimization",
                            "recommendation": "Consider premium pricing strategy validation",
                            "impact": "medium",
                            "implementation": "Review premium category margins and competitive positioning",
                        }
                    )

                if len(budget_categories) > 3:
                    pricing_analysis["pricing_recommendations"].append(  # type: ignore
                        {
                            "type": "value_enhancement",
                            "recommendation": "Enhance value perception in budget categories",
                            "impact": "high",
                            "implementation": "Improve product quality or bundle offerings",
                        }
                    )

                # Competitive insights
                pricing_analysis["competitive_pricing_insights"] = {
                    "market_positioning": "Balanced pricing across categories with opportunities for premium positioning",
                    "pricing_advantages": [
                        "Diverse pricing tiers across categories",
                        "Moderate pricing flexibility",
                        "Strong mid-market positioning",
                    ],
                    "pricing_vulnerabilities": [
                        "High price variance in some categories",
                        "Potential premium market underutilization",
                        "Need for value-tier differentiation",
                    ],
                    "competitive_response_options": [
                        "Dynamic pricing implementation",
                        "Category-specific pricing strategies",
                        "Value-added service bundling",
                    ],
                }

                return pricing_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing competitive pricing: {e}")
            return {}

    async def _generate_strategic_recommendations(
        self,
        competitors: List[CompetitorProfile],
        threats: List[MarketThreat],
        opportunities: List[CompetitiveOpportunity],
        pricing_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on competitive analysis."""
        try:
            recommendations = []

            # Threat response recommendations
            critical_threats = [t for t in threats if t.severity == "critical"]
            if critical_threats:
                recommendations.append(
                    {
                        "category": "threat_response",
                        "priority": "immediate",
                        "title": "Address Critical Market Threats",
                        "description": f"{len(critical_threats)} critical threats requiring immediate response",
                        "strategic_actions": [
                            "Implement emergency competitive response protocols",
                            "Accelerate product development and innovation",
                            "Strengthen customer retention programs",
                            "Review pricing strategy for affected categories",
                        ],
                        "timeline": "1-2 weeks",
                        "expected_impact": "Prevent market share erosion",
                        "success_metrics": [
                            "Market share maintenance",
                            "Customer retention rate",
                            "Revenue protection",
                        ],
                    }
                )

            # Opportunity capture recommendations
            high_value_opportunities = [
                o for o in opportunities if float(o.revenue_potential) > 40000
            ]
            if high_value_opportunities:
                recommendations.append(
                    {
                        "category": "market_expansion",
                        "priority": "high",
                        "title": "Capture High-Value Market Opportunities",
                        "description": f"{len(high_value_opportunities)} opportunities with ${sum(float(o.revenue_potential) for o in high_value_opportunities):,.0f} potential",
                        "strategic_actions": [
                            "Prioritize digital transformation initiatives",
                            "Develop private label product strategy",
                            "Enhance customer experience differentiation",
                            "Expand underperforming categories",
                        ],
                        "timeline": "3-6 months",
                        "expected_impact": f"${sum(float(o.revenue_potential) for o in high_value_opportunities):,.0f} revenue potential",
                        "success_metrics": [
                            "Revenue growth",
                            "Market share expansion",
                            "Customer acquisition",
                        ],
                    }
                )

            # Competitive positioning recommendations
            premium_competitors = [
                c
                for c in competitors
                if c.pricing_strategy == "premium"
                and c.threat_level in ["high", "critical"]
            ]
            if premium_competitors:
                recommendations.append(
                    {
                        "category": "competitive_positioning",
                        "priority": "medium",
                        "title": "Counter Premium Competitor Pressure",
                        "description": f"Respond to {len(premium_competitors)} premium competitors with high threat levels",
                        "strategic_actions": [
                            "Develop premium product line extensions",
                            "Enhance brand positioning and marketing",
                            "Implement loyalty program improvements",
                            "Focus on customer experience excellence",
                        ],
                        "timeline": "2-4 months",
                        "expected_impact": "Improved competitive positioning in premium segments",
                        "success_metrics": [
                            "Brand perception",
                            "Premium category market share",
                            "Customer loyalty",
                        ],
                    }
                )

            # Pricing strategy recommendations
            if pricing_analysis.get("pricing_recommendations"):
                recommendations.append(
                    {
                        "category": "pricing_optimization",
                        "priority": "medium",
                        "title": "Optimize Competitive Pricing Strategy",
                        "description": "Enhance pricing competitiveness across categories",
                        "strategic_actions": [
                            "Implement dynamic pricing capabilities",
                            "Develop category-specific pricing strategies",
                            "Create value-added bundling options",
                            "Monitor competitive pricing intelligence",
                        ],
                        "timeline": "1-3 months",
                        "expected_impact": "Improved pricing competitiveness and margin optimization",
                        "success_metrics": [
                            "Price competitiveness index",
                            "Margin improvement",
                            "Sales volume growth",
                        ],
                    }
                )

            # Innovation and differentiation recommendations
            recommendations.append(
                {
                    "category": "innovation_strategy",
                    "priority": "medium",
                    "title": "Drive Innovation and Differentiation",
                    "description": "Build sustainable competitive advantages through innovation",
                    "strategic_actions": [
                        "Establish innovation labs and R&D capabilities",
                        "Develop unique product and service offerings",
                        "Create customer co-innovation programs",
                        "Implement technology-driven solutions",
                    ],
                    "timeline": "6-12 months",
                    "expected_impact": "Sustainable competitive differentiation",
                    "success_metrics": [
                        "Innovation pipeline",
                        "Unique offering adoption",
                        "Customer satisfaction",
                    ],
                }
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating strategic recommendations: {e}")
            return []

    def _calculate_competitive_strength(
        self,
        competitors: List[CompetitorProfile],
        threats: List[MarketThreat],
        opportunities: List[CompetitiveOpportunity],
    ) -> float:
        """Calculate overall competitive strength score."""
        try:
            # Base score
            strength_score = 70.0

            # Adjust for competitor threats
            critical_competitors = len(
                [c for c in competitors if c.threat_level == "critical"]
            )
            high_threat_competitors = len(
                [c for c in competitors if c.threat_level == "high"]
            )

            strength_score -= critical_competitors * 15 + high_threat_competitors * 8

            # Adjust for market threats
            critical_threats = len([t for t in threats if t.severity == "critical"])
            high_threats = len([t for t in threats if t.severity == "high"])

            strength_score -= critical_threats * 10 + high_threats * 5

            # Adjust for opportunities
            high_value_opportunities = len(
                [o for o in opportunities if float(o.revenue_potential) > 40000]
            )
            strategic_opportunities = len(
                [o for o in opportunities if o.strategic_importance == "high"]
            )

            strength_score += high_value_opportunities * 5 + strategic_opportunities * 3

            # Ensure score is within reasonable bounds
            return max(20.0, min(95.0, strength_score))

        except Exception as e:
            self.logger.error(f"Error calculating competitive strength: {e}")
            return 60.0

    async def _store_competitive_analysis(
        self,
        city_id: Optional[int],
        competitors: List[CompetitorProfile],
        threats: List[MarketThreat],
        opportunities: List[CompetitiveOpportunity],
        recommendations: List[Dict[str, Any]],
    ) -> None:
        """Store competitive analysis results in database."""
        try:
            async with get_db_connection() as conn:
                # Store main competitive intelligence record
                await conn.execute(
                    """
                    INSERT INTO competitive_intelligence (
                        city_id, competitor_name, market_share_estimate, pricing_strategy,
                        threat_level, competitive_advantages, weaknesses, strategic_focus,
                        threat_assessment, opportunity_analysis, strategic_recommendations,
                        analysis_date
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                """,
                    city_id,
                    "Market Analysis Summary",
                    (
                        sum(c.market_share_estimate for c in competitors)
                        / len(competitors)
                        if competitors
                        else 0
                    ),
                    "balanced",
                    "medium",
                    json.dumps(
                        [adv for c in competitors for adv in c.competitive_advantages]
                    ),
                    json.dumps([weak for c in competitors for weak in c.weaknesses]),
                    json.dumps(
                        [focus for c in competitors for focus in c.strategic_focus]
                    ),
                    json.dumps(
                        [
                            {
                                "threat_id": t.threat_id,
                                "type": t.threat_type,
                                "severity": t.severity,
                                "impact": float(t.estimated_impact),
                                "response": t.recommended_response,
                            }
                            for t in threats
                        ]
                    ),
                    json.dumps(
                        [
                            {
                                "opportunity_id": o.opportunity_id,
                                "type": o.opportunity_type,
                                "revenue_potential": float(o.revenue_potential),
                                "success_probability": o.success_probability,
                                "importance": o.strategic_importance,
                            }
                            for o in opportunities
                        ]
                    ),
                    json.dumps(recommendations),
                )

        except Exception as e:
            self.logger.error(f"Error storing competitive analysis: {e}")

    async def get_competitor_monitoring_dashboard(
        self, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get real-time competitor monitoring dashboard."""
        try:
            # This would integrate with external data sources in a real system
            # For now, return simulated monitoring data

            return {
                "monitoring_timestamp": datetime.now().isoformat(),
                "market_scope": f"City {city_id}" if city_id else "All Markets",
                "competitor_activity": {
                    "recent_price_changes": [
                        {
                            "competitor": "Premium Market Leader",
                            "category": "Electronics",
                            "change_type": "price_reduction",
                            "magnitude": "5-10%",
                            "detected_date": (
                                datetime.now() - timedelta(days=2)
                            ).isoformat(),
                            "threat_level": "medium",
                        },
                        {
                            "competitor": "Value Competitor",
                            "category": "Groceries",
                            "change_type": "promotion_launch",
                            "magnitude": "15% discount",
                            "detected_date": (
                                datetime.now() - timedelta(days=1)
                            ).isoformat(),
                            "threat_level": "high",
                        },
                    ],
                    "new_store_openings": [
                        {
                            "competitor": "Regional Chain",
                            "location": "Downtown Area",
                            "opening_date": (
                                datetime.now() + timedelta(days=30)
                            ).isoformat(),
                            "threat_assessment": "medium",
                            "recommended_response": "Strengthen local marketing and customer loyalty programs",
                        }
                    ],
                    "product_launches": [
                        {
                            "competitor": "Category Specialist",
                            "product_category": "Organic Foods",
                            "launch_date": (
                                datetime.now() - timedelta(days=7)
                            ).isoformat(),
                            "market_impact": "moderate",
                            "response_opportunity": "Develop competing organic product line",
                        }
                    ],
                },
                "market_intelligence": {
                    "trending_categories": [
                        "Health & Wellness",
                        "Sustainable Products",
                        "Home Office",
                    ],
                    "declining_categories": [
                        "Traditional Media",
                        "Non-Essential Accessories",
                    ],
                    "pricing_trends": {
                        "overall_direction": "stable",
                        "high_pressure_categories": ["Electronics", "Apparel"],
                        "premium_opportunities": ["Organic", "Artisanal"],
                    },
                },
                "alert_summary": {
                    "total_competitive_alerts": 8,
                    "high_priority_alerts": 3,
                    "response_required": 2,
                    "monitoring_coverage": "85% market visibility",
                },
                "recommended_actions": [
                    {
                        "priority": "high",
                        "action": "Respond to Value Competitor grocery promotion",
                        "timeline": "immediate",
                        "expected_impact": "Maintain market share in price-sensitive segment",
                    },
                    {
                        "priority": "medium",
                        "action": "Monitor Premium Market Leader electronics pricing",
                        "timeline": "ongoing",
                        "expected_impact": "Identify pricing optimization opportunities",
                    },
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting competitor monitoring dashboard: {e}")
            return {}

    async def get_market_share_analysis(
        self, category_id: Optional[int] = None, city_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get detailed market share analysis for specific category or overall market."""
        try:
            async with get_db_connection() as conn:
                # Analyze our market performance
                query = """
                WITH market_analysis AS (
                    SELECT 
                        ph.first_category_name,
                        ph.first_category_id,
                        SUM(s.units_sold) as our_units,
                        SUM(s.units_sold * 25) as our_revenue,  -- Estimated revenue
                        COUNT(DISTINCT s.store_id) as our_store_count,
                        COUNT(DISTINCT s.product_id) as our_product_count
                    FROM sales_data s
                    JOIN product_hierarchy ph ON s.product_id = ph.product_id
                    WHERE s.date >= NOW() - INTERVAL '90 days'
                """

                params = []
                if city_id:
                    query += " AND s.store_id IN (SELECT store_id FROM store_hierarchy WHERE city_id = $1)"
                    params.append(city_id)

                if category_id:
                    param_num = len(params) + 1
                    query += f" AND ph.first_category_id = ${param_num}"
                    params.append(category_id)

                query += """
                    GROUP BY ph.first_category_name, ph.first_category_id
                    ORDER BY our_revenue DESC
                )
                SELECT 
                    *,
                    our_revenue / SUM(our_revenue) OVER() as category_share
                FROM market_analysis
                """

                result = await conn.fetch(query, *params)

                # Simulate total market size (in a real system, this would come from market research)
                market_multiplier = 3.5  # Assume we see ~30% of total market

                market_analysis = {
                    "analysis_scope": {
                        "category_id": category_id,
                        "city_id": city_id,
                        "analysis_period": "90 days",
                    },
                    "our_performance": [],
                    "market_estimates": {
                        "total_market_size_estimate": 0,
                        "our_estimated_market_share": 0,
                        "market_growth_estimate": 0.05,  # 5% estimated growth
                    },
                    "competitive_landscape": {
                        "market_concentration": "moderate",
                        "competitive_intensity": "high",
                        "barriers_to_entry": "medium",
                    },
                    "strategic_position": {
                        "position_strength": "moderate",
                        "growth_opportunities": [],
                        "defensive_priorities": [],
                    },
                }

                total_our_revenue = sum(row["our_revenue"] for row in result)
                estimated_total_market = total_our_revenue * market_multiplier

                market_analysis["market_estimates"]["total_market_size_estimate"] = (  # type: ignore
                    float(estimated_total_market)
                )
                market_analysis["market_estimates"]["our_estimated_market_share"] = (  # type: ignore
                    1 / market_multiplier
                )

                for row in result:
                    category_analysis = {
                        "category_name": row["first_category_name"],
                        "category_id": row["first_category_id"],
                        "our_revenue": float(row["our_revenue"]),
                        "our_units": row["our_units"],
                        "our_store_presence": row["our_store_count"],
                        "our_product_count": row["our_product_count"],
                        "category_share_of_our_business": float(row["category_share"]),
                        "estimated_market_share": 1 / market_multiplier,
                        "competitive_position": (
                            "challenger" if row["category_share"] > 0.15 else "follower"
                        ),
                    }
                    market_analysis["our_performance"].append(category_analysis)  # type: ignore

                # Add strategic insights
                top_categories = sorted(
                    result, key=lambda x: x["our_revenue"], reverse=True
                )[:3]
                market_analysis["strategic_position"]["growth_opportunities"] = [  # type: ignore
                    f"Expand in {cat['first_category_name']} - strong current performance"
                    for cat in top_categories
                ]

                weak_categories = sorted(result, key=lambda x: x["our_revenue"])[:2]
                market_analysis["strategic_position"]["defensive_priorities"] = [  # type: ignore
                    f"Strengthen position in {cat['first_category_name']} - underperforming"
                    for cat in weak_categories
                ]

                return market_analysis

        except Exception as e:
            self.logger.error(f"Error getting market share analysis: {e}")
            return {}
