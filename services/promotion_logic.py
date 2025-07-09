from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def simulate_and_recommend_promotions(
    df: pd.DataFrame,
    model,
    group_cols: List[str],
    max_discount: float = 0.5,
    count: int = 10,
    target_uplift: Optional[float] = None,
    target_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Simulate different discount levels and recommend promotions based on uplift model.
    Returns recommendations and summary statistics.
    """
    discount_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    discount_levels = [d for d in discount_levels if d <= max_discount]
    recommendations = []

    if df is None or len(df) < 10:
        return {"recommendations": [], "summary": {}}

    combinations = df[group_cols].drop_duplicates()

    for _, combo in combinations.iterrows():
        combo_filter = True
        for col in group_cols:
            combo_filter = combo_filter & (df[col] == combo[col])
        combo_data = df[combo_filter].copy()
        if len(combo_data) < 10:
            continue
        baseline = (
            combo_data[~combo_data['promo_flag']]['sale_amount'].mean()
            if 'promo_flag' in combo_data.columns else combo_data['sale_amount'].mean()
        )
        for discount in discount_levels:
            sim_data = combo_data.copy()
            sim_data['discount'] = discount
            sim_data['promo_flag'] = True
            try:
                pred_uplift = model.predict(sim_data).mean()
                if pred_uplift <= 0:
                    continue
                discount_cost = baseline * discount
                roi = (pred_uplift - discount_cost) / discount_cost if discount_cost > 0 else 0
                rec = {
                    "discount_percentage": float(discount),
                    "estimated_uplift": float(pred_uplift),
                    "estimated_roi": float(roi),
                    "baseline_sales": float(baseline),
                    "projected_sales": float(baseline + pred_uplift),
                    "promotion_type": "Discount"
                }
                for col in group_cols:
                    rec[col] = int(combo[col]) if pd.notnull(combo[col]) else None
                recommendations.append(rec)
            except Exception as e:
                logger.warning(f"Error predicting uplift for {combo}: {str(e)}", exc_info=True)

    recommendations.sort(key=lambda x: x["estimated_roi"], reverse=True)
    top_recommendations = recommendations[:count]
    if target_uplift is not None:
        top_recommendations = [rec for rec in top_recommendations if rec["estimated_uplift"] >= target_uplift]
    avg_uplift = sum(rec["estimated_uplift"] for rec in top_recommendations) / len(top_recommendations) if top_recommendations else 0
    avg_roi = sum(rec["estimated_roi"] for rec in top_recommendations) / len(top_recommendations) if top_recommendations else 0
    avg_discount = sum(rec["discount_percentage"] for rec in top_recommendations) / len(top_recommendations) if top_recommendations else 0
    return {
        "recommendations": top_recommendations,
        "summary": {
            "total_recommendations": len(top_recommendations),
            "average_uplift": float(avg_uplift),
            "average_roi": float(avg_roi),
            "average_discount": float(avg_discount),
            "target_date": target_date.strftime('%Y-%m-%d') if target_date else None
        }
    } 