"""
FastAPI endpoints for promotion management and recommendation.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import logging
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.promo_uplift_model import PromoUpliftModel
from database.connection import get_db_engine, get_supabase_client
from api.forecast import get_promo_model, fetch_historical_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/promotions",
    tags=["promotions"],
    responses={404: {"description": "Not found"}},
)

# Data models for API requests and responses
class PromotionBase(BaseModel):
    """Base model for promotion data."""
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    start_date: str
    end_date: str
    promotion_type: str
    discount_percentage: float
    display_location: Optional[str] = None
    campaign_id: Optional[str] = None

class PromotionCreate(PromotionBase):
    """Model for creating a new promotion."""
    pass

class PromotionUpdate(BaseModel):
    """Model for updating an existing promotion."""
    promotion_id: int
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    promotion_type: Optional[str] = None
    discount_percentage: Optional[float] = None
    display_location: Optional[str] = None
    campaign_id: Optional[str] = None

class PromotionResponse(PromotionBase):
    """Response model for promotion data."""
    id: int
    created_at: datetime

class PromotionRecommendationRequest(BaseModel):
    """Request model for promotion recommendations."""
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    city_id: Optional[int] = None
    target_date: Optional[str] = None
    target_uplift: Optional[float] = None
    max_discount: Optional[float] = 0.5
    count: int = 10

class PromotionRecommendationResponse(BaseModel):
    """Response model for promotion recommendations."""
    recommendations: List[Dict[str, Any]]
    summary: Dict[str, Any]

@router.post("/", response_model=PromotionResponse)
def create_promotion(promotion: PromotionCreate):
    """
    Create a new promotion.
    
    Parameters:
    - store_id: Store ID (optional, can be null for all stores)
    - product_id: Product ID (optional, can be null for all products)
    - start_date: Start date for the promotion
    - end_date: End date for the promotion
    - promotion_type: Type of promotion (e.g., "Discount", "BOGO", "Bundle")
    - discount_percentage: Discount percentage (0.0-1.0)
    - display_location: Location in the store for promotion display (optional)
    - campaign_id: ID of the campaign this promotion belongs to (optional)
    
    Returns:
    - Created promotion with ID and timestamp
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Parse dates to ensure proper format
        start_date = pd.to_datetime(promotion.start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(promotion.end_date).strftime('%Y-%m-%d')
        
        # Prepare data for insertion
        promo_data = {
            "store_id": promotion.store_id,
            "product_id": promotion.product_id,
            "start_date": start_date,
            "end_date": end_date,
            "promotion_type": promotion.promotion_type,
            "discount_percentage": promotion.discount_percentage,
            "display_location": promotion.display_location,
            "campaign_id": promotion.campaign_id,
            "created_at": datetime.now().isoformat()
        }
        
        # Insert into Supabase
        response = supabase.table("promotion_events").insert(promo_data).execute()
        
        if response.data and len(response.data) > 0:
            # Add ID to the response
            result = response.data[0]
            promo_data["id"] = result.get("id")
            return PromotionResponse(**promo_data)
        else:
            raise HTTPException(status_code=500, detail="Failed to create promotion")
    
    except Exception as e:
        logger.error(f"Error creating promotion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating promotion: {str(e)}")

@router.put("/{promotion_id}", response_model=PromotionResponse)
def update_promotion(promotion_id: int, promotion: PromotionUpdate):
    """
    Update an existing promotion.
    
    Parameters:
    - promotion_id: ID of the promotion to update
    - promotion: Updated promotion data (only fields to update)
    
    Returns:
    - Updated promotion
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Check if promotion exists
        existing = supabase.table("promotion_events").select("*").eq("id", promotion_id).execute()
        
        if not existing.data or len(existing.data) == 0:
            raise HTTPException(status_code=404, detail=f"Promotion with ID {promotion_id} not found")
        
        # Build update data - only include fields that are not None
        update_data = {}
        if promotion.store_id is not None:
            update_data["store_id"] = promotion.store_id
        if promotion.product_id is not None:
            update_data["product_id"] = promotion.product_id
        if promotion.start_date is not None:
            update_data["start_date"] = pd.to_datetime(promotion.start_date).strftime('%Y-%m-%d')
        if promotion.end_date is not None:
            update_data["end_date"] = pd.to_datetime(promotion.end_date).strftime('%Y-%m-%d')
        if promotion.promotion_type is not None:
            update_data["promotion_type"] = promotion.promotion_type
        if promotion.discount_percentage is not None:
            update_data["discount_percentage"] = promotion.discount_percentage
        if promotion.display_location is not None:
            update_data["display_location"] = promotion.display_location
        if promotion.campaign_id is not None:
            update_data["campaign_id"] = promotion.campaign_id
        
        # Update in Supabase
        response = supabase.table("promotion_events").update(update_data).eq("id", promotion_id).execute()
        
        if response.data and len(response.data) > 0:
            return PromotionResponse(**response.data[0])
        else:
            raise HTTPException(status_code=500, detail="Failed to update promotion")
    
    except Exception as e:
        logger.error(f"Error updating promotion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating promotion: {str(e)}")

@router.delete("/{promotion_id}")
def delete_promotion(promotion_id: int):
    """
    Delete an existing promotion.
    
    Parameters:
    - promotion_id: ID of the promotion to delete
    
    Returns:
    - Success message
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Check if promotion exists
        existing = supabase.table("promotion_events").select("*").eq("id", promotion_id).execute()
        
        if not existing.data or len(existing.data) == 0:
            raise HTTPException(status_code=404, detail=f"Promotion with ID {promotion_id} not found")
        
        # Delete from Supabase
        response = supabase.table("promotion_events").delete().eq("id", promotion_id).execute()
        
        if response.data is not None:
            return {"message": f"Promotion with ID {promotion_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete promotion")
    
    except Exception as e:
        logger.error(f"Error deleting promotion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting promotion: {str(e)}")

@router.get("/", response_model=List[PromotionResponse])
def list_promotions(
    store_id: Optional[int] = None, 
    product_id: Optional[int] = None,
    category_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    active: Optional[bool] = None
):
    """
    List promotions with optional filters.
    
    Parameters:
    - store_id: Filter by store ID
    - product_id: Filter by product ID
    - category_id: Filter by category ID
    - start_date: Filter by start date
    - end_date: Filter by end date
    - active: If true, return only active promotions (current date between start and end date)
    
    Returns:
    - List of matching promotions
    """
    try:
        engine = get_db_engine()
        
        # Base query
        query = """
        SELECT 
            pe.id,
            pe.store_id,
            pe.product_id,
            pe.start_date,
            pe.end_date,
            pe.promotion_type,
            pe.discount_percentage,
            pe.display_location,
            pe.campaign_id,
            pe.created_at
        FROM promotion_events pe
        """
        
        # Add join if filtering by category
        if category_id is not None:
            query += """
            JOIN product_hierarchy ph ON pe.product_id = ph.product_id
            """
        
        # Where clause
        filters = []
        params = {}
        
        if store_id is not None:
            filters.append("pe.store_id = :store_id")
            params["store_id"] = store_id
            
        if product_id is not None:
            filters.append("pe.product_id = :product_id")
            params["product_id"] = product_id
            
        if category_id is not None:
            filters.append("ph.first_category_id = :category_id")
            params["category_id"] = category_id
            
        if start_date is not None:
            filters.append("pe.start_date >= :start_date")
            params["start_date"] = start_date
            
        if end_date is not None:
            filters.append("pe.end_date <= :end_date")
            params["end_date"] = end_date
            
        if active is not None and active:
            today = datetime.now().strftime('%Y-%m-%d')
            filters.append("pe.start_date <= :today AND pe.end_date >= :today")
            params["today"] = today
        
        # Add filters to query
        if filters:
            query += " WHERE " + " AND ".join(filters)
            
        # Order by start date descending (newest first)
        query += " ORDER BY pe.start_date DESC"
        
        # Execute query
        df = pd.read_sql(query, engine, params=params)
        
        # Convert to response format
        promotions = []
        for _, row in df.iterrows():
            promotion = PromotionResponse(
                id=row["id"],
                store_id=row["store_id"],
                product_id=row["product_id"],
                start_date=row["start_date"],
                end_date=row["end_date"],
                promotion_type=row["promotion_type"],
                discount_percentage=row["discount_percentage"],
                display_location=row["display_location"],
                campaign_id=row["campaign_id"],
                created_at=row["created_at"]
            )
            promotions.append(promotion)
            
        return promotions
    
    except Exception as e:
        logger.error(f"Error listing promotions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing promotions: {str(e)}")

@router.post("/recommend", response_model=PromotionRecommendationResponse)
def recommend_promotions(request: PromotionRecommendationRequest, model: PromoUpliftModel = Depends(get_promo_model)):
    """
    Recommend promotions based on historical data and uplift model.
    
    Parameters:
    - store_id: Optional store ID to filter by
    - product_id: Optional product ID to filter by
    - category_id: Optional category ID to filter by
    - city_id: Optional city ID to filter by
    - target_date: Target date for recommendations (default: today)
    - target_uplift: Target uplift percentage (optional)
    - max_discount: Maximum discount percentage (default: 0.5)
    - count: Number of recommendations to return (default: 10)
    
    Returns:
    - List of recommended promotions with estimated uplift
    """
    try:
        # Set default target date to today if not provided
        target_date = pd.to_datetime(request.target_date) if request.target_date else datetime.now().date()
        
        # Set date range for historical data analysis
        start_date = target_date - timedelta(days=90)  # Use last 90 days of data
        end_date = target_date - timedelta(days=1)     # Up to yesterday
        
        # Get historical data
        df = fetch_historical_data(
            store_id=request.store_id,
            product_id=request.product_id,
            category_id=request.category_id,
            city_id=request.city_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if len(df) < 30:
            raise HTTPException(status_code=400, 
                               detail=f"Not enough historical data ({len(df)} records) for promotion recommendations")
        
        # Train model if not already trained
        if not model.trained:
            logger.info("Training promotion uplift model for recommendations")
            model.train(df)
        
        # Group data by store/product combinations
        group_cols = []
        if request.store_id is None:
            group_cols.append('store_id')
        if request.product_id is None:
            group_cols.append('product_id')
        if request.category_id is None and 'first_category_id' in df.columns:
            group_cols.append('first_category_id')
        
        # Add default grouping if none specified
        if not group_cols:
            group_cols = ['product_id']
        
        # Get unique combinations
        combinations = df[group_cols].drop_duplicates()
        
        # For each combination, simulate different discount levels and predict uplift
        recommendations = []
        discount_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        discount_levels = [d for d in discount_levels if d <= request.max_discount]
        
        for _, combo in combinations.iterrows():
            # Filter data for this combination
            combo_filter = True
            for col in group_cols:
                combo_filter = combo_filter & (df[col] == combo[col])
            
            combo_data = df[combo_filter].copy()
            
            if len(combo_data) < 10:  # Skip if too little data
                continue
            
            # Calculate baseline metrics (no promo)
            baseline = combo_data[~combo_data['promo_flag']]['sale_amount'].mean() if 'promo_flag' in combo_data.columns else combo_data['sale_amount'].mean()
            
            for discount in discount_levels:
                # Create a copy with simulated discount
                sim_data = combo_data.copy()
                sim_data['discount'] = discount
                sim_data['promo_flag'] = True
                
                # Predict uplift
                try:
                    pred_uplift = model.predict(sim_data).mean()
                    
                    # Skip negative uplift predictions
                    if pred_uplift <= 0:
                        continue
                    
                    # Calculate ROI
                    incremental_sales = pred_uplift
                    discount_cost = baseline * discount
                    roi = (incremental_sales - discount_cost) / discount_cost if discount_cost > 0 else 0
                    
                    # Add to recommendations
                    rec = {
                        "discount_percentage": float(discount),
                        "estimated_uplift": float(pred_uplift),
                        "estimated_roi": float(roi),
                        "baseline_sales": float(baseline),
                        "projected_sales": float(baseline + pred_uplift),
                        "promotion_type": "Discount"
                    }
                    
                    # Add grouping columns
                    for col in group_cols:
                        rec[col] = int(combo[col]) if pd.notnull(combo[col]) else None
                    
                    recommendations.append(rec)
                except Exception as e:
                    logger.warning(f"Error predicting uplift for {combo}: {str(e)}")
        
        # Sort by estimated ROI and take the top n
        recommendations.sort(key=lambda x: x["estimated_roi"], reverse=True)
        top_recommendations = recommendations[:request.count]
        
        # If target uplift is specified, filter to meet target
        if request.target_uplift is not None:
            top_recommendations = [rec for rec in top_recommendations if rec["estimated_uplift"] >= request.target_uplift]
        
        # Calculate summary statistics
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
                "target_date": target_date.strftime('%Y-%m-%d')
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating promotion recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Add router to the main FastAPI app in api/__init__.py
