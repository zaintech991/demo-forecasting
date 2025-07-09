from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ForecastRequest(BaseModel):
    """Request model for forecasting."""
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    category_id: Optional[int] = None  # Used for filtering via product_hierarchy
    city_id: Optional[int] = None      # Used for filtering via store_hierarchy
    start_date: str
    periods: int = 30
    freq: str = "D"
    include_weather: bool = False
    include_holidays: bool = True
    include_promotions: bool = False
    return_components: bool = False

class ForecastResponse(BaseModel):
    """Response model for forecasting."""
    forecast: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None
    components: Optional[Dict[str, List[Dict[str, Any]]]] = None

class PromotionAnalysisRequest(BaseModel):
    """Request model for promotion analysis."""
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    city_id: Optional[int] = None
    start_date: str
    end_date: str
    promotion_type: Optional[str] = None
    discount_min: Optional[float] = None
    discount_max: Optional[float] = None

class StockoutAnalysisRequest(BaseModel):
    """Request model for stockout impact analysis."""
    store_id: int
    product_id: int
    start_date: str
    end_date: str

class HolidayImpactRequest(BaseModel):
    """Request model for holiday impact analysis."""
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    holiday_name: Optional[str] = None
    start_date: str
    end_date: str 