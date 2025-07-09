import os
import logging
from models.prophet_forecaster import ProphetForecaster
from models.promo_uplift_model import PromoUpliftModel

logger = logging.getLogger(__name__)

_forecast_model = None
_promo_model = None

def get_forecast_model():
    """
    Get or initialize the forecast model. Loads from disk if not already loaded.
    """
    global _forecast_model
    if _forecast_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/saved/forecast_model.json')
        _forecast_model = ProphetForecaster()
        if os.path.exists(model_path):
            logger.info(f"Loading forecast model from {model_path}")
            _forecast_model.load_model(model_path)
    return _forecast_model

def get_promo_model():
    """
    Get or initialize the promotion uplift model. Loads from disk if not already loaded.
    """
    global _promo_model
    if _promo_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/saved/promo_uplift_model.joblib')
        _promo_model = PromoUpliftModel(model_type="gradient_boost")
        if os.path.exists(model_path):
            logger.info(f"Loading promo uplift model from {model_path}")
            _promo_model.load_model(model_path)
    return _promo_model 