"""
FreshRetail Forecasting API initialization.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Import API modules
from api.forecast import app as forecast_app
from api.promotions import router as promotions_router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create main API application
app = FastAPI(
    title="FreshRetail Forecasting API",
    description="API for sales forecasting, promotion impact, and inventory optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from other modules
app.include_router(promotions_router)

# Mount the forecast app directly
app.mount("/forecast", forecast_app)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the FreshRetail Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "forecast": "/forecast/",
            "promotions": "/promotions/"
        }
    }

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # This is used when running locally
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
