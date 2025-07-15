"""
Main application entry point.
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import app as api_app
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve index.html at root
@app.get("/")
def root():
    return FileResponse("static/index.html")


# Serve professional dashboard
@app.get("/professional_dashboard.html")
def professional_dashboard():
    return FileResponse("static/professional_dashboard.html")


# Serve enhanced dashboard
@app.get("/enhanced_index.html")
def enhanced_dashboard():
    return FileResponse("static/enhanced_index.html")


# Mount the API under /api
app.mount("/api", api_app)

# Add enhanced endpoints directly to main app
try:
    from api.enhanced_multi_modal_api import router as enhanced_router

    app.include_router(enhanced_router)
except ImportError as e:
    print(f"Warning: Could not load enhanced API: {e}")
    pass


# Debug endpoint to find valid data combinations
@app.get("/debug/data")
async def debug_data():
    """Debug endpoint to find valid data combinations"""
    try:
        import asyncpg

        DATABASE_URL = os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/dbname"
        )
        conn = await asyncpg.connect(DATABASE_URL)
        # Get some sample data
        query = """
        SELECT DISTINCT 
            sd.store_id, 
            sd.product_id, 
            sh.city_id,
            COUNT(*) as record_count
        FROM sales_data sd
        JOIN store_hierarchy sh ON sd.store_id = sh.store_id
        GROUP BY sd.store_id, sd.product_id, sh.city_id
        HAVING COUNT(*) >= 5
        ORDER BY record_count DESC
        LIMIT 10
        """
        records = await conn.fetch(query)
        await conn.close()
        return {
            "available_combinations": [
                {
                    "store_id": r["store_id"],
                    "product_id": r["product_id"],
                    "city_id": r["city_id"],
                    "record_count": r["record_count"],
                }
                for r in records
            ]
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
