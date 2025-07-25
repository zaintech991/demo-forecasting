"""
Main application entry point.
"""

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from app.api import app as api_app
from dotenv import load_dotenv
from database.connection import DatabaseManager # Import DatabaseManager class directly
import logging # Import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {websocket.client}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except RuntimeError as e:
                logger.error(f"Error sending message to WebSocket {connection.client}: {e}")
                self.active_connections.remove(connection) # Remove broken connection

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# Serve enhanced multi-dimensional dashboard
@app.get("/enhanced_multi_dimensional.html")
def enhanced_multi_dimensional():
    return FileResponse("static/enhanced_multi_dimensional.html")


# WebSocket endpoint for real-time notifications
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await app.state.websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, or handle incoming messages if needed
            await websocket.receive_text()
    except WebSocketDisconnect:
        app.state.websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        app.state.websocket_manager.disconnect(websocket)


# Mount the API under /api
# app.mount("/api", api_app) # Removed app.mount
app.include_router(api_app, prefix="/api") # Use include_router instead to share app.state

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    app.state.db_manager = DatabaseManager() # Instantiate DatabaseManager directly
    await app.state.db_manager.initialize()
    app.state.websocket_manager = ConnectionManager() # Instantiate WebSocket ConnectionManager

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.db_manager.close()

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
        # Access db_manager from app.state
        async with app.state.db_manager.get_connection() as conn:
            records = await conn.fetch(query)
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

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
