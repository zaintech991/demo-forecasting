from fastapi import FastAPI
from api import forecast

app = FastAPI(title="Boolmind Forecasting API")
app.include_router(forecast.router, prefix="/api")
