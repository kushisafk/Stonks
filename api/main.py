import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import live, backtest, features

app = FastAPI(title="Stonks API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(live.router, prefix="/live", tags=["live"])
app.include_router(backtest.router, prefix="/backtest", tags=["backtest"])
app.include_router(features.router, prefix="/features", tags=["features"])
