from typing import List, Union
from fastapi import APIRouter
from pydantic import BaseModel
import datetime

from model.config.settings import TICKERS
from model.live import run_live

router = APIRouter()

class LiveResultSuccess(BaseModel):
    ticker: str
    last_close: float
    predicted_next_close: float
    signal: str
    fib_levels: dict

class LiveResultError(BaseModel):
    ticker: str
    error: str

class LiveResponse(BaseModel):
    as_of_date: str
    results: List[Union[LiveResultSuccess, LiveResultError]]

@router.get("", response_model=LiveResponse)
def get_live():
    today_iso = datetime.date.today().isoformat()
    results = []
    
    for ticker in TICKERS:
        try:
            res = run_live(ticker)
            # Ensure proper rounding for float values
            res["last_close"] = round(res["last_close"], 2)
            res["predicted_next_close"] = round(res["predicted_next_close"], 2)
            res["fib_levels"] = {k: round(v, 2) for k, v in res["fib_levels"].items()}
            results.append(LiveResultSuccess(**res))
        except Exception:
            results.append(LiveResultError(ticker=ticker, error="Model file not found. Run training pipeline first."))
            
    return LiveResponse(as_of_date=today_iso, results=results)
