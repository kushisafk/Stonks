from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model.config.settings import TICKERS, START_DATE, END_DATE
from model.src.forester_model import load_model
from model.src.data_pipeline import run_data_pipeline

router = APIRouter()

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class FeaturesResponse(BaseModel):
    ticker: str
    feature_importances: List[FeatureImportance]

@router.get("/{ticker}", response_model=FeaturesResponse)
def get_features(ticker: str):
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail="Invalid ticker.")
        
    try:
        model = load_model(ticker)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Model not found for {ticker}. Run training pipeline first.")
        
    try:
        raw_df, X, y = run_data_pipeline(ticker, START_DATE, END_DATE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    importances = list(model.feature_importances_)
    features_list = list(X.columns)
    
    paired = []
    for f, imp in zip(features_list, importances):
        paired.append({"feature": str(f), "importance": round(float(imp), 4)})
        
    paired.sort(key=lambda x: x["importance"], reverse=True)
    
    feature_importances = [FeatureImportance(**p) for p in paired]
    
    return FeaturesResponse(
        ticker=ticker,
        feature_importances=feature_importances
    )
