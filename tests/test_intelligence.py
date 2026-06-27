import pytest
from src.intelligence.market_reasoner import MarketReasoner
from src.intelligence.confidence_analyzer import ConfidenceAnalyzer
from src.intelligence.risk_assessor import RiskAssessor
from src.intelligence.trade_planner import TradePlanner
from src.intelligence.recommendation_engine import RecommendationEngine

def test_market_reasoner():
    reasoner = MarketReasoner()
    
    # 1. Test standard trusted alignment
    features = {
        "market_regime": 1.0,
        "relative_strength_20d": 0.05,
        "volume_ratio": 1.5,
        "abnormal_volume_flag": 1.0,
        "sentiment_score": 0.30,
        "rsi": 55.0
    }
    res = reasoner.evaluate_market("BUY", 0.78, features)
    assert res["is_trusted"] is True
    assert len(res["confirmations"]) >= 3
    assert len(res["warnings"]) == 0
    
    # 2. Test counter-trend bearish mismatch
    features_bear = {
        "market_regime": -1.0,
        "relative_strength_20d": -0.05,
        "volume_ratio": 0.5,
        "abnormal_volume_flag": 0.0,
        "sentiment_score": -0.40,
        "rsi": 40.0
    }
    res_bear = reasoner.evaluate_market("BUY", 0.72, features_bear)
    assert res_bear["is_trusted"] is False
    assert any("broader S&P 500 regime is Bearish" in w for w in res_bear["warnings"])
    assert any("lacks volume support" in w for w in res_bear["warnings"])

def test_confidence_analyzer():
    analyzer = ConfidenceAnalyzer()
    
    # 1. Very High Confidence
    res = analyzer.analyze_confidence(
        probability=0.82,
        confirmations=["Confirm 1", "Confirm 2"],
        warnings=[]
    )
    assert res["tier"] == "Very High"
    
    # 2. Downgraded due to warnings
    res_down = analyzer.analyze_confidence(
        probability=0.82,
        confirmations=["Confirm 1"],
        warnings=["Warning critical conflict"]
    )
    assert res_down["tier"] == "Medium"
    
    # 3. Near 50% neutral zone
    res_neutral = analyzer.analyze_confidence(
        probability=0.51,
        confirmations=[],
        warnings=[]
    )
    assert res_neutral["tier"] == "Low"

def test_risk_assessor():
    assessor = RiskAssessor()
    
    # 1. Low risk scenario
    features_low = {
        "volatility_20d": 0.01,
        "spy_volatility_20d": 0.005,
        "market_regime": 1.0,
        "sentiment_score": 0.20
    }
    res_low = assessor.assess_risk("BUY", "High", features_low)
    assert res_low["score"] <= 35
    assert res_low["tier"] == "Low Risk"
    
    # 2. High risk scenario (high volatility, regime mismatch, negative sentiment)
    features_high = {
        "volatility_20d": 0.04,  # high vol
        "spy_volatility_20d": 0.02,
        "market_regime": -1.0,  # mismatch
        "sentiment_score": -0.30  # negative sentiment
    }
    res_high = assessor.assess_risk("BUY", "Low", features_high)
    assert res_high["score"] > 70
    assert res_high["tier"] == "High Risk"
    assert len(res_high["factors"]) >= 3

def test_trade_planner():
    planner = TradePlanner()
    
    # 1. Take Profit
    features_tp = {
        "rsi": 78.0,
        "return_5d": 0.12
    }
    res_tp = planner.plan_trade("BUY", "Very High", "Low Risk", True, features_tp)
    assert res_tp["action"] == "TAKE PROFITS"
    
    # 2. Safe Buy
    features_buy = {
        "rsi": 50.0,
        "return_5d": 0.01
    }
    res_buy = planner.plan_trade("BUY", "High", "Low Risk", True, features_buy)
    assert res_buy["action"] == "BUY"
    
    # 3. Defensive Buy
    res_def = planner.plan_trade("BUY", "Low", "High Risk", False, features_buy)
    assert res_def["action"] in ("BUY SMALL POSITION", "WAIT")

def test_recommendation_engine():
    engine = RecommendationEngine()
    features = {
        "market_regime": 1.0,
        "relative_strength_20d": 0.04,
        "volume_ratio": 1.3,
        "abnormal_volume_flag": 1.0,
        "sentiment_score": 0.25,
        "rsi": 60.0,
        "volatility_20d": 0.015,
        "spy_volatility_20d": 0.008
    }
    
    res = engine.generate_recommendation("AAPL", 0.74, "BUY", features)
    
    # Check JSON structure
    json_rep = res["json_report"]
    assert json_rep["ticker"] == "AAPL"
    assert json_rep["prediction"] == "BUY"
    assert json_rep["recommendation"] == "BUY"
    assert "confidence_tier" in json_rep
    assert "risk_score" in json_rep
    
    # Check Markdown report
    md_rep = res["markdown_report"]
    assert "## STONKS Intelligence Report: **AAPL**" in md_rep
    assert "Executive Setup Summary" in md_rep
    assert "Reasoning & Rationale" in md_rep
    assert "Rejected Alternatives" in md_rep
