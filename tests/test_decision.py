import pytest
from stonks.decision.decision_engine import DecisionEngine
from stonks.ai_layer.explainer import RuleBasedExplainer, LLMExplainer

def test_decision_engine_rules():
    """Verify that predictions resolve correctly against parameterized buy/sell boundaries."""
    engine = DecisionEngine(buy_threshold=0.65, sell_threshold=0.40)
    
    # Test BUY trigger
    buy_res = engine.make_decision(0.72)
    assert buy_res["signal"] == "BUY"
    assert buy_res["confidence"] == 0.72
    
    # Test SELL trigger
    sell_res = engine.make_decision(0.35)
    assert sell_res["signal"] == "SELL"
    assert sell_res["confidence"] == 0.35
    
    # Test HOLD trigger
    hold_res = engine.make_decision(0.50)
    assert hold_res["signal"] == "HOLD"
    assert hold_res["confidence"] == 0.50
    
    # Verify out-of-bounds inputs raise ValueError
    with pytest.raises(ValueError):
        engine.make_decision(-0.10)
    with pytest.raises(ValueError):
        engine.make_decision(1.05)

def test_rule_based_explainer():
    """Verify that RuleBasedExplainer successfully evaluates and reports technical indicator states."""
    explainer = RuleBasedExplainer()
    
    # Mock features representing bullish momentum and oversold bounce
    mock_features = {
        "rsi": 25.5,
        "macd": 0.5,
        "macd_signal": 0.2,
        "daily_return": 0.015,
        "return_20d": 0.07
    }
    
    explanation = explainer.explain("BUY", 0.72, mock_features)
    
    # Verify indicator strings exist in explanation
    assert "BUY" in explanation
    assert "72.00%" in explanation
    assert "RSI is oversold at 25.50" in explanation
    assert "MACD shows bullish crossover" in explanation
    assert "Strong daily positive return of 1.50%" in explanation
    assert "Bullish 20-day trend" in explanation

def test_llm_explainer_stub():
    """Verify that the future LLMExplainer stub returns expected placeholder text without crash."""
    explainer = LLMExplainer()
    explanation = explainer.explain("SELL", 0.35, {})
    
    assert "[Stub LLM Explanation]" in explanation
    assert "SELL" in explanation
    assert "35.00%" in explanation
