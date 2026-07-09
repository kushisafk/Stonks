import json
from typing import Dict, Any
from stonks.intelligence.market_reasoner import MarketReasoner
from stonks.intelligence.confidence_analyzer import ConfidenceAnalyzer
from stonks.intelligence.risk_assessor import RiskAssessor
from stonks.intelligence.trade_planner import TradePlanner

class RecommendationEngine:
    """Coordinates the reasoning sub-modules to compile comprehensive JSON and Markdown trading recommendations."""
    
    def __init__(self):
        self.reasoner = MarketReasoner()
        self.analyzer = ConfidenceAnalyzer()
        self.risk_assessor = RiskAssessor()
        self.planner = TradePlanner()
        
    def generate_recommendation(
        self, 
        ticker: str, 
        probability: float, 
        signal: str, 
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Runs reasoning flow and consolidates decision payload.
        
        Args:
            ticker: Ticker symbol (e.g. AAPL)
            probability: Ensemble prediction probability of price increase
            signal: Raw signal (BUY, SELL, HOLD)
            features: Dictionary of current feature values
            
        Returns:
            Dict containing:
                json_report: Dict (Structured machine-readable recommendation payload)
                markdown_report: str (Premium formatted human-readable report)
        """
        ticker = ticker.strip().upper()
        
        # 1. Market reasoning
        reason_res = self.reasoner.evaluate_market(signal, probability, features)
        is_trusted = reason_res["is_trusted"]
        warnings = reason_res["warnings"]
        confirmations = reason_res["confirmations"]
        
        # 2. Confidence analysis
        conf_res = self.analyzer.analyze_confidence(probability, confirmations, warnings)
        confidence_tier = conf_res["tier"]
        confidence_rationale = conf_res["rationale"]
        
        # 3. Risk assessment
        risk_res = self.risk_assessor.assess_risk(signal, confidence_tier, features)
        risk_score = risk_res["score"]
        risk_tier = risk_res["tier"]
        risk_factors = risk_res["factors"]
        
        # 4. Action planning
        plan_res = self.planner.plan_trade(signal, confidence_tier, risk_tier, is_trusted, features)
        action = plan_res["action"]
        rejected_actions = plan_res["rejected_actions"]
        plan_rationale = plan_res["rationale"]
        
        # Build JSON Payload
        json_report = {
            "ticker": ticker,
            "prediction": signal,
            "probability": f"{probability:.2%}",
            "confidence_tier": confidence_tier,
            "confidence_rationale": confidence_rationale,
            "market_regime": "Bullish" if features.get("market_regime", 0.0) == 1.0 else ("Bearish" if features.get("market_regime", 0.0) == -1.0 else "Sideways"),
            "news_sentiment": f"{features.get('sentiment_score', 0.0):.2f}",
            "relative_strength_20d": f"{features.get('relative_strength_20d', 0.0):+.2%}",
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "risk_factors": risk_factors,
            "recommendation": action,
            "rejected_actions": rejected_actions,
            "reasoning": plan_rationale,
            "warnings": warnings,
            "confirmations": confirmations
        }
        
        # Build Markdown Report
        md = f"""## STONKS Intelligence Report: **{ticker}**

* **Recommended Action**: `{' '.join([w.upper() for w in action.split()])}`
* **Risk Profile**: **{risk_tier}** (Score: `{risk_score}/100`)
* **Confidence Level**: **{confidence_tier}**

---

### Executive Setup Summary
* **Base Prediction**: `{signal}` (Probability: `{probability:.1%}`)
* **News Sentiment Alignment**: `{features.get('sentiment_score', 0.0):.2f}`
* **Relative Strength (20d vs S&P 500)**: `{features.get('relative_strength_20d', 0.0):+.2%}`
* **Index Regime Context**: `{json_report['market_regime']}`

---

### Reasoning & Rationale
{plan_rationale}
* {confidence_rationale}

"""
        if confirmations:
            md += "#### Market Confirmations:\n"
            for c in confirmations:
                md += f"- ✓ {c}\n"
            md += "\n"
            
        if warnings:
            md += "#### Contextual Warnings:\n"
            for w in warnings:
                md += f"- ⚠ {w}\n"
            md += "\n"
            
        if risk_factors:
            md += "#### Active Risk Factors:\n"
            for rf in risk_factors:
                md += f"- 🗲 {rf}\n"
            md += "\n"
            
        if rejected_actions:
            md += "#### Rejected Alternatives:\n"
            md += f"- *Rejected actions*: {', '.join([f'`{a}`' for a in rejected_actions])}\n"
            md += "- *Rejection basis*: These options were discarded because they do not satisfy the current risk-to-reward boundaries or run counter to the prevailing market regime.\n"
            
        return {
            "json_report": json_report,
            "markdown_report": md
        }
