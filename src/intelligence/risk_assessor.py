from typing import Dict, Any, List

class RiskAssessor:
    """Estimates a risk score from 0-100 and classifies it into Risk Tiers based on volatility, regimes, and sentiment."""
    
    def assess_risk(self, signal: str, confidence_tier: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantifies risk metrics across multiple trading dimensions.
        
        Args:
            signal: Trading signal (BUY, SELL, HOLD)
            confidence_tier: Qualitative confidence tier from ConfidenceAnalyzer
            features: Dictionary containing current features
            
        Returns:
            Dict containing:
                score: int (0 to 100)
                tier: str (Low Risk, Medium Risk, High Risk)
                factors: List[str] (List of specific risk factors contributing to the score)
        """
        score = 0.0
        factors = []
        
        # 1. Volatility contribution (max 45 points)
        # Standard daily stock volatility ranges 1-3%. Let's scale 2.0% daily to 30 points.
        daily_vol = features.get("volatility_20d", 0.015)
        stock_vol_contrib = min(daily_vol * 1500.0, 30.0)
        score += stock_vol_contrib
        if daily_vol > 0.025:
            factors.append(f"High asset-level volatility detected (daily volatility = {daily_vol:.2%}).")
            
        # Index volatility: scale 1.5% daily SPY to 15 points.
        daily_spy_vol = features.get("spy_volatility_20d", 0.01)
        spy_vol_contrib = min(daily_spy_vol * 1000.0, 15.0)
        score += spy_vol_contrib
        if daily_spy_vol > 0.015:
            factors.append(f"Elevated S&P 500 index volatility (daily SPY volatility = {daily_spy_vol:.2%}).")
            
        # 2. Trend Regime Mismatch contribution (max 25 points)
        regime = features.get("market_regime", 0.0)
        if signal == "BUY" and regime == -1.0:
            score += 25.0
            factors.append("Counter-trend trade: executing BUY during S&P 500 bearish regime.")
        elif signal == "SELL" and regime == 1.0:
            score += 20.0
            factors.append("Counter-trend trade: executing SELL during S&P 500 bullish regime.")
            
        # 3. Sentiment Divergence contribution (max 15 points)
        sentiment = features.get("sentiment_score", 0.0)
        if signal == "BUY" and sentiment < -0.15:
            score += 15.0
            factors.append(f"Sentiment divergence: bullish signal with negative news sentiment ({sentiment:.2f}).")
        elif signal == "SELL" and sentiment > 0.15:
            score += 10.0
            factors.append(f"Sentiment divergence: bearish signal with positive news sentiment ({sentiment:.2f}).")
            
        # 4. Confidence Tier contribution (max 15 points)
        if confidence_tier == "Low":
            score += 15.0
            factors.append("Low prediction confidence tier adds default parameter uncertainty.")
        elif confidence_tier == "Medium":
            score += 5.0
        elif confidence_tier == "Very High":
            score -= 5.0  # Volatility offset for highly confirmed trades
            
        # Ensure bounding
        score = max(0.0, min(100.0, score))
        score_int = int(round(score))
        
        # Classify Tiers
        if score_int <= 35:
            tier = "Low Risk"
        elif score_int <= 70:
            tier = "Medium Risk"
        else:
            tier = "High Risk"
            
        return {
            "score": score_int,
            "tier": tier,
            "factors": factors
        }
