from typing import Dict, Any, List

class MarketReasoner:
    """Evaluates stock technical/fundamental context relative to index regimes and volume signals to decide if prediction is trusted."""
    
    def evaluate_market(self, signal: str, confidence: float, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes prediction context to determine trend confirmation or divergence.
        
        Args:
            signal: Raw signal from decision engine (BUY, SELL, HOLD)
            confidence: Raw prediction probability / confidence
            features: Dictionary containing current feature values
            
        Returns:
            Dict containing:
                is_trusted: bool (if prediction is supported by market context)
                warnings: List[str] (contextual warnings detected)
                confirmations: List[str] (supporting indicators detected)
        """
        warnings = []
        confirmations = []
        is_trusted = True
        
        market_regime = features.get("market_regime", 0.0)
        spy_trend = features.get("spy_trend_strength", 0.0)
        rel_strength = features.get("relative_strength_20d", 0.0)
        rel_momentum = features.get("relative_momentum_score", 0.0)
        vol_ratio = features.get("volume_ratio", 1.0)
        abnormal_vol = features.get("abnormal_volume_flag", 0.0)
        sentiment = features.get("sentiment_score", 0.0)
        rsi = features.get("rsi", 50.0)
        
        # 1. Market Regime vs Signal Alignment
        if signal == "BUY":
            if market_regime == -1.0:
                warnings.append("Prediction is BUY but broader S&P 500 regime is Bearish (SPY is in a downtrend).")
                is_trusted = False
            elif market_regime == 1.0:
                confirmations.append("Bullish prediction is supported by a Bullish broader S&P 500 regime.")
                
        elif signal == "SELL":
            if market_regime == 1.0:
                warnings.append("Prediction is SELL but broader S&P 500 regime is Bullish (SPY is in a strong uptrend).")
                is_trusted = False
            elif market_regime == -1.0:
                confirmations.append("Bearish prediction is supported by a Bearish broader S&P 500 regime.")
                
        # 2. Relative Strength Alignment
        if signal == "BUY" and rel_strength < -0.02:
            warnings.append(f"Stock is underperforming the S&P 500 (relative_strength_20d = {rel_strength:.2%}), indicating weak relative momentum.")
        elif signal == "BUY" and rel_strength > 0.02:
            confirmations.append(f"Stock shows strong relative outperformance vs S&P 500 (relative_strength_20d = {rel_strength:.2%}).")
            
        if signal == "SELL" and rel_strength > 0.02:
            warnings.append(f"Stock is outperforming the S&P 500 (relative_strength_20d = {rel_strength:.2%}), indicating strong counter-trend relative strength.")
        elif signal == "SELL" and rel_strength < -0.02:
            confirmations.append(f"Stock shows relative underperformance vs S&P 500 (relative_strength_20d = {rel_strength:.2%}), confirming bearish momentum.")

        # 3. Volume Breakouts
        if signal == "BUY":
            if vol_ratio < 0.80:
                warnings.append(f"Bullish trend lacks volume support (volume_ratio = {vol_ratio:.2f}x of 20-day SMA).")
            elif abnormal_vol == 1.0:
                confirmations.append(f"Abnormal volume breakout confirmed ({vol_ratio:.2f}x of 20-day SMA), indicating institutional accumulation.")
                
        # 4. Sentiment Alignment
        if signal == "BUY" and sentiment < -0.15:
            warnings.append(f"Bullish signal conflicted by negative news sentiment (sentiment_score = {sentiment:.2f}).")
        elif signal == "BUY" and sentiment > 0.15:
            confirmations.append("Recent news sentiment is supportive and positive.")
            
        if signal == "SELL" and sentiment > 0.15:
            warnings.append(f"Bearish signal conflicted by positive news sentiment (sentiment_score = {sentiment:.2f}).")
        elif signal == "SELL" and sentiment < -0.15:
            confirmations.append("Recent news sentiment confirms bearish outlook.")
            
        # 5. Technical Overextensions
        if signal == "BUY" and rsi > 70.0:
            warnings.append(f"Stock is technically overbought (RSI = {rsi:.1f}). Buying now carries elevated correction risk.")
        if signal == "SELL" and rsi < 30.0:
            warnings.append(f"Stock is technically oversold (RSI = {rsi:.1f}). Selling now carries short squeeze risk.")
            
        return {
            "is_trusted": is_trusted,
            "warnings": warnings,
            "confirmations": confirmations
        }
