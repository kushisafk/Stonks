from abc import ABC, abstractmethod
from typing import Dict, Any
from src.logging.logger import logger

class BaseExplainer(ABC):
    """Abstract Base Class for generating natural language explanations for trading decisions."""
    
    @abstractmethod
    def explain(self, signal: str, confidence: float, features: Dict[str, Any]) -> str:
        """
        Generates an explanation of a trading signal based on decision parameters and features.
        
        Args:
            signal: Trading signal (BUY, SELL, HOLD)
            confidence: Decision confidence level
            features: Dictionary containing current feature values
            
        Returns:
            str: Natural language explanation of the decision
        """
        pass

class RuleBasedExplainer(BaseExplainer):
    """Generates explanations using standard quant rules and feature value boundaries."""
    
    def explain(self, signal: str, confidence: float, features: Dict[str, Any]) -> str:
        if not features:
            return f"Decision signal is {signal} with a confidence of {confidence:.2%}. Technical features are unavailable."
            
        reasons = []
        rsi = features.get("rsi")
        macd = features.get("macd")
        macd_signal = features.get("macd_signal")
        daily_return = features.get("daily_return")
        return_20d = features.get("return_20d")
        
        # 1. Analyze RSI
        if rsi is not None:
            if rsi < 30.0:
                reasons.append(f"RSI is oversold at {rsi:.2f}")
            elif rsi > 70.0:
                reasons.append(f"RSI is overbought at {rsi:.2f}")
            else:
                reasons.append(f"RSI is neutral at {rsi:.2f}")
                
        # 2. Analyze MACD
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                reasons.append("MACD shows bullish crossover above signal line")
            else:
                reasons.append("MACD shows bearish crossover below signal line")
                
        # 3. Analyze short-term returns momentum
        if daily_return is not None:
            if daily_return > 0.01:
                reasons.append(f"Strong daily positive return of {daily_return:.2%}")
            elif daily_return < -0.01:
                reasons.append(f"Strong daily negative return of {daily_return:.2%}")
                
        # 4. Analyze longer term 20-day trend
        if return_20d is not None:
            if return_20d > 0.05:
                reasons.append(f"Bullish 20-day trend with a {return_20d:.2%} gain")
            elif return_20d < -0.05:
                reasons.append(f"Bearish 20-day trend with a {return_20d:.2%} drop")
                
        # 5. Analyze market regime
        market_regime = features.get("market_regime")
        if market_regime is not None:
            if market_regime == 1.0:
                reasons.append("broader market is in a bullish regime (SPY in strong uptrend)")
            elif market_regime == -1.0:
                reasons.append("broader market is in a bearish regime (SPY in downtrend)")
            else:
                reasons.append("broader market is consolidating sideways")

        # 6. Analyze relative strength vs S&P 500 (SPY)
        rel_strength_20d = features.get("relative_strength_20d")
        if rel_strength_20d is not None:
            if rel_strength_20d > 0.02:
                reasons.append(f"outperforming SPY by {rel_strength_20d:.2%} over the last 20 days (strong relative strength)")
            elif rel_strength_20d < -0.02:
                reasons.append(f"underperforming SPY by {abs(rel_strength_20d):.2%} over the last 20 days (weak relative strength)")

        # 7. Analyze volume intelligence
        vol_ratio = features.get("volume_ratio")
        abnormal_vol = features.get("abnormal_volume_flag")
        if vol_ratio is not None:
            if abnormal_vol == 1.0:
                reasons.append(f"abnormal volume breakout detected ({vol_ratio:.2f}x the 20-day average), indicating institutional activity")
            elif vol_ratio > 1.25:
                reasons.append(f"above-average volume accumulation ({vol_ratio:.2f}x of 20-day SMA)")
            elif vol_ratio < 0.75:
                reasons.append(f"below-average trading volume ({vol_ratio:.2f}x of 20-day SMA)")

        # 8. Analyze news sentiment
        sentiment_score = features.get("sentiment_score")
        article_count = features.get("article_count")
        if sentiment_score is not None and article_count is not None and article_count > 0.0:
            if sentiment_score > 0.15:
                reasons.append(f"recent news sentiment is strongly positive (score: {sentiment_score:.2f} across {int(article_count)} articles)")
            elif sentiment_score < -0.15:
                reasons.append(f"recent news sentiment is strongly negative (score: {sentiment_score:.2f} across {int(article_count)} articles)")
            else:
                reasons.append(f"recent news sentiment is neutral (score: {sentiment_score:.2f} across {int(article_count)} articles)")
                
        reasons_str = ", ".join(reasons)
        explanation = f"{signal} signal generated with {confidence:.2%} confidence due to: {reasons_str}."
        return explanation

class LLMExplainer(BaseExplainer):
    """Stub implementation of LLM/LangChain-based explanation generation."""
    
    def explain(self, signal: str, confidence: float, features: Dict[str, Any]) -> str:
        logger.warning("LLMExplainer.explain: LLM explanation is not active. Stub returns mock response.")
        return f"[Stub LLM Explanation] {signal} signal backed by {confidence:.2%} confidence. Positive indicators suggest asset accumulation."
