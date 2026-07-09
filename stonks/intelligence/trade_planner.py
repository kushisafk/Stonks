from typing import Dict, Any, List

class TradePlanner:
    """Maps prediction outcomes and risk constraints to multi-level, non-binary suggested trading actions (e.g. TAKE PROFITS)."""
    
    def plan_trade(
        self, 
        signal: str, 
        confidence_tier: str, 
        risk_tier: str, 
        is_trusted: bool, 
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Derives suggested plan action and alternative rejections.
        
        Args:
            signal: Raw signal (BUY, SELL, HOLD)
            confidence_tier: Qualitative confidence tier
            risk_tier: Qualitative risk tier
            is_trusted: Trustworthiness from MarketReasoner
            features: Current feature values
            
        Returns:
            Dict containing:
                action: str (BUY, BUY SMALL POSITION, WAIT, HOLD, TAKE PROFITS, SELL, STRONG SELL)
                rejected_actions: List[str] (Actions considered but discarded)
                rationale: str (Reason for choosing this plan)
        """
        rsi = features.get("rsi", 50.0)
        return_5d = features.get("return_5d", 0.0)
        
        action = "HOLD"
        rejected_actions = []
        rationale = ""
        
        # 1. Take Profit Check (RSI is extremely overbought)
        if rsi >= 75.0 and return_5d > 0.05:
            action = "TAKE PROFITS"
            rejected_actions = ["BUY", "BUY SMALL POSITION", "HOLD"]
            rationale = f"Stock is in extreme overbought territory (RSI = {rsi:.1f}) with 5-day gains of {return_5d:.2%}. Booking profits is recommended over buying or holding."
            return {
                "action": action,
                "rejected_actions": rejected_actions,
                "rationale": rationale
            }
            
        # 2. Bullish Signal Logic
        if signal == "BUY":
            rejected_actions = ["SELL", "STRONG SELL"]
            if not is_trusted:
                if risk_tier == "High Risk":
                    action = "WAIT"
                    rejected_actions.extend(["BUY", "BUY SMALL POSITION"])
                    rationale = "Bullish prediction is untrusted and carries High Risk due to strong counter-trend market context. Suggest waiting for macro confirmation."
                else:
                    action = "BUY SMALL POSITION"
                    rejected_actions.extend(["BUY", "WAIT"])
                    rationale = "Bullish prediction is untrusted due to trend mismatches, but lower risk tier warrants a defensive mini-exposure."
            else:
                if risk_tier == "High Risk":
                    action = "BUY SMALL POSITION"
                    rejected_actions.extend(["BUY", "WAIT"])
                    rationale = "Bullish prediction is trusted, but high volatility or risk requires scaling down to a small entry position."
                elif confidence_tier == "Low":
                    action = "BUY SMALL POSITION"
                    rejected_actions.extend(["BUY"])
                    rationale = "Bullish prediction has low conviction. A reduced position size is recommended."
                else:
                    action = "BUY"
                    rejected_actions.extend(["BUY SMALL POSITION", "WAIT"])
                    rationale = "Bullish prediction is fully trusted with high confidence and acceptable risk. Full buy position supported."
                    
        # 3. Bearish Signal Logic
        elif signal == "SELL":
            rejected_actions = ["BUY", "BUY SMALL POSITION"]
            if not is_trusted:
                if risk_tier == "High Risk":
                    action = "WAIT"
                    rejected_actions.extend(["SELL", "STRONG SELL"])
                    rationale = "Bearish prediction contradicts strong bullish market trend. High risk suggests waiting rather than shorting."
                else:
                    action = "HOLD"
                    rejected_actions.extend(["SELL", "STRONG SELL"])
                    rationale = "Bearish prediction lacks confirmation under a bullish regime. Hold current positions and defer selling."
            else:
                market_regime = features.get("market_regime", 0.0)
                if confidence_tier in ("Very High", "High") and market_regime == -1.0:
                    action = "STRONG SELL"
                    rejected_actions.extend(["SELL", "HOLD"])
                    rationale = "Bearish prediction is high conviction and aligned with a macro market downturn. Liquidation or shorting recommended."
                else:
                    action = "SELL"
                    rejected_actions.extend(["STRONG SELL", "HOLD"])
                    rationale = "Bearish prediction is confirmed. Exiting long exposure is suggested."
                    
        # 4. Neutral Signal Logic
        else:
            rejected_actions = ["BUY", "SELL", "STRONG SELL"]
            if rsi < 25.0:
                action = "BUY SMALL POSITION"
                rejected_actions.extend(["HOLD", "WAIT"])
                rationale = f"Ensemble is neutral, but the stock is deeply oversold (RSI = {rsi:.1f}). Suggesting a small mean-reversion buy position."
            else:
                action = "HOLD"
                rejected_actions.extend(["BUY SMALL POSITION", "WAIT"])
                rationale = "Ensemble prediction is neutral and stock is within normal technical boundaries. Maintain holding."
                
        return {
            "action": action,
            "rejected_actions": rejected_actions,
            "rationale": rationale
        }
