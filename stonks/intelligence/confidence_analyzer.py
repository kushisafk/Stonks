from typing import Dict, Any, List

class ConfidenceAnalyzer:
    """Evaluates prediction probabilities and contextual validations to assign qualitative confidence tiers and explain updates."""
    
    def analyze_confidence(self, probability: float, confirmations: List[str], warnings: List[str]) -> Dict[str, Any]:
        """
        Derives qualitative confidence levels based on prediction convergence and contextual support.
        
        Args:
            probability: Prediction probability of class 1 (price increase)
            confirmations: Confirming market signals resolved by MarketReasoner
            warnings: Warning market signals resolved by MarketReasoner
            
        Returns:
            Dict containing:
                tier: str (Very High, High, Medium, Low)
                rationale: str (Explanation of confidence assignment)
        """
        # Distance from neutral probability (0.50) represents prediction strength
        pred_strength = abs(probability - 0.50)
        
        # Determine base tier from raw probability strength
        if pred_strength >= 0.25:  # prob >= 0.75 or prob <= 0.25 (Strong signal)
            if len(warnings) == 0 and len(confirmations) >= 2:
                tier = "Very High"
                rationale = f"Ensemble prediction is highly conviction-driven ({probability:.1%}), fully confirmed by clean market and volume trends."
            elif len(warnings) > 0:
                tier = "Medium"
                rationale = f"Ensemble prediction has strong raw probability ({probability:.1%}), but is downgraded to Medium due to active context conflicts: {'; '.join(warnings)}"
            else:
                tier = "High"
                rationale = f"Ensemble prediction has strong conviction ({probability:.1%}) with standard confirmation and no context conflicts."
                
        elif pred_strength >= 0.15:  # prob >= 0.65 or prob <= 0.35 (Moderate signal)
            if len(warnings) >= 2:
                tier = "Low"
                rationale = f"Moderate prediction signal ({probability:.1%}) is downgraded to Low due to multiple severe context conflicts."
            elif len(warnings) == 0:
                tier = "High"
                rationale = f"Moderate prediction signal ({probability:.1%}) is upgraded to High because market context is clean and offers zero warnings."
            else:
                tier = "Medium"
                rationale = f"Moderate prediction signal ({probability:.1%}) is supported by a stable, though partially mixed, market context."
                
        else:  # Neutral range (0.35 < prob < 0.65)
            tier = "Low"
            if len(confirmations) >= 2 and len(warnings) == 0:
                tier = "Medium"
                rationale = f"Raw prediction is in the neutral zone ({probability:.1%}), but has been upgraded to Medium due to strong supporting macro/volume confirmation."
            else:
                rationale = f"Prediction probability ({probability:.1%}) is near the neutral 50% boundary, indicating low directional conviction."
                
        return {
            "tier": tier,
            "rationale": rationale
        }
