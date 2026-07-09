from typing import Dict, Any
from stonks.config.settings import settings
from stonks.logging.logger import logger

class DecisionEngine:
    """Evaluates probability predictions against parameterized thresholds to generate trading signals."""
    
    def __init__(self, buy_threshold: Optional[float] = None, sell_threshold: Optional[float] = None):
        self.buy_threshold = buy_threshold if buy_threshold is not None else settings.BUY_THRESHOLD
        self.sell_threshold = sell_threshold if sell_threshold is not None else settings.SELL_THRESHOLD
        
    def make_decision(self, probability: float) -> Dict[str, Any]:
        """
        Generates a trading signal based on model prediction confidence.
        
        Args:
            probability: Prediction probability of price increasing [0.0, 1.0]
            
        Returns:
            Dict[str, Any]: Dict containing 'signal' (BUY, SELL, HOLD) and 'confidence' (float)
        """
        # Force cast and check
        probability = float(probability)
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"DecisionEngine: Confidence probability {probability} is out of bounds [0.0, 1.0].")
            
        if probability > self.buy_threshold:
            signal = "BUY"
        elif probability < self.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
            
        logger.info(f"Decision Engine: Prob: {probability:.4f} -> Signal: {signal}")
        
        return {
            "signal": signal,
            "confidence": probability
        }

# Import typing helper
from typing import Optional

# Global decision engine instance
decision_engine = DecisionEngine()
