"""
Conformal LLM Judge Calibrator.
"""
import numpy as np

class ConformalJudge:
    """
    Wraps an LLM judge to output calibrated certainty ratings using 
    Conformal Prediction. If a score is within the conformal uncertainty 
    boundary, it is labeled ambiguous and its SPRT weight is downgraded.
    """
    def __init__(self, calibration_scores: list[float], alpha: float = 0.10):
        self.alpha = alpha
        self.non_conformity_scores = np.abs(np.array(calibration_scores) - 0.5) # Example heuristic
        
        n = len(self.non_conformity_scores)
        if n == 0:
            self.q_hat = 0.0
        else:
            # Conformal quantile
            val = np.ceil((n + 1) * (1 - alpha)) / n
            val = min(1.0, max(0.0, val))
            self.q_hat = np.quantile(self.non_conformity_scores, val, method='higher')
            
    def calibrate(self, llm_score: float) -> dict:
        """
        Evaluate an LLM score and determine if it's statistically ambiguous.
        """
        # Distance from certainty (0.0 or 1.0)
        uncertainty = 0.5 - abs(llm_score - 0.5)
        
        is_ambiguous = uncertainty > self.q_hat
        
        # Downgrade weight for SPRT if ambiguous
        weight = 0.5 if is_ambiguous else 1.0
        
        return {
            "score": llm_score,
            "is_ambiguous": is_ambiguous,
            "weight": weight,
            "interval": [max(0.0, llm_score - self.q_hat), min(1.0, llm_score + self.q_hat)]
        }
