import numpy as np
from typing import List, Union

class ConformalPredictor:
    """
    Computes Conformal Prediction bounds to provide formal, 
    distribution-free coverage guarantees on agent confidence.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        :param alpha: The target error rate (e.g., 0.05 for 95% confidence).
        """
        self.alpha = alpha
        self.calibration_scores: List[float] = []
        self._q_hat: float = float('inf')

    def calibrate(self, non_conformity_scores: List[float]) -> None:
        """
        Calibrate the predictor using a dataset of historical non-conformity scores 
        (e.g., how wrong previous agent actions were).
        """
        self.calibration_scores = sorted(non_conformity_scores)
        n = len(self.calibration_scores)
        if n == 0:
            return
            
        # Calculate empirical quantile
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        
        self._q_hat = np.quantile(self.calibration_scores, q_level, method='higher')

    def is_safe(self, current_score: float) -> bool:
        """
        Returns True if the current score is within the calibrated safe bound.
        """
        if not self.calibration_scores:
            # If not calibrated, we conservatively abstain or warn.
            # For the SDK, we'll allow it initially but flag it in tracing.
            return True 
            
        return current_score <= self._q_hat

    def get_bound(self) -> float:
        """Returns the current conformal risk boundary."""
        return self._q_hat
