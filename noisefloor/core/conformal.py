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


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive Conformal Predictor (Rolling Calibration Window).
    Manages distribution drift in production by keeping a rolling FIFO window 
    of calibration scores and dynamically updating the conformal bounds.
    """
    def __init__(self, alpha: float = 0.05, max_window_size: int = 500, learning_rate: float = 0.01):
        super().__init__(alpha)
        self.max_window_size = max_window_size
        self.learning_rate = learning_rate
        self.window: List[float] = []
        self.target_alpha = alpha
        
    def calibrate(self, non_conformity_scores: List[float]) -> None:
        """Initialize the rolling window with baseline scores."""
        self.window = list(non_conformity_scores)[-self.max_window_size:]
        super().calibrate(self.window)
        
    def update_calibration(self, new_score: float, was_correct: bool) -> None:
        """
        Dynamically adjusts the calibration boundary after receiving ground truth feedback.
        Uses a learning rate on the target alpha error rate to adjust q_hat dynamically,
        alongside maintaining the FIFO rolling window.
        """
        # 1. Update rolling FIFO window
        self.window.append(new_score)
        if len(self.window) > self.max_window_size:
            self.window.pop(0)
            
        # 2. Adjust alpha based on error feedback
        # Error signal is 1 if incorrect (out of bounds), 0 if correct
        error_signal = 0.0 if was_correct else 1.0
        self.alpha = self.alpha + self.learning_rate * (self.target_alpha - error_signal)
        self.alpha = min(max(self.alpha, 0.01), 0.50)
        
        # 3. Re-calibrate using the rolling window and adjusted alpha
        super().calibrate(self.window)

