from typing import List
import numpy as np

class OnlineCalibrator:
    """
    Implements a self-tuning feedback loop for Conformal Prediction bounds.
    Adjusts the expected error rate (alpha) or dynamically recalibrates 
    the base non-conformity scores based on user feedback.
    """
    def __init__(self, initial_alpha: float = 0.10, learning_rate: float = 0.01):
        self.current_alpha = initial_alpha
        self.learning_rate = learning_rate
        self.feedback_history = []
        
    def log_feedback(self, was_false_positive: bool):
        """
        Takes human or secondary-judge feedback.
        If it was a false positive (system blocked a safe action), the bound was too strict.
        If it was a false negative (system allowed an unsafe action), the bound was too loose.
        """
        self.feedback_history.append(was_false_positive)
        
        if len(self.feedback_history) > 100:
            self.feedback_history.pop(0)
            
        # Very simple online adjustment heuristic
        if was_false_positive:
            # System was too strict, slightly increase alpha (allow more errors)
            # to widen the threshold and reduce false positive rate.
            self.current_alpha = min(0.99, self.current_alpha + self.learning_rate)
        else:
            # System was too loose (false negative), decrease alpha (allow fewer errors)
            # to tighten the threshold and increase safety.
            self.current_alpha = max(0.01, self.current_alpha - self.learning_rate)
            
        return self.current_alpha

# Global calibrator
online_calibrator = OnlineCalibrator()
