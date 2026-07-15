"""
Adaptive Conformal Inference (ACI) for non-exchangeable streaming data.
"""

class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference to maintain valid coverage guarantees 
    on non-exchangeable or drifting streaming data.
    """
    def __init__(self, alpha: float = 0.05, eta: float = 0.01, initial_gamma: float = 0.0):
        self.alpha = alpha
        self.eta = eta
        self.gamma = initial_gamma
    
    def update(self, was_covered: bool) -> float:
        """
        Update the gamma threshold parameter based on whether the last 
        observation was covered by the prediction set.
        """
        # If not covered, error indicator is 1
        error = 1.0 if not was_covered else 0.0
        self.gamma += self.eta * (self.alpha - error)
        return self.gamma
