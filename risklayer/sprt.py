"""
Wald's Sequential Probability Ratio Test (SPRT).
"""
import numpy as np
from .core.verdicts import SPRTDecision

class SPRTGate:
    def __init__(
        self,
        h0_rate: float,
        h1_rate: float,
        alpha: float = 0.05,
        beta: float = 0.20,
        max_samples: int = 500
    ):
        self.p0 = h0_rate
        self.p1 = h1_rate
        self.alpha = alpha
        self.beta = beta
        self.max_samples = max_samples
        
        self.a = np.log(self.beta / (1 - self.alpha))
        self.b = np.log((1 - self.beta) / self.alpha)
        
        self.log_ratio_success = np.log(self.p1 / self.p0)
        self.log_ratio_failure = np.log((1 - self.p1) / (1 - self.p0))
        
        self.log_lambda = 0.0
        self.n_samples = 0
        self.success_count = 0
        
    def update(self, score: float) -> SPRTDecision:
        """Update the SPRT with a new observation."""
        self.n_samples += 1
        
        # Treat score as binary success/failure based on > 0.5 or strict bool
        # For a more advanced continuous implementation, normal SPRT could be used.
        is_success = score > 0.5
        
        if is_success:
            self.log_lambda += self.log_ratio_success
            self.success_count += 1
        else:
            self.log_lambda += self.log_ratio_failure
            
        decision = "CONTINUE"
        savings_pct = max(0.0, 100.0 * (self.max_samples - self.n_samples) / self.max_samples)
        
        if self.log_lambda >= self.b:
            decision = "ACCEPT_H1"
        elif self.log_lambda <= self.a:
            decision = "ACCEPT_H0"
        elif self.n_samples >= self.max_samples:
            decision = "MAX_REACHED"
            
        return SPRTDecision(
            decision=decision,
            n_samples=self.n_samples,
            log_lambda=self.log_lambda,
            lower_boundary=self.a,
            upper_boundary=self.b,
            savings_pct=savings_pct,
            cumulative_pass_rate=self.success_count / self.n_samples
        )

def sprt_gate(h0_rate: float, h1_rate: float, alpha: float = 0.05, beta: float = 0.20, max_samples: int = 500) -> SPRTGate:
    """Return an SPRTGate instance for streaming evaluations."""
    return SPRTGate(h0_rate, h1_rate, alpha, beta, max_samples)
