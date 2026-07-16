from risklayer.integrations.tiered_guard import CascadingGuard
from risklayer.core.conformal import ConformalPredictor
import re

class SecurityGuard(CascadingGuard):
    """
    Pre-configured Guardrail for detecting Prompt Injections and Jailbreaks.
    """
    def __init__(self, base_alpha: float = 0.05): # Tighter bound for security
        predictor = ConformalPredictor(alpha=base_alpha)
        # Calibrate on internal dataset of known jailbreaks (high error/risk scores)
        predictor.calibrate([0.2, 0.4, 0.6, 0.8, 0.95])
        super().__init__(predictor=predictor)

    def security_score_fn(self, prompt: str) -> float:
        """
        Fast heuristic mapping to a conformal score.
        """
        risk = 0.1
        
        # Simple regex tier
        suspicious_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"you are now",
            r"DAN",
            r"bypass"
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                risk += 0.4
                
        return min(risk, 0.99)
        
    def __call__(self):
        return self.guard(score_fn=self.security_score_fn)

# Singleton for easy import
security_guard = SecurityGuard()
