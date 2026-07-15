"""
Custom exceptions for the Noisefloor library.
"""

class NoisefloorError(Exception):
    """Base exception for all Noisefloor errors."""
    pass

class OutOfBudgetError(NoisefloorError):
    """Raised when an evaluator exhausts its token/time budget before concluding."""
    pass


class ConformalRiskException(NoisefloorError):
    """
    Raised when an autonomous agent action breaches its formal 
    conformal prediction uncertainty bounds.
    """
    pass

class ConfigurationError(NoisefloorError):
    """Raised when configuration parameters are invalid."""
    pass

class EvaluationError(NoisefloorError):
    """Raised when an evaluation fails (e.g., malformed score)."""
    pass

class RateLimitExceededError(NoisefloorError):
    """Raised when a rate limit from an external service is encountered."""
    
    def __init__(self, message: str, retry_after: float = 2.0):
        super().__init__(message)
        self.retry_after = retry_after

class StatisticalConvergenceError(NoisefloorError):
    """Raised when an algorithm fails to converge statistically."""
    pass
