"""
Custom exceptions for the risklayer library.
"""

class risklayerError(Exception):
    """Base exception for all risklayer errors."""
    pass

class OutOfBudgetError(risklayerError):
    """Raised when an evaluator exhausts its token/time budget before concluding."""
    pass


class ConformalRiskException(risklayerError):
    """
    Raised when an autonomous agent action breaches its formal 
    conformal prediction uncertainty bounds.
    """
    pass

class ConfigurationError(risklayerError):
    """Raised when configuration parameters are invalid."""
    pass

class EvaluationError(risklayerError):
    """Raised when an evaluation fails (e.g., malformed score)."""
    pass

class RateLimitExceededError(risklayerError):
    """Raised when a rate limit from an external service is encountered."""
    
    def __init__(self, message: str, retry_after: float = 2.0):
        super().__init__(message)
        self.retry_after = retry_after

class StatisticalConvergenceError(risklayerError):
    """Raised when an algorithm fails to converge statistically."""
    pass
