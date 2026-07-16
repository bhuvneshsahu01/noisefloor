"""
Rate Limiter implementations.
"""
from ..config import AIMDRateLimitSettings
from .interfaces import RateLimiter
from ..log import get_logger

logger = get_logger(__name__)

class AIMDRateLimiter:
    """
    Additive-Increase / Multiplicative-Decrease Rate Limiter.
    """
    def __init__(self, settings: AIMDRateLimitSettings):
        self.settings = settings
        self._concurrency = settings.initial_concurrency
        
    @property
    def current_concurrency(self) -> int:
        return self._concurrency
        
    def on_success(self) -> None:
        if self._concurrency < self.settings.max_concurrency:
            self._concurrency = min(
                self.settings.max_concurrency, 
                self._concurrency + self.settings.increase_factor
            )
            
    def on_rate_limit(self) -> float:
        old_concurrency = self._concurrency
        self._concurrency = max(1, int(self._concurrency * self.settings.decrease_factor))
        logger.warning(
            "Rate limit hit. Scaled down concurrency.", 
            old_concurrency=old_concurrency, 
            new_concurrency=self._concurrency,
            backoff=self.settings.backoff_seconds
        )
        return self.settings.backoff_seconds
