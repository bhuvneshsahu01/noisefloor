"""
Interfaces for the async runner architecture.
"""
from typing import Protocol, Any, Awaitable, TypeVar

T = TypeVar('T')

class Evaluator(Protocol):
    """Protocol defining how an evaluation should be executed."""
    async def evaluate(self, item: Any) -> float:
        ...

class RateLimiter(Protocol):
    """Protocol for concurrency and rate limit management."""
    
    @property
    def current_concurrency(self) -> int:
        ...
        
    def on_success(self) -> None:
        """Called when an evaluation succeeds to potentially scale up concurrency."""
        ...
        
    def on_rate_limit(self) -> float:
        """
        Called when a rate limit is hit to scale down concurrency.
        Returns the required backoff time in seconds.
        """
        ...
