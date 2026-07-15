"""
Async Queue Manager using EvaluatorPool pattern.
"""
import asyncio
from typing import Iterable, Any, List

from .interfaces import Evaluator, RateLimiter
from ..sprt import SPRTGate
from ..exceptions import RateLimitExceededError
from ..log import get_logger

logger = get_logger(__name__)

class EvaluatorPool:
    """
    Manages parallel async execution of evaluation cases with dynamic 
    rate limit throttling via a provided RateLimiter.
    """
    def __init__(self, sprt_gate: SPRTGate, rate_limiter: RateLimiter, evaluator: Evaluator):
        self.sprt_gate = sprt_gate
        self.rate_limiter = rate_limiter
        self.evaluator = evaluator
        self.active_tasks: set[asyncio.Task[Any]] = set()
        
    async def run_stream(self, test_items: Iterable[Any]) -> str:
        """
        Stream items into the async evaluator, updating the SPRT gate,
        and dynamically adjusting concurrency on rate limits.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()
        for item in test_items:
            queue.put_nowait(item)
            
        verdict = "CONTINUE"
        
        async def worker() -> None:
            nonlocal verdict
            while True:
                if verdict != "CONTINUE":
                    break
                
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
                try:
                    # Run evaluation
                    score = await self.evaluator.evaluate(item)
                    self.rate_limiter.on_success()
                    
                    # Feed to SPRT
                    if verdict == "CONTINUE":
                        decision = self.sprt_gate.update(score)
                        verdict = decision.decision
                        
                        logger.info(
                            "Evaluated item", 
                            score=score, 
                            sprt_decision=verdict, 
                            concurrency=self.rate_limiter.current_concurrency
                        )
                        
                except Exception as e:
                    # Use a custom exception or check specifically
                    if isinstance(e, RateLimitExceededError) or "429" in str(e):
                        backoff = self.rate_limiter.on_rate_limit()
                        queue.put_nowait(item)  # Reschedule the item
                        await asyncio.sleep(backoff)
                    else:
                        logger.error("Evaluation failed with unexpected error.", exc_info=e)
                        raise
                finally:
                    queue.task_done()
                    
        # Launch workers based on dynamic concurrency
        while not queue.empty() and verdict == "CONTINUE":
            while len(self.active_tasks) < self.rate_limiter.current_concurrency and not queue.empty():
                task = asyncio.create_task(worker())
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
            
            # Allow event loop to yield to workers
            await asyncio.sleep(0.05)
            
        # Cancel remaining tasks if verdict reached or queue empty
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
                
        # Wait for cancellations to propagate
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            
        logger.info("Evaluation stream finished.", final_verdict=verdict)
        return verdict
