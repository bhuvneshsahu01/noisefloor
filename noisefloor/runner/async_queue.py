"""
Async Queue Manager with AIMD (Additive-Increase/Multiplicative-Decrease) Rate Limiting.
"""
import asyncio
import time

class AsyncQueueRunner:
    """
    Manages parallel async execution of evaluation cases with dynamic 
    rate limit throttling via AIMD.
    """
    def __init__(self, sprt_gate, max_concurrency: int = 20):
        self.sprt_gate = sprt_gate
        self.max_concurrency = max_concurrency
        self.current_concurrency = 1
        self.active_tasks = set()
        
    async def run_stream(self, test_items, eval_coro):
        """
        Stream items into the async evaluator, updating the SPRT gate,
        and dynamically adjusting concurrency on rate limits.
        """
        queue = asyncio.Queue()
        for item in test_items:
            queue.put_nowait(item)
            
        verdict = "CONTINUE"
        
        async def worker():
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
                    score = await eval_coro(item)
                    
                    # Additive increase
                    if self.current_concurrency < self.max_concurrency:
                        self.current_concurrency = min(self.max_concurrency, self.current_concurrency + 1)
                        
                    # Feed to SPRT
                    if verdict == "CONTINUE":
                        decision = self.sprt_gate.update(score)
                        verdict = decision.decision
                        
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        # Multiplicative decrease
                        self.current_concurrency = max(1, self.current_concurrency // 2)
                        queue.put_nowait(item)  # Reschedule
                        await asyncio.sleep(2.0)  # Backoff
                    else:
                        raise e
                finally:
                    queue.task_done()
                    
        # Launch workers based on dynamic concurrency
        while not queue.empty() and verdict == "CONTINUE":
            while len(self.active_tasks) < self.current_concurrency and not queue.empty():
                task = asyncio.create_task(worker())
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
            
            # Allow event loop to process
            await asyncio.sleep(0.1)
            
        # Cancel remaining if verdict reached
        for task in self.active_tasks:
            task.cancel()
            
        return verdict
