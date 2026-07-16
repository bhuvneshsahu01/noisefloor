import logging
from typing import AsyncGenerator, Callable, Any
from risklayer.core.conformal import ConformalPredictor
from risklayer.exceptions import ConformalRiskException
from risklayer.telemetry import get_tracer

logger = logging.getLogger("risklayer.judges.stream")
tracer = get_tracer("risklayer.stream_guard")

class StreamGuard:
    """
    Early-Exit interceptor for LLM streaming responses.
    Evaluates chunks dynamically and aborts the stream if the risk boundary is breached,
    saving token costs and preventing malicious/hallucinated outputs from completing.
    """
    
    def __init__(self, predictor: ConformalPredictor, chunk_size: int = 50, shadow_mode: bool = False):
        """
        :param predictor: ConformalPredictor instance for bound checking.
        :param chunk_size: Number of characters to accumulate before triggering an evaluation.
        :param shadow_mode: If true, logs the breach but allows the stream to continue.
        """
        self.predictor = predictor
        self.chunk_size = chunk_size
        self.shadow_mode = shadow_mode
        
    async def intercept(self, stream: AsyncGenerator[str, None], score_fn: Callable[[str], float]) -> AsyncGenerator[str, None]:
        """
        Wraps an async text generator. Yields tokens while they are safe.
        Aborts if the accumulated text breaches conformal risk bounds.
        """
        buffer = ""
        evaluated_buffer = ""
        step = 1
        
        with tracer.start_as_current_span("stream_guard.intercept") as span:
            try:
                async for chunk in stream:
                    buffer += chunk
                    
                    # If we have accumulated enough new characters, run a safety check
                    if len(buffer) - len(evaluated_buffer) >= self.chunk_size:
                        # Evaluate the current buffer
                        current_score = score_fn(buffer)
                        bound = self.predictor.get_adjusted_bound(step)
                        
                        if not self.predictor.is_safe(current_score, step):
                            if self.shadow_mode:
                                logger.warning(f"SHADOW BLOCKED STREAM: Step {step} | Score {current_score:.3f} > {bound:.3f}. Continuing...")
                            else:
                                logger.error(f"STREAM ABORTED: Step {step} | Risk score {current_score:.3f} > Conformal bound {bound:.3f}!")
                                span.set_attribute("stream.aborted", True)
                                span.set_attribute("stream.abort_step", step)
                                raise ConformalRiskException(f"Stream aborted at step {step} due to conformal risk breach.")
                        
                        evaluated_buffer = buffer
                        step += 1
                    
                    # Yield the chunk safely to the consumer
                    yield chunk
                    
            except Exception as e:
                span.record_exception(e)
                raise
