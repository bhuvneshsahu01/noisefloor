import logging
from typing import List, Callable, Tuple
import time
from risklayer.telemetry import get_tracer

logger = logging.getLogger("risklayer.tiered_guard")
tracer = get_tracer("risklayer.tiered_guard")

class CascadingGuard:
    """
    Implements a multi-tiered hybrid guardrail system.
    Evaluates inputs using extremely fast/cheap models first (e.g. Regex, ONNX),
    escalating to slower/expensive models (e.g. LLM Judge) only on ambiguity.
    """
    
    def __init__(self):
        self.tiers: List[Tuple[Callable[[str], float], float, float, str]] = []
        
    def add_tier(self, name: str, evaluator: Callable[[str], float], safe_threshold: float, block_threshold: float):
        """
        Adds an evaluation tier. Tiers should be added in order of ascending cost/latency.
        
        :param name: Name of the tier (e.g. 'Regex', 'ONNX', 'LLM')
        :param evaluator: Function that takes text and returns a non-conformity score (0.0 to 1.0)
        :param safe_threshold: If score <= safe_threshold, instantly accept (exit early).
        :param block_threshold: If score >= block_threshold, instantly block (exit early).
                                If safe_threshold < score < block_threshold, escalate to next tier.
        """
        self.tiers.append((evaluator, safe_threshold, block_threshold, name))
        
    def score(self, text: str, **kwargs) -> float:
        """
        Executes the cascading evaluation.
        Returns the final non-conformity score.
        """
        if not self.tiers:
            raise ValueError("No tiers configured for CascadingGuard.")
            
        with tracer.start_as_current_span("cascading_guard.score") as span:
            start_time = time.time()
            
            for i, (evaluator, safe_threshold, block_threshold, name) in enumerate(self.tiers):
                tier_start = time.time()
                
                # Execute the evaluator
                score = evaluator(text, **kwargs)
                tier_latency = (time.time() - tier_start) * 1000
                
                logger.debug(f"Tier {i+1} [{name}] evaluated in {tier_latency:.2f}ms | Score: {score:.3f}")
                
                # Check early exit conditions
                if score <= safe_threshold:
                    logger.info(f"Tier {i+1} [{name}] definitively SAFE. Exiting early.")
                    span.set_attribute("guard.exit_tier", name)
                    span.set_attribute("guard.latency_ms", (time.time() - start_time) * 1000)
                    return score
                    
                if score >= block_threshold:
                    logger.warning(f"Tier {i+1} [{name}] definitively UNSAFE. Exiting early.")
                    span.set_attribute("guard.exit_tier", name)
                    span.set_attribute("guard.latency_ms", (time.time() - start_time) * 1000)
                    return score
                    
                # If ambiguous and not the last tier, escalate!
                if i < len(self.tiers) - 1:
                    logger.info(f"Tier {i+1} [{name}] returned ambiguous score {score:.3f}. Escalating to {self.tiers[i+1][3]}...")
                    
            # If we reach the last tier and it's still ambiguous, we return its score
            # and let the ConformalPredictor decide based on formal calibration.
            span.set_attribute("guard.exit_tier", self.tiers[-1][3])
            span.set_attribute("guard.latency_ms", (time.time() - start_time) * 1000)
            return score
