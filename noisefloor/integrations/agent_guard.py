import functools
import logging
from typing import Callable, Any
from noisefloor.core.conformal import ConformalPredictor
from noisefloor.core.trajectory import active_trace_id, trajectory_tracker
from noisefloor.exceptions import ConformalRiskException
from noisefloor.telemetry import get_tracer

logger = logging.getLogger("noisefloor.risk")
tracer = get_tracer("noisefloor.agent_guard")

class RiskGuard:
    """
    The control-flow primitive for autonomous agents.
    Wraps agent tools or generation steps and halts execution if 
    the Conformal Risk bounds are breached.
    """
    def __init__(self, conformal_predictor: ConformalPredictor):
        self.predictor = conformal_predictor

    def guard(self, score_fn: Callable[..., float]):
        """
        Decorator for agent actions.
        :param score_fn: A function that calculates the non-conformity score 
                         (e.g. predictive uncertainty) of the input arguments.
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(f"agent_action.{func.__name__}") as span:
                    # 1. Resolve trace_id for trajectory-level calibration
                    trace_id = kwargs.get("trace_id") or active_trace_id.get()
                    step_number = 1
                    
                    # 2. Calculate uncertainty of the impending action
                    current_score = score_fn(*args, **kwargs)
                    span.set_attribute("agent.conformal_score", current_score)
                    
                    if trace_id:
                        step_number = trajectory_tracker.add_step(trace_id, current_score)
                        span.set_attribute("agent.trace_id", trace_id)
                        span.set_attribute("agent.step_number", step_number)
                        
                    bound = self.predictor.get_adjusted_bound(step_number)
                    span.set_attribute("agent.conformal_bound", bound)
                    
                    # 3. Check against the formal coverage guarantee
                    if not self.predictor.is_safe(current_score, step_number):
                        span.set_attribute("agent.action_status", "BLOCKED")
                        span.record_exception(ConformalRiskException("Risk bound exceeded"))
                        
                        logger.warning(
                            f"Action {func.__name__} blocked at step {step_number}! "
                            f"Risk score {current_score} exceeds conformal bound {bound}."
                        )
                        raise ConformalRiskException(
                            f"Action blocked by RiskLayer. Conformal bound exceeded at step {step_number}."
                        )
                    
                    # 4. Safe to execute
                    span.set_attribute("agent.action_status", "APPROVED")
                    return func(*args, **kwargs)
            return wrapper
        return decorator

# Global primitive instance for easy importing
risk = RiskGuard(ConformalPredictor(alpha=0.05))

