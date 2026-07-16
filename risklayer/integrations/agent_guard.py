import functools
import logging
import contextvars
from typing import Callable, Any
from risklayer.core.conformal import ConformalPredictor
from risklayer.core.trajectory import active_trace_id, trajectory_tracker
from risklayer.exceptions import ConformalRiskException
from risklayer.telemetry import get_tracer

logger = logging.getLogger("risklayer.risk")
tracer = get_tracer("risklayer.agent_guard")

# Context var to dynamically override shadow mode in thread/async execution
active_shadow_mode = contextvars.ContextVar("active_shadow_mode", default=None)

class RiskGuard:
    """
    The control-flow primitive for autonomous agents.
    Wraps agent tools or generation steps and halts execution if 
    the Conformal Risk bounds are breached.
    """
    def __init__(self, conformal_predictor: ConformalPredictor, shadow_mode: bool = False):
        self.predictor = conformal_predictor
        self.shadow_mode = shadow_mode

    def guard(self, score_fn: Callable[..., float], fallback_handler: Any = None):
        """
        Decorator for agent actions.
        :param score_fn: A function that calculates the non-conformity score 
                         (e.g. predictive uncertainty) of the input arguments.
        :param fallback_handler: A callable that attempts to heal failed parameters,
                                 returning a tuple of (new_args, new_kwargs).
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(f"agent_action.{func.__name__}") as span:
                    # Create a copy or pop keys to avoid passing them to score_fn/func
                    kwargs_clean = kwargs.copy()
                    trace_id = kwargs_clean.pop("trace_id", None) or active_trace_id.get()
                    is_shadow = kwargs_clean.pop("shadow_mode", None)
                    step_number = 1
                    
                    # 2. Resolve shadow mode if not explicitly popped from kwargs
                    if is_shadow is None:
                        ctx_val = active_shadow_mode.get()
                        is_shadow = ctx_val if ctx_val is not None else self.shadow_mode
                    
                    # 3. Calculate uncertainty of the impending action
                    current_score = score_fn(*args, **kwargs_clean)
                    span.set_attribute("agent.conformal_score", current_score)
                    
                    if trace_id:
                        metadata = {
                            "action_name": func.__name__,
                            "args": str(args),
                            "kwargs": str(kwargs_clean)
                        }
                        step_number = trajectory_tracker.add_step(trace_id, current_score, metadata=metadata)
                        span.set_attribute("agent.trace_id", trace_id)
                        span.set_attribute("agent.step_number", step_number)
                        
                    bound = self.predictor.get_adjusted_bound(step_number)
                    span.set_attribute("agent.conformal_bound", bound)
                    
                    # 4. Check against the formal coverage guarantee
                    if not self.predictor.is_safe(current_score, step_number):
                        # Attempt self-healing if a fallback handler is registered
                        if fallback_handler is not None:
                            try:
                                logger.info(f"Self-Healing: Triggering fallback handler for {func.__name__}...")
                                healed_args, healed_kwargs = fallback_handler(*args, **kwargs_clean)
                                
                                # Recheck risk boundary with healed parameters
                                healed_score = score_fn(*healed_args, **healed_kwargs)
                                if self.predictor.is_safe(healed_score, step_number):
                                    span.set_attribute("agent.action_status", "HEALED")
                                    span.set_attribute("agent.healed_score", healed_score)
                                    logger.info(f"Self-Healing: Action {func.__name__} successfully healed!")
                                    return func(*healed_args, **healed_kwargs)
                                else:
                                    logger.warning("Self-Healing: Healed parameters still exceed risk bounds.")
                            except Exception as fallback_err:
                                logger.error(f"Self-Healing: Fallback handler failed: {str(fallback_err)}")
                                
                        if is_shadow:
                            span.set_attribute("agent.action_status", "SHADOW_BLOCKED")
                            logger.warning(
                                f"SHADOW BLOCKED: Action {func.__name__} at step {step_number} "
                                f"exceeds bound {bound} but allowed to execute (Shadow Mode active)."
                            )
                        else:
                            span.set_attribute("agent.action_status", "BLOCKED")
                            span.record_exception(ConformalRiskException("Risk bound exceeded"))
                            
                            logger.warning(
                                f"Action {func.__name__} blocked at step {step_number}! "
                                f"Risk score {current_score} exceeds conformal bound {bound}."
                            )
                            raise ConformalRiskException(
                                f"Action blocked by RiskLayer. Conformal bound exceeded at step {step_number}."
                            )
                    else:
                        # Safe to execute
                        span.set_attribute("agent.action_status", "APPROVED")
                        
                    return func(*args, **kwargs_clean)
            return wrapper
        return decorator


# Global primitive instance for easy importing
risk = RiskGuard(ConformalPredictor(alpha=0.05))


