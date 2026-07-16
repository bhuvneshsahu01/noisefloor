import functools
import logging
from typing import Any, Callable

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from risklayer.integrations.tiered_guard import CascadingGuard
from risklayer.core.conformal import ConformalPredictor

logger = logging.getLogger("risklayer.instrumentation.openai")

def patch_openai(guard: CascadingGuard = None):
    """
    Monkey-patches the OpenAI Python SDK to automatically route calls through risklayer guardrails.
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI SDK not found. Cannot patch.")
        return

    if guard is None:
        # Default strict guard if none provided
        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate([0.05, 0.1, 0.15])
        guard = CascadingGuard(predictor=predictor)

    # Store original methods to avoid infinite recursion or double-patching
    original_create = openai.chat.completions.create

    @functools.wraps(original_create)
    def guarded_create(*args, **kwargs) -> Any:
        messages = kwargs.get("messages", [])
        prompt_text = " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])
        
        # We need a dynamic score_fn for the guard. 
        # In a real scenario, this would use Phase 7 evaluators.
        # For now, we mock a simple score extractor.
        def mock_score_fn(prompt):
            # E.g. Check length or specific keywords as a placeholder
            return 0.1 if len(prompt) < 100 else 0.5
            
        @guard.guard(score_fn=mock_score_fn)
        def _execute_openai():
            return original_create(*args, **kwargs)
            
        # Execute the guarded version
        # We pass the prompt_text to the inner function so the guard can score it
        return _execute_openai(prompt_text)

    # Apply the patch
    openai.chat.completions.create = guarded_create
    logger.info("OpenAI SDK successfully instrumented with risklayer Guardrails.")

def unpatch_openai():
    """Removes the risklayer patches from the OpenAI SDK."""
    if not OPENAI_AVAILABLE:
        return
    import importlib
    importlib.reload(openai)
    logger.info("OpenAI SDK restored to original state.")
