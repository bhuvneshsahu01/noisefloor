import functools
import logging
from typing import Any

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from risklayer.integrations.tiered_guard import CascadingGuard
from risklayer.core.conformal import ConformalPredictor

logger = logging.getLogger("risklayer.instrumentation.langchain")

def patch_langchain(guard: CascadingGuard = None):
    """
    Monkey-patches LangChain's BaseChatModel to route all LLM calls through risklayer guardrails.
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain Core not found. Cannot patch.")
        return

    if guard is None:
        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate([0.05, 0.1, 0.15])
        guard = CascadingGuard(predictor=predictor)

    original_invoke = BaseChatModel.invoke

    @functools.wraps(original_invoke)
    def guarded_invoke(self, input: Any, config: Any = None, **kwargs) -> Any:
        # Extract text from LangChain message format
        prompt_text = str(input)
        
        def mock_score_fn(prompt):
            return 0.1 if len(prompt) < 100 else 0.5
            
        @guard.guard(score_fn=mock_score_fn)
        def _execute_langchain():
            return original_invoke(self, input, config=config, **kwargs)
            
        return _execute_langchain(prompt_text)

    BaseChatModel.invoke = guarded_invoke
    logger.info("LangChain BaseChatModel successfully instrumented with risklayer Guardrails.")

def unpatch_langchain():
    """Removes the risklayer patches from LangChain."""
    if not LANGCHAIN_AVAILABLE:
        return
    import importlib
    import langchain_core.language_models.chat_models
    importlib.reload(langchain_core.language_models.chat_models)
    logger.info("LangChain restored to original state.")
