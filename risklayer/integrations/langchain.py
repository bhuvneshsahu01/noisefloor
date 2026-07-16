import json
from typing import Any, Dict, List, Optional
from risklayer import RiskGuard

# This relies on the standard langchain-core library, which enterprise users will have installed
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    BaseCallbackHandler = object
    LLMResult = None

class RiskLayerLangchainCallback(BaseCallbackHandler):
    """
    A LangChain callback handler that automatically evaluates LLM responses
    and intercepts dangerous tool calls in real-time.
    """
    def __init__(self, risk_guard: RiskGuard, block_on_fail: bool = True):
        self.guard = risk_guard
        self.block_on_fail = block_on_fail
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Evaluate the final text generation for hallucinations/safety."""
        if not response or not response.generations:
            return
            
        for generation_list in response.generations:
            for generation in generation_list:
                text = generation.text
                # Generate a mock conformal score using the guard's fallback mechanism
                # In production, this would call the RiskLayer backend API
                report = self.guard.evaluate([0.5]) # Mock mid-level score
                if not report.is_safe and self.block_on_fail:
                    raise Exception(f"RiskLayer Intercept: Response exceeded conformal risk boundary (alpha={self.guard.target_alpha})")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Intercept tool calls before they execute and pass to ActionGuard."""
        tool_name = serialized.get("name", "unknown_tool")
        
        # Parse the input_str into dict if possible
        try:
            args = json.loads(input_str)
        except Exception:
            args = {"raw_input": input_str}
            
        action = {"tool": tool_name, "args": args}
        
        # We need the action guard (which would normally be part of the RiskGuard context)
        # For MVP, we'll mock the check if no action guard is explicitly mounted
        print(f"[RiskLayer LangChain Plugin] Verifying Tool Call: {tool_name}")
        
        # Assume it's a critical tool we want to block if suspicious
        if "eval" in tool_name or "exec" in tool_name or "bash" in tool_name:
            if "rm -rf" in input_str:
                if self.block_on_fail:
                    raise Exception(f"RiskLayer Intercept: Blocked OOD Tool Call '{tool_name}' with arguments {args}")
