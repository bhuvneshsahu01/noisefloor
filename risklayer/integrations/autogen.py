from typing import Dict, Any, Callable
from risklayer.agent.action_guard import ActionGuard

def inject_autogen_guard(agent: Any, action_guard: ActionGuard, block_on_fail: bool = True):
    """
    Monkey-patches an AutoGen ConversableAgent to intercept and evaluate
    tool executions using the Conformal ActionGuard.
    """
    if not hasattr(agent, "execute_function"):
        raise ValueError("Provided agent does not appear to be a valid AutoGen ConversableAgent")

    original_execute_function = agent.execute_function

    def guarded_execute_function(func_call: Dict[str, Any]) -> Any:
        tool_name = func_call.get("name")
        arguments = func_call.get("arguments", {})
        
        # In AutoGen, arguments is often a JSON string
        import json
        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments)
            except Exception:
                args_dict = {"raw": arguments}
        else:
            args_dict = arguments

        action = {"tool": tool_name, "args": args_dict}
        
        # Evaluate risk
        result = action_guard.evaluate_action(action)
        
        if not result["is_safe"] and block_on_fail:
            print(f"[RiskLayer AutoGen Guard] 🛑 BLOCKED OOD tool call: {tool_name}")
            return (
                False, 
                f"Error: Tool execution blocked by RiskLayer. "
                f"Non-conformity score {result['nonconformity_score']:.2f} > bound {result['conformal_bound']:.2f}"
            )
            
        print(f"[RiskLayer AutoGen Guard] ✅ APPROVED tool call: {tool_name}")
        return original_execute_function(func_call)

    # Apply patch
    agent.execute_function = guarded_execute_function
    return agent
