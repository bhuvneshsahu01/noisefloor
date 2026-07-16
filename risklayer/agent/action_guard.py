from typing import Any, Dict
import numpy as np

class ActionGuard:
    """
    Evaluates the risk of an Agent's Tool Call (Action) using Conformal Prediction.
    It blocks Out-of-Distribution (OOD) tool calls or sensitive arguments.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.reference_actions: list[Dict[str, Any]] = []
        
    def fit(self, safe_actions: list[Dict[str, Any]]):
        """
        Calibrates the guard on a dataset of known safe actions.
        Expected format: [{"tool": "sql_query", "args": {"query": "SELECT *..."}}, ...]
        """
        self.reference_actions = safe_actions
        
    def _compute_nonconformity(self, action: Dict[str, Any]) -> float:
        """
        A heuristic nonconformity score for an action.
        In a full implementation, this would use an embeddings-based distance
        metric or a lightweight LLM judge to determine how "weird" the action is.
        Here we use a simple schema/string overlap heuristic.
        """
        if not self.reference_actions:
            return 1.0 # Maximum risk if untrained
            
        tool_name = action.get("tool")
        
        # Are there any reference actions for this tool?
        tool_references = [a for a in self.reference_actions if a.get("tool") == tool_name]
        if not tool_references:
            return 0.9 # Tool never seen before
            
        # Very naive score: difference in argument keys
        args = action.get("args", {})
        arg_keys = set(args.keys())
        
        scores = []
        for ref in tool_references:
            ref_args = ref.get("args", {})
            ref_keys = set(ref_args.keys())
            
            # Nonconformity = Jaccard distance of keys + length penalty
            intersection = len(arg_keys.intersection(ref_keys))
            union = len(arg_keys.union(ref_keys))
            jaccard_dist = 1.0 - (intersection / union) if union > 0 else 1.0
            
            scores.append(jaccard_dist)
            
        return float(np.min(scores)) if scores else 1.0

    def evaluate_action(self, action: Dict[str, Any]) -> dict:
        """
        Evaluates a new action against the reference set.
        """
        score = self._compute_nonconformity(action)
        
        # Conformal quantile
        if self.reference_actions:
            ref_scores = [self._compute_nonconformity(r) for r in self.reference_actions]
            n = len(ref_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            q_hat = np.quantile(ref_scores, q_level)
        else:
            q_hat = 0.5 # Default boundary
            
        is_safe = bool(score <= q_hat)
        
        return {
            "action": action.get("tool"),
            "nonconformity_score": score,
            "conformal_bound": q_hat,
            "is_safe": is_safe,
            "decision": "PROCEED" if is_safe else "BLOCK"
        }
