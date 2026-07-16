import numpy as np
from typing import Dict, Any, List, Tuple
from risklayer.compare import compare_evals

class FdrPromptGuard:
    """
    FDR & FWER Adjustment Guard for iterative Prompt Optimization (e.g. DSPy, MIPRO).
    Prevents multiple-testing bias ("p-value hacking" / overfitting prompts to small test sets).
    """
    def __init__(self, target_alpha: float = 0.05):
        self.target_alpha = target_alpha

    def verify_optimization_improvement(
        self,
        baseline_scores: List[float],
        iteration_scores: List[List[float]],
        method: str = "holm"
    ) -> Dict[str, Any]:
        """
        Evaluates a set of prompt optimization iterations against baseline.
        Applies Holm-Bonferroni (FWER) or Benjamini-Hochberg (FDR) adjustments
        across all test iterations.
        """
        n_iters = len(iteration_scores)
        raw_p_values = []
        comparisons = []
        
        # 1. Run comparisons for each iteration
        for i, cand in enumerate(iteration_scores):
            comp = compare_evals(baseline_scores, cand, alpha=self.target_alpha)
            raw_p_values.append(comp.p_value)
            comparisons.append(comp)
            
        # 2. Adjust p-values based on chosen method
        p_vals = np.array(raw_p_values)
        adjusted_p_vals = np.ones_like(p_vals)
        
        if method == "holm":
            # Holm-Bonferroni (FWER) correction
            sorted_indices = np.argsort(p_vals)
            for rank, idx in enumerate(sorted_indices):
                multiplier = n_iters - rank
                adjusted_p_vals[idx] = min(1.0, p_vals[idx] * multiplier)
            # Ensure monotonicity
            for i in range(1, n_iters):
                idx_prev = sorted_indices[i - 1]
                idx_curr = sorted_indices[i]
                adjusted_p_vals[idx_curr] = max(adjusted_p_vals[idx_curr], adjusted_p_vals[idx_prev])
                
        elif method == "bh":
            # Benjamini-Hochberg (FDR) correction
            sorted_indices = np.argsort(p_vals)
            for rank, idx in enumerate(sorted_indices):
                multiplier = n_iters / (rank + 1)
                adjusted_p_vals[idx] = min(1.0, p_vals[idx] * multiplier)
            # Ensure reverse monotonicity
            for i in range(n_iters - 2, -1, -1):
                idx_next = sorted_indices[i + 1]
                idx_curr = sorted_indices[i]
                adjusted_p_vals[idx_curr] = min(adjusted_p_vals[idx_curr], adjusted_p_vals[idx_next])
                
        # 3. Find if any iteration passes the adjusted alpha boundary
        passed_indices = np.where(adjusted_p_vals < self.target_alpha)[0]
        
        results = []
        for i in range(n_iters):
            results.append({
                "iteration": i + 1,
                "raw_p_value": float(p_vals[i]),
                "adjusted_p_value": float(adjusted_p_vals[i]),
                "delta": comparisons[i].delta,
                "is_significant": i in passed_indices
            })
            
        is_real_improvement = len(passed_indices) > 0
        best_iter_idx = int(np.argmax([r["delta"] for r in results]))
        best_iter = results[best_iter_idx]
        
        verdict = "STATIONARY/NOISE (Do not deploy)"
        if is_real_improvement:
            if best_iter["is_significant"]:
                verdict = f"REAL IMPROVEMENT (Deploy Iteration {best_iter['iteration']})"
            else:
                verdict = "WARNING: Best iteration is not statistically significant after FDR adjustment!"
                
        return {
            "is_real_improvement": is_real_improvement,
            "verdict": verdict,
            "best_iteration": best_iter,
            "all_iterations": results
        }
