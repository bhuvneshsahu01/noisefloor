"""
Multiple comparison correction.
"""
from dataclasses import dataclass
from typing import Dict

@dataclass
class MultipleCompareResult:
    corrected_pvalues: Dict[str, float]
    rejected: Dict[str, bool]
    adjusted_alpha: Dict[str, float]
    method: str
    n_comparisons: int
    family_wise_alpha: float
    summary: str

def correct_multiple(
    results: Dict[str, float],
    method: str = "holm",
    alpha: float = 0.05
) -> MultipleCompareResult:
    """
    Applies multiple-comparison correction to a dict of p-values.
    """
    n = len(results)
    if n == 0:
        raise ValueError("Must provide at least one result.")
        
    sorted_items = sorted(results.items(), key=lambda x: x[1])
    
    corrected_pvalues = {}
    rejected = {}
    adjusted_alpha = {}
    
    if method == "holm":
        # Step-down procedure
        keep_rejecting = True
        for i, (key, p_val) in enumerate(sorted_items):
            adj_alpha = alpha / (n - i)
            adjusted_alpha[key] = adj_alpha
            
            # Holm corrected p-value
            corrected_p = min(1.0, p_val * (n - i))
            # Ensure monotonicity
            if i > 0:
                prev_key = sorted_items[i-1][0]
                corrected_p = max(corrected_p, corrected_pvalues[prev_key])
            
            corrected_pvalues[key] = corrected_p
            
            if keep_rejecting and p_val <= adj_alpha:
                rejected[key] = True
            else:
                keep_rejecting = False
                rejected[key] = False
                
    elif method == "bonferroni":
        for key, p_val in results.items():
            adj_alpha = alpha / n
            adjusted_alpha[key] = adj_alpha
            corrected_pvalues[key] = min(1.0, p_val * n)
            rejected[key] = p_val <= adj_alpha
            
    else:
        raise ValueError(f"Unknown method {method}")
        
    return MultipleCompareResult(
        corrected_pvalues=corrected_pvalues,
        rejected=rejected,
        adjusted_alpha=adjusted_alpha,
        method=method,
        n_comparisons=n,
        family_wise_alpha=alpha,
        summary=f"Corrected {n} comparisons using {method}."
    )
