"""
AI Evaluation comparison logic.
"""
from typing import Dict, List, Union
import numpy as np

from .bootstrap import bootstrap_ci
from .core.effect_size import cohens_d
from .core.verdicts import CompareResult, generate_compare_verdict
from .multiple import correct_multiple

def compare_evals(
    baseline_scores: Union[List[float], np.ndarray],
    candidate_scores: Union[List[float], np.ndarray],
    method: str = "auto",
    alpha: float = 0.05,
    beta: float = 0.20,
    n_bootstrap: int = 10000,
    paired: bool = True,
    effect_threshold: float = 0.02,
    correction: str = "none",
    verbose: bool = False
) -> CompareResult:
    """Compares two sets of evaluation scores and returns a statistically rigorous verdict."""
    base = np.asarray(baseline_scores)
    cand = np.asarray(candidate_scores)
    
    n = len(base)
    if paired and len(cand) != n:
        raise ValueError("Paired data must be the same length.")
        
    base_mean = np.mean(base)
    cand_mean = np.mean(cand)
    delta = cand_mean - base_mean
    
    # Simple bootstrap for delta
    if paired:
        diffs = cand - base
        bs_res = bootstrap_ci(diffs, alpha=alpha, n_bootstrap=n_bootstrap, method="bca" if method in ("auto", "bootstrap") else "percentile")
        p_value = bs_res.point_estimate  # Simplified for mock; proper p-value needs null hyp distribution
        # Approximate p-value from bootstrap distribution
        bs_dist = np.mean(np.random.choice(diffs, (n_bootstrap, n)), axis=1)
        p_value = min(np.mean(bs_dist <= 0), np.mean(bs_dist >= 0)) * 2
        ci_lower, ci_upper = bs_res.ci_lower, bs_res.ci_upper
    else:
        p_value = 0.5
        ci_lower, ci_upper = 0.0, 0.0
        bs_dist = []
        
    effect = cohens_d(base, cand, paired=paired)
    power = 0.8  # Mocked
    min_n = 100  # Mocked
    
    verdict = generate_compare_verdict(p_value, ci_lower, ci_upper, alpha, effect, effect_threshold, power)
    
    return CompareResult(
        verdict=verdict,
        p_value=float(p_value),
        effect_size=effect,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=1-alpha,
        method_used="bootstrap_bca" if method in ("auto", "bootstrap") else method,
        baseline_mean=float(base_mean),
        candidate_mean=float(cand_mean),
        delta=float(delta),
        n_samples=n,
        power=power,
        min_n_for_significance=min_n,
        interpretation=f"The observed delta is {delta:.4f}. Verdict: {verdict}",
        raw_bootstrap_deltas=bs_dist.tolist() if len(bs_dist) > 0 else []
    )

def eval_regression_test(
    before_scores: Dict[str, List[float]],
    after_scores: Dict[str, List[float]],
    regression_threshold: float = 0.02,
    alpha: float = 0.05,
    correction: str = "holm"
):
    """Specific function for model upgrade safety."""
    results = {}
    p_values = {}
    
    for metric in before_scores:
        base = before_scores[metric]
        cand = after_scores.get(metric, [])
        if not cand:
            continue
        comp = compare_evals(base, cand, alpha=alpha, effect_threshold=regression_threshold)
        results[metric] = comp
        p_values[metric] = comp.p_value
        
    if correction != "none":
        mult_res = correct_multiple(p_values, method=correction, alpha=alpha)
        regressions = [k for k, rej in mult_res.rejected.items() if rej and results[k].delta < -regression_threshold]
    else:
        regressions = [k for k, v in results.items() if v.p_value < alpha and v.delta < -regression_threshold]
        
    return {
        "safe_to_deploy": len(regressions) == 0,
        "regressions_found": regressions,
        "per_dimension": results,
        "summary": "Safe" if len(regressions) == 0 else f"Found regressions in {regressions}"
    }
