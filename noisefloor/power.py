"""
Power analysis for Noisefloor.
"""
from typing import Optional, Any
import numpy as np
from scipy import stats  # type: ignore
from .core.verdicts import PowerResult

def power_analysis(
    effect_size: Optional[float] = None,
    baseline_rate: Optional[float] = None,
    target_rate: Optional[float] = None,
    alpha: float = 0.05,
    power_target: float = 0.80,
    test_type: str = "two-sided"
) -> PowerResult:
    """
    Computes required sample size to achieve target statistical power.
    Supports either continuous (via Cohen's d effect_size) or proportions 
    (via baseline_rate and target_rate).
    """
    if effect_size is None and (baseline_rate is None or target_rate is None):
        raise ValueError("Must provide either effect_size OR (baseline_rate and target_rate).")
        
    z_alpha = stats.norm.ppf(1 - alpha / 2) if test_type == "two-sided" else stats.norm.ppf(1 - alpha)
    z_power = stats.norm.ppf(power_target)
    
    if effect_size is not None:
        # Continuous metric approximation
        n = 2 * ((z_alpha + z_power) / effect_size) ** 2
        d = effect_size
    elif baseline_rate is not None and target_rate is not None:
        # Two proportions
        p1 = baseline_rate
        p2 = target_rate
        p_pool = (p1 + p2) / 2
        
        num = (z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) + z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        den = (p2 - p1) ** 2
        n = num / den
        
        # Approximate effect size (Cohen's h)
        h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
        d = abs(h)
    else:
        raise ValueError("Must provide either effect_size OR (baseline_rate and target_rate).")

    min_n = int(np.ceil(n))
    
    power_curve = {}
    for test_n in [max(10, min_n // 2), min_n, min_n * 2]:
        # Approximate achieved power
        z_achieved = np.sqrt(test_n / 2) * d - z_alpha
        power_curve[test_n] = float(stats.norm.cdf(z_achieved))
        
    return PowerResult(
        min_n_per_group=min_n,
        achieved_power=power_curve.get(min_n, power_target),
        effect_size=d,
        detectable_delta=abs(target_rate - baseline_rate) if target_rate else d,
        power_curve=power_curve,
        recommendation=f"You need at least {min_n} samples per group to detect this effect with {power_target*100:.0f}% power."
    )
