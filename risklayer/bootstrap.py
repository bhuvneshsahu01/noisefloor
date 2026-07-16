"""
Bootstrap Confidence Intervals (BCa).
"""
import numpy as np
from scipy import stats
from .core.verdicts import BootstrapResult
from .core.utils import coerce_to_array

def bootstrap_ci(
    scores,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    stat_func=np.mean,
    method: str = "bca"
) -> BootstrapResult:
    """
    Computes bootstrap confidence intervals for any metric on a set of scores.
    """
    scores = coerce_to_array(scores)
    n = len(scores)
    
    if n < 5:
        raise ValueError("Require at least 5 samples for bootstrap CI.")
        
    point_estimate = stat_func(scores)
    
    # Generate bootstrap samples
    indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_samples = scores[indices]
    
    # Calculate statistic for each sample
    # Apply stat_func along axis 1 (across the samples)
    if stat_func == np.mean:
        bootstrap_stats = np.mean(bootstrap_samples, axis=1)
    else:
        bootstrap_stats = np.array([stat_func(s) for s in bootstrap_samples])
        
    se = np.std(bootstrap_stats, ddof=1)
    bias = np.mean(bootstrap_stats) - point_estimate
    
    if method == "bca":
        # Bias correction factor (z0)
        p_less = np.mean(bootstrap_stats < point_estimate)
        # Avoid infinity if p_less is 0 or 1
        p_less = max(min(p_less, 0.999), 0.001)
        z0 = stats.norm.ppf(p_less)
        
        # Acceleration factor (a) via jackknife
        jackknife_stats = np.zeros(n)
        for i in range(n):
            idx = np.ones(n, dtype=bool)
            idx[i] = False
            jackknife_stats[i] = stat_func(scores[idx])
            
        mean_jack = np.mean(jackknife_stats)
        num = np.sum((mean_jack - jackknife_stats) ** 3)
        den = 6.0 * (np.sum((mean_jack - jackknife_stats) ** 2) ** 1.5)
        
        a = num / den if den != 0 else 0.0
        
        # Calculate corrected percentiles
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1_alpha = stats.norm.ppf(1 - alpha / 2)
        
        alpha_l = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        alpha_u = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
        
        # Convert to percentages for np.percentile
        pl = max(min(alpha_l * 100, 100), 0)
        pu = max(min(alpha_u * 100, 100), 0)
        
        ci_lower = np.percentile(bootstrap_stats, pl)
        ci_upper = np.percentile(bootstrap_stats, pu)
    else:
        # Fallback to percentile
        ci_lower = np.percentile(bootstrap_stats, (alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        method = "percentile"
        
    return BootstrapResult(
        point_estimate=float(point_estimate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=1 - alpha,
        se=float(se),
        bias=float(bias),
        n_samples=n,
        n_bootstrap=n_bootstrap,
        method=method
    )
