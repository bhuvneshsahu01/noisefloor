"""
Effect size calculations.
"""
import numpy as np

def cohens_d(x: np.ndarray, y: np.ndarray, paired: bool = False) -> float:
    """Calculate Cohen's d effect size for two groups."""
    x = np.asarray(x)
    y = np.asarray(y)
    n1, n2 = len(x), len(y)
    
    mean_diff = np.mean(y) - np.mean(x)
    
    if paired:
        if n1 != n2:
            raise ValueError("Arrays must be the same length for paired Cohen's d.")
        diffs = y - x
        s = np.std(diffs, ddof=1)
        if s == 0:
            return 0.0
        return mean_diff / s
    else:
        var1 = np.var(x, ddof=1)
        var2 = np.var(y, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return mean_diff / pooled_std

def hedges_g(x: np.ndarray, y: np.ndarray, paired: bool = False) -> float:
    """Calculate Hedge's g effect size (Cohen's d with small sample correction)."""
    d = cohens_d(x, y, paired)
    n1, n2 = len(x), len(y)
    N = n1 if paired else n1 + n2
    if N < 3:
        return d
    correction = 1 - (3 / (4 * N - 9))
    return d * correction
