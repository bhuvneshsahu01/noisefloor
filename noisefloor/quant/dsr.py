"""
Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR).
"""
import numpy as np
from scipy import stats

def expected_maximum_sharpe(num_trials: int) -> float:
    """
    Computes the expected maximum Sharpe ratio from m independent trials
    using the Eubank-Gordon-Rojo formula.
    """
    if num_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    m = num_trials
    
    # E[max(X_1...X_m)] for standard normal
    part1 = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / m)
    part2 = euler_mascheroni * stats.norm.ppf(1 - 1 / (m * np.e))
    return part1 + part2

def probabilistic_sharpe_ratio(
    sr_obs: float,
    sr_benchmark: float,
    t: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Computes the Probabilistic Sharpe Ratio (PSR), the probability that
    the true Sharpe ratio exceeds the benchmark.
    
    kurtosis should be the standard kurtosis (normal = 3.0).
    """
    if t < 3:
        return 0.5
        
    num = (sr_obs - sr_benchmark) * np.sqrt(t - 1)
    den = np.sqrt(1 - skewness * sr_obs + ((kurtosis - 1) / 4) * sr_obs**2)
    
    if den <= 0:
        return 0.0
        
    z = num / den
    return stats.norm.cdf(z)

def deflated_sharpe_ratio(
    returns: np.ndarray,
    num_trials: int,
    freq: str = "daily",
    risk_free_rate: float = 0.0,
    return_components: bool = False
):
    """
    Computes the Deflated Sharpe Ratio (Bailey & López de Prado, 2012).
    Returns (dsr, psr) tuple, or full components if return_components is True.
    """
    returns = np.asarray(returns)
    t = len(returns)
    if t < 3:
        return (0.5, 0.5) if not return_components else (0.5, 0.5, 0.0, t, 0.0, 3.0)
        
    mean_ret = np.mean(returns) - risk_free_rate
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        sr_obs = 0.0
    else:
        sr_obs = mean_ret / std_ret
        
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns, fisher=False)  # normal = 3
    
    # Convert observed Sharpe to non-annualized base for the formula
    sr_star = expected_maximum_sharpe(num_trials)
    
    dsr = probabilistic_sharpe_ratio(sr_obs, sr_star, t, skewness, kurtosis)
    psr = probabilistic_sharpe_ratio(sr_obs, 0.0, t, skewness, kurtosis)
    
    if return_components:
        return dsr, psr, sr_star, t, skewness, kurtosis
    return dsr, psr
