"""
Financial metrics (Sharpe, Calmar, Max Drawdown).
"""
import numpy as np
from typing import Any

def max_drawdown(returns: np.ndarray[Any, Any]) -> float:
    """Calculate maximum drawdown of a return series."""
    returns = np.asarray(returns)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(np.min(drawdowns))

def annualized_return(returns: np.ndarray[Any, Any], periods_per_year: int = 252) -> float:
    """Calculate annualized return."""
    returns = np.asarray(returns)
    compounded_growth = np.prod(1 + returns)
    n_years = len(returns) / periods_per_year
    if n_years == 0:
        return 0.0
    return float(compounded_growth ** (1 / n_years) - 1)

def sharpe_ratio(returns: np.ndarray[Any, Any], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio."""
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate / periods_per_year
    mean = np.mean(excess_returns)
    std = np.std(excess_returns, ddof=1)
    if std == 0:
        return 0.0
    return float((mean / std) * np.sqrt(periods_per_year))

def calmar_ratio(returns: np.ndarray[Any, Any], periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio."""
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(ann_ret / mdd)
