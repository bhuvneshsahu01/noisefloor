"""
Quant Validation Entrypoint.
"""
import numpy as np

from ..core.verdicts import AuditResult
from .metrics import sharpe_ratio, annualized_return, max_drawdown, calmar_ratio
from .dsr import deflated_sharpe_ratio

def audit_backtest(
    returns,
    num_trials_tried: int,
    freq: str = "daily",
    risk_free_rate: float = 0.0,
    include_implementation_risk: bool = True,
    benchmark_returns = None,
    verbose: bool = True
) -> AuditResult:
    """
    The one-function entry point for quant validation.
    """
    returns = np.asarray(returns)
    periods = {"daily": 252, "weekly": 52, "monthly": 12}.get(freq, 252)
    
    # Raw metrics
    raw_sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year=periods)
    ann_ret = annualized_return(returns, periods_per_year=periods)
    mdd = max_drawdown(returns)
    calmar = calmar_ratio(returns, periods_per_year=periods)
    
    # Overfitting correction
    dsr, psr = deflated_sharpe_ratio(returns, num_trials_tried, freq, risk_free_rate)
    
    # Min track record heuristic (years)
    # Simple rule of thumb: ~ (1 + skew^2/4) / SR^2
    sr = raw_sharpe / np.sqrt(periods) # non-annualized
    min_track_record = 1.0 / (sr**2) / periods if sr > 0 else float("inf")
    
    # PBO stub (requires full returns matrix across all trials)
    pbo = 0.5  # placeholder since we don't have the matrix here
    
    if dsr > 0.95:
        verdict = "STRONG BUY"
        code = 0
        conf = "HIGH"
        rec = ["Allocate capital", "Monitor out of sample"]
    elif dsr > 0.5:
        verdict = "ALLOCATE WITH CAUTION"
        code = 1
        conf = "MEDIUM"
        rec = ["Scale in slowly", "Set tight stop loss"]
    else:
        verdict = "LIKELY OVERFIT — DO NOT ALLOCATE"
        code = 2
        conf = "HIGH"
        rec = [f"Extend track record by {max(0, min_track_record - (len(returns)/periods)):.1f} years",
               "Reduce number of variants tested"]
               
    interp = f"Your {raw_sharpe:.2f} Sharpe ratio has a {(1-dsr)*100:.0f}% chance of being a statistical artifact."
               
    return AuditResult(
        reported_sharpe=raw_sharpe,
        annualized_return=ann_ret,
        max_drawdown=mdd,
        calmar_ratio=calmar,
        deflated_sharpe=dsr,
        probabilistic_sharpe=psr,
        probability_of_overfitting=pbo,
        min_track_record_length=min_track_record,
        implementation_risk_score=0.0,
        cross_engine_variance=0.0,
        verdict=verdict,
        verdict_code=code,
        confidence=conf,
        recommendations=rec,
        interpretation=interp
    )
