"""
Probability of Backtest Overfitting (PBO).
"""
import numpy as np
from dataclasses import dataclass
from typing import List
from .cpcv import CombinatorialPurgedCV

@dataclass
class PBOResult:
    pbo: float
    is_sharpes: np.ndarray
    oos_sharpes: np.ndarray
    logit_values: np.ndarray
    interpretation: str

def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    S: int = 16,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
    n_jobs: int = -1
) -> PBOResult:
    """
    Computes the Probability of Backtest Overfitting (PBO) using 
    Combinatorial Purged Cross-Validation (CPCV).
    """
    T, N = returns_matrix.shape
    if S % 2 != 0:
        S += 1  # Must be even
        
    cv = CombinatorialPurgedCV(
        n_splits=S,
        n_test_splits=S // 2,
        purge_gap=int(T * purge_pct),
        embargo_gap=int(T * embargo_pct)
    )
    
    is_sharpes_list = []
    oos_sharpes_list = []
    
    # In a full implementation, we'd use joblib for parallelizing this loop
    # For now, a simplified sequential approach
    
    for train_idx, test_idx in cv.split(np.arange(T)):
        # Calculate Sharpe for all N strategies In-Sample
        is_returns = returns_matrix[train_idx]
        is_means = np.mean(is_returns, axis=0)
        is_stds = np.std(is_returns, axis=0, ddof=1)
        
        # Avoid div by zero
        is_stds[is_stds == 0] = 1e-8
        is_sr = is_means / is_stds
        
        # Find best IS strategy
        best_strat_idx = np.argmax(is_sr)
        
        # Calculate Sharpe Out-Of-Sample for the best IS strategy
        oos_returns = returns_matrix[test_idx, best_strat_idx]
        oos_mean = np.mean(oos_returns)
        oos_std = np.std(oos_returns, ddof=1)
        if oos_std == 0:
            oos_sr = 0
        else:
            oos_sr = oos_mean / oos_std
            
        # Also need OOS sharpes for all other strategies to determine rank
        all_oos_returns = returns_matrix[test_idx]
        all_oos_means = np.mean(all_oos_returns, axis=0)
        all_oos_stds = np.std(all_oos_returns, axis=0, ddof=1)
        all_oos_stds[all_oos_stds == 0] = 1e-8
        all_oos_sr = all_oos_means / all_oos_stds
        
        is_sharpes_list.append(np.max(is_sr))
        
        # Rank of the chosen strategy out-of-sample
        rank_oos = np.mean(all_oos_sr <= oos_sr)
        oos_sharpes_list.append(rank_oos)
        
    is_sharpes = np.array(is_sharpes_list)
    oos_ranks = np.array(oos_sharpes_list)
    
    # PBO is fraction of pairs where best IS strategy performs below median OOS
    pbo = np.mean(oos_ranks < 0.5)
    
    # Logit calculation (approximate for the plot)
    logit = lambda p: np.log(p / (1 - p)) if 0 < p < 1 else 0
    logits = np.array([logit(r) for r in oos_ranks])
    
    return PBOResult(
        pbo=float(pbo),
        is_sharpes=is_sharpes,
        oos_sharpes=oos_ranks,
        logit_values=logits,
        interpretation=f"There is a {pbo*100:.1f}% chance this strategy is overfit."
    )
