"""
Noisefloor: The convergent statistical verdict engine for AI decisions and quantitative backtesting.
"""

from .compare import compare_evals, eval_regression_test
from .sprt import sprt_gate
from .bootstrap import bootstrap_ci
from .power import power_analysis
from .multiple import correct_multiple

__all__ = [
    "compare_evals",
    "eval_regression_test",
    "sprt_gate",
    "bootstrap_ci",
    "power_analysis",
    "correct_multiple",
]
