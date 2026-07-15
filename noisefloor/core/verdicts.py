"""
Verdict generation and interpretations.
"""
from dataclasses import dataclass

@dataclass
class CompareResult:
    verdict: str
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    method_used: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    n_samples: int
    power: float
    min_n_for_significance: int
    interpretation: str
    raw_bootstrap_deltas: list[float]

@dataclass
class SPRTDecision:
    decision: str
    n_samples: int
    log_lambda: float
    lower_boundary: float
    upper_boundary: float
    savings_pct: float
    cumulative_pass_rate: float

@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    se: float
    bias: float
    n_samples: int
    n_bootstrap: int
    method: str

@dataclass
class PowerResult:
    min_n_per_group: int
    achieved_power: float
    effect_size: float
    detectable_delta: float
    power_curve: dict
    recommendation: str

@dataclass
class AuditResult:
    reported_sharpe: float
    annualized_return: float
    max_drawdown: float
    calmar_ratio: float
    deflated_sharpe: float
    probabilistic_sharpe: float
    probability_of_overfitting: float
    min_track_record_length: float
    implementation_risk_score: float
    cross_engine_variance: float
    verdict: str
    verdict_code: int
    confidence: str
    recommendations: list[str]
    interpretation: str

def generate_compare_verdict(p_value: float, ci_lower: float, ci_upper: float, alpha: float, effect_size: float, effect_threshold: float, power: float) -> str:
    """Generate human readable verdict string for eval comparison."""
    if p_value < alpha and abs(effect_size) > effect_threshold:
        return "✅ SHIP — statistically significant improvement"
    elif p_value > alpha and (ci_lower <= 0 <= ci_upper):
        if power < 0.5:
            return "⚠️ INCONCLUSIVE — likely underpowered, need N more samples"
        else:
            return "❌ DO NOT SHIP — not statistically significant"
    elif p_value < alpha and abs(effect_size) <= effect_threshold:
        return "⚠️ STATISTICALLY SIGNIFICANT BUT PRACTICALLY NEGLIGIBLE"
    return "⚠️ INCONCLUSIVE"
