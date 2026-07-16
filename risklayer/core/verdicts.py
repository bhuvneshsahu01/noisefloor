"""
Verdict generation and interpretations using Pydantic for validation.
"""
from pydantic import BaseModel, Field
from typing import List

class CompareResult(BaseModel):
    verdict: str = Field(description="The final verdict string (e.g., SHIP, DO NOT SHIP)")
    p_value: float = Field(ge=0.0, le=1.0, description="The statistical p-value")
    effect_size: float = Field(description="The calculated effect size (Cohen's d or similar)")
    ci_lower: float = Field(description="Lower bound of the confidence interval")
    ci_upper: float = Field(description="Upper bound of the confidence interval")
    ci_level: float = Field(description="The confidence level (e.g., 0.95)")
    method_used: str = Field(description="The statistical method used (e.g., bootstrap_bca)")
    baseline_mean: float = Field(description="Mean score of the baseline group")
    candidate_mean: float = Field(description="Mean score of the candidate group")
    delta: float = Field(description="Difference between candidate and baseline means")
    n_samples: int = Field(gt=0, description="Number of samples evaluated")
    power: float = Field(ge=0.0, le=1.0, description="Statistical power achieved or assumed")
    min_n_for_significance: int = Field(ge=0, description="Minimum sample size needed for target power")
    interpretation: str = Field(description="Human readable explanation of the verdict")
    raw_bootstrap_deltas: List[float] = Field(default_factory=list, description="Raw bootstrap sample deltas")

class SPRTDecision(BaseModel):
    decision: str = Field(description="Current decision state: CONTINUE, ACCEPT_H0, ACCEPT_H1, MAX_REACHED")
    n_samples: int = Field(ge=0, description="Number of samples evaluated so far")
    log_lambda: float = Field(description="Current log-likelihood ratio")
    lower_boundary: float = Field(description="Log boundary to accept H0")
    upper_boundary: float = Field(description="Log boundary to accept H1")
    savings_pct: float = Field(ge=0.0, le=100.0, description="Percentage of sample budget saved")
    cumulative_pass_rate: float = Field(ge=0.0, le=1.0, description="Running average pass rate")

class BootstrapResult(BaseModel):
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    se: float
    bias: float
    n_samples: int
    n_bootstrap: int
    method: str

class PowerResult(BaseModel):
    min_n_per_group: int
    achieved_power: float
    effect_size: float
    detectable_delta: float
    power_curve: dict[int, float]
    recommendation: str

class AuditResult(BaseModel):
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
    recommendations: List[str]
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
