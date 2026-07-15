# Noisefloor — Exhaustive Implementation Plan
## The Convergent Statistical Verdict Engine for AI Decisions

> **Status:** Pre-build. This document is the complete knowledge dump and build bible.  
> **Last Updated:** July 2026  
> **Author's Note:** This is not a startup pitch deck. This is everything you need to build, ship, and grow Noisefloor. Read it before writing a single line of code, and re-read it whenever you feel lost.

---

# PART 0: THE CORE THESIS — WHY THIS EXISTS

## The Problem In One Sentence

> Every AI team in the world is making shipping decisions — "prompt A is better than B", "model X outperforms Y", "this strategy has a 1.8 Sharpe ratio" — based on point estimates with no statistical significance testing. They are flipping coins and calling them decisions.

## The Deeper Problem: Two Identical Crises In Different Worlds

There are two professional communities in 2026, separated by vocabulary but suffering the exact same mathematical problem:

### World 1: AI/ML Engineers
A team ships Claude 3.7 → GPT-4.5 migration. They run 100 eval test cases. Score goes from 0.73 to 0.76. They declare victory and ship. **What they don't know**: with 100 samples and a 3-point difference, the 95% confidence interval is (-0.02, +0.08). The improvement is not statistically distinguishable from random noise. They just shipped something that might be *worse*.

**Evidence this is happening at scale:**
- DeepEval: 400K+ monthly PyPI downloads. No bootstrap CIs.
- Ragas: 14.7K GitHub stars. No significance testing.
- Promptfoo: 22.9K stars, acquired by OpenAI in March 2026. Never added CIs.
- Braintrust: $45M raised. No sequential testing.
- ICML 2025 paper "When Not to Rely on CLT in LLM Evaluation" — proves current methods are statistically broken for typical eval sizes (n < 300).

### World 2: Quant Researchers
A trader runs 47 variations of a strategy. The best one has a 1.83 Sharpe ratio. They allocate $2M. **What they don't know**: the Deflated Sharpe Ratio, corrected for 47 trials, is 0.31. The Probability of Backtest Overfitting is 0.62. They just deployed capital to a noise-fitted strategy.

**Evidence this is happening at scale:**
- 5+ independent developers have rebuilt PBO/DSR from scratch over 8 years — all abandoned
- mlfinlab (the reference implementation) went closed-source
- Hudson & Thames charged £100/month for these algorithms — people paid
- AlphaAssay ($0.05/call API) is live and charging — proven WTP
- SEBI algo ID framework (April 2026): 53% of NSE cash volume is algorithmic, creating regulatory pressure for defensible validation

### The Convergence Insight

**The math is ~80% identical.** Both crises share:
- Bootstrap confidence intervals
- Multiple-comparison correction (Holm-Bonferroni)
- Sequential testing (SPRT)
- Effect size estimation
- Sample size / power analysis

**The failure mode is identical**: "I tried many things until one looked good, then I committed to it based on the best-looking number."

**No existing tool bridges both worlds.** `evalstats` does AI eval only. AlphaAssay does quant only. Noisefloor serves both with the same codebase and brand.

---

# PART 1: PRODUCT VISION & POSITIONING

## Name: **Noisefloor**

**Why this name:** In audio engineering, the "noise floor" is the background noise level below which signals cannot be distinguished from random variation. You can't hear a signal that's below the noise floor. In data science: you can't trust a measurement that's within the noise floor of your methodology. The name is instantly understood by technical people, memorable, and captures the core value proposition.

**Alternative names considered:**
- `robustbt` (too quant-specific)
- `sprt-eval` (too narrow, only SPRT)
- `calibraeval` (too evaluation-only)
- `sigtest` (too generic)
- `verdikt` (pronunciation ambiguity)

## Tagline Options
- *"The statistical verdict layer for AI decisions"*
- *"Is your improvement real, or are you below the noise floor?"*
- *"Stop shipping noise."*
- *"Every A/B decision in AI is statistically broken. Fix it."*

## Core Value Proposition
**One function call** that tells you whether your measured improvement is real or an artifact of how you measured it. Works for both:
- AI/ML engineers comparing prompts, models, or RAG configs
- Quant researchers validating trading strategy backtests

## Who Are The Users (Personas)

### Primary Persona A: The AI/ML Engineer
- **Who:** ML engineer, prompt engineer, MLOps engineer at a startup or mid-size company shipping LLM-based products
- **Stack:** Python, pytest, GitHub Actions, maybe DeepEval/Ragas
- **Pain moment:** CI run returns a 2% accuracy improvement on 80 test cases. Everyone celebrates. Ships to prod. Turns out it was noise.
- **What they currently do:** "Vibe check" the outputs, trust the point estimate, write custom bootstrap scripts in Jupyter notebooks
- **What they want:** One function that tells them "SHIP" or "DO NOT SHIP" with mathematical backing
- **Persona size:** 500K+ globally (all DeepEval/Ragas/Promptfoo users + the Braintrust audience)

### Primary Persona B: The Quant Researcher
- **Who:** Quant researcher at a prop trading shop, hedge fund, or solo systematic trader
- **Stack:** Python, pandas, vectorbt, QuantConnect, or custom backtesting engine
- **Pain moment:** Spent 3 weeks backtesting. Best strategy has 1.8 Sharpe. Allocates capital. Underperforms. Realizes it was data-mined.
- **What they currently do:** Apply gut feel, maybe simple hold-out tests, nothing formal
- **What they want:** A formal "PASS/FAIL" verdict that's defensible — especially under SEBI's new algo ID framework
- **Persona size:** ~50K globally (but very high WTP — capital decisions)

### Secondary Persona: LLM Agent Developers
- **Who:** Builders of agentic pipelines using LangGraph, CrewAI, AutoGen, PydanticAI
- **Pain moment:** Agent v2 seems better in a 20-run test. But multi-step agent evaluation is expensive. Are those 20 runs enough?
- **Special need:** Sequential testing (SPRT) to stop evaluations early once significance is reached — saves 70–90% of API costs

### Secondary Persona: Research Scientists / ML Researchers
- **Who:** Academic or industry researcher publishing results comparing model variants
- **Pain moment:** Comparing two fine-tuned models on a benchmark. Need to report not just accuracy but whether the difference is statistically significant.
- **Special need:** Proper multiple-comparison correction, effect sizes, and power analysis for publication

---

# PART 2: FULL FEATURE SPECIFICATION

## 2.1 The Core API — All Functions

### MODULE 1: `noisefloor.compare` — AI/LLM Evaluation

#### `compare_evals(baseline, candidate, method, alpha, n_bootstrap)`
**What:** Compares two sets of evaluation scores and returns a statistically rigorous verdict.

**Why:** The most common operation in LLM development — "is version B better than version A?" — is currently done with raw score comparison, which has no statistical validity for typical eval sizes (n=50-300).

**Signature:**
```python
from noisefloor import compare_evals

result = compare_evals(
    baseline_scores: list[float] | np.ndarray,   # scores from baseline model/prompt
    candidate_scores: list[float] | np.ndarray,  # scores from new model/prompt
    method: str = "auto",       # "bootstrap" | "sprt" | "paired_t" | "wilcoxon" | "auto"
    alpha: float = 0.05,        # significance level (Type I error rate)
    beta: float = 0.20,         # Type II error rate (for SPRT mode)
    n_bootstrap: int = 10000,   # bootstrap resampling iterations
    paired: bool = True,        # True if same test cases used for both
    effect_threshold: float = 0.02,  # minimum effect size worth caring about
    correction: str = "none",   # "none" | "holm" | "bonferroni" (for multi-comparison)
    verbose: bool = False
) -> CompareResult
```

**Output:**
```python
@dataclass
class CompareResult:
    verdict: str          # "SHIP" | "DO NOT SHIP" | "INCONCLUSIVE — need more data"
    p_value: float        # e.g., 0.034
    effect_size: float    # Cohen's d or equivalent
    ci_lower: float       # lower bound of 95% CI for the difference
    ci_upper: float       # upper bound of 95% CI for the difference
    ci_level: float       # 0.95 by default
    method_used: str      # which method was actually used
    baseline_mean: float
    candidate_mean: float
    delta: float          # candidate_mean - baseline_mean
    n_samples: int        # number of test cases
    power: float          # observed statistical power
    min_n_for_significance: int  # "you need at least N more test cases"
    interpretation: str   # human-readable explanation
    raw_bootstrap_deltas: np.ndarray | None  # for plotting
```

**Example output (human readable):**
```
[noisefloor] Comparing 87 eval scores (paired)
[noisefloor] Baseline: 0.731 ± 0.042
[noisefloor] Candidate: 0.758 ± 0.039
[noisefloor] Observed delta: +0.027 (+3.7%)
[noisefloor] 95% CI of delta: (-0.008, +0.062)
[noisefloor] p-value: 0.134 (bootstrap, 10K iterations)
[noisefloor] Effect size (Cohen's d): 0.31 (small)
[noisefloor] Power: 0.42 (underpowered)
[noisefloor] ──────────────────────────────────────
[noisefloor] VERDICT: DO NOT SHIP
[noisefloor] Reason: The observed improvement is not statistically significant.
[noisefloor] The 95% CI includes zero — the candidate could be worse.
[noisefloor] You need at least 187 test cases to detect this effect reliably.
```

---

#### `sprt_gate(scores_stream, h0_rate, h1_rate, alpha, beta)`
**What:** A streaming/sequential evaluation gate. As scores come in one by one, it computes whether to stop early (accept H1 = improvement, or H0 = no improvement) or continue.

**Why:** Running 500 LLM eval test cases when the regression is obvious after 15 tests wastes 98% of the API budget. SPRT (Wald's Sequential Probability Ratio Test, 1947) provides mathematically guaranteed early stopping with controlled error rates.

**The Math:**
- H₀: true pass rate = p₀ (baseline performance)
- H₁: true pass rate = p₁ (new, improved performance)
- After each observation, update log-likelihood ratio: Λₙ = Σ log(P(x|H₁)/P(x|H₀))
- If Λₙ ≥ log((1-β)/α): stop, accept H₁ (improvement detected)
- If Λₙ ≤ log(β/(1-α)): stop, accept H₀ (no improvement, or regression)
- Otherwise: continue

**Signature:**
```python
gate = sprt_gate(
    h0_rate: float,           # baseline pass rate (e.g., 0.73)
    h1_rate: float,           # minimum improvement worth detecting (e.g., 0.78)
    alpha: float = 0.05,      # false positive rate
    beta: float = 0.20,       # false negative rate
    max_samples: int = 500    # hard limit even if SPRT hasn't concluded
)

for score in eval_scores_stream:
    decision = gate.update(score)  # score is float 0-1 or bool
    if decision != "CONTINUE":
        print(f"Stopped at sample {gate.n_samples}: {decision}")
        print(f"API savings: {gate.savings_pct:.1f}%")
        break
```

**Output at each step:**
```python
@dataclass
class SPRTDecision:
    decision: str      # "CONTINUE" | "ACCEPT_H1" | "ACCEPT_H0" | "MAX_REACHED"
    n_samples: int
    log_lambda: float  # current log-likelihood ratio
    lower_boundary: float
    upper_boundary: float
    savings_pct: float  # % of max_samples saved by stopping early
    cumulative_pass_rate: float
```

**CLI usage:**
```bash
# Pipe your eval results through sprt_gate
cat eval_results.jsonl | noisefloor sprt-gate --h0 0.73 --h1 0.78 --alpha 0.05
# Output:
# [SPRT] Sample 18/500: log_lambda=-2.34
# [SPRT] Stopped at sample 18: ACCEPT H0 (no improvement)
# [SPRT] API savings: 96.4% ($43.20 saved at $0.003/call)
```

---

#### `bootstrap_ci(scores, alpha, n_bootstrap, stat)`
**What:** Computes bootstrap confidence intervals for any metric on a set of scores.

**Why:** CLT-based standard errors fail for typical eval sizes (n < 300). The ICML 2025 paper shows that CLT dramatically underestimates uncertainty in LLM evaluation. Bootstrap is distribution-free and correct.

**The Math:**
1. Compute observed statistic θ̂ = stat(scores)
2. Resample with replacement n_bootstrap times → θ̂*₁, θ̂*₂, ..., θ̂*B
3. 95% CI = [percentile(2.5%), percentile(97.5%)] of the bootstrap distribution
4. BCa correction applied by default (bias-corrected and accelerated — more accurate)

**Signature:**
```python
result = bootstrap_ci(
    scores: list[float] | np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    stat: callable = np.mean,    # default: mean accuracy
    method: str = "bca"          # "percentile" | "bca" | "basic"
) -> BootstrapResult

@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    se: float              # bootstrap standard error
    bias: float            # bootstrap bias estimate
    n_samples: int
    n_bootstrap: int
    method: str
```

---

#### `power_analysis(effect_size, alpha, power_target, baseline_rate)`
**What:** Answers "how many test cases do I need to reliably detect a real improvement?"

**Why:** Teams run 50 test cases and wonder why their results are inconclusive. Power analysis before running evals prevents wasted compute and false-negative decisions.

**The Math:**
For binary metrics (pass/fail): use two-proportion z-test power formula
For continuous metrics: use Cohen's d and standard power formula
Both cross-checked with bootstrap simulation for small n

**Signature:**
```python
result = power_analysis(
    effect_size: float | None = None,  # Cohen's d, or None to compute from baseline/target
    baseline_rate: float | None = None, # e.g., 0.73
    target_rate: float | None = None,   # e.g., 0.78 (minimum detectable effect)
    alpha: float = 0.05,
    power_target: float = 0.80,         # desired power (0.80 = 80% chance of detecting real effect)
    test_type: str = "two-sided"        # "one-sided" | "two-sided"
) -> PowerResult

@dataclass
class PowerResult:
    min_n_per_group: int    # minimum samples needed
    achieved_power: float   # power at min_n
    effect_size: float      # Cohen's d or equivalent
    detectable_delta: float # minimum detectable difference at this n
    power_curve: dict       # {n: power} for plotting
    recommendation: str     # human-readable
```

---

#### `correct_multiple(results_dict, method)`
**What:** Applies multiple-comparison correction to a dict of p-values (for when you're comparing many variants simultaneously).

**Why:** If you compare 10 prompt variants and declare the best one "significant" at p<0.05, your actual false-positive rate is ~40%, not 5%. Holm-Bonferroni correction controls this.

**The Math (Holm-Bonferroni, step-down procedure):**
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₖ
2. For each i: reject H₀ᵢ if pᵢ ≤ α/(k - i + 1)
3. Stop rejecting as soon as one fails (step-down)

Advantages over Bonferroni: uniformly more powerful while maintaining FWER control.

**Signature:**
```python
corrected = correct_multiple(
    results: dict[str, float],  # {"variant_a": 0.03, "variant_b": 0.04, "variant_c": 0.18}
    method: str = "holm",       # "holm" | "bonferroni" | "bh" (Benjamini-Hochberg for FDR)
    alpha: float = 0.05
) -> MultipleCompareResult

@dataclass
class MultipleCompareResult:
    corrected_pvalues: dict[str, float]    # {"variant_a": 0.09, ...}
    rejected: dict[str, bool]              # {"variant_a": False, ...}
    adjusted_alpha: dict[str, float]       # threshold used per comparison
    method: str
    n_comparisons: int
    family_wise_alpha: float  # = alpha
    summary: str
```

---

#### `eval_regression_test(before_scores, after_scores, regression_threshold)`
**What:** Specific function for model upgrade safety — "did my new model regress on any dimension?"

**Why:** When upgrading from GPT-4o to GPT-4.5, teams need to know not just if the average improved but whether any specific capability degraded. This is a one-sided test with a regression threshold.

**Signature:**
```python
result = eval_regression_test(
    before_scores: dict[str, list[float]],  # {"accuracy": [...], "faithfulness": [...]}
    after_scores: dict[str, list[float]],
    regression_threshold: float = 0.02,    # alert if any dimension drops by >2%
    alpha: float = 0.05,
    correction: str = "holm"               # must correct for multiple dimensions
) -> RegressionTestResult

@dataclass
class RegressionTestResult:
    safe_to_deploy: bool
    regressions_found: list[str]  # ["faithfulness"] or []
    per_dimension: dict[str, CompareResult]
    summary: str
```

---

### MODULE 2: `noisefloor.quant` — Quantitative Backtest Validation

#### `audit_backtest(returns, num_trials, freq)`
**What:** The one-function entry point for quant validation. Takes a strategy's return series and the number of variants tried, and outputs a full overfitting audit.

**Why:** This is the "sitcom test" function — visceral, immediate, humbling. Show a quant that their 1.83 Sharpe is actually 0.31 after deflation for 47 trials, and they'll never forget it.

**Signature:**
```python
from noisefloor.quant import audit_backtest

result = audit_backtest(
    returns: pd.Series | np.ndarray,    # daily/weekly/monthly returns
    num_trials_tried: int,              # how many variants/parameters were tried
    freq: str = "daily",               # "daily" | "weekly" | "monthly"
    risk_free_rate: float = 0.0,
    include_implementation_risk: bool = True,
    benchmark_returns: pd.Series | None = None,
    verbose: bool = True
) -> AuditResult

@dataclass
class AuditResult:
    # Raw metrics
    reported_sharpe: float
    annualized_return: float
    max_drawdown: float
    calmar_ratio: float
    
    # Overfitting correction
    deflated_sharpe: float          # DSR — the key number
    probabilistic_sharpe: float     # PSR — P(Sharpe > 0)
    probability_of_overfitting: float  # PBO from CPCV (0-1)
    min_track_record_length: float  # years needed to validate at current Sharpe
    
    # Implementation Risk (March 2026 paper by Yin et al.)
    implementation_risk_score: float    # 0-10 (0 = low risk)
    cross_engine_variance: float | None # if multiple engines provided
    
    # Verdict
    verdict: str   # "STRONG BUY" | "ALLOCATE WITH CAUTION" | "LIKELY OVERFIT — DO NOT ALLOCATE"
    verdict_code: int  # 0=pass, 1=borderline, 2=fail
    confidence: str    # "HIGH" | "MEDIUM" | "LOW"
    
    # Recommendations
    recommendations: list[str]
    interpretation: str  # full human-readable explanation
```

**Example output:**
```
[noisefloor.quant] Auditing strategy: 1247 daily returns (4.97 years)
[noisefloor.quant] Trials attempted: 47
[noisefloor.quant] ────────────────────────────────────────────────
[noisefloor.quant] Reported Sharpe Ratio:     1.83  (unadjusted)
[noisefloor.quant] Deflated Sharpe Ratio:     0.31  (adjusted for 47 trials)
[noisefloor.quant] Probabilistic Sharpe:      0.41  (P(true Sharpe > 0) = 41%)
[noisefloor.quant] Probability of Overfitting: 0.62 (62% chance this is overfit)
[noisefloor.quant] Min Track Record Length:   8.4 years (you have 4.97 years)
[noisefloor.quant] ────────────────────────────────────────────────
[noisefloor.quant] VERDICT: ⚠️  LIKELY OVERFIT — DO NOT ALLOCATE
[noisefloor.quant] Your 1.83 Sharpe ratio has a 62% chance of being a statistical artifact
[noisefloor.quant] of the 47 parameter variants you tested. The true expected Sharpe after
[noisefloor.quant] selection bias correction is approximately 0.31.
[noisefloor.quant] Recommendation: Extend track record by 3.4+ years on OOS data
[noisefloor.quant] OR reduce the number of variants tested in next iteration to < 10.
```

---

#### `deflated_sharpe_ratio(returns, num_trials, freq, skewness, kurtosis)`
**What:** Computes the Deflated Sharpe Ratio (Bailey & López de Prado, 2012).

**Why:** The Sharpe ratio is fundamentally biased when selected from multiple trials. If you test 47 strategies and pick the best, the expected best Sharpe from random data is approximately √(2 log(47)) ≈ 2.8. The DSR corrects for this selection bias, non-normality (skewness and excess kurtosis), and sample length.

**The Math:**
```
DSR = PSR(SR*) where:
SR* = benchmark Sharpe (expected Sharpe from selection)
PSR(x) = Φ((√(T-1) * (SR - x)) / √(1 - γ₃*SR + (γ₄-1)/4 * SR²))

where:
- T = number of observations
- γ₃ = skewness of returns
- γ₄ = kurtosis of returns
- SR* = estimated maximum expected Sharpe from m trials
- SR* ≈ (1 - γ_E) * Φ⁻¹(1 - 1/m) + γ_E * Φ⁻¹(1 - 1/(m*e))
  (Eubank, Gordon, Rojo formula for expected maximum)
```

**Key references:**
- Bailey, D.H., López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier"
- Bailey, D.H., López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"

**Signature:**
```python
dsr, psr = deflated_sharpe_ratio(
    returns: pd.Series,
    num_trials: int,
    freq: str = "daily",
    risk_free_rate: float = 0.0,
    return_components: bool = False  # if True, also return SR*, T, skew, kurt
) -> tuple[float, float]
```

---

#### `probability_of_backtest_overfitting(returns_matrix, S)`
**What:** Computes the Probability of Backtest Overfitting (PBO) using the Combinatorial Purged Cross-Validation (CPCV) framework from Bailey et al. (2017).

**Why:** Unlike simple train/test splits, PBO generates multiple IS/OOS scenario pairs through combinatorial path enumeration. For each pair, if the best IS strategy is also best OOS, it counts as "not overfit." PBO is the fraction of pairs where the best IS strategy performs *below median* OOS.

**The Math (CSCV Algorithm):**
1. Partition the returns matrix into T equal-length sub-matrices
2. For each combination C of T/2 sub-matrices as IS: compute IS Sharpe for each strategy
3. Assign remaining T/2 sub-matrices as OOS: compute OOS Sharpe for optimal IS strategy
4. PBO = (# of pairs where OOS Sharpe rank < 0.5) / (total # of pairs)

**Why CPCV over plain CSCV:**
- CPCV purges overlapping returns from training/test boundaries (prevents information leakage in time-series data)
- Embargoed periods after training data prevent look-ahead bias from market impact

**Signature:**
```python
pbo_result = probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,  # rows=time, cols=strategy variants
    S: int = 16,                   # number of sub-matrices (must be even)
    purge_pct: float = 0.01,       # % of observations to purge at boundaries
    embargo_pct: float = 0.01,     # % to embargo after training
    n_jobs: int = -1               # parallelism
) -> PBOResult

@dataclass
class PBOResult:
    pbo: float                    # 0-1, probability of overfitting
    is_sharpes: np.ndarray        # distribution of IS Sharpe ratios
    oos_sharpes: np.ndarray       # distribution of OOS Sharpe ratios
    logit_values: np.ndarray      # for plotting the PBO degradation curve
    interpretation: str
```

---

#### `combinatorial_purged_cv(X, y, n_splits, purge_pct, embargo_pct)`
**What:** A scikit-learn-compatible cross-validator that implements CPCV correctly for financial time-series.

**Why:** This is the scikit-learn gap that has been open for 4.5 years (issue #22229). skfolio has it, but for portfolio optimization only — not for general time-series ML. Noisefloor ships it as a standalone, MIT-licensed, properly maintained CV splitter.

**The Problem With Standard CV:**
- KFold: leaks future data into training (time-series contamination)
- TimeSeriesSplit: only single-path, doesn't generate distribution of OOS performance
- Standard train_test_split: only one realization, no distribution estimate

**CPCV Fixes This By:**
- Purging: removing observations near the training/test boundary where labels overlap
- Embargoing: removing a buffer after training data (for strategies with latency)
- Combinatorial: generating T_choose_(T/2) distinct IS/OOS scenario pairs

**Signature:**
```python
# Drop-in replacement for sklearn cross-validators
from noisefloor.quant import CombinatorialPurgedCV

cv = CombinatorialPurgedCV(
    n_splits: int = 6,           # number of folds (T in the paper)
    n_test_splits: int = 2,      # number of folds used for testing per combination
    purge_gap: int = 0,          # samples to purge at boundaries
    embargo_gap: int = 0,        # samples to embargo post-training
    samples_info_sets: pd.Series | None = None  # for event-driven purging
)

# Use exactly like sklearn:
for train_idx, test_idx in cv.split(X):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

---

#### `implementation_risk_audit(returns_by_engine, strategy_params)`
**What:** Quantifies the Implementation Risk of a strategy — how much the backtested performance depends on the specific backtesting engine used.

**Why:** This is the brand-new dimension from the March 2026 paper "Implementation Risk in Portfolio Backtesting" (Yin, Miki, Lesnichenko & Gural). Running the same strategy in vectorbt, backtrader, and zipline produces different Sharpe ratios once realistic transaction costs, slippage, and fill assumptions are introduced. This cross-engine variance is a measurable "implementation risk."

**Metrics Computed:**
- **Implementation Uncertainty Index (IUI):** Standard deviation of Sharpe ratios across engines
- **Decision Agreement Factor (DAF):** Fraction of engines that agree on PASS/FAIL verdict
- **Conclusion Stability Index (CSI):** Robustness of the investment conclusion across engine assumptions

**Signature:**
```python
risk = implementation_risk_audit(
    returns_by_engine: dict[str, pd.Series],  # {"vectorbt": series, "backtrader": series}
    verdict_threshold: float = 0.5,           # Sharpe > threshold = PASS
) -> ImplementationRiskResult

@dataclass
class ImplementationRiskResult:
    iui: float              # Implementation Uncertainty Index (stdev of Sharpes)
    daf: float              # Decision Agreement Factor (0-1, 1=all agree)
    csi: float              # Conclusion Stability Index
    engine_sharpes: dict    # {"vectorbt": 1.2, "backtrader": 0.8, ...}
    risk_level: str         # "LOW" | "MEDIUM" | "HIGH"
    recommendation: str
```

---

### MODULE 3: `noisefloor.core` — Shared Statistical Primitives

These are the shared math building blocks used by both modules above.

#### `effect_size(a, b, method)`
Computes Cohen's d, Hedge's g, Glass's Δ, or the common language effect size.

#### `sample_size_needed(effect, alpha, power, test_type)`
Returns minimum n to detect effect_size with given power.

#### `run_permutation_test(a, b, n_permutations, stat)`
Permutation test — distribution-free, no assumptions, best for small samples.

#### `adaptive_conformal_inference(scores_stream, alpha, method)`
Adaptive Conformal Inference (ACI) for non-exchangeable streaming data. Maintains coverage guarantees even when score distributions drift over time. Key for agent evaluation where score distributions change as the agent learns.

**The Math (ACI):**
- Standard conformal prediction assumes exchangeability — fails for drifting distributions
- ACI uses a running correction to the non-conformity score threshold
- If recent coverage drops below 1-α, the threshold adapts downward (gets stricter)
- Provides distribution-free coverage guarantees for non-stationary streams

#### `minimum_track_record_length(sharpe, alpha, sr_star)`
Computes the minimum number of years of track record needed to be 95% confident that a strategy with the observed Sharpe ratio has a true Sharpe > SR* (usually 0.0 or the risk-free rate).

---

### MODULE 4: `noisefloor.ci` — CI/CD Integration Layer

#### `pytest` Plugin
```python
# conftest.py
import noisefloor

@noisefloor.sprt_eval(
    baseline="baselines/prod_scores.jsonl",
    h0_rate=0.73,
    h1_rate=0.78,
    alpha=0.05,
    fail_on="regression"  # or "no_improvement"
)
def test_model_quality(eval_output):
    ...
```

#### GitHub Action
```yaml
# .github/workflows/eval.yml
- uses: noisefloor-dev/noisefloor-action@v1
  with:
    baseline-file: baselines/prod_scores.jsonl
    candidate-file: eval_output.jsonl
    method: sprt
    h0-rate: 0.73
    h1-rate: 0.78
    fail-on: regression
```

#### CLI
```bash
# Compare two eval result files
noisefloor compare baseline.jsonl candidate.jsonl --method bootstrap --alpha 0.05

# Stream SPRT gate
cat eval_scores.jsonl | noisefloor sprt --h0 0.73 --h1 0.78

# Audit a backtest
noisefloor quant-audit returns.csv --trials 47 --freq daily

# Power analysis
noisefloor power --baseline 0.73 --target 0.78 --alpha 0.05 --power 0.80
```

---

## 2.2 Planned Integrations

### AI Eval Side
| Integration | How | Why |
|---|---|---|
| **DeepEval** | Plugin/wrapper around `compare_evals` | 400K monthly users — largest surface |
| **Ragas** | Wrapper for `compare_evals` on RAG metrics | 14.7K stars, growing fast |
| **LangSmith** | Export traces → noisefloor compare | LangChain ecosystem |
| **Weights & Biases** | Log noisefloor verdicts as W&B runs | Standard MLOps integration |
| **Braintrust** | API plugin | $45M-funded platform |
| **Langfuse** | OSS integration | OSS-to-OSS natural fit |

### Quant Side
| Integration | How | Why |
|---|---|---|
| **skfolio** | Drop-in replacement CV + audit wrapper | 1.9K stars, commercially backed |
| **VectorBT** | Post-backtesting audit pipeline | Popular vectorized backtester |
| **QuantConnect** | Algorithm result → audit | Largest retail quant platform |
| **Zipline-Reloaded** | Tearsheet extension | Maintained Zipline fork |
| **backtrader** | Custom analyzer | Large installed base |

### CI/CD
| Integration | How | Why |
|---|---|---|
| **pytest** | Plugin (`pytest-noisefloor`) | Where developers already test |
| **GitHub Actions** | Official action | Industry standard CI |
| **GitLab CI** | Template | Enterprise GitLab users |
| **pre-commit** | Hook | Prevents bad commits early |

---

## 2.3 The Verdict System (Brand Differentiator)

**What:** Every function in Noisefloor returns a human-readable `verdict` string, not just numbers. This is a conscious design decision.

**Why:** p-values and confidence intervals are intimidating. Developers don't want to interpret statistics — they want a decision. The verdict system bridges the gap:

| Condition | Verdict |
|---|---|
| p < α AND delta > effect_threshold | `✅ SHIP — statistically significant improvement` |
| p > α AND CI includes zero | `❌ DO NOT SHIP — not statistically significant` |
| CI includes zero but power < 0.5 | `⚠️ INCONCLUSIVE — likely underpowered, need N more samples` |
| p < α but delta < effect_threshold | `⚠️ STATISTICALLY SIGNIFICANT BUT PRACTICALLY NEGLIGIBLE` |
| PBO > 0.5 | `🚫 LIKELY OVERFIT — do not allocate` |
| DSR > 0.5 | `✅ STRONG SIGNAL — low overfitting risk` |

---

# PART 3: TECHNICAL ARCHITECTURE

## 3.1 Repository Structure

```
noisefloor/
├── noisefloor/
│   ├── __init__.py               # top-level public API
│   ├── compare.py               # AI/LLM eval comparison functions
│   ├── sprt.py                  # SPRT implementation
│   ├── bootstrap.py             # Bootstrap CI + BCa
│   ├── power.py                 # Power analysis
│   ├── multiple.py              # Multiple comparison correction
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── audit.py             # audit_backtest (main entry)
│   │   ├── dsr.py               # Deflated Sharpe Ratio
│   │   ├── pbo.py               # Probability of Backtest Overfitting
│   │   ├── cpcv.py              # Combinatorial Purged Cross-Validation
│   │   ├── implementation_risk.py  # March 2026 paper metrics
│   │   └── metrics.py           # Sharpe, Calmar, max drawdown, etc.
│   ├── core/
│   │   ├── __init__.py
│   │   ├── effect_size.py
│   │   ├── aci.py               # Adaptive Conformal Inference
│   │   ├── utils.py
│   │   └── verdicts.py          # Verdict system
│   ├── ci/
│   │   ├── __init__.py
│   │   ├── pytest_plugin.py
│   │   ├── github_action.py     # GitHub Action entrypoint
│   │   └── cli.py               # CLI (using Click or Typer)
│   └── integrations/
│       ├── deepeval_wrapper.py
│       ├── ragas_wrapper.py
│       ├── skfolio_wrapper.py
│       └── vectorbt_wrapper.py
├── tests/
│   ├── test_compare.py
│   ├── test_sprt.py
│   ├── test_bootstrap.py
│   ├── test_dsr.py
│   ├── test_pbo.py
│   ├── test_cpcv.py
│   ├── test_cli.py
│   └── fixtures/
│       ├── known_overfit_strategy.py    # synthetic strategy that should fail
│       ├── known_good_strategy.py       # synthetic strategy that should pass
│       └── sample_eval_scores.jsonl
├── docs/
│   ├── getting-started.md
│   ├── math/
│   │   ├── bootstrap.md          # math derivations
│   │   ├── sprt.md
│   │   ├── dsr.md
│   │   └── pbo.md
│   └── cookbook/
│       ├── ci_cd_integration.md
│       ├── deepeval_integration.md
│       └── sebi_compliance.md
├── examples/
│   ├── llm_eval_comparison.py
│   ├── agent_evaluation.py
│   ├── backtest_audit.py
│   └── sebi_report.py
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

## 3.2 Dependencies

### Core Dependencies (minimal)
```toml
[project.dependencies]
numpy = ">=1.24"
scipy = ">=1.10"
pandas = ">=2.0"  # optional — for quant module
```

**Why minimal:** The library must be `pip install noisefloor` in 5 seconds with zero friction. Heavy dependencies kill adoption. numpy + scipy are essentially universal.

### Optional Dependencies
```toml
[project.optional-dependencies]
quant = ["pandas>=2.0", "numba>=0.57"]      # for quant module + speed
ci = ["click>=8.0", "rich>=13.0"]           # for CLI
integrations = ["deepeval", "ragas"]        # for integration wrappers
dev = ["pytest", "hypothesis", "black", "mypy", "ruff"]
```

### Why NOT to add:
- `scikit-learn`: adds 50MB to install, creates version conflicts, not needed
- `statsmodels`: has most things but heavy, prefer our own implementations for control
- `numba`: optional only — speeds up CPCV simulation but not required

## 3.3 Implementation Notes — Critical Correctness Points

### Bootstrap CI: BCa Is Mandatory
Don't use naive percentile bootstrap. The **BCa (Bias-Corrected and Accelerated)** method is significantly more accurate for small samples and non-symmetric distributions. It's more complex but the right default.

The BCa correction requires:
1. **Bias correction factor**: ẑ₀ = Φ⁻¹(fraction of bootstrap samples < observed)
2. **Acceleration factor**: â = (Σ(θ̄ - θᵢ)³) / (6 * (Σ(θ̄ - θᵢ)²)^(3/2)) (jackknife influence values)

### SPRT: Handle Edge Cases
- Both p₀ and p₁ must be in (0, 1), not 0 or 1 (causes log(0))
- Must handle non-binary outcomes (continuous scores) via binning or parametric extension
- Max samples limit is mandatory — SPRT has no guaranteed termination for intermediate effect sizes

### DSR: Return Series Must Be De-Meaned
The DSR formula assumes the return series represents excess returns over the risk-free rate. Must handle this correctly.

### CPCV: The Purging Math
The key formula for purge indices: for training on fold i ending at time t_i_end, and test on fold j starting at t_j_start, purge all training observations with label-generating times beyond t_j_start - purge_gap. This is non-trivial to implement correctly — reference Advances in Financial Machine Learning, Chapter 7.

### PBO: Correct CSCV Implementation
The original Bailey et al. paper uses T=16 sub-matrices with C(16,8) = 12,870 combinations. This is computationally expensive. For practical use:
- Default T=16, use parallel processing (n_jobs=-1)
- For quick estimates, T=8 gives C(8,4)=70 combinations — much faster, similar results
- Results should match the original R implementation `CSCV` package output

## 3.4 Performance Requirements

| Function | Input Size | Target Time |
|---|---|---|
| `compare_evals()` | 1000 scores, 10K bootstrap | < 2 seconds |
| `sprt_gate.update()` | Single observation | < 1ms |
| `deflated_sharpe_ratio()` | Any length returns | < 100ms |
| `probability_of_backtest_overfitting()` | 100 strategies, 1000 obs | < 30s |
| `combinatorial_purged_cv()` | 10K obs, 6 folds | < 5s |

Speed strategy:
- Use `numba` JIT for hot loops in CPCV and bootstrap (optional dep)
- Vectorize SPRT update with numpy
- Cache bootstrap distribution when inputs unchanged

---

# PART 4: PHASED BUILD ROADMAP

## Phase 0: Pre-Build (Week 0 — Before Any Code)

### Step 1: User Interviews (5 conversations)

**AI eval side:** Find 3 engineers using DeepEval/Ragas/Braintrust
Questions:
- "Walk me through how you decide if a prompt improvement is real"
- "Have you ever shipped something that turned out to be noise?"
- "What do you currently use for significance testing?"
- "Would you use a function that gives you p-value + verdict in one call?"

**Quant side:** Find 2 quant researchers (WQ BRAIN community, QuantInsti, NSE/BSE algo traders)
Questions:
- "How many strategy variants did you test before deploying your current strategy?"
- "What's your process for validating that the Sharpe ratio is real?"
- "Have you heard of the Deflated Sharpe Ratio?"
- "SEBI now requires documented validation — does that change anything for you?"

**Hypothesis to validate:**
1. At least 4/5 say they use point estimates or vibe checks (no statistical testing)
2. At least 3/5 have shipped or allocated based on what turned out to be noise
3. At least 3/5 say they'd use a one-function solution

**Kill condition:** If 4+/5 say they already have rigorous pipelines they're happy with → don't build.

### Step 2: Competitive Audit

Run `noisefloor compare` conceptually against these:
- `evalstats` (pip install evalstats) — what does it actually do vs our design?
- `skfolio` CPCV — how buggy/complete is it really?
- `mlfinpy` — does DSR implementation actually work?
- `sprtt` (R/Python) — what's missing for our use case?
- AlphaAssay API — what does $0.05/call actually get you?

**Document all gaps found.**

### Step 3: Public API Design Review

Before writing implementation, write the README with code examples first (README-Driven Development). Post the draft API to:
- r/MachineLearning: "Would you use this for LLM eval?"
- r/quant: "Would you use this for backtest validation?"

Get 20+ comments. Revise API based on feedback. **Then** write the code.

---

## Phase 1: Core Library — AI Eval Side (Weeks 1–4)

**Goal:** A publishable, correct, well-tested Python library with the AI eval functions.

### Week 1: Foundation + Bootstrap

**Build:**
- `noisefloor/bootstrap.py`: BCa bootstrap CI implementation
- `noisefloor/power.py`: Power analysis for proportion and continuous metrics
- `noisefloor/core/effect_size.py`: Cohen's d, Hedge's g
- `noisefloor/core/verdicts.py`: Verdict system logic
- Basic test suite with synthetic data

**Correctness checks:**
- BCa CI output must match scipy's bootstrap on same inputs (within tolerance)
- Power formula output must match G*Power software for standard scenarios
- Cohen's d must match standard textbook formula

### Week 2: Compare + Multiple Correction

**Build:**
- `noisefloor/compare.py`: `compare_evals()` full implementation
- `noisefloor/multiple.py`: Holm-Bonferroni, Bonferroni, Benjamini-Hochberg
- `noisefloor/core/utils.py`: Input validation, type coercion, etc.

**Testing strategy:**
- Property-based tests with Hypothesis: "for any two identical score distributions, p-value should be > 0.05 ~95% of the time"
- Known-answer tests: manually computed cases from statistics textbooks
- Edge cases: n=1, n=2, all scores identical, extreme skewness

### Week 3: SPRT + Regression Test

**Build:**
- `noisefloor/sprt.py`: Full SPRT with streaming interface
- Regression test function: `eval_regression_test()`
- Power analysis for SPRT (expected sample size under H0 and H1)
- Integration with compare results

**Key SPRT correctness test:**
```python
# Under H0: should accept H0 with probability >= (1-alpha) = 95%
import random
results = []
for _ in range(1000):
    gate = sprt_gate(h0_rate=0.5, h1_rate=0.6, alpha=0.05, beta=0.20)
    scores = [random.random() > 0.5 for _ in range(500)]  # H0 is true
    decision = "CONTINUE"
    for s in scores:
        decision = gate.update(s).decision
        if decision != "CONTINUE":
            break
    results.append(decision)

# Should have ~5% false positives (ACCEPT_H1 when H0 is true)
assert sum(r == "ACCEPT_H1" for r in results) / 1000 < 0.08  # allow some Monte Carlo variance
```

### Week 4: CLI + First Blog Post

**Build:**
- `noisefloor/ci/cli.py`: CLI with Click/Typer
- pytest plugin scaffold
- pyproject.toml, README, CHANGELOG, LICENSE (MIT)
- Publish to PyPI: `pip install noisefloor`

**The First Blog Post (Do This Before Full Launch):**
Title: *"We ran 200 LLM eval comparisons. 67% were statistically meaningless."*

Content structure:
1. The problem: show a real eval where "improvement" isn't significant
2. The math: bootstrap CI explained visually (no formulas)
3. The demo: 5-line code example with `compare_evals()`
4. The results: show the verdict system output
5. Call to action: `pip install noisefloor`

**Where to post:**
- HN: Show HN post
- r/MachineLearning
- Twitter/X AI community
- Personal blog (crosspost to Substack/Medium)

**Target: 200+ GitHub stars from this post alone**

---

## Phase 2: Quant Module (Weeks 5–8)

**Goal:** Add the quantitative backtest validation module and launch on r/quant.

### Week 5: DSR + PSR + MinTRL

**Build:**
- `noisefloor/quant/dsr.py`: Deflated Sharpe Ratio (exact Bailey & López de Prado formula)
- Probabilistic Sharpe Ratio (PSR)
- Minimum Track Record Length (MinTRL)
- Basic `AuditResult` dataclass

**Reference implementation to cross-check:**
- The original Marcos López de Prado Python code (published in his books)
- `mlfinpy` DSR implementation (verify our output matches)

### Week 6: PBO + CPCV

**Build:**
- `noisefloor/quant/pbo.py`: CSCV-based PBO implementation
- `noisefloor/quant/cpcv.py`: CPCV sklearn-compatible splitter
- Parallel CSCV with joblib
- Performance optimization with numba (optional dep)

**This is the hardest part of the build.** Plan for iteration.

### Week 7: audit_backtest() + Implementation Risk

**Build:**
- `noisefloor/quant/audit.py`: Full `audit_backtest()` wrapper
- `noisefloor/quant/implementation_risk.py`: Yin et al. metrics (IUI, DAF, CSI)
- Integration tests with real historical data (e.g., SPY daily returns)

### Week 8: Distribution + Second Blog Post

**The Second Blog Post:**
Title: *"I backtested my best strategy and got a 1.83 Sharpe. Noisefloor said 0.31."*

Content:
1. Show a realistic backtest with great-looking metrics
2. Run `audit_backtest()` and show the deflation
3. Explain the math (briefly, no formulas)
4. Explain why this matters for SEBI compliance (India-specific angle)
5. Demo code

**Where to post:**
- r/quant
- r/algotrading
- FinTwit (Twitter quant community)
- Quantocracy blog aggregator
- Hacker News (second Show HN)

---

## Phase 3: Integrations + Cloud (Months 3–6)

### Integration PRs (Month 3)
Submit PRs or plugins to:
1. **DeepEval**: `noisefloor` as an optional statistical backend
2. **Ragas**: wrapper that adds CIs to Ragas metrics
3. **VectorBT**: post-backtest audit plugin

Each integration is a distribution channel. When DeepEval users google "statistical significance for deepeval", they find Noisefloor.

### Cloud API (Month 4)
Build a hosted version of the API:
- Input: evaluation scores or return series (JSON)
- Output: full audit result with verdict (JSON)
- Signing: Ed25519 cryptographic signatures on results (like AlphaAssay)
  - **Why signing?** SEBI-regulated algo traders need defensible, immutable audit records
  - A signed verdict can be attached to a SEBI compliance filing
- Pricing: $0.05/call for quant audits, $0.01/call for eval comparisons (match AlphaAssay pricing)
- Stack: FastAPI + Railway/Fly.io (cheapest, fastest to deploy)

### Persistent Trial Tracking (Month 5)
The API should track how many times you've called `audit_backtest` on variants of the same strategy, and auto-apply the deflation correction for the total number of trials. This is a **statefulness moat** — local library can't do this, cloud API can.

User story: "I've tested 12 variants of my momentum strategy. Noisefloor knows about all 12 and tells me my current 1.4 Sharpe is deflated to 0.8 based on all 12 trials."

---

## Phase 4: Enterprise (Month 6–12)

### Enterprise Features
- **Audit-ready PDF reports**: Full statistical report suitable for SEBI filing or bank MRM documentation
- **SR 26-2 compliance mode**: AI governance report format matching Federal Reserve model risk management requirements
- **SSO + team dashboards**: Share audit results within a team
- **CI/CD enterprise gateway**: Self-hosted version with persistent trial tracking

### Enterprise Target Customers
- **India quant side**: Algo trading firms under SEBI, domestic hedge funds, prop trading desks at ICICI, Kotak, Zerodha
- **Global AI eval side**: Mid-size AI product companies ($10M-$100M ARR) with serious eval pipelines
- **Banks with SR 26-2**: Model risk teams validating GenAI models (India: RBI compliance; US: Fed compliance)

---

# PART 5: MATHEMATICAL DEEP DIVE

## 5.1 Why Bootstrap Over Parametric Tests

**The choice:** For AI eval comparison, should we use:
- (a) Paired t-test
- (b) Wilcoxon signed-rank test
- (c) Bootstrap

**Answer: Bootstrap (BCa) by default, with fallback to parametric.**

**Why not t-test:**
- Assumes normally distributed score differences — false for LLM eval (often bimodal: 0.0 or 1.0)
- Breaks for n < 30 without normality
- Can't handle complex statistics (e.g., F1 score, BLEU)

**Why not Wilcoxon:**
- Non-parametric but still limited to comparing medians
- Can't produce confidence intervals for the mean difference easily
- Loses statistical power for complex metrics

**Why Bootstrap works:**
- Distribution-free — makes no assumptions about score distribution
- Works for any summary statistic: mean, median, F1, BLEU, exact match
- BCa variant handles skewness and non-constant variance correctly
- 10,000 iterations is sufficient for 95% CI (variance of CI estimate < 0.001)

**When to use parametric fallback:**
- n > 500: t-test or z-test is fine, faster, equivalent
- Paired continuous scores with approximately normal differences: paired t-test has slightly higher power
- Use `method="auto"` to let Noisefloor choose

## 5.2 SPRT — The Full Math

Wald's SPRT for Bernoulli (binary pass/fail) outcomes:

**Setup:**
- H₀: θ = θ₀ (baseline pass rate)
- H₁: θ = θ₁ (improved pass rate, θ₁ > θ₀)
- α = P(reject H₀ | H₀ true) — false positive rate
- β = P(accept H₀ | H₁ true) — false negative rate

**Boundaries:**
- A = β / (1 - α) — lower boundary
- B = (1 - β) / α — upper boundary

**Update rule after observation xₙ (0 or 1):**
```
Λₙ = Λₙ₋₁ × P(xₙ | H₁) / P(xₙ | H₀)
   = Λₙ₋₁ × (θ₁^xₙ × (1-θ₁)^(1-xₙ)) / (θ₀^xₙ × (1-θ₀)^(1-xₙ))
```

**Decision:**
- If Λₙ ≥ B: reject H₀ (improvement detected)
- If Λₙ ≤ A: accept H₀ (no improvement)
- Otherwise: continue

**Working in log-space (numerical stability):**
```python
log_lambda += x * log(θ₁/θ₀) + (1-x) * log((1-θ₁)/(1-θ₀))
upper_log = log(B)  # log((1-β)/α)
lower_log = log(A)  # log(β/(1-α))
```

**Extension to continuous scores:**
Option 1: Bin scores into pass/fail using a threshold
Option 2: Use normal SPRT with running mean and variance
Option 3: Use a one-sided t-test sequential procedure (SPRT for means, not proportions)

Noisefloor implements all three, auto-selecting based on score type.

**Expected sample size under H₀:**
E[N|H₀] ≈ (β log(β/(1-α)) + (1-β) log((1-β)/α)) / (θ₀ log(θ₀/θ₁) + (1-θ₀) log((1-θ₀)/(1-θ₁)))

**Expected sample size under H₁:**
E[N|H₁] ≈ (α log(β/(1-α)) + (1-α) log((1-β)/α)) / (θ₁ log(θ₁/θ₀) + (1-θ₁) log((1-θ₁)/(1-θ₀)))

These formulas allow computing "expected API cost savings" before running the evaluation.

## 5.3 Deflated Sharpe Ratio — Full Derivation

**The Sharpe Ratio (unadjusted):**
SR = (μ - r_f) / σ

where μ = annualized mean return, r_f = risk-free rate, σ = annualized std dev

**Problem:** This is a realized Sharpe from ONE sample path. It's inflated by:
1. **Selection bias**: picking the best from m trials
2. **Non-normality**: returns have fat tails and negative skewness
3. **Short track record**: small-sample estimation error

**Step 1 — Probabilistic Sharpe Ratio (PSR):**
PSR(SR*) = P(SR_true > SR*) = Φ[(√(T-1) × (SR_obs - SR*)) / √(1 - γ₃×SR_obs + ((γ₄-1)/4)×SR_obs²)]

where:
- T = number of observations
- γ₃ = skewness of returns
- γ₄ = kurtosis of returns
- SR* = benchmark (often 0.0 or Sharpe of a passive strategy)

**Step 2 — Expected Maximum Sharpe from m Trials:**
SR* = (1 - γ_E) × Φ⁻¹(1 - 1/m) + γ_E × Φ⁻¹(1 - 1/(m × e))

where γ_E ≈ 0.5772 (Euler-Mascheroni constant) and e = 2.71828

**Step 3 — Deflated Sharpe Ratio:**
DSR = PSR(SR*) — i.e., the probability that the TRUE Sharpe exceeds the expected maximum from random selection

**Interpretation:**
- DSR > 0.95: very likely real alpha
- DSR 0.50–0.95: borderline, need more data
- DSR < 0.50: more likely noise than signal — probably overfit

## 5.4 Adaptive Conformal Inference (ACI)

**For agent evaluation specifically:**

Standard conformal prediction assumes *exchangeability* — that the calibration data and test data are drawn from the same distribution. This fails for:
- Agent evaluation where the agent's behavior changes over time (as it adapts)
- Streaming evaluation where prompt distribution evolves
- Multi-step agent trajectories where early steps affect later distributions

**ACI Fix (Gibbs et al., 2021; Angelopoulos et al., 2023):**
Maintain a correction parameter γₜ that adapts the non-conformity threshold:
```
γ_t+1 = γ_t + η × (α - 1{y_t ∉ Ĉ_t})
```

where:
- η = step size (learning rate for adaptation)
- α = target error rate
- 1{y_t ∉ Ĉ_t} = 1 if true label falls outside predicted set

**Effect:** If recent coverage drops below 1-α (too many misses), γ adapts to make confidence sets wider. Guarantees long-run coverage ≥ 1-α even for non-exchangeable sequences.

**Use in Noisefloor:** `adaptive_conformal_inference()` wraps this for agent trajectory evaluation — maintaining coverage guarantees on multi-step agent quality scores even as the agent's behavior distribution drifts.

---

# PART 6: DISTRIBUTION STRATEGY

## 6.1 Blog/Content Strategy (Ordered By Priority)

### Post 1: The Launch Post (Week 4)
**Title:** *"Your LLM eval improvements are probably statistical noise. Here's how to find out."*
**Platform:** HN (Show HN), personal blog
**Angle:** Data-driven, show real examples of false positives in LLM eval
**Code demo:** `compare_evals()` returning "DO NOT SHIP" on a seemingly-good improvement
**CTA:** `pip install noisefloor` + GitHub stars

### Post 2: The Quant Post (Week 8)
**Title:** *"I deflated my 1.83 Sharpe to 0.31 after 47 backtests. Then I found out about SEBI's new audit rules."*
**Platform:** r/quant, r/algotrading, FinTwit, Quantocracy
**Angle:** The SEBI April 2026 regulatory angle (India-specific) + universal overfitting story
**Code demo:** `audit_backtest()` with the humbling verdict
**CTA:** `pip install noisefloor` + quant-focused docs

### Post 3: The Technical Deep Dive (Month 2)
**Title:** *"The math behind 'Is this AI improvement real?'"*
**Platform:** HN, dev.to, Substack
**Angle:** Statistical education — explain SPRT, BCa, DSR for a technical audience
**Goal:** SEO and long-term reference traffic

### Post 4: The Integration Post (Month 3)
**Title:** *"How we added statistical rigor to DeepEval in 3 lines"*
**Platform:** HN, DeepEval Discord, r/MachineLearning
**Angle:** Show the integration plugin — drives DeepEval users to Noisefloor

### Post 5: The ICML Response Post (Month 3)
**Title:** *"We built the tool that ICML 2025's 'Don't Use CLT for LLM Eval' paper calls for"*
**Platform:** HN, Twitter/X AI community, academic Twitter
**Angle:** Position Noisefloor as the production implementation of the ICML 2025 research

### Post 6: The Cost Savings Post (Month 4)
**Title:** *"We reduced our LLM eval API costs by 78% using sequential testing"*
**Platform:** HN, dev.to, company engineering blogs
**Angle:** Pure economic angle — SPRT early stopping saves money
**CTA:** Cloud API pricing page

## 6.2 Community Strategy

### For AI Eval Audience
- **DeepEval Discord**: Regular contributor, answer questions about statistical eval
- **r/MachineLearning**: Helpful comments on eval-related threads
- **AI Engineering Discord** (various servers): Presence
- **Twitter/X**: Regular posts on LLM eval statistics (build authority before launching)

### For Quant Audience
- **r/quant**: Answer overfitting questions, reference Noisefloor where relevant
- **QuantLib mailing list**: Announce the library
- **WorldQuant BRAIN community**: India-specific outreach
- **QuantInsti community**: India algo trading education platform
- **EliteTrader forums**: Known appetite for "your backtest is fake" content

## 6.3 GitHub Strategy

- **README**: Absolutely critical. Must have: 1-line install, 5-line demo, output screenshot, link to docs
- **Topics/tags**: `llm-evaluation`, `statistical-testing`, `backtesting`, `quantitative-finance`, `sprt`, `bootstrap`, `multiple-testing`
- **Releasing**: Tag v0.1.0 as soon as Week 4 code is publishable
- **Issues**: Respond to every issue within 24 hours for first 6 months — community trust is built here
- **CONTRIBUTING.md**: Make it easy for others to contribute — especially integration wrappers
- **Notebooks**: Jupyter notebooks in `examples/` directory — these get stars too

---

# PART 7: MONETIZATION

## 7.1 Revenue Model (Staged)

### Stage 1: MIT Open-Source (Month 1-6)
Build community. Zero revenue. Every user is a future customer.

### Stage 2: Cloud API (Month 4-12)
**Product:** `api.noisefloor.dev`

| Tier | Price | Features |
|---|---|---|
| Free | $0/month | 100 calls/month, no persistence, no signing |
| Developer | $29/month | 5000 calls/month, persistent trial tracking |
| Team | $99/month | 25000 calls/month, team dashboards, signed verdicts |
| Enterprise | Custom | Unlimited, SSO, audit reports, on-premise option |

**Why this pricing:**
- AlphaAssay charges $0.05/call for quant audits — that's $50/1000 calls
- Our Developer tier ($29 for 5000 calls = $0.006/call) is dramatically cheaper
- Trial tracking is the upsell — can't be replicated by the open-source library

### Stage 3: Enterprise Reports (Month 9-24)
**Product:** Noisefloor Compliance Reports

- SEBI algo validation report: ₹50,000–₹2,00,000 per strategy
- SR 26-2 (Federal Reserve) AI model risk evidence pack: $2,000–$10,000 per model
- These are high-ACV, low-volume — good margin

### Stage 4: B2B SaaS (Month 12-36)
- Integration into LLM eval platforms as statistical backend (revenue share)
- White-label for enterprise AI teams
- Training/workshops on statistical eval practices

## 7.2 Why Open-Source First

1. **Trust**: Statistical tools must be auditable. Closed-source statistical engine has zero credibility.
2. **Distribution**: OSS gets PyPI downloads, GitHub stars, blog posts, citations — paid product doesn't
3. **Feedback**: Real users find bugs and edge cases before enterprise contracts depend on correctness
4. **Moat**: By the time competitors build, Noisefloor is the default citation in 50+ papers and blog posts

---

# PART 8: COMPETITIVE ANALYSIS (DETAILED)

## AI Eval Side

| Tool | Stars | What They Do | What They Don't Do | Threat Level |
|---|---|---|---|---|
| **evalstats** | ~200 | Bootstrap CIs, comparative p-values for LLM eval | No quant features, no SPRT, no power analysis, narrow | Low (too narrow) |
| **DeepEval** | 16.7K | Eval metrics, pytest integration, LLM-as-judge | No statistical inference — point estimates only | Medium (could add, hasn't) |
| **Ragas** | 14.7K | RAG evaluation metrics | No statistical inference | Low (metrics-focused, not stats) |
| **Braintrust** | closed | Eval + experiment tracking | Closed-source, no CIs | Medium (could ship stats layer) |
| **LangSmith** | closed | Tracing + eval | Closed-source, vendor-locked to LangChain | Low (ecosystem lock-in, not statistical) |
| **Promptfoo** | 22.9K | Red-teaming, eval | Acquired by OpenAI, pivoting to security | Low (different direction now) |

**Key insight:** Every major eval tool has known about bootstrap CIs and SPRT for years and has not built it. Their architecture is metric-first (accumulate metrics, display them). Noisefloor is verdict-first (run stats, give a decision). These are different architectural approaches, not just feature differences.

## Quant Side

| Tool | Type | What They Do | Gap vs. Noisefloor |
|---|---|---|---|
| **AlphaAssay** | API/Platform | DSR, placebo, robustness, signed verdicts | Platform only (not a library), no AI eval side, $0.05/call |
| **VARRD** | Platform | Quant validation, AI agent integration | Platform, closed-source, no AI eval side |
| **skfolio** | Library | Portfolio optimization, CPCV | CPCV only, no DSR/PBO/IR, portfolio-focused not validation-focused |
| **VectorBT PRO** | Library | Backtesting + some validation | $500 lifetime, bundled inside backtester, no standalone |
| **quantskills/skill-backtest-overfit** | Script | PBO, DSR (basic) | Not maintained, not pip-installable, no AI eval side |
| **mlfinpy** | Library | DSR, PSR, various | Partial implementation, no CPCV, no PBO, not well-maintained |

**Key insight:** None of the quant tools have an AI eval side. None of the AI eval tools have a quant side. Noisefloor is the only product that bridges both with the same API and brand.

---

# PART 9: RISKS AND MITIGATIONS

## Risk 1: Absorption by DeepEval/Braintrust
**Probability:** Medium (30-40% within 18 months)
**What would happen:** DeepEval adds `--confidence-interval` flag. Most users don't switch.
**Mitigation:**
- Build deep integration that makes Noisefloor *inside* DeepEval (not competing with it)
- Own the quant side — impossible for DeepEval to follow
- Build the cloud API persistence moat before they copy the library

## Risk 2: Two-Persona Brand Confusion
**Probability:** Medium (25-35%)
**What would happen:** Quant users think it's for AI people. AI people think it's for quants.
**Mitigation:**
- Separate landing pages: `noisefloor.dev/quant` and `noisefloor.dev/eval`
- Two separate documentation paths (but same PyPI package)
- Same-day split into two packages if this proves to be the problem: `noisefloor-eval` and `noisefloor-quant` that both import from `noisefloor-core`

## Risk 3: Users Don't Care About Statistical Rigor
**Probability:** Low-Medium (15-25%)
**What would happen:** Posts on HN/r/quant get low engagement. Nobody installs.
**Mitigation:**
- This is the #1 thing to validate in the user interviews (Phase 0)
- The "API cost savings" angle for SPRT is immune to this — people care about money even if they don't care about statistics
- Kill condition: < 300 GitHub stars after first two posts

## Risk 4: AlphaAssay or evalstats Builds the Convergent Product
**Probability:** Low (10-15%)
**What would happen:** Direct competitor with first-mover advantage
**Mitigation:**
- Speed to market (4-week V1 for AI eval side)
- Open-source moat — they can't out-OSS us if we're already the standard

## Risk 5: LLM API Costs Collapse to Zero
**Probability:** Low-Medium in 3+ years
**What would happen:** SPRT early stopping less valuable (the cost savings angle weakens)
**Mitigation:**
- SPRT saves time (CI pipeline latency) even if cost → 0
- The statistical rigor angle (don't ship noise) is independent of API costs
- Quant side (DSR/PBO) has no analog to "API cost collapse"

---

# PART 10: RESEARCH AGENDA

## Papers to Read (In Priority Order)

1. **Bailey, D.H., López de Prado, M. (2014).** "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management.* — The foundational paper for the DSR.

2. **Bailey, D.H., Borwein, J., López de Prado, M., Salehipour, A., Zhu, Q.J. (2017).** "The Probability of Backtest Overfitting." *Journal of Computational Finance.* — The PBO paper.

3. **"When Not to Rely on CLT in LLM Evaluation." ICML 2025 position paper.** — The paper that proves current LLM eval methods are statistically broken.

4. **Yin, Miki, Lesnichenko & Gural (March 2026).** "Implementation Risk in Portfolio Backtesting." — The new paper that gives us the IUI, DAF, CSI metrics.

5. **Gibbs, I., Candès, E. (2021).** "Adaptive Conformal Inference Under Distribution Shift." *NeurIPS.* — The ACI paper.

6. **Wald, A. (1947).** "Sequential Analysis." — The SPRT book. Yes, read the original.

7. **Angelopoulos, A.N., Bates, S. (2023).** "Conformal Risk Control." *ICLR.* — For the ACI module.

8. **López de Prado, M. (2018).** "Advances in Financial Machine Learning." *Wiley.* — Chapters 6-8 on CPCV and cross-validation for finance.

## Code to Study

1. **`evalstats`** (pip install evalstats): Study what they've built and where the gaps are. Our API must be a superset.

2. **`skfolio` CPCV implementation**: `skfolio/model_selection/_combinatorial.py`. Understand their approach, identify any correctness issues.

3. **`mlfinpy`** (if available): Study their DSR implementation for reference.

4. **`pypbo`**: Study the PBO CSCV implementation.

5. **`sprtt`** (Python port): Study the SPRT implementation for inspiration.

6. **AlphaAssay API docs**: What does their $0.05/call actually return? Document the gap.

---

# PART 11: NOTES AND BUILDER WISDOM

## Critical Implementation Notes

> **Note 1:** The BCa bootstrap requires at least n ≥ 10 samples to be meaningful. Below n=10, warn the user. Below n=5, refuse to compute and tell them why.

> **Note 2:** SPRT has no guaranteed termination when the true effect size is exactly between H₀ and H₁. Always enforce `max_samples` as a hard limit.

> **Note 3:** The DSR formula uses the Eubank-Gordon-Rojo estimate for the expected maximum Sharpe from m trials. This estimate assumes normally-distributed Sharpe ratios — it's an approximation. Document this clearly. For very non-normal return distributions, the PBO (which is simulation-based) is more reliable.

> **Note 4:** The CPCV implementation is computationally expensive. C(16,8) = 12,870 combinations × n observations can be slow. Default to n_jobs=-1 with joblib. Also cache results when inputs haven't changed (use hashlib on returns data).

> **Note 5:** The verdict strings must be calibrated carefully. "DO NOT SHIP" is strong language. Consider: "LIKELY NOT SIGNIFICANT" instead. But also consider that weak language causes people to ship noise anyway. User test which framing drives better behavior.

> **Note 6:** Never claim that Noisefloor proves something is real or fake. It computes the *probability* based on statistical assumptions. Document this caveat in every function's docstring.

## What Makes This Hard (And Where To Be Careful)

1. **Correctness matters more than usual.** A bug in a UI is embarrassing. A bug in a statistical testing library can cause a quant to allocate capital incorrectly or a team to ship a degraded model. Every function needs rigorous testing.

2. **The BCa bootstrap is subtle.** The acceleration factor computation (jackknife influence values) is easy to get wrong. Cross-check against `scipy.stats.bootstrap` on every output.

3. **Financial time-series are non-IID.** Standard bootstrap assumes IID data. For the quant module, must use block bootstrap or moving-block bootstrap. Noisefloor should support both.

4. **p-values are not what most people think they are.** The README must include a clear explanation of what p < 0.05 means and doesn't mean. Misinterpretation of p-values is rampant. Noisefloor's "verdict" system is partly designed to avoid this — but must be careful not to create new misinterpretations.

5. **The "auto" method selection in `compare_evals` must be documented.** If the user doesn't know which method was chosen and why, they can't trust the result.

## Observations About The Competitive Landscape

The most interesting competitive observation: **AlphaAssay exists, charges money, and people use it.** This is the strongest possible evidence that WTP exists for quant validation-as-a-service. They proved the market. Noisefloor's job on the quant side is to be the open-source, pip-installable, no-cost-per-call version — and then monetize on cloud API with persistence.

The LangSmith lesson: LangSmith proved that LLM observability is valuable by getting 50K+ users before monetizing. Noisefloor should follow the same pattern — get the users on the open-source library, then monetize through cloud services.

## The One Metric That Matters

**GitHub stars at 30 days, 90 days, and 1 year.**

- 30 days: 200+ stars → blog posts worked, keep going
- 90 days: 1000+ stars → product-market fit signal, start cloud API
- 1 year: 5000+ stars → category leader position, start enterprise sales

If at 30 days you have <100 stars: the content strategy isn't working. Pivot the messaging (try the "save API costs" angle instead of the "statistical rigor" angle).

---

# APPENDIX A: JSONL Format Specifications

Input format for `compare_evals` via CLI:
```json
{"sample_id": "q1", "score": 0.73, "passed": true, "latency_ms": 234}
{"sample_id": "q2", "score": 0.45, "passed": false, "latency_ms": 891}
```

Input format for `audit_backtest` via CLI:
```json
{"date": "2023-01-03", "return": 0.0023}
{"date": "2023-01-04", "return": -0.0041}
```

Output format (cloud API):
```json
{
  "verdict": "DO NOT SHIP",
  "p_value": 0.134,
  "ci_lower": -0.008,
  "ci_upper": 0.062,
  "effect_size": 0.31,
  "power": 0.42,
  "min_n_needed": 187,
  "method": "bootstrap_bca",
  "noisefloor_version": "0.3.1",
  "timestamp": "2026-07-15T02:31:30Z",
  "signature": "ed25519:abc123..."  # cloud API only
}
```

# APPENDIX B: Synthetic Test Data Generation

For testing, generate known-overfit strategies:
```python
import numpy as np

def generate_overfit_strategy(n_obs=500, true_sharpe=0.0, seed=42):
    """Generate a strategy that looks good IS but is pure noise OOS."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n_obs)  # true Sharpe = 0
    return returns

def generate_real_strategy(n_obs=500, true_sharpe=1.5, seed=42):
    """Generate a strategy with genuine positive expected return."""
    rng = np.random.default_rng(seed)
    daily_return = true_sharpe * 0.01 / np.sqrt(252)
    returns = rng.normal(daily_return, 0.01, n_obs)
    return returns
```

# APPENDIX C: Key Papers Reference Table

| Paper | Year | Relevance | Key Formula |
|---|---|---|---|
| Wald (1947) | 1947 | SPRT foundations | Likelihood ratio boundaries |
| Bailey & López de Prado (2012) | 2012 | PSR | Φ[(√T-1)(SR-SR*)/√(...)] |
| Bailey & López de Prado (2014) | 2014 | DSR | Expected max Sharpe from m trials |
| Bailey et al. (2017) | 2017 | PBO/CSCV | CSCV algorithm |
| López de Prado (2018) | 2018 | CPCV | Purging + embargoing |
| Gibbs & Candès (2021) | 2021 | ACI | Adaptive threshold update |
| ICML (2025) | 2025 | LLM eval statistics | CLT failure modes |
| Yin et al. (2026) | 2026 | Implementation Risk | IUI, DAF, CSI |
