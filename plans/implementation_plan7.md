# Comprehensive Technical & Mathematical Implementation Plan: `sprt-eval`

This document details the architecture, mathematical formulations, execution lifecycle, and research directives for **`sprt-eval`**—a statistics-native compute governor and early-stopping test runner for LLMs and AI agents in CI/CD pipelines.

---

## 1. Executive Vision: The "What" & "Why"

### What is `sprt-eval`?
`sprt-eval` is an open-source, local-first test runner and pytest plugin. It streams evaluation cases and executes them in parallel, continuously feeding outputs into a sequential statistical decision engine. The moment the candidate prompt/model is proven to be statistically different (better or worse) than the baseline, `sprt-eval` cancels all active API workers and terminates the test run, outputting a rigorous verdict.

```
                  [ Code Change / Prompt Edit ]
                               │
                               ▼
                    [ sprt-eval CLI Runner ]
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[Pytest Plugin Engine]                     [Stochastic Evaluation Stream]
   - Intercept Test Loops                     - Rate Limit Adaptive Queue
   - Hook pytest_runtestloop                  - Parallel LLM Judge Calls
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
                  [Sequential Decision Engine]
                     - Wald's Binomial/Normal SPRT
                     - Siegmund Boundary Corrections
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
            [Boundary Crossed]   [Within Boundaries]
                     │                   │
         ┌───────────┴───────────┐       └──► [Fetch Next Mini-Batch]
         ▼                       ▼
   [Halt Signal]        [Cancel Workers]
         │
         ▼
[Print Savings & Exit]
```

### Why is it Required? (The Point-Estimate Fallacy)
Traditional unit testing is deterministic: a function outputs `x`, and we assert `x == target`. LLM and agent outputs are probabilistic. 
Currently, AI engineers try to evaluate changes by running fixed-size evaluation runs (e.g., 200 or 500 test cases) on every commit, using frontier models as judges. This results in:
1.  **Exorbitant API Costs & Latency:** Running 500 multi-turn agent test cases can take 15 minutes and cost $30 in token usage per commit.
2.  **Statistically Invalid Decisions:** Developers deploy prompts based on a "vibe check" or a marginal raw score improvement (e.g., 81% to 83% accuracy) that is statistically indistinguishable from noise.
3.  **Wasted Compute:** If a code change introduces a catastrophic parsing bug, standard pipelines still run all 500 cases to the end.

`sprt-eval` solves this by treating evaluation as a streaming sequential hypothesis test. It mathematically determines the minimum number of samples needed to reach a decision, cutting evaluation costs and runtime by up to 80% while establishing formal statistical confidence.

---

## 2. Statistical & Mathematical Research Requirements

To build a statistically sound runner, we must address four research topics:

### Research Area 1: Wald's Sequential Probability Ratio Test (SPRT)
We must implement sequential hypothesis testing for two distinct data distributions:

#### 1. Binomial SPRT (Pass/Fail Outcomes)
Used for binary assertions (e.g., schema matches, regex hits, exact equality checks).
*   **Hypotheses:** $H_0: p = p_0$ (baseline accuracy) vs. $H_1: p = p_1$ (candidate target accuracy or regression threshold).
*   **Likelihood Ratio (LR):** For a test outcome $x_i \in \{0, 1\}$ (success/failure):
    $$\Lambda(x_i) = \left(\frac{p_1}{p_0}\right)^{x_i} \left(\frac{1 - p_1}{1 - p_0}\right)^{1 - x_i}$$
*   **Cumulative Log-Likelihood Ratio ($S_n$):**
    $$S_n = S_{n-1} + x_i \ln\left(\frac{p_1}{p_0}\right) + (1 - x_i) \ln\left(\frac{1 - p_1}{1 - p_0}\right)$$

#### 2. Normal SPRT (Continuous Scores)
Used for continuous evaluation ratings (e.g., semantic similarity, LLM-as-a-judge score cards on a 0-1 scale).
*   **Hypotheses:** $H_0: \mu = \mu_0$ vs. $H_1: \mu = \mu_1$ with assumed variance $\sigma^2$.
*   **Cumulative Log-Likelihood Ratio ($S_n$):**
    $$S_n = S_{n-1} + \frac{\mu_1 - \mu_0}{\sigma^2} \left(x_i - \frac{\mu_0 + \mu_1}{2}\right)$$

### Research Area 2: Siegmund's Boundary Correction (Overshoot Mitigation)
Because test cases are evaluated in discrete steps (or mini-batches), the score $S_n$ will overshoot the decision boundaries. Naive boundaries:
$$A = \ln \left(\frac{\beta}{1 - \alpha}\right) \quad \text{and} \quad B = \ln \left(\frac{1 - \beta}{\alpha}\right)$$
will result in actual error rates exceeding the nominal target rates $\alpha$ and $\beta$. 

*   **Research Task:** Implement Siegmund's correction to adjust the boundaries based on the expected overshoot:
    $$A^* = A + \rho \quad \text{and} \quad B^* = B - \rho$$
    where $\rho \approx 0.583$ for standard normal variables (or computed dynamically based on the likelihood ratio variance).

### Research Area 3: BCa Bootstrap Confidence Intervals
Upon early termination, the developer needs to know the candidate's estimated accuracy and its uncertainty range. Since the sample size is small and dynamically determined (truncated), standard Central Limit Theorem intervals are invalid.

*   **Research Task:** Implement the **Bias-Corrected and Accelerated (BCa) Bootstrap** method. This corrects for bias (deviation of the bootstrap distribution median from the sample estimate) and acceleration (skewness of the distribution) to output exact, distribution-free confidence intervals.

### Research Area 4: Multiple Testing Corrections
If a user compares multiple candidates in parallel (e.g., testing 3 prompt variations against a baseline), the probability of a false positive scales rapidly.
*   **Research Task:** Implement Holm-Bonferroni and Benjamini-Hochberg false discovery rate (FDR) corrections to adjust the boundaries when executing multi-arm candidate evaluations.

---

## 3. Concurrency & Integration Engineering

### Component 1: Pytest Test Loop Interception
To support early stopping, we cannot rely on pytest’s default sequential execution. We must intercept the pytest run loop:
*   **Hook to Implement:** `pytest_runtestloop`. 
*   **Mechanism:** Override this hook to run a custom async loop. The loop pulls test cases from the pytest collection, schedules them in batches to an async worker queue, feeds results to `stats.py`, and cancels pending tasks using `asyncio.Task.cancel()` when a boundary is crossed.

### Component 2: Rate Limit and Throttle Coordination
Parallel LLM judge calls will rapidly hit token-per-minute (TPM) and request-per-minute (RPM) limits. 
*   **Mechanism:** Implement an adaptive concurrency queue utilizing an Additive-Increase/Multiplicative-Decrease (AIMD) throttling algorithm. If the LLM provider returns a `429 Rate Limit` or `Retry-After` header, the queue manager must dynamically reduce concurrency and reschedule the failed test cases.

### Component 3: Conformal LLM Judge Calibration
LLM judges are themselves non-deterministic.
*   **Mechanism:** PayerProof maintains a calibration database of 100 human-graded outcomes. If the LLM judge grades a test case, the engine calculates the conformal prediction interval around the grade. If the interval is wide, the test outcome is flagged as "statistically ambiguous" and down-weighted in the cumulative SPRT calculation.

---

## 4. Proposed Folder Layout & Blueprints

```
sprt-eval/
├── core/
│   ├── __init__.py
│   ├── stats.py         # Binomial/Normal SPRT and BCa Bootstrap calculations
│   └── runner.py        # Async batch queue manager with AIMD throttle
├── judges/
│   ├── __init__.py
│   ├── judge.py         # LLM Judge interface with Conformal calibration
│   └── rate_limiter.py  # Adaptive rate-limit backoff handler
├── pytest_plugin/
│   ├── __init__.py
│   └── plugin.py        # Pytest runtestloop hook override
└── cli/
    ├── __init__.py
    └── main.py          # CLI parser and terminal reporter
```

### [NEW] [stats.py](file:///E:/CODE_TOP/Project/sprt_eval/core/stats.py)
```python
# stats.py
import numpy as np

class WaldSPRT:
    def __init__(self, p0, p1, alpha=0.05, beta=0.10, variance=0.25, metric_type="binomial"):
        self.p0 = p0
        self.p1 = p1
        self.metric_type = metric_type
        self.variance = variance
        
        # Siegmund boundary correction factor
        rho = 0.583 
        
        # Calculate nominal boundaries
        raw_a = np.log(beta / (1 - alpha))
        raw_b = np.log((1 - beta) / alpha)
        
        # Apply Siegmund correction
        self.a = raw_a + (rho if metric_type == "normal" else 0)
        self.b = raw_b - (rho if metric_type == "normal" else 0)
        
        if metric_type == "binomial":
            self.log_ratio_success = np.log(p1 / p0)
            self.log_ratio_failure = np.log((1 - p1) / (1 - p0))
            
        self.score = 0.0
        self.samples = []

    def update(self, val):
        self.samples.append(val)
        if self.metric_type == "binomial":
            step = self.log_ratio_success if val == 1 else self.log_ratio_failure
        else:  # Normal distribution
            step = ((self.p1 - self.p0) / self.variance) * (val - (self.p0 + self.p1) / 2.0)
            
        self.score += step
        
        if self.score >= self.b:
            return "UPGRADE"
        elif self.score <= self.a:
            return "REGRESSION"
        return "CONTINUE"

    def compute_bca_ci(self, confidence=0.95, num_bootstrap=2000):
        # Implement Bias-Corrected and Accelerated Bootstrap
        data = np.array(self.samples)
        n = len(data)
        if n < 5:
            return np.mean(data), np.mean(data)
            
        bootstrap_means = np.zeros(num_bootstrap)
        for i in range(num_bootstrap):
            resample = np.random.choice(data, size=n, replace=True)
            bootstrap_means[i] = np.mean(resample)
            
        theta_hat = np.mean(data)
        z0 = norm_inv(np.mean(bootstrap_means < theta_hat))
        
        # Calculate acceleration parameter via jackknife
        jackknife_means = np.zeros(n)
        for i in range(n):
            jackknife_means[i] = np.mean(np.delete(data, i))
        mean_jack = np.mean(jackknife_means)
        num = np.sum((mean_jack - jackknife_means) ** 3)
        den = 6 * (np.sum((mean_jack - jackknife_means) ** 2) ** 1.5)
        a = num / (den + 1e-12)
        
        alpha_l = norm_cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        alpha_u = norm_cdf(z0 + (z0 - z_alpha) / (1 - a * (z0 - z_alpha)))
        
        ci_lower = np.percentile(bootstrap_means, alpha_l * 100)
        ci_upper = np.percentile(bootstrap_means, alpha_u * 100)
        return ci_lower, ci_upper
```

---

### [NEW] [plugin.py](file:///E:/CODE_TOP/Project/sprt_eval/pytest_plugin/plugin.py)
```python
# plugin.py
import pytest
import asyncio
from sprt_eval.core.stats import WaldSPRT
from sprt_eval.core.runner import AsyncQueueRunner

def pytest_addoption(parser):
    group = parser.getgroup("sprt-eval")
    group.addoption("--sprt", action="store_true", help="Enable SPRT early stopping")
    group.addoption("--sprt-baseline", type=float, default=0.80, help="Baseline expected accuracy")
    group.addoption("--sprt-alpha", type=float, default=0.05, help="Target Type I error rate")

@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if not session.config.getoption("--sprt"):
        return None  # Fall back to default pytest run loop

    loop = asyncio.get_event_loop()
    session.results = loop.run_until_complete(run_sprt_test_loop(session))
    return True  # Stop further execution of default pytest loop

async def run_sprt_test_loop(session):
    baseline = session.config.getoption("--sprt-baseline")
    alpha = session.config.getoption("--sprt-alpha")
    
    # Initialize SPRT: assume p0 = baseline, p1 = baseline - 0.10 (detect 10% regression)
    sprt = WaldSPRT(p0=baseline, p1=baseline - 0.10, alpha=alpha, beta=0.10)
    runner = AsyncQueueRunner(session.items, sprt)
    
    verdict = await runner.start_stream()
    return verdict
```

---

## 5. Construction Steps

### Phase 1: Pure Statistical Engine (Weeks 1-4)
*   **Goal:** Build the mathematical backend.
*   **Steps:**
    1.  Implement `stats.py` with Binomial and Normal sequential estimators.
    2.  Write Siegmund's boundary correction formulas.
    3.  Implement BCa bootstrapping logic for accurate confidence intervals at arbitrary termination points.
    4.  Verify statistics using a simulation harness: generate 10,000 synthetic paths with known drift parameters and verify that the Type I/II error rates match target bounds.

### Phase 2: Pytest Lifecycle Hook Integration (Weeks 5-8)
*   **Goal:** Intercept and parallelize test runner execution.
*   **Steps:**
    1.  Override `pytest_runtestloop` in `plugin.py` to support asynchronous execution.
    2.  Implement `runner.py` using `asyncio` to manage parallel worker pools.
    3.  Add support for command-line arguments to configure baseline target accuracy, Type I error, and Type II error.

### Phase 3: Rate Limiting & Concurrency Queue (Weeks 9-12)
*   **Goal:** Prevent rate limit errors when executing parallel judge calls.
*   **Steps:**
    1.  Build the AIMD throttling queue in `rate_limiter.py`.
    2.  Intercept HTTP error codes (e.g., `429 Too Many Requests`) from LLM judge APIs.
    3.  Implement backoff delays and automatically reschedule failed evaluations.

### Phase 4: Conformal LLM Judge Calibrator (Weeks 13-16)
*   **Goal:** Calibrate non-deterministic LLM-as-a-judge outputs.
*   **Steps:**
    1.  Implement `judge.py` to compare LLM grades with a small human-graded golden dataset.
    2.  Map output scores to conformal validation ranges.
    3.  Downgrade the impact of ambiguous evaluations in the cumulative SPRT calculation.

### Phase 5: CLI Engine & CI/CD Pipeline Integration (Weeks 17-20)
*   **Goal:** Build the GitHub Actions CI connector.
*   **Steps:**
    1.  Configure CLI commands: `sprt-eval run --baseline=x.jsonl --candidate=y.jsonl`.
    2.  Implement exit status codes to support automated PR checks (exit code `0` for no regression, exit code `1` for proven regression).
    3.  Build terminal report output to display savings statistics and BCa confidence ranges.

---

## 6. Verification & Test Plan

### Automated Simulation Tests
We will verify `sprt-eval` using synthetic runs:
1.  **Wald boundary validation:**
    *   Simulate a candidate model with a true accuracy of 70% against a baseline of 80% (regression).
    *   Assert that `sprt-eval` stops the evaluation early and outputs a "REGRESSION" verdict.
    *   Measure and verify that the average sample path length (ASN) is reduced by at least 60% compared to a fixed run of 500.
2.  **Concurrency checks:**
    *   Verify that if the halt boundary is crossed on sample 20 of a 500-sample queue, all async workers scheduled for samples 21–500 are cancelled immediately, preventing network payload executions.

### Manual Verification
1.  **CI/CD Pipeline Run:**
    *   Create a demo repository using GitHub Actions.
    *   Add a test step using `sprt-eval`. Introduce a bug in the prompt helper logic.
    *   Verify that the GitHub Action terminates the test execution early, displays the statistics on the runner terminal, and exits with code `1`, blocking the pull request.
