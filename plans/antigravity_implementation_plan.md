# Goal: Build "Noisefloor" (formerly sprt-eval)

This plan outlines the implementation of **Noisefloor**, the convergent statistical verdict engine for AI decisions and quantitative backtesting. It synthesizes all requested features across the provided implementation plans, giving preference to the architecture and naming conventions in `noisefloor_implementation_plan.md` while ensuring no features from the other plans (like `sprt-eval`'s AIMD concurrency throttling, conformal LLM judge calibration, and async runner) are missed.

## User Review Required

> [!CAUTION]
> **Project Name and Directory Structure:** The preferred plan `noisefloor_implementation_plan.md` rebrands the project from `sprt-eval` to `noisefloor` and changes the main package directory from `sprt_eval` to `noisefloor`. The root project folder is still `E:\CODE_TOP\Project\sprt-eval`, but the Python package inside it will be `noisefloor`. Please confirm this is acceptable.

> [!IMPORTANT]
> **Feature Synthesis:** To ensure no features are missed from `plan7`, I have added an `llm_judges/` directory (for conformal calibration) and a `runner/` directory (for async AIMD rate-limiting/concurrency) into the `noisefloor` package structure. These are critical for the `sprt-eval` use cases.

## Proposed Changes

We will create the Python project structure inside `E:\CODE_TOP\Project\sprt-eval`.

### Project Scaffolding
#### [NEW] `pyproject.toml`
Defines the `noisefloor` package, its dependencies (`numpy`, `scipy`, `pandas`, `aiohttp` for async runners), and the CLI entry points.
#### [NEW] `README.md`
Documentation for the project, covering both AI Eval and Quant workflows.

---

### Module 1: Core Statistics & Utilities (`noisefloor/core/`)
Provides shared mathematical primitives used by both AI Eval and Quant modules.
#### [NEW] `noisefloor/core/__init__.py`
#### [NEW] `noisefloor/core/effect_size.py` (Cohen's d, Hedge's g)
#### [NEW] `noisefloor/core/aci.py` (Adaptive Conformal Inference)
#### [NEW] `noisefloor/core/verdicts.py` (Verdict system dataclasses and logic)
#### [NEW] `noisefloor/core/utils.py`

---

### Module 2: AI Eval Engine (`noisefloor/`)
Top-level modules for AI/LLM evaluation.
#### [NEW] `noisefloor/__init__.py`
#### [NEW] `noisefloor/bootstrap.py` (BCa bootstrap CI implementation)
#### [NEW] `noisefloor/sprt.py` (Wald's SPRT logic, Siegmund's Boundary Correction)
#### [NEW] `noisefloor/power.py` (Power analysis)
#### [NEW] `noisefloor/multiple.py` (Holm-Bonferroni, Benjamini-Hochberg)
#### [NEW] `noisefloor/compare.py` (`compare_evals` and regression testing)

---

### Module 3: Quantitative Backtest Validation (`noisefloor/quant/`)
Implements quantitative finance validation (DSR, PBO, CPCV).
#### [NEW] `noisefloor/quant/__init__.py`
#### [NEW] `noisefloor/quant/audit.py` (Main `audit_backtest` entry point)
#### [NEW] `noisefloor/quant/dsr.py` (Deflated Sharpe Ratio, PSR)
#### [NEW] `noisefloor/quant/pbo.py` (Probability of Backtest Overfitting)
#### [NEW] `noisefloor/quant/cpcv.py` (Combinatorial Purged Cross-Validation)
#### [NEW] `noisefloor/quant/implementation_risk.py` (Implementation risk metrics)
#### [NEW] `noisefloor/quant/metrics.py` (Sharpe, Calmar, Drawdown, etc.)

---

### Module 4: Asynchronous Execution & LLM Judges (`noisefloor/runner/` & `noisefloor/judges/`)
Brought over from `implementation_plan7.md` to ensure concurrent execution and LLM non-determinism are handled.
#### [NEW] `noisefloor/runner/__init__.py`
#### [NEW] `noisefloor/runner/async_queue.py` (Async queue manager with AIMD throttle for API rate limits)
#### [NEW] `noisefloor/judges/__init__.py`
#### [NEW] `noisefloor/judges/conformal.py` (Conformal LLM Judge Calibrator to down-weight ambiguous LLM outputs)

---

### Module 5: Integrations & CLI (`noisefloor/ci/` & `noisefloor/integrations/`)
#### [NEW] `noisefloor/ci/__init__.py`
#### [NEW] `noisefloor/ci/cli.py` (Click/Typer-based CLI for all features)
#### [NEW] `noisefloor/ci/pytest_plugin.py` (Pytest hook overrides for early stopping via `pytest_runtestloop`)
#### [NEW] `noisefloor/integrations/__init__.py` (Placeholder wrappers for DeepEval, Ragas, skfolio)

---

## Verification Plan

### Automated Tests
We will build a comprehensive test suite in `tests/`:
- **Statistics Tests:** Verify `BCa` outputs against `scipy.stats.bootstrap`. Validate SPRT Type I/II error rates via Monte Carlo simulation (as specified in plan 7).
- **Quant Tests:** Validate CPCV splits and DSR against known synthetic overfit strategies.
- **Concurrency Tests:** Ensure AIMD rate limiter properly backs off on mocked `429 Too Many Requests` responses.

### Manual Verification
- We will write dummy Python scripts using `pytest` to ensure the `pytest_plugin.py` properly hooks into the run loop and stops evaluations early.
- We will run the CLI commands with synthetic `jsonl` files to ensure formatted console output works correctly.
