# Noisefloor

**The convergent statistical verdict engine for AI decisions and quantitative backtesting.**

Is your LLM improvement real, or is it statistical noise? 
Is your 1.8 Sharpe ratio strategy real, or is it overfit to the test set?

Noisefloor answers these questions with mathematical certainty, providing clear "SHIP" / "DO NOT SHIP" verdicts using sequential testing, bootstrap confidence intervals, and multiple-comparison correction.

## Installation

```bash
pip install noisefloor
```

For quant features:
```bash
pip install noisefloor[quant]
```

## Features
- **AI/LLM Evaluation:** Compare prompts, models, and RAG configs using robust bootstrap confidence intervals.
- **Sequential Testing (SPRT):** Save up to 80% on API costs by stopping evaluations early when a verdict is already statistically guaranteed.
- **Quant Backtest Validation:** Calculate the Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO) using Combinatorial Purged Cross-Validation (CPCV).
- **Integrations:** Integrates natively with `pytest` for CI/CD gates.

## Usage

```bash
noisefloor compare baseline.jsonl candidate.jsonl --method bootstrap --alpha 0.05
```
