# Implementation Plan: `sprt-eval` (Statistical LLM Evaluation)

**Goal:** Build **`sprt-eval`**, a developer-first, pytest-native statistical utility that uses Wald's Sequential Probability Ratio Test (SPRT) to dynamically halt LLM and AI agent evaluations. 

This document serves as a **standalone, highly detailed encyclopedia** for this product concept. It is designed so that even if the original research reports are lost, a founder can read this and understand the entire product, the problem space, and the statistical math behind it.

---

## 1. The "What" (Extensive Product Definition)

To understand what `sprt-eval` is, we must first understand why the current status quo of testing AI is fundamentally broken.

### 1.1 The Status Quo: Point Estimates and Wasted Money
In traditional software, a unit test is deterministic: you run it once, it passes or fails, and you move on. 
Large Language Models (LLMs) and AI Agents are non-deterministic. If you change a prompt, you cannot run one test case. You must run the new prompt against a dataset of 100, 500, or 1,000 test cases to see if the overall accuracy improved. 
Currently, developers do this naively:
1. They run the old prompt on 500 test cases and get **82% accuracy**.
2. They write a new prompt, run it on 500 test cases, and get **84% accuracy**.
3. They merge the PR, assuming it's better.

**The Problem:** Because they didn't calculate a Confidence Interval (CI) or a p-value, that 2% difference is likely just statistical noise. More importantly, running 500 test cases through GPT-4 takes **2 hours of CI/CD compute time** and costs **$50 in API credits** *per pull request*. 

### 1.2 The "Peeking" Penalty (Why Developers Can't Just Write a `break` Statement)
Developers often ask: *"Why run all 500 tests? If the new prompt fails the first 20 tests, I know it's garbage. I'll just write a python script that checks the score after every 10 tests and `break`s the loop if it's failing."*

In statistics, this is called **"Peeking."** If you peek at data during an experiment and stop early without adjusting your math, you drastically inflate your False Positive (Type I) error rate. You will frequently merge prompts that look good early on but are actually terrible.

### 1.3 The Product: `sprt-eval`
`sprt-eval` is a library that mathematically solves the Peeking Penalty using Abraham Wald's **Sequential Probability Ratio Test (SPRT)**—a technique invented during WWII to test the quality of military munitions without blowing all of them up.

SPRT allows you to check the results *after every single test*. It tracks the random walk of the LLM's performance. If the performance hits a mathematically calculated "Upper Boundary," it stops the test and declares the prompt a **Success**. If it hits the "Lower Boundary," it stops the test and declares it a **Failure**. 

### 1.4 The User Journey (Concrete Example)
Imagine a developer is building a RAG (Retrieval-Augmented Generation) bot. They tweak the system prompt and push a PR.
1. **The Setup:** In their GitHub Actions pipeline, they run `pytest --sprt --alpha=0.05 --beta=0.05 --baseline=0.80`. (This means: "I want 95% confidence that this new prompt is better than my old 80% baseline.")
2. **The Execution:** The evaluation begins. 
   - Test 1: Fail.
   - Test 2: Fail.
   - Test 3: Pass.
   - Test 4-12: Fail.
3. **The Early Stop:** The new prompt is doing terribly. At Test 12, the internal SPRT engine calculates the log-likelihood ratio. The math proves that it is *statistically impossible* (within the 5% error bound) for this prompt to eventually reach the 80% baseline, even if it aced the remaining 488 tests. 
4. **The Result:** `sprt-eval` immediately halts the test suite. It prints a beautiful graph to the terminal showing the random walk crossing the failure boundary. The CI pipeline fails in **3 minutes** instead of 2 hours, saving the developer **488 unnecessary API calls**.

---

## 2. The "Why" (Market & Moat)

**Why is this the absolute #1 opportunity for a Constrained / Solo Founder?**

1. **Immediate Developer Pain (API Cost):** Developers hate waiting for CI pipelines, and engineering managers hate massive OpenAI/Anthropic bills. `sprt-eval` provides a "60-second Time-To-First-Value" (TTFV). A developer pip-installs it, runs their eval, and immediately sees their AWS/OpenAI bill drop by 80%. 
2. **Open-Source Virality:** This is a pure infrastructure primitive. It does not require a sales team. It spreads via word-of-mouth on HackerNews, Twitter, and developer blogs ("How we cut our LLM Eval costs by 80% using SPRT").
3. **A Math Moat, Not a SaaS Moat:** Large platforms (LangSmith, Braintrust) currently focus on displaying dashboards and traces. Very few of their engineers are deeply focused on statistical theory (Wald's sequential analysis, Combinatorial Purged Cross-Validation, etc.). Building a perfectly rigorous, open-source statistical standard forces those giants to integrate *your* library rather than building it themselves.

---

## 3. Exhaustive Use Cases & Feature Set

### 3.1 CI/CD Regression Testing (The Wedge)
As described in the user journey above, this is the primary go-to-market. A Pytest plugin that wraps existing evaluation scripts.

### 3.2 Self-Consistency Early Stopping (Production Compute Efficiency)
*   **The Concept:** Advanced LLM apps use "Self-Consistency" (asking the LLM to solve a complex math problem 15 times and taking the majority vote). 
*   **The SPRT Application:** `sprt-eval` can be imported directly into production app code. If the LLM outputs the exact same answer for the first 4 tries, SPRT can mathematically prove that the remaining 11 tries cannot possibly change the majority vote. It halts generation instantly, saving massive token costs for live users.

### 3.3 Multi-Agent Debate Governor
*   **The Concept:** When using frameworks like AutoGen, two agents debate an answer. Sometimes they get stuck in an infinite loop.
*   **The SPRT Application:** `sprt-eval` acts as a compute governor. It monitors the log-likelihood ratio of the agents converging. Once the debate crosses the statistical threshold of "useful convergence," it terminates the debate loop.

---

## 4. Architecture & Technical Stack

### 4.1 The SPRT Mathematical Core (Python/Rust)
*   **Distributions Supported:** 
    *   **Bernoulli/Binomial:** For binary pass/fail evals (e.g., "Did the LLM output valid JSON?").
    *   **Categorical/Multinomial:** For multiple-choice evaluations or self-consistency clustering.
*   **Threshold Engine:** Calculates the upper bound $A$ and lower bound $B$ based on user-defined Type I ($\alpha$) and Type II ($\beta$) error tolerances.

### 4.2 The Test Runner Integrations
*   **Pytest Plugin:** `pytest-sprt` handles the orchestration, overriding standard test collection to run dynamically.

### 4.3 Telemetry & Logging
*   **Integrations:** Native export to LangSmith, Braintrust, and Langfuse to visually display *why* the evaluation stopped early (showing the log-likelihood random walk crossing the boundary).

---

## 5. Phased Implementation Strategy

### Phase 1: The Core Library (Weeks 1-2)
*   Implement the pure mathematical logic (`sprt.calculate_ll_ratio`, `sprt.check_boundaries`).
*   Publish the core `sprt-eval` library to PyPI.
*   **Validation:** Write exhaustive unit tests verifying the mathematical boundaries against known academic datasets to ensure zero statistical flaws.

### Phase 2: The Developer UX (Weeks 3-4)
*   Build the `pytest-sprt` plugin.
*   Create a beautiful CLI output that shows a dynamic progress bar and a visualization of the SPRT random walk hitting the threshold.

### Phase 3: The Orchestration Hooks (Weeks 5-6)
*   Build LangChain and LlamaIndex callback handlers for production self-consistency stopping.

---

## 6. Final Notes & Edge Cases (The "Gotchas")

If you build this, you must engineer around these three critical statistical traps:

1. **Item Selection Bias (The "Easy First" Problem):** In standard evals, developers run test cases in a fixed order. If the first 20 test cases happen to be incredibly easy, SPRT might stop the test early with a "Pass", completely ignoring the hard test cases at the end of the dataset. **Solution:** `sprt-eval` MUST forcibly implement randomized shuffling of the dataset before execution to ensure I.I.D. (Independent and Identically Distributed) sampling.
2. **The "Cost vs. Variance" Trap:** If the LLM's performance is highly volatile and the new prompt is only *marginally* better than the old one, the SPRT random walk will bounce around the center line and never hit the early-stopping boundaries. **Solution:** The library must include a `max_steps` truncation failsafe so it doesn't run infinitely.
3. **Calibration Blindness:** SPRT assumes that the underlying evaluations (e.g., if you use "LLM-as-a-judge" to score the outputs) are perfectly calibrated. If the judge LLM has a systematic bias, SPRT will confidently and rapidly stop the test based on biased data. This isn't a flaw in SPRT, but a flaw in the user's setup. **Solution:** Write extensive documentation warning users that SPRT accelerates decisions based on the *assumption* that their scoring metric is accurate.

---

> [!IMPORTANT]
> ## Open Questions for the User
> 1. **Language Choice:** Python is mandatory for the Pytest plugin, but should we write the core mathematical engine in **Rust** (and bind to Python via PyO3) to ensure it can be easily ported to TypeScript/Node.js later?
> 2. **Monetization Strategy:** The CLI and Pytest plugin must be free and open-source (MIT License) to get viral adoption. However, do you want to plan for a Cloud Dashboard (e.g., `sprt-cloud`) where teams can track their historical API cost savings over time?
> 3. **Initial Focus:** Should Phase 1 strictly target **CI/CD Regression Testing** (evaluating prompts during development), or should we pivot to prioritize **Self-Consistency Early Stopping** (saving costs in production)?
