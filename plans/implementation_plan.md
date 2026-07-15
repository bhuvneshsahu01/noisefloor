# Implementation Plan: `sprt-eval` (Statistical LLM Evaluation)

**Goal:** Build **`sprt-eval`**, a developer-first, pytest-native statistical utility that uses Wald's Sequential Probability Ratio Test (SPRT) to dynamically halt LLM and AI agent evaluations. 

This is the definitive **"Constrained Path" winner**. It is designed for a solo founder or lean team to build rapidly, publish to open-source package managers (PyPI/npm), and achieve viral adoption by solving a universal developer pain point: *The extreme API cost and slowness of testing non-deterministic AI models.*

> [!CAUTION]
> **User Review Required:** This plan defines the technical architecture for a mathematically rigorous statistics library. Please review the "Core Use Cases" and "Edge Cases" sections to ensure the feature scope aligns with your vision before we begin execution.

---

## 1. The Problem Space & Market Dynamics

### 1.1 The "Eval" Bottleneck
When developers change a prompt or swap an LLM (e.g., moving from GPT-4 to Claude 3.5), they need to know if the application got better or worse. Because LLMs are non-deterministic, running a test case once is meaningless. Developers must run evaluations 50 to 100 times to achieve statistical significance.
- **The Pain:** Running an agent 100 times in a GitHub Actions CI/CD pipeline takes 2 hours and costs $20 in API credits *per PR*.
- **The Solution:** SPRT is a statistical technique developed during WWII for quality control. It allows you to check the results *after every single test*. If the new prompt is overwhelmingly better (or horribly worse) after just 12 tests, SPRT mathematically guarantees that you can stop the evaluation immediately without sacrificing statistical confidence. This saves up to 80% on API costs and CI/CD compute time.

---

## 2. Exhaustive Use Cases & Feature Set

### 2.1 CI/CD Regression Testing (The Core Wedge)
*   **Workflow:** A developer writes a test suite using `pytest`. They decorate the test with `@sprt(alpha=0.05, beta=0.05, baseline_accuracy=0.8)`.
*   **Execution:** The test runner starts evaluating the LLM against the dataset. After every iteration, it calculates the log-likelihood ratio. If the LLM's accuracy drops so low that the SPRT threshold is crossed (e.g., on test 14), the entire test suite immediately halts, fails the CI pipeline, and prevents the bad prompt from being merged.

### 2.2 Self-Consistency Early Stopping (Compute Efficiency)
*   **Workflow:** Advanced LLM apps use "Self-Consistency" (asking the LLM to solve a math problem 10 times and taking the majority vote). 
*   **Execution:** `sprt-eval` can be used inside the application code. If the LLM outputs the same answer for the first 3 tries, SPRT can mathematically prove that the remaining 7 tries will not change the majority vote. It halts generation, saving massive token costs in production.

### 2.3 Multi-Agent Debate Governor
*   **Workflow:** When using multi-agent frameworks (like AutoGen), two agents debate an answer.
*   **Execution:** `sprt-eval` acts as a compute governor. It monitors the log-likelihood ratio of the agents converging. Once the debate crosses the statistical threshold of "useful convergence," it terminates the debate loop, preventing infinite API burns.

### 2.4 Agent Anomaly & Security Detection
*   **Workflow:** Monitoring a live agent's tool calls.
*   **Execution:** If the agent begins making repeated, anomalous tool calls (e.g., trying to read restricted files), SPRT can be used as a rolling statistical threshold to instantly kill the agent session before a security breach occurs.

---

## 3. Architecture & Technical Stack

Building `sprt-eval` requires clean API design and rigorous mathematical implementation.

### 3.1 The SPRT Mathematical Core (Python/Rust)
*   **Distributions Supported:** 
    *   **Bernoulli/Binomial:** For binary pass/fail evals (e.g., "Did the LLM output valid JSON?").
    *   **Categorical/Multinomial:** For multiple-choice evaluations or self-consistency clustering.
*   **Threshold Engine:** Calculates the upper bound $A$ and lower bound $B$ based on user-defined Type I ($\alpha$) and Type II ($\beta$) error tolerances.

### 3.2 The Test Runner Integrations
*   **Pytest Plugin:** `pytest-sprt` handles the orchestration, overriding standard test collection to run dynamically.
*   **LangChain / LlamaIndex Callbacks:** Native callback handlers that inject SPRT logic directly into standard LLM orchestration chains.

### 3.3 Telemetry & Logging
*   **Integrations:** Native export to LangSmith, Braintrust, and Langfuse to visually display *why* the evaluation stopped early (showing the log-likelihood random walk crossing the boundary).

---

## 4. Phased Implementation Strategy

### Phase 1: The Core Library (Weeks 1-2)
*   Implement the pure mathematical logic (`sprt.calculate_ll_ratio`, `sprt.check_boundaries`).
*   Publish the core `sprt-eval` library to PyPI.
*   **Validation:** Write exhaustive unit tests verifying the mathematical boundaries against known academic datasets to ensure zero statistical flaws.

### Phase 2: The Developer UX (Weeks 3-4)
*   Build the `pytest-sprt` plugin.
*   Create a beautiful CLI output that shows a dynamic progress bar and a visualization of the SPRT random walk hitting the threshold.

### Phase 3: The Orchestration Hooks (Weeks 5-6)
*   Build the LangChain and LlamaIndex callback handlers for production self-consistency stopping.

---

## 5. Final Notes & Edge Cases (The "Gotchas")

Before we consider this plan final, we must document three critical architectural risks:

1. **The "Cost vs. Variance" Trap:** If the LLM's performance is incredibly noisy (variance is high) and the new prompt is only marginally better than the old one, the SPRT random walk will bounce around the center and *never* hit the early-stopping boundaries. The library must include a `max_steps` failsafe so it doesn't run infinitely waiting for statistical significance that doesn't exist.
2. **Calibration Blindness:** SPRT assumes that the underlying evaluations (e.g., LLM-as-a-judge scoring) are perfectly calibrated. If the judge LLM has a systematic bias, SPRT will confidently and rapidly stop the test based on biased data. We must document this clearly to users.
3. **Item Selection Bias:** In standard evals, developers run test cases in a fixed order. If the first 10 test cases happen to be the easiest ones, SPRT might stop the test early with a "Pass", completely ignoring the hard test cases at the end of the dataset. `sprt-eval` MUST implement randomized shuffling of the dataset before execution to ensure I.I.D. (Independent and Identically Distributed) sampling.

---

> [!IMPORTANT]
> ## Open Questions for the User
> 1. **Language Choice:** Python is mandatory for the Pytest plugin, but should we write the core mathematical engine in **Rust** (and bind to Python via PyO3) to ensure it can be easily ported to TypeScript/Node.js later?
> 2. **Monetization Strategy:** The CLI and Pytest plugin must be free and open-source (MIT License) to get viral adoption. However, do you want to plan for a Cloud Dashboard (e.g., `sprt-cloud`) where teams can track their historical API cost savings over time?
> 3. **Initial Focus:** Should Phase 1 strictly target **CI/CD Regression Testing** (evaluating prompts during development), or should we pivot to prioritize **Self-Consistency Early Stopping** (saving costs in production)?
