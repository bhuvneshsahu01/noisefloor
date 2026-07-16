<div align="center">
  <img src="https://img.icons8.com/color/144/000000/shield.png" width="100"/>
  <h1>RiskLayer Enterprise</h1>
  <p><strong>Adaptive Conformal Risk Control for Autonomous AI Systems</strong></p>

  <p>
    <a href="https://github.com/your-org/risklayer/actions"><img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="Build Status"></a>
    <a href="https://pypi.org/project/risklayer/"><img src="https://img.shields.io/badge/PyPI-v1.0.0-blue.svg" alt="PyPI Version"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License"></a>
  </p>
</div>

---

## 🛑 The Market's Fatal Flaw
Current LLM evaluation relies on "LLMs-as-a-Judge," which are fundamentally broken. They are slow (adding 1-3 seconds of latency), wildly expensive, and suffer from undocumented hallucinations. No enterprise bank, healthcare provider, or insurance company will deploy autonomous agents if the only guarantee of safety is "GPT-4 said it was safe."

## 🛡️ Enter RiskLayer
RiskLayer replaces heuristic prompt-engineering with **mathematically guaranteed statistical bounds**. By utilizing **Conformal Prediction** and **Sequential Probability Ratio Tests (SPRT)**, RiskLayer guarantees that your agent's error rate will strictly adhere to a chosen tolerance ($\alpha$)—all in under 50ms.

### Key Features
- **Conformal Action Guard:** Intercepts out-of-distribution (OOD) agent tool executions before they happen.
- **Cryptographic Audit Trails:** Uses `Ed25519` to sign every single trace and verdict, guaranteeing SOC2 and HIPAA compliance.
- **Dynamic Model Routing:** Uses conformal bounds to intelligently route queries between cheap open-source models and premium frontier models, saving up to 80% on inference costs.
- **Real-time Dashboard:** A high-throughput React/Next.js dashboard to visualize the mathematical random walks and blocked tool calls.
- **Drop-in SDKs:** Native intercepts for LangChain and AutoGen.

## 🚀 Quick Start

### Installation
```bash
pip install risklayer
```

### 1. The Developer CLI
Run backtests, evaluate datasets, or spin up the dashboard instantly:
```bash
risklayer serve
risklayer eval path/to/dataset.json
risklayer verify <trace_id>
```

### 2. AutoGen Integration
Protect your agents from prompt injections and dangerous tool calls with 3 lines of code:
```python
from risklayer.agent import ActionGuard
from risklayer.integrations.autogen import inject_autogen_guard

action_guard = ActionGuard(safe_action_embeddings, target_alpha=0.05)
safe_agent = inject_autogen_guard(my_autogen_agent, action_guard)
```

## 📖 The Math (Whitepaper)
Curious about how we achieve sub-50ms deterministic bounds? Read our [WHITEPAPER.md](./WHITEPAPER.md).

---
*RiskLayer is open-source, built for the Enterprise.*
