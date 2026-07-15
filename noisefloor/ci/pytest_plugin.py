"""
Pytest Plugin for Noisefloor SPRT Early Stopping.
"""
import pytest
import asyncio
from typing import Any
from ..sprt import sprt_gate
from ..runner.async_queue import EvaluatorPool
from ..runner.rate_limiters import AIMDRateLimiter
from ..config import AIMDRateLimitSettings

def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("noisefloor")
    group.addoption("--sprt", action="store_true", help="Enable SPRT early stopping for evals")
    group.addoption("--sprt-h0", type=float, default=0.70, help="Baseline expected accuracy (H0)")
    group.addoption("--sprt-h1", type=float, default=0.80, help="Target expected accuracy (H1)")
    group.addoption("--sprt-alpha", type=float, default=0.05, help="Target Type I error rate")

@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session: Any) -> bool | None:
    if not session.config.getoption("--sprt"):
        return None  # Fall back to default pytest run loop

    loop = asyncio.get_event_loop()
    session.results = loop.run_until_complete(run_sprt_test_loop(session))
    return True  # Stop further execution of default pytest loop

class DummyEvaluator:
    async def evaluate(self, item: Any) -> float:
        return 0.8  # dummy score

async def run_sprt_test_loop(session: Any) -> str:
    h0 = session.config.getoption("--sprt-h0")
    h1 = session.config.getoption("--sprt-h1")
    alpha = session.config.getoption("--sprt-alpha")
    
    gate = sprt_gate(h0_rate=h0, h1_rate=h1, alpha=alpha, beta=0.20, max_samples=len(session.items))
    rate_limiter = AIMDRateLimiter(AIMDRateLimitSettings(max_concurrency=10))
    runner = EvaluatorPool(gate, rate_limiter, DummyEvaluator())
        
    verdict = await runner.run_stream(session.items)
    print(f"\n[noisefloor] SPRT Run Terminated Early. Verdict: {verdict}")
    return verdict
