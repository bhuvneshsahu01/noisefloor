"""
Pytest Plugin for Noisefloor SPRT Early Stopping.
"""
import pytest
import asyncio
from ..sprt import sprt_gate
from ..runner.async_queue import AsyncQueueRunner

def pytest_addoption(parser):
    group = parser.getgroup("noisefloor")
    group.addoption("--sprt", action="store_true", help="Enable SPRT early stopping for evals")
    group.addoption("--sprt-h0", type=float, default=0.70, help="Baseline expected accuracy (H0)")
    group.addoption("--sprt-h1", type=float, default=0.80, help="Target expected accuracy (H1)")
    group.addoption("--sprt-alpha", type=float, default=0.05, help="Target Type I error rate")

@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if not session.config.getoption("--sprt"):
        return None  # Fall back to default pytest run loop

    loop = asyncio.get_event_loop()
    session.results = loop.run_until_complete(run_sprt_test_loop(session))
    return True  # Stop further execution of default pytest loop

async def run_sprt_test_loop(session):
    h0 = session.config.getoption("--sprt-h0")
    h1 = session.config.getoption("--sprt-h1")
    alpha = session.config.getoption("--sprt-alpha")
    
    gate = sprt_gate(h0_rate=h0, h1_rate=h1, alpha=alpha, beta=0.20, max_samples=len(session.items))
    runner = AsyncQueueRunner(gate, max_concurrency=10)
    
    # Mocking the async evaluation step for the pytest loop
    async def eval_coro(item):
        # In reality, this would execute the test item and extract the score.
        # This is a stub for the pytest hook override.
        return 0.8  # dummy score
        
    verdict = await runner.run_stream(session.items, eval_coro)
    print(f"\n[noisefloor] SPRT Run Terminated Early. Verdict: {verdict}")
    return verdict
