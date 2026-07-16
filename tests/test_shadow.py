import pytest
from risklayer.core.conformal import ConformalPredictor
from risklayer.integrations.agent_guard import RiskGuard, active_shadow_mode
from risklayer.exceptions import ConformalRiskException

def test_shadow_mode_exception_bypass():
    """Verify that shadow mode intercepts blocks without raising exceptions."""
    predictor = ConformalPredictor(alpha=0.10)
    predictor.calibrate([0.05, 0.1, 0.15]) # q_hat ~0.15
    
    # 1. Initialize RiskGuard with shadow_mode = True
    guard_shadow = RiskGuard(predictor, shadow_mode=True)
    
    @guard_shadow.guard(score_fn=lambda x: x)
    def action_shadow(score):
        return "executed"
        
    # High score 0.50 (exceeds 0.15) should be allowed to run
    assert action_shadow(0.50) == "executed"

def test_shadow_mode_kwargs_override():
    """Verify shadow_mode can be overridden via function keyword arguments."""
    predictor = ConformalPredictor(alpha=0.10)
    predictor.calibrate([0.05, 0.1, 0.15])
    
    # Global default is active block (shadow_mode = False)
    guard_block = RiskGuard(predictor, shadow_mode=False)
    
    @guard_block.guard(score_fn=lambda x: x)
    def action_block(score, **kwargs):
        return "executed"
        
    # Standard run: should raise exception
    with pytest.raises(ConformalRiskException):
        action_block(0.50)
        
    # Keyword override: should run successfully
    assert action_block(0.50, shadow_mode=True) == "executed"

def test_shadow_mode_context_override():
    """Verify shadow_mode can be overridden using thread-safe contextvars."""
    predictor = ConformalPredictor(alpha=0.10)
    predictor.calibrate([0.05, 0.1, 0.15])
    
    guard_block = RiskGuard(predictor, shadow_mode=False)
    
    @guard_block.guard(score_fn=lambda x: x)
    def action_block(score):
        return "executed"
        
    # Contextvar override to True: should run successfully
    active_shadow_mode.set(True)
    assert action_block(0.50) == "executed"
    
    # Reset contextvar: should block again
    active_shadow_mode.set(False)
    with pytest.raises(ConformalRiskException):
        action_block(0.50)
        
    active_shadow_mode.set(None)
