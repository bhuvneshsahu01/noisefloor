import pytest
from risklayer.core.conformal import ConformalPredictor
from risklayer.integrations.agent_guard import RiskGuard
from risklayer.exceptions import ConformalRiskException

def test_recovery_successful_healing():
    """Verify that a fallback handler can modify arguments to safely bypass a risk block."""
    predictor = ConformalPredictor(alpha=0.10)
    predictor.calibrate([0.05, 0.1, 0.15]) # q_hat ~0.15
    
    # Score function: uncertainty equals transaction value / 1000
    def score_fn(amount):
        return amount / 1000.0
        
    # Fallback handler: reduces amount to 100 (which yields score = 0.10, safe!)
    def fallback_helper(amount):
        return (100,), {}
        
    guard = RiskGuard(predictor)
    
    @guard.guard(score_fn=score_fn, fallback_handler=fallback_helper)
    def transfer_money(amount):
        return f"Transferred ${amount}"
        
    # Standard run: amount = 500 (score = 0.50, exceeds 0.15).
    # It should trigger fallback, change amount to 100, re-evaluate (score = 0.10, safe), and run!
    result = transfer_money(500)
    assert result == "Transferred $100"

def test_recovery_unsuccessful_healing():
    """Verify that if healed arguments still breach risk, it throws ConformalRiskException."""
    predictor = ConformalPredictor(alpha=0.10)
    predictor.calibrate([0.05, 0.1, 0.15])
    
    def score_fn(amount):
        return amount / 1000.0
        
    # Fallback handler: reduces amount to 300 (which yields score = 0.30, still unsafe!)
    def fallback_helper(amount):
        return (300,), {}
        
    guard = RiskGuard(predictor)
    
    @guard.guard(score_fn=score_fn, fallback_handler=fallback_helper)
    def transfer_money(amount):
        return f"Transferred ${amount}"
        
    # Should throw ConformalRiskException because score = 0.30 is still > q_hat
    with pytest.raises(ConformalRiskException):
        transfer_money(500)
