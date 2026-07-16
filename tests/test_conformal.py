import pytest
from risklayer.core.conformal import ConformalPredictor
from risklayer.integrations.agent_guard import RiskGuard
from risklayer.exceptions import ConformalRiskException

def test_conformal_predictor():
    predictor = ConformalPredictor(alpha=0.1) # 90% confidence
    
    # Simulate historical agent uncertainty scores (lower is better)
    historical_scores = [0.1, 0.2, 0.15, 0.4, 0.9, 0.8, 0.5, 0.3, 0.25, 0.12]
    predictor.calibrate(historical_scores)
    
    # 90th percentile of 10 items should be around 0.8 / 0.9
    bound = predictor.get_bound()
    
    assert predictor.is_safe(0.5) == True
    assert predictor.is_safe(0.95) == False

def test_agent_guard_interceptor():
    predictor = ConformalPredictor(alpha=0.05)
    # Give it strict historical data so the bound is tight
    predictor.calibrate([0.1, 0.15, 0.12, 0.05, 0.2]) 
    
    guard = RiskGuard(predictor)
    
    # A fake agent function
    def calculate_uncertainty(prompt: str) -> float:
        if "DELETE" in prompt:
            return 0.9 # High uncertainty
        return 0.1
        
    @guard.guard(score_fn=calculate_uncertainty)
    def execute_sql(prompt: str):
        return f"Executed: {prompt}"
        
    # Safe action should pass
    result = execute_sql("SELECT * FROM users")
    assert result == "Executed: SELECT * FROM users"
    
    # Dangerous action should be blocked by the RiskLayer
    with pytest.raises(ConformalRiskException):
        execute_sql("DROP TABLE users; DELETE FROM logs;")
