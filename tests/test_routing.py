import pytest
from sqlmodel import SQLModel, create_engine, Session
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from noisefloor.judges.conformal import ConformalJudge
from noisefloor.integrations.routing import ConformalRouter
from noisefloor.server.api import app, get_session
from noisefloor.server.models import VerdictModel

class MockJudge:
    def __init__(self, score: float):
        self.score = score
        
    def evaluate_correctness(self, q: str, r: str) -> float:
        return self.score

def test_conformal_routing_no_escalation():
    """Verify that if the cheap judge is confident (not ambiguous), we route to cheap."""
    cheap_judge = MockJudge(0.95) # Highly confident correct answer (score = 0.95)
    premium_judge = MockJudge(0.98)
    
    # Normal calibration errors: range [0.45, 0.55] is ambiguous
    historical_errors = [0.05, 0.1, 0.08, 0.15]
    calibrator = ConformalJudge(calibration_scores=historical_errors, alpha=0.10)
    
    router = ConformalRouter(cheap_judge, premium_judge, calibrator)
    result = router.route_and_evaluate("What is 1+1?", "1+1 is 2.")
    
    assert result["routed_to"] == "cheap"
    assert result["score"] == 0.95
    assert result["is_ambiguous"] == False
    assert result["cost_saved"] > 0.0
    assert result["decision"] == "H1"

def test_conformal_routing_escalation():
    """Verify that if the cheap judge is ambiguous, we escalate to premium."""
    cheap_judge = MockJudge(0.51) # Ambiguous score (0.51 is close to 0.5)
    premium_judge = MockJudge(0.92) # Premium judge resolves it to 0.92
    
    historical_errors = [0.05, 0.1, 0.08, 0.15]
    calibrator = ConformalJudge(calibration_scores=historical_errors, alpha=0.10)
    
    router = ConformalRouter(cheap_judge, premium_judge, calibrator)
    result = router.route_and_evaluate("What is 1+1?", "1+1 is 2.")
    
    assert result["routed_to"] == "premium"
    assert result["score"] == 0.92
    assert result["cost_saved"] == 0.0 # No savings on escalation
    assert result["decision"] == "H1"

def test_routing_api_endpoint():
    """Verify FastAPI routing endpoint persists database schemas correctly."""
    from sqlalchemy.pool import StaticPool
    from sqlmodel import select
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    
    with Session(engine) as session:
        client = TestClient(app)
        
        def get_session_override():
            return session
            
        app.dependency_overrides[get_session] = get_session_override
        
        payload = {
            "test_name": "Cost-Optimization-Test",
            "question": "What is 3 * 3?",
            "response": "3 * 3 is 9."
        }
        
        response = client.post("/api/v1/evaluations/route", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "Cost-Optimization-Test" in data["test_name"]
        assert data["samples"] == 1
        assert 0.0 <= data["log_lambda"] <= 1.0
        assert data["decision"] in ["H1", "H0"]
        assert data["cryptographic_signature"] is not None
        
        # Verify the record in DB has cost_saved populated
        db_verdict = session.exec(select(VerdictModel)).first()
        assert db_verdict is not None
        assert db_verdict.cost_saved >= 0.0
        
        app.dependency_overrides.clear()
