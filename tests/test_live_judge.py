import pytest
from sqlmodel import SQLModel, create_engine, Session
from fastapi.testclient import TestClient
from noisefloor.judges.llm import LiveLLMJudge
from noisefloor.server.api import app, get_session
from noisefloor.server.models import VerdictModel

def test_live_llm_judge_initialization():
    """Verify loading from .env occurs and provider is selected."""
    try:
        judge = LiveLLMJudge()
        assert judge.provider in ["groq", "openrouter"]
        assert judge.api_key is not None
    except ValueError as e:
        # Skip if no keys are found in environment (e.g. running in keyless CI)
        pytest.skip(f"Live API keys missing: {str(e)}")

def test_live_llm_evaluation():
    """Trigger a live evaluation check against Groq/OpenRouter."""
    try:
        judge = LiveLLMJudge()
        # Evaluate a obviously correct response
        score_good = judge.evaluate_correctness(
            question="What is the capital of France?",
            response="The capital of France is Paris."
        )
        assert 0.0 <= score_good <= 1.0
        
        # Evaluate a completely wrong response
        score_bad = judge.evaluate_correctness(
            question="What is the capital of France?",
            response="The capital of France is Tokyo."
        )
        assert 0.0 <= score_bad <= 1.0
        # A correct answer should score higher than a wrong one
        assert score_good >= score_bad
        
    except ValueError:
        pytest.skip("Skipping live LLM connection test (keys not found).")

def test_api_endpoint_evaluation():
    """Verify FastAPI endpoint performs live evaluation and DB commits."""
    # Use isolated SQLModel DB in StaticPool
    from sqlalchemy.pool import StaticPool
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    
    with Session(engine) as session:
        client = TestClient(app)
        
        def get_session_override():
            return session
            
        app.dependency_overrides[get_session] = get_session_override
        
        payload = {
            "test_name": "Groq-Correctness-Check",
            "question": "What is 2 + 2?",
            "response": "2 + 2 is equal to 4."
        }
        
        response = client.post("/api/v1/evaluations/judge", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["test_name"] == "Groq-Correctness-Check"
        assert data["samples"] == 1
        assert 0.0 <= data["log_lambda"] <= 1.0
        assert data["decision"] in ["H1", "CONTINUING"]
        assert data["cryptographic_signature"] is not None
        
        app.dependency_overrides.clear()
