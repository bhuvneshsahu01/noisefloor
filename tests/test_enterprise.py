import pytest
from sqlmodel import SQLModel, create_engine, Session, select
from fastapi.testclient import TestClient

from noisefloor.core.state import InMemoryStateStore, get_state_store
from noisefloor.core.signer import AuditSigner
from noisefloor.server.models import AgentTraceModel, VerdictModel
from noisefloor.server.api import app, get_session

# --- Unit Test 1: State Store Fallbacks ---
def test_state_store_fallback():
    # InMemory store should work out of the box
    store = InMemoryStateStore()
    store.set("test_key", "value")
    assert store.get("test_key") == "value"
    
    val = store.incr("metric:test")
    assert val == 1
    assert store.get("metric:test") == "1"
    
    # Factory check: if local Redis isn't running, it should fallback to InMemoryStateStore
    from unittest.mock import patch
    import redis
    
    with patch("redis.Redis.ping") as mock_ping:
        mock_ping.side_effect = redis.ConnectionError("Mocked connection error")
        dynamic_store = get_state_store(port=9999)
        assert isinstance(dynamic_store, InMemoryStateStore)

# --- Unit Test 2: Cryptographic Audit Signatures ---
def test_cryptographic_signer():
    priv, pub = AuditSigner.generate_keypair()
    signer = AuditSigner(priv)
    
    verdict_payload = {
        "test_name": "Backtest Alpha",
        "samples": 120,
        "log_lambda": 3.12,
        "decision": "H1"
    }
    
    signature = signer.sign_verdict(verdict_payload)
    assert len(signature) > 0
    
    # Verification with correct public key
    is_valid = signer.verify_verdict(verdict_payload, signature, pub)
    assert is_valid == True
    
    # Tampering test (simulate someone trying to alter the logs)
    tampered_payload = verdict_payload.copy()
    tampered_payload["decision"] = "H0" # Changed outcome
    
    is_tampered_valid = signer.verify_verdict(tampered_payload, signature, pub)
    assert is_tampered_valid == False
    
    # Invalid key test
    _, bad_pub = AuditSigner.generate_keypair()
    is_bad_key_valid = signer.verify_verdict(verdict_payload, signature, bad_pub)
    assert is_bad_key_valid == False

# --- Unit Test 3: DB Session and API Integration ---
@pytest.fixture(name="session")
def session_fixture():
    # Use in-memory SQLite for isolated test DB with StaticPool
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        "sqlite:///:memory:", 
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

def test_api_endpoints(session):
    client = TestClient(app)
    
    # Override FastAPI dependency to use isolated memory DB
    def get_session_override():
        return session
        
    app.dependency_overrides[get_session] = get_session_override
    
    # 1. Post Conformal Agent Trace
    trace_payload = {
        "step_id": "step-0",
        "action": "Execute SQL Query",
        "conformal_score": 0.12,
        "conformal_bound": 0.35,
        "status": "APPROVED",
        "trace_id": "otel-trace-123"
    }
    
    response = client.post("/api/v1/traces/agent", json=trace_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] is not None
    assert data["status"] == "APPROVED"
    
    # Verify in DB
    db_trace = session.exec(select(AgentTraceModel)).first()
    assert db_trace is not None
    assert db_trace.action == "Execute SQL Query"
    
    # 2. Post Verdict (Triggering Crypto Auto-Signature)
    verdict_payload = {
        "test_name": "SPRT Backtest",
        "samples": 45,
        "log_lambda": 3.01,
        "decision": "H1"
    }
    
    response = client.post("/api/v1/verdicts", json=verdict_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["cryptographic_signature"] is not None
    
    # Verify signature in DB
    db_verdict = session.exec(select(VerdictModel)).first()
    assert db_verdict is not None
    assert db_verdict.cryptographic_signature is not None
    
    # Clean override
    app.dependency_overrides.clear()
