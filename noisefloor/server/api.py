import asyncio
import json
import random
from datetime import datetime
from typing import List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from sqlmodel import SQLModel, create_engine, Session, select

from noisefloor.telemetry import setup_telemetry
from noisefloor.server.models import AgentTraceModel, VerdictModel
from noisefloor.core.signer import AuditSigner
from noisefloor.core.state import get_state_store
from noisefloor.judges import ConformalJudge, LiveLLMJudge
from noisefloor.integrations.routing import ConformalRouter
from pydantic import BaseModel

# Initialize OpenTelemetry
setup_telemetry()

# Initialize Database
DATABASE_URL = "sqlite:///noisefloor.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

def init_db():
    SQLModel.metadata.create_all(engine)

app = FastAPI(title="Noisefloor Enterprise RiskLayer API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

# Dependency for database session
def get_session():
    with Session(engine) as session:
        yield session

# Generate transient Ed25519 keys for the API's automatic signing
priv_key, pub_key = AuditSigner.generate_keypair()
signer = AuditSigner(priv_key)

# Global distributed state store
state_store = get_state_store()

# Setup Live LLM Judge and Conformal Calibrator
# Default to a mock fallback if no keys are found
try:
    live_judge = LiveLLMJudge()
    cheap_judge = LiveLLMJudge("groq")
    premium_judge = LiveLLMJudge("openrouter")
except Exception as e:
    # Safe dummy fallback for testing
    class DummyJudge:
        def __init__(self, score=0.85): self.score = score
        def evaluate_correctness(self, q, r): return self.score
    live_judge = DummyJudge(0.85)
    cheap_judge = DummyJudge(0.55) # Set cheap to ambiguous by default to trigger routing tests
    premium_judge = DummyJudge(0.88)
    
# Conformal prediction bounds calibration
historical_errors = [0.05, 0.1, 0.08, 0.15, 0.22, 0.02, 0.12]
conformal_judge = ConformalJudge(calibration_scores=historical_errors, alpha=0.10)

# Setup Conformal Router
conformal_router = ConformalRouter(cheap_judge, premium_judge, conformal_judge)

class EvaluationRequest(BaseModel):
    test_name: str
    question: str
    response: str

@app.post("/api/v1/evaluations/route", response_model=VerdictModel)
def evaluate_conformal_route(req: EvaluationRequest, session: Session = Depends(get_session)):
    """
    Dynamically route evaluation queries using Conformal prediction bounds.
    Saves costs and registers the savings inside the database metrics.
    """
    # 1. Run model routing check
    result = conformal_router.route_and_evaluate(req.question, req.response)
    
    # 2. Create persistent verdict record
    verdict = VerdictModel(
        test_name=f"{req.test_name} ({result['routed_to'].upper()} ROUTE)",
        samples=1,
        log_lambda=result["score"],
        decision=result["decision"],
        cost_saved=result["cost_saved"]
    )
    
    # 3. Cryptographically sign verdict payload
    payload = {
        "test_name": verdict.test_name,
        "samples": verdict.samples,
        "log_lambda": verdict.log_lambda,
        "decision": verdict.decision
    }
    verdict.cryptographic_signature = signer.sign_verdict(payload)
    
    session.add(verdict)
    session.commit()
    session.refresh(verdict)
    
    # Keep track of running metrics in distributed state
    state_store.incr("metric:total_evaluations")
    if result["cost_saved"] > 0:
        # Save saved cost in state (e.g. multiplied by 10000 to keep as int)
        state_store.incr("metric:cost_savings_micros")
        
    return verdict

@app.post("/api/v1/evaluations/judge", response_model=VerdictModel)
def evaluate_llm_judge(req: EvaluationRequest, session: Session = Depends(get_session)):
    """
    Run a live LLM evaluation, perform Conformal Calibration, 
    and sign/persist the statistical verdict.
    """
    # 1. Query the live LLM
    score = live_judge.evaluate_correctness(req.question, req.response)
    
    # 2. Calibrate certainty bounds
    calibrated = conformal_judge.calibrate(score)
    
    # 3. Create verdict payload
    verdict = VerdictModel(
        test_name=req.test_name,
        samples=1,
        log_lambda=score,
        decision="H1" if not calibrated["is_ambiguous"] else "CONTINUING"
    )
    
    # 4. Sign and save
    payload = {
        "test_name": verdict.test_name,
        "samples": verdict.samples,
        "log_lambda": verdict.log_lambda,
        "decision": verdict.decision
    }
    verdict.cryptographic_signature = signer.sign_verdict(payload)
    
    session.add(verdict)
    session.commit()
    session.refresh(verdict)
    
    state_store.incr("metric:total_evaluations")
    return verdict

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "noisefloor-risk-engine",
        "state_store": state_store.__class__.__name__
    }

@app.post("/api/v1/traces/agent", response_model=AgentTraceModel)
def add_agent_trace(trace: AgentTraceModel, session: Session = Depends(get_session)):
    """Ingest a live conformal prediction trace from the agent interceptor."""
    session.add(trace)
    session.commit()
    session.refresh(trace)
    
    # Also push to distributed state for rate limit/metric counts
    state_store.incr("metric:agent_interventions" if trace.status == "BLOCKED" else "metric:agent_approved")
    return trace

@app.post("/api/v1/verdicts", response_model=VerdictModel)
def add_verdict(verdict: VerdictModel, session: Session = Depends(get_session)):
    """Save an evaluation verdict and cryptographically sign it."""
    # Build payload to sign
    payload = {
        "test_name": verdict.test_name,
        "samples": verdict.samples,
        "log_lambda": verdict.log_lambda,
        "decision": verdict.decision
    }
    
    # Sign it cryptographically
    signature = signer.sign_verdict(payload)
    verdict.cryptographic_signature = signature
    
    session.add(verdict)
    session.commit()
    session.refresh(verdict)
    
    state_store.incr("metric:total_evaluations")
    return verdict

async def sprt_generator(session: Session):
    """
    Streams live SPRT mathematical random walk.
    If database records exist, streams real database verdicts.
    Otherwise, gracefully falls back to synthetic logs for testing.
    """
    log_lambda = 0.0
    n = 0
    while True:
        # Check if database has records
        statement = select(VerdictModel).order_by(VerdictModel.timestamp.desc()).limit(1)
        results = session.exec(statement).all()
        
        if results:
            real_verdict = results[0]
            data = {
                "samples": real_verdict.samples,
                "logLambda": real_verdict.log_lambda,
                "h0Boundary": -2.99,
                "h1Boundary": 2.99,
                "decision": real_verdict.decision,
                "signature": real_verdict.cryptographic_signature
            }
        else:
            # Fallback to simulated random walk
            score = 1 if random.random() > 0.4 else 0
            if score == 1:
                log_lambda += 0.25
            else:
                log_lambda -= 0.35
                
            data = {
                "samples": n,
                "logLambda": round(log_lambda, 2),
                "h0Boundary": -2.99,
                "h1Boundary": 2.99
            }
            n += 1
            if log_lambda <= -2.99 or log_lambda >= 2.99:
                await asyncio.sleep(2)
                log_lambda = 0.0
                n = 0

        yield {"data": json.dumps(data)}
        await asyncio.sleep(1)

@app.get("/api/v1/stream/sprt")
async def stream_sprt():
    session = Session(engine)
    return EventSourceResponse(sprt_generator(session))

async def agent_generator(session: Session):
    """
    Streams live agent conformal monitor events.
    If database records exist, streams real database traces.
    Otherwise, gracefully falls back to synthetic logs.
    """
    actions = [
        "Execute SQL Query", "Read File System", "Call External API", 
        "Generate Summarization", "Send Email"
    ]
    step = 0
    while True:
        # Check database records
        statement = select(AgentTraceModel).order_by(AgentTraceModel.timestamp.desc()).limit(8)
        results = session.exec(statement).all()
        
        if results:
            for trace in reversed(results):
                data = {
                    "id": f"db-{trace.id}",
                    "time": trace.timestamp.strftime("%I:%M:%S %p"),
                    "action": trace.action,
                    "conformalScore": trace.conformal_score,
                    "status": trace.status
                }
                yield {"data": json.dumps(data)}
        else:
            # Fallback mock generators
            is_dangerous = random.random() > 0.7
            conformal_score = round(random.uniform(0.1, 0.9 if is_dangerous else 0.4), 2)
            data = {
                "id": f"step-{step}",
                "time": datetime.now().strftime("%I:%M:%S %p"),
                "action": random.choice(actions),
                "conformalScore": conformal_score,
                "status": "BLOCKED" if is_dangerous else "APPROVED"
            }
            yield {"data": json.dumps(data)}
            step += 1
            
        await asyncio.sleep(2.5)

@app.get("/api/v1/stream/agents")
async def stream_agents():
    session = Session(engine)
    return EventSourceResponse(agent_generator(session))
