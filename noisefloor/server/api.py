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
