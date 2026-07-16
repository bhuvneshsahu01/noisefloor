from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime

class AgentTraceModel(SQLModel, table=True):
    """
    Persistent table for Autonomous Agent conformal risk checks.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    step_id: str = Field(index=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str
    conformal_score: float
    conformal_bound: float
    status: str # APPROVED, BLOCKED
    trace_id: Optional[str] = None # OpenTelemetry trace context

class VerdictModel(SQLModel, table=True):
    """
    Persistent table for SPRT evaluation verdicts.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    test_name: str
    samples: int
    log_lambda: float
    decision: str # H0, H1, CONTINUING
    cost_saved: float = Field(default=0.0)
    cryptographic_signature: Optional[str] = None
