"""
Pydantic configuration models for risklayer.
"""
from pydantic import BaseModel, Field

class SPRTSettings(BaseModel):
    """Configuration for Wald's Sequential Probability Ratio Test."""
    h0_rate: float = Field(..., ge=0.0, le=1.0, description="Baseline expected accuracy (H0)")
    h1_rate: float = Field(..., ge=0.0, le=1.0, description="Target expected accuracy (H1)")
    alpha: float = Field(default=0.05, ge=0.0, le=1.0, description="Target Type I error rate")
    beta: float = Field(default=0.20, ge=0.0, le=1.0, description="Target Type II error rate")
    max_samples: int = Field(default=500, gt=0, description="Maximum number of samples before forced truncation")

class AIMDRateLimitSettings(BaseModel):
    """Configuration for the Additive-Increase/Multiplicative-Decrease concurrency rate limiter."""
    max_concurrency: int = Field(default=20, gt=0, description="Absolute maximum concurrency allowed")
    initial_concurrency: int = Field(default=1, gt=0, description="Starting concurrency")
    increase_factor: int = Field(default=1, gt=0, description="Number of workers to add on successful evaluation")
    decrease_factor: float = Field(default=0.5, gt=0.0, lt=1.0, description="Multiplier to apply to concurrency on rate limit")
    backoff_seconds: float = Field(default=2.0, gt=0.0, description="Base sleep time after hitting a rate limit")

class ConformalJudgeSettings(BaseModel):
    """Configuration for the Conformal LLM Judge Calibrator."""
    alpha: float = Field(default=0.10, ge=0.0, le=1.0, description="Target error rate for conformal prediction sets")
    ambiguity_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight multiplier for ambiguous evaluations in SPRT")
