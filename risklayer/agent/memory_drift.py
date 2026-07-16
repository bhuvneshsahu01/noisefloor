from typing import List
from risklayer.drift.ks_test import KSDriftDetector
import numpy as np

class MemoryDriftDetector:
    """
    Detects if an agent's working memory or context window has
    drifted into an unsafe or hijacked state compared to the initial
    baseline distribution of the system prompt and early interactions.
    """
    def __init__(self, p_value_threshold: float = 0.05):
        self.drift_detector = KSDriftDetector(p_value_threshold=p_value_threshold)
        self._baseline_embeddings = []

    def _mock_embed(self, text: str) -> float:
        """
        Mock embedding for demonstration purposes.
        In production, use OpenAI / sentence-transformers to get
        a semantic representation, then return the vector norm or 
        distance from the system prompt.
        """
        # A deterministic mock value based on length and ascii sum
        return float(sum(ord(c) for c in text) / (len(text) + 1))

    def fit_baseline(self, system_prompts: List[str]):
        """
        Establish the baseline memory distribution from the system prompts
        and expected user constraints.
        """
        if not system_prompts:
            return
            
        scores = [self._mock_embed(p) for p in system_prompts]
        self._baseline_embeddings = scores
        self.drift_detector.fit(scores)

    def check_memory_drift(self, current_memory_buffer: List[str]) -> dict:
        """
        Compare the current memory buffer against the baseline.
        If drift is detected, it could indicate a context hijacking or
        goal-drift (e.g. agent forgot the original instructions).
        """
        if not self._baseline_embeddings:
            return {"drift_detected": False, "status": "UNKNOWN_UNFIT"}
            
        scores = [self._mock_embed(m) for m in current_memory_buffer]
        return self.drift_detector.detect(scores)
