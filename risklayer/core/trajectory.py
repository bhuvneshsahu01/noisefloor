import contextvars
import json
from typing import Dict, List, Any
from risklayer.core.state import StateStore, get_state_store

# Thread-safe context var to track active agent traces
active_trace_id = contextvars.ContextVar("active_trace_id", default=None)

class TrajectoryTracker:
    """
    Tracks sequential steps, cumulative non-conformity scores, and rich 
    event metadata (Event Sourcing) for multi-step agent trajectories.
    Utilizes a StateStore to track execution graphs across distributed serverless environments.
    """
    def __init__(self, state_store: StateStore = None):
        self.store = state_store or get_state_store()

    def _key(self, trace_id: str) -> str:
        return f"nf:trace:{trace_id}"

    def add_step(self, trace_id: str, score: float, metadata: Dict[str, Any] = None) -> int:
        """
        Adds a step score and metadata to the trajectory in the distributed store. 
        Returns the step index (1-indexed).
        """
        key = self._key(trace_id)
        payload = {
            "score": score,
            "metadata": metadata or {}
        }
        # rpush appends and returns the new length atomically
        step_idx = self.store.rpush(key, json.dumps(payload))
        return step_idx

    def log_state_transition(self, trace_id: str, state_type: str, state_snapshot: Any) -> int:
        """
        Logs a specific state transition (e.g., TOOL_CALL, MEMORY_UPDATE) without necessarily
        requiring a conformal score (defaults to 0.0). This supports tracking agent evolution.
        """
        metadata = {
            "transition_type": state_type,
            "state_snapshot": state_snapshot
        }
        return self.add_step(trace_id, score=0.0, metadata=metadata)

    def get_history(self, trace_id: str) -> List[float]:
        """Returns the history of non-conformity scores for a trace."""
        key = self._key(trace_id)
        raw_list = self.store.lrange(key, 0, -1)
        
        scores = []
        for item in raw_list:
            try:
                parsed = json.loads(item)
                scores.append(float(parsed.get("score", 0.0)))
            except json.JSONDecodeError:
                # Backwards compatibility if old keys just stored floats
                scores.append(float(item))
        return scores
        
    def get_full_trajectory(self, trace_id: str) -> List[Dict[str, Any]]:
        """Returns the complete event-sourced trajectory including all metadata."""
        key = self._key(trace_id)
        raw_list = self.store.lrange(key, 0, -1)
        
        events = []
        for i, item in enumerate(raw_list):
            try:
                parsed = json.loads(item)
                parsed["step"] = i + 1
                events.append(parsed)
            except json.JSONDecodeError:
                events.append({"step": i + 1, "score": float(item), "metadata": {}})
        return events

    def clear(self, trace_id: str) -> None:
        """Removes the trajectory state from the distributed store."""
        key = self._key(trace_id)
        self.store.delete(key)

# Global tracker instance
trajectory_tracker = TrajectoryTracker()

