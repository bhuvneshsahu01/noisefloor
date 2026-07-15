import contextvars
from typing import Dict, List

# Thread-safe context var to track active agent traces
active_trace_id = contextvars.ContextVar("active_trace_id", default=None)

class TrajectoryTracker:
    """
    Tracks sequential steps and cumulative non-conformity scores
    for multi-step agent trajectories.
    """
    def __init__(self):
        self._trajectories: Dict[str, List[float]] = {}

    def add_step(self, trace_id: str, score: float) -> int:
        """
        Adds a step score to the trajectory. 
        Returns the step index (1-indexed).
        """
        if trace_id not in self._trajectories:
            self._trajectories[trace_id] = []
        self._trajectories[trace_id].append(score)
        return len(self._trajectories[trace_id])

    def get_history(self, trace_id: str) -> List[float]:
        return self._trajectories.get(trace_id, [])

    def clear(self, trace_id: str) -> None:
        if trace_id in self._trajectories:
            del self._trajectories[trace_id]

# Global tracker instance
trajectory_tracker = TrajectoryTracker()
