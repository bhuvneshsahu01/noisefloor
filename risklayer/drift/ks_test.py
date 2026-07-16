import numpy as np

class KSDriftDetector:
    """
    Kolmogorov-Smirnov (KS) test for detecting distribution shift
    between reference data (fit) and inference data (evaluate/monitor).
    """
    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold
        self.reference_scores = None

    def fit(self, reference_scores: list[float]):
        """Store the baseline distribution."""
        self.reference_scores = np.array(reference_scores)

    def detect(self, current_scores: list[float]) -> dict:
        """
        Compare current scores against the reference distribution.
        Returns a dictionary with drift status and metrics.
        """
        if self.reference_scores is None or len(self.reference_scores) == 0:
            return {"drift_detected": False, "p_value": 1.0, "status": "UNKNOWN"}
            
        if not current_scores:
            return {"drift_detected": False, "p_value": 1.0, "status": "UNKNOWN"}

        current_scores = np.array(current_scores)
        
        # Scipy KS 2-sample test
        from scipy.stats import ks_2samp
        statistic, p_value = ks_2samp(self.reference_scores, current_scores)
        
        drift_detected = p_value < self.p_value_threshold
        
        status = "HIGH" if drift_detected else ("MEDIUM" if p_value < (self.p_value_threshold * 5) else "LOW")
        
        return {
            "drift_detected": drift_detected,
            "p_value": p_value,
            "statistic": statistic,
            "status": status
        }
