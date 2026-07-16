from risklayer.core.conformal import ConformalPredictor
from risklayer.drift.ks_test import KSDriftDetector
from risklayer.core.reports import EvaluationReport
from risklayer.integrations.tiered_guard import CascadingGuard
import numpy as np

class RiskGuard:
    """
    The top-level scikit-learn style API for RiskLayer.
    Wraps Conformal Prediction, Sequential Testing, and Drift Detection
    into a simple DX for AI engineers.
    """
    def __init__(self, target_alpha: float = 0.1):
        self.target_alpha = target_alpha
        self.predictor = ConformalPredictor(alpha=target_alpha)
        self.drift_detector = KSDriftDetector()
        
        self.reference_scores = None

    def fit(self, reference_data: list[float], labels=None):
        """
        Calibrate the conformal bounds and baseline distribution using 
        a reference set of nonconformity scores (e.g. LLM judge scores).
        """
        self.reference_scores = reference_data
        
        # Fit conformal bounds
        self.predictor.calibrate(reference_data)
        
        # Fit drift baseline
        self.drift_detector.fit(reference_data)
        
        return self

    def evaluate(self, predictions: list[float], labels=None) -> EvaluationReport:
        """
        Evaluate a new batch of predictions.
        Checks coverage, calculates conformal radius, and runs drift detection.
        Returns a rich EvaluationReport.
        """
        if not self.reference_scores:
            raise ValueError("Must call fit() before evaluate()")
            
        # 1. Drift Detection
        drift_results = self.drift_detector.detect(predictions)
        
        # 2. Conformal Math
        q_hat = self.predictor._q_hat
        
        # Empirical coverage (how many predictions are below q_hat)
        covered = sum(1 for p in predictions if p <= q_hat)
        empirical_coverage = covered / len(predictions) if len(predictions) > 0 else 0.0
        
        target_coverage = 1.0 - self.target_alpha
        calibration_error = abs(empirical_coverage - target_coverage)
        
        # 3. Assemble Metrics
        metrics = {
            "calibration_error": calibration_error,
            "coverage": empirical_coverage,
            "guaranteed_coverage": empirical_coverage >= target_coverage,
            "distribution_shift": drift_results["status"],
            "conformal_radius": q_hat,
            "expected_risk": np.mean(predictions)
        }
        
        return EvaluationReport(metrics)

    def monitor(self, stream):
        """
        Placeholder for online monitoring (CUSUM, Page-Hinkley).
        """
        pass

    def wrap(self, agent_or_llm):
        """
        Syntactic sugar to return a configured CascadingGuard for this LLM/Agent.
        """
        return CascadingGuard(predictor=self.predictor)
