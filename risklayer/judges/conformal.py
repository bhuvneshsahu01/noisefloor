"""
Conformal LLM Judge Calibrator.
"""
import numpy as np
from risklayer.conformal.weighted import WeightedConformalPredictor

class ConformalJudge:
    """
    Wraps an LLM judge to output calibrated certainty ratings using 
    Conformal Prediction. If a score is within the conformal uncertainty 
    boundary, it is labeled ambiguous and its SPRT weight is downgraded.
    Supports Weighted Conformal Prediction to adapt to covariate shift (distribution drift).
    """
    def __init__(self, calibration_scores: list[float], alpha: float = 0.10, calibration_features: np.ndarray = None):
        self.alpha = alpha
        self.non_conformity_scores = np.abs(np.array(calibration_scores) - 0.5) # Example heuristic
        self.calibration_features = calibration_features
        
        n = len(self.non_conformity_scores)
        if n == 0:
            self.q_hat = 0.0
        else:
            # Conformal quantile
            val = np.ceil((n + 1) * (1 - alpha)) / n
            val = min(1.0, max(0.0, val))
            self.q_hat = np.quantile(self.non_conformity_scores, val, method='higher')
            
        self.weighted_predictor = None
        if calibration_features is not None:
            self.weighted_predictor = WeightedConformalPredictor(calibration_features, self.non_conformity_scores)

    def fit_covariate_shift(self, production_features: np.ndarray):
        """Fit density ratio estimator on current production feature stream."""
        if self.weighted_predictor is not None:
            self.weighted_predictor.fit_covariate_shift(production_features)

    def calibrate(self, llm_score: float, prompt_feature: np.ndarray = None) -> dict:
        """
        Evaluate an LLM score and determine if it's statistically ambiguous.
        Adapts the quantile boundary dynamically if a prompt embedding is supplied
        and covariate shift calibration is active.
        """
        q_hat = self.q_hat
        
        # Use weighted boundary if prompt embedding is supplied
        if prompt_feature is not None and self.weighted_predictor is not None and self.weighted_predictor._is_trained:
            q_hat = self.weighted_predictor.get_weighted_quantile(prompt_feature, self.alpha)
            
        # Distance from certainty (0.0 or 1.0)
        uncertainty = 0.5 - abs(llm_score - 0.5)
        
        is_ambiguous = uncertainty > q_hat
        
        # Downgrade weight for SPRT if ambiguous
        weight = 0.5 if is_ambiguous else 1.0
        
        return {
            "score": llm_score,
            "is_ambiguous": is_ambiguous,
            "weight": weight,
            "q_hat_used": q_hat,
            "interval": [max(0.0, llm_score - q_hat), min(1.0, llm_score + q_hat)]
        }
