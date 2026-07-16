import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any

class WeightedConformalPredictor:
    """
    Weighted Conformal Prediction for Covariate Shift adaptation (Tibshirani et al., 2019).
    Adjusts conformal prediction bounds dynamically when the production prompt
    distribution drifts from the offline calibration set.
    """
    def __init__(self, calibration_features: np.ndarray, calibration_scores: np.ndarray):
        """
        calibration_features: (N, D) array of prompt vector embeddings.
        calibration_scores: (N,) array of heuristic non-conformity scores.
        """
        self.cal_X = np.asarray(calibration_features)
        self.cal_scores = np.asarray(calibration_scores)
        self.n_cal = len(calibration_scores)
        
        # Binary classifier to compute density ratio w(x) = P(prod)/P(calib)
        self.classifier = LogisticRegression(class_weight="balanced", max_iter=500)
        self._is_trained = False
        
    def fit_covariate_shift(self, production_features: np.ndarray):
        """
        Trains the density ratio estimator to distinguish between
        calibration features (class 0) and production features (class 1).
        """
        prod_X = np.asarray(production_features)
        n_prod = len(prod_X)
        
        # Combine datasets
        X = np.vstack([self.cal_X, prod_X])
        y = np.concatenate([np.zeros(self.n_cal), np.ones(n_prod)])
        
        # Fit logistic regression
        self.classifier.fit(X, y)
        self._is_trained = True
        
    def compute_density_ratio(self, x: np.ndarray) -> float:
        """
        Calculates the weight w(x) = p_prod(x) / p_cal(x).
        Using the relation w(x) = p(y=1|x)/p(y=0|x) * (N_cal / N_prod)
        """
        if not self._is_trained:
            return 1.0  # Default to unweighted if not trained
            
        x_reshaped = x.reshape(1, -1)
        probs = self.classifier.predict_proba(x_reshaped)[0]
        
        # Avoid division by zero
        p_cal = max(probs[0], 1e-8)
        p_prod = probs[1]
        
        # We assume equivalent weights class multiplier is handled by "balanced"
        return p_prod / p_cal

    def get_weighted_quantile(self, new_x: np.ndarray, alpha: float) -> float:
        """
        Calculates the dynamic weighted conformal prediction boundary for a new prompt new_x.
        """
        new_x = np.asarray(new_x)
        w_new = self.compute_density_ratio(new_x)
        
        # Compute weights for all calibration points
        w_cal = np.array([self.compute_density_ratio(x) for x in self.cal_X])
        
        # Conformal sum of weights
        total_weight = np.sum(w_cal) + w_new
        normalized_w_cal = w_cal / total_weight
        normalized_w_new = w_new / total_weight
        
        # Sort calibration scores
        sorted_indices = np.argsort(self.cal_scores)
        sorted_scores = self.cal_scores[sorted_indices]
        sorted_weights = normalized_w_cal[sorted_indices]
        
        # Find the weighted quantile
        cumulative_weight = 0.0
        target = 1.0 - alpha
        
        for score, weight in zip(sorted_scores, sorted_weights):
            cumulative_weight += weight
            if cumulative_weight >= target:
                return float(score)
                
        return float(sorted_scores[-1])
