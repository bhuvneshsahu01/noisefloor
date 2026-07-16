import numpy as np
import pandas as pd
from typing import Dict, Any, List

class ConformalBacktester:
    """
    Vectorized quantitative backtesting engine for Conformal Prediction.
    Evaluates how different target alphas (risk tolerance) would have
    performed over thousands of historical LLM predictions.
    """
    def __init__(self, reference_scores: np.ndarray):
        """
        Initializes the backtester with a calibration (reference) dataset.
        """
        self.reference_scores = np.sort(reference_scores)
        self.n = len(self.reference_scores)

    def _get_q_hat(self, alpha: float) -> float:
        """Calculates the empirical quantile for a given alpha."""
        q_level = np.ceil((self.n + 1) * (1 - alpha)) / self.n
        q_level = min(q_level, 1.0)
        return float(np.quantile(self.reference_scores, q_level))

    def evaluate_alpha(self, test_scores: np.ndarray, ground_truth_labels: np.ndarray, alpha: float) -> Dict[str, float]:
        """
        Evaluates a specific alpha against a test dataset with ground truth.
        ground_truth_labels: 1 (safe/correct), 0 (unsafe/incorrect).
        """
        q_hat = self._get_q_hat(alpha)
        
        # Vectorized prediction
        # Score <= q_hat means model is confident/safe
        predictions = (test_scores <= q_hat).astype(int)
        
        # Calculate metrics
        true_positives = np.sum((predictions == 1) & (ground_truth_labels == 1))
        false_positives = np.sum((predictions == 1) & (ground_truth_labels == 0))
        false_negatives = np.sum((predictions == 0) & (ground_truth_labels == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
        
        rejection_rate = 1.0 - np.mean(predictions)
        
        return {
            "alpha": alpha,
            "q_hat": q_hat,
            "precision": precision,
            "recall": recall,
            "rejection_rate": rejection_rate
        }

    def generate_curve(self, test_scores: np.ndarray, ground_truth_labels: np.ndarray, alphas: List[float] = None) -> pd.DataFrame:
        """
        Generates a DataFrame mapping different alpha values to Precision, Recall, and Rejection Rates.
        Used to plot the tradeoff curve and find the optimal operating point.
        """
        if alphas is None:
            alphas = list(np.linspace(0.01, 0.5, 50))
            
        results = []
        for alpha in alphas:
            res = self.evaluate_alpha(test_scores, ground_truth_labels, alpha)
            results.append(res)
            
        return pd.DataFrame(results)
