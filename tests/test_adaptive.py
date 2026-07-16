import pytest
from risklayer.core.conformal import AdaptiveConformalPredictor

def test_adaptive_conformal_predictor_fifo_window():
    """Verify that the sliding window maintains its max size correctly."""
    predictor = AdaptiveConformalPredictor(alpha=0.10, max_window_size=5)
    
    initial_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    predictor.calibrate(initial_scores)
    
    # FIFO queue should truncate to size 5
    assert len(predictor.window) == 5
    assert predictor.window == [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Update calibration should slide the window
    predictor.update_calibration(new_score=0.8, was_correct=True)
    assert len(predictor.window) == 5
    assert predictor.window == [0.4, 0.5, 0.6, 0.7, 0.8]

def test_adaptive_conformal_predictor_adaptation():
    """Verify that target alpha is adjusted strictly based on error signals."""
    predictor = AdaptiveConformalPredictor(alpha=0.10, max_window_size=10, learning_rate=0.10)
    
    # Calibrate with baseline scores
    baseline = [0.1] * 10
    predictor.calibrate(baseline)
    
    original_alpha = predictor.alpha
    assert original_alpha == 0.10
    
    # 1. Simulate an INCORRECT action (should make bounds stricter -> decrease alpha)
    predictor.update_calibration(new_score=0.3, was_correct=False)
    # New alpha: 0.10 + 0.10 * (0.10 - 1.0) = 0.10 - 0.09 = 0.01
    assert predictor.alpha < original_alpha
    
    # 2. Simulate CORRECT actions (should relax bounds -> increase alpha back towards target)
    previous_alpha = predictor.alpha
    predictor.update_calibration(new_score=0.1, was_correct=True)
    # New alpha: previous + 0.10 * (0.10 - 0.0) -> increases
    assert predictor.alpha > previous_alpha
