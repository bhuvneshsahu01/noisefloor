import pytest
import json
import os
from click.testing import CliRunner
from risklayer.core.autotune import AutoTuner
from risklayer.ci.cli import main

class MockLiveJudge:
    def evaluate_correctness(self, q, r):
        # Dummy evaluator
        if "correct" in r:
            return 0.95
        return 0.15

def test_autotune_calculations(tmp_path):
    """Verify that autotuning correctly calculates conformity scores against ground truth."""
    dataset_file = tmp_path / "benchmark.jsonl"
    
    # Create test benchmark
    rows = [
        {"question": "Q1", "response": "this is correct", "ground_truth": 1.0},
        {"question": "Q2", "response": "this is wrong", "ground_truth": 0.0}
    ]
    with open(dataset_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            
    judge = MockLiveJudge()
    tuner = AutoTuner(judge)
    
    scores = tuner.tune(str(dataset_file))
    
    assert len(scores) == 2
    # Score 1: abs(0.95 - 1.0) = 0.05
    assert pytest.approx(scores[0]) == 0.05
    # Score 2: abs(0.15 - 0.0) = 0.15
    assert pytest.approx(scores[1]) == 0.15

def test_autotune_cli_execution(tmp_path):
    """Verify Click CLI subcommand works and outputs JSON file."""
    dataset_file = tmp_path / "benchmark.jsonl"
    output_profile = tmp_path / "calibrated_profile.json"
    
    rows = [
        {"question": "Q1", "response": "correct answer", "ground_truth": 1.0}
    ]
    with open(dataset_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            
    runner = CliRunner()
    result = runner.invoke(main, [
        "autotune", 
        str(dataset_file), 
        "--output", str(output_profile),
        "--provider", "groq"
    ])
    
    assert result.exit_code == 0
    assert "Auto-tuning completed successfully!" in result.output
    
    # Verify the JSON structure was generated correctly
    assert os.path.exists(output_profile)
    with open(output_profile, "r") as f:
        profile = json.load(f)
        assert "calibration_scores" in profile
        assert len(profile["calibration_scores"]) == 1
