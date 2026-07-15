import json
import os
from typing import List
from noisefloor.judges.llm import LiveLLMJudge

class AutoTuner:
    """
    Benchmarks LLM judges and auto-tunes conformal prediction boundaries.
    """
    def __init__(self, judge: LiveLLMJudge):
        self.judge = judge
        
    def tune(self, dataset_path: str) -> List[float]:
        """
        Runs the judge over the benchmark dataset and computes non-conformity scores.
        """
        non_conformity_scores = []
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Benchmark dataset not found at: {dataset_path}")
            
        with open(dataset_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    q = data.get("question", "")
                    r = data.get("response", "")
                    # Default ground truth to 1.0 (completely correct) if not specified
                    gt = float(data.get("ground_truth", 1.0))
                    
                    score = self.judge.evaluate_correctness(q, r)
                    # Non-conformity score: distance between prediction and ground truth
                    non_conformity = abs(score - gt)
                    non_conformity_scores.append(non_conformity)
                except Exception:
                    # Skip malformed lines or transient evaluation failures
                    continue
                    
        return non_conformity_scores
