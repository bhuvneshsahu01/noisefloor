from typing import List, Dict, Any, Callable
from risklayer.judges.conformal import ConformalJudge

class ConformalModelCascade:
    """
    Multi-Model Conformal Cascade Router.
    Sequentially queries cheaper models, checks if their conformal prediction set size
    is exactly 1 (meaning it's highly confident and safe).
    Cascades to the next premium model only if the current model's prediction is ambiguous.
    """
    def __init__(self, models: List[Dict[str, Any]], conformal_judges: List[ConformalJudge]):
        """
        models: List of dicts representing model clients and their costs, e.g.:
            [{"name": "llama3", "eval_fn": query_llama}, {"name": "gpt4", "eval_fn": query_gpt}]
        conformal_judges: List of ConformalJudges, mapping 1-to-1 with the models.
        """
        self.models = models
        self.judges = conformal_judges
        
    def execute_cascade(self, question: str, response: str) -> Dict[str, Any]:
        """
        Runs the model cascade sequentially.
        """
        for i, model in enumerate(self.models):
            judge = self.judges[i]
            
            # Query the model evaluator (e.g. LLM judge rating)
            # In a production router, this would call the LLM API
            score = model["eval_fn"](question, response)
            
            # Calibrate certainty boundary
            verdict = judge.calibrate(score)
            
            # If the model prediction is not ambiguous, we stop the cascade!
            if not verdict["is_ambiguous"]:
                return {
                    "routed_to": model["name"],
                    "cost": model.get("cost", 0.0),
                    "score": score,
                    "verdict": verdict,
                    "steps_executed": i + 1
                }
                
        # If all cheaper models were ambiguous, return the premium model (last in the cascade)
        premium_model = self.models[-1]
        premium_judge = self.judges[-1]
        score = premium_model["eval_fn"](question, response)
        verdict = premium_judge.calibrate(score)
        
        return {
            "routed_to": premium_model["name"],
            "cost": premium_model.get("cost", 0.0),
            "score": score,
            "verdict": verdict,
            "steps_executed": len(self.models)
        }
