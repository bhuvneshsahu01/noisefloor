import logging
from typing import Dict, Any, Tuple
from noisefloor.judges.llm import LiveLLMJudge
from noisefloor.judges.conformal import ConformalJudge

logger = logging.getLogger("noisefloor.routing")

class ConformalRouter:
    """
    Conformal Model Router (CCPO / Cost Optimizer).
    Queries a cheap, low-latency LLM judge (e.g. Groq Llama 3 8B) first.
    If the uncertainty interval is ambiguous, escalates to a premium judge 
    (e.g. OpenRouter Gemma 9B).
    """
    def __init__(
        self, 
        cheap_judge: LiveLLMJudge, 
        premium_judge: LiveLLMJudge, 
        calibrator: ConformalJudge,
        cheap_cost: float = 0.0001, # Estimated API cost per query
        premium_cost: float = 0.005
    ):
        self.cheap_judge = cheap_judge
        self.premium_judge = premium_judge
        self.calibrator = calibrator
        self.cheap_cost = cheap_cost
        self.premium_cost = premium_cost

    def route_and_evaluate(self, question: str, response: str) -> Dict[str, Any]:
        """
        Evaluate response using the cheapest route. 
        Escalates only if conformal certainty boundaries are breached.
        """
        # 1. Query the cheap judge
        cheap_score = self.cheap_judge.evaluate_correctness(question, response)
        
        # 2. Check conformal calibration of the cheap judge
        calibration = self.calibrator.calibrate(cheap_score)
        
        if not calibration["is_ambiguous"]:
            # Safe to use cheap rating
            cost_saved = self.premium_cost - self.cheap_cost
            logger.info(f"Conformal Router: Approved cheap model rating. Cost saved: ${cost_saved:.4f}")
            return {
                "routed_to": "cheap",
                "score": cheap_score,
                "is_ambiguous": False,
                "cost_saved": cost_saved,
                "interval": calibration["interval"],
                "decision": "H1" if cheap_score >= 0.7 else "H0" # Simple threshold decision
            }
            
        # 3. Escalation required (Uncertainty is too high)
        logger.warning("Conformal Router: Cheap rating ambiguous. Escalating to premium judge...")
        premium_score = self.premium_judge.evaluate_correctness(question, response)
        
        # Calibrate premium judge score
        premium_calibration = self.calibrator.calibrate(premium_score)
        
        return {
            "routed_to": "premium",
            "score": premium_score,
            "is_ambiguous": premium_calibration["is_ambiguous"],
            "cost_saved": 0.0, # Escalated, no savings
            "interval": premium_calibration["interval"],
            "decision": "H1" if premium_score >= 0.7 else "H0"
        }
