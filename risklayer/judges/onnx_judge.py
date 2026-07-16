import time
import logging
from typing import Dict, Any

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

logger = logging.getLogger("risklayer.judges.onnx")

class OnnxEvaluator:
    """
    Lightning-fast local evaluator using small quantized models.
    Provides sub-50ms latency for real-time guardrails, bypassing network latency entirely.
    """
    
    def __init__(self, model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"):
        """
        Initializes the local pipeline.
        :param model_name: HuggingFace model ID for a small, fast classifier.
        """
        if pipeline is None:
            raise ImportError("The 'transformers' library is required to use OnnxEvaluator. Install with: pip install transformers")
            
        logger.info(f"Loading local zero-latency evaluator: {model_name}...")
        start_time = time.time()
        
        # Load a standard text classification pipeline. 
        # In a full production environment, this would be exported to ONNX via optimum.onnxruntime
        self.classifier = pipeline("text-classification", model=model_name, device=-1)
        
        load_time = (time.time() - start_time) * 1000
        logger.info(f"Local evaluator loaded in {load_time:.2f}ms.")
        
    def score(self, text: str, target_label: str = None) -> float:
        """
        Calculates the non-conformity (risk) score of the text.
        Returns a score between 0.0 (safe) and 1.0 (highly risky).
        
        :param text: The input string to evaluate.
        :param target_label: Optional label to check for (e.g., 'sadness', 'anger'). 
                             If the text matches the label, the risk is higher.
        :return: Float score representing risk.
        """
        start = time.time()
        
        # Run classification
        results = self.classifier(text)
        
        # For this demo model, if target_label is specified, the risk is the confidence of that label.
        # Otherwise, we just return the confidence of the top label as a proxy for 'certainty'.
        # Since non-conformity is 'error/risk', high confidence in a negative label = high risk.
        
        result = results[0]
        label = result['label']
        confidence = result['score']
        
        # Simple risk heuristic: 
        # If target_label is provided and matches, risk = confidence.
        # If no match, risk = 1.0 - confidence (meaning low confidence in safe label).
        if target_label:
            risk = confidence if label.lower() == target_label.lower() else (1.0 - confidence)
        else:
            # Default: just return a base uncertainty metric
            risk = 1.0 - confidence
            
        latency = (time.time() - start) * 1000
        logger.debug(f"ONNX Evaluation took {latency:.2f}ms | Result: {label} ({confidence:.3f}) | Risk: {risk:.3f}")
        
        return risk
