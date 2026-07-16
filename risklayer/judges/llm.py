import os
import requests
import json
import logging
from typing import Optional

logger = logging.getLogger("risklayer.judges")

def load_env(path: str = ".env") -> None:
    """Lightweight helper to load environment variables from a .env file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                if "=" in stripped and not stripped.startswith("#"):
                    k, v = stripped.split("=", 1)
                    # Strip spaces and quotes
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

# Load environment variables on import
load_env()

class LiveLLMJudge:
    """
    Queries actual LLMs (via Groq or OpenRouter) to evaluate text.
    Provides real correctness/helpfulness ratings.
    """
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM Judge. Defaults to Groq, falling back to OpenRouter.
        """
        self.provider = provider or ("groq" if os.getenv("GROQ_API_KEY") else "openrouter")
        self.api_key = os.getenv("GROQ_API_KEY") if self.provider == "groq" else os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key missing for provider: {self.provider}. Please verify your .env file.")
            
        if self.provider == "groq":
            self.url = "https://api.groq.com/openai/v1/chat/completions"
            self.model = "llama-3.1-8b-instant"
        else:
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = "meta-llama/llama-3.1-8b-instruct:free"

    def evaluate_correctness(self, question: str, response: str) -> float:
        """
        Queries the LLM to score the response correctness from 0.0 (completely wrong)
        to 1.0 (completely correct).
        """
        prompt = (
            f"You are an expert AI evaluator.\n"
            f"Evaluate the correctness and quality of this response to the question.\n\n"
            f"Question: {question}\n"
            f"Response: {response}\n\n"
            f"Output a single numerical score strictly between 0.0 (completely wrong) and 1.0 (perfectly correct).\n"
            f"Do not write any markdown, introduction, or explanation. Only output a float (e.g. 0.85)."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # OpenRouter requires HTTP-Referer or App-Title headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://risklayer.ai"
            headers["X-Title"] = "risklayer RiskLayer"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10
        }

        try:
            res = requests.post(self.url, headers=headers, json=payload, timeout=8.0)
            if not res.ok:
                logger.error(f"Live LLM Evaluation failed: {res.status_code} {res.reason} - {res.text}")
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"].strip()
            
            # Parse the float score
            score = float(content)
            # Ensure boundaries are respected
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"Live LLM Exception: {str(e)}")
            # Default to a safe median uncertainty score if the API call fails
            return 0.5
