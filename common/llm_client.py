import os
import json
import hashlib
from typing import Dict, Any, Optional
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
import logging

logger = logging.getLogger("llm_client")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".llm_cache")

def get_prompt_hash(model: str, system: str, user: str, kwargs: dict) -> str:
    """Generates a stable hash for a prompt and its parameters."""
    payload = {
        "model": model,
        "system": system,
        "user": user,
        "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"} 
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

class LLMClient:
    """
    OpenAI-compatible client wrapper for vLLM and API models, 
    with retries and a disk cache.
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: str = "ollama", default_model: str = "qwen2.5:7b-instruct-q4_K_M"):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = default_model
        os.makedirs(CACHE_DIR, exist_ok=True)
        
    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
        
    def _write_cache(self, cache_key: str, data: Dict[str, Any]):
        path = os.path.join(CACHE_DIR, f"{cache_key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def chat(self, user: str, system: Optional[str] = None, model: Optional[str] = None, logprobs: bool = False, temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Sends a chat request with caching and retries.
        """
        model = model or self.default_model
        system_msg = system or "You are a helpful assistant."
        
        cache_key = get_prompt_hash(model, system_msg, user, {"temperature": temperature, "logprobs": logprobs, **kwargs})
        
        cached = self._read_cache(cache_key)
        if cached:
            return cached
            
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user}
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            logprobs=logprobs,
            **kwargs
        )
        
        result = {
            "text": response.choices[0].message.content,
            "model": response.model,
        }
        
        logger.info(f"LLM Generation completed - Model: {model}, Prompt Hash: {cache_key}")
        
        if logprobs and response.choices[0].logprobs:
            # Depending on the backend (vLLM/Ollama), parse the logprobs array
            result["token_logprobs"] = [
                token.logprob for token in response.choices[0].logprobs.content
            ]
            
        self._write_cache(cache_key, result)
        return result
