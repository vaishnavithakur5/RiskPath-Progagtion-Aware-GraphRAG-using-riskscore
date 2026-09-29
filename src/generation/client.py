"""Local LLM client interface supporting Ollama and deterministic offline mock execution."""

from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import requests


class BaseLLMClient(ABC):
    """Abstract base class for local language model generation."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        """Generates completion text given input prompt."""
        pass


class OllamaClient(BaseLLMClient):
    """Client for local Ollama HTTP API (e.g. Qwen2.5-1.5B or Llama-3.2-3B)."""

    def __init__(self, model_name: str = "qwen2.5:1.5b", api_base: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.endpoint = f"{self.api_base}/api/generate"

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            # If Ollama is not active, fallback or raise informative message
            raise RuntimeError(
                f"Ollama generation failed at {self.endpoint}: {e}. Ensure Ollama is running or switch to 'mock' backend."
            )


class MockLocalClient(BaseLLMClient):
    """Deterministic, CPU-friendly rule-based inference engine.

    Extracts multi-hop relational conclusions directly from the grounded facts in the prompt.
    Enables instant unit testing and offline benchmarking on laptops without GPU.
    """

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        # Parse facts section
        facts_match = re.search(r"Facts:\s*(.*?)\s*Question:", prompt, re.DOTALL)
        if not facts_match:
            return "Information not available"

        facts_text = facts_match.group(1).strip()
        if "No supporting graph facts" in facts_text or not facts_text:
            return "Information not available"

        lines = [line.strip() for line in facts_text.splitlines() if line.strip()]
        if not lines:
            return "Information not available"

        # Multi-hop chain resolution: follow transitions s -> r -> t
        # Lines format: "1. Source -> Relation -> Target"
        transitions: Dict[str, str] = {}
        first_source = None

        for line in lines:
            clean = re.sub(r"^\d+\.\s*", "", line)
            parts = [p.strip() for p in clean.split("->")]
            if len(parts) == 3:
                src, rel, tgt = parts
                transitions[src.lower()] = tgt
                if first_source is None:
                    first_source = src.lower()

        # Follow chain from first source to destination
        curr = first_source
        hops = 0
        while curr in transitions and hops < 5:
            nxt = transitions[curr]
            curr = nxt.lower()
            hops += 1

        if curr and hops > 0:
            # Return title-cased target entity
            return curr.title()

        # Fallback to last line target
        last_line = lines[-1]
        parts = [p.strip() for p in last_line.split("->")]
        if len(parts) >= 2:
            return parts[-1].title()

        return "Information not available"


def get_llm_client(backend: str = "mock", model_name: str = "qwen2.5:1.5b", api_base: str = "http://localhost:11434") -> BaseLLMClient:
    """Factory creating appropriate LLM client instance."""
    if backend.lower() == "ollama":
        return OllamaClient(model_name=model_name, api_base=api_base)
    elif backend.lower() == "mock":
        return MockLocalClient()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}. Supported: 'ollama', 'mock'.")
