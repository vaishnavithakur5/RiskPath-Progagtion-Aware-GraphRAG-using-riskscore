"""Evaluation metrics assessing factual answer quality, corrupted edge avoidance, and repair success."""

from __future__ import annotations
import re
import string
from dataclasses import dataclass
from typing import Dict, Any, List

from src.graph.schema import ReasoningPath


def normalize_answer(text: str) -> str:
    """Lowercases text, strips punctuation, articles, and extra whitespace for standardized scoring."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Normalize whitespace
    return " ".join(text.split())


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Returns 1.0 if normalized strings match, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Calculates token-level harmonic mean of precision and recall."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = set(pred_tokens).intersection(set(gold_tokens))
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return max(0.0, min(1.0, f1))


@dataclass
class QueryEvaluationResult:
    """Evaluation summary for a single query."""
    query_id: str
    method_name: str
    exact_match: float
    token_f1: float
    contains_corrupted_edge: bool
    avoided_corrupted_edge: bool
    repaired_successfully: bool
    latency_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "method_name": self.method_name,
            "exact_match": round(self.exact_match, 4),
            "token_f1": round(self.token_f1, 4),
            "contains_corrupted_edge": self.contains_corrupted_edge,
            "avoided_corrupted_edge": self.avoided_corrupted_edge,
            "repaired_successfully": self.repaired_successfully,
            "latency_seconds": round(self.latency_seconds, 4),
        }


def evaluate_prediction(
    query_id: str,
    method_name: str,
    prediction: str,
    ground_truth: str,
    selected_path: ReasoningPath,
    was_repaired: bool,
    latency: float,
) -> QueryEvaluationResult:
    """Evaluates answer correctness and graph path risk compliance."""
    em = compute_exact_match(prediction, ground_truth)
    f1 = compute_token_f1(prediction, ground_truth)

    has_corrupt = any(e.is_corrupted for e in selected_path.edges)
    avoided = not has_corrupt
    success_repair = was_repaired and (em > 0.5)

    return QueryEvaluationResult(
        query_id=query_id,
        method_name=method_name,
        exact_match=em,
        token_f1=f1,
        contains_corrupted_edge=has_corrupt,
        avoided_corrupted_edge=avoided,
        repaired_successfully=success_repair,
        latency_seconds=latency,
    )
