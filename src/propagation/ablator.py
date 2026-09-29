"""Counterfactual leave-one-out edge ablation measuring downstream generative answer sensitivity."""

from __future__ import annotations
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set

from src.graph.schema import Edge, ReasoningPath
from src.generation.client import BaseLLMClient
from src.generation.prompts import build_qa_prompt


@dataclass
class AblationRecord:
    """Records the outcome of ablating a specific edge from the reasoning path."""
    edge_id: str
    baseline_answer: str
    ablated_answer: str
    token_jaccard_distance: float
    semantic_distance: float
    causal_importance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "baseline_answer": self.baseline_answer,
            "ablated_answer": self.ablated_answer,
            "token_jaccard_distance": round(self.token_jaccard_distance, 4),
            "semantic_distance": round(self.semantic_distance, 4),
            "causal_importance": round(self.causal_importance, 4),
        }


class CausalAblator:
    """Estimates the causal answer-level influence of reasoning path edges via bounded ablation."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_ablation_edges: int = 3,
        weight_jaccard: float = 0.60,
        weight_semantic: float = 0.40,
    ):
        self.llm = llm_client
        self.max_edges = max_ablation_edges
        total = weight_jaccard + weight_semantic
        self.w_jaccard = weight_jaccard / total
        self.w_semantic = weight_semantic / total

    @staticmethod
    def _compute_token_jaccard_distance(text_a: str, text_b: str) -> float:
        """Computes 1.0 - Jaccard similarity over word tokens."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a and not tokens_b:
            return 0.0
        union = tokens_a.union(tokens_b)
        if not union:
            return 0.0
        intersection = tokens_a.intersection(tokens_b)
        similarity = len(intersection) / len(union)
        return max(0.0, min(1.0, 1.0 - similarity))

    @staticmethod
    def _compute_semantic_string_distance(text_a: str, text_b: str) -> float:
        """Computes sequence distance using Ratcliff-Obershelp similarity."""
        matcher = difflib.SequenceMatcher(None, text_a.lower(), text_b.lower())
        ratio = matcher.ratio()
        return max(0.0, min(1.0, 1.0 - ratio))

    def evaluate_path_causality(
        self, query: str, path: ReasoningPath
    ) -> Tuple[str, List[AblationRecord]]:
        """Performs bounded leave-one-out ablation on edges within path."""
        if not path.edges:
            return "Information not available", []

        # 1. Generate baseline answer with complete path
        prompt_0 = build_qa_prompt(query, path)
        y_0 = self.llm.generate(prompt_0)

        # 2. Select edges to ablate (prioritize edges with highest base risk or lowest confidence)
        edges_to_ablate = path.edges[: self.max_edges]
        ablation_records: List[AblationRecord] = []

        for edge in edges_to_ablate:
            # Construct ablated path (P \ {edge})
            ablated_edges = [e for e in path.edges if e.id != edge.id]
            ablated_path = ReasoningPath(edges=ablated_edges)

            # Generate counterfactual answer
            prompt_ablated = build_qa_prompt(query, ablated_path)
            y_ablated = self.llm.generate(prompt_ablated)

            # Measure displacement
            dist_jaccard = self._compute_token_jaccard_distance(y_0, y_ablated)
            dist_sem = self._compute_semantic_string_distance(y_0, y_ablated)
            importance = self.w_jaccard * dist_jaccard + self.w_semantic * dist_sem
            importance = max(0.0, min(1.0, importance))

            # Update edge causal metric
            edge.causal_importance = round(importance, 4)

            record = AblationRecord(
                edge_id=edge.id,
                baseline_answer=y_0,
                ablated_answer=y_ablated,
                token_jaccard_distance=dist_jaccard,
                semantic_distance=dist_sem,
                causal_importance=importance,
            )
            ablation_records.append(record)

        return y_0, ablation_records
