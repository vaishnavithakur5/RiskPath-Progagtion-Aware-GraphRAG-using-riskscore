"""Propagation-Weighted Risk Score (PWRS) decision-theoretic scoring engine."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from src.graph.schema import Edge, ReasoningPath


@dataclass
class PathRiskSummary:
    """Comprehensive risk evaluation of a reasoning path."""
    path_linearized: str
    max_edge_pwrs: float
    cumulative_pwrs: float
    bottleneck_edge: Optional[Edge]
    composite_utility: float
    is_acceptable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_linearized": self.path_linearized,
            "max_edge_pwrs": round(self.max_edge_pwrs, 4),
            "cumulative_pwrs": round(self.cumulative_pwrs, 4),
            "bottleneck_edge_id": self.bottleneck_edge.id if self.bottleneck_edge else None,
            "composite_utility": round(self.composite_utility, 4),
            "is_acceptable": self.is_acceptable,
        }


class PWRSScorer:
    """Calculates decision-theoretic Propagation-Weighted Risk Score: PWRS(e) = BaseRisk(e) * [I_prop(e)]^gamma."""

    def __init__(
        self,
        gamma: float = 1.0,
        risk_threshold: float = 0.45,
        relevance_penalty_lambda: float = 0.70,
    ):
        self.gamma = max(0.1, float(gamma))
        self.risk_threshold = max(0.0, min(1.0, float(risk_threshold)))
        self.risk_lambda = max(0.0, float(relevance_penalty_lambda))

    def compute_edge_pwrs(self, edge: Edge) -> float:
        """Computes PWRS for a single edge: PWRS(e) = BaseRisk(e) * [I_prop(e)]^gamma."""
        base_risk = edge.base_risk
        causal_imp = edge.causal_importance

        # Decision-theoretic expected loss
        scaled_importance = math.pow(causal_imp, self.gamma)
        score = base_risk * scaled_importance
        score = max(0.0, min(1.0, score))

        edge.pwrs = round(score, 4)
        return edge.pwrs

    def evaluate_path(self, path: ReasoningPath) -> PathRiskSummary:
        """Evaluates bottleneck risk, cumulative risk, and utility for an entire reasoning path."""
        if not path.edges:
            return PathRiskSummary(
                path_linearized="Empty Path",
                max_edge_pwrs=0.0,
                cumulative_pwrs=0.0,
                bottleneck_edge=None,
                composite_utility=0.0,
                is_acceptable=False,
            )

        max_risk = 0.0
        cum_prob_safe = 1.0
        bottleneck_edge: Optional[Edge] = None

        for edge in path.edges:
            pwrs_val = self.compute_edge_pwrs(edge)
            if pwrs_val > max_risk or bottleneck_edge is None:
                max_risk = pwrs_val
                bottleneck_edge = edge
            cum_prob_safe *= (1.0 - pwrs_val)

        cum_risk = 1.0 - cum_prob_safe

        # Path utility balancing semantic relevance against bottleneck risk
        composite_utility = path.semantic_score - (self.risk_lambda * max_risk)

        # Update path fields
        path.max_risk = round(max_risk, 4)
        path.cum_risk = round(cum_risk, 4)
        path.pwrs_score = round(max_risk, 4)
        path.status = "SAFE" if max_risk <= self.risk_threshold else "FLAGGED"

        return PathRiskSummary(
            path_linearized=path.linearize(),
            max_edge_pwrs=path.max_risk,
            cumulative_pwrs=path.cum_risk,
            bottleneck_edge=bottleneck_edge,
            composite_utility=round(composite_utility, 4),
            is_acceptable=(max_risk <= self.risk_threshold),
        )

    def rank_candidate_paths(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """Ranks paths by composite utility descending."""
        for p in paths:
            self.evaluate_path(p)

        return sorted(
            paths,
            key=lambda p: (p.semantic_score - self.risk_lambda * p.max_risk),
            reverse=True,
        )
