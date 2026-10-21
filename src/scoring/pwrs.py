"""Propagation-Weighted Risk Score (PWRS) decision-theoretic scoring and propagation engine.

Exact Mathematical Formulation (Thesis Slide 10):
  1. Own Risk:
     R_own(f) = 1.0 - Rel(f), where Rel(f) = w1*S_prov + w2*S_freq + w3*S_topo + w4*S_sem

  2. Inherited Risk:
     R_inh(v) = lambda * Agg({ R_tot(u) : u in parents(v) })
     Agg in {max, mean, probabilistic union}, default lambda = 0.60

  3. Total Risk:
     R_tot(v) = 1.0 - (1.0 - R_own(v)) * (1.0 - R_inh(v))

  4. Causal Importance:
     I_causal(f) = 1.0 - sim(Answer_full, Answer_without_f)  (via leave-one-out ablation)

  5. PWRS:
     PWRS(f) = R_tot(f) * [I_causal(f)]^gamma

  6. Path-Level Decision:
     PathRisk(P) = max_{f in P} PWRS(f)
     Trigger repair / reroute if PathRisk(P) > tau (default tau = 0.55)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from src.graph.schema import Edge, ReasoningPath


@dataclass
class PathRiskSummary:
    """Comprehensive risk evaluation of a reasoning path conforming to Slide 10."""
    path_linearized: str
    path_risk: float  # max_{f in P} PWRS(f)
    max_edge_pwrs: float  # alias for path_risk
    cumulative_pwrs: float
    bottleneck_edge: Optional[Edge]
    composite_utility: float
    is_acceptable: bool
    risk_breakdown: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_linearized": self.path_linearized,
            "path_risk": round(self.path_risk, 4),
            "max_edge_pwrs": round(self.max_edge_pwrs, 4),
            "cumulative_pwrs": round(self.cumulative_pwrs, 4),
            "bottleneck_edge_id": self.bottleneck_edge.id if self.bottleneck_edge else None,
            "composite_utility": round(self.composite_utility, 4),
            "is_acceptable": self.is_acceptable,
            "risk_breakdown": self.risk_breakdown,
        }


class PWRSScorer:
    """Calculates decision-theoretic PWRS fusing own risk, inherited risk, and causal importance."""

    def __init__(
        self,
        gamma: float = 1.0,
        risk_threshold: float = 0.55,
        lambda_propagation: float = 0.60,
        relevance_penalty_lambda: float = 0.70,
        agg_mode: str = "max",
    ):
        self.gamma = max(0.1, float(gamma))
        self.risk_threshold = max(0.0, min(1.0, float(risk_threshold)))
        self.lambda_propagation = max(0.0, min(1.0, float(lambda_propagation)))
        self.relevance_lambda = max(0.0, float(relevance_penalty_lambda))
        self.agg_mode = agg_mode.lower()

    def propagate_chain_risk(self, edges: List[Edge]) -> None:
        """Propagates inherited risk sequentially through a reasoning chain e_1 -> e_2 -> ... -> e_m.

        For e_1 (root hop):
          R_inh(e_1) = 0.0
          R_tot(e_1) = R_own(e_1)
        For e_i (downstream hop):
          R_inh(e_i) = lambda * R_tot(e_{i-1})
          R_tot(e_i) = 1.0 - (1.0 - R_own(e_i)) * (1.0 - R_inh(e_i))
        """
        if not edges:
            return

        # Upstream total risk accumulator
        prev_r_tot = 0.0

        for i, edge in enumerate(edges):
            # Ensure own risk is fresh: R_own = 1 - Rel
            edge.own_risk = max(0.0, min(1.0, 1.0 - edge.base_reliability))

            if i == 0:
                edge.inherited_risk = 0.0
            else:
                edge.inherited_risk = max(0.0, min(1.0, self.lambda_propagation * prev_r_tot))

            # Total risk: probabilistic union of own risk and inherited risk
            edge.total_risk = max(0.0, min(1.0, 1.0 - (1.0 - edge.own_risk) * (1.0 - edge.inherited_risk)))
            prev_r_tot = edge.total_risk

            # Recompute PWRS with fresh total risk
            self.compute_edge_pwrs(edge)

    def compute_edge_pwrs(self, edge: Edge) -> float:
        """Computes PWRS for a single edge: PWRS(e) = TotalRisk(e) * [I_causal(e)]^gamma."""
        tot_risk = edge.total_risk
        causal_imp = edge.causal_importance

        # Decision-theoretic expected loss
        scaled_importance = math.pow(causal_imp, self.gamma) if causal_imp > 0.0 else 0.0
        score = tot_risk * scaled_importance
        score = max(0.0, min(1.0, score))

        edge.pwrs = round(score, 4)
        return edge.pwrs

    def evaluate_path(self, path: ReasoningPath) -> PathRiskSummary:
        """Propagates risk along path, computes PWRS for all edges, and evaluates bottleneck risk."""
        if not path.edges:
            return PathRiskSummary(
                path_linearized="Empty Path",
                path_risk=0.0,
                max_edge_pwrs=0.0,
                cumulative_pwrs=0.0,
                bottleneck_edge=None,
                composite_utility=0.0,
                is_acceptable=False,
                risk_breakdown=[],
            )

        # 1. Propagate inherited risk along the sequence
        self.propagate_chain_risk(path.edges)

        max_risk = 0.0
        cum_prob_safe = 1.0
        bottleneck_edge: Optional[Edge] = None
        breakdown: List[Dict[str, Any]] = []

        for edge in path.edges:
            pwrs_val = edge.pwrs
            if pwrs_val > max_risk or bottleneck_edge is None:
                max_risk = pwrs_val
                bottleneck_edge = edge
            cum_prob_safe *= (1.0 - pwrs_val)
            breakdown.append(edge.to_risk_breakdown())

        cum_risk = 1.0 - cum_prob_safe

        # Composite utility balancing relevance against path bottleneck risk
        composite_utility = path.semantic_score - (self.relevance_lambda * max_risk)

        # Update path fields
        path.path_risk = round(max_risk, 4)
        path.max_risk = round(max_risk, 4)
        path.cum_risk = round(cum_risk, 4)
        path.pwrs_score = round(max_risk, 4)
        path.status = "SAFE" if max_risk <= self.risk_threshold else "FLAGGED"

        return PathRiskSummary(
            path_linearized=path.linearize(),
            path_risk=path.path_risk,
            max_edge_pwrs=path.path_risk,
            cumulative_pwrs=path.cum_risk,
            bottleneck_edge=bottleneck_edge,
            composite_utility=round(composite_utility, 4),
            is_acceptable=(max_risk <= self.risk_threshold),
            risk_breakdown=breakdown,
        )

    def rank_candidate_paths(self, paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """Ranks paths by composite utility descending (maximizing relevance, penalizing risk)."""
        for p in paths:
            self.evaluate_path(p)

        return sorted(
            paths,
            key=lambda p: (p.semantic_score - self.relevance_lambda * p.path_risk),
            reverse=True,
        )
