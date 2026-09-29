"""Base reliability scorer evaluating multi-factor evidence signals for knowledge graph triples."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import networkx as nx

from src.graph.schema import Edge
from src.graph.manager import KnowledgeGraphManager


@dataclass
class ReliabilityBreakdown:
    """Detailed decomposition of static reliability signals for explainability."""
    edge_id: str
    provenance_score: float
    frequency_score: float
    topology_score: float
    semantic_score: float
    composite_reliability: float
    base_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "provenance_score": round(self.provenance_score, 4),
            "frequency_score": round(self.frequency_score, 4),
            "topology_score": round(self.topology_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "composite_reliability": round(self.composite_reliability, 4),
            "base_risk": round(self.base_risk, 4),
        }


class BaseReliabilityScorer:
    """Calculates intrinsic static edge reliability: Rel(e) in [0, 1] and BaseRisk(e) = 1 - Rel(e)."""

    def __init__(
        self,
        kg_manager: KnowledgeGraphManager,
        w_prov: float = 0.35,
        w_freq: float = 0.25,
        w_topo: float = 0.20,
        w_sem: float = 0.20,
        k_target_provenance: int = 3,
        max_expected_frequency: int = 10,
    ):
        self.kg = kg_manager
        # Normalize weights to sum to 1.0
        total_w = w_prov + w_freq + w_topo + w_sem
        self.w_prov = w_prov / total_w
        self.w_freq = w_freq / total_w
        self.w_topo = w_topo / total_w
        self.w_sem = w_sem / total_w
        self.k_target_prov = max(1, k_target_provenance)
        self.max_freq = max(1, max_expected_frequency)

    def score_provenance(self, edge: Edge) -> float:
        """Evaluates supporting evidence count and extractor confidence."""
        if not edge.provenance:
            return 0.10  # Baseline minimal score for ungrounded extracted triples
        count_factor = min(1.0, len(edge.provenance) / self.k_target_prov)
        avg_conf = sum(p.confidence for p in edge.provenance) / len(edge.provenance)
        return min(1.0, max(0.0, 0.5 * count_factor + 0.5 * avg_conf))

    def score_frequency(self, edge: Edge) -> float:
        """Evaluates log-scaled occurrence frequency."""
        freq = max(1, edge.frequency)
        score = math.log(1.0 + freq) / math.log(1.0 + self.max_freq)
        return min(1.0, max(0.0, score))

    def score_topology(self, edge: Edge) -> float:
        """Evaluates local structural graph connectivity (Jaccard neighborhood overlap)."""
        u, v = edge.source, edge.target
        if not (self.kg.graph.has_node(u) and self.kg.graph.has_node(v)):
            return 0.20

        # Successors/neighbors in undirected view to capture mutual association
        succ_u = set(self.kg.graph.successors(u)) | set(self.kg.graph.predecessors(u))
        succ_v = set(self.kg.graph.successors(v)) | set(self.kg.graph.predecessors(v))

        # Exclude endpoints themselves
        succ_u.discard(v)
        succ_v.discard(u)

        union = succ_u.union(succ_v)
        if not union:
            # If no other neighbors, return a neutral prior
            return 0.35

        overlap = succ_u.intersection(succ_v)
        jaccard = len(overlap) / len(union)
        # Scaled with base prior
        return min(1.0, max(0.0, 0.20 + 0.80 * jaccard))

    def score_semantics(self, edge: Edge) -> float:
        """Evaluates lexical/semantic alignment between relation name and provenance texts."""
        if not edge.provenance:
            return 0.30

        rel_tokens = set(edge.relation.lower().replace("_", " ").split())
        matched_scores = []
        for prov in edge.provenance:
            text_lower = prov.text_snippet.lower()
            matches = sum(1 for tok in rel_tokens if tok in text_lower)
            matched_scores.append(matches / max(1, len(rel_tokens)))

        avg_match = sum(matched_scores) / len(matched_scores)
        return min(1.0, max(0.20, avg_match))

    def compute_edge_reliability(self, edge: Edge) -> ReliabilityBreakdown:
        """Computes composite reliability and returns detailed signal breakdown."""
        s_prov = self.score_provenance(edge)
        s_freq = self.score_frequency(edge)
        s_topo = self.score_topology(edge)
        s_sem = self.score_semantics(edge)

        composite_rel = (
            self.w_prov * s_prov
            + self.w_freq * s_freq
            + self.w_topo * s_topo
            + self.w_sem * s_sem
        )
        composite_rel = max(0.01, min(0.99, composite_rel))
        base_risk = 1.0 - composite_rel

        # Update edge in place
        edge.update_reliability(composite_rel)

        return ReliabilityBreakdown(
            edge_id=edge.id,
            provenance_score=s_prov,
            frequency_score=s_freq,
            topology_score=s_topo,
            semantic_score=s_sem,
            composite_reliability=composite_rel,
            base_risk=base_risk,
        )

    def score_entire_graph(self) -> Dict[str, ReliabilityBreakdown]:
        """Iterates over all edges in the graph, updating their reliability attributes."""
        breakdowns: Dict[str, ReliabilityBreakdown] = {}
        for edge_id, edge in self.kg._edges.items():
            breakdown = self.compute_edge_reliability(edge)
            breakdowns[edge_id] = breakdown
        return breakdowns
