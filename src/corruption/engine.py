"""Interactive Adversarial Robustness Protocol (IARP) execution engine."""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

from src.graph.schema import Edge
from src.graph.manager import KnowledgeGraphManager
from .operators import CorruptionOperator, get_operator


@dataclass
class CorruptionAuditRecord:
    """Audit log of an applied corruption event."""
    original_edge_id: str
    corrupted_edge_id: str
    operator_name: str
    source: str
    original_relation: str
    corrupted_relation: str
    original_target: str
    corrupted_target: str
    original_reliability: float
    corrupted_reliability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_edge_id": self.original_edge_id,
            "corrupted_edge_id": self.corrupted_edge_id,
            "operator_name": self.operator_name,
            "source": self.source,
            "original_relation": self.original_relation,
            "corrupted_relation": self.corrupted_relation,
            "original_target": self.original_target,
            "corrupted_target": self.corrupted_target,
            "original_reliability": round(self.original_reliability, 4),
            "corrupted_reliability": round(self.corrupted_reliability, 4),
        }


class CorruptionEngine:
    """Orchestrates deterministic corruption generation over a knowledge graph."""

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed

    def inject_noise(
        self,
        kg: KnowledgeGraphManager,
        operator_name: str = "entity_swap",
        noise_rate: float = 0.10,
        replace_original: bool = True,
    ) -> Tuple[KnowledgeGraphManager, List[CorruptionAuditRecord]]:
        """Applies corruption operator to a specified fraction of graph edges deterministically."""
        rng = random.Random(self.seed)
        operator = get_operator(operator_name)
        corrupted_kg = kg.copy()

        all_edges = list(corrupted_kg._edges.values())
        if not all_edges or noise_rate <= 0.0:
            return corrupted_kg, []

        num_to_corrupt = max(1, int(len(all_edges) * noise_rate))
        sampled_edges = rng.sample(all_edges, min(num_to_corrupt, len(all_edges)))

        audit_records: List[CorruptionAuditRecord] = []

        for edge in sampled_edges:
            corrupted_edge = operator.apply(corrupted_kg, edge, rng)
            if not corrupted_edge:
                continue

            if replace_original:
                corrupted_kg.remove_edge_by_id(edge.id)

            corrupted_kg.add_edge(corrupted_edge)

            audit_records.append(
                CorruptionAuditRecord(
                    original_edge_id=edge.id,
                    corrupted_edge_id=corrupted_edge.id,
                    operator_name=operator_name,
                    source=edge.source,
                    original_relation=edge.relation,
                    corrupted_relation=corrupted_edge.relation,
                    original_target=edge.target,
                    corrupted_target=corrupted_edge.target,
                    original_reliability=edge.base_reliability,
                    corrupted_reliability=corrupted_edge.base_reliability,
                )
            )

        return corrupted_kg, audit_records

    def inject_single_targeted_corruption(
        self,
        kg: KnowledgeGraphManager,
        target_edge_id: str,
        operator_name: str = "entity_swap",
    ) -> Tuple[KnowledgeGraphManager, Optional[CorruptionAuditRecord]]:
        """Corrupts a specific designated edge for interactive testing and live viva demo."""
        edge = kg.get_edge_by_id(target_edge_id)
        if not edge:
            return kg, None

        rng = random.Random(self.seed)
        operator = get_operator(operator_name)
        corrupted_kg = kg.copy()

        corrupted_edge = operator.apply(corrupted_kg, edge, rng)
        if not corrupted_edge:
            return kg, None

        corrupted_kg.remove_edge_by_id(edge.id)
        corrupted_kg.add_edge(corrupted_edge)

        record = CorruptionAuditRecord(
            original_edge_id=edge.id,
            corrupted_edge_id=corrupted_edge.id,
            operator_name=operator_name,
            source=edge.source,
            original_relation=edge.relation,
            corrupted_relation=corrupted_edge.relation,
            original_target=edge.target,
            corrupted_target=corrupted_edge.target,
            original_reliability=edge.base_reliability,
            corrupted_reliability=corrupted_edge.base_reliability,
        )
        return corrupted_kg, record
