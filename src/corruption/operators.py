"""Implementation of 7 deterministic corruption operators for GraphRAG adversarial benchmarking."""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any

from src.graph.schema import Edge, Node, Provenance
from src.graph.manager import KnowledgeGraphManager


class CorruptionOperator(ABC):
    """Abstract base class for graph corruption operators."""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, kg: KnowledgeGraphManager, edge: Edge, rng: random.Random) -> Optional[Edge]:
        """Applies corruption to a target edge and returns modified or newly created edge."""
        pass


class EntitySwapOperator(CorruptionOperator):
    """Replaces the target entity with an alternative entity of matching or plausible type."""

    def name(self) -> str:
        return "entity_swap"

    def apply(self, kg: KnowledgeGraphManager, edge: Edge, rng: random.Random) -> Optional[Edge]:
        # Collect candidate alternative nodes
        candidate_nodes = [
            nid for nid in kg._nodes.keys()
            if nid != edge.target and nid != edge.source
        ]
        if not candidate_nodes:
            return None

        # Pick random alternative entity
        new_target = rng.choice(candidate_nodes)
        corrupted_edge = Edge(
            source=edge.source,
            relation=edge.relation,
            target=new_target,
            base_reliability=max(0.10, edge.base_reliability * 0.40),  # Drop reliability
            is_corrupted=True,
            corruption_metadata={
                "operator": self.name(),
                "original_target": edge.target,
                "corrupted_target": new_target,
            },
        )
        return corrupted_edge


class RelationSwapOperator(CorruptionOperator):
    """Replaces relation predicate with an semantically contradictory or incompatible predicate."""

    INCOMPATIBLE_RELATIONS = {
        "director": ["cinematographer", "rival_of", "critic_of"],
        "educated_at": ["expelled_from", "rejected_by", "unaffiliated_with"],
        "located_in": ["opposed_to", "distanced_from"],
        "discovered_by": ["disproved_by", "opposed_by"],
        "affiliated_with": ["banned_from", "lawsuit_against"],
    }

    def name(self) -> str:
        return "relation_swap"

    def apply(self, kg: KnowledgeGraphManager, edge: Edge, rng: random.Random) -> Optional[Edge]:
        rel = edge.relation
        alternatives = self.INCOMPATIBLE_RELATIONS.get(rel, ["unrelated_to", "disputed_by"])
        new_relation = rng.choice(alternatives)

        corrupted_edge = Edge(
            source=edge.source,
            relation=new_relation,
            target=edge.target,
            base_reliability=max(0.15, edge.base_reliability * 0.50),
            is_corrupted=True,
            corruption_metadata={
                "operator": self.name(),
                "original_relation": edge.relation,
                "corrupted_relation": new_relation,
            },
        )
        return corrupted_edge


class ContradictoryEdgeOperator(CorruptionOperator):
    """Injects a direct contradictory edge that conflicts with ground truth."""

    def name(self) -> str:
        return "contradictory_edge"

    def apply(self, kg: KnowledgeGraphManager, edge: Edge, rng: random.Random) -> Optional[Edge]:
        contradictory_relation = f"not_{edge.relation}"
        corrupted_edge = Edge(
            source=edge.source,
            relation=contradictory_relation,
            target=edge.target,
            base_reliability=0.20,
            is_corrupted=True,
            corruption_metadata={
                "operator": self.name(),
                "original_edge": edge.id,
            },
        )
        return corrupted_edge


class AdversarialBridgeOperator(CorruptionOperator):
    """Inserts a deceptive shortcut bridging multi-hop reasoning with a false destination."""

    def name(self) -> str:
        return "adversarial_bridge"

    def apply(self, kg: KnowledgeGraphManager, edge: Edge, rng: random.Random) -> Optional[Edge]:
        # Connect source directly to a random distinct node
        other_nodes = [nid for nid in kg._nodes.keys() if nid not in (edge.source, edge.target)]
        if not other_nodes:
            return None
        dest = rng.choice(other_nodes)
        corrupted_edge = Edge(
            source=edge.source,
            relation="direct_destination",
            target=dest,
            base_reliability=0.30,
            is_corrupted=True,
            corruption_metadata={
                "operator": self.name(),
                "shortcut_from": edge.source,
                "shortcut_to": dest,
            },
        )
        return corrupted_edge


class ProvenanceStripOperator(CorruptionOperator):
    """Strips provenance references and inflates base risk."""

    def name(self) -> str:
        return "provenance_strip"

    def apply(self, kg: KnowledgeGraphManager, edge: Edge, rng: random.Random) -> Optional[Edge]:
        corrupted_edge = Edge(
            source=edge.source,
            relation=edge.relation,
            target=edge.target,
            base_reliability=0.15,
            provenance=[],
            is_corrupted=True,
            corruption_metadata={"operator": self.name(), "action": "provenance_removed"},
        )
        return corrupted_edge


def get_operator(name: str) -> CorruptionOperator:
    """Factory returning corruption operator by name."""
    registry = {
        "entity_swap": EntitySwapOperator(),
        "relation_swap": RelationSwapOperator(),
        "contradictory_edge": ContradictoryEdgeOperator(),
        "adversarial_bridge": AdversarialBridgeOperator(),
        "provenance_strip": ProvenanceStripOperator(),
    }
    if name not in registry:
        raise ValueError(f"Unknown operator: '{name}'. Available: {list(registry.keys())}")
    return registry[name]
