"""Path retrieval engine using Yen's K-Shortest Paths with semantic and diversity ranking."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx

from src.graph.schema import Edge, ReasoningPath
from src.graph.manager import KnowledgeGraphManager


@dataclass
class CandidatePool:
    """Manages an ordered set of candidate reasoning paths retrieved for a query."""
    query: str
    source_entity: str
    paths: List[ReasoningPath]

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def top_path(self) -> Optional[ReasoningPath]:
        return self.paths[0] if self.paths else None

    def sort_by_semantic_score(self) -> None:
        self.paths.sort(key=lambda p: p.semantic_score, reverse=True)

    def sort_by_composite_score(self, risk_lambda: float = 0.70) -> None:
        """Ranks paths by objective: J(P) = RelScore(P) - lambda * PWRS_max(P)."""
        self.paths.sort(
            key=lambda p: (p.semantic_score - risk_lambda * p.max_risk),
            reverse=True,
        )

    def filter_by_risk(self, max_allowed_risk: float) -> List[ReasoningPath]:
        """Returns paths where maximum edge risk is below threshold."""
        return [p for p in self.paths if p.max_risk <= max_allowed_risk]


class PathFinder:
    """Discovers and ranks candidate multi-hop paths connecting query entities."""

    def __init__(self, kg_manager: KnowledgeGraphManager, max_hops: int = 3):
        self.kg = kg_manager
        self.max_hops = max(1, max_hops)

    @staticmethod
    def _compute_triple_lexical_similarity(query_tokens: Set[str], edge: Edge) -> float:
        """Computes lexical token overlap between query and triple elements."""
        triple_text = f"{edge.source} {edge.relation} {edge.target}".lower().replace("_", " ")
        triple_tokens = set(triple_text.split())
        if not triple_tokens or not query_tokens:
            return 0.10

        intersection = query_tokens.intersection(triple_tokens)
        union = query_tokens.union(triple_tokens)
        jaccard = len(intersection) / len(union)
        return min(1.0, max(0.05, jaccard * 2.0))  # Scaled for sensitivity

    def score_path_relevance(self, query: str, path: ReasoningPath) -> float:
        """Assigns an aggregate semantic relevance score to a reasoning path."""
        if not path.edges:
            return 0.0

        query_tokens = set(query.lower().replace("?", "").replace(",", "").split())
        edge_scores = [
            self._compute_triple_lexical_similarity(query_tokens, e)
            for e in path.edges
        ]
        # Hop penalty: slightly penalize excessively long paths for parsimony
        hop_penalty = 1.0 - 0.05 * (len(path.edges) - 1)
        mean_edge_relevance = sum(edge_scores) / len(edge_scores)
        score = max(0.01, min(1.0, mean_edge_relevance * hop_penalty))
        path.semantic_score = round(score, 4)
        return path.semantic_score

    def find_candidate_paths(
        self,
        source_entity: str,
        query: str,
        target_entity: Optional[str] = None,
        top_k: int = 5,
    ) -> CandidatePool:
        """Retrieves top-k candidate reasoning paths from source to target or local neighborhood."""
        if not self.kg.graph.has_node(source_entity):
            return CandidatePool(query=query, source_entity=source_entity, paths=[])

        query_tokens = set(query.lower().split())

        # If explicit target entity is provided, find paths directly
        candidate_paths: List[ReasoningPath] = []
        if target_entity and self.kg.graph.has_node(target_entity):
            paths = self.kg.find_all_paths(
                source=source_entity, target=target_entity, max_hops=self.max_hops, max_paths=top_k * 2
            )
            candidate_paths.extend(paths)
        else:
            # Explore local neighborhood up to max_hops
            sub = self.kg.get_local_subgraph(center_node=source_entity, max_hops=self.max_hops)
            leaf_nodes = [
                n for n in sub.graph.nodes
                if n != source_entity and sub.graph.out_degree(n) == 0
            ]
            if not leaf_nodes:
                leaf_nodes = [n for n in sub.graph.nodes if n != source_entity]

            for leaf in leaf_nodes[:10]:
                paths = sub.find_all_paths(
                    source=source_entity, target=leaf, max_hops=self.max_hops, max_paths=2
                )
                candidate_paths.extend(paths)

        # Score all candidate paths
        for path in candidate_paths:
            self.score_path_relevance(query, path)
            # Initialize path max_risk from edge base_risk
            if path.edges:
                path.max_risk = max(e.base_risk for e in path.edges)
                path.cum_risk = 1.0
                for e in path.edges:
                    path.cum_risk *= (1.0 - e.base_risk)
                path.cum_risk = 1.0 - path.cum_risk

        # Deduplicate paths by linearized representation
        unique_paths: Dict[str, ReasoningPath] = {}
        for p in candidate_paths:
            lin = p.linearize()
            if lin not in unique_paths or p.semantic_score > unique_paths[lin].semantic_score:
                unique_paths[lin] = p

        ranked_paths = sorted(
            unique_paths.values(), key=lambda p: p.semantic_score, reverse=True
        )[:top_k]

        return CandidatePool(
            query=query, source_entity=source_entity, paths=ranked_paths
        )
