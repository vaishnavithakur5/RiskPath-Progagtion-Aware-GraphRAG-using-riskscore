"""Entity Linker resolving natural language query mentions to Knowledge Graph nodes."""

from __future__ import annotations
import re
from typing import List, Dict, Tuple, Optional
from src.graph.manager import KnowledgeGraphManager
from src.graph.schema import Node


class EntityLinker:
    """Identifies and resolves entity mentions in text queries to Knowledge Graph nodes."""

    def __init__(self, kg_manager: KnowledgeGraphManager):
        self.kg = kg_manager
        # Build normalization lookup table: normalized_name -> node_id
        self._name_index: Dict[str, str] = {}
        self._alias_index: Dict[str, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Constructs lowercase indices for rapid token resolution."""
        for node_id, node in self.kg._nodes.items():
            norm_name = self._normalize(node.name)
            self._name_index[norm_name] = node_id
            for alias in node.aliases:
                norm_alias = self._normalize(alias)
                self._alias_index[norm_alias] = node_id

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, remove punctuation and extra whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    def link_entities(
        self, query: str, top_k: int = 3, min_similarity: float = 0.50
    ) -> List[Tuple[Node, float]]:
        """Extracts candidate entities mentioned in the query ranked by confidence."""
        norm_query = self._normalize(query)
        candidates: Dict[str, float] = {}

        # 1. Exact substring matches for full names
        for norm_name, node_id in self._name_index.items():
            if norm_name and norm_name in norm_query:
                # Weight by proportion of name length relative to query
                score = 0.95 + min(0.05, len(norm_name) / max(1, len(norm_query)))
                candidates[node_id] = max(candidates.get(node_id, 0.0), score)

        # 2. Exact substring matches for aliases
        for norm_alias, node_id in self._alias_index.items():
            if norm_alias and norm_alias in norm_query:
                candidates[node_id] = max(candidates.get(node_id, 0.0), 0.90)

        # 3. Token-level overlap (Jaccard) fallback if few matches found
        if len(candidates) < top_k:
            query_tokens = set(norm_query.split())
            for norm_name, node_id in self._name_index.items():
                name_tokens = set(norm_name.split())
                if not name_tokens:
                    continue
                intersection = query_tokens.intersection(name_tokens)
                if intersection:
                    jaccard = len(intersection) / len(query_tokens.union(name_tokens))
                    if jaccard >= min_similarity:
                        candidates[node_id] = max(candidates.get(node_id, 0.0), jaccard)

        # Sort candidates by score descending
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[Tuple[Node, float]] = []
        for node_id, score in sorted_candidates:
            node = self.kg.get_node(node_id)
            if node:
                results.append((node, round(score, 3)))

        return results

    def extract_source_anchor(self, query: str) -> Optional[Node]:
        """Convenience method returning the highest confidence candidate node."""
        matches = self.link_entities(query, top_k=1)
        return matches[0][0] if matches else None
