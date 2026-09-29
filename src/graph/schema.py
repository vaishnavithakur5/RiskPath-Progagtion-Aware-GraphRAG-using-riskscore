"""Data schemas and domain models for Knowledge Graph entities, edges, provenance, and paths."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class Provenance:
    """Represents documentary evidence supporting an extracted knowledge graph edge."""
    doc_id: str
    text_snippet: str
    confidence: float = 1.0
    source_url: str = ""
    extractor: str = "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text_snippet": self.text_snippet,
            "confidence": self.confidence,
            "source_url": self.source_url,
            "extractor": self.extractor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Provenance:
        return cls(
            doc_id=data.get("doc_id", ""),
            text_snippet=data.get("text_snippet", ""),
            confidence=float(data.get("confidence", 1.0)),
            source_url=data.get("source_url", ""),
            extractor=data.get("extractor", "manual"),
        )


@dataclass
class Node:
    """Represents a node/entity within the Knowledge Graph."""
    id: str
    name: str
    entity_type: str = "Entity"
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "aliases": self.aliases,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Node:
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data.get("entity_type", "Entity"),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            attributes=data.get("attributes", {}),
        )


@dataclass
class Edge:
    """Represents a directed semantic relationship between two nodes, along with risk metrics."""
    source: str
    relation: str
    target: str
    id: Optional[str] = None
    base_reliability: float = 0.50
    base_risk: float = 0.50
    causal_importance: float = 0.0
    pwrs: float = 0.0
    provenance: List[Provenance] = field(default_factory=list)
    frequency: int = 1
    is_corrupted: bool = False
    corruption_metadata: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            # Deterministic ID based on source, relation, target if not explicitly given
            self.id = f"{self.source}__[{self.relation}]__{self.target}"
        # Maintain invariant: BaseRisk = 1.0 - BaseReliability
        self.base_risk = max(0.0, min(1.0, 1.0 - self.base_reliability))

    def update_reliability(self, reliability: float) -> None:
        """Update base reliability and maintain base_risk invariant."""
        self.base_reliability = max(0.0, min(1.0, float(reliability)))
        self.base_risk = 1.0 - self.base_reliability

    def update_causal_metrics(self, importance: float, pwrs: float) -> None:
        """Update downstream causal importance and composite PWRS."""
        self.causal_importance = max(0.0, min(1.0, float(importance)))
        self.pwrs = max(0.0, min(1.0, float(pwrs)))

    def to_triple_string(self) -> str:
        """Linearize edge into natural language subject-predicate-object triple."""
        return f"({self.source}, {self.relation}, {self.target})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "base_reliability": self.base_reliability,
            "base_risk": self.base_risk,
            "causal_importance": self.causal_importance,
            "pwrs": self.pwrs,
            "provenance": [p.to_dict() for p in self.provenance],
            "frequency": self.frequency,
            "is_corrupted": self.is_corrupted,
            "corruption_metadata": self.corruption_metadata,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Edge:
        prov = [Provenance.from_dict(p) for p in data.get("provenance", [])]
        edge = cls(
            id=data.get("id"),
            source=data["source"],
            relation=data["relation"],
            target=data["target"],
            base_reliability=float(data.get("base_reliability", 0.50)),
            causal_importance=float(data.get("causal_importance", 0.0)),
            pwrs=float(data.get("pwrs", 0.0)),
            provenance=prov,
            frequency=int(data.get("frequency", 1)),
            is_corrupted=bool(data.get("is_corrupted", False)),
            corruption_metadata=data.get("corruption_metadata"),
            attributes=data.get("attributes", {}),
        )
        return edge


# Alias Triple to Edge for conceptual clarity in multi-hop reasoning
Triple = Edge


@dataclass
class ReasoningPath:
    """Represents a continuous sequence of edges forming a multi-hop reasoning path."""
    edges: List[Edge]
    semantic_score: float = 0.0
    max_risk: float = 0.0
    cum_risk: float = 0.0
    pwrs_score: float = 0.0
    status: str = "SAFE"  # Options: 'SAFE', 'FLAGGED', 'REPAIRED', 'ABSTAINED'
    repair_history: Optional[List[Dict[str, Any]]] = None

    @property
    def hop_count(self) -> int:
        return len(self.edges)

    @property
    def source_node(self) -> Optional[str]:
        return self.edges[0].source if self.edges else None

    @property
    def target_node(self) -> Optional[str]:
        return self.edges[-1].target if self.edges else None

    def linearize(self) -> str:
        """Formats the reasoning path into an ordered sequence of triples."""
        if not self.edges:
            return "No path evidence."
        return " -> ".join([e.to_triple_string() for e in self.edges])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "semantic_score": self.semantic_score,
            "max_risk": self.max_risk,
            "cum_risk": self.cum_risk,
            "pwrs_score": self.pwrs_score,
            "status": self.status,
            "hop_count": self.hop_count,
            "linearized": self.linearize(),
            "repair_history": self.repair_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningPath:
        edges = [Edge.from_dict(e) for e in data.get("edges", [])]
        return cls(
            edges=edges,
            semantic_score=float(data.get("semantic_score", 0.0)),
            max_risk=float(data.get("max_risk", 0.0)),
            cum_risk=float(data.get("cum_risk", 0.0)),
            pwrs_score=float(data.get("pwrs_score", 0.0)),
            status=data.get("status", "SAFE"),
            repair_history=data.get("repair_history"),
        )
