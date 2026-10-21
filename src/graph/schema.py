"""Data schemas and domain models for Knowledge Graph entities, edges, provenance, and paths."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


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
    """Represents a directed semantic relationship between two nodes with multi-hop risk metrics.

    Mathematical Formulation (Thesis Slide 10):
      Own Risk:        R_own(e) = 1.0 - BaseReliability(e)
      Inherited Risk:  R_inh(e) = lambda * Agg({ R_tot(p) for p in parents(e) })
      Total Risk:      R_tot(e) = 1.0 - (1.0 - R_own(e)) * (1.0 - R_inh(e))
      PWRS Score:      PWRS(e)  = R_tot(e) * [I_causal(e)]^gamma
    """
    source: str
    relation: str
    target: str
    id: Optional[str] = None
    base_reliability: float = 0.50
    own_risk: float = 0.50
    inherited_risk: float = 0.0
    total_risk: float = 0.50
    causal_importance: float = 0.0
    pwrs: float = 0.0
    provenance: List[Provenance] = field(default_factory=list)
    frequency: int = 1
    is_corrupted: bool = False
    corruption_metadata: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.source}__[{self.relation}]__{self.target}"
        # Maintain invariant: R_own = 1.0 - BaseReliability
        self.base_reliability = max(0.0, min(1.0, float(self.base_reliability)))
        self.own_risk = max(0.0, min(1.0, 1.0 - self.base_reliability))
        self.compute_total_risk()

    @property
    def base_risk(self) -> float:
        """Alias for own_risk maintaining backward compatibility."""
        return self.own_risk

    @base_risk.setter
    def base_risk(self, val: float) -> None:
        self.own_risk = max(0.0, min(1.0, float(val)))
        self.base_reliability = 1.0 - self.own_risk
        self.compute_total_risk()

    def update_reliability(self, reliability: float) -> None:
        """Update base reliability and recompute own risk."""
        self.base_reliability = max(0.0, min(1.0, float(reliability)))
        self.own_risk = 1.0 - self.base_reliability
        self.compute_total_risk()

    def compute_total_risk(self) -> float:
        """Computes total risk as probabilistic union: R_tot = 1 - (1 - R_own) * (1 - R_inh)."""
        self.total_risk = max(0.0, min(1.0, 1.0 - (1.0 - self.own_risk) * (1.0 - self.inherited_risk)))
        return self.total_risk

    def update_causal_metrics(self, importance: float, pwrs: float) -> None:
        """Update downstream causal importance and composite PWRS."""
        self.causal_importance = max(0.0, min(1.0, float(importance)))
        self.pwrs = max(0.0, min(1.0, float(pwrs)))

    def to_triple_string(self) -> str:
        """Linearize edge into natural language subject-predicate-object triple."""
        return f"({self.source}, {self.relation}, {self.target})"

    def to_risk_breakdown(self) -> Dict[str, Any]:
        """Returns the full 6-variable risk breakdown conforming to Slide 10."""
        return {
            "edge_id": self.id,
            "triple": self.to_triple_string(),
            "base_reliability": round(self.base_reliability, 4),
            "own_risk": round(self.own_risk, 4),
            "inherited_risk": round(self.inherited_risk, 4),
            "total_risk": round(self.total_risk, 4),
            "causal_importance": round(self.causal_importance, 4),
            "pwrs": round(self.pwrs, 4),
            "is_corrupted": self.is_corrupted,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "base_reliability": self.base_reliability,
            "own_risk": self.own_risk,
            "base_risk": self.own_risk,
            "inherited_risk": self.inherited_risk,
            "total_risk": self.total_risk,
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
        rel = float(data.get("base_reliability", 0.50))
        own_r = float(data.get("own_risk", data.get("base_risk", 1.0 - rel)))
        inh_r = float(data.get("inherited_risk", 0.0))
        tot_r = float(data.get("total_risk", 1.0 - (1.0 - own_r) * (1.0 - inh_r)))
        edge = cls(
            id=data.get("id"),
            source=data["source"],
            relation=data["relation"],
            target=data["target"],
            base_reliability=rel,
            own_risk=own_r,
            inherited_risk=inh_r,
            total_risk=tot_r,
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
    path_risk: float = 0.0  # max_{f in P} PWRS(f)
    max_risk: float = 0.0   # alias for path_risk
    cum_risk: float = 0.0
    pwrs_score: float = 0.0
    status: str = "SAFE"  # Options: 'SAFE', 'REPAIRED', 'ABSTAINED', 'FLAGGED'
    repair_history: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.path_risk == 0.0 and self.max_risk != 0.0:
            self.path_risk = self.max_risk
        elif self.path_risk != 0.0 and self.max_risk == 0.0:
            self.max_risk = self.path_risk

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

    def get_full_risk_table(self) -> List[Dict[str, Any]]:
        """Returns step-by-step risk metrics along the path."""
        return [e.to_risk_breakdown() for e in self.edges]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "semantic_score": self.semantic_score,
            "path_risk": self.path_risk,
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
        p_risk = float(data.get("path_risk", data.get("max_risk", 0.0)))
        return cls(
            edges=edges,
            semantic_score=float(data.get("semantic_score", 0.0)),
            path_risk=p_risk,
            max_risk=p_risk,
            cum_risk=float(data.get("cum_risk", 0.0)),
            pwrs_score=float(data.get("pwrs_score", p_risk)),
            status=data.get("status", "SAFE"),
            repair_history=data.get("repair_history"),
        )
