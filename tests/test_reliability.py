"""Unit tests for Base Reliability Model."""

import pytest
from src.graph.schema import Edge, Provenance
from src.graph.manager import KnowledgeGraphManager
from src.reliability.base_scorer import BaseReliabilityScorer, ReliabilityBreakdown


def test_base_reliability_provenance_sensitivity():
    kg = KnowledgeGraphManager("TestReliabilityKG")

    # High provenance edge (3 supporting documents, high confidence)
    e_strong = Edge(
        source="EntityA",
        relation="founded",
        target="EntityB",
        frequency=5,
        provenance=[
            Provenance(doc_id="d1", text_snippet="EntityA founded EntityB in 2005.", confidence=0.95),
            Provenance(doc_id="d2", text_snippet="EntityA founded EntityB.", confidence=0.90),
            Provenance(doc_id="d3", text_snippet="Historical record shows EntityA founded EntityB.", confidence=0.92),
        ],
    )

    # Weak ungrounded edge
    e_weak = Edge(
        source="EntityA",
        relation="acquired",
        target="EntityC",
        frequency=1,
        provenance=[],
    )

    kg.add_edge(e_strong)
    kg.add_edge(e_weak)

    scorer = BaseReliabilityScorer(kg)
    b_strong = scorer.compute_edge_reliability(e_strong)
    b_weak = scorer.compute_edge_reliability(e_weak)

    assert b_strong.composite_reliability > b_weak.composite_reliability
    assert b_strong.provenance_score > b_weak.provenance_score
    assert b_strong.base_risk < b_weak.base_risk
    assert abs((b_strong.composite_reliability + b_strong.base_risk) - 1.0) < 1e-5
    assert abs((b_weak.composite_reliability + b_weak.base_risk) - 1.0) < 1e-5


def test_score_entire_graph():
    kg = KnowledgeGraphManager("WholeGraphKG")
    for i in range(5):
        e = Edge(source=f"Node_{i}", relation="relates", target=f"Node_{i+1}")
        kg.add_edge(e)

    scorer = BaseReliabilityScorer(kg)
    breakdowns = scorer.score_entire_graph()

    assert len(breakdowns) == 5
    for edge_id, edge in kg._edges.items():
        assert edge.base_reliability > 0.0
        assert edge.base_risk > 0.0
        assert abs((edge.base_reliability + edge.base_risk) - 1.0) < 1e-5
