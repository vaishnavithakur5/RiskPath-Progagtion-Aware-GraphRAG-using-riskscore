"""Unit tests for IARP corruption operators and CorruptionEngine."""

import pytest
from src.graph.schema import Edge
from src.graph.manager import KnowledgeGraphManager
from src.corruption.engine import CorruptionEngine
from src.corruption.operators import get_operator


def test_corruption_operators_execution():
    kg = KnowledgeGraphManager("OpKG")
    e1 = Edge(source="NodeA", relation="director", target="NodeB", base_reliability=0.9)
    e2 = Edge(source="NodeB", relation="educated_at", target="NodeC", base_reliability=0.9)
    kg.add_edge(e1)
    kg.add_edge(e2)

    engine = CorruptionEngine(random_seed=42)
    corrupted_kg, records = engine.inject_noise(kg, operator_name="entity_swap", noise_rate=0.5)

    assert len(records) >= 1
    assert corrupted_kg.edge_count == 2
    corrupted_edge = corrupted_kg.get_edge_by_id(records[0].corrupted_edge_id)
    assert corrupted_edge is not None
    assert corrupted_edge.is_corrupted is True
    assert corrupted_edge.base_reliability < 0.90


def test_targeted_single_edge_corruption():
    kg = KnowledgeGraphManager("TargetKG")
    e = Edge(source="Inception", relation="director", target="Christopher_Nolan", base_reliability=0.95)
    kg.add_edge(e)

    engine = CorruptionEngine(random_seed=123)
    corrupted_kg, record = engine.inject_single_targeted_corruption(
        kg, target_edge_id=e.id, operator_name="relation_swap"
    )

    assert record is not None
    assert record.original_edge_id == e.id
    assert corrupted_kg.get_edge_by_id(e.id) is None  # Original replaced
    assert corrupted_kg.edge_count == 1
    new_edge = corrupted_kg.get_edge_by_id(record.corrupted_edge_id)
    assert new_edge is not None
    assert new_edge.is_corrupted is True
