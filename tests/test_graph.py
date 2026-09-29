"""Unit tests for Graph schema, data structures, and KnowledgeGraphManager."""

import os
import tempfile
import pytest

from src.graph.schema import Node, Edge, Provenance, ReasoningPath
from src.graph.manager import KnowledgeGraphManager


def test_node_schema():
    node = Node(
        id="Christopher_Nolan",
        name="Christopher Nolan",
        entity_type="Person",
        description="British-American filmmaker",
        aliases=["C. Nolan"],
        attributes={"birth_year": 1970},
    )
    assert node.id == "Christopher_Nolan"
    data = node.to_dict()
    assert data["name"] == "Christopher Nolan"
    reconstructed = Node.from_dict(data)
    assert reconstructed.id == node.id
    assert reconstructed.aliases == ["C. Nolan"]


def test_edge_invariants_and_serialization():
    prov = Provenance(
        doc_id="wiki_doc_1",
        text_snippet="Christopher Nolan was educated at University College London.",
        confidence=0.95,
    )
    edge = Edge(
        source="Christopher_Nolan",
        relation="educated_at",
        target="University_College_London",
        base_reliability=0.85,
        provenance=[prov],
    )

    # Invariant: BaseRisk = 1.0 - BaseReliability
    assert abs(edge.base_risk - 0.15) < 1e-6
    assert edge.id == "Christopher_Nolan__[educated_at]__University_College_London"
    assert edge.to_triple_string() == "(Christopher_Nolan, educated_at, University_College_London)"

    # Test updating reliability
    edge.update_reliability(0.90)
    assert abs(edge.base_reliability - 0.90) < 1e-6
    assert abs(edge.base_risk - 0.10) < 1e-6

    # Test serialization
    edge_dict = edge.to_dict()
    reconstructed = Edge.from_dict(edge_dict)
    assert reconstructed.source == edge.source
    assert reconstructed.target == edge.target
    assert len(reconstructed.provenance) == 1
    assert reconstructed.provenance[0].doc_id == "wiki_doc_1"


def test_kg_manager_construction_and_traversal():
    mgr = KnowledgeGraphManager("TestKG")

    # Add 2-hop chain: Inception -> director -> Christopher Nolan -> educated_at -> UCL
    e1 = Edge(source="Inception", relation="director", target="Christopher_Nolan", base_reliability=0.95)
    e2 = Edge(source="Christopher_Nolan", relation="educated_at", target="UCL", base_reliability=0.90)
    # Add parallel noisy branch: Inception -> director -> John_Doe
    e3 = Edge(source="Inception", relation="director", target="John_Doe", base_reliability=0.20)

    mgr.add_edge(e1)
    mgr.add_edge(e2)
    mgr.add_edge(e3)

    assert mgr.node_count == 4  # Inception, Christopher_Nolan, UCL, John_Doe
    assert mgr.edge_count == 3

    # Neighbor lookup
    neighbors = mgr.get_neighbors("Inception")
    assert "Christopher_Nolan" in neighbors
    assert "John_Doe" in neighbors

    # Outgoing edges
    out_edges = mgr.get_outgoing_edges("Inception")
    assert len(out_edges) == 2

    # Path discovery
    paths = mgr.find_all_paths(source="Inception", target="UCL", max_hops=3)
    assert len(paths) == 1
    path = paths[0]
    assert path.hop_count == 2
    assert path.source_node == "Inception"
    assert path.target_node == "UCL"
    assert "(Inception, director, Christopher_Nolan) -> (Christopher_Nolan, educated_at, UCL)" in path.linearize()


def test_kg_subgraph_extraction():
    mgr = KnowledgeGraphManager("BigKG")
    mgr.add_edge(Edge(source="A", relation="r1", target="B"))
    mgr.add_edge(Edge(source="B", relation="r2", target="C"))
    mgr.add_edge(Edge(source="C", relation="r3", target="D"))
    mgr.add_edge(Edge(source="X", relation="r4", target="Y"))  # Disconnected component

    # Extract 1-hop subgraph from A
    sub1 = mgr.get_local_subgraph(center_node="A", max_hops=1)
    assert sub1.node_count == 2  # A, B
    assert sub1.edge_count == 1

    # Extract 2-hop subgraph from A
    sub2 = mgr.get_local_subgraph(center_node="A", max_hops=2)
    assert sub2.node_count == 3  # A, B, C
    assert sub2.edge_count == 2
    assert "D" not in sub2.graph.nodes


def test_kg_manager_json_persistence():
    mgr = KnowledgeGraphManager("PersistKG")
    e = Edge(
        source="Node_1",
        relation="connects_to",
        target="Node_2",
        base_reliability=0.75,
        is_corrupted=True,
    )
    mgr.add_edge(e)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        mgr.save_to_json(tmp_path)
        loaded = KnowledgeGraphManager.load_from_json(tmp_path)
        assert loaded.node_count == 2
        assert loaded.edge_count == 1
        loaded_edge = loaded.get_edge_by_id(e.id)
        assert loaded_edge is not None
        assert loaded_edge.is_corrupted is True
        assert abs(loaded_edge.base_reliability - 0.75) < 1e-6
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_kg_manager_deep_copy():
    mgr = KnowledgeGraphManager("OriginalKG")
    edge = Edge(source="P", relation="r", target="Q", base_reliability=0.8)
    mgr.add_edge(edge)

    copy_mgr = mgr.copy()
    assert copy_mgr.node_count == 2
    assert copy_mgr.edge_count == 1

    # Modify copy: remove edge
    copy_mgr.remove_edge_by_id(edge.id)
    assert copy_mgr.edge_count == 0
    assert mgr.edge_count == 1  # Original remains intact
