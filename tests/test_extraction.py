"""Unit tests for EntityLinker and BenchmarkDataset."""

import os
import tempfile
import pytest

from src.extraction.benchmark_loader import BenchmarkDataset, BenchmarkQuery
from src.extraction.entity_linker import EntityLinker


def test_benchmark_dataset_creation_and_stats():
    ds = BenchmarkDataset.create_curated_multihop_benchmark(num_distractor_clusters=5)
    stats = ds.kg.get_stats()

    assert stats["nodes"] > 30
    assert stats["edges"] > 30
    assert len(ds.queries) >= 20  # 10 cinema 2-hop + 10 cinema 3-hop + 8 science

    # Verify query path edges exist in the graph
    sample_query = ds.queries[0]
    for edge_id in sample_query.ground_truth_path_edges:
        edge = ds.kg.get_edge_by_id(edge_id)
        assert edge is not None, f"Ground truth edge {edge_id} must exist in graph"


def test_entity_linker_resolution():
    ds = BenchmarkDataset.create_curated_multihop_benchmark(num_distractor_clusters=2)
    linker = EntityLinker(ds.kg)

    # Test exact query containing film name
    query1 = "Where was the director of Inception educated?"
    matches1 = linker.link_entities(query1, top_k=3)
    assert len(matches1) > 0
    assert matches1[0][0].id == "Inception"
    assert matches1[0][1] >= 0.90

    # Test extract_source_anchor convenience method
    anchor = linker.extract_source_anchor("What did Albert Einstein discover?")
    assert anchor is not None
    assert anchor.id == "Albert_Einstein"


def test_benchmark_dataset_json_serialization():
    ds = BenchmarkDataset.create_curated_multihop_benchmark(num_distractor_clusters=2)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ds.save(tmp_path)
        loaded = BenchmarkDataset.load(tmp_path)
        assert loaded.kg.node_count == ds.kg.node_count
        assert loaded.kg.edge_count == ds.kg.edge_count
        assert len(loaded.queries) == len(ds.queries)
        assert loaded.queries[0].query_id == ds.queries[0].query_id
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_2wikimultihopqa_benchmark_loading():
    """Validates that official 2WikiMultihopQA benchmark loads with nodes, edges, queries, and provenance."""
    ds = BenchmarkDataset.load_benchmark("2wiki")

    assert ds.kg.node_count >= 1500
    assert ds.kg.edge_count >= 350
    assert len(ds.queries) == 200

    q0 = ds.queries[0]
    assert q0.source_entity != ""
    assert q0.target_entity != ""
    assert q0.ground_truth_answer != ""

    # Verify edge provenance
    sample_edge = next(iter(ds.kg._edges.values()))
    assert sample_edge.provenance is not None
    assert sample_edge.base_reliability > 0.0

