"""Unit tests for standardized baseline pipelines."""

import pytest
from src.extraction.benchmark_loader import BenchmarkDataset
from src.generation.client import MockLocalClient
from src.baselines.pipeline import create_pipeline


def test_baseline_pipelines_execution():
    ds = BenchmarkDataset.create_curated_multihop_benchmark(num_distractor_clusters=2)
    llm = MockLocalClient()
    sample_query = ds.queries[0]

    baselines_to_test = [
        "standard_graphrag",
        "shortest_path",
        "reliability_only",
        "causal_rag",
        "propagation_rag",
        "riskpath_ablated",
        "riskpath_full",
    ]

    for b_name in baselines_to_test:
        pipeline = create_pipeline(b_name, ds.kg, llm)
        output = pipeline.execute(
            query=sample_query.question,
            source_entity=sample_query.source_entity,
            target_entity=sample_query.target_entity,
        )

        assert output.method_name is not None
        assert output.generated_answer != ""
        assert output.selected_path is not None
        assert output.latency_seconds >= 0.0
        assert "Trust Narrative" in output.trust_narrative
