"""Unit tests for evaluation metrics and ExperimentRunner."""

import pytest
import pandas as pd
from src.extraction.benchmark_loader import BenchmarkDataset
from src.generation.client import MockLocalClient
from src.evaluation.metrics import compute_exact_match, compute_token_f1
from src.evaluation.runner import ExperimentRunner


def test_metric_calculation_accuracy():
    # Exact match test with articles and punctuation
    assert compute_exact_match("University College London.", "University College London") == 1.0
    assert compute_exact_match("The University of Cambridge", "University of Cambridge") == 1.0
    assert compute_exact_match("Oxford", "Cambridge") == 0.0

    # Token F1
    assert compute_token_f1("University College London", "University College London") == 1.0
    assert compute_token_f1("London University College", "University College London") == 1.0
    f1_partial = compute_token_f1("University of Cambridge", "Cambridge")
    assert 0.0 < f1_partial < 1.0


def test_experiment_runner_mini_matrix():
    ds = BenchmarkDataset.create_curated_multihop_benchmark(num_distractor_clusters=2)
    llm = MockLocalClient()
    runner = ExperimentRunner(dataset=ds, llm_client=llm, random_seed=42)

    df_summary = runner.run_matrix(
        methods=["standard_graphrag", "riskpath_full"],
        noise_rates=[0.0, 0.10],
        operator_name="entity_swap",
        max_queries=4,  # Fast test slice
    )

    assert isinstance(df_summary, pd.DataFrame)
    assert len(df_summary) == 4  # 2 methods * 2 noise rates
    assert "mean_exact_match" in df_summary.columns
    assert "cear" in df_summary.columns
    assert "avg_latency_s" in df_summary.columns
