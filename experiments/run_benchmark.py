"""CLI script to run full offline GraphRAG comparative evaluation experiments."""

import os
import sys
import argparse

# Add riskpath root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extraction.benchmark_loader import BenchmarkDataset
from src.generation.client import get_llm_client
from src.evaluation.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run RiskPath GraphRAG Benchmark")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "ollama"], help="LLM backend")
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b", help="Model name for Ollama")
    parser.add_argument("--queries", type=int, default=10, help="Number of queries to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=================================================================")
    print("RiskPath: Propagation-Aware Risk Retrieval for GraphRAG Benchmark")
    print("=================================================================")
    print(f"Backend: {args.backend} | Model: {args.model} | Queries: {args.queries} | Seed: {args.seed}")

    ds = BenchmarkDataset.create_curated_multihop_benchmark(num_distractor_clusters=10)
    print(f"Loaded Benchmark KG with {ds.kg.node_count} nodes and {ds.kg.edge_count} edges.")
    print(f"Total available benchmark queries: {len(ds.queries)}")

    llm = get_llm_client(backend=args.backend, model_name=args.model)
    runner = ExperimentRunner(dataset=ds, llm_client=llm, random_seed=args.seed)

    methods = [
        "standard_graphrag",
        "shortest_path",
        "reliability_only",
        "riskpath_ablated",
        "riskpath_full",
    ]
    noise_rates = [0.0, 0.10, 0.20]

    print("\nExecuting evaluation matrix...")
    df_summary = runner.run_matrix(
        methods=methods,
        noise_rates=noise_rates,
        operator_name="entity_swap",
        max_queries=args.queries,
    )

    print("\nBenchmark Summary Results:")
    print(df_summary.to_string(index=False))
    print(f"\nResults saved to '{runner.output_dir}'")


if __name__ == "__main__":
    main()
