"""Batch experiment runner executing the GraphRAG comparative evaluation matrix."""

from __future__ import annotations
import os
import time
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.extraction.benchmark_loader import BenchmarkDataset, BenchmarkQuery
from src.generation.client import BaseLLMClient
from src.corruption.engine import CorruptionEngine
from src.baselines.pipeline import create_pipeline
from .metrics import evaluate_prediction, QueryEvaluationResult


@dataclass
class BenchmarkSummary:
    """Aggregated evaluation results for a specific method and noise condition."""
    method_name: str
    noise_rate: float
    noise_operator: str
    num_queries: int
    mean_exact_match: float
    mean_token_f1: float
    corrupted_edge_avoidance_rate: float
    successful_recovery_rate: float
    avg_latency_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "noise_rate": round(self.noise_rate, 2),
            "noise_operator": self.noise_operator,
            "num_queries": self.num_queries,
            "mean_exact_match": round(self.mean_exact_match, 4),
            "mean_token_f1": round(self.mean_token_f1, 4),
            "cear": round(self.corrupted_edge_avoidance_rate, 4),
            "srr": round(self.successful_recovery_rate, 4),
            "avg_latency_s": round(self.avg_latency_seconds, 4),
        }


class ExperimentRunner:
    """Orchestrates multi-method, multi-noise batch experiments for the M.Tech thesis."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        llm_client: BaseLLMClient,
        random_seed: int = 42,
        output_dir: str = "experiments/results",
    ):
        self.dataset = dataset
        self.llm = llm_client
        self.seed = random_seed
        self.output_dir = output_dir
        self.corruption_engine = CorruptionEngine(random_seed=random_seed)
        os.makedirs(output_dir, exist_ok=True)

    def run_matrix(
        self,
        methods: List[str],
        noise_rates: List[float] = [0.0, 0.10, 0.20],
        operator_name: str = "entity_swap",
        max_queries: Optional[int] = None,
    ) -> pd.DataFrame:
        """Executes full evaluation matrix across specified methods and noise rates."""
        queries_to_eval = self.dataset.queries[:max_queries] if max_queries else self.dataset.queries
        summaries: List[BenchmarkSummary] = []
        raw_rows: List[Dict[str, Any]] = []

        for noise in noise_rates:
            # Prepare graph condition
            if noise == 0.0:
                active_kg = self.dataset.kg.copy()
            else:
                active_kg, _ = self.corruption_engine.inject_noise(
                    self.dataset.kg, operator_name=operator_name, noise_rate=noise
                )

            for method in methods:
                pipeline = create_pipeline(method, active_kg, self.llm)
                method_results: List[QueryEvaluationResult] = []

                for q in queries_to_eval:
                    t0 = time.time()
                    out = pipeline.execute(
                        query=q.question,
                        source_entity=q.source_entity,
                        target_entity=q.target_entity,
                    )
                    lat = time.time() - t0

                    eval_res = evaluate_prediction(
                        query_id=q.query_id,
                        method_name=method,
                        prediction=out.generated_answer,
                        ground_truth=q.ground_truth_answer,
                        selected_path=out.selected_path,
                        was_repaired=(out.repair_result.status == "REPAIRED") if out.repair_result else False,
                        latency=lat,
                    )
                    method_results.append(eval_res)

                    row_dict = eval_res.to_dict()
                    row_dict["noise_rate"] = noise
                    row_dict["operator"] = operator_name
                    row_dict["prediction"] = out.generated_answer
                    row_dict["ground_truth"] = q.ground_truth_answer
                    row_dict["path"] = out.selected_path.linearize()
                    raw_rows.append(row_dict)

                # Aggregate summary
                n = len(method_results)
                mean_em = sum(r.exact_match for r in method_results) / max(1, n)
                mean_f1 = sum(r.token_f1 for r in method_results) / max(1, n)
                cear = sum(1 for r in method_results if r.avoided_corrupted_edge) / max(1, n)
                srr = sum(1 for r in method_results if r.repaired_successfully) / max(1, n)
                avg_lat = sum(r.latency_seconds for r in method_results) / max(1, n)

                summary = BenchmarkSummary(
                    method_name=method,
                    noise_rate=noise,
                    noise_operator=operator_name,
                    num_queries=n,
                    mean_exact_match=mean_em,
                    mean_token_f1=mean_f1,
                    corrupted_edge_avoidance_rate=cear,
                    successful_recovery_rate=srr,
                    avg_latency_seconds=avg_lat,
                )
                summaries.append(summary)

        # Convert to DataFrames
        summary_df = pd.DataFrame([s.to_dict() for s in summaries])
        raw_df = pd.DataFrame(raw_rows)

        # Save to disk
        timestamp = int(time.time())
        summary_path = os.path.join(self.output_dir, f"matrix_summary_{timestamp}.csv")
        raw_path = os.path.join(self.output_dir, f"matrix_raw_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        raw_df.to_csv(raw_path, index=False)

        return summary_df
