"""Dynamic rerouting and evidence repair avoiding high-risk causal bottleneck edges."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from src.graph.schema import Edge, ReasoningPath
from src.graph.manager import KnowledgeGraphManager
from src.scoring.pwrs import PWRSScorer
from src.retrieval.path_finder import PathFinder


@dataclass
class RepairResult:
    """Encapsulates the outcome of a risk-aware repair operation."""
    original_path: ReasoningPath
    repaired_path: ReasoningPath
    flagged_edge: Optional[Edge]
    original_pwrs: float
    repaired_pwrs: float
    status: str  # 'UNTOUCHED', 'REPAIRED', 'ABSTAINED'
    retry_count: int
    repair_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_path": self.original_path.linearize(),
            "repaired_path": self.repaired_path.linearize(),
            "flagged_edge_id": self.flagged_edge.id if self.flagged_edge else None,
            "original_pwrs": round(self.original_pwrs, 4),
            "repaired_pwrs": round(self.repaired_pwrs, 4),
            "status": self.status,
            "retry_count": self.retry_count,
            "repair_log": self.repair_log,
        }


class GraphRepairEngine:
    """Detects high-risk bottleneck edges and dynamically discovers clean alternative paths."""

    def __init__(
        self,
        kg_manager: KnowledgeGraphManager,
        pwrs_scorer: PWRSScorer,
        path_finder: PathFinder,
        max_retries: int = 2,
    ):
        self.kg = kg_manager
        self.scorer = pwrs_scorer
        self.finder = path_finder
        self.max_retries = max(1, max_retries)

    def repair_path(
        self, query: str, primary_path: ReasoningPath
    ) -> RepairResult:
        """Inspects primary path; if risk exceeds threshold, isolates worst edge and reroutes."""
        summary = self.scorer.evaluate_path(primary_path)

        # 1. Check if repair is necessary
        if summary.is_acceptable or not summary.bottleneck_edge:
            return RepairResult(
                original_path=primary_path,
                repaired_path=primary_path,
                flagged_edge=None,
                original_pwrs=summary.max_edge_pwrs,
                repaired_pwrs=summary.max_edge_pwrs,
                status="UNTOUCHED",
                retry_count=0,
                repair_log=["Primary path risk within acceptable threshold."],
            )

        # 2. Begin recovery on working copy of graph
        logs: List[str] = []
        working_kg = self.kg.copy()
        current_path = primary_path
        flagged_edge = summary.bottleneck_edge
        retry = 0

        logs.append(
            f"Flagged bottleneck edge '{flagged_edge.id}' with PWRS={summary.max_edge_pwrs:.4f} > threshold {self.scorer.risk_threshold}"
        )

        while retry < self.max_retries:
            retry += 1
            # Prune the culprit edge from working graph
            working_kg.remove_edge_by_id(flagged_edge.id)
            logs.append(f"Attempt {retry}: Pruned edge '{flagged_edge.id}'. Searching alternative paths.")

            # Create working path finder
            working_finder = PathFinder(working_kg, max_hops=self.finder.max_hops)
            pool = working_finder.find_candidate_paths(
                source_entity=primary_path.source_node or "",
                query=query,
                target_entity=primary_path.target_node,
                top_k=5,
            )

            if not pool.paths:
                logs.append(f"Attempt {retry}: No alternative path found between endpoints.")
                break

            # Find best alternative path
            best_alt = pool.paths[0]
            alt_summary = self.scorer.evaluate_path(best_alt)
            logs.append(
                f"Attempt {retry}: Found alternative path '{best_alt.linearize()}' with PWRS={alt_summary.max_edge_pwrs:.4f}"
            )

            if alt_summary.is_acceptable:
                best_alt.status = "REPAIRED"
                logs.append("Successful recovery: Selected alternative safe path.")
                return RepairResult(
                    original_path=primary_path,
                    repaired_path=best_alt,
                    flagged_edge=flagged_edge,
                    original_pwrs=summary.max_edge_pwrs,
                    repaired_pwrs=alt_summary.max_edge_pwrs,
                    status="REPAIRED",
                    retry_count=retry,
                    repair_log=logs,
                )
            else:
                # Still risky; update bottleneck and retry if quota permits
                flagged_edge = alt_summary.bottleneck_edge
                if not flagged_edge:
                    break

        # If loop exhausts without safe path, abstain or return safest found
        logs.append("Exceeded max repair retries. Returning original path with FLAGGED warning.")
        primary_path.status = "FLAGGED"
        return RepairResult(
            original_path=primary_path,
            repaired_path=primary_path,
            flagged_edge=summary.bottleneck_edge,
            original_pwrs=summary.max_edge_pwrs,
            repaired_pwrs=summary.max_edge_pwrs,
            status="FLAGGED",
            retry_count=retry,
            repair_log=logs,
        )
