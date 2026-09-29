"""Interchangeable baseline retrieval pipelines for fair comparative experiments."""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.graph.schema import ReasoningPath, Edge
from src.graph.manager import KnowledgeGraphManager
from src.retrieval.path_finder import PathFinder, CandidatePool
from src.generation.client import BaseLLMClient
from src.generation.prompts import build_qa_prompt
from src.generation.trust_narrative import TrustNarrativeGenerator
from src.propagation.ablator import CausalAblator
from src.scoring.pwrs import PWRSScorer
from src.repair.recovery import GraphRepairEngine, RepairResult


@dataclass
class RetrievalOutput:
    """Standardized output produced by any GraphRAG retrieval pipeline."""
    method_name: str
    query: str
    selected_path: ReasoningPath
    generated_answer: str
    repair_result: Optional[RepairResult]
    trust_narrative: str
    latency_seconds: float
    candidate_paths_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "query": self.query,
            "selected_path": self.selected_path.linearize(),
            "selected_path_status": self.selected_path.status,
            "max_risk": self.selected_path.max_risk,
            "generated_answer": self.generated_answer,
            "repaired": self.repair_result.status == "REPAIRED" if self.repair_result else False,
            "trust_narrative": self.trust_narrative,
            "latency_seconds": round(self.latency_seconds, 4),
            "candidate_paths_count": self.candidate_paths_count,
        }


class BaseRetrievalPipeline(ABC):
    """Abstract interface governing all comparative GraphRAG retrieval pipelines."""

    def __init__(self, name: str, kg: KnowledgeGraphManager, llm: BaseLLMClient):
        self.name = name
        self.kg = kg
        self.llm = llm
        self.finder = PathFinder(kg, max_hops=3)

    @abstractmethod
    def execute(
        self,
        query: str,
        source_entity: str,
        target_entity: Optional[str] = None,
    ) -> RetrievalOutput:
        pass


class StandardGraphRAGPipeline(BaseRetrievalPipeline):
    """B1: Standard GraphRAG selecting candidate path with highest lexical/semantic relevance."""

    def __init__(self, kg: KnowledgeGraphManager, llm: BaseLLMClient):
        super().__init__("Standard_GraphRAG", kg, llm)

    def execute(
        self, query: str, source_entity: str, target_entity: Optional[str] = None
    ) -> RetrievalOutput:
        t0 = time.time()
        pool = self.finder.find_candidate_paths(source_entity, query, target_entity, top_k=5)
        selected_path = pool.top_path or ReasoningPath(edges=[])

        prompt = build_qa_prompt(query, selected_path)
        answer = self.llm.generate(prompt)
        dt = time.time() - t0

        narrative = TrustNarrativeGenerator.generate_narrative(
            query, selected_path, method_name=self.name
        )
        return RetrievalOutput(
            method_name=self.name,
            query=query,
            selected_path=selected_path,
            generated_answer=answer,
            repair_result=None,
            trust_narrative=narrative,
            latency_seconds=dt,
            candidate_paths_count=len(pool),
        )


class ShortestPathPipeline(BaseRetrievalPipeline):
    """B2: Shortest-Path RAG selecting the path with minimum hop count."""

    def __init__(self, kg: KnowledgeGraphManager, llm: BaseLLMClient):
        super().__init__("Shortest_Path_RAG", kg, llm)

    def execute(
        self, query: str, source_entity: str, target_entity: Optional[str] = None
    ) -> RetrievalOutput:
        t0 = time.time()
        pool = self.finder.find_candidate_paths(source_entity, query, target_entity, top_k=5)
        # Sort purely by hop count ascending, breaking ties with semantic score
        sorted_paths = sorted(pool.paths, key=lambda p: (p.hop_count, -p.semantic_score))
        selected_path = sorted_paths[0] if sorted_paths else ReasoningPath(edges=[])

        prompt = build_qa_prompt(query, selected_path)
        answer = self.llm.generate(prompt)
        dt = time.time() - t0

        narrative = TrustNarrativeGenerator.generate_narrative(
            query, selected_path, method_name=self.name
        )
        return RetrievalOutput(
            method_name=self.name,
            query=query,
            selected_path=selected_path,
            generated_answer=answer,
            repair_result=None,
            trust_narrative=narrative,
            latency_seconds=dt,
            candidate_paths_count=len(pool),
        )


class ReliabilityOnlyPipeline(BaseRetrievalPipeline):
    """B3: Reliability-Only RAG selecting the path with highest average static base reliability."""

    def __init__(self, kg: KnowledgeGraphManager, llm: BaseLLMClient):
        super().__init__("Reliability_Only_RAG", kg, llm)

    def execute(
        self, query: str, source_entity: str, target_entity: Optional[str] = None
    ) -> RetrievalOutput:
        t0 = time.time()
        pool = self.finder.find_candidate_paths(source_entity, query, target_entity, top_k=5)

        def mean_rel(p: ReasoningPath) -> float:
            return sum(e.base_reliability for e in p.edges) / max(1, len(p.edges))

        sorted_paths = sorted(pool.paths, key=lambda p: mean_rel(p), reverse=True)
        selected_path = sorted_paths[0] if sorted_paths else ReasoningPath(edges=[])

        prompt = build_qa_prompt(query, selected_path)
        answer = self.llm.generate(prompt)
        dt = time.time() - t0

        narrative = TrustNarrativeGenerator.generate_narrative(
            query, selected_path, method_name=self.name
        )
        return RetrievalOutput(
            method_name=self.name,
            query=query,
            selected_path=selected_path,
            generated_answer=answer,
            repair_result=None,
            trust_narrative=narrative,
            latency_seconds=dt,
            candidate_paths_count=len(pool),
        )


class RiskPathAblatedPipeline(BaseRetrievalPipeline):
    """B4: RiskPath without dynamic causal propagation (treats I_prop=1.0 uniformly)."""

    def __init__(self, kg: KnowledgeGraphManager, llm: BaseLLMClient, risk_threshold: float = 0.45):
        super().__init__("RiskPath_Ablated_Static", kg, llm)
        self.scorer = PWRSScorer(risk_threshold=risk_threshold)
        self.repair = GraphRepairEngine(kg, self.scorer, self.finder)

    def execute(
        self, query: str, source_entity: str, target_entity: Optional[str] = None
    ) -> RetrievalOutput:
        t0 = time.time()
        pool = self.finder.find_candidate_paths(source_entity, query, target_entity, top_k=5)
        # Assign static risk without ablation
        for p in pool.paths:
            for e in p.edges:
                e.causal_importance = 1.0
                e.pwrs = e.base_risk

        ranked = self.scorer.rank_candidate_paths(pool.paths)
        primary = ranked[0] if ranked else ReasoningPath(edges=[])

        repair_res = self.repair.repair_path(query, primary)
        final_path = repair_res.repaired_path

        prompt = build_qa_prompt(query, final_path)
        answer = self.llm.generate(prompt)
        dt = time.time() - t0

        narrative = TrustNarrativeGenerator.generate_narrative(
            query, final_path, repair_result=repair_res, method_name=self.name
        )
        return RetrievalOutput(
            method_name=self.name,
            query=query,
            selected_path=final_path,
            generated_answer=answer,
            repair_result=repair_res,
            trust_narrative=narrative,
            latency_seconds=dt,
            candidate_paths_count=len(pool),
        )


class RiskPathPipeline(BaseRetrievalPipeline):
    """B5: Full Proposed RiskPath with causal ablation, PWRS decision theory, and dynamic repair."""

    def __init__(
        self,
        kg: KnowledgeGraphManager,
        llm: BaseLLMClient,
        risk_threshold: float = 0.45,
        risk_lambda: float = 0.70,
        gamma: float = 1.0,
    ):
        super().__init__("RiskPath_Full", kg, llm)
        self.ablator = CausalAblator(llm, max_ablation_edges=3)
        self.scorer = PWRSScorer(gamma=gamma, risk_threshold=risk_threshold, relevance_penalty_lambda=risk_lambda)
        self.repair = GraphRepairEngine(kg, self.scorer, self.finder)

    def execute(
        self, query: str, source_entity: str, target_entity: Optional[str] = None
    ) -> RetrievalOutput:
        t0 = time.time()
        pool = self.finder.find_candidate_paths(source_entity, query, target_entity, top_k=5)
        if not pool.paths:
            return RetrievalOutput(
                method_name=self.name,
                query=query,
                selected_path=ReasoningPath(edges=[]),
                generated_answer="Information not available",
                repair_result=None,
                trust_narrative="No paths found.",
                latency_seconds=time.time() - t0,
                candidate_paths_count=0,
            )

        # 1. Take top candidate path and estimate causal importance via bounded ablation
        top_cand = pool.top_path
        self.ablator.evaluate_path_causality(query, top_cand)

        # 2. Evaluate PWRS and check if repair is necessary
        repair_res = self.repair.repair_path(query, top_cand)
        final_path = repair_res.repaired_path

        # 3. Final grounded answer generation
        prompt = build_qa_prompt(query, final_path)
        answer = self.llm.generate(prompt)
        dt = time.time() - t0

        narrative = TrustNarrativeGenerator.generate_narrative(
            query, final_path, repair_result=repair_res, method_name=self.name
        )
        return RetrievalOutput(
            method_name=self.name,
            query=query,
            selected_path=final_path,
            generated_answer=answer,
            repair_result=repair_res,
            trust_narrative=narrative,
            latency_seconds=dt,
            candidate_paths_count=len(pool),
        )


def create_pipeline(
    method_name: str,
    kg: KnowledgeGraphManager,
    llm: BaseLLMClient,
    config: Optional[Dict[str, Any]] = None,
) -> BaseRetrievalPipeline:
    """Factory creating any of the 5 standardized baseline pipelines."""
    cfg = config or {}
    thresh = cfg.get("risk_threshold", 0.45)
    r_lambda = cfg.get("relevance_risk_lambda", 0.70)
    gamma = cfg.get("gamma_sensitivity", 1.0)

    method_clean = method_name.lower().replace("-", "_").replace(" ", "_")
    if "standard" in method_clean:
        return StandardGraphRAGPipeline(kg, llm)
    elif "shortest" in method_clean:
        return ShortestPathPipeline(kg, llm)
    elif "reliability_only" in method_clean:
        return ReliabilityOnlyPipeline(kg, llm)
    elif "ablated" in method_clean:
        return RiskPathAblatedPipeline(kg, llm, risk_threshold=thresh)
    elif "riskpath" in method_clean:
        return RiskPathPipeline(kg, llm, risk_threshold=thresh, risk_lambda=r_lambda, gamma=gamma)
    else:
        raise ValueError(f"Unknown pipeline: '{method_name}'")
