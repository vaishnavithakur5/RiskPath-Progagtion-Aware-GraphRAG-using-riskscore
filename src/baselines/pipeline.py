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
    is_abstained: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "query": self.query,
            "selected_path": self.selected_path.linearize(),
            "selected_path_status": self.selected_path.status,
            "max_risk": self.selected_path.max_risk,
            "generated_answer": self.generated_answer,
            "repaired": self.repair_result.status == "REPAIRED" if self.repair_result else False,
            "is_abstained": self.is_abstained,
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


class CausalRAGPipeline(BaseRetrievalPipeline):
    """B4: Causal-RAG (Slide 13: Own risk ✓, Propagation -, Causal ✓).
    
    Evaluates static own risk and causal ablation, without graph-level inherited risk propagation.
    """

    def __init__(
        self,
        kg: KnowledgeGraphManager,
        llm: BaseLLMClient,
        risk_threshold: float = 0.55,
        gamma: float = 1.0,
    ):
        super().__init__("Causal_RAG", kg, llm)
        self.risk_threshold = risk_threshold
        self.gamma = gamma
        self.ablator = CausalAblator(llm, max_ablation_edges=3)
        # PWRSScorer with lambda_propagation=0.0 so R_inh is strictly 0 and R_tot = R_own
        self.scorer = PWRSScorer(gamma=gamma, risk_threshold=risk_threshold, lambda_propagation=0.0)
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
                is_abstained=False,
            )

        top_cand = pool.top_path
        # Ablation to estimate causal importance
        self.ablator.evaluate_path_causality(query, top_cand)

        # In Causal-RAG, no propagation: R_inh = 0, R_tot = R_own
        for e in top_cand.edges:
            e.inherited_risk = 0.0
            e.total_risk = e.own_risk
            self.scorer.compute_edge_pwrs(e)

        repair_res = self.repair.repair_path(query, top_cand)
        final_path = repair_res.repaired_path
        is_abstained = repair_res.status == "ABSTAINED"

        if is_abstained:
            final_path.status = "ABSTAINED"
            answer = "Information abstained: All retrieved reasoning paths exceed epistemic risk threshold (PWRS > 0.55)."
        else:
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
            is_abstained=is_abstained,
        )


class PropagationRAGPipeline(BaseRetrievalPipeline):
    """B5: Propagation-RAG (Slide 13: Own risk ✓, Propagation ✓, Causal -).
    
    Evaluates own risk and sequential inherited risk propagation, without dynamic causal ablation.
    """

    def __init__(
        self,
        kg: KnowledgeGraphManager,
        llm: BaseLLMClient,
        risk_threshold: float = 0.55,
        lambda_propagation: float = 0.60,
    ):
        super().__init__("Propagation_RAG", kg, llm)
        self.risk_threshold = risk_threshold
        self.scorer = PWRSScorer(gamma=1.0, risk_threshold=risk_threshold, lambda_propagation=lambda_propagation)
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
                is_abstained=False,
            )

        # In Propagation-RAG, no causal ablation: I_causal = 1.0 uniformly
        for p in pool.paths:
            for e in p.edges:
                e.causal_importance = 1.0

        ranked = self.scorer.rank_candidate_paths(pool.paths)
        primary = ranked[0] if ranked else ReasoningPath(edges=[])

        repair_res = self.repair.repair_path(query, primary)
        final_path = repair_res.repaired_path
        is_abstained = repair_res.status == "ABSTAINED"

        if is_abstained:
            final_path.status = "ABSTAINED"
            answer = "Information abstained: All retrieved reasoning paths exceed epistemic risk threshold (PWRS > 0.55)."
        else:
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
            is_abstained=is_abstained,
        )


class RiskPathAblatedPipeline(PropagationRAGPipeline):
    """Backward-compatible alias for Propagation-RAG ablation pipeline."""
    def __init__(self, kg: KnowledgeGraphManager, llm: BaseLLMClient, risk_threshold: float = 0.55):
        super().__init__(kg, llm, risk_threshold=risk_threshold)
        self.name = "RiskPath_Ablated_Static"


class RiskPathPipeline(BaseRetrievalPipeline):
    """B6: Full Proposed RiskPath (Slide 13: Own risk ✓, Propagation ✓, Causal ✓).
    
    Fuses own risk, sequential propagation, causal ablation, and dynamic repair/abstention.
    """

    def __init__(
        self,
        kg: KnowledgeGraphManager,
        llm: BaseLLMClient,
        risk_threshold: float = 0.55,
        risk_lambda: float = 0.70,
        gamma: float = 1.0,
        lambda_propagation: float = 0.60,
    ):
        super().__init__("RiskPath_Full", kg, llm)
        self.ablator = CausalAblator(llm, max_ablation_edges=3)
        self.scorer = PWRSScorer(
            gamma=gamma,
            risk_threshold=risk_threshold,
            lambda_propagation=lambda_propagation,
            relevance_penalty_lambda=risk_lambda,
        )
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
                is_abstained=False,
            )

        # 1. Take top candidate path and estimate causal importance via bounded ablation
        top_cand = pool.top_path
        self.ablator.evaluate_path_causality(query, top_cand)

        # 2. Evaluate PWRS and check if repair/abstention is necessary
        repair_res = self.repair.repair_path(query, top_cand)
        final_path = repair_res.repaired_path
        is_abstained = repair_res.status == "ABSTAINED"

        # 3. Final grounded answer generation
        if is_abstained:
            final_path.status = "ABSTAINED"
            answer = "Information abstained: All retrieved reasoning paths exceed epistemic risk threshold (PWRS > 0.55)."
        else:
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
            is_abstained=is_abstained,
        )


def create_pipeline(
    method_name: str,
    kg: KnowledgeGraphManager,
    llm: BaseLLMClient,
    config: Optional[Dict[str, Any]] = None,
) -> BaseRetrievalPipeline:
    """Factory creating any of the 6 standardized comparative pipelines (Slide 13)."""
    cfg = config or {}
    thresh = cfg.get("risk_threshold", 0.55)
    r_lambda = cfg.get("relevance_risk_lambda", 0.70)
    gamma = cfg.get("gamma_sensitivity", 1.0)
    lambda_prop = cfg.get("lambda_propagation", 0.60)

    method_clean = method_name.lower().replace("-", "_").replace(" ", "_")
    if "standard" in method_clean:
        return StandardGraphRAGPipeline(kg, llm)
    elif "shortest" in method_clean:
        return ShortestPathPipeline(kg, llm)
    elif "reliability_only" in method_clean or method_clean == "reliability":
        return ReliabilityOnlyPipeline(kg, llm)
    elif "causal" in method_clean:
        return CausalRAGPipeline(kg, llm, risk_threshold=thresh, gamma=gamma)
    elif "propagation" in method_clean:
        return PropagationRAGPipeline(kg, llm, risk_threshold=thresh, lambda_propagation=lambda_prop)
    elif "ablated" in method_clean:
        return RiskPathAblatedPipeline(kg, llm, risk_threshold=thresh)
    elif "riskpath" in method_clean:
        return RiskPathPipeline(
            kg,
            llm,
            risk_threshold=thresh,
            risk_lambda=r_lambda,
            gamma=gamma,
            lambda_propagation=lambda_prop,
        )
    else:
        raise ValueError(f"Unknown pipeline: '{method_name}'")
