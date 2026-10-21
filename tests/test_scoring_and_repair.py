"""Unit tests for PWRSScorer and GraphRepairEngine."""

import pytest
from src.graph.schema import Edge, ReasoningPath
from src.graph.manager import KnowledgeGraphManager
from src.retrieval.path_finder import PathFinder
from src.scoring.pwrs import PWRSScorer
from src.repair.recovery import GraphRepairEngine


def test_pwrs_mathematical_invariants():
    scorer = PWRSScorer(gamma=1.0, risk_threshold=0.45)

    # Edge 1: Perfectly reliable (base_risk = 0), highly important (causal_imp = 1.0)
    e_safe = Edge(source="A", relation="r", target="B", base_reliability=1.0, causal_importance=1.0)
    assert scorer.compute_edge_pwrs(e_safe) == 0.0

    # Edge 2: Unreliable (base_risk = 0.8), but completely inert (causal_imp = 0.0)
    e_inert = Edge(source="A", relation="r", target="B", base_reliability=0.2, causal_importance=0.0)
    assert scorer.compute_edge_pwrs(e_inert) == 0.0

    # Edge 3: Highly unreliable (base_risk = 0.8) and highly critical (causal_imp = 0.9)
    e_critical = Edge(source="A", relation="r", target="B", base_reliability=0.2, causal_importance=0.9)
    assert abs(scorer.compute_edge_pwrs(e_critical) - 0.72) < 1e-4

    # Test path evaluation
    path = ReasoningPath(edges=[e_safe, e_critical])
    summary = scorer.evaluate_path(path)
    assert summary.max_edge_pwrs == 0.72
    assert summary.is_acceptable is False
    assert summary.bottleneck_edge == e_critical


def test_graph_repair_recovery():
    kg = KnowledgeGraphManager("RepairKG")

    # Path 1 (Shortest, but corrupted): Inception -> director -> Corrupted_Director (high risk) -> educated_at -> UCL
    e_corrupt_1 = Edge(source="Inception", relation="director", target="Fake_Director", base_reliability=0.1, causal_importance=0.9)
    e_corrupt_2 = Edge(source="Fake_Director", relation="educated_at", target="UCL", base_reliability=0.9, causal_importance=0.8)

    # Path 2 (Alternative clean): Inception -> writer -> True_Writer -> educated_at -> UCL
    e_clean_1 = Edge(source="Inception", relation="writer", target="True_Writer", base_reliability=0.95, causal_importance=0.9)
    e_clean_2 = Edge(source="True_Writer", relation="educated_at", target="UCL", base_reliability=0.95, causal_importance=0.9)

    kg.add_edge(e_corrupt_1)
    kg.add_edge(e_corrupt_2)
    kg.add_edge(e_clean_1)
    kg.add_edge(e_clean_2)

    primary_path = ReasoningPath(edges=[e_corrupt_1, e_corrupt_2])
    scorer = PWRSScorer(risk_threshold=0.45)
    finder = PathFinder(kg, max_hops=3)
    repair_engine = GraphRepairEngine(kg, scorer, finder, max_retries=2)

    query = "Where was the filmmaker of Inception educated?"
    result = repair_engine.repair_path(query, primary_path)

    assert result.status == "REPAIRED"
    assert result.flagged_edge is not None
    assert result.flagged_edge.id == e_corrupt_1.id
    assert result.repaired_pwrs < result.original_pwrs
    assert "True_Writer" in result.repaired_path.linearize()


def test_slide_10_numerical_example():
    """Validates the exact numerical values from Thesis Defense Slide 10."""
    eA = Edge(source="nodeA", relation="r1", target="nodeB", base_reliability=0.95, causal_importance=0.20)
    eB = Edge(source="nodeB", relation="r2", target="nodeC", base_reliability=0.90, causal_importance=0.30)
    eC = Edge(source="nodeC", relation="r3", target="nodeD", base_reliability=0.40, causal_importance=0.90)
    eD = Edge(source="nodeD", relation="r4", target="nodeE", base_reliability=0.95, causal_importance=0.40)

    chain = ReasoningPath(edges=[eA, eB, eC, eD])
    scorer = PWRSScorer(gamma=1.0, risk_threshold=0.55, lambda_propagation=0.60)
    summary = scorer.evaluate_path(chain)

    # Edge A: R_own=0.05, R_inh=0.00, R_tot=0.05, I_causal=0.20, PWRS=0.01
    assert round(eA.own_risk, 2) == 0.05
    assert round(eA.inherited_risk, 2) == 0.00
    assert round(eA.total_risk, 2) == 0.05
    assert round(eA.pwrs, 2) == 0.01

    # Edge B: R_own=0.10, R_inh=0.03, R_tot=0.13, I_causal=0.30, PWRS=0.04
    assert round(eB.own_risk, 2) == 0.10
    assert round(eB.inherited_risk, 2) == 0.03
    assert round(eB.total_risk, 2) == 0.13
    assert round(eB.pwrs, 2) == 0.04

    # Edge C: R_own=0.60, R_inh=0.08, R_tot=0.63, I_causal=0.90, PWRS=0.57
    assert round(eC.own_risk, 2) == 0.60
    assert round(eC.inherited_risk, 2) == 0.08
    assert round(eC.total_risk, 2) == 0.63
    assert round(eC.pwrs, 2) == 0.57

    # Edge D: R_own=0.05, R_inh=0.38, R_tot=0.41, I_causal=0.40, PWRS=0.16
    assert round(eD.own_risk, 2) == 0.05
    assert round(eD.inherited_risk, 2) == 0.38
    assert round(eD.total_risk, 2) == 0.41
    assert round(eD.pwrs, 2) == 0.16

    # C is flagged because 0.57 exceeds tau = 0.55
    assert summary.bottleneck_edge == eC
    assert summary.path_risk > 0.55
    assert summary.is_acceptable is False


def test_epistemic_abstention_decision():
    """Validates that when no alternative path is under threshold tau=0.55, system cleanly abstains."""
    kg = KnowledgeGraphManager("AbstainKG")
    # Only one path exists, and it is severely corrupted
    e_bad = Edge(source="A", relation="r", target="B", base_reliability=0.1, causal_importance=1.0)
    kg.add_edge(e_bad)

    primary_path = ReasoningPath(edges=[e_bad])
    scorer = PWRSScorer(risk_threshold=0.55)
    finder = PathFinder(kg, max_hops=2)
    repair_engine = GraphRepairEngine(kg, scorer, finder, max_retries=2)

    result = repair_engine.repair_path("Where is B?", primary_path)
    assert result.status == "ABSTAINED"
    assert primary_path.status == "ABSTAINED"

