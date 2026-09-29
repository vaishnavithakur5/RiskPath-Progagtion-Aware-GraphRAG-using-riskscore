"""Unit tests for PathFinder and CandidatePool."""

import pytest
from src.graph.schema import Edge
from src.graph.manager import KnowledgeGraphManager
from src.retrieval.path_finder import PathFinder, CandidatePool


def test_path_finder_and_candidate_pool():
    kg = KnowledgeGraphManager("RetrievalKG")

    # Path 1 (Target): Inception -> director -> Christopher_Nolan -> educated_at -> UCL
    e1 = Edge(source="Inception", relation="director", target="Christopher_Nolan", base_reliability=0.9)
    e2 = Edge(source="Christopher_Nolan", relation="educated_at", target="UCL", base_reliability=0.9)

    # Path 2 (Alternative): Inception -> producer -> Emma_Thomas -> educated_at -> UCL
    e3 = Edge(source="Inception", relation="producer", target="Emma_Thomas", base_reliability=0.8)
    e4 = Edge(source="Emma_Thomas", relation="educated_at", target="UCL", base_reliability=0.8)

    kg.add_edge(e1)
    kg.add_edge(e2)
    kg.add_edge(e3)
    kg.add_edge(e4)

    finder = PathFinder(kg, max_hops=3)
    query = "Where was the director of Inception educated?"
    pool = finder.find_candidate_paths(
        source_entity="Inception", query=query, target_entity="UCL", top_k=5
    )

    assert len(pool) >= 2
    # Verify director path scores higher than producer path due to query token alignment
    top = pool.top_path
    assert top is not None
    assert "director" in top.linearize()
    assert top.semantic_score >= pool.paths[1].semantic_score

    # Test CandidatePool composite sorting
    pool.sort_by_composite_score(risk_lambda=0.5)
    assert len(pool.paths) >= 2
