"""Unit tests for LLM generation client and CausalAblator."""

import pytest
from src.graph.schema import Edge, ReasoningPath
from src.generation.client import MockLocalClient, get_llm_client
from src.generation.prompts import build_qa_prompt
from src.propagation.ablator import CausalAblator


def test_mock_llm_chain_resolution():
    client = MockLocalClient()

    # 2-hop chain
    e1 = Edge(source="Inception", relation="director", target="Christopher_Nolan")
    e2 = Edge(source="Christopher_Nolan", relation="educated_at", target="University_College_London")
    path = ReasoningPath(edges=[e1, e2])

    query = "Where was the director of Inception educated?"
    prompt = build_qa_prompt(query, path)
    answer = client.generate(prompt)

    assert "University College London" in answer


def test_causal_ablator_bridge_sensitivity():
    client = MockLocalClient()
    ablator = CausalAblator(llm_client=client, max_ablation_edges=2)

    e1 = Edge(source="Inception", relation="director", target="Christopher_Nolan", base_reliability=0.9)
    e2 = Edge(source="Christopher_Nolan", relation="educated_at", target="University_College_London", base_reliability=0.9)
    path = ReasoningPath(edges=[e1, e2])

    query = "Where was the director of Inception educated?"
    y_0, records = ablator.evaluate_path_causality(query, path)

    assert "University College London" in y_0
    assert len(records) == 2

    # When e2 is ablated, the education destination is missing -> answer changes -> high causal importance
    rec_e2 = next(r for r in records if r.edge_id == e2.id)
    assert rec_e2.causal_importance > 0.60
    assert rec_e2.ablated_answer != rec_e2.baseline_answer
