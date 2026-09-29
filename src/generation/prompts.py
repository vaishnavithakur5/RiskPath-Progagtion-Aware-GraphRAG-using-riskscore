"""Grounded prompt templates for multi-hop graph generation and counterfactual ablation."""

from typing import List
from src.graph.schema import Edge, ReasoningPath


def format_triples_for_prompt(edges: List[Edge]) -> str:
    """Formats edges into clear, enumerated factual statements."""
    if not edges:
        return "No supporting graph facts available."
    lines = []
    for i, e in enumerate(edges, 1):
        s = e.source.replace("_", " ")
        r = e.relation.replace("_", " ")
        t = e.target.replace("_", " ")
        lines.append(f"{i}. {s} -> {r} -> {t}")
    return "\n".join(lines)


def build_qa_prompt(query: str, path: ReasoningPath) -> str:
    """Builds a strictly grounded prompt instructing the model to answer based ONLY on facts."""
    facts = format_triples_for_prompt(path.edges)
    prompt = (
        f"You are a factual reasoning assistant. Answer the question using ONLY the provided facts.\n"
        f"If the facts do not contain sufficient information, respond with 'Information not available'.\n"
        f"Keep the answer brief and directly name the entity.\n\n"
        f"Facts:\n{facts}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt
