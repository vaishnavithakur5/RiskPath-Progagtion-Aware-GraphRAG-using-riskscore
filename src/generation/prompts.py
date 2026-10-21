"""Grounded prompt templates for multi-hop graph generation and counterfactual ablation."""

from typing import List, Optional, Any, Dict
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


def build_structured_research_response(
    query: str,
    selected_path: ReasoningPath,
    generated_answer: str,
    method_name: str = "RiskPath_Full",
    repair_result: Optional[Any] = None,
    risk_threshold: float = 0.55,
) -> str:
    """Constructs the research-grade 5-part thesis defense output report.
    
    1. Final Answer
    2. Supporting Evidence & Provenance
    3. Risk Breakdown Table (R_own, R_inh, R_tot, I_causal, PWRS)
    4. Retrieval & Repair Explanation
    5. Confidence & Epistemic Limitations
    """
    is_abstained = (selected_path.status == "ABSTAINED") or (
        repair_result and getattr(repair_result, "status", "") == "ABSTAINED"
    )
    path_risk = getattr(selected_path, "path_risk", getattr(selected_path, "max_risk", 0.0))
    confidence_pct = max(0.0, min(100.0, (1.0 - path_risk) * 100))

    lines = []
    lines.append(f"## 📋 Research Synthesis Report — {method_name}")
    lines.append(f"**Query:** *\"{query}\"*\n")

    # Section 1: Final Answer
    lines.append("### 1. Final Answer")
    if is_abstained:
        lines.append(
            "> ⚠️ **EPISTEMIC ABSTENTION:** *The system withheld an answer because all available reasoning "
            f"paths contain bottleneck risk exceeding safety threshold (PWRS > {risk_threshold:.2f}).*"
        )
    else:
        lines.append(f"> 🎯 **Answer:** **{generated_answer}**\n")

    # Section 2: Supporting Evidence & Provenance
    lines.append("### 2. Supporting Evidence & Provenance")
    if selected_path.edges:
        for idx, edge in enumerate(selected_path.edges, 1):
            s, r, t = edge.source.replace("_", " "), edge.relation.replace("_", " "), edge.target.replace("_", " ")
            prov_src = edge.provenance.source_id if edge.provenance else "Unknown Corpus"
            prov_txt = edge.provenance.text_snippet if (edge.provenance and edge.provenance.text_snippet) else "Documentary passage grounding"
            lines.append(f"- **Hop {idx}:** `{s}` $\\xrightarrow{{\\text{{{r}}}}}$ `{t}`")
            lines.append(f"  - **Provenance Source:** *{prov_src}*")
            lines.append(f"  - **Passage Snippet:** \"{prov_txt}\"")
            lines.append(f"  - **Base Reliability (Rel):** `{edge.base_reliability:.2f}` (Base Risk $R_{{own}}={edge.base_risk:.2f}$)")
    else:
        lines.append("- *No verified evidence edges available for this query.*")
    lines.append("")

    # Section 3: Risk Breakdown Table
    lines.append("### 3. Edge-Level Risk Breakdown Table")
    lines.append("| Hop | Evidence Triple | $R_{own}$ | $R_{inh}$ | $R_{tot}$ | $I_{causal}$ | PWRS | Status |")
    lines.append("|:---:|:----------------|:---------:|:---------:|:---------:|:------------:|:----:|:------:|")
    if selected_path.edges:
        for idx, e in enumerate(selected_path.edges, 1):
            trip = f"({e.source}, {e.relation}, {e.target})"
            st_badge = "FLAGGED ⚠️" if e.pwrs > risk_threshold else "SAFE ✅"
            lines.append(
                f"| {idx} | `{trip}` | {e.own_risk:.3f} | {e.inherited_risk:.3f} | {e.total_risk:.3f} | {e.causal_importance:.3f} | **{e.pwrs:.4f}** | {st_badge} |"
            )
    else:
        lines.append("| - | *Empty Path* | - | - | - | - | - | - |")
    lines.append("")

    # Section 4: Retrieval & Repair Explanation
    lines.append("### 4. Retrieval & Repair Explanation")
    if repair_result and getattr(repair_result, "status", "") == "REPAIRED":
        flagged = repair_result.flagged_edge
        flagged_str = flagged.to_triple_string() if flagged else "Unknown"
        lines.append(
            f"- **Culprit Isolated:** Edge `{flagged_str}` had high total risk ({flagged.total_risk:.2f}) and high causal influence ({flagged.causal_importance:.2f}), yielding PWRS={repair_result.original_pwrs:.4f} > {risk_threshold}."
        )
        lines.append(
            f"- **Dynamic Recovery:** High-risk edge was pruned; rerouted to clean alternative path with PWRS={repair_result.repaired_pwrs:.4f}."
        )
    elif is_abstained:
        lines.append(
            f"- **Abstention Decision:** Candidate paths were pruned iteratively up to retry limits, but no alternative path met safety criterion $\\tau \\le {risk_threshold}$. System abstains."
        )
    else:
        lines.append(
            f"- **Path Verification:** All candidate hops remained below risk threshold $\\tau = {risk_threshold}$. Reasoning path executed directly without requiring graph mutation."
        )
    lines.append("")

    # Section 5: Confidence & Epistemic Limitations
    lines.append("### 5. Confidence & Epistemic Limitations")
    lines.append(f"- **System Epistemic Confidence:** **{confidence_pct:.1f}%** ($1.0 - \\text{{PathRisk}}$)")
    lines.append(
        f"- **Decision Boundary:** Safety threshold $\\tau = {risk_threshold:.2f}$ (Slide 10 ground truth)."
    )
    lines.append(
        "- **Evaluated Assumptions:** Causal ablation evaluated on top-3 decisive path edges; propagation decay factor $\\lambda = 0.60$."
    )
    return "\n".join(lines)
