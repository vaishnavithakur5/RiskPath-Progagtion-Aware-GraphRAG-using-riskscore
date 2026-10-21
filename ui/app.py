"""Streamlit interactive research dashboard for RiskPath M.Tech thesis viva defense.

Supports:
  1. Official Research Benchmark: 2WikiMultihopQA (200 multi-hop queries, 1,600+ entities).
  2. Curated Illustrative Demo: Cinema & Science multi-hop graph (28 queries, 86 entities).
  3. Four Thesis Review 2 Protocol Scenarios (Clean KG, Corrupted Evidence, Alternative-Path Recovery, Epistemic Abstention).
  4. All 6 Slide 13 comparative baselines.
  5. Full 5-part research synthesis card for thesis defense.
"""

from __future__ import annotations
import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extraction.benchmark_loader import BenchmarkDataset, BenchmarkQuery
from src.generation.client import get_llm_client
from src.generation.prompts import build_structured_research_response
from src.corruption.engine import CorruptionEngine
from src.baselines.pipeline import create_pipeline
from src.evaluation.metrics import compute_exact_match, compute_token_f1
from src.graph.schema import Edge, ReasoningPath


# Page configuration
st.set_page_config(
    page_title="RiskPath: Propagation-Aware GraphRAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ RiskPath: Propagation-Aware Risk Retrieval for GraphRAG")
st.caption(
    "M.Tech AI/ML Thesis Prototype — Robust Knowledge Graph Retrieval, Sequential Risk Propagation & Counterfactual Ablation"
)


# Sidebar: Dataset Benchmark Selection
st.sidebar.header("📊 Dataset Benchmark")
dataset_mode = st.sidebar.radio(
    "Select Benchmark Mode",
    ["Official 2WikiMultihopQA (Research Benchmark)", "Curated Cinema & Science (Illustrative Demo)"],
    index=0,
)

is_official_2wiki = "2WikiMultihopQA" in dataset_mode


@st.cache_resource
def load_cached_benchmark(mode: str) -> BenchmarkDataset:
    if "2WikiMultihopQA" in mode:
        return BenchmarkDataset.load_benchmark("2wiki")
    else:
        return BenchmarkDataset.load_benchmark("demo")


ds = load_cached_benchmark(dataset_mode)

if is_official_2wiki:
    st.sidebar.success(
        f"🏆 **Research Benchmark:** 2WikiMultihopQA\n\n"
        f"• Queries: {len(ds.queries)}\n"
        f"• Entities: {ds.kg.node_count}\n"
        f"• Verified Triples: {ds.kg.edge_count}"
    )
else:
    st.sidebar.info(
        f"💡 **Illustrative Demo:** Cinema & Science\n\n"
        f"• Queries: {len(ds.queries)}\n"
        f"• Entities: {ds.kg.node_count}\n"
        f"• Verified Triples: {ds.kg.edge_count}\n\n"
        "*(Convenient for instant viva demonstration)*"
    )

# Sidebar: Environment & Theoretical Parameters (Slide 10)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Theoretical Parameters (Slide 10)")
backend_choice = st.sidebar.selectbox("LLM Backend", ["mock", "ollama"], index=0)
model_name = st.sidebar.text_input("Model Name", value="qwen2.5:1.5b")

risk_threshold = st.sidebar.slider(
    "PWRS Safety Threshold (τ)",
    min_value=0.10,
    max_value=0.90,
    value=0.55,
    step=0.05,
    help="Slide 10 ground truth τ = 0.55 defines the decision boundary for path rerouting / epistemic abstention.",
)
lambda_propagation = st.sidebar.slider(
    "Propagation Decay (λ)",
    min_value=0.10,
    max_value=1.00,
    value=0.60,
    step=0.05,
    help="Slide 10 ground truth λ = 0.60 controls inherited risk attenuation across connected hops.",
)
gamma_exp = st.sidebar.slider(
    "Causal Sensitivity Exponent (γ)",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Slide 10 ground truth γ = 1.0 weights causal importance in PWRS(f) = R_tot(f) * [I_causal(f)]^γ.",
)

# Sidebar: Comparative Baseline Selector (Slide 13)
st.sidebar.markdown("---")
st.sidebar.header("⚖️ Baseline System (Slide 13)")
baseline_choice = st.sidebar.selectbox(
    "Compare RiskPath Against",
    [
        "standard_graphrag",
        "shortest_path",
        "reliability_only",
        "causal_rag",
        "propagation_rag",
    ],
    format_func=lambda s: {
        "standard_graphrag": "B1: Standard GraphRAG (Relevance Only)",
        "shortest_path": "B2: Shortest-Path RAG (Min Hops)",
        "reliability_only": "B3: Reliability-Only RAG (Static Own Risk)",
        "causal_rag": "B4: Causal-RAG (Ablation, No Propagation)",
        "propagation_rag": "B5: Propagation-RAG (Propagation, No Causal)",
    }.get(s, s),
    index=0,
)

# Sidebar: Adversarial Protocol (IARP)
st.sidebar.markdown("---")
st.sidebar.header("🧪 Adversarial Protocol (IARP)")
inject_noise_active = st.sidebar.checkbox("Inject Adversarial Graph Noise", value=False)
noise_operator = st.sidebar.selectbox(
    "Corruption Operator",
    ["entity_swap", "relation_swap", "contradictory_edge", "adversarial_bridge", "provenance_strip"],
)
noise_rate = st.sidebar.slider("Corruption Rate (η)", 0.05, 0.40, 0.15, 0.05)


# Main Layout: 1. Demonstration Protocol Scenario
st.subheader("1. Demonstration Protocol Scenario")
scenario = st.selectbox(
    "Choose a viva presentation scenario:",
    [
        "Scenario 1: Clean Knowledge Graph (Ground-Truth Evidence Verification)",
        "Scenario 2: Corrupted Evidence (Adversarial Detection & PWRS Flagging)",
        "Scenario 3: Dynamic Alternative-Path Recovery (Bottleneck Isolation & Rerouting)",
        "Scenario 4: Epistemic Abstention (Unrecoverable Risk & Hallucination Prevention)",
        "Custom: Manual Benchmark Query Selection",
    ],
    index=0,
)

# Working KG setup based on noise settings and scenario
corr_engine = CorruptionEngine(random_seed=42)
working_kg = ds.kg.copy()
audit_records = []

if inject_noise_active:
    working_kg, audit_records = corr_engine.inject_noise(
        working_kg, operator_name=noise_operator, noise_rate=noise_rate
    )

# Filter or select query based on scenario
query_dict = {q.question: q for q in ds.queries}

if "Scenario 1" in scenario:
    st.info("📌 **Scenario 1:** Knowledge graph is clean. Verifies that both systems answer accurately while RiskPath computes auditable risk propagation and provenance.")
    selected_query = ds.queries[0]
elif "Scenario 2" in scenario:
    st.warning("⚠️ **Scenario 2:** High-relevance candidate evidence is corrupted. Standard GraphRAG is deceived by lexical relevance, whereas RiskPath isolates high PWRS.")
    selected_query = ds.queries[0]
    # Ensure working_kg has a corrupt edge on the top path
    source_node = selected_query.source_entity
    out_edges = working_kg.get_outgoing_edges(source_node)
    if out_edges:
        target_edge = out_edges[0]
        working_kg.remove_edge_by_id(target_edge.id)
        corrupt_e = Edge(
            source=target_edge.source,
            relation=target_edge.relation,
            target="Corrupted_Entity_Fake",
            base_reliability=0.10,
            is_corrupted=True,
            corruption_type="entity_swap",
        )
        working_kg.add_edge(corrupt_e)
        audit_records.append({"scenario": "Scenario 2", "edge_id": corrupt_e.id, "corrupted": True})
elif "Scenario 3" in scenario:
    st.info("🔄 **Scenario 3:** Primary candidate path is poisoned, but an alternative clean multi-hop path exists. Demonstrates dynamic rerouting and safe answer recovery.")
    selected_query = ds.queries[0]
    source_node = selected_query.source_entity
    out_edges = working_kg.get_outgoing_edges(source_node)
    if out_edges:
        target_edge = out_edges[0]
        # Introduce a high-relevance corrupted shortcut
        corrupt_shortcut = Edge(
            source=source_node,
            relation=target_edge.relation,
            target="Corrupted_Shortcut_Entity",
            base_reliability=0.15,
            is_corrupted=True,
            corruption_type="adversarial_bridge",
        )
        working_kg.add_edge(corrupt_shortcut)
        audit_records.append({"scenario": "Scenario 3", "edge_id": corrupt_shortcut.id, "corrupted": True})
elif "Scenario 4" in scenario:
    st.error("🛑 **Scenario 4:** All candidate reasoning paths to target are compromised (PWRS > 0.55). RiskPath cleanly triggers epistemic abstention to prevent hallucination.")
    selected_query = ds.queries[0]
    # Corrupt all edges leaving source node so no safe path exists
    source_node = selected_query.source_entity
    for out_e in working_kg.get_outgoing_edges(source_node):
        out_e.base_reliability = 0.05
        out_e.base_risk = 0.95
        out_e.is_corrupted = True
else:
    # Custom selector
    selected_q_text = st.selectbox(
        "Select a verified multi-hop query:",
        list(query_dict.keys()),
        index=0,
    )
    selected_query = query_dict[selected_q_text]

# Query metadata card
col_q1, col_q2, col_q3 = st.columns(3)
col_q1.info(f"**Source Entity:** `{selected_query.source_entity}`")
col_q2.info(f"**Ground Truth Target:** `{selected_query.ground_truth_answer}`")
col_q3.info(f"**Reasoning Hops:** `{selected_query.hop_count} hops`")

st.markdown(f"**Active Question:** *\"{selected_query.question}\"*")

# Execution trigger
col_btn1, col_btn2 = st.columns([1, 4])
run_analysis = col_btn1.button("🚀 Execute Retrieval & Audit", type="primary")

if run_analysis or "last_run" in st.session_state:
    st.session_state["last_run"] = True
    llm = get_llm_client(backend=backend_choice, model_name=model_name)
    config = {
        "risk_threshold": risk_threshold,
        "gamma_sensitivity": gamma_exp,
        "lambda_propagation": lambda_propagation,
        "relevance_risk_lambda": 0.70,
    }

    # Execute chosen Baseline and RiskPath Full
    baseline_pipe = create_pipeline(baseline_choice, working_kg, llm, config)
    risk_pipe = create_pipeline("riskpath_full", working_kg, llm, config)

    base_out = baseline_pipe.execute(
        selected_query.question,
        source_entity=selected_query.source_entity,
        target_entity=selected_query.target_entity,
    )
    risk_out = risk_pipe.execute(
        selected_query.question,
        source_entity=selected_query.source_entity,
        target_entity=selected_query.target_entity,
    )

    st.markdown("---")
    st.subheader("2. Comparative Baseline vs. RiskPath Results")

    col_base, col_risk = st.columns(2)

    with col_base:
        st.markdown(f"### 🔴 Baseline: `{base_out.method_name}`")
        st.write(f"**Selected Reasoning Path:** `{base_out.selected_path.linearize()}`")
        st.write(f"**Generated Answer:** `{base_out.generated_answer}`")

        base_em = compute_exact_match(base_out.generated_answer, selected_query.ground_truth_answer)
        base_f1 = compute_token_f1(base_out.generated_answer, selected_query.ground_truth_answer)
        base_has_corrupt = any(e.is_corrupted for e in base_out.selected_path.edges)

        bm1, bm2, bm3 = st.columns(3)
        bm1.metric("Exact Match", f"{base_em:.1f}")
        bm2.metric("Token F1", f"{base_f1:.2f}")
        bm3.metric("Corrupted Fact Used?", "YES ⚠️" if base_has_corrupt else "NO ✅")

    with col_risk:
        st.markdown("### 🟢 Proposed: `RiskPath_Full`")
        st.write(f"**Selected Reasoning Path:** `{risk_out.selected_path.linearize()}`")
        st.write(f"**Generated Answer:** `{risk_out.generated_answer}`")

        risk_em = compute_exact_match(risk_out.generated_answer, selected_query.ground_truth_answer)
        risk_f1 = compute_token_f1(risk_out.generated_answer, selected_query.ground_truth_answer)
        risk_has_corrupt = any(e.is_corrupted for e in risk_out.selected_path.edges)

        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("Exact Match", f"{risk_em:.1f}")
        rm2.metric("Token F1", f"{risk_f1:.2f}")
        rm3.metric("Corrupted Fact Used?", "YES ⚠️" if risk_has_corrupt else "NO ✅")

    # Section 3: 5-Part Research Synthesis Card
    st.markdown("---")
    st.subheader("3. 📋 5-Part Research Synthesis Card (Thesis Viva Format)")

    report_markdown = build_structured_research_response(
        query=selected_query.question,
        selected_path=risk_out.selected_path,
        generated_answer=risk_out.generated_answer,
        method_name="RiskPath_Full",
        repair_result=risk_out.repair_result,
        risk_threshold=risk_threshold,
    )
    st.markdown(report_markdown)

    # Section 4: Mathematical Risk Propagation Trace (Slide 10 Decomposition)
    st.markdown("---")
    st.subheader("4. 📐 Mathematical Risk Propagation Trace (Slide 10)")
    st.latex(r"R_{own}(f) = 1.0 - \text{Rel}(f)")
    st.latex(r"R_{inh}(v) = \lambda \cdot R_{tot}(u), \quad R_{tot}(v) = 1 - (1 - R_{own}(v))(1 - R_{inh}(v))")
    st.latex(r"\text{PWRS}(f) = R_{tot}(f) \times [I_{causal}(f)]^\gamma, \quad \text{PathRisk}(\mathcal{P}) = \max_{f \in \mathcal{P}} \text{PWRS}(f)")

    if risk_out.selected_path.edges:
        calc_rows = []
        for idx, e in enumerate(risk_out.selected_path.edges, 1):
            calc_rows.append(
                {
                    "Hop": idx,
                    "Triple": e.to_triple_string(),
                    "Rel(f)": f"{e.base_reliability:.3f}",
                    "R_own (1-Rel)": f"{e.own_risk:.3f}",
                    "R_inh": f"{e.inherited_risk:.3f}",
                    "R_tot": f"{e.total_risk:.3f}",
                    "I_causal": f"{e.causal_importance:.3f}",
                    "PWRS": f"{e.pwrs:.4f}",
                    "Status": "FLAGGED ⚠️" if e.pwrs > risk_threshold else "SAFE ✅",
                }
            )
        st.dataframe(pd.DataFrame(calc_rows), use_container_width=True)

    # Section 5: Trust Narrative & Adversarial Trace
    st.markdown("---")
    st.subheader("5. Auditable Trust Narrative & Repair Trace")
    st.markdown(risk_out.trust_narrative)

    if audit_records:
        with st.expander("🔍 View Active Injected Corruptions (IARP Trace)"):
            st.dataframe(pd.DataFrame(audit_records))

st.sidebar.markdown("---")
st.sidebar.caption("RiskPath M.Tech Prototype • Local CPU Mode • NetworkX Core")
