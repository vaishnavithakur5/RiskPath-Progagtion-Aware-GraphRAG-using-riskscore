"""Streamlit interactive dashboard for the RiskPath M.Tech thesis viva defense."""

import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extraction.benchmark_loader import BenchmarkDataset
from src.generation.client import get_llm_client
from src.corruption.engine import CorruptionEngine
from src.baselines.pipeline import create_pipeline
from src.evaluation.metrics import compute_exact_match, compute_token_f1


# Page configuration
st.set_page_config(
    page_title="RiskPath: Propagation-Aware GraphRAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ RiskPath: Propagation-Aware Risk Retrieval for GraphRAG")
st.caption(
    "M.Tech AI/ML Thesis Prototype — Robust Knowledge Graph Retrieval & Counterfactual Risk Weighting"
)


@st.cache_resource
def load_base_system():
    """Initializes and caches the benchmark knowledge graph and queries."""
    ds = BenchmarkDataset.create_curated_multihop_benchmark(num_distractor_clusters=10)
    return ds


ds = load_base_system()

# Sidebar: Environment & Adversarial Controls
st.sidebar.header("⚙️ System & Inference Settings")
backend_choice = st.sidebar.selectbox("LLM Backend", ["mock", "ollama"], index=0)
model_name = st.sidebar.text_input("Model Name", value="qwen2.5:1.5b")
risk_threshold = st.sidebar.slider("PWRS Risk Threshold (τ)", 0.10, 0.90, 0.45, 0.05)
gamma_exp = st.sidebar.slider("Causal Sensitivity Exponent (γ)", 0.5, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("🧪 Adversarial Protocol (IARP)")
inject_noise_active = st.sidebar.checkbox("Enable Graph Corruption", value=False)

noise_operator = st.sidebar.selectbox(
    "Corruption Operator",
    ["entity_swap", "relation_swap", "contradictory_edge", "adversarial_bridge", "provenance_strip"],
)
noise_rate = st.sidebar.slider("Noise Rate (η)", 0.05, 0.40, 0.15, 0.05)

# Working KG setup based on noise settings
corr_engine = CorruptionEngine(random_seed=42)
if inject_noise_active:
    working_kg, audit_records = corr_engine.inject_noise(
        ds.kg, operator_name=noise_operator, noise_rate=noise_rate
    )
    st.sidebar.warning(f"⚠️ Injected {len(audit_records)} corrupted edges into Knowledge Graph.")
else:
    working_kg = ds.kg
    audit_records = []

# Main Layout: 1. Query Selection
st.subheader("1. Multi-Hop Query Selection")
query_options = {q.question: q for q in ds.queries}
selected_q_text = st.selectbox(
    "Choose a verified multi-hop benchmark query (or observe preset):",
    list(query_options.keys()),
    index=0,
)
selected_query = query_options[selected_q_text]

col_q1, col_q2, col_q3 = st.columns(3)
col_q1.info(f"**Source Entity:** `{selected_query.source_entity}`")
col_q2.info(f"**Ground Truth Target:** `{selected_query.ground_truth_answer}`")
col_q3.info(f"**Reasoning Hops:** `{selected_query.hop_count} hops`")

# Execution trigger
col_btn1, col_btn2 = st.columns([1, 4])
run_analysis = col_btn1.button("🚀 Execute Retrieval & Audit", type="primary")

if run_analysis or "last_run" in st.session_state:
    st.session_state["last_run"] = True
    llm = get_llm_client(backend=backend_choice, model_name=model_name)
    config = {
        "risk_threshold": risk_threshold,
        "gamma_sensitivity": gamma_exp,
        "relevance_risk_lambda": 0.70,
    }

    # Execute both pipelines for side-by-side viva comparison
    std_pipeline = create_pipeline("standard_graphrag", working_kg, llm, config)
    risk_pipeline = create_pipeline("riskpath_full", working_kg, llm, config)

    std_out = std_pipeline.execute(
        selected_query.question,
        source_entity=selected_query.source_entity,
        target_entity=selected_query.target_entity,
    )
    risk_out = risk_pipeline.execute(
        selected_query.question,
        source_entity=selected_query.source_entity,
        target_entity=selected_query.target_entity,
    )

    st.markdown("---")
    st.subheader("2. Comparative Baseline vs. RiskPath Results")

    col_std, col_risk = st.columns(2)

    with col_std:
        st.markdown("### 🔴 Standard GraphRAG (Relevance Only)")
        st.write(f"**Selected Path:** `{std_out.selected_path.linearize()}`")
        st.write(f"**Generated Answer:** `{std_out.generated_answer}`")

        std_em = compute_exact_match(std_out.generated_answer, selected_query.ground_truth_answer)
        std_f1 = compute_token_f1(std_out.generated_answer, selected_query.ground_truth_answer)
        std_has_corrupt = any(e.is_corrupted for e in std_out.selected_path.edges)

        m1, m2, m3 = st.columns(3)
        m1.metric("Exact Match", f"{std_em:.1f}")
        m2.metric("Token F1", f"{std_f1:.2f}")
        m3.metric("Corrupted Fact Used?", "YES ⚠️" if std_has_corrupt else "NO ✅")

    with col_risk:
        st.markdown("### 🟢 Proposed RiskPath (PWRS + Recovery)")
        st.write(f"**Selected Path:** `{risk_out.selected_path.linearize()}`")
        st.write(f"**Generated Answer:** `{risk_out.generated_answer}`")

        risk_em = compute_exact_match(risk_out.generated_answer, selected_query.ground_truth_answer)
        risk_f1 = compute_token_f1(risk_out.generated_answer, selected_query.ground_truth_answer)
        risk_has_corrupt = any(e.is_corrupted for e in risk_out.selected_path.edges)

        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("Exact Match", f"{risk_em:.1f}")
        rm2.metric("Token F1", f"{risk_f1:.2f}")
        rm3.metric("Corrupted Fact Used?", "YES ⚠️" if risk_has_corrupt else "NO ✅")

    # Section 3: Edge Risk Assessment Table
    st.markdown("---")
    st.subheader("3. Edge-Level PWRS Risk & Causal Influence Breakdown")

    if risk_out.selected_path.edges:
        table_rows = []
        for e in risk_out.selected_path.edges:
            table_rows.append(
                {
                    "Triple": e.to_triple_string(),
                    "Base Reliability (Rel)": f"{e.base_reliability:.2f}",
                    "Base Risk (1 - Rel)": f"{e.base_risk:.2f}",
                    "Causal Importance (I_prop)": f"{e.causal_importance:.2f}",
                    "PWRS Score": f"{e.pwrs:.4f}",
                    "Status": "FLAGGED ⚠️" if e.pwrs > risk_threshold else "SAFE ✅",
                    "Is Corrupted?": "YES" if e.is_corrupted else "NO",
                }
            )
        st.table(pd.DataFrame(table_rows))
    else:
        st.info("No edges in path.")

    # Section 4: Explainability Trust Narrative
    st.markdown("---")
    st.subheader("4. Trust Narrative (Auditable Explainability)")
    st.markdown(risk_out.trust_narrative)

    # Section 5: Adversarial Audit Log
    if audit_records:
        with st.expander("🔍 View Active Injected Corruptions (IARP Trace)"):
            st.dataframe(pd.DataFrame([r.to_dict() for r in audit_records]))

st.sidebar.markdown("---")
st.sidebar.caption("RiskPath M.Tech Prototype • Local CPU Mode • NetworkX Core")
