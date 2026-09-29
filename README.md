# RiskPath: Propagation-Aware Risk Retrieval for GraphRAG

**M.Tech Computer Science & Engineering (AI/ML) Research Prototype**  
*Thesis Title:* RiskPath: Propagation-Aware Risk Retrieval for GraphRAG  
*Target Environment:* Commodity Student Laptop (Intel i3 CPU, 4–16 GB RAM, No Dedicated GPU, Windows 11)

---

## 📌 Project Overview
Graph Retrieval-Augmented Generation (GraphRAG) enriches LLMs by extracting multi-hop reasoning paths from Knowledge Graphs (KGs). However, unverified or noisy relations can mislead generation. 

Traditional denoising filters facts based purely on **static reliability** (e.g. extraction confidence). **RiskPath** introduces **Propagation-Weighted Risk Scoring (PWRS)** grounded in statistical decision theory (expected loss):

$$\text{PWRS}(e \mid q) = R_{\text{base}}(e) \cdot \left[ I_{\text{prop}}(e \mid q) \right]^\gamma$$

Where:
- $R_{\text{base}}(e) = 1.0 - \text{Rel}(e)$ represents the intrinsic epistemic uncertainty of edge $e$ derived from provenance citations, occurrence frequency, and graph topology.
- $I_{\text{prop}}(e \mid q) \in [0.0, 1.0]$ represents the downstream causal influence of $e$ on the generated answer, measured via bounded counterfactual leave-one-out ablation.
- $\gamma \ge 0$ is the causal sensitivity exponent.

---

## 🚀 Key Features
1. **Decision-Theoretic Path Retrieval:** Prioritizes paths maximizing factual relevance while penalizing causal risk bottlenecks.
2. **Interactive Adversarial Robustness Protocol (IARP):** 7 deterministic corruption operators (`entity_swap`, `relation_swap`, `contradictory_edge`, `adversarial_bridge`, `provenance_strip`, etc.) to simulate realistic noise regimes (5%, 10%, 20%).
3. **Dynamic Path Repair:** Automatically flags high-risk bottleneck edges, isolates them, and reroutes through clean alternative paths.
4. **Explainable Trust Narrative:** Generates natural-language audits explaining why specific facts were trusted or rejected.
5. **Laptop-Optimized Architecture:** Entire pipeline runs in pure Python with NetworkX and local SLMs (or offline mock engine) with a RAM footprint $<1.5\text{ GB}$.

---

## 🛠️ Project Structure
```
riskpath/
├── config/
│   └── default_config.yaml         # System parameters, weights, and thresholds
├── src/
│   ├── graph/                      # NetworkX manager, schemas, persistence
│   ├── extraction/                 # Entity linker and benchmark dataset loader
│   ├── reliability/                # Base reliability scorer (multi-factor signals)
│   ├── retrieval/                  # Path finder and candidate pool ranking
│   ├── propagation/                # Causal leave-one-out ablation engine
│   ├── scoring/                    # PWRS calculator and path utility
│   ├── corruption/                 # IARP adversarial engine & 7 operators
│   ├── repair/                     # Dynamic rerouting and evidence recovery
│   ├── generation/                 # LLM client (Ollama + Mock) and Trust Narrative
│   ├── baselines/                  # Standard GraphRAG, Shortest Path, Reliability-Only
│   └── evaluation/                 # Metrics (EM, F1, CEAR, SRR) and matrix runner
├── experiments/
│   └── run_benchmark.py            # CLI batch experiment runner
├── ui/
│   └── app.py                      # Interactive Streamlit viva defense dashboard
├── tests/                          # 21 unit & integration tests (100% pass)
└── requirements.txt
```

---

## 💻 Quick Start & Reproduction

### 1. Run Automated Test Suite
Verify that all 21 unit and integration tests pass:
```bash
python -m pytest tests/ -v
```

### 2. Run Comparative Baseline Benchmark (CLI)
Execute an offline matrix evaluation across 5 baselines and noise rates:
```bash
python experiments/run_benchmark.py --queries 10 --backend mock
```

To run with a real local LLM via Ollama:
```bash
# Ensure Ollama is running with Qwen2.5-1.5B (or Llama-3.2-3B)
ollama run qwen2.5:1.5b
python experiments/run_benchmark.py --queries 10 --backend ollama --model qwen2.5:1.5b
```

### 3. Launch Interactive Viva Defense Dashboard (Streamlit)
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Baselines Evaluated
- **B1 (Standard GraphRAG):** Relevance-only path selection without risk modeling.
- **B2 (Shortest-Path RAG):** Hop-minimizing Dijkstra selection.
- **B3 (Reliability-Only RAG):** Static edge reliability filtering without causal weighting.
- **B4 (RiskPath Ablated):** RiskPath without dynamic causal propagation ($I_{\text{prop}}=1.0$).
- **B5 (RiskPath Full):** Joint optimization of relevance, causal PWRS, and dynamic repair.

---

## 🎓 Academic Integrity Notice
This codebase is developed for an M.Tech CSE (AI/ML) thesis. All benchmark results, noise injections, and metric evaluations are deterministic, reproducible, and verifiable without external closed APIs.
