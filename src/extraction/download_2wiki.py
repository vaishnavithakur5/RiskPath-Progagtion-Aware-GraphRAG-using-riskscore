"""CLI utility to download and build the official 2WikiMultihopQA benchmark dataset."""

from __future__ import annotations
import os
import sys
import argparse
import urllib.request
import pandas as pd
import json

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DEV_PARQUET_URL = "https://huggingface.co/datasets/somebody-had-to-do-it/2WikiMultihopQA/resolve/main/dev.parquet"


def download_2wiki_raw(dest_path: str = "data/raw/2wikimultihopqa_dev.parquet", force: bool = False) -> str:
    """Downloads official 2WikiMultihopQA validation split in parquet format (~30 MB)."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if os.path.exists(dest_path) and not force:
        print(f"[Dataset] Raw 2WikiMultihopQA file already exists at '{dest_path}'.")
        return dest_path

    print(f"[Dataset] Downloading 2WikiMultihopQA dev split from {DEV_PARQUET_URL}...")
    urllib.request.urlretrieve(DEV_PARQUET_URL, dest_path)
    print(f"[Dataset] Download completed successfully: '{dest_path}' ({os.path.getsize(dest_path)} bytes).")
    return dest_path


def build_benchmark_graph(
    raw_path: str = "data/raw/2wikimultihopqa_dev.parquet",
    output_path: str = "data/processed/2wikimultihop_benchmark_200.json",
    num_queries: int = 200,
) -> str:
    """Parses raw parquet into verified multi-hop reasoning chains with documentary provenance."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset file not found at '{raw_path}'. Run with --download first."
        )

    print(f"[Dataset] Reading raw dataset from '{raw_path}'...")
    df = pd.read_parquet(raw_path)
    comp_df = df[df["type"].isin(["compositional", "inference"])].copy()
    print(f"[Dataset] Found {len(comp_df)} compositional/inference multi-hop QA pairs.")

    nodes_dict = {}
    edges_dict = {}
    queries = []

    for _, row in comp_df.head(num_queries).iterrows():
        qid = str(row["_id"])
        q_text = str(row["question"])
        ans = str(row["answer"])
        ev_raw = row["evidences"]
        evidences = json.loads(ev_raw) if isinstance(ev_raw, str) else ev_raw
        supp_raw = row["supporting_facts"]
        supporting = json.loads(supp_raw) if isinstance(supp_raw, str) else supp_raw
        ctx_raw = row["context"]
        context = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw

        # Map paragraph title -> list of sentence strings
        ctx_map = {item[0]: item[1] for item in context if len(item) == 2}

        # Extract documentary sentences as provenance
        supp_texts = []
        for sf in supporting:
            if len(sf) >= 2:
                title, s_idx = sf[0], sf[1]
                if title in ctx_map and s_idx < len(ctx_map[title]):
                    supp_texts.append(ctx_map[title][s_idx])

        q_edge_ids = []
        if evidences and len(evidences) >= 2:
            for ev in evidences:
                if len(ev) == 3:
                    s, r, o = str(ev[0]).strip(), str(ev[1]).strip(), str(ev[2]).strip()
                    if s and r and o:
                        edge_id = f"{s}__[{r}]__{o}"
                        q_edge_ids.append(edge_id)

                        if s not in nodes_dict:
                            nodes_dict[s] = {"id": s, "name": s, "entity_type": "Entity"}
                        if o not in nodes_dict:
                            nodes_dict[o] = {"id": o, "name": o, "entity_type": "Entity"}

                        if edge_id not in edges_dict:
                            prov_list = []
                            for i, stext in enumerate(supp_texts):
                                prov_list.append({
                                    "doc_id": f"{qid}_prov_{i}",
                                    "text_snippet": stext,
                                    "confidence": 0.95
                                })
                            edges_dict[edge_id] = {
                                "id": edge_id,
                                "source": s,
                                "relation": r,
                                "target": o,
                                "base_reliability": 0.90,
                                "provenance": prov_list,
                                "frequency": 1
                            }
                        else:
                            edges_dict[edge_id]["frequency"] += 1

            queries.append({
                "query_id": qid,
                "question": q_text,
                "ground_truth_answer": ans,
                "source_entity": str(evidences[0][0]),
                "target_entity": str(evidences[-1][2]),
                "ground_truth_path_edges": q_edge_ids,
                "hop_count": len(evidences),
                "category": str(row["type"])
            })

    # Add context entities as distractor nodes for realistic graph exploration
    for _, row in comp_df.head(num_queries).iterrows():
        ctx_raw = row["context"]
        context = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        for item in context:
            if len(item) == 2:
                title = str(item[0]).strip()
                if title and title not in nodes_dict and len(nodes_dict) < 3000:
                    nodes_dict[title] = {"id": title, "name": title, "entity_type": "Entity"}

    out_data = {
        "name": f"2WikiMultihopQA_Benchmark_{len(queries)}",
        "description": "Official 2WikiMultihopQA multi-hop benchmark with ground-truth evidence triples and provenance.",
        "graph": {
            "nodes": list(nodes_dict.values()),
            "edges": list(edges_dict.values())
        },
        "queries": queries
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(
        f"[Dataset] Successfully constructed benchmark graph with {len(nodes_dict)} nodes, "
        f"{len(edges_dict)} edges, and {len(queries)} multi-hop evaluation queries."
    )
    print(f"[Dataset] Saved to: '{output_path}'")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download and build 2WikiMultihopQA benchmark for RiskPath.")
    parser.add_argument("--download", action="store_true", help="Download raw dev.parquet from Hugging Face.")
    parser.add_argument("--queries", type=int, default=200, help="Number of benchmark queries to process (default: 200).")
    parser.add_argument("--raw-file", type=str, default="data/raw/2wikimultihopqa_dev.parquet", help="Path to raw parquet file.")
    parser.add_argument("--out-file", type=str, default="data/processed/2wikimultihop_benchmark_200.json", help="Output JSON path.")
    args = parser.parse_args()

    if args.download or not os.path.exists(args.raw_file):
        download_2wiki_raw(args.raw_file)

    build_benchmark_graph(raw_path=args.raw_file, output_path=args.out_file, num_queries=args.queries)


if __name__ == "__main__":
    main()
