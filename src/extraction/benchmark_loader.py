"""Benchmark dataset loader and generator for multi-hop GraphRAG evaluation.

Explicitly separates:
  1. Official Research Benchmark: 2WikiMultihopQA (200 multi-hop queries, 1,600+ nodes, ground-truth evidence triples).
  2. Illustrative Demo Dataset: Cinema & Science multi-hop graph (28 queries, 86 nodes, for instant viva testing).
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.graph.schema import Node, Edge, Provenance
from src.graph.manager import KnowledgeGraphManager


@dataclass
class BenchmarkQuery:
    """Represents a verified multi-hop question-answer pair for GraphRAG evaluation."""
    query_id: str
    question: str
    ground_truth_answer: str
    source_entity: str
    target_entity: str
    ground_truth_path_edges: List[str]  # List of edge IDs
    hop_count: int = 2
    category: str = "multi_hop"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "ground_truth_answer": self.ground_truth_answer,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "ground_truth_path_edges": self.ground_truth_path_edges,
            "hop_count": self.hop_count,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkQuery:
        return cls(
            query_id=data["query_id"],
            question=data["question"],
            ground_truth_answer=data["ground_truth_answer"],
            source_entity=data["source_entity"],
            target_entity=data["target_entity"],
            ground_truth_path_edges=data.get("ground_truth_path_edges", []),
            hop_count=data.get("hop_count", 2),
            category=data.get("category", "multi_hop"),
        )


class BenchmarkDataset:
    """Holds a benchmark knowledge graph and its corresponding evaluation query set."""

    def __init__(self, name: str = "RiskPathBenchmark", description: str = ""):
        self.name = name
        self.description = description
        self.kg: KnowledgeGraphManager = KnowledgeGraphManager(name=name)
        self.queries: List[BenchmarkQuery] = []

    def add_query(self, query: BenchmarkQuery) -> None:
        self.queries.append(query)

    def save(self, filepath: str) -> None:
        """Saves entire dataset (graph + queries) to a single JSON archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "graph": {
                "nodes": [n.to_dict() for n in self.kg._nodes.values()],
                "edges": [e.to_dict() for e in self.kg._edges.values()],
            },
            "queries": [q.to_dict() for q in self.queries],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> BenchmarkDataset:
        """Loads dataset from JSON archive."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Benchmark file not found at: '{filepath}'")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(name=data.get("name", "RiskPathBenchmark"), description=data.get("description", ""))
        for n_dict in data.get("graph", {}).get("nodes", []):
            dataset.kg.add_node(Node.from_dict(n_dict))
        for e_dict in data.get("graph", {}).get("edges", []):
            dataset.kg.add_edge(Edge.from_dict(e_dict))
        for q_dict in data.get("queries", []):
            dataset.add_query(BenchmarkQuery.from_dict(q_dict))
        return dataset

    @classmethod
    def create_curated_multihop_benchmark(cls, num_distractor_clusters: int = 10) -> BenchmarkDataset:
        """Convenience alias for demo dataset maintaining backward compatibility."""
        return DemoDataset.create_cinema_science_demo(num_distractor_clusters=num_distractor_clusters)

    @classmethod
    def load_benchmark(
        cls,
        benchmark_name: str = "2wiki",
        processed_path: str = "data/processed/2wikimultihop_benchmark_200.json",
        raw_path: str = "data/raw/2wikimultihopqa_dev.parquet",
    ) -> BenchmarkDataset:
        """Unified factory loading either the official 2WikiMultihopQA benchmark or the toy demo."""
        name_clean = benchmark_name.lower()
        if "2wiki" in name_clean:
            if os.path.exists(processed_path):
                return cls.load(processed_path)
            elif os.path.exists(raw_path):
                from src.extraction.download_2wiki import build_benchmark_graph
                build_benchmark_graph(raw_path=raw_path, output_path=processed_path, num_queries=200)
                return cls.load(processed_path)
            else:
                # If neither exists, fallback gracefully with explanation or build demo
                print(
                    f"[Benchmark] Warning: 2WikiMultihopQA processed benchmark not found at '{processed_path}'. "
                    f"Falling back to curated DemoDataset. Run 'python -m src.extraction.download_2wiki' to build 2Wiki."
                )
                return DemoDataset.create_cinema_science_demo()
        else:
            return DemoDataset.create_cinema_science_demo()


class DemoDataset:
    """Small, verified illustrative demo dataset (Cinema & Science) for instant viva demonstration.

    NOTE: This is an illustrative demo dataset, NOT the 2WikiMultihopQA research benchmark.
    """

    @classmethod
    def create_cinema_science_demo(cls, num_distractor_clusters: int = 10) -> BenchmarkDataset:
        ds = BenchmarkDataset(
            name="RiskPath_Demo_Curated",
            description="Illustrative demo dataset covering Cinema and Science for fast viva walk-through.",
        )

        # ----------------- Domain 1: Cinema & Academy Awards -----------------
        directors = [
            ("Inception", "Christopher_Nolan", "University_College_London", "London", "United_Kingdom"),
            ("Oppenheimer", "Christopher_Nolan", "University_College_London", "London", "United_Kingdom"),
            ("Titanic", "James_Cameron", "Fullerton_College", "Fullerton", "United_States"),
            ("Avatar", "James_Cameron", "Fullerton_College", "Fullerton", "United_States"),
            ("Pulp_Fiction", "Quentin_Tarantino", "Narbonne_High_School", "Los_Angeles", "United_States"),
            ("Schindlers_List", "Steven_Spielberg", "California_State_University_Long_Beach", "Long_Beach", "United_States"),
            ("Parasite", "Bong_Joon_ho", "Yonsei_University", "Seoul", "South_Korea"),
            ("Spirited_Away", "Hayao_Miyazaki", "Gakushuin_University", "Tokyo", "Japan"),
            ("The_Godfather", "Francis_Ford_Coppola", "Hofstra_University", "Hempstead", "United_States"),
            ("Blade_Runner_2049", "Denis_Villeneuve", "Universite_du_Quebec_a_Montreal", "Montreal", "Canada"),
        ]

        for movie, director, alma_mater, city, country in directors:
            ds.kg.add_node(Node(id=movie, name=movie.replace("_", " "), entity_type="Film"))
            ds.kg.add_node(Node(id=director, name=director.replace("_", " "), entity_type="Person"))
            ds.kg.add_node(Node(id=alma_mater, name=alma_mater.replace("_", " "), entity_type="University"))
            ds.kg.add_node(Node(id=city, name=city.replace("_", " "), entity_type="City"))
            ds.kg.add_node(Node(id=country, name=country.replace("_", " "), entity_type="Country"))

            e1 = Edge(
                source=movie, relation="director", target=director, base_reliability=0.95,
                provenance=[Provenance(doc_id=f"doc_{movie}_dir", text_snippet=f"{movie.replace('_', ' ')} is directed by {director.replace('_', ' ')}.", confidence=0.98)],
            )
            e2 = Edge(
                source=director, relation="educated_at", target=alma_mater, base_reliability=0.92,
                provenance=[Provenance(doc_id=f"doc_{director}_edu", text_snippet=f"{director.replace('_', ' ')} completed their studies at {alma_mater.replace('_', ' ')}.", confidence=0.95)],
            )
            e3 = Edge(
                source=alma_mater, relation="located_in", target=city, base_reliability=0.96,
                provenance=[Provenance(doc_id=f"doc_{alma_mater}_loc", text_snippet=f"{alma_mater.replace('_', ' ')} is located in {city.replace('_', ' ')}.", confidence=0.99)],
            )
            e4 = Edge(
                source=city, relation="country", target=country, base_reliability=0.98,
                provenance=[Provenance(doc_id=f"doc_{city}_country", text_snippet=f"{city.replace('_', ' ')} is situated in {country.replace('_', ' ')}.", confidence=0.99)],
            )

            ds.kg.add_edge(e1)
            ds.kg.add_edge(e2)
            ds.kg.add_edge(e3)
            ds.kg.add_edge(e4)

            # 2-hop query
            ds.add_query(
                BenchmarkQuery(
                    query_id=f"Q_cinema_{movie}_alma_mater",
                    question=f"Where was the director of {movie.replace('_', ' ')} educated?",
                    ground_truth_answer=alma_mater.replace("_", " "),
                    source_entity=movie,
                    target_entity=alma_mater,
                    ground_truth_path_edges=[e1.id, e2.id],
                    hop_count=2,
                    category="cinema",
                )
            )

            # 3-hop query
            ds.add_query(
                BenchmarkQuery(
                    query_id=f"Q_cinema_{movie}_city",
                    question=f"In which city is the alma mater of {movie.replace('_', ' ')}'s director located?",
                    ground_truth_answer=city.replace("_", " "),
                    source_entity=movie,
                    target_entity=city,
                    ground_truth_path_edges=[e1.id, e2.id, e3.id],
                    hop_count=3,
                    category="cinema",
                )
            )

        # ----------------- Domain 2: Science & Nobel Laureates -----------------
        scientists = [
            ("Theory_of_General_Relativity", "Albert_Einstein", "ETH_Zurich", "Zurich", "Switzerland"),
            ("Discovery_of_Radium", "Marie_Curie", "University_of_Paris", "Paris", "France"),
            ("Quantum_Electrodynamics", "Richard_Feynman", "Princeton_University", "Princeton", "United_States"),
            ("DNA_Double_Helix", "Francis_Crick", "University_College_London", "London", "United_Kingdom"),
            ("Penicillin_Discovery", "Alexander_Fleming", "Imperial_College_London", "London", "United_Kingdom"),
            ("Computer_Architecture", "John_von_Neumann", "University_of_Budapest", "Budapest", "Hungary"),
            ("Uncertainty_Principle", "Werner_Heisenberg", "University_of_Munich", "Munich", "Germany"),
            ("Electromagnetism", "James_Clerk_Maxwell", "University_of_Cambridge", "Cambridge", "United_Kingdom"),
        ]

        for discovery, scientist, institution, city, country in scientists:
            ds.kg.add_node(Node(id=discovery, name=discovery.replace("_", " "), entity_type="ScientificContribution"))
            ds.kg.add_node(Node(id=scientist, name=scientist.replace("_", " "), entity_type="Scientist"))
            ds.kg.add_node(Node(id=institution, name=institution.replace("_", " "), entity_type="University"))
            ds.kg.add_node(Node(id=city, name=city.replace("_", " "), entity_type="City"))
            ds.kg.add_node(Node(id=country, name=country.replace("_", " "), entity_type="Country"))

            es1 = Edge(
                source=discovery, relation="discovered_by", target=scientist, base_reliability=0.98,
                provenance=[Provenance(doc_id=f"doc_{discovery}", text_snippet=f"{discovery} was formulated by {scientist}.")],
            )
            es2 = Edge(
                source=scientist, relation="affiliated_with", target=institution, base_reliability=0.93,
                provenance=[Provenance(doc_id=f"doc_{scientist}_inst", text_snippet=f"{scientist} conducted research at {institution}.")],
            )
            es3 = Edge(
                source=institution, relation="headquarters", target=city, base_reliability=0.97,
                provenance=[Provenance(doc_id=f"doc_{institution}_hq", text_snippet=f"{institution} is located in {city}.")],
            )

            ds.kg.add_edge(es1)
            ds.kg.add_edge(es2)
            ds.kg.add_edge(es3)

            ds.add_query(
                BenchmarkQuery(
                    query_id=f"Q_science_{discovery}_institution",
                    question=f"Which academic institution was the discoverer of {discovery.replace('_', ' ')} affiliated with?",
                    ground_truth_answer=institution.replace("_", " "),
                    source_entity=discovery,
                    target_entity=institution,
                    ground_truth_path_edges=[es1.id, es2.id],
                    hop_count=2,
                    category="science",
                )
            )

        # Distractor nodes
        distractor_universities = ["Harvard_University", "Stanford_University", "MIT", "Oxford_University"]
        for u in distractor_universities:
            ds.kg.add_node(Node(id=u, name=u.replace("_", " "), entity_type="University"))

        for i in range(num_distractor_clusters):
            d_person = f"Distractor_Researcher_{i}"
            ds.kg.add_node(Node(id=d_person, name=f"Researcher {i}", entity_type="Person"))
            ds.kg.add_edge(
                Edge(
                    source=d_person, relation="educated_at",
                    target=distractor_universities[i % len(distractor_universities)],
                    base_reliability=0.60,
                )
            )

        return ds
