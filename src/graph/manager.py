"""Knowledge Graph manager providing NetworkX-backed indexing, traversal, and persistence."""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Any, Set, Tuple
import networkx as nx

from .schema import Node, Edge, Provenance, ReasoningPath


class KnowledgeGraphManager:
    """Encapsulates a directed multigraph representing entities and relations with risk attributes."""

    def __init__(self, name: str = "RiskPathKG"):
        self.name = name
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph(name=name)
        # Fast lookup dictionaries
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}  # Key: edge.id

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def add_node(self, node: Node) -> None:
        """Inserts or updates a node in the graph."""
        self._nodes[node.id] = node
        self.graph.add_node(
            node.id,
            name=node.name,
            entity_type=node.entity_type,
            description=node.description,
            aliases=node.aliases,
            attributes=node.attributes,
        )

    def add_edge(self, edge: Edge) -> None:
        """Inserts or updates a directed edge in the graph."""
        # Ensure endpoints exist
        if edge.source not in self._nodes:
            self.add_node(Node(id=edge.source, name=edge.source))
        if edge.target not in self._nodes:
            self.add_node(Node(id=edge.target, name=edge.target))

        self._edges[edge.id] = edge
        self.graph.add_edge(
            edge.source,
            edge.target,
            key=edge.relation,
            id=edge.id,
            relation=edge.relation,
            base_reliability=edge.base_reliability,
            base_risk=edge.base_risk,
            causal_importance=edge.causal_importance,
            pwrs=edge.pwrs,
            is_corrupted=edge.is_corrupted,
            frequency=edge.frequency,
        )

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieves Node dataclass by ID."""
        return self._nodes.get(node_id)

    def get_edge_by_id(self, edge_id: str) -> Optional[Edge]:
        """Retrieves Edge dataclass by its unique ID."""
        return self._edges.get(edge_id)

    def get_edge(self, source: str, relation: str, target: str) -> Optional[Edge]:
        """Looks up an edge by triple components."""
        edge_id = f"{source}__[{relation}]__{target}"
        if edge_id in self._edges:
            return self._edges[edge_id]
        # Fallback search if custom ID was used
        for e in self._edges.values():
            if e.source == source and e.relation == relation and e.target == target:
                return e
        return None

    def remove_edge_by_id(self, edge_id: str) -> bool:
        """Removes an edge by ID from both NetworkX graph and cache."""
        edge = self._edges.get(edge_id)
        if not edge:
            return False
        if self.graph.has_edge(edge.source, edge.target, key=edge.relation):
            self.graph.remove_edge(edge.source, edge.target, key=edge.relation)
        del self._edges[edge_id]
        return True

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Returns all directed edges starting from node_id."""
        if not self.graph.has_node(node_id):
            return []
        edges = []
        for _, v, key in self.graph.out_edges(node_id, keys=True):
            e = self.get_edge(node_id, key, v)
            if e:
                edges.append(e)
        return edges

    def get_neighbors(self, node_id: str) -> List[str]:
        """Returns adjacent successor node IDs."""
        if not self.graph.has_node(node_id):
            return []
        return list(self.graph.successors(node_id))

    def get_local_subgraph(self, center_node: str, max_hops: int = 2) -> KnowledgeGraphManager:
        """Extracts an egocentric subgraph around center_node up to max_hops."""
        sub_mgr = KnowledgeGraphManager(name=f"Sub_{center_node}_{max_hops}h")
        if not self.graph.has_node(center_node):
            return sub_mgr

        visited_nodes: Set[str] = {center_node}
        current_frontier: Set[str] = {center_node}

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for u in current_frontier:
                for v in self.graph.successors(u):
                    next_frontier.add(v)
            visited_nodes.update(next_frontier)
            current_frontier = next_frontier
            if not current_frontier:
                break

        # Copy nodes
        for nid in visited_nodes:
            if nid in self._nodes:
                sub_mgr.add_node(self._nodes[nid])

        # Copy edges between visited nodes
        for u in visited_nodes:
            for v in visited_nodes:
                if self.graph.has_edge(u, v):
                    for key in self.graph[u][v]:
                        e = self.get_edge(u, key, v)
                        if e:
                            sub_mgr.add_edge(e)

        return sub_mgr

    def find_all_paths(
        self, source: str, target: str, max_hops: int = 3, max_paths: int = 10
    ) -> List[ReasoningPath]:
        """Discovers reasoning paths from source to target within max_hops."""
        if not (self.graph.has_node(source) and self.graph.has_node(target)):
            return []

        # Find simple paths using NetworkX
        try:
            raw_paths = list(
                nx.all_simple_edge_paths(self.graph, source, target, cutoff=max_hops)
            )
        except Exception:
            return []

        paths_result: List[ReasoningPath] = []
        for raw_path in raw_paths[:max_paths]:
            path_edges: List[Edge] = []
            for u, v, key in raw_path:
                edge = self.get_edge(u, key, v)
                if edge:
                    path_edges.append(edge)
            if path_edges:
                paths_result.append(ReasoningPath(edges=path_edges))

        return paths_result

    def get_stats(self) -> Dict[str, Any]:
        """Computes topological summary statistics of the graph."""
        n = self.node_count
        m = self.edge_count
        corrupted_count = sum(1 for e in self._edges.values() if e.is_corrupted)
        avg_out_degree = float(m / n) if n > 0 else 0.0

        return {
            "name": self.name,
            "nodes": n,
            "edges": m,
            "corrupted_edges": corrupted_count,
            "corruption_ratio": (corrupted_count / m) if m > 0 else 0.0,
            "avg_out_degree": round(avg_out_degree, 2),
            "is_connected": nx.is_weakly_connected(self.graph) if n > 0 else False,
        }

    def copy(self) -> KnowledgeGraphManager:
        """Creates a deep copy of the manager and underlying graph."""
        new_mgr = KnowledgeGraphManager(name=f"{self.name}_copy")
        for node in self._nodes.values():
            new_mgr.add_node(
                Node(
                    id=node.id,
                    name=node.name,
                    entity_type=node.entity_type,
                    description=node.description,
                    aliases=list(node.aliases),
                    attributes=dict(node.attributes),
                )
            )
        for edge in self._edges.values():
            new_edge = Edge(
                id=edge.id,
                source=edge.source,
                relation=edge.relation,
                target=edge.target,
                base_reliability=edge.base_reliability,
                causal_importance=edge.causal_importance,
                pwrs=edge.pwrs,
                provenance=[Provenance(**p.to_dict()) for p in edge.provenance],
                frequency=edge.frequency,
                is_corrupted=edge.is_corrupted,
                corruption_metadata=dict(edge.corruption_metadata) if edge.corruption_metadata else None,
                attributes=dict(edge.attributes),
            )
            new_mgr.add_edge(new_edge)
        return new_mgr

    def save_to_json(self, filepath: str) -> None:
        """Serializes nodes and edges to a lightweight JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "name": self.name,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> KnowledgeGraphManager:
        """Deserializes a KnowledgeGraphManager from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        mgr = cls(name=data.get("name", "RiskPathKG"))
        for n_dict in data.get("nodes", []):
            mgr.add_node(Node.from_dict(n_dict))
        for e_dict in data.get("edges", []):
            mgr.add_edge(Edge.from_dict(e_dict))
        return mgr
