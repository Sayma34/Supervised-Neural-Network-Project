"""
Music Structure Graph Builder
Constructs temporal adjacency and acoustic similarity graphs as PyTorch Geometric Data objects.
"""

import json
import torch
import numpy as np
from typing import Dict, Any, Optional
from torch_geometric.data import Data


class MusicGraphBuilder:
    def __init__(
        self,
        similarity_threshold: float = 0.70,
        add_self_loops: bool = True,
        directed: bool = False,
    ):
        self.similarity_threshold = similarity_threshold
        self.add_self_loops = add_self_loops
        self.directed = directed

    def build_graph_from_features(
        self,
        node_features: np.ndarray,
        label: Optional[torch.Tensor] = None,
        track_id: Optional[str] = None,
    ) -> Data:
        """
        Builds a PyG Data graph from extracted node feature vectors h_i^(0).

        Edges created:
        1. Temporal adjacency: (i, i+1) and (i+1, i)
        2. Acoustic similarity: (i, j) if cosine_sim(h_i, h_j) > threshold
        3. Self-loops: (i, i) if add_self_loops is True
        """
        num_nodes = node_features.shape[0]
        x = torch.tensor(node_features, dtype=torch.float32)

        # Normalize rows for cosine similarity calculation
        norms = np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8
        norm_feats = node_features / norms
        sim_matrix = np.dot(norm_feats, norm_feats.T)

        edges = set()
        edge_weights = []

        # 1. Temporal Adjacency Edges
        for i in range(num_nodes - 1):
            edges.add((i, i + 1))
            if not self.directed:
                edges.add((i + 1, i))

        # 2. Acoustic Similarity Edges (beyond adjacent)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and abs(i - j) > 1:
                    sim = sim_matrix[i, j]
                    if sim >= self.similarity_threshold:
                        edges.add((i, j))
                        if not self.directed:
                            edges.add((j, i))

        # 3. Optional Self-Loops
        if self.add_self_loops:
            for i in range(num_nodes):
                edges.add((i, i))

        edge_list = sorted(list(edges))
        if len(edge_list) == 0:
            # Fallback to pure temporal line graph
            for i in range(num_nodes - 1):
                edge_list.append((i, i + 1))
                edge_list.append((i + 1, i))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Compute edge weights (cosine similarity or 1.0 for temporal)
        weights = []
        for src, dst in edge_list:
            if src == dst:
                weights.append(1.0)
            else:
                weights.append(float(sim_matrix[src, dst]))
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if label is not None:
            data.y = label
        if track_id is not None:
            data.track_id = track_id

        return data

    @staticmethod
    def save_graph_pt(data: Data, filepath: str):
        """Saves a PyG graph object to .pt format."""
        torch.save(data, filepath)

    @staticmethod
    def load_graph_pt(filepath: str) -> Data:
        """Loads a PyG graph object from .pt format."""
        return torch.load(filepath)

    @staticmethod
    def export_graph_json(data: Data, filepath: str):
        """Exports graph structure and features to JSON format for inspectability."""
        graph_dict = {
            "num_nodes": int(data.x.size(0)),
            "feature_dim": int(data.x.size(1)),
            "edge_index": data.edge_index.tolist(),
            "edge_weights": data.edge_attr.squeeze().tolist() if data.edge_attr is not None else [],
            "node_features": data.x.tolist(),
            "track_id": getattr(data, "track_id", "unknown"),
        }
        with open(filepath, "w") as f:
            json.dump(graph_dict, f, indent=2)
