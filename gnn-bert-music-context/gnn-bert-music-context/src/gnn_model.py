"""
Task 2: Graph Neural Network on Music Structure Graphs
Implements GraphSAGE and GAT architectures with global mean pooling readout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool
from typing import Optional, Dict


class MusicGNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 39,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_classes: int = 8,
        num_layers: int = 2,
        gnn_type: str = "GraphSAGE",
        dropout: float = 0.2,
        is_multilabel: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.is_multilabel = is_multilabel

        self.convs = nn.ModuleList()
        if gnn_type == "GAT":
            self.convs.append(GATConv(in_dim, hidden_dim // 2, heads=2))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // 2, heads=2))
            self.convs.append(GATConv(hidden_dim, out_dim, heads=1))
        else:
            # Default: GraphSAGE
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, out_dim))

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
            + [nn.BatchNorm1d(out_dim)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        if is_multilabel:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass:
        - x: (total_nodes_in_batch, in_dim)
        - edge_index: (2, total_edges_in_batch)
        - batch: (total_nodes_in_batch,) assigning each node to its graph in the mini-batch
        - edge_weight: (total_edges_in_batch, 1) optional edge weights
        """
        h = x
        for i, conv in enumerate(self.convs):
            if self.gnn_type == "GAT":
                h = conv(h, edge_index)
            else:
                h = conv(h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Graph Readout: Global Mean Pooling
        # g has shape (batch_size, out_dim)
        g = global_mean_pool(h, batch)

        logits = self.classifier(g)
        probs = torch.sigmoid(logits) if self.is_multilabel else F.softmax(logits, dim=-1)

        result = {
            "node_embeddings": h,
            "graph_embedding": g,
            "logits": logits,
            "probs": probs,
        }

        if labels is not None:
            if self.is_multilabel:
                loss = self.loss_fn(logits, labels.float())
            else:
                loss = self.loss_fn(logits, labels.long())
            result["loss"] = loss

        return result

    def extract_graph_embedding(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts pooled graph-level representation vector g for multimodal fusion.
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
        g = global_mean_pool(h, batch)
        return g
