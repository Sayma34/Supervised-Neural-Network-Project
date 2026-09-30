"""
Task 4: Cross-Modal Contrastive Alignment (InfoNCE Dual-Encoder)
Aligns structural audio graph embeddings and text embeddings in a shared metric space.
Supports cross-modal retrieval (Audio -> Text, Text -> Audio) and computes Recall@K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ContrastiveGNNBERT(nn.Module):
    def __init__(
        self,
        gnn_encoder: nn.Module,
        bert_encoder: nn.Module,
        projection_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.gnn = gnn_encoder
        self.bert = bert_encoder
        self.temperature = nn.Parameter(torch.tensor(temperature))

        graph_dim = gnn_encoder.out_dim
        text_dim = bert_encoder.hidden_dim

        # Projection heads into shared multimodal embedding space
        self.audio_proj = nn.Sequential(
            nn.Linear(graph_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # 1. Audio graph embedding
        g_raw = self.gnn.extract_graph_embedding(x, edge_index, batch)
        g = self.audio_proj(g_raw)
        g = F.normalize(g, p=2, dim=-1)

        # 2. Text CLS embedding
        _, t_cls = self.bert.get_token_embeddings(input_ids, attention_mask)
        t = self.text_proj(t_cls)
        t = F.normalize(t, p=2, dim=-1)

        # 3. InfoNCE symmetric contrastive loss
        tau = torch.clamp(self.temperature, min=0.01, max=1.0)
        # Similarity matrix: (N, N)
        sim_matrix = torch.matmul(g, t.t()) / tau

        # Targets are diagonal indices (0, 1, 2, ..., N-1)
        targets = torch.arange(g.size(0), device=g.device)

        # Symmetric InfoNCE loss (Audio -> Text + Text -> Audio)
        loss_audio_to_text = F.cross_entropy(sim_matrix, targets)
        loss_text_to_audio = F.cross_entropy(sim_matrix.t(), targets)
        total_loss = (loss_audio_to_text + loss_text_to_audio) / 2.0

        return {
            "loss": total_loss,
            "sim_matrix": sim_matrix,
            "audio_embeddings": g,
            "text_embeddings": t,
        }

    @torch.no_grad()
    def compute_retrieval_metrics(
        self, audio_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes Recall@1, Recall@5, Recall@10 for both Audio -> Text and Text -> Audio.
        - audio_embeds: (N, D)
        - text_embeds: (N, D)
        """
        sim_matrix = torch.matmul(audio_embeds, text_embeds.t()).cpu().numpy()
        N = sim_matrix.shape[0]

        # Audio -> Text (for each audio, rank text candidates)
        ranks_a2t = []
        for i in range(N):
            sorted_indices = sim_matrix[i].argsort()[::-1]
            rank = (sorted_indices == i).nonzero()[0][0]
            ranks_a2t.append(rank)

        # Text -> Audio (for each text, rank audio candidates)
        ranks_t2a = []
        for i in range(N):
            sorted_indices = sim_matrix[:, i].argsort()[::-1]
            rank = (sorted_indices == i).nonzero()[0][0]
            ranks_t2a.append(rank)

        def recall_at_k(ranks, k):
            return float(sum(r < k for r in ranks) / len(ranks))

        return {
            "Audio2Text_R@1": recall_at_k(ranks_a2t, 1),
            "Audio2Text_R@5": recall_at_k(ranks_a2t, 5),
            "Audio2Text_R@10": recall_at_k(ranks_a2t, 10),
            "Text2Audio_R@1": recall_at_k(ranks_t2a, 1),
            "Text2Audio_R@5": recall_at_k(ranks_t2a, 5),
            "Text2Audio_R@10": recall_at_k(ranks_t2a, 10),
        }
