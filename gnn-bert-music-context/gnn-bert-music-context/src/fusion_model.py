"""
Task 3: GNN-BERT Fusion for Multi-Context Understanding
Implements Cross-Attention and Concatenation fusion between Graph representations and BERT embeddings.
Supports multi-task loss (tags + optional valence/arousal emotion regression).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim: int, text_dim: int, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for Query (from Graph g) and Key/Value (from BERT H_text)
        self.w_q = nn.Linear(graph_dim, d_model)
        self.w_k = nn.Linear(text_dim, d_model)
        self.w_v = nn.Linear(text_dim, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.scale = (self.head_dim) ** 0.5
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, g: torch.Tensor, h_text: torch.Tensor, text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - g: (batch_size, graph_dim) graph summary vector
        - h_text: (batch_size, seq_len, text_dim) contextual token sequence from BERT
        - text_mask: (batch_size, seq_len) binary attention mask (1 for token, 0 for pad)
        """
        batch_size = g.size(0)
        seq_len = h_text.size(1)

        # Q from graph: (batch_size, 1, d_model)
        q = self.w_q(g).unsqueeze(1)
        # K and V from text: (batch_size, seq_len, d_model)
        k = self.w_k(h_text)
        v = self.w_v(h_text)

        # Reshape for multi-head attention: (batch_size, n_heads, len, head_dim)
        q = q.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale # (batch_size, n_heads, 1, seq_len)

        if text_mask is not None:
            mask = text_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1) # (batch_size, n_heads, 1, seq_len)
        context = torch.matmul(attn_weights, v)   # (batch_size, n_heads, 1, head_dim)

        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model).squeeze(1)
        context = self.layer_norm(self.w_out(context))

        # Average attention weights across heads for visualization
        avg_attn = attn_weights.squeeze(2).mean(dim=1) # (batch_size, seq_len)

        return context, avg_attn


class GNNBertFusionModel(nn.Module):
    def __init__(
        self,
        gnn_encoder: nn.Module,
        bert_encoder: nn.Module,
        num_tags: int = 50,
        fusion_type: str = "cross_attention", # "cross_attention" or "concat"
        d_model: int = 128,
        predict_emotion: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn = gnn_encoder
        self.bert = bert_encoder
        self.num_tags = num_tags
        self.fusion_type = fusion_type
        self.predict_emotion = predict_emotion

        graph_dim = gnn_encoder.out_dim
        text_dim = bert_encoder.hidden_dim

        if fusion_type == "cross_attention":
            self.cross_attn = CrossAttentionFusion(graph_dim, text_dim, d_model=d_model)
            fused_dim = graph_dim + d_model
        else:
            # Early Concat ablation: CONCAT(g, t_CLS)
            fused_dim = graph_dim + text_dim

        self.dropout = nn.Dropout(dropout)
        self.tag_classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_tags),
        )

        if predict_emotion:
            # Predicts continuous valence and arousal: (v, a) in range [1, 9]
            self.emotion_head = nn.Sequential(
                nn.Linear(fused_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        self.tag_loss_fn = nn.BCEWithLogitsLoss()
        self.emotion_loss_fn = nn.MSELoss()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tag_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        # 1. Graph representation g
        g = self.gnn.extract_graph_embedding(x, edge_index, batch)

        # 2. Text representations
        h_text, t_cls = self.bert.get_token_embeddings(input_ids, attention_mask)

        # 3. Fusion
        attn_weights = None
        if self.fusion_type == "cross_attention":
            context, attn_weights = self.cross_attn(g, h_text, text_mask=attention_mask)
            z = torch.cat([g, context], dim=-1)
        else:
            # Early concat
            z = torch.cat([g, t_cls], dim=-1)

        z_out = self.dropout(z)
        tag_logits = self.tag_classifier(z_out)
        tag_probs = torch.sigmoid(tag_logits)

        result = {
            "fused_embedding": z,
            "tag_logits": tag_logits,
            "tag_probs": tag_probs,
            "attention_weights": attn_weights,
        }

        if self.predict_emotion:
            emotion_preds = self.emotion_head(z_out)
            result["emotion_preds"] = emotion_preds

        # 4. Multi-task loss computation
        total_loss = 0.0
        if tag_labels is not None:
            loss_tags = self.tag_loss_fn(tag_logits, tag_labels.float())
            result["loss_tags"] = loss_tags
            total_loss = total_loss + loss_tags

        if self.predict_emotion and emotion_labels is not None:
            # (valence, arousal)
            loss_v = self.emotion_loss_fn(emotion_preds[:, 0], emotion_labels[:, 0])
            loss_a = self.emotion_loss_fn(emotion_preds[:, 1], emotion_labels[:, 1])
            loss_emotion = alpha * loss_v + beta * loss_a
            result["loss_emotion"] = loss_emotion
            total_loss = total_loss + loss_emotion

        if tag_labels is not None or emotion_labels is not None:
            result["loss"] = total_loss

        return result
