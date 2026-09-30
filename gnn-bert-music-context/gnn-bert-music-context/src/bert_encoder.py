"""
Task 1: BERT Multi-Label Tag Classifier
Implements DistilBERT/BERT text encoder with a multi-label classification head.
Also exports contextual token embeddings H_text for downstream Task 3 Fusion.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional, List


class BertTagClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_tags: int = 50,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_tags = num_tags
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.bert.config.hidden_size

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, num_tags)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass:
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - labels: (batch_size, num_tags) binary target matrix
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Sequence of contextual token embeddings H_text: (batch_size, seq_len, hidden_dim)
        h_text = outputs.last_hidden_state

        # CLS token representation t: (batch_size, hidden_dim)
        cls_token = h_text[:, 0, :]
        pooled = self.dropout(cls_token)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probs": probs,
            "cls_embedding": cls_token,
            "h_text": h_text,
        }

        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            result["loss"] = loss

        return result

    def get_token_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts H_text and CLS token embeddings for multimodal fusion.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_text = outputs.last_hidden_state
        cls_token = h_text[:, 0, :]
        return h_text, cls_token


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Loads Hugging Face tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)
