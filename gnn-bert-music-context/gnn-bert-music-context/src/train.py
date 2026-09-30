"""
Unified Training Pipeline
Contains training and validation loops for Task 1 (BERT), Task 2 (GNN & CNN), Task 3 (Fusion), and Task 4 (Contrastive).
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader

from src.evaluate import (
    compute_multilabel_metrics,
    compute_multiclass_metrics,
    compute_emotion_metrics,
)


def train_bert_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Trains Task 1 BERT multi-label classifier for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["tag_labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_bert(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Evaluates Task 1 BERT multi-label classifier."""
    model.eval()
    all_preds = []
    all_targets = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["tag_labels"].cpu().numpy()

        out = model(input_ids, attention_mask)
        probs = out["probs"].cpu().numpy()

        all_preds.append(probs)
        all_targets.append(labels)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    return compute_multilabel_metrics(y_true, y_pred)


def train_gnn_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_multilabel: bool = False,
) -> float:
    """Trains Task 2 GNN model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        pyg_batch = batch["graph_batch"].to(device)
        labels = batch["tag_labels" if is_multilabel else "genre_labels"].to(device)

        optimizer.zero_grad()
        out = model(
            x=pyg_batch.x,
            edge_index=pyg_batch.edge_index,
            batch=pyg_batch.batch,
            labels=labels,
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_gnn(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_multilabel: bool = False,
) -> Dict[str, float]:
    """Evaluates Task 2 GNN model."""
    model.eval()
    all_preds = []
    all_targets = []
    for batch in dataloader:
        pyg_batch = batch["graph_batch"].to(device)
        labels = batch["tag_labels" if is_multilabel else "genre_labels"].cpu().numpy()

        out = model(
            x=pyg_batch.x,
            edge_index=pyg_batch.edge_index,
            batch=pyg_batch.batch,
        )
        probs = out["probs"].cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets) if is_multilabel else np.concatenate(all_targets)

    if is_multilabel:
        return compute_multilabel_metrics(y_true, y_pred)
    else:
        return compute_multiclass_metrics(y_true, y_pred)


def train_fusion_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Trains Task 3 Multimodal Cross-Attention Fusion model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        pyg_batch = batch["graph_batch"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tag_labels = batch.get("tag_labels", None)
        if tag_labels is not None:
            tag_labels = tag_labels.to(device)
        emotion_labels = batch.get("emotion_labels", None)
        if emotion_labels is not None:
            emotion_labels = emotion_labels.to(device)

        optimizer.zero_grad()
        out = model(
            x=pyg_batch.x,
            edge_index=pyg_batch.edge_index,
            batch=pyg_batch.batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tag_labels=tag_labels,
            emotion_labels=emotion_labels,
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_fusion(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> Dict[str, Any]:
    """Evaluates Task 3 Fusion model."""
    model.eval()
    all_tag_preds = []
    all_tag_targets = []
    all_embeddings = []

    for batch in dataloader:
        pyg_batch = batch["graph_batch"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tag_labels = batch["tag_labels"].cpu().numpy()

        out = model(
            x=pyg_batch.x,
            edge_index=pyg_batch.edge_index,
            batch=pyg_batch.batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        all_tag_preds.append(out["tag_probs"].cpu().numpy())
        all_tag_targets.append(tag_labels)
        all_embeddings.append(out["fused_embedding"].cpu().numpy())

    y_pred = np.vstack(all_tag_preds)
    y_true = np.vstack(all_tag_targets)
    metrics = compute_multilabel_metrics(y_true, y_pred)
    metrics["fused_embeddings"] = np.vstack(all_embeddings)
    return metrics


def train_contrastive_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Trains Task 4 Contrastive InfoNCE model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        pyg_batch = batch["graph_batch"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        out = model(
            x=pyg_batch.x,
            edge_index=pyg_batch.edge_index,
            batch=pyg_batch.batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_contrastive(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Evaluates Task 4 retrieval metrics."""
    model.eval()
    audio_embeds = []
    text_embeds = []
    for batch in dataloader:
        pyg_batch = batch["graph_batch"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(
            x=pyg_batch.x,
            edge_index=pyg_batch.edge_index,
            batch=pyg_batch.batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        audio_embeds.append(out["audio_embeddings"])
        text_embeds.append(out["text_embeddings"])

    all_a = torch.cat(audio_embeds, dim=0)
    all_t = torch.cat(text_embeds, dim=0)
    return model.compute_retrieval_metrics(all_a, all_t)
