"""
Evaluation Metrics Pipeline
Computes Macro-F1, Micro-F1, AUC-PR, MAE, R^2, and Retrieval Recall@K as defined in Section 6.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    mean_absolute_error,
    r2_score,
)
from typing import Dict, Any, List, Optional


def compute_multilabel_metrics(
    y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes Macro-F1, Micro-F1, Precision, Recall, and mean AUC-PR over tags.
    - y_true: (N, K) binary ground-truth matrix
    - y_pred_probs: (N, K) continuous probabilities [0, 1]
    """
    y_pred_bin = (y_pred_probs >= threshold).astype(int)

    # 1. Macro and Micro F1
    macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)

    # 2. AUC-PR (Area Under Precision-Recall Curve) per tag
    num_tags = y_true.shape[1]
    auc_prs = []
    for k in range(num_tags):
        if y_true[:, k].sum() > 0:
            p, r, _ = precision_recall_curve(y_true[:, k], y_pred_probs[:, k])
            auc_val = auc(r, p)
            auc_prs.append(auc_val)

    mean_auc_pr = float(np.mean(auc_prs)) if len(auc_prs) > 0 else 0.0

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "mean_auc_pr": float(mean_auc_pr),
    }


def compute_multiclass_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> Dict[str, float]:
    """
    For single-label multi-class (e.g. FMA-small 8 genre classes).
    """
    y_pred = np.argmax(y_pred_probs, axis=-1)
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE and R^2 score for valence and arousal emotion regression.
    """
    mae_v = float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
    mae_a = float(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))
    r2_v = float(r2_score(y_true[:, 0], y_pred[:, 0]))
    r2_a = float(r2_score(y_true[:, 1], y_pred[:, 1]))

    return {
        "mae_valence": mae_v,
        "mae_arousal": mae_a,
        "mae_mean": (mae_v + mae_a) / 2.0,
        "r2_valence": r2_v,
        "r2_arousal": r2_a,
    }


def compute_graph_coherence(node_features: np.ndarray, edge_index: np.ndarray, threshold: float = 0.7) -> float:
    """
    Computes graph coherence score: fraction of edges that align with high feature similarity.
    S_graph = 1/|E| * sum_{(i,j) in E} I[cos(h_i, h_j) > tau]
    """
    norms = np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8
    norm_feats = node_features / norms

    coherent_edges = 0
    num_edges = edge_index.shape[1]
    for e in range(num_edges):
        u, v = edge_index[0, e], edge_index[1, e]
        sim = float(np.dot(norm_feats[u], norm_feats[v]))
        if sim > threshold:
            coherent_edges += 1

    return float(coherent_edges / max(num_edges, 1))
