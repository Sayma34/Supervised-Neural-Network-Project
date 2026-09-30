"""
PyTorch & PyTorch Geometric Dataset Loaders
Manages FMA-small, MagnaTagATune annotations, and multimodal paired mini-batches.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer

from src.audio_features import AudioFeatureExtractor
from src.graph_builder import MusicGraphBuilder


class MusicMultimodalDataset(Dataset):
    def __init__(
        self,
        audio_files: List[str],
        text_descriptions: List[str],
        tag_labels: Optional[np.ndarray] = None,
        genre_labels: Optional[np.ndarray] = None,
        emotion_labels: Optional[np.ndarray] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        max_token_len: int = 128,
        precomputed_graphs_dir: Optional[str] = None,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        graph_builder: Optional[MusicGraphBuilder] = None,
    ):
        self.audio_files = audio_files
        self.text_descriptions = text_descriptions
        self.tag_labels = tag_labels
        self.genre_labels = genre_labels
        self.emotion_labels = emotion_labels
        self.max_token_len = max_token_len
        self.precomputed_graphs_dir = precomputed_graphs_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.graph_builder = graph_builder or MusicGraphBuilder()

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audio_path = self.audio_files[idx]
        text = self.text_descriptions[idx]

        # 1. Text Tokenization
        tokenized = self.tokenizer(
            text,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # 2. Graph and Mel-Spectrogram
        graph_loaded = False
        if self.precomputed_graphs_dir:
            track_name = os.path.splitext(os.path.basename(audio_path))[0]
            pt_path = os.path.join(self.precomputed_graphs_dir, f"{track_name}.pt")
            if os.path.exists(pt_path):
                graph = torch.load(pt_path)
                graph_loaded = True

        if not graph_loaded:
            # Extract features on-the-fly or generate fallback
            try:
                processed = self.feature_extractor.process_track(audio_path)
                mel_spec = torch.tensor(processed["mel_spectrogram"], dtype=torch.float32)
                graph = self.graph_builder.build_graph_from_features(
                    processed["node_features"], track_id=str(idx)
                )
            except Exception:
                # Fallback synthetic graph (10 segments, 39 features)
                dummy_feats = np.random.randn(10, 39).astype(np.float32)
                mel_spec = torch.randn(128, 1292, dtype=torch.float32)
                graph = self.graph_builder.build_graph_from_features(dummy_feats, track_id=str(idx))
        else:
            mel_spec = torch.zeros((128, 1292), dtype=torch.float32)

        item = {
            "graph": graph,
            "mel_spectrogram": mel_spec,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text,
            "audio_path": audio_path,
        }

        if self.tag_labels is not None:
            item["tag_labels"] = torch.tensor(self.tag_labels[idx], dtype=torch.float32)

        if self.genre_labels is not None:
            item["genre_labels"] = torch.tensor(self.genre_labels[idx], dtype=torch.long)

        if self.emotion_labels is not None:
            item["emotion_labels"] = torch.tensor(self.emotion_labels[idx], dtype=torch.float32)

        return item


def multimodal_collate_fn(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that batches PyG graphs using PyG's Batch.from_data_list,
    along with standard PyTorch tensors for text and labels.
    """
    graphs = [item["graph"] for item in batch_list]
    pyg_batch = Batch.from_data_list(graphs)

    input_ids = torch.stack([item["input_ids"] for item in batch_list])
    attention_mask = torch.stack([item["attention_mask"] for item in batch_list])
    mel_specs = torch.stack([item["mel_spectrogram"] for item in batch_list])
    texts = [item["text"] for item in batch_list]

    result = {
        "graph_batch": pyg_batch,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mel_spectrogram": mel_specs,
        "texts": texts,
    }

    if "tag_labels" in batch_list[0]:
        result["tag_labels"] = torch.stack([item["tag_labels"] for item in batch_list])

    if "genre_labels" in batch_list[0]:
        result["genre_labels"] = torch.stack([item["genre_labels"] for item in batch_list])

    if "emotion_labels" in batch_list[0]:
        result["emotion_labels"] = torch.stack([item["emotion_labels"] for item in batch_list])

    return result
