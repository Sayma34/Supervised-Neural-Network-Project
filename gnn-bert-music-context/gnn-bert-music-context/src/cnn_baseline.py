"""
Baseline B2: 2D CNN on Mel-Spectrogram
Standard Convolutional Neural Network baseline operating on 2D Log-Mel Spectrograms.
Provides direct comparison against the GNN graph-based approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioCNNBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 1,
        base_channels: int = 32,
        is_multilabel: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.is_multilabel = is_multilabel

        self.conv1 = ConvBlock(in_channels, base_channels)       # 32
        self.conv2 = ConvBlock(base_channels, base_channels * 2) # 64
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4) # 128
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8) # 256

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        if is_multilabel:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        mel_spec: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        - mel_spec: (batch_size, 1, 128, time_steps) or (batch_size, 128, time_steps)
        """
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        x = self.conv1(mel_spec)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x).flatten(1) # (batch_size, 256)
        logits = self.fc(x)
        probs = torch.sigmoid(logits) if self.is_multilabel else F.softmax(logits, dim=-1)

        result = {
            "embedding": x,
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
