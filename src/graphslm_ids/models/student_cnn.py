from __future__ import annotations

import torch
from torch import nn


class Student1DCNN(nn.Module):
    """Compact 1D CNN that maps 256-byte payload vectors into semantic embeddings."""

    def __init__(self, embedding_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        if payload.ndim == 2:
            payload = payload.unsqueeze(1)
        x = payload.float() / 255.0
        x = self.features(x)
        return self.head(x)
