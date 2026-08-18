"""PyTorch models for text, vision, and intermediate fusion."""

from __future__ import annotations

import torch
from torch import nn


class RiskLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        output, (hidden, _) = self.lstm(embedded)
        return hidden[-1]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.encode(token_ids))
        return self.classifier(features)


class RiskCNN(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding_dim = embedding_dim
        self.project = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self.features(images).flatten(1)
        return torch.relu(self.project(pooled))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images))


class IntermediateFusionNet(nn.Module):
    def __init__(
        self,
        struct_dim: int = 64,
        text_dim: int = 64,
        vision_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
        )
        self.text_encoder = nn.Sequential(nn.Linear(text_dim, 32), nn.ReLU())
        self.vision_encoder = nn.Sequential(nn.Linear(vision_dim, 32), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, structured: torch.Tensor, text: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.struct_encoder(structured),
                self.text_encoder(text),
                self.vision_encoder(vision),
            ],
            dim=1,
        )

    def forward(
        self,
        structured: torch.Tensor,
        text: torch.Tensor,
        vision: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(self.encode(structured, text, vision))
