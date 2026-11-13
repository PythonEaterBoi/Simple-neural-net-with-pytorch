"""Video models: shared per-frame CNN encoder + temporal pooling classifier."""

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """Encodes a single frame into a fixed-size vector."""

    def __init__(self, emb_dim: int = 128, base: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(base * 4 * 16, emb_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


class VideoClassifier(nn.Module):
    """Classifies a clip by encoding each frame and mean-pooling over time."""

    def __init__(self, n_classes: int, emb_dim: int = 128, base: int = 32, dropout: float = 0.3):
        super().__init__()
        self.encoder = FrameEncoder(emb_dim=emb_dim, base=base)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(emb_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t = x.shape[:2]
        feats = self.encoder(x.reshape(b * t, *x.shape[2:]))
        pooled = feats.reshape(b, t, -1).mean(dim=1)
        return self.head(pooled)
