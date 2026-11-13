"""Tabular model: MLP for classification or regression."""

import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden),
                       nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
