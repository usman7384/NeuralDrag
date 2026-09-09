"""Mesh-aware GNN surrogate: per-vertex pressure head + global drag-coefficient head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FeaStConv, global_max_pool, global_mean_pool


class MeshSurrogate(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: tuple[int, ...] = (64, 128, 256),
        heads: int = 4,
    ):
        super().__init__()
        channels = (in_channels,) + hidden_channels
        self.convs = nn.ModuleList(
            [
                FeaStConv(channels[i], channels[i + 1], heads=heads)
                for i in range(len(channels) - 1)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:]])

        last = channels[-1]
        self.pressure_head = nn.Sequential(
            nn.Linear(last, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.cd_head = nn.Sequential(
            nn.Linear(last * 2, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        h = self.encode(x, edge_index)
        pressure = self.pressure_head(h).squeeze(-1)

        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        cd = self.cd_head(pooled).squeeze(-1)

        return pressure, cd
