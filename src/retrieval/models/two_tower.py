from __future__ import annotations

import torch
import torch.nn as nn


class TwoTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_banners: int,
        emb_dim: int = 64,
        hidden_dims: tuple[int, ...] | list[int] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.user_tower = nn.Embedding(n_users, emb_dim)
        self.banner_tower = nn.Embedding(n_banners, emb_dim)
        self.embedding_dim = emb_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        self.user_mlp = self._build_tower_mlp(emb_dim, self.hidden_dims, self.dropout)
        self.banner_mlp = self._build_tower_mlp(emb_dim, self.hidden_dims, self.dropout)

    @staticmethod
    def _build_tower_mlp(
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> nn.Module:
        if not hidden_dims:
            return nn.Identity()

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, input_dim))
        return nn.Sequential(*layers)

    def encode_user(self, user_ids):
        return self.user_mlp(self.user_tower(user_ids))

    def encode_banner(self, banner_ids):
        return self.banner_mlp(self.banner_tower(banner_ids))

    def encode_all_users(self) -> torch.Tensor:
        return self.user_mlp(self.user_tower.weight)

    def encode_all_banners(self) -> torch.Tensor:
        return self.banner_mlp(self.banner_tower.weight)

    def score_all_banners(self):
        return self.encode_all_users() @ self.encode_all_banners().T

    def forward(self, user_ids, banner_ids):
        user_vec = self.encode_user(user_ids)
        banner_vec = self.encode_banner(banner_ids)
        return (user_vec * banner_vec).sum(dim=1)
