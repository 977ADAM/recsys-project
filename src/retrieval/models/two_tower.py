from __future__ import annotations

import torch
import torch.nn as nn


class TwoTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_banners: int,
        emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.user_tower = nn.Embedding(n_users, emb_dim)
        self.banner_tower = nn.Embedding(n_banners, emb_dim)
        self.embedding_dim = emb_dim
        self.hidden_dims: tuple[int, ...] = ()
        self.dropout = 0.0

    def encode_user(self, user_ids):
        return self.user_tower(user_ids)

    def encode_banner(self, banner_ids):
        return self.banner_tower(banner_ids)

    def encode_all_users(self) -> torch.Tensor:
        return self.user_tower.weight

    def encode_all_banners(self) -> torch.Tensor:
        return self.banner_tower.weight

    def score_all_banners(self):
        return self.encode_all_users() @ self.encode_all_banners().T

    def forward(self, user_ids, banner_ids):
        user_vec = self.encode_user(user_ids)
        banner_vec = self.encode_banner(banner_ids)
        return (user_vec * banner_vec).sum(dim=1)
