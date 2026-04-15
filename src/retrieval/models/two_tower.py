from __future__ import annotations

import torch.nn as nn


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_banners: int, emb_dim: int = 64) -> None:
        super().__init__()
        self.user_tower = nn.Embedding(n_users, emb_dim)
        self.banner_tower = nn.Embedding(n_banners, emb_dim)
        self.embedding_dim = emb_dim

    def encode_user(self, user_ids):
        return self.user_tower(user_ids)

    def encode_banner(self, banner_ids):
        return self.banner_tower(banner_ids)

    def score_all_banners(self):
        return self.user_tower.weight @ self.banner_tower.weight.T

    def forward(self, user_ids, banner_ids):
        user_vec = self.encode_user(user_ids)
        banner_vec = self.encode_banner(banner_ids)
        return (user_vec * banner_vec).sum(dim=1)
