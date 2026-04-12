from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc


class TwoTower(nn.Module):
    """Отдельно кодирует пользователя и баннер в одно embedding-пространство."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        user_feature_dim: int = 0,
        item_feature_dim: int = 0,
        user_feature_table: torch.Tensor | None = None,
        item_feature_table: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_feature_proj = (
            nn.Linear(user_feature_dim, emb_dim) if user_feature_dim > 0 else None
        )
        self.item_feature_proj = (
            nn.Linear(item_feature_dim, emb_dim) if item_feature_dim > 0 else None
        )
        self.register_buffer("user_feature_table", user_feature_table)
        self.register_buffer("item_feature_table", item_feature_table)

        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        user_vec = self.encode_user(user_idx)
        item_vec = self.encode_item(item_idx)
        return self.score(user_vec, item_vec)

    def encode_user(self, user_idx: torch.Tensor) -> torch.Tensor:
        x = self.user_emb(user_idx)
        if self.user_feature_proj is not None and self.user_feature_table is not None:
            x = x + self.user_feature_proj(self.user_feature_table[user_idx])
        x = self.user_mlp(x)
        return F.normalize(x, dim=-1)

    def encode_item(self, item_idx: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(item_idx)
        if self.item_feature_proj is not None and self.item_feature_table is not None:
            x = x + self.item_feature_proj(self.item_feature_table[item_idx])
        x = self.item_mlp(x)
        return F.normalize(x, dim=-1)

    @staticmethod
    def score(user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        # Чем выше скалярное произведение, тем релевантнее баннер пользователю.
        return (user_vec * item_vec).sum(dim=-1)


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # BPR заставляет позитивный баннер иметь скор выше, чем негативный.
    loss = -F.logsigmoid(pos_scores - neg_scores)
    if weight is not None:
        loss = loss * weight
    return loss.mean()
