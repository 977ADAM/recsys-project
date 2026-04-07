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

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

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

    def encode_user(self, user_idx: torch.Tensor) -> torch.Tensor:
        x = self.user_emb(user_idx)
        x = self.user_mlp(x)
        return F.normalize(x, dim=-1)

    def encode_item(self, item_idx: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(item_idx)
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
