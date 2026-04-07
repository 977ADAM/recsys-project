from __future__ import annotations

import random

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc

from pytorch_recsys.model import TwoTower, bpr_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def train_step(
    model: TwoTower,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad()

    user_idx = batch["user_idx"].to(device)
    pos_item_idx = batch["pos_item_idx"].to(device)
    neg_item_idx = batch["neg_item_idx"].to(device)
    weight = batch["weight"].to(device)

    user_vec = model.encode_user(user_idx)
    pos_item_vec = model.encode_item(pos_item_idx)
    neg_item_vec = model.encode_item(neg_item_idx)

    pos_scores = model.score(user_vec, pos_item_vec)
    neg_scores = model.score(user_vec, neg_item_vec)

    loss = bpr_loss(pos_scores, neg_scores, weight)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_epoch(
    model: TwoTower,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    losses = [train_step(model, batch, optimizer, device) for batch in train_loader]
    return float(np.mean(losses)) if losses else float("nan")
