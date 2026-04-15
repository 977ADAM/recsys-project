from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.retrieval.datasets import resolve_dataset_path
from src.retrieval.preprocessing import build_mappings
from src.retrieval.twotower_minimal import TwoTower

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_EPOCHS = 25
DEFAULT_LR = 1e-2


def _load_interactions_frame(interactions_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(interactions_csv, parse_dates=["event_date"])
    df["user_id"] = df["user_id"].astype(str)
    df["banner_id"] = df["banner_id"].astype(str)
    return df


def _load_saved_runtime(
    artifact_dir: str,
) -> tuple[TwoTower, dict[str, int], dict[str, int], dict[int, str]]:
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "model.pt"
    mappings_path = artifact_path / "mappings.json"
    if not model_path.exists() or not mappings_path.exists():
        raise FileNotFoundError("Saved retrieval artifacts were not found.")

    checkpoint = torch.load(model_path, map_location="cpu")
    with mappings_path.open("r", encoding="utf-8") as file_obj:
        mappings = json.load(file_obj)

    user2idx = {str(user_id): int(idx) for user_id, idx in mappings["user2idx"].items()}
    item2idx = {str(item_id): int(idx) for item_id, idx in mappings["item2idx"].items()}
    idx2item = {int(idx): str(item_id) for idx, item_id in mappings["idx2item"].items()}

    model = TwoTower(
        n_users=int(checkpoint["n_users"]),
        n_banners=int(checkpoint["n_banners"]),
        emb_dim=int(checkpoint["embedding_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, user2idx, item2idx, idx2item


@lru_cache(maxsize=8)
def _train_runtime(artifact_dir: str) -> tuple[TwoTower, dict[str, int], dict[str, int], dict[int, str]]:
    try:
        return _load_saved_runtime(artifact_dir)
    except FileNotFoundError:
        pass

    interactions_csv = resolve_dataset_path("banner_interactions.csv", artifact_dir)
    banners_csv = resolve_dataset_path("banners.csv", artifact_dir)
    interactions = _load_interactions_frame(interactions_csv)
    user2idx, item2idx, idx2item = build_mappings(interactions, str(banners_csv))

    filtered = interactions[
        interactions["user_id"].isin(user2idx) & interactions["banner_id"].isin(item2idx)
    ].copy()
    if filtered.empty:
        raise ValueError("Retrieval training dataset is empty after applying user/banner mappings.")

    user_ids = torch.tensor(
        filtered["user_id"].map(user2idx).to_numpy(),
        dtype=torch.long,
    )
    banner_ids = torch.tensor(
        filtered["banner_id"].map(item2idx).to_numpy(),
        dtype=torch.long,
    )
    labels = torch.tensor(
        (filtered["clicks"] > 0).astype("float32").to_numpy(),
        dtype=torch.float32,
    )

    torch.manual_seed(42)
    model = TwoTower(
        n_users=len(user2idx),
        n_banners=len(item2idx),
        emb_dim=DEFAULT_EMBEDDING_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(DEFAULT_EPOCHS):
        logits = model(user_ids, banner_ids)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model, user2idx, item2idx, idx2item


def load_retrieval_model(
    artifact_dir: str,
    device: torch.device | None = None,
):
    device = device or torch.device("cpu")
    model, user2idx, item2idx, idx2item = _train_runtime(artifact_dir)
    model = model.to(device)
    model.eval()
    return model, user2idx, item2idx, idx2item, model.embedding_dim, device


def reset_runtime_caches() -> None:
    _train_runtime.cache_clear()


def load_item_embeddings(
    artifact_dir: str,
    model: TwoTower,
    num_items: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    del artifact_dir, num_items
    device = device or torch.device("cpu")
    return model.banner_tower.weight.detach().to(device)


def recommend_top_n(
    artifact_dir: str,
    user_id: str,
    top_n: int,
    exclude_seen: bool = False,
    interactions_csv: str | None = None,
) -> list[str]:
    model, user2idx, item2idx, idx2item, _, device = load_retrieval_model(
        artifact_dir=artifact_dir,
        device=torch.device("cpu"),
    )

    interactions_path = (
        Path(interactions_csv)
        if interactions_csv is not None
        else resolve_dataset_path("banner_interactions.csv", artifact_dir)
    )
    interactions = _load_interactions_frame(interactions_path)

    if user_id not in user2idx:
        popularity = (
            interactions.groupby("banner_id", as_index=False)[["clicks", "impressions"]]
            .sum()
            .sort_values(["clicks", "impressions", "banner_id"], ascending=[False, False, True])
        )
        return popularity["banner_id"].astype(str).head(top_n).tolist()

    with torch.no_grad():
        user_tensor = torch.tensor([user2idx[user_id]], dtype=torch.long, device=device)
        scores = torch.matmul(
            model.encode_user(user_tensor),
            model.banner_tower.weight.detach().to(device).T,
        ).squeeze(0)

        if exclude_seen:
            seen_banner_ids = set(
                interactions.loc[interactions["user_id"] == user_id, "banner_id"].astype(str)
            )
            seen_indices = sorted(
                item2idx[banner_id]
                for banner_id in seen_banner_ids
                if banner_id in item2idx
            )
            if seen_indices:
                scores[torch.tensor(seen_indices, dtype=torch.long, device=device)] = -torch.inf

        k = min(top_n, scores.numel())
        top_indices = torch.topk(scores, k=k).indices.cpu().tolist()
    return [idx2item[item_idx] for item_idx in top_indices if item_idx in idx2item]
