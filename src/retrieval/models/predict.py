from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from src.retrieval.data.ingest import load_interactions_frame, resolve_dataset_path
from src.retrieval.data.preprocess import (
    build_mappings,
    encode_frame,
    filter_known_entities,
)
from src.retrieval.models.export import load_saved_runtime
from src.retrieval.models.train import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LR,
    DEFAULT_RUNTIME_EPOCHS,
    DEFAULT_TOWER_DROPOUT,
    DEFAULT_TOWER_HIDDEN_DIMS,
    train_model,
)
from src.retrieval.models.two_tower import TwoTower


@lru_cache(maxsize=8)
def _train_runtime(artifact_dir: str) -> tuple[TwoTower, dict[str, int], dict[str, int], dict[int, str]]:
    try:
        return load_saved_runtime(artifact_dir)
    except FileNotFoundError:
        pass

    interactions_csv = resolve_dataset_path("banner_interactions.csv", artifact_dir)
    banners_csv = resolve_dataset_path("banners.csv", artifact_dir)
    interactions = load_interactions_frame(interactions_csv, require_event_date=False)
    user2idx, item2idx, idx2item = build_mappings(interactions, str(banners_csv))

    filtered = filter_known_entities(interactions, user2idx, item2idx)
    user_ids, banner_ids, labels = encode_frame(filtered, user2idx, item2idx)
    if user_ids.numel() == 0:
        raise ValueError("Retrieval training dataset is empty after applying user/banner mappings.")

    torch.manual_seed(42)
    model = TwoTower(
        n_users=len(user2idx),
        n_banners=len(item2idx),
        emb_dim=DEFAULT_EMBEDDING_DIM,
        hidden_dims=DEFAULT_TOWER_HIDDEN_DIMS,
        dropout=DEFAULT_TOWER_DROPOUT,
    )
    train_model(
        model,
        user_ids,
        banner_ids,
        labels,
        epochs=DEFAULT_RUNTIME_EPOCHS,
        lr=DEFAULT_LR,
    )
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
    return model.encode_all_banners().detach().to(device)


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
    interactions = load_interactions_frame(interactions_path, require_event_date=False)

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
            model.encode_all_banners().detach().to(device).T,
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

