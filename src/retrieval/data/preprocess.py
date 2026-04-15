from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.retrieval.data.ingest import load_banners_frame


def build_mappings(
    train_df: pd.DataFrame,
    banners_csv: str | Path | None = None,
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    if banners_csv is not None:
        banners = load_banners_frame(banners_csv)
        banner_ids = banners["banner_id"].drop_duplicates().astype(str).tolist()
    else:
        banner_ids = train_df["banner_id"].drop_duplicates().astype(str).tolist()

    user_ids = train_df["user_id"].drop_duplicates().astype(str).tolist()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item2idx = {banner_id: idx for idx, banner_id in enumerate(banner_ids)}
    idx2item = {idx: banner_id for banner_id, idx in item2idx.items()}
    return user2idx, item2idx, idx2item


def filter_known_entities(
    frame: pd.DataFrame,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> pd.DataFrame:
    return frame[
        frame["user_id"].isin(user2idx) & frame["banner_id"].isin(item2idx)
    ].copy()


def encode_frame(
    frame: pd.DataFrame,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if frame.empty:
        empty_users = torch.empty(0, dtype=torch.long)
        empty_banners = torch.empty(0, dtype=torch.long)
        empty_labels = torch.empty(0, dtype=torch.float32)
        return empty_users, empty_banners, empty_labels

    users = torch.tensor(frame["user_id"].map(user2idx).to_numpy(), dtype=torch.long)
    banners = torch.tensor(frame["banner_id"].map(item2idx).to_numpy(), dtype=torch.long)

    if "label" in frame.columns:
        label_values = frame["label"].astype("float32").to_numpy()
    else:
        label_values = (frame["clicks"] > 0).astype("float32").to_numpy()

    labels = torch.tensor(label_values, dtype=torch.float32)
    return users, banners, labels


def build_positive_pairs(
    frame: pd.DataFrame,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> pd.DataFrame:
    positive_df = frame.loc[frame["clicks"] > 0, ["user_id", "banner_id"]].copy()
    if positive_df.empty:
        return pd.DataFrame(columns=["user_idx", "banner_idx"])

    positive_df["user_idx"] = positive_df["user_id"].map(user2idx)
    positive_df["banner_idx"] = positive_df["banner_id"].map(item2idx)
    positive_df = positive_df.dropna(subset=["user_idx", "banner_idx"])
    positive_df["user_idx"] = positive_df["user_idx"].astype(int)
    positive_df["banner_idx"] = positive_df["banner_idx"].astype(int)
    return positive_df[["user_idx", "banner_idx"]]



def build_negatives_pairs(
    frame: pd.DataFrame,
    *,
    negatives_per_positive: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "banner_id", "label"])
    
    rng = np.random.default_rng(seed)

    positives = (
        frame.loc[frame["clicks"] > 0, ["user_id", "banner_id"]]
        .drop_duplicates()
        .copy()
    )
    positives["label"] = 1.0

    all_banners = frame["banner_id"].drop_duplicates().astype(str).to_numpy()

    user_seen = (
        frame.loc[frame["clicks"] > 0, ["user_id", "banner_id"]]
        .drop_duplicates()
        .groupby("user_id")["banner_id"]
        .agg(lambda x: set(x.astype(str)))
        .to_dict()
    )

    negative_rows: list[dict[str, object]] = []

    for row in positives.itertuples(index=False):
        user_id = str(row.user_id)
        pos_banner = str(row.banner_id)
        seen = user_seen.get(user_id, set())

        candidate_pool = [b for b in all_banners if b not in seen]
        if not candidate_pool:
            continue

        sample_size = min(negatives_per_positive, len(candidate_pool))
        sampled_negatives = rng.choice(candidate_pool, size=sample_size, replace=False)

        for neg_banner in sampled_negatives:
            negative_rows.append(
                {
                    "user_id": user_id,
                    "banner_id": str(neg_banner),
                    "label": 0.0,
                }
            )

    negatives = pd.DataFrame(negative_rows, columns=["user_id", "banner_id", "label"])

    result = pd.concat(
        [
            positives[["user_id", "banner_id", "label"]],
            negatives,
        ],
        ignore_index=True,
    )

    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)