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
    hard_negative_ratio: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "banner_id", "label"])

    if negatives_per_positive <= 0:
        return pd.DataFrame(columns=["user_id", "banner_id", "label"])
    if not 0.0 <= hard_negative_ratio <= 1.0:
        raise ValueError("hard_negative_ratio must be between 0.0 and 1.0.")

    rng = np.random.default_rng(seed)

    positives = (
        frame.loc[frame["clicks"] > 0, ["user_id", "banner_id"]]
        .drop_duplicates()
        .copy()
    )
    positives["label"] = 1.0

    all_banners = frame["banner_id"].drop_duplicates().astype(str).to_numpy()
    banner_popularity = (
        frame.groupby("banner_id", as_index=False)
        .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        .sort_values(["clicks", "impressions", "banner_id"], ascending=[False, False, True])
    )
    popular_banners = banner_popularity["banner_id"].astype(str).tolist()

    user_seen = (
        frame.loc[frame["clicks"] > 0, ["user_id", "banner_id"]]
        .drop_duplicates()
        .groupby("user_id")["banner_id"]
        .agg(lambda x: set(x.astype(str)))
        .to_dict()
    )
    hard_negative_pool = (
        frame.loc[
            (frame["impressions"] > 0) & (frame["clicks"] <= 0),
            ["user_id", "banner_id", "impressions"],
        ]
        .copy()
        .sort_values(["user_id", "impressions", "banner_id"], ascending=[True, False, True])
        .groupby("user_id")["banner_id"]
        .agg(lambda banners: list(dict.fromkeys(banners.astype(str))))
        .to_dict()
    )

    negative_rows: list[dict[str, object]] = []

    for row in positives.itertuples(index=False):
        user_id = str(row.user_id)
        clicked_banners = user_seen.get(user_id, set())

        selected_negatives: list[str] = []

        hard_candidates = [
            banner_id
            for banner_id in hard_negative_pool.get(user_id, [])
            if banner_id not in clicked_banners
        ]
        hard_target = min(
            len(hard_candidates),
            int(np.ceil(negatives_per_positive * hard_negative_ratio)),
        )
        if hard_target > 0:
            selected_negatives.extend(hard_candidates[:hard_target])

        selected_set = set(selected_negatives)
        remaining_budget = negatives_per_positive - len(selected_negatives)

        popular_candidates = [
            banner_id
            for banner_id in popular_banners
            if banner_id not in clicked_banners and banner_id not in selected_set
        ]
        if remaining_budget > 0 and popular_candidates:
            sample_size = min(remaining_budget, len(popular_candidates))
            sampled_popular = rng.choice(popular_candidates, size=sample_size, replace=False)
            sampled_popular = np.atleast_1d(sampled_popular).tolist()
            selected_negatives.extend(str(banner_id) for banner_id in sampled_popular)
            selected_set.update(str(banner_id) for banner_id in sampled_popular)

        remaining_budget = negatives_per_positive - len(selected_negatives)
        if remaining_budget > 0:
            random_candidates = [
                banner_id
                for banner_id in all_banners
                if banner_id not in clicked_banners and banner_id not in selected_set
            ]
            if random_candidates:
                sample_size = min(remaining_budget, len(random_candidates))
                sampled_random = rng.choice(random_candidates, size=sample_size, replace=False)
                sampled_random = np.atleast_1d(sampled_random).tolist()
                selected_negatives.extend(str(banner_id) for banner_id in sampled_random)

        for neg_banner in selected_negatives:
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
