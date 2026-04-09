from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc

from pytorch_recsys.config import TRAIN_END, VALID_END


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(
        "./data/db/banner_interactions.csv",
        parse_dates=["event_date"],
    )

    # Делим данные по времени: модель видит только прошлое.
    train_df = interactions[interactions["event_date"] <= TRAIN_END].copy()
    valid_df = interactions[
        (interactions["event_date"] > TRAIN_END)
        & (interactions["event_date"] <= VALID_END)
    ].copy()
    test_df = interactions[interactions["event_date"] > VALID_END].copy()
    return train_df, valid_df, test_df


def build_mappings(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    banners = pd.read_csv("./data/db/banners.csv")

    user_ids = train_df["user_id"].drop_duplicates().tolist()
    banner_ids = banners["banner_id"].drop_duplicates().tolist()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item2idx = {banner_id: idx for idx, banner_id in enumerate(banner_ids)}
    idx2item = {idx: banner_id for banner_id, idx in item2idx.items()}
    return user2idx, item2idx, idx2item


def prepare_positive_pairs(
    frame: pd.DataFrame,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> pd.DataFrame:
    # Для implicit-feedback считаем позитивом только записи с кликом.
    positive = frame[frame["clicks"] > 0].copy()
    positive["user_idx"] = positive["user_id"].map(user2idx)
    positive["item_idx"] = positive["banner_id"].map(item2idx)
    positive = positive.dropna(subset=["user_idx", "item_idx"]).copy()
    positive["user_idx"] = positive["user_idx"].astype(np.int64)
    positive["item_idx"] = positive["item_idx"].astype(np.int64)

    grouped = (
        positive.groupby(["user_idx", "item_idx"], as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        )
        .copy()
    )

    # Вес усиливает пары, где кликов было больше.
    grouped["weight"] = grouped["clicks"] + 0.1 * grouped["impressions"]
    return grouped[["user_idx", "item_idx", "weight"]]


def build_user_history(pairs: pd.DataFrame) -> dict[int, set[int]]:
    history: dict[int, set[int]] = {}
    for row in pairs.itertuples(index=False):
        history.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    return history


class BPRDataset(Dataset):
    """Возвращает триплеты (user, positive item, negative item) для BPR-loss."""

    def __init__(
        self,
        positive_pairs: pd.DataFrame,
        user_history: dict[int, set[int]],
        num_items: int,
    ) -> None:
        self.user_idx = positive_pairs["user_idx"].to_numpy(dtype=np.int64)
        self.item_idx = positive_pairs["item_idx"].to_numpy(dtype=np.int64)
        self.weight = positive_pairs["weight"].to_numpy(dtype=np.float32)
        self.user_history = user_history
        self.num_items = num_items
        self.all_items = np.arange(num_items, dtype=np.int64)
        self.available_negatives = self._build_negative_pools()

    def __len__(self) -> int:
        return len(self.user_idx)

    def _build_negative_pools(self) -> dict[int, np.ndarray]:
        pools: dict[int, np.ndarray] = {}
        for user_idx, seen_items in self.user_history.items():
            candidate_items = np.setdiff1d(
                self.all_items,
                np.fromiter(seen_items, dtype=np.int64),
                assume_unique=False,
            )
            if len(candidate_items) == 0:
                raise ValueError(
                    f"user_idx={user_idx} has interactions with every item; "
                    "negative sampling is impossible"
                )
            pools[user_idx] = candidate_items
        return pools

    def _sample_negative(self, user_idx: int) -> int:
        candidate_items = self.available_negatives[user_idx]
        sampled_index = np.random.randint(0, len(candidate_items))
        return int(candidate_items[sampled_index])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_idx = int(self.user_idx[index])
        pos_item_idx = int(self.item_idx[index])
        neg_item_idx = self._sample_negative(user_idx)

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "pos_item_idx": torch.tensor(pos_item_idx, dtype=torch.long),
            "neg_item_idx": torch.tensor(neg_item_idx, dtype=torch.long),
            "weight": torch.tensor(self.weight[index], dtype=torch.float32),
        }
