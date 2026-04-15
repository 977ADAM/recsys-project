from pathlib import Path

import pandas as pd
from rich.console import Console
import torch

from src.retrieval.preprocessing import build_mappings

console = Console()


def _encode_frame(
    frame: pd.DataFrame,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users = torch.tensor(frame["user_id"].map(user2idx).to_numpy(), dtype=torch.long)
    banners = torch.tensor(frame["banner_id"].map(item2idx).to_numpy(), dtype=torch.long)
    labels = torch.tensor((frame["clicks"] > 0).astype("float32").to_numpy(), dtype=torch.float32)
    return users, banners, labels


def _build_positive_pairs(
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


def load_data(path: Path, TRAIN_END: pd.Timestamp, VALID_END: pd.Timestamp):
    console.print(f"Загружаем датасет {path}")
    df = pd.read_csv(path, parse_dates=["event_date"])
    df["user_id"] = df["user_id"].astype(str)
    df["banner_id"] = df["banner_id"].astype(str)

    train_df = df[df["event_date"] <= TRAIN_END].copy()
    valid_df = df[(df["event_date"] > TRAIN_END) & (df["event_date"] <= VALID_END)].copy()
    test_df = df[df["event_date"] > VALID_END].copy()

    user2idx, item2idx, idx2item = build_mappings(train_df)

    train_encoded = train_df[
        train_df["user_id"].isin(user2idx) & train_df["banner_id"].isin(item2idx)
    ].copy()
    valid_encoded = valid_df[
        valid_df["user_id"].isin(user2idx) & valid_df["banner_id"].isin(item2idx)
    ].copy()
    test_encoded = test_df[
        test_df["user_id"].isin(user2idx) & test_df["banner_id"].isin(item2idx)
    ].copy()

    train_users, train_banners, train_labels = _encode_frame(train_encoded, user2idx, item2idx)
    valid_users, valid_banners, valid_labels = _encode_frame(valid_encoded, user2idx, item2idx)
    test_users, test_banners, test_labels = _encode_frame(test_encoded, user2idx, item2idx)

    return {
        "users": train_users,
        "banners": train_banners,
        "labels": train_labels,
        "n_users": len(user2idx),
        "n_banners": len(item2idx),
        "positive_pairs": _build_positive_pairs(train_encoded, user2idx, item2idx),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "train_users": train_users,
        "train_banners": train_banners,
        "train_labels": train_labels,
        "valid_users": valid_users,
        "valid_banners": valid_banners,
        "valid_labels": valid_labels,
        "test_users": test_users,
        "test_banners": test_banners,
        "test_labels": test_labels,
        "train_positive_pairs": _build_positive_pairs(train_encoded, user2idx, item2idx),
        "valid_positive_pairs": _build_positive_pairs(valid_encoded, user2idx, item2idx),
        "test_positive_pairs": _build_positive_pairs(test_encoded, user2idx, item2idx),
        "train_rows": len(train_encoded),
        "valid_rows": len(valid_encoded),
        "test_rows": len(test_encoded),
    }
