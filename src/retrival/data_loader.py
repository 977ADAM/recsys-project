from pathlib import Path

import pandas as pd
from rich.console import Console
import torch

console = Console()

def load_data(path: Path, TRAIN_END: pd.Timestamp, VALID_END: pd.Timestamp):
    console.print(f"Загружаем датасет {path}")
    df = pd.read_csv(path, parse_dates=["event_date"])

    train_df = df[df["event_date"] <= TRAIN_END].copy()
    valid_df = df[(df["event_date"] > TRAIN_END) & (df["event_date"] <= VALID_END)].copy()
    test_df = df[df["event_date"] > VALID_END].copy()


    user_codes, user_index = pd.factorize(df["user_id"])

    banner_codes, banner_index = pd.factorize(df["banner_id"])

    labels = (df["clicks"] > 0).astype("float32")

    users = torch.tensor(user_codes, dtype=torch.long)
    banners = torch.tensor(banner_codes, dtype=torch.long)
    labels = torch.tensor(labels.to_numpy(), dtype=torch.float32)

    positive_df = df.loc[df["clicks"] > 0, ["user_id", "banner_id"]].copy()

    positive_df["user_idx"] = pd.Categorical(
        positive_df["user_id"], categories=user_index
    ).codes

    positive_df["banner_idx"] = pd.Categorical(
        positive_df["banner_id"], categories=banner_index
    ).codes

    positive_df = positive_df.loc[
        (positive_df["user_idx"] >= 0) & (positive_df["banner_idx"] >= 0),
        ["user_idx", "banner_idx"],
    ]

    return {
        "users": users,
        "banners": banners,
        "labels": labels,
        "n_users": len(user_index),
        "n_banners": len(banner_index),
        "positive_pairs": positive_df,
    }
