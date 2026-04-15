from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()


def build_mappings(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    banners = pd.read_csv("./data/db/banners.csv")

    user_ids = train_df["user_id"].drop_duplicates().tolist()
    banner_ids = banners["banner_id"].drop_duplicates().tolist()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item2idx = {banner_id: idx for idx, banner_id in enumerate(banner_ids)}
    idx2item = {idx: banner_id for banner_id, idx in item2idx.items()}
    return user2idx, item2idx, idx2item