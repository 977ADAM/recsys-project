from __future__ import annotations

from typing import Any

import pandas as pd


def recall_at_k(model: Any, positive_pairs: pd.DataFrame, k: int = 100) -> float:
    scores = model.score_all_banners()
    topk = scores.topk(k=min(k, scores.size(1)), dim=1).indices

    hits = 0
    total = 0
    for user_idx, group in positive_pairs.groupby("user_idx"):
        true_banners = set(group["banner_idx"].tolist())
        predicted = set(topk[user_idx].tolist())
        hits += len(true_banners & predicted)
        total += len(true_banners)

    return 0.0 if total == 0 else hits / total


def evaluate_recalls(model: Any, positive_pairs: pd.DataFrame, ks: list[int]) -> dict[str, float]:
    return {f"recall@{k}": round(recall_at_k(model, positive_pairs, k=k), 6) for k in ks}


__all__ = ["recall_at_k", "evaluate_recalls"]
