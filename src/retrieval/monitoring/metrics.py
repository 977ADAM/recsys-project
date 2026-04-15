from __future__ import annotations

from functools import lru_cache

from src.retrieval.data.ingest import load_banners_frame, load_interactions_frame


@lru_cache(maxsize=8)
def load_active_banner_ids(banners_csv: str) -> set[str]:
    banners = load_banners_frame(banners_csv)
    if "is_active" not in banners.columns:
        raise ValueError("banners_csv must contain banner_id and is_active columns.")
    active_ids = banners.loc[banners["is_active"] == 1, "banner_id"].astype(str)
    return set(active_ids.tolist())


@lru_cache(maxsize=8)
def load_popular_banner_scores(interactions_csv: str) -> list[tuple[str, float]]:
    interactions = load_interactions_frame(interactions_csv, require_event_date=False)
    if "clicks" not in interactions.columns:
        raise ValueError("interactions_csv must contain banner_id and clicks columns.")

    popularity = (
        interactions.groupby("banner_id", as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        )
        .sort_values(["clicks", "impressions", "banner_id"], ascending=[False, False, True])
    )
    return [
        (str(row.banner_id), float(row.clicks + 0.1 * row.impressions))
        for row in popularity.itertuples(index=False)
    ]


@lru_cache(maxsize=8)
def load_seen_banner_history(interactions_csv: str) -> dict[str, set[str]]:
    interactions = load_interactions_frame(interactions_csv, require_event_date=False)
    grouped = interactions.groupby("user_id")["banner_id"].agg(list)
    return {
        str(user_id): {str(banner_id) for banner_id in banner_ids}
        for user_id, banner_ids in grouped.items()
    }


def merge_seen_histories(*histories: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for history in histories:
        for user_id, banner_ids in history.items():
            merged.setdefault(str(user_id), set()).update(str(banner_id) for banner_id in banner_ids)
    return merged


def clear_state_metric_caches() -> None:
    load_active_banner_ids.cache_clear()
    load_popular_banner_scores.cache_clear()
    load_seen_banner_history.cache_clear()
