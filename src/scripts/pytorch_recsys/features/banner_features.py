from __future__ import annotations

import numpy as np
import pandas as pd


BANNER_CATEGORICAL_FEATURES = [
    "brand",
    "category",
    "subcategory",
    "banner_format",
    "campaign_goal",
    "target_gender",
    "is_active",
]

BANNER_NUMERICAL_FEATURES = [
    "target_age_min",
    "target_age_max",
    "cpm_bid",
    "quality_score",
    "banner_age_days",
]

BANNER_DERIVED_FEATURES = [
    "target_age_bucket",
    "bid_bucket",
    "quality_bucket",
]


def build_banner_feature_frame(banners_csv_path: str = "./data/db/banners.csv") -> pd.DataFrame:
    banners = pd.read_csv(banners_csv_path, parse_dates=["created_at"]).copy()

    banners["is_active"] = banners["is_active"].fillna(0).astype("int64").astype(str)
    banners["target_age_min"] = banners["target_age_min"].fillna(18).astype("int64")
    banners["target_age_max"] = banners["target_age_max"].fillna(65).astype("int64")
    banners["cpm_bid"] = banners["cpm_bid"].fillna(banners["cpm_bid"].median()).astype("float32")
    banners["quality_score"] = (
        banners["quality_score"].fillna(banners["quality_score"].median()).astype("float32")
    )

    max_created_at = banners["created_at"].max()
    banners["banner_age_days"] = (max_created_at - banners["created_at"]).dt.days.astype("int64")
    banners["target_age_bucket"] = (
        banners["target_age_min"].astype(str) + "_" + banners["target_age_max"].astype(str)
    )
    banners["bid_bucket"] = pd.qcut(
        banners["cpm_bid"],
        q=4,
        labels=["low_bid", "mid_bid", "high_bid", "top_bid"],
        duplicates="drop",
    ).astype(str)
    banners["quality_bucket"] = pd.qcut(
        banners["quality_score"],
        q=4,
        labels=["low_quality", "mid_quality", "high_quality", "top_quality"],
        duplicates="drop",
    ).astype(str)

    selected_columns = (
        ["banner_id"]
        + BANNER_CATEGORICAL_FEATURES
        + BANNER_NUMERICAL_FEATURES
        + BANNER_DERIVED_FEATURES
    )
    return banners[selected_columns].copy()


def build_banner_feature_matrix(
    banner_ids: list[str],
    banners_csv_path: str = "./data/db/banners.csv",
) -> np.ndarray:
    banners = build_banner_feature_frame(banners_csv_path)
    banners = banners.set_index("banner_id").reindex(banner_ids)

    categorical = pd.get_dummies(
        banners[BANNER_CATEGORICAL_FEATURES + BANNER_DERIVED_FEATURES].fillna("unknown").astype(str),
        dummy_na=False,
    )
    numerical = banners[BANNER_NUMERICAL_FEATURES].astype("float32")
    numerical = (numerical - numerical.mean()) / numerical.std().replace(0, 1)
    features = pd.concat([categorical, numerical.fillna(0.0)], axis=1)
    return features.to_numpy(dtype=np.float32, copy=True)
