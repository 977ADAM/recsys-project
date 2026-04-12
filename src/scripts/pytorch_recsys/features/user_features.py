from __future__ import annotations

import numpy as np
import pandas as pd


USER_CATEGORICAL_FEATURES = [
    "gender",
    "city_tier",
    "device_os",
    "platform",
    "income_band",
    "activity_segment",
    "interest_1",
    "interest_2",
    "interest_3",
    "country",
    "is_premium",
]

USER_NUMERICAL_FEATURES = [
    "age",
    "signup_days_ago",
]

USER_DERIVED_FEATURES = [
    "age_bucket",
    "signup_recency_bucket",
]


def build_user_feature_frame(users_csv_path: str = "./data/db/users.csv") -> pd.DataFrame:
    users = pd.read_csv(users_csv_path).copy()

    users["age"] = users["age"].fillna(users["age"].median()).astype("int64")
    users["signup_days_ago"] = (
        users["signup_days_ago"].fillna(users["signup_days_ago"].median()).astype("int64")
    )
    users["is_premium"] = users["is_premium"].fillna(0).astype("int64").astype(str)

    users["age_bucket"] = pd.cut(
        users["age"],
        bins=[0, 24, 34, 44, 54, 120],
        labels=["18_24", "25_34", "35_44", "45_54", "55_plus"],
        include_lowest=True,
    ).astype(str)
    users["signup_recency_bucket"] = pd.cut(
        users["signup_days_ago"],
        bins=[-1, 30, 90, 180, 365, 10000],
        labels=["new_30d", "recent_90d", "warm_180d", "mature_365d", "loyal_365d_plus"],
    ).astype(str)

    selected_columns = (
        ["user_id"]
        + USER_CATEGORICAL_FEATURES
        + USER_NUMERICAL_FEATURES
        + USER_DERIVED_FEATURES
    )
    return users[selected_columns].copy()


def build_user_feature_matrix(
    user_ids: list[str],
    users_csv_path: str = "./data/db/users.csv",
) -> np.ndarray:
    users = build_user_feature_frame(users_csv_path)
    users = users.set_index("user_id").reindex(user_ids)

    categorical = pd.get_dummies(
        users[USER_CATEGORICAL_FEATURES + USER_DERIVED_FEATURES].fillna("unknown").astype(str),
        dummy_na=False,
    )
    numerical = users[USER_NUMERICAL_FEATURES].astype("float32")
    numerical = (numerical - numerical.mean()) / numerical.std().replace(0, 1)
    features = pd.concat([categorical, numerical.fillna(0.0)], axis=1)
    return features.to_numpy(dtype=np.float32, copy=True)
