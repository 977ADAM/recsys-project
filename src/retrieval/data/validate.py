from __future__ import annotations

import pandas as pd

REQUIRED_INTERACTIONS_COLUMNS = frozenset({"user_id", "banner_id", "clicks", "impressions"})
REQUIRED_BANNERS_COLUMNS = frozenset({"banner_id"})


def _missing_columns(df: pd.DataFrame, required_columns: set[str]) -> list[str]:
    return sorted(column for column in required_columns if column not in df.columns)


def validate_interactions_frame(
    df: pd.DataFrame,
    *,
    require_event_date: bool = True,
) -> None:
    required_columns = set(REQUIRED_INTERACTIONS_COLUMNS)
    if require_event_date:
        required_columns.add("event_date")

    missing_columns = _missing_columns(df, required_columns)
    if missing_columns:
        raise ValueError(
            "Interactions dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if require_event_date and df["event_date"].isna().any():
        raise ValueError("Interactions dataset contains invalid event_date values.")


def validate_banners_frame(df: pd.DataFrame) -> None:
    missing_columns = _missing_columns(df, set(REQUIRED_BANNERS_COLUMNS))
    if missing_columns:
        raise ValueError(
            "Banners dataset is missing required columns: "
            + ", ".join(missing_columns)
        )


def validate_split_boundaries(
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> None:
    if train_end > valid_end:
        raise ValueError("train_end must be less than or equal to valid_end.")
