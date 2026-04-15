from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.retrieval.data.validate import validate_split_boundaries


@dataclass(frozen=True)
class RetrievalDataSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_interactions_by_time(
    interactions: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> RetrievalDataSplits:
    validate_split_boundaries(train_end, valid_end)

    train_df = interactions[interactions["event_date"] <= train_end].copy()
    valid_df = interactions[
        (interactions["event_date"] > train_end) & (interactions["event_date"] <= valid_end)
    ].copy()
    test_df = interactions[interactions["event_date"] > valid_end].copy()

    return RetrievalDataSplits(
        train=train_df,
        valid=valid_df,
        test=test_df,
    )
