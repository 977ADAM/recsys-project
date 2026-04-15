from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.retrieval.pipeline.registry import PreparedRetrievalData, RetrievalDataConfig
from src.retrieval.pipeline.stages import (
    run_ingest_stage,
    run_preprocess_stage,
    run_split_stage,
)


def prepare_training_data(config: RetrievalDataConfig) -> PreparedRetrievalData:
    interactions = run_ingest_stage(config)
    splits = run_split_stage(interactions, config)
    return run_preprocess_stage(splits, config)


def load_data(
    path: Path,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    banners_csv: str | Path | None = None,
) -> dict[str, object]:
    config = RetrievalDataConfig(
        interactions_path=Path(path),
        train_end=train_end,
        valid_end=valid_end,
        banners_path=Path(banners_csv) if banners_csv is not None else None,
    )
    return prepare_training_data(config).as_dict()
