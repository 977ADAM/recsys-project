from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.retrieval.data.ingest import load_interactions_frame
from src.retrieval.data.preprocess import (
    build_binary_interactions,
    build_mappings,
    build_positive_pairs,
    encode_frame,
    filter_known_entities,
)
from src.retrieval.data.split import split_interactions_by_time
from src.retrieval.pipeline.registry import (
    EncodedSplit,
    PreparedRetrievalData,
    RetrievalDataConfig,
)


def _prepare_split(
    frame: pd.DataFrame,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> EncodedSplit:
    prepared_frame = build_binary_interactions(filter_known_entities(frame, user2idx, item2idx))
    users, banners, labels = encode_frame(prepared_frame, user2idx, item2idx)
    positive_pairs = build_positive_pairs(prepared_frame, user2idx, item2idx)
    return EncodedSplit(
        frame=prepared_frame,
        users=users,
        banners=banners,
        labels=labels,
        positive_pairs=positive_pairs,
    )


def prepare_training_data(config: RetrievalDataConfig) -> PreparedRetrievalData:
    interactions = load_interactions_frame(config.interactions_path, require_event_date=True)
    raw_splits = split_interactions_by_time(
        interactions,
        config.train_end,
        config.valid_end,
    )

    user2idx, item2idx, idx2item = build_mappings(raw_splits.train, config.banners_path)
    train = _prepare_split(raw_splits.train, user2idx, item2idx)
    valid = _prepare_split(raw_splits.valid, user2idx, item2idx)
    test = _prepare_split(raw_splits.test, user2idx, item2idx)

    return PreparedRetrievalData(
        n_users=len(user2idx),
        n_banners=len(item2idx),
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        raw_splits=raw_splits,
        train=train,
        valid=valid,
        test=test,
    )


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
