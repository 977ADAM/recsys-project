from __future__ import annotations

import pandas as pd

from src.retrieval.data.ingest import load_interactions_frame
from src.retrieval.data.preprocess import (
    build_mappings,
    build_positive_pairs,
    encode_frame,
    filter_known_entities,
)
from src.retrieval.data.split import RetrievalDataSplits, split_interactions_by_time
from src.retrieval.pipeline.registry import (
    EncodedSplit,
    PreparedRetrievalData,
    RetrievalDataConfig,
)

def run_ingest_stage(config: RetrievalDataConfig) -> pd.DataFrame:
    return load_interactions_frame(config.interactions_path, require_event_date=True)


def run_split_stage(
    interactions: pd.DataFrame,
    config: RetrievalDataConfig,
) -> RetrievalDataSplits:
    return split_interactions_by_time(interactions, config.train_end, config.valid_end)


def _encode_split(
    frame: pd.DataFrame,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> EncodedSplit:
    encoded_frame = filter_known_entities(frame, user2idx, item2idx)
    users, banners, labels = encode_frame(encoded_frame, user2idx, item2idx)
    positive_pairs = build_positive_pairs(encoded_frame, user2idx, item2idx)
    return EncodedSplit(
        frame=encoded_frame,
        users=users,
        banners=banners,
        labels=labels,
        positive_pairs=positive_pairs,
    )


def run_preprocess_stage(
    splits: RetrievalDataSplits,
    config: RetrievalDataConfig,
) -> PreparedRetrievalData:
    user2idx, item2idx, idx2item = build_mappings(splits.train, config.banners_path)

    return PreparedRetrievalData(
        n_users=len(user2idx),
        n_banners=len(item2idx),
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        raw_splits=splits,
        train=_encode_split(splits.train, user2idx, item2idx),
        valid=_encode_split(splits.valid, user2idx, item2idx),
        test=_encode_split(splits.test, user2idx, item2idx),
    )
