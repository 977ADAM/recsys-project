from __future__ import annotations

import pandas as pd

from src.retrieval.data.ingest import load_interactions_frame
from src.retrieval.data.preprocess import (
    build_mappings,
    build_positive_pairs,
    encode_frame,
    filter_known_entities,
    build_negatives_pairs,
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

    train_frame = filter_known_entities(splits.train, user2idx, item2idx)
    train_pairs = build_negatives_pairs(
        train_frame,
        negatives_per_positive=3,
        seed=42,
    )
    train_users, train_banners, train_labels = encode_frame(train_pairs, user2idx, item2idx)
    train_positive_pairs = build_positive_pairs(train_frame, user2idx, item2idx)

    valid_encoded = _encode_split(splits.valid, user2idx, item2idx)
    test_encoded = _encode_split(splits.test, user2idx, item2idx)

    return PreparedRetrievalData(
        n_users=len(user2idx),
        n_banners=len(item2idx),
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        raw_splits=splits,
        train=EncodedSplit(
            frame=train_pairs,
            users=train_users,
            banners=train_banners,
            labels=train_labels,
            positive_pairs=train_positive_pairs,
        ),
        valid=valid_encoded,
        test=test_encoded,
    )
