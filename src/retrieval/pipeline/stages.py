from __future__ import annotations

import pandas as pd

from src.retrieval.data.ingest import load_interactions_frame
from src.retrieval.data.split import RetrievalDataSplits, split_interactions_by_time
from src.retrieval.pipeline.orchestrator import prepare_training_data
from src.retrieval.pipeline.registry import PreparedRetrievalData, RetrievalDataConfig

def run_ingest_stage(config: RetrievalDataConfig) -> pd.DataFrame:
    return load_interactions_frame(config.interactions_path, require_event_date=True)


def run_split_stage(
    interactions: pd.DataFrame,
    config: RetrievalDataConfig,
) -> RetrievalDataSplits:
    return split_interactions_by_time(interactions, config.train_end, config.valid_end)


def run_preprocess_stage(
    splits: RetrievalDataSplits,
    config: RetrievalDataConfig,
) -> PreparedRetrievalData:
    del splits
    return prepare_training_data(config)
