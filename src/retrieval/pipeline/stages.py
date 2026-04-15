from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

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
    MODEL_VERSION,
    PreparedRetrievalData,
    PROJECT_ROOT,
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


def validate_training_ready_data(data: PreparedRetrievalData) -> None:
    if data.train.users.numel() == 0:
        raise ValueError("Training split is empty after encoding.")
    if data.valid.positive_pairs.empty:
        raise ValueError("Validation split has no positive pairs for recall@k evaluation.")


def recall_at_k(model: Any, positive_pairs: pd.DataFrame, k: int = 100) -> float:
    scores = model.score_all_banners()
    topk = scores.topk(k=min(k, scores.size(1)), dim=1).indices

    hits = 0
    total = 0
    for user_idx, group in positive_pairs.groupby("user_idx"):
        true_banners = set(group["banner_idx"].tolist())
        predicted = set(topk[user_idx].tolist())
        hits += len(true_banners & predicted)
        total += len(true_banners)

    return 0.0 if total == 0 else hits / total


def evaluate_recalls(model: Any, positive_pairs: pd.DataFrame, ks: list[int]) -> dict[str, float]:
    return {f"recall@{k}": round(recall_at_k(model, positive_pairs, k=k), 6) for k in ks}


def _normalize_metadata_path(path: str | Path) -> str:
    resolved_path = Path(path).resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved_path)


def save_artifacts(
    output_dir: Path,
    model: Any,
    data: PreparedRetrievalData,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "n_users": data.n_users,
            "n_banners": data.n_banners,
        },
        output_dir / "model.pt",
    )

    with (output_dir / "mappings.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "user2idx": data.user2idx,
                "item2idx": data.item2idx,
                "idx2item": {str(idx): banner_id for idx, banner_id in data.idx2item.items()},
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "model_version": MODEL_VERSION,
                "model_type": "two_tower",
                "artifact_dir": _normalize_metadata_path(output_dir),
                "embedding_dim": model.embedding_dim,
                "train_end": config["train_end"],
                "valid_end": config["valid_end"],
                "latest_event_date": data.latest_event_date,
                "train_rows": data.train.rows,
                "valid_rows": data.valid.rows,
                "test_rows": data.test.rows,
                "validation_metrics": metrics,
                "training_data": {
                    "interactions_csv": _normalize_metadata_path(config["data_path"]),
                },
                "training_config": {
                    "embedding_dim": config["emb_dim"],
                    "epochs": config["epochs"],
                    "learning_rate": config["lr"],
                    "random_seed": config["seed"],
                    "recall_k": config["recall_k"],
                },
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )


__all__ = [
    "run_ingest_stage",
    "run_split_stage",
    "run_preprocess_stage",
    "validate_training_ready_data",
    "recall_at_k",
    "evaluate_recalls",
    "save_artifacts",
]
