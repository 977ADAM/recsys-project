from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from src.retrieval.data.split import RetrievalDataSplits
from src.retrieval.utils.common import project_root_from

PROJECT_ROOT = project_root_from(__file__)
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pytorch_retrieval"
DEFAULT_TRAIN_END = pd.Timestamp("2026-02-28")
DEFAULT_VALID_END = pd.Timestamp("2026-03-15")
MODEL_VERSION = "pytorch_retrieval"


@dataclass(frozen=True)
class RetrievalDataConfig:
    interactions_path: Path
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    banners_path: Path | None = None


@dataclass(frozen=True)
class EncodedSplit:
    frame: pd.DataFrame
    users: torch.Tensor
    banners: torch.Tensor
    labels: torch.Tensor
    positive_pairs: pd.DataFrame

    @property
    def rows(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class PreparedRetrievalData:
    n_users: int
    n_banners: int
    user2idx: dict[str, int]
    item2idx: dict[str, int]
    idx2item: dict[int, str]
    raw_splits: RetrievalDataSplits
    train: EncodedSplit
    valid: EncodedSplit
    test: EncodedSplit

    @property
    def latest_event_date(self) -> str | None:
        for frame in (
            self.raw_splits.test,
            self.raw_splits.valid,
            self.raw_splits.train,
        ):
            if not frame.empty:
                return str(frame["event_date"].max().date())
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "users": self.train.users,
            "banners": self.train.banners,
            "labels": self.train.labels,
            "n_users": self.n_users,
            "n_banners": self.n_banners,
            "positive_pairs": self.train.positive_pairs,
            "user2idx": self.user2idx,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
            "train_df": self.raw_splits.train,
            "valid_df": self.raw_splits.valid,
            "test_df": self.raw_splits.test,
            "train_users": self.train.users,
            "train_banners": self.train.banners,
            "train_labels": self.train.labels,
            "valid_users": self.valid.users,
            "valid_banners": self.valid.banners,
            "valid_labels": self.valid.labels,
            "test_users": self.test.users,
            "test_banners": self.test.banners,
            "test_labels": self.test.labels,
            "train_positive_pairs": self.train.positive_pairs,
            "valid_positive_pairs": self.valid.positive_pairs,
            "test_positive_pairs": self.test.positive_pairs,
            "train_rows": self.train.rows,
            "valid_rows": self.valid.rows,
            "test_rows": self.test.rows,
        }


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_TRAIN_END",
    "DEFAULT_VALID_END",
    "MODEL_VERSION",
    "RetrievalDataConfig",
    "EncodedSplit",
    "PreparedRetrievalData",
]
