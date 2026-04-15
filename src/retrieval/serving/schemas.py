from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor
import torch


@dataclass(frozen=True)
class RetrievalRuntime:
    artifact_dir: str
    model: Any
    user2idx: dict[str, int]
    item2idx: dict[str, int]
    idx2item: dict[int, str]
    embedding_dim: int
    device: torch.device
    item_embeddings: Tensor
    metadata: dict[str, Any]
    model_version: str


@dataclass(frozen=True)
class RetrievalCandidate:
    banner_id: str
    retrieval_rank: int
    retrieval_score: float


@dataclass(frozen=True)
class RetrievalResult:
    user_id: str
    source: str
    model_version: str
    items: list[RetrievalCandidate]


@dataclass
class RetrievalState:
    active_banner_ids: set[str]
    popular_banner_scores: list[tuple[str, float]]
    seen_banner_history: dict[str, set[str]]
