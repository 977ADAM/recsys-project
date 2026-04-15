from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import torch

from src.retrieval.models.predict import (
    load_item_embeddings,
    load_retrieval_model,
    reset_runtime_caches,
)
from src.retrieval.serving.schemas import RetrievalRuntime


def _load_runtime_metadata(artifact_path: Path) -> dict[str, object]:
    metadata_path = artifact_path / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    return {"model_version": "pytorch_retrieval"}


@lru_cache(maxsize=8)
def load_runtime(artifact_dir: str | Path) -> RetrievalRuntime:
    artifact_path = Path(artifact_dir)
    metadata = _load_runtime_metadata(artifact_path)

    model, user2idx, item2idx, idx2item, embedding_dim, device = load_retrieval_model(
        artifact_dir=str(artifact_path),
        device=torch.device("cpu"),
    )
    item_embeddings = load_item_embeddings(
        artifact_dir=str(artifact_path),
        model=model,
        num_items=len(item2idx),
        device=device,
    )

    model_version = str(metadata.get("model_version") or artifact_path.name)
    return RetrievalRuntime(
        artifact_dir=str(artifact_path),
        model=model,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        embedding_dim=embedding_dim,
        device=device,
        item_embeddings=item_embeddings,
        metadata=metadata,
        model_version=model_version,
    )


def clear_runtime_caches() -> None:
    reset_runtime_caches()
    load_runtime.cache_clear()


