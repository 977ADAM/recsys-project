from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.retrieval.models.two_tower import TwoTower
from src.retrieval.pipeline.registry import MODEL_VERSION, PreparedRetrievalData
from src.retrieval.utils.common import normalize_project_path, project_root_from

PROJECT_ROOT = project_root_from(__file__)


def save_artifacts(
    output_dir: Path,
    model: TwoTower,
    data: PreparedRetrievalData,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "tower_hidden_dims": list(model.hidden_dims),
            "tower_dropout": model.dropout,
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
                "artifact_dir": normalize_project_path(output_dir, PROJECT_ROOT),
                "embedding_dim": model.embedding_dim,
                "tower_hidden_dims": list(model.hidden_dims),
                "tower_dropout": model.dropout,
                "train_end": config["train_end"],
                "valid_end": config["valid_end"],
                "latest_event_date": data.latest_event_date,
                "train_rows": data.train.rows,
                "valid_rows": data.valid.rows,
                "test_rows": data.test.rows,
                "validation_metrics": metrics,
                "training_data": {
                    "interactions_csv": normalize_project_path(config["data_path"], PROJECT_ROOT),
                },
                "training_config": {
                    "embedding_dim": config["emb_dim"],
                    "tower_hidden_dims": list(model.hidden_dims),
                    "tower_dropout": model.dropout,
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


def load_saved_runtime(
    artifact_dir: str | Path,
) -> tuple[TwoTower, dict[str, int], dict[str, int], dict[int, str]]:
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "model.pt"
    mappings_path = artifact_path / "mappings.json"
    if not model_path.exists() or not mappings_path.exists():
        raise FileNotFoundError("Saved retrieval artifacts were not found.")

    checkpoint = torch.load(model_path, map_location="cpu")
    with mappings_path.open("r", encoding="utf-8") as file_obj:
        mappings = json.load(file_obj)

    user2idx = {str(user_id): int(idx) for user_id, idx in mappings["user2idx"].items()}
    item2idx = {str(item_id): int(idx) for item_id, idx in mappings["item2idx"].items()}
    idx2item = {int(idx): str(item_id) for idx, item_id in mappings["idx2item"].items()}

    model = TwoTower(
        n_users=int(checkpoint["n_users"]),
        n_banners=int(checkpoint["n_banners"]),
        emb_dim=int(checkpoint["embedding_dim"]),
        hidden_dims=tuple(checkpoint.get("tower_hidden_dims", ())),
        dropout=float(checkpoint.get("tower_dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, user2idx, item2idx, idx2item


