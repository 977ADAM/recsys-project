from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc

from pytorch_recsys.model import TwoTower


def save_retrieval_artifacts(
    model: TwoTower,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
    idx2item: dict[int, str],
    embedding_dim: int,
    output_dir: str,
    save_item_embeddings: bool,
    device: torch.device,
) -> Path:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Чекпоинт хранит всё, что нужно, чтобы потом поднять retrieval-модель без retrain.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "embedding_dim": embedding_dim,
    }
    torch.save(checkpoint, artifact_dir / "retrieval_model.pt")

    metadata = {
        "embedding_dim": embedding_dim,
        "num_users": len(user2idx),
        "num_items": len(item2idx),
        "files": {
            "checkpoint": "retrieval_model.pt",
        },
    }

    if save_item_embeddings:
        model.eval()
        with torch.no_grad():
            all_items = torch.arange(len(item2idx), dtype=torch.long, device=device)
            item_embeddings = model.encode_item(all_items).detach().cpu().numpy().astype(np.float32)
        np.save(artifact_dir / "item_embeddings.npy", item_embeddings)
        metadata["files"]["item_embeddings"] = "item_embeddings.npy"

    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    return artifact_dir
