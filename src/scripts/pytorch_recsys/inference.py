from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc

from src.scripts.pytorch_recsys.model import TwoTower


def load_retrieval_model(
    artifact_dir: str,
    device: torch.device | None = None,
) -> tuple[TwoTower, dict[str, int], dict[str, int], dict[int, str], int, torch.device]:
    artifact_path = Path(artifact_dir)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        artifact_path / "retrieval_model.pt",
        map_location=resolved_device,
    )

    user2idx = checkpoint["user2idx"]
    item2idx = checkpoint["item2idx"]
    idx2item = checkpoint["idx2item"]
    embedding_dim = int(checkpoint["embedding_dim"])

    # Размеры эмбеддингов берём из сохранённых словарей, чтобы восстановить модель один в один.
    model = TwoTower(
        n_users=len(user2idx),
        n_items=len(item2idx),
        emb_dim=embedding_dim,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, user2idx, item2idx, idx2item, embedding_dim, resolved_device


def load_item_embeddings(
    artifact_dir: str,
    model: TwoTower,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    artifact_path = Path(artifact_dir)
    embeddings_path = artifact_path / "item_embeddings.npy"

    if embeddings_path.exists():
        item_embeddings = torch.from_numpy(np.load(embeddings_path))
        return item_embeddings.to(device)

    with torch.no_grad():
        all_items = torch.arange(num_items, dtype=torch.long, device=device)
        return model.encode_item(all_items)


def load_seen_items(
    interactions_csv: str,
    user_id: str,
    item2idx: dict[str, int],
) -> set[int]:
    interactions = pd.read_csv(interactions_csv)
    seen_banner_ids = set(interactions.loc[interactions["user_id"] == user_id, "banner_id"].tolist())
    return {item2idx[banner_id] for banner_id in seen_banner_ids if banner_id in item2idx}


def recommend_top_n(
    artifact_dir: str,
    user_id: str,
    top_n: int,
    exclude_seen: bool = False,
    interactions_csv: str | None = None,
    device: torch.device | None = None,
) -> list[str]:
    model, user2idx, item2idx, idx2item, _, resolved_device = load_retrieval_model(
        artifact_dir=artifact_dir,
        device=device,
    )

    if user_id not in user2idx:
        raise ValueError(f"user_id={user_id!r} is missing from retrieval vocabulary")

    with torch.no_grad():
        user_tensor = torch.tensor([user2idx[user_id]], dtype=torch.long, device=resolved_device)
        user_vector = model.encode_user(user_tensor)
        item_vectors = load_item_embeddings(
            artifact_dir=artifact_dir,
            model=model,
            num_items=len(item2idx),
            device=resolved_device,
        )
        scores = torch.matmul(user_vector, item_vectors.T).squeeze(0)

        if exclude_seen:
            if interactions_csv is None:
                raise ValueError("exclude_seen=True requires interactions_csv")

            seen_items = load_seen_items(interactions_csv, user_id, item2idx)
            if seen_items:
                seen_tensor = torch.tensor(sorted(seen_items), dtype=torch.long, device=resolved_device)
                scores[seen_tensor] = -torch.inf

        top_indices = torch.topk(scores, k=min(top_n, len(item2idx))).indices.cpu().tolist()
        return [idx2item[item_idx] for item_idx in top_indices]
