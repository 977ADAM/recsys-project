from __future__ import annotations

import numpy as np
import torch

from src.retrieval.serving.schemas import RetrievalCandidate, RetrievalRuntime


def search_top_k(
    runtime: RetrievalRuntime,
    user_id: str,
    top_k: int,
    seen_items: set[int] | None = None,
    active_banner_ids: set[str] | None = None,
) -> list[RetrievalCandidate]:
    user_idx = runtime.user2idx[user_id]
    seen_items = seen_items or set()

    with torch.no_grad():
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=runtime.device)
        user_vector = runtime.model.encode_user(user_tensor)
        scores = torch.matmul(user_vector, runtime.item_embeddings.T).squeeze(0).clone()

        if seen_items:
            seen_tensor = torch.tensor(sorted(seen_items), dtype=torch.long, device=runtime.device)
            scores[seen_tensor] = -torch.inf

        if active_banner_ids is not None:
            inactive_indices = [
                item_idx
                for item_idx, banner_id in runtime.idx2item.items()
                if banner_id not in active_banner_ids
            ]
            if inactive_indices:
                inactive_tensor = torch.tensor(
                    inactive_indices,
                    dtype=torch.long,
                    device=runtime.device,
                )
                scores[inactive_tensor] = -torch.inf

        k = min(top_k, scores.numel())
        top_scores, top_indices = torch.topk(scores, k=k)

    items: list[RetrievalCandidate] = []
    for rank, (item_idx, score) in enumerate(
        zip(top_indices.cpu().tolist(), top_scores.cpu().tolist(), strict=False),
        start=1,
    ):
        if not np.isfinite(score):
            continue
        items.append(
            RetrievalCandidate(
                banner_id=runtime.idx2item[item_idx],
                retrieval_rank=rank,
                retrieval_score=float(score),
            )
        )
    return items


def build_fallback_candidates(
    popular_banner_scores: list[tuple[str, float]],
    top_k: int,
    active_banner_ids: set[str] | None = None,
) -> list[RetrievalCandidate]:
    items: list[RetrievalCandidate] = []
    for banner_id, score in popular_banner_scores:
        if active_banner_ids is not None and banner_id not in active_banner_ids:
            continue
        items.append(
            RetrievalCandidate(
                banner_id=banner_id,
                retrieval_rank=len(items) + 1,
                retrieval_score=score,
            )
        )
        if len(items) >= top_k:
            break
    return items


__all__ = ["search_top_k", "build_fallback_candidates"]
