from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc

from pytorch_recsys.data import build_user_history
from pytorch_recsys.model import TwoTower


@dataclass
class EvalResult:
    users: int
    precision_at_k: float
    recall_at_k: float
    map_at_k: float
    ndcg_at_k: float


def evaluate_topk(
    model: TwoTower,
    eval_pairs: pd.DataFrame,
    seen_history: dict[int, set[int]],
    num_items: int,
    device: torch.device,
    k: int,
) -> EvalResult:
    if eval_pairs.empty:
        return EvalResult(0, float("nan"), float("nan"), float("nan"), float("nan"))

    truth = build_user_history(eval_pairs)
    eval_users = sorted(truth.keys())

    model.eval()
    with torch.no_grad():
        # Считаем вектора всех баннеров один раз, чтобы не делать это на каждого user.
        all_items = torch.arange(num_items, device=device)
        item_vectors = model.encode_item(all_items)

        precisions: list[float] = []
        recalls: list[float] = []
        average_precisions: list[float] = []
        ndcgs: list[float] = []

        for user_idx in eval_users:
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
            user_vector = model.encode_user(user_tensor)
            scores = torch.matmul(user_vector, item_vectors.T).squeeze(0)

            seen_items = seen_history.get(user_idx, set())
            if seen_items:
                seen_tensor = torch.tensor(sorted(seen_items), dtype=torch.long, device=device)
                scores[seen_tensor] = -torch.inf

            top_items = torch.topk(scores, k=min(k, num_items)).indices.cpu().tolist()
            true_items = truth[user_idx]

            hits = [1 if item in true_items else 0 for item in top_items]
            hit_count = sum(hits)

            precisions.append(hit_count / len(top_items))
            recalls.append(hit_count / len(true_items))

            running_hits = 0
            ap_sum = 0.0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    running_hits += 1
                    ap_sum += running_hits / rank
            average_precisions.append(ap_sum / min(len(true_items), len(top_items)))

            dcg = sum(
                1.0 / np.log2(rank + 1)
                for rank, hit in enumerate(hits, start=1)
                if hit
            )
            ideal_hits = min(len(true_items), len(top_items))
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return EvalResult(
        users=len(eval_users),
        precision_at_k=float(np.mean(precisions)),
        recall_at_k=float(np.mean(recalls)),
        map_at_k=float(np.mean(average_precisions)),
        ndcg_at_k=float(np.mean(ndcgs)),
    )


def print_eval(split_name: str, result: EvalResult, k: int) -> None:
    print(f"{split_name} users: {result.users}")
    print(f"{split_name} Precision@{k}: {result.precision_at_k:.6f}")
    print(f"{split_name} Recall@{k}:    {result.recall_at_k:.6f}")
    print(f"{split_name} MAP@{k}:       {result.map_at_k:.6f}")
    print(f"{split_name} NDCG@{k}:      {result.ndcg_at_k:.6f}")
