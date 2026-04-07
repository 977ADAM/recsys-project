from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the current environment. "
        "Install it first, then rerun this script."
    ) from exc


TRAIN_END = pd.Timestamp("2026-02-28")
VALID_END = pd.Timestamp("2026-03-15")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(
        "./data/db/banner_interactions.csv",
        parse_dates=["event_date"],
    )

    train_df = interactions[interactions["event_date"] <= TRAIN_END].copy()
    valid_df = interactions[
        (interactions["event_date"] > TRAIN_END)
        & (interactions["event_date"] <= VALID_END)
    ].copy()
    test_df = interactions[interactions["event_date"] > VALID_END].copy()
    return train_df, valid_df, test_df


def build_mappings() -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    users = pd.read_csv("./data/db/users.csv")
    banners = pd.read_csv("./data/db/banners.csv")

    user_ids = users["user_id"].drop_duplicates().tolist()
    banner_ids = banners["banner_id"].drop_duplicates().tolist()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item2idx = {banner_id: idx for idx, banner_id in enumerate(banner_ids)}
    idx2item = {idx: banner_id for banner_id, idx in item2idx.items()}
    return user2idx, item2idx, idx2item


def prepare_positive_pairs(
    frame: pd.DataFrame,
    user2idx: dict[str, int],
    item2idx: dict[str, int],
) -> pd.DataFrame:
    positive = frame[frame["clicks"] > 0].copy()
    positive["user_idx"] = positive["user_id"].map(user2idx)
    positive["item_idx"] = positive["banner_id"].map(item2idx)
    positive = positive.dropna(subset=["user_idx", "item_idx"]).copy()
    positive["user_idx"] = positive["user_idx"].astype(np.int64)
    positive["item_idx"] = positive["item_idx"].astype(np.int64)

    grouped = (
        positive.groupby(["user_idx", "item_idx"], as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        )
        .copy()
    )
    grouped["weight"] = grouped["clicks"] + 0.1 * grouped["impressions"]
    return grouped[["user_idx", "item_idx", "weight"]]


def build_user_history(pairs: pd.DataFrame) -> dict[int, set[int]]:
    history: dict[int, set[int]] = {}
    for row in pairs.itertuples(index=False):
        history.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    return history


class BPRDataset(Dataset):
    def __init__(
        self,
        positive_pairs: pd.DataFrame,
        user_history: dict[int, set[int]],
        num_items: int,
    ) -> None:
        self.user_idx = positive_pairs["user_idx"].to_numpy(dtype=np.int64)
        self.item_idx = positive_pairs["item_idx"].to_numpy(dtype=np.int64)
        self.weight = positive_pairs["weight"].to_numpy(dtype=np.float32)
        self.user_history = user_history
        self.num_items = num_items

    def __len__(self) -> int:
        return len(self.user_idx)

    def _sample_negative(self, user_idx: int) -> int:
        seen_items = self.user_history[user_idx]
        negative = np.random.randint(0, self.num_items)
        while negative in seen_items:
            negative = np.random.randint(0, self.num_items)
        return int(negative)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_idx = int(self.user_idx[index])
        pos_item_idx = int(self.item_idx[index])
        neg_item_idx = self._sample_negative(user_idx)

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "pos_item_idx": torch.tensor(pos_item_idx, dtype=torch.long),
            "neg_item_idx": torch.tensor(neg_item_idx, dtype=torch.long),
            "weight": torch.tensor(self.weight[index], dtype=torch.float32),
        }


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def encode_user(self, user_idx: torch.Tensor) -> torch.Tensor:
        x = self.user_emb(user_idx)
        x = self.user_mlp(x)
        return F.normalize(x, dim=-1)

    def encode_item(self, item_idx: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(item_idx)
        x = self.item_mlp(x)
        return F.normalize(x, dim=-1)

    @staticmethod
    def score(user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        return (user_vec * item_vec).sum(dim=-1)


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = -F.logsigmoid(pos_scores - neg_scores)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def train_step(
    model: TwoTower,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad()

    user_idx = batch["user_idx"].to(device)
    pos_item_idx = batch["pos_item_idx"].to(device)
    neg_item_idx = batch["neg_item_idx"].to(device)
    weight = batch["weight"].to(device)

    user_vec = model.encode_user(user_idx)
    pos_item_vec = model.encode_item(pos_item_idx)
    neg_item_vec = model.encode_item(neg_item_idx)

    pos_scores = model.score(user_vec, pos_item_vec)
    neg_scores = model.score(user_vec, neg_item_vec)

    loss = bpr_loss(pos_scores, neg_scores, weight)
    loss.backward()
    optimizer.step()
    return float(loss.item())


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
                seen_tensor = torch.tensor(
                    sorted(seen_items),
                    dtype=torch.long,
                    device=device,
                )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch two-tower recommender.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_df, valid_df, test_df = load_data()
    user2idx, item2idx, idx2item = build_mappings()

    train_pairs = prepare_positive_pairs(train_df, user2idx, item2idx)
    valid_pairs = prepare_positive_pairs(valid_df, user2idx, item2idx)
    test_pairs = prepare_positive_pairs(test_df, user2idx, item2idx)

    train_history = build_user_history(train_pairs)
    train_dataset = BPRDataset(
        positive_pairs=train_pairs,
        user_history=train_history,
        num_items=len(item2idx),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTower(
        n_users=len(user2idx),
        n_items=len(item2idx),
        emb_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"device: {device}")
    print(f"train positive pairs: {len(train_pairs)}")
    print(f"valid positive pairs: {len(valid_pairs)}")
    print(f"test positive pairs: {len(test_pairs)}")

    for epoch in range(1, args.epochs + 1):
        losses = [train_step(model, batch, optimizer, device) for batch in train_loader]
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch {epoch}/{args.epochs} loss: {epoch_loss:.6f}")

        valid_result = evaluate_topk(
            model=model,
            eval_pairs=valid_pairs,
            seen_history=train_history,
            num_items=len(item2idx),
            device=device,
            k=args.k,
        )
        print_eval("valid", valid_result, args.k)

    test_result = evaluate_topk(
        model=model,
        eval_pairs=test_pairs,
        seen_history=train_history,
        num_items=len(item2idx),
        device=device,
        k=args.k,
    )
    print_eval("test", test_result, args.k)

    sample_users = sorted(build_user_history(test_pairs).keys())[:3]
    if sample_users:
        model.eval()
        with torch.no_grad():
            all_items = torch.arange(len(item2idx), device=device)
            item_vectors = model.encode_item(all_items)
            print("sample predictions:")
            for user_idx in sample_users:
                user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
                scores = torch.matmul(model.encode_user(user_tensor), item_vectors.T).squeeze(0)

                seen_items = train_history.get(user_idx, set())
                if seen_items:
                    scores[torch.tensor(sorted(seen_items), device=device)] = -torch.inf

                top_items = torch.topk(scores, k=min(args.k, len(item2idx))).indices.cpu().tolist()
                top_banner_ids = [idx2item[item_idx] for item_idx in top_items]
                print(f"  user_idx={user_idx}: {top_banner_ids}")


if __name__ == "__main__":
    main()
