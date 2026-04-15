from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd

import torch
import torch.nn as nn
from rich.console import Console

from src.retrieval.data.ingest import DEFAULT_INTERACTIONS_CSV
from src.retrieval.pipeline.orchestrator import prepare_training_data
from src.retrieval.pipeline.registry import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_TRAIN_END,
    DEFAULT_VALID_END,
    RetrievalDataConfig,
)
from src.retrieval.pipeline.stages import (
    evaluate_recalls,
    recall_at_k,
    save_artifacts,
    validate_training_ready_data,
)


DATA_PATH = DEFAULT_INTERACTIONS_CSV
ARTIFACTS_DIR = DEFAULT_ARTIFACTS_DIR
TRAIN_END = DEFAULT_TRAIN_END
VALID_END = DEFAULT_VALID_END
console = Console()


class TwoTower(nn.Module):
    def __init__(self, n_users, n_banners, emb_dim=64):
        super().__init__()
        self.user_tower = nn.Embedding(n_users, emb_dim)
        self.banner_tower = nn.Embedding(n_banners, emb_dim)
        self.embedding_dim = emb_dim

    def encode_user(self, user_ids):
        return self.user_tower(user_ids)

    def encode_banner(self, banner_ids):
        return self.banner_tower(banner_ids)

    def score_all_banners(self):
        score = self.user_tower.weight @ self.banner_tower.weight.T
        return score

    def forward(self, user_ids, banner_ids):
        user_vec = self.encode_user(user_ids)
        banner_vec = self.encode_banner(banner_ids)
        return (user_vec * banner_vec).sum(dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a minimal TwoTower retrieval model.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--output-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--train-end", default=str(TRAIN_END.date()))
    parser.add_argument("--valid-end", default=str(VALID_END.date()))
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-k", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)
    output_dir = Path(args.output_dir)

    torch.manual_seed(args.seed)
    data = prepare_training_data(
        RetrievalDataConfig(
            interactions_path=data_path,
            train_end=train_end,
            valid_end=valid_end,
        )
    )
    validate_training_ready_data(data)

    train_users = data.train.users
    train_banners = data.train.banners
    train_labels = data.train.labels
    valid_positive_pairs = data.valid.positive_pairs

    model = TwoTower(
        n_users=data.n_users,
        n_banners=data.n_banners,
        emb_dim=args.emb_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_recall = float("-inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        logits = model(train_users, train_banners)
        loss = loss_fn(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            recall = recall_at_k(model, valid_positive_pairs, k=args.recall_k)

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        console.print(
            f"epoch={epoch + 1} train_loss={loss.item():.4f} valid_recall@{args.recall_k}={recall:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    metrics = evaluate_recalls(model, valid_positive_pairs, ks=[20, 50, args.recall_k])
    metrics["best_epoch"] = best_epoch
    save_artifacts(
        output_dir,
        model,
        data,
        metrics,
        {
            "train_end": str(train_end.date()),
            "valid_end": str(valid_end.date()),
            "data_path": data_path,
            "emb_dim": args.emb_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "recall_k": args.recall_k,
        },
    )
    console.print(json.dumps(metrics, ensure_ascii=False, indent=2))
    console.print(f"best_valid_recall@{args.recall_k}={best_recall:.4f} epoch={best_epoch}")
    console.print(f"saved_artifacts={output_dir}")


if __name__ == "__main__":
    main()
