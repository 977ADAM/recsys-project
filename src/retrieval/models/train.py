from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
import torch
import torch.nn as nn

from src.retrieval.data.ingest import DEFAULT_INTERACTIONS_CSV
from src.retrieval.models.evaluate import evaluate_recalls, recall_at_k
from src.retrieval.models.export import save_artifacts
from src.retrieval.models.two_tower import TwoTower
from src.retrieval.pipeline.orchestrator import prepare_training_data
from src.retrieval.pipeline.registry import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_TRAIN_END,
    DEFAULT_VALID_END,
    PreparedRetrievalData,
    RetrievalDataConfig,
)

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_TRAIN_EPOCHS = 100
DEFAULT_RUNTIME_EPOCHS = 25
DEFAULT_LR = 1e-2

console = Console()


def validate_training_data(data: PreparedRetrievalData) -> None:
    if data.train.users.numel() == 0:
        raise ValueError("Training split is empty after encoding.")
    if data.valid.positive_pairs.empty:
        raise ValueError("Validation split has no positive pairs for recall@k evaluation.")


def train_model(
    model: TwoTower,
    users: torch.Tensor,
    banners: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    epoch_callback=None,
) -> None:
    if users.numel() == 0:
        raise ValueError("Training split is empty after encoding.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        logits = model(users, banners)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch_callback is not None:
            epoch_callback(model, epoch + 1, float(loss.item()))


def train_two_tower_model(
    data: PreparedRetrievalData,
    *,
    emb_dim: int = DEFAULT_EMBEDDING_DIM,
    epochs: int = DEFAULT_TRAIN_EPOCHS,
    lr: float = DEFAULT_LR,
    seed: int = 42,
    recall_k: int = 100,
    progress_console: Console | None = None,
) -> tuple[TwoTower, dict[str, float]]:
    validate_training_data(data)
    torch.manual_seed(seed)

    model = TwoTower(
        n_users=data.n_users,
        n_banners=data.n_banners,
        emb_dim=emb_dim,
    )
    best_state = None
    best_recall = float("-inf")
    best_epoch = 0

    def on_epoch(current_model: TwoTower, epoch: int, loss_value: float) -> None:
        nonlocal best_state, best_recall, best_epoch
        current_model.eval()
        with torch.no_grad():
            recall = recall_at_k(current_model, data.valid.positive_pairs, k=recall_k)

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in current_model.state_dict().items()
            }

        if progress_console is not None:
            progress_console.print(
                f"epoch={epoch} train_loss={loss_value:.4f} valid_recall@{recall_k}={recall:.4f}"
            )

    train_model(
        model,
        data.train.users,
        data.train.banners,
        data.train.labels,
        epochs=epochs,
        lr=lr,
        epoch_callback=on_epoch,
    )

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    metrics = evaluate_recalls(model, data.valid.positive_pairs, ks=[20, 50, recall_k])
    metrics["best_epoch"] = best_epoch
    return model, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a minimal TwoTower retrieval model.")
    parser.add_argument("--data-path", default=str(DEFAULT_INTERACTIONS_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--train-end", default=str(DEFAULT_TRAIN_END.date()))
    parser.add_argument("--valid-end", default=str(DEFAULT_VALID_END.date()))
    parser.add_argument("--emb-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)
    output_dir = Path(args.output_dir)

    data = prepare_training_data(
        RetrievalDataConfig(
            interactions_path=data_path,
            train_end=train_end,
            valid_end=valid_end,
        )
    )
    model, metrics = train_two_tower_model(
        data,
        emb_dim=args.emb_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        recall_k=args.recall_k,
        progress_console=console,
    )

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
    console.print(
        f"best_valid_recall@{args.recall_k}={metrics[f'recall@{args.recall_k}']:.4f} "
        f"epoch={int(metrics['best_epoch'])}"
    )
    console.print(f"saved_artifacts={output_dir}")


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_TRAIN_EPOCHS",
    "DEFAULT_RUNTIME_EPOCHS",
    "DEFAULT_LR",
    "validate_training_data",
    "train_model",
    "train_two_tower_model",
    "main",
]
