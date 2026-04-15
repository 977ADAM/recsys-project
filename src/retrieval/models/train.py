from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich.console import Console
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.retrieval.utils.args import parse_args
from src.retrieval.models.evaluate import evaluate_recalls, recall_at_k
from src.retrieval.models.export import save_artifacts
from src.retrieval.models.two_tower import TwoTower
from src.retrieval.pipeline.orchestrator import prepare_training_data
from src.retrieval.pipeline.registry import (
    PreparedRetrievalData,
    RetrievalDataConfig,
)

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_TRAIN_EPOCHS = 20
DEFAULT_RUNTIME_EPOCHS = 10
DEFAULT_LR = 0.01

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
    batch_size: int = 2048,
    shuffle: bool = True,
) -> list[float]:
    if users.numel() == 0:
        raise ValueError("Training split is empty after encoding.")

    if not (len(users) == len(banners) == len(labels)):
        raise ValueError("Users, banners, and labels must have the same length.")

    dataset = TensorDataset(users, banners, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_examples = 0

        for batch_users, batch_banners, batch_labels in loader:
            logits = model(batch_users, batch_banners)
            loss = loss_fn(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_users.size(0)
            total_loss += float(loss.item()) * batch_size_actual
            total_examples += batch_size_actual

        epoch_loss = total_loss / total_examples if total_examples > 0 else 0.0
        epoch_losses.append(epoch_loss)

    return epoch_losses


def train_two_tower_model(
    data: PreparedRetrievalData,
    *,
    emb_dim: int = DEFAULT_EMBEDDING_DIM,
    epochs: int = DEFAULT_TRAIN_EPOCHS,
    lr: float = DEFAULT_LR,
    seed: int = 42,
    recall_k: int = 100,
    batch_size: int = 2048,
    progress_console: Console | None = None,
) -> tuple[TwoTower, dict[str, float]]:
    validate_training_data(data)
    torch.manual_seed(seed)

    model = TwoTower(
        n_users=data.n_users,
        n_banners=data.n_banners,
        emb_dim=emb_dim,
    )
    epoch_losses = train_model(
        model,
        data.train.users,
        data.train.banners,
        data.train.labels,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    model.eval()

    metrics = evaluate_recalls(model, data.valid.positive_pairs, ks=[20, 50, recall_k])
    metrics["final_epoch"] = epochs
    metrics["train_loss"] = round(epoch_losses[-1], 6) if epoch_losses else 0.0
    if progress_console is not None:
        progress_console.print(
            f"train_loss={metrics['train_loss']:.4f} valid_recall@{recall_k}={metrics[f'recall@{recall_k}']:.4f}"
        )
    return model, metrics

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
        batch_size=args.batch_size,
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
        f"valid_recall@{args.recall_k}={metrics[f'recall@{args.recall_k}']:.4f} "
        f"epoch={int(metrics['final_epoch'])}"
    )
    console.print(f"saved_artifacts={output_dir}")
