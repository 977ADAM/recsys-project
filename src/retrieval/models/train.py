from __future__ import annotations

import argparse
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
DEFAULT_TRAIN_EPOCHS = 100
DEFAULT_RUNTIME_EPOCHS = 25
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
    weight_decay: float = 1e-5,
    epoch_callback=None,
) -> None:
    


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
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

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

        if epoch_callback is not None:
            epoch_callback(model, epoch + 1, epoch_loss)


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
        batch_size=batch_size,
        epoch_callback=on_epoch,
    )

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    metrics = evaluate_recalls(model, data.valid.positive_pairs, ks=[20, 50, recall_k])
    metrics["best_epoch"] = best_epoch
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
        batch_size=getattr(args, "batch_size", 512),
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
