from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd


TRAIN_END = pd.Timestamp("2026-02-28")
VALID_END = pd.Timestamp("2026-03-15")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    embedding_dim: int
    learning_rate: float
    weight_decay: float
    k: int
    seed: int
    output_dir: str
    save_item_embeddings: bool
    early_stopping_patience: int
    early_stopping_min_delta: float


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a PyTorch two-tower recommender.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/pytorch_retrieval")
    parser.add_argument("--save-item-embeddings", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    args = parser.parse_args()
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        k=args.k,
        seed=args.seed,
        output_dir=args.output_dir,
        save_item_embeddings=args.save_item_embeddings,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
