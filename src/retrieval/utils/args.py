import argparse

from src.retrieval.pipeline.registry import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_TRAIN_END,
    DEFAULT_VALID_END,
)
from src.retrieval.data.ingest import DEFAULT_INTERACTIONS_CSV

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_TRAIN_EPOCHS = 100
DEFAULT_RUNTIME_EPOCHS = 25
DEFAULT_LR = 0.01
DEFAULT_BATCH_SIZE = 1024
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_PATIENCE = 50
DEFAULT_MIN_DELTA = 1e-4


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train a minimal TwoTower retrieval model.")
    parser.add_argument("--data-path", default=str(DEFAULT_INTERACTIONS_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--train-end", default=str(DEFAULT_TRAIN_END.date()))
    parser.add_argument("--valid-end", default=str(DEFAULT_VALID_END.date()))
    parser.add_argument("--emb-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-k", type=int, default=100)
    return parser.parse_args(argv)
