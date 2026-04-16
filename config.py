import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal TwoTower retrieval model on data/raw."
    )
    parser.add_argument("--data-dir",        type=Path,  default=Path("data/raw"))
    parser.add_argument("--output",          type=Path,  default=Path("artifacts/twotower.pt"))
    parser.add_argument("--epochs",          type=int,   default=1)
    parser.add_argument("--batch-size",      type=int,   default=1024)
    parser.add_argument("--eval-batch-size", type=int,   default=4096)
    parser.add_argument("--embedding-dim",   type=int,   default=16)
    parser.add_argument("--hidden-dim",      type=int,   default=64)
    parser.add_argument("--output-dim",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--temperature",     type=float, default=0.05)
    parser.add_argument("--recall-k",        type=int,   default=100)
    parser.add_argument("--patience",        type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--train-end",       type=str,   default="2026-02-28")
    parser.add_argument("--valid-end",       type=str,   default="2026-03-15")
    parser.add_argument("--device",          type=str,   default="cpu")
    parser.add_argument("--num-workers",     type=int,   default=0)
    parser.add_argument("--pin-memory",      action="store_true")
    parser.add_argument("--drop-last-train", action="store_true")
    parser.add_argument(
        "--shuffle-train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )



    return parser.parse_args()