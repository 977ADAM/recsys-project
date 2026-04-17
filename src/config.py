import argparse
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TwoTowerConfig:
    data_dir: Path = Path("data/raw")
    output: Path = Path("artifacts/twotower.pt")
    epochs: int = 1
    batch_size: int = 1024
    eval_batch_size: int = 4096
    embedding_dim: int = 16
    hidden_dim: int = 64
    output_dim: int = 32
    lr: float = 1e-3
    temperature: float = 0.05
    recall_k: int = 100
    patience: int = 0
    seed: int = 42
    train_end: str = "2026-02-28"
    valid_end: str = "2026-03-15"
    device: str = "cpu"
    num_workers: int = 0
    pin_memory: bool = False
    drop_last_train: bool = False
    shuffle_train: bool = True

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, overrides: dict | None = None) -> "TwoTowerConfig":
        if overrides is None:
            return cls()
        config = cls()
        for key, value in overrides.items():
            if not hasattr(config, key):
                continue
            current = getattr(config, key)
            if isinstance(current, Path) and not isinstance(value, Path):
                setattr(config, key, Path(value))
            else:
                setattr(config, key, value)
        return config

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "TwoTowerConfig":
        return cls.from_dict(vars(args))


DEFAULT_CONFIG = TwoTowerConfig().to_dict()


def build_namespace(overrides: dict | None = None) -> argparse.Namespace:
    return TwoTowerConfig.from_dict(overrides).to_namespace()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal TwoTower retrieval model on data/raw."
    )
    parser.add_argument("--data-dir",        type=Path,  default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output",          type=Path,  default=DEFAULT_CONFIG["output"])
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size",      type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--eval-batch-size", type=int,   default=DEFAULT_CONFIG["eval_batch_size"])
    parser.add_argument("--embedding-dim",   type=int,   default=DEFAULT_CONFIG["embedding_dim"])
    parser.add_argument("--hidden-dim",      type=int,   default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--output-dim",      type=int,   default=DEFAULT_CONFIG["output_dim"])
    parser.add_argument("--lr",              type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--temperature",     type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--recall-k",        type=int,   default=DEFAULT_CONFIG["recall_k"])
    parser.add_argument("--patience",        type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--seed",            type=int,   default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--train-end",       type=str,   default=DEFAULT_CONFIG["train_end"])
    parser.add_argument("--valid-end",       type=str,   default=DEFAULT_CONFIG["valid_end"])
    parser.add_argument("--device",          type=str,   default=DEFAULT_CONFIG["device"])
    parser.add_argument("--num-workers",     type=int,   default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--pin-memory",      action="store_true")
    parser.add_argument("--drop-last-train", action="store_true")
    parser.add_argument(
        "--shuffle-train",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["shuffle_train"],
    )
    return parser.parse_args()
