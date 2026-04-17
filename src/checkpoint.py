import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data import EncodedTable
from src.model import TwoTowerModel


def _serialize_config(config: argparse.Namespace | dict) -> dict[str, str | int | float | bool | None]:
    serialized: dict[str, str | int | float | bool | None] = {}
    if isinstance(config, dict):
        items = config.items()
    else:
        items = vars(config).items()
    for key, value in items:
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def save_checkpoint(
    output_path: Path,
    model: TwoTowerModel,
    config: argparse.Namespace | dict,
    metrics: dict[str, float],
    history: list[dict[str, float]],
    split_dates: dict[str, str],
    user_table: EncodedTable,
    item_table: EncodedTable,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": _serialize_config(config),
        "metrics": metrics,
        "history": history,
        "split_dates": split_dates,
        "user_ids": user_table.ids,
        "item_ids": item_table.ids,
        "user_categorical": user_table.categorical.cpu(),
        "user_numerical": user_table.numerical.cpu(),
        "item_categorical": item_table.categorical.cpu(),
        "item_numerical": item_table.numerical.cpu(),
        "user_cardinalities": user_table.cardinalities,
        "item_cardinalities": item_table.cardinalities,
    }
    torch.save(checkpoint, output_path)


def save_run_summary(output_path: Path, summary: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(output_path: Path, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(output_path, map_location=map_location, weights_only=False)
    checkpoint["user_table"] = EncodedTable(
        ids=checkpoint["user_ids"],
        id_to_row={entity_id: idx for idx, entity_id in enumerate(checkpoint["user_ids"])},
        categorical=checkpoint["user_categorical"],
        numerical=checkpoint["user_numerical"],
        cardinalities=checkpoint["user_cardinalities"],
    )
    checkpoint["item_table"] = EncodedTable(
        ids=checkpoint["item_ids"],
        id_to_row={entity_id: idx for idx, entity_id in enumerate(checkpoint["item_ids"])},
        categorical=checkpoint["item_categorical"],
        numerical=checkpoint["item_numerical"],
        cardinalities=checkpoint["item_cardinalities"],
    )
    return checkpoint


def build_run_summary(
    config,
    device: torch.device,
    data,
    fit_result,
    test_recall: float,
    test_users: int,
) -> dict:
    config_dict = _serialize_config(config)
    namespace = SimpleNamespace(**config_dict)
    output_path = Path(namespace.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    split_dates = {
        "train_end": namespace.train_end,
        "valid_end": namespace.valid_end,
    }
    if isinstance(data, dict):
        dataset = {
            "num_users": data["num_users"],
            "num_items": data["num_items"],
            "num_train_pairs": data["num_train_pairs"],
            "num_valid_pairs": data["num_valid_pairs"],
            "num_test_pairs": data["num_test_pairs"],
        }
    else:
        dataset = {
            "num_users": data.num_users,
            "num_items": data.num_items,
            "num_train_pairs": data.num_train_pairs,
            "num_valid_pairs": data.num_valid_pairs,
            "num_test_pairs": data.num_test_pairs,
        }
    metrics = {
        "best_epoch": fit_result.best_epoch,
        "best_valid_recall_at_k": fit_result.best_valid_recall_at_k,
        "best_valid_users": fit_result.best_valid_users,
        "test_recall_at_k": test_recall,
        "test_users": test_users,
    }
    return {
        "artifacts": {
            "checkpoint_path": str(output_path),
            "summary_path": str(summary_path),
        },
        "config": config_dict,
        "split_dates": split_dates,
        "dataset": dataset,
        "metrics": metrics,
        "history": fit_result.history,
        "device": device.type,
    }
