import argparse
from pathlib import Path

import torch

from src.data import EncodedTable
from src.model import TwoTowerModel


def save_checkpoint(
    output_path: Path,
    model: TwoTowerModel,
    args: argparse.Namespace,
    metrics: dict[str, float],
    user_table: EncodedTable,
    item_table: EncodedTable,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "metrics": metrics,
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