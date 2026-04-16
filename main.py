import random
import numpy as np
import pandas as pd
import torch

from src.data import RecSysDataModule
from src.checkpoint import save_checkpoint
from src.model import init_model
from src.config import parse_args

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    data = RecSysDataModule(args).setup()
    print(
    "Loaded data:",
    f"users={len(data.user_table.ids)}",
    f"items={len(data.item_table.ids)}",
    f"train_pairs={len(data.train_pairs)}",
    f"valid_pairs={len(data.valid_pairs)}",
    f"test_pairs={len(data.test_pairs)}",
    f"device={device.type}",
    )

    model = init_model(args, data.user_table, data.item_table, device)

    fit_result = model.fit(
        args=args,
        train_loader=data.train_loader,
        valid_pairs=data.valid_pairs,
        user_table=data.user_table,
        item_table=data.item_table,
        device=device,
    )

    test_recall, test_users = model.evaluate_recall_at_k(
        user_table=data.user_table,
        item_table=data.item_table,
        positives=data.test_pairs,
        k=args.recall_k,
        device=device,
        batch_size=args.eval_batch_size,
    )
    print(
        f"test_recall@{args.recall_k}={test_recall:.4f}",
        f"test_users={test_users}",
    )

    metrics = {
        "best_epoch": float(fit_result.best_epoch),
        "valid_recall_at_k": fit_result.best_valid_recall_at_k,
        "test_recall_at_k": test_recall,
    }
    save_checkpoint(
        output_path=args.output,
        model=model,
        args=args,
        metrics=metrics,
        user_table=data.user_table,
        item_table=data.item_table,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
