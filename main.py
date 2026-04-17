import random
import numpy as np
import torch
from src.config import TwoTowerConfig, parse_args
from src.data import RecSysDataModule
from src.twotower import TwoTower


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    args = parse_args()
    config = TwoTowerConfig.from_namespace(args)
    set_seed(args.seed)
    data = RecSysDataModule(args).setup()
    print(
        "Loaded data:",
        f"users={data.num_users}",
        f"items={data.num_items}",
        f"train_pairs={data.num_train_pairs}",
        f"valid_pairs={data.num_valid_pairs}",
        f"test_pairs={data.num_test_pairs}",
        f"device={args.device}",
    )
    train_data = data.train_data()
    valid_data = data.valid_data()
    test_data = data.test_data()

    model = TwoTower(config=config)
    model.fit(train_data, valid_data)
    metrics = model.evaluate(test_data)
    checkpoint_path = model.save_model(config.output)
    summary_path = model.save_summary()
    print(
        f"test_recall@{config.recall_k}={metrics['test_recall_at_k']:.4f}",
        f"test_users={int(metrics['test_users'])}",
    )
    print(
        "Artifacts saved:",
        f"checkpoint={checkpoint_path}",
        f"summary={summary_path}",
    )


if __name__ == "__main__":
    main()
