from pathlib import Path
import argparse
import json
import pandas as pd
from rich.console import Console

import torch
import torch.nn as nn

try:
    from src.retrieval.data_loader import load_data
except ImportError:
    from data_loader import load_data


DATA_PATH = Path(__file__).with_name("banner_interactions.csv")
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "pytorch_retrieval"
TRAIN_END = pd.Timestamp("2026-02-28")
VALID_END = pd.Timestamp("2026-03-15")
console = Console()


class TwoTower(nn.Module):
    def __init__(self, n_users, n_banners, emb_dim=64):
        super().__init__()
        self.user_tower = nn.Embedding(n_users, emb_dim)
        self.banner_tower = nn.Embedding(n_banners, emb_dim)
        self.embedding_dim = emb_dim

    def encode_user(self, user_ids):
        return self.user_tower(user_ids)

    def encode_banner(self, banner_ids):
        return self.banner_tower(banner_ids)

    def score_all_banners(self):
        score = self.user_tower.weight @ self.banner_tower.weight.T
        return score

    def forward(self, user_ids, banner_ids):
        user_vec = self.encode_user(user_ids)
        banner_vec = self.encode_banner(banner_ids)
        return (user_vec * banner_vec).sum(dim=1)


def recall_at_k(model, positive_pairs, k=100):
    scores = model.score_all_banners()
    topk = scores.topk(k=min(k, scores.size(1)), dim=1).indices

    hits = 0
    total = 0
    for user_idx, group in positive_pairs.groupby("user_idx"):
        true_banners = set(group["banner_idx"].tolist())
        predicted = set(topk[user_idx].tolist())
        hits += len(true_banners & predicted)
        total += len(true_banners)

    return 0.0 if total == 0 else hits / total


def evaluate_recalls(model, positive_pairs, ks: list[int]) -> dict[str, float]:
    return {f"recall@{k}": round(recall_at_k(model, positive_pairs, k=k), 6) for k in ks}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a minimal TwoTower retrieval model.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--output-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--train-end", default=str(TRAIN_END.date()))
    parser.add_argument("--valid-end", default=str(VALID_END.date()))
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-k", type=int, default=100)
    return parser.parse_args()


def save_artifacts(output_dir: Path, model: TwoTower, data: dict, metrics: dict, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "n_users": data["n_users"],
            "n_banners": data["n_banners"],
        },
        output_dir / "model.pt",
    )

    with (output_dir / "mappings.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "user2idx": data["user2idx"],
                "item2idx": data["item2idx"],
                "idx2item": {str(idx): banner_id for idx, banner_id in data["idx2item"].items()},
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "model_version": "pytorch_retrieval",
                "model_type": "two_tower",
                "embedding_dim": model.embedding_dim,
                "train_end": config["train_end"],
                "valid_end": config["valid_end"],
                "latest_event_date": config["latest_event_date"],
                "train_rows": data["train_rows"],
                "valid_rows": data["valid_rows"],
                "test_rows": data["test_rows"],
                "validation_metrics": metrics,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)
    output_dir = Path(args.output_dir)

    torch.manual_seed(args.seed)
    data = load_data(data_path, train_end, valid_end)

    train_users = data["train_users"]
    train_banners = data["train_banners"]
    train_labels = data["train_labels"]
    valid_positive_pairs = data["valid_positive_pairs"]

    model = TwoTower(
        n_users=data["n_users"],
        n_banners=data["n_banners"],
        emb_dim=args.emb_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_recall = float("-inf")
    best_epoch = 0

    if train_users.numel() == 0:
        raise ValueError("Training split is empty after encoding.")

    if valid_positive_pairs.empty:
        raise ValueError("Validation split has no positive pairs for recall@k evaluation.")

    for epoch in range(args.epochs):
        model.train()
        logits = model(train_users, train_banners)
        loss = loss_fn(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            recall = recall_at_k(model, valid_positive_pairs, k=args.recall_k)

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        console.print(
            f"epoch={epoch + 1} train_loss={loss.item():.4f} valid_recall@{args.recall_k}={recall:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    metrics = evaluate_recalls(model, valid_positive_pairs, ks=[20, 50, args.recall_k])
    metrics["best_epoch"] = best_epoch
    save_artifacts(
        output_dir,
        model,
        data,
        metrics,
        {
            "train_end": str(train_end.date()),
            "valid_end": str(valid_end.date()),
            "latest_event_date": (
                str(data["test_df"]["event_date"].max().date())
                if not data["test_df"].empty
                else (
                    str(data["valid_df"]["event_date"].max().date())
                    if not data["valid_df"].empty
                    else str(data["train_df"]["event_date"].max().date())
                )
            ),
        },
    )
    console.print(json.dumps(metrics, ensure_ascii=False, indent=2))
    console.print(f"best_valid_recall@{args.recall_k}={best_recall:.4f} epoch={best_epoch}")
    console.print(f"saved_artifacts={output_dir}")


if __name__ == "__main__":
    main()
