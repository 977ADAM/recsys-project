from pathlib import Path
from rich.console import Console
import pandas as pd

import torch
import torch.nn as nn

from data_loader import load_data


DATA_PATH = Path(__file__).with_name("banner_interactions.csv")
TRAIN_END = pd.Timestamp("2026-02-28")
VALID_END = pd.Timestamp("2026-03-15")
console = Console()


class TwoTower(nn.Module):
    def __init__(self, n_users, n_banners, emb_dim=64):
        super().__init__()
        self.user_tower = nn.Embedding(n_users, emb_dim)
        self.banner_tower = nn.Embedding(n_banners, emb_dim)

    def score_all_banners(self):
        score = self.user_tower.weight @ self.banner_tower.weight.T
        return score

    def forward(self, user_ids, banner_ids):

        user_vec = self.user_tower(user_ids)
        banner_vec = self.banner_tower(banner_ids)

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


def main():
    data = load_data(DATA_PATH)

    

    users = data["users"]
    banners = data["banners"]
    labels = data["labels"]
    positive_pairs = data["positive_pairs"]

    model = TwoTower(n_users=data["n_users"], n_banners=data["n_banners"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        logits = model(users, banners)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recall = recall_at_k(model, positive_pairs, k=100)
        console.print(f"epoch={epoch + 1} loss={loss.item():.4f} recall@100={recall:.4f}")


if __name__ == "__main__":
    main()
