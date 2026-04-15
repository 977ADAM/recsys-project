from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


USER_CATEGORICAL_COLUMNS = [
    "user_id",
    "gender",
    "city_tier",
    "device_os",
    "platform",
    "income_band",
    "activity_segment",
    "interest_1",
    "interest_2",
    "interest_3",
    "country",
]
USER_NUMERIC_COLUMNS = ["age", "signup_days_ago", "is_premium"]

ITEM_CATEGORICAL_COLUMNS = [
    "banner_id",
    "brand",
    "category",
    "subcategory",
    "banner_format",
    "campaign_goal",
    "target_gender",
]
ITEM_NUMERIC_COLUMNS = [
    "target_age_min",
    "target_age_max",
    "cpm_bid",
    "quality_score",
    "is_active",
    "banner_age_days",
]


@dataclass
class EncodedTable:
    ids: list[str]
    id_to_row: dict[str, int]
    categorical: torch.Tensor
    numerical: torch.Tensor
    cardinalities: list[int]


@dataclass
class PairTensors:
    user_indices: torch.Tensor
    item_indices: torch.Tensor
    weights: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.user_indices.shape[0])


class Tower(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        num_numeric_features: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(num_embeddings=cardinality, embedding_dim=embedding_dim)
            for cardinality in cardinalities
        )
        input_dim = len(cardinalities) * embedding_dim + num_numeric_features
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        parts = [embedding(categorical[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        if numerical.shape[1] > 0:
            parts.append(numerical)
        tower_input = torch.cat(parts, dim=1)
        return F.normalize(self.network(tower_input), dim=1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_cardinalities: list[int],
        item_cardinalities: list[int],
        user_num_features: int,
        item_num_features: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.user_tower = Tower(
            cardinalities=user_cardinalities,
            num_numeric_features=user_num_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.item_tower = Tower(
            cardinalities=item_cardinalities,
            num_numeric_features=item_num_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def encode_users(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        return self.user_tower(categorical, numerical)

    def encode_items(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        return self.item_tower(categorical, numerical)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal TwoTower retrieval model on data/raw."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/twotower.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-end", type=str, default="2026-02-28")
    parser.add_argument("--valid-end", type=str, default="2026-03-15")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(data_dir / "users.csv")
    banners = pd.read_csv(data_dir / "banners.csv", parse_dates=["created_at"])
    interactions = pd.read_csv(
        data_dir / "banner_interactions.csv",
        parse_dates=["event_date"],
    )
    return users, banners, interactions


def prepare_banners(banners: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    prepared = banners.copy()
    prepared["banner_age_days"] = (
        (reference_date - prepared["created_at"]).dt.days.clip(lower=0).astype(np.float32)
    )
    return prepared


def split_interactions(
    interactions: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = interactions[interactions["event_date"] <= train_end].copy()
    valid_df = interactions[
        (interactions["event_date"] > train_end)
        & (interactions["event_date"] <= valid_end)
    ].copy()
    test_df = interactions[interactions["event_date"] > valid_end].copy()
    return train_df, valid_df, test_df


def build_positive_pairs(interactions: pd.DataFrame) -> pd.DataFrame:
    positives = interactions[interactions["clicks"] > 0].copy()
    if positives.empty:
        raise ValueError("No positive interactions with clicks > 0 were found.")
    pairs = (
        positives.groupby(["user_id", "banner_id"], as_index=False)
        .agg(clicks=("clicks", "sum"))
        .sort_values(["user_id", "banner_id"])
        .reset_index(drop=True)
    )
    pairs["weight"] = np.log1p(pairs["clicks"]).astype(np.float32)
    return pairs


def encode_table(
    frame: pd.DataFrame,
    id_column: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> EncodedTable:
    table = frame.drop_duplicates(subset=id_column).sort_values(id_column).reset_index(drop=True).copy()
    for column in categorical_columns:
        table[column] = table[column].fillna("__missing__").astype(str)

    numeric_frame = table[numerical_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric_mean = numeric_frame.mean()
    numeric_std = numeric_frame.std(ddof=0).replace(0.0, 1.0)
    standardized_numeric = ((numeric_frame - numeric_mean) / numeric_std).astype(np.float32)

    categorical_arrays: list[np.ndarray] = []
    cardinalities: list[int] = []
    for column in categorical_columns:
        values = table[column].astype(str)
        vocabulary = {value: idx + 1 for idx, value in enumerate(sorted(values.unique()))}
        encoded = values.map(vocabulary).fillna(0).astype(np.int64).to_numpy()
        categorical_arrays.append(encoded)
        cardinalities.append(len(vocabulary) + 1)

    categorical_matrix = np.stack(categorical_arrays, axis=1).astype(np.int64)
    ids = table[id_column].astype(str).tolist()

    return EncodedTable(
        ids=ids,
        id_to_row={entity_id: idx for idx, entity_id in enumerate(ids)},
        categorical=torch.tensor(categorical_matrix, dtype=torch.long),
        numerical=torch.tensor(standardized_numeric.to_numpy(), dtype=torch.float32),
        cardinalities=cardinalities,
    )


def pairs_to_tensors(
    pairs: pd.DataFrame,
    user_table: EncodedTable,
    item_table: EncodedTable,
) -> PairTensors:
    mapped = pairs.copy()
    mapped["user_index"] = mapped["user_id"].map(user_table.id_to_row)
    mapped["item_index"] = mapped["banner_id"].map(item_table.id_to_row)
    mapped = mapped.dropna(subset=["user_index", "item_index"]).reset_index(drop=True)
    if mapped.empty:
        raise ValueError("No training pairs remained after mapping ids to feature tables.")
    return PairTensors(
        user_indices=torch.tensor(mapped["user_index"].astype(np.int64).to_numpy(), dtype=torch.long),
        item_indices=torch.tensor(mapped["item_index"].astype(np.int64).to_numpy(), dtype=torch.long),
        weights=torch.tensor(mapped["weight"].astype(np.float32).to_numpy(), dtype=torch.float32),
    )


def iterate_minibatches(num_examples: int, batch_size: int) -> torch.Tensor:
    order = torch.randperm(num_examples)
    for start in range(0, num_examples, batch_size):
        batch = order[start : start + batch_size]
        if batch.numel() > 1:
            yield batch


def train_one_epoch(
    model: TwoTowerModel,
    optimizer: torch.optim.Optimizer,
    pairs: PairTensors,
    user_table: EncodedTable,
    item_table: EncodedTable,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in iterate_minibatches(pairs.size, batch_size):
        user_idx = pairs.user_indices[batch]
        item_idx = pairs.item_indices[batch]
        weights = pairs.weights[batch].to(device)

        user_categorical = user_table.categorical[user_idx].to(device)
        user_numerical = user_table.numerical[user_idx].to(device)
        item_categorical = item_table.categorical[item_idx].to(device)
        item_numerical = item_table.numerical[item_idx].to(device)

        user_embeddings = model.encode_users(user_categorical, user_numerical)
        item_embeddings = model.encode_items(item_categorical, item_numerical)
        logits = user_embeddings @ item_embeddings.T / temperature
        targets = torch.arange(logits.shape[0], device=device)

        loss_users = F.cross_entropy(logits, targets, reduction="none")
        loss_items = F.cross_entropy(logits.T, targets, reduction="none")
        sample_weights = weights / weights.mean().clamp(min=1e-6)
        loss = (((loss_users + loss_items) * 0.5) * sample_weights).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.inference_mode()
def encode_full_table(
    model: TwoTowerModel,
    table: EncodedTable,
    tower: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoded_batches: list[torch.Tensor] = []
    for start in range(0, len(table.ids), batch_size):
        stop = start + batch_size
        categorical = table.categorical[start:stop].to(device)
        numerical = table.numerical[start:stop].to(device)
        if tower == "user":
            batch_embeddings = model.encode_users(categorical, numerical)
        else:
            batch_embeddings = model.encode_items(categorical, numerical)
        encoded_batches.append(batch_embeddings.cpu())
    return torch.cat(encoded_batches, dim=0)


def evaluate_recall_at_k(
    model: TwoTowerModel,
    user_table: EncodedTable,
    item_table: EncodedTable,
    positives: pd.DataFrame,
    k: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, int]:
    grouped = positives.groupby("user_id")["banner_id"].agg(lambda values: set(values)).to_dict()
    if not grouped:
        return float("nan"), 0

    model.eval()
    user_embeddings = encode_full_table(model, user_table, "user", device, batch_size)
    item_embeddings = encode_full_table(model, item_table, "item", device, batch_size)

    candidate_count = item_embeddings.shape[0]
    top_k = min(k, candidate_count)
    scores = user_embeddings @ item_embeddings.T
    top_indices = torch.topk(scores, k=top_k, dim=1).indices
    item_ids = item_table.ids

    recalls: list[float] = []
    for user_id, relevant_items in grouped.items():
        user_row = user_table.id_to_row.get(str(user_id))
        if user_row is None:
            continue
        predicted_items = {item_ids[idx] for idx in top_indices[user_row].tolist()}
        recalls.append(len(predicted_items & relevant_items) / len(relevant_items))

    if not recalls:
        return float("nan"), 0
    return float(np.mean(recalls)), len(recalls)


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)

    users, banners, interactions = load_tables(args.data_dir)
    banners = prepare_banners(banners, reference_date=train_end)

    train_df, valid_df, test_df = split_interactions(interactions, train_end, valid_end)
    train_pairs = build_positive_pairs(train_df)
    valid_pairs = build_positive_pairs(valid_df)
    test_pairs = build_positive_pairs(test_df)

    user_table = encode_table(
        frame=users,
        id_column="user_id",
        categorical_columns=USER_CATEGORICAL_COLUMNS,
        numerical_columns=USER_NUMERIC_COLUMNS,
    )
    item_table = encode_table(
        frame=banners,
        id_column="banner_id",
        categorical_columns=ITEM_CATEGORICAL_COLUMNS,
        numerical_columns=ITEM_NUMERIC_COLUMNS,
    )
    train_tensors = pairs_to_tensors(train_pairs, user_table, item_table)

    model = TwoTowerModel(
        user_cardinalities=user_table.cardinalities,
        item_cardinalities=item_table.cardinalities,
        user_num_features=user_table.numerical.shape[1],
        item_num_features=item_table.numerical.shape[1],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        "Loaded data:",
        f"users={len(user_table.ids)}",
        f"items={len(item_table.ids)}",
        f"train_pairs={train_tensors.size}",
        f"valid_pairs={len(valid_pairs)}",
        f"test_pairs={len(test_pairs)}",
        f"device={device.type}",
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            pairs=train_tensors,
            user_table=user_table,
            item_table=item_table,
            batch_size=args.batch_size,
            temperature=args.temperature,
            device=device,
        )
        valid_recall, valid_users = evaluate_recall_at_k(
            model=model,
            user_table=user_table,
            item_table=item_table,
            positives=valid_pairs,
            k=args.recall_k,
            device=device,
            batch_size=args.eval_batch_size,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "valid_recall_at_k": valid_recall,
            }
        )
        print(
            f"epoch={epoch:02d}",
            f"train_loss={train_loss:.4f}",
            f"valid_recall@{args.recall_k}={valid_recall:.4f}",
            f"valid_users={valid_users}",
        )

    test_recall, test_users = evaluate_recall_at_k(
        model=model,
        user_table=user_table,
        item_table=item_table,
        positives=test_pairs,
        k=args.recall_k,
        device=device,
        batch_size=args.eval_batch_size,
    )
    print(
        f"test_recall@{args.recall_k}={test_recall:.4f}",
        f"test_users={test_users}",
    )

    metrics = {
        "valid_recall_at_k": history[-1]["valid_recall_at_k"] if history else float("nan"),
        "test_recall_at_k": test_recall,
    }
    save_checkpoint(
        output_path=args.output,
        model=model,
        args=args,
        metrics=metrics,
        user_table=user_table,
        item_table=item_table,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
