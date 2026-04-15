#!/usr/bin/env python3
import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.ranker.artifacts import (
    RANKER_MODEL_VERSION,
    default_ranker_artifacts_path,
    normalize_metadata_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

USER_COLS = [
    "user_id",
    "age",
    "gender",
    "city_tier",
    "device_os",
    "platform",
    "income_band",
    "activity_segment",
    "interest_1",
    "interest_2",
    "interest_3",
    "signup_days_ago",
    "is_premium",
]

BANNER_COLS = [
    "banner_id",
    "brand",
    "category",
    "subcategory",
    "banner_format",
    "campaign_goal",
    "target_gender",
    "target_age_min",
    "target_age_max",
    "cpm_bid",
    "quality_score",
    "created_at",
    "is_active",
]

CAT_FEATURES = [
    "user_id",
    "banner_id",
    "gender",
    "city_tier",
    "device_os",
    "platform",
    "income_band",
    "activity_segment",
    "interest_1",
    "interest_2",
    "interest_3",
    "brand",
    "category",
    "subcategory",
    "banner_format",
    "campaign_goal",
    "target_gender",
]

FEATURE_COLS = [
    "user_id",
    "banner_id",
    "age",
    "gender",
    "city_tier",
    "device_os",
    "platform",
    "income_band",
    "activity_segment",
    "interest_1",
    "interest_2",
    "interest_3",
    "signup_days_ago",
    "is_premium",
    "brand",
    "category",
    "subcategory",
    "banner_format",
    "campaign_goal",
    "target_gender",
    "target_age_min",
    "target_age_max",
    "cpm_bid",
    "quality_score",
    "is_active",
    "age_match",
    "gender_match",
    "interest_match_1",
    "interest_match_2",
    "interest_match_3",
    "interest_match_count",
    "interest_match_any",
    "banner_age_days",
    "target_age_span",
    "age_distance_to_target",
    "event_dow",
    "event_day",
    "event_month",
    "banner_ctr_prior",
    "banner_ctr_prior_impr",
    "user_ctr_prior",
    "user_ctr_prior_impr",
    "subcategory_ctr_prior",
    "subcategory_ctr_prior_impr",
    "brand_ctr_prior",
    "brand_ctr_prior_impr",
    "user_subcategory_ctr_prior",
    "user_subcategory_ctr_prior_impr",
    "user_banner_ctr_prior",
    "user_banner_ctr_prior_impr",
]

DENSE_FEATURES = [col for col in FEATURE_COLS if col not in CAT_FEATURES]


class TabularDataset(Dataset):
    def __init__(self, cat_array, dense_array, clicks, impressions):
        self.cat_array = torch.as_tensor(cat_array, dtype=torch.long)
        self.dense_array = torch.as_tensor(dense_array, dtype=torch.float32)
        self.clicks = torch.as_tensor(clicks, dtype=torch.float32)
        self.impressions = torch.as_tensor(impressions, dtype=torch.float32)

    def __len__(self):
        return self.cat_array.shape[0]

    def __getitem__(self, idx):
        return (
            self.cat_array[idx],
            self.dense_array[idx],
            self.clicks[idx],
            self.impressions[idx],
        )


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        return self.net(x)


class DeepFM(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Dict[str, int],
        dense_dim: int,
        hidden_dims: List[int],
        dropout: float,
        emb_dim: int,
    ):
        super().__init__()
        self.cat_features = list(cat_cardinalities.keys())
        self.embedding_dim = emb_dim

        self.first_order_embeddings = nn.ModuleDict(
            {feat: nn.Embedding(card, 1) for feat, card in cat_cardinalities.items()}
        )
        self.fm_embeddings = nn.ModuleDict(
            {feat: nn.Embedding(card, self.embedding_dim) for feat, card in cat_cardinalities.items()}
        )
        self.linear_dense = nn.Linear(dense_dim, 1)

        deep_input_dim = dense_dim + len(self.cat_features) * self.embedding_dim
        self.deep = MLP(deep_input_dim, hidden_dims, dropout)
        self.deep_out = nn.Linear(self.deep.output_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for emb in self.first_order_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.fm_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.linear_dense.weight)
        nn.init.zeros_(self.linear_dense.bias)
        for module in self.deep.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.deep_out.weight)
        nn.init.zeros_(self.deep_out.bias)

    def forward(self, cat_x: torch.Tensor, dense_x: torch.Tensor) -> torch.Tensor:
        linear_cat_terms = []
        fm_vectors = []
        deep_vectors = []

        for idx, feat in enumerate(self.cat_features):
            feat_ids = cat_x[:, idx]
            linear_cat_terms.append(self.first_order_embeddings[feat](feat_ids))
            emb = self.fm_embeddings[feat](feat_ids)
            fm_vectors.append(emb)
            deep_vectors.append(emb)

        linear_part = self.bias + self.linear_dense(dense_x)
        linear_part = linear_part + torch.stack(linear_cat_terms, dim=0).sum(dim=0)

        stacked = torch.stack(fm_vectors, dim=1)  # [B, F, D]
        summed = stacked.sum(dim=1)
        fm_part = 0.5 * (summed.pow(2) - stacked.pow(2).sum(dim=1)).sum(dim=1, keepdim=True)

        deep_input = torch.cat([dense_x] + deep_vectors, dim=1)
        deep_part = self.deep_out(self.deep(deep_input))

        logits = linear_part + fm_part + deep_part
        return logits.squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepFM pCTR model on aggregated banner interactions."
    )
    parser.add_argument("--interactions-csv", required=True, default="./data/db/banner_interactions.csv")
    parser.add_argument("--users-csv", required=True, default="./data/db/users.csv")
    parser.add_argument("--banners-csv", required=True, default="./data/db/banners.csv")
    parser.add_argument(
        "--output-dir",
        required=True,
        default=str(default_ranker_artifacts_path(PROJECT_ROOT)),
    )
    parser.add_argument("--valid-days", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dims", type=str, default="256,128,64")
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Auto if omitted")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(interactions_csv, users_csv, banners_csv):
    interactions = pd.read_csv(interactions_csv, parse_dates=["event_date"])
    users = pd.read_csv(users_csv)
    banners = pd.read_csv(banners_csv, parse_dates=["created_at"])

    banners = banners.copy()
    banners["created_at"] = pd.to_datetime(banners["created_at"], errors="coerce")

    df = interactions.merge(users[USER_COLS], on="user_id", how="left")
    df = df.merge(banners[BANNER_COLS], on="banner_id", how="left")
    df["target_ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan)
    df["target_ctr"] = df["target_ctr"].fillna(0.0).clip(0.0, 1.0)
    return df


def add_base_features(df):
    df = df.copy()
    df["age_match"] = (
        (df["age"] >= df["target_age_min"]) & (df["age"] <= df["target_age_max"])
    ).astype("int8")
    df["gender_match"] = (
        (df["target_gender"] == "U")
        | (df["gender"] == "U")
        | (df["gender"] == df["target_gender"])
    ).astype("int8")
    df["interest_match_1"] = (df["interest_1"] == df["subcategory"]).astype("int8")
    df["interest_match_2"] = (df["interest_2"] == df["subcategory"]).astype("int8")
    df["interest_match_3"] = (df["interest_3"] == df["subcategory"]).astype("int8")
    df["interest_match_count"] = (
        df["interest_match_1"] + df["interest_match_2"] + df["interest_match_3"]
    ).astype("int8")
    df["interest_match_any"] = (df["interest_match_count"] > 0).astype("int8")
    df["banner_age_days"] = ((df["event_date"] - df["created_at"]).dt.days.fillna(0).clip(lower=0))
    df["target_age_span"] = (df["target_age_max"] - df["target_age_min"]).clip(lower=0)
    df["age_distance_to_target"] = np.where(
        df["age"] < df["target_age_min"],
        df["target_age_min"] - df["age"],
        np.where(df["age"] > df["target_age_max"], df["age"] - df["target_age_max"], 0),
    )
    df["event_dow"] = df["event_date"].dt.dayofweek.astype("int8")
    df["event_day"] = df["event_date"].dt.day.astype("int8")
    df["event_month"] = df["event_date"].dt.month.astype("int8")
    return df


def add_date_prior_feature(df, group_cols, feature_name, alpha, global_ctr):
    work = (
        df.groupby(group_cols + ["event_date"], as_index=False)[["clicks", "impressions"]]
        .sum()
        .sort_values(group_cols + ["event_date"])
    )
    grp = work.groupby(group_cols, sort=False)
    work["prior_clicks"] = grp["clicks"].cumsum() - work["clicks"]
    work["prior_impressions"] = grp["impressions"].cumsum() - work["impressions"]
    work[feature_name] = (work["prior_clicks"] + alpha * global_ctr) / (work["prior_impressions"] + alpha)
    count_feature = f"{feature_name}_impr"
    work[count_feature] = work["prior_impressions"]
    df = df.merge(
        work[group_cols + ["event_date", feature_name, count_feature]],
        on=group_cols + ["event_date"],
        how="left",
    )
    df[feature_name] = df[feature_name].fillna(global_ctr)
    df[count_feature] = df[count_feature].fillna(0.0)
    return df


def build_training_table(df):
    df = df.sort_values(["event_date", "user_id", "banner_id"]).reset_index(drop=True)
    global_ctr = df["clicks"].sum() / df["impressions"].sum()
    prior_specs = [
        (["banner_id"], "banner_ctr_prior", 100.0),
        (["user_id"], "user_ctr_prior", 100.0),
        (["subcategory"], "subcategory_ctr_prior", 100.0),
        (["brand"], "brand_ctr_prior", 100.0),
        (["user_id", "subcategory"], "user_subcategory_ctr_prior", 20.0),
        (["user_id", "banner_id"], "user_banner_ctr_prior", 10.0),
    ]
    for group_cols, feature_name, alpha in prior_specs:
        df = add_date_prior_feature(df, group_cols, feature_name, alpha, global_ctr)
    return df, global_ctr, prior_specs


def compute_full_history_tables(df, global_ctr):
    history_specs = {
        "banner": (["banner_id"], "banner_ctr_prior", 100.0),
        "user": (["user_id"], "user_ctr_prior", 100.0),
        "subcategory": (["subcategory"], "subcategory_ctr_prior", 100.0),
        "brand": (["brand"], "brand_ctr_prior", 100.0),
        "user_subcategory": (["user_id", "subcategory"], "user_subcategory_ctr_prior", 20.0),
        "user_banner": (["user_id", "banner_id"], "user_banner_ctr_prior", 10.0),
    }
    tables = {}
    for table_name, (group_cols, feature_name, alpha) in history_specs.items():
        agg = df.groupby(group_cols, as_index=False)[["clicks", "impressions"]].sum()
        agg[feature_name] = (agg["clicks"] + alpha * global_ctr) / (agg["impressions"] + alpha)
        agg[f"{feature_name}_impr"] = agg["impressions"]
        tables[table_name] = agg[group_cols + [feature_name, f"{feature_name}_impr"]]
    return tables, history_specs


def weighted_rmse(y_true, y_pred, weights):
    mse = np.average((y_true - y_pred) ** 2, weights=weights)
    return float(np.sqrt(mse))


def ndcg_at_k(df, pred_col="pred_ctr", label_col="target_ctr", user_col="user_id", k=5):
    scores = []
    for _, g in df.groupby(user_col):
        g = g.sort_values(pred_col, ascending=False).head(k)
        rel = g[label_col].to_numpy()
        if rel.size == 0:
            continue
        discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
        dcg = float((rel * discounts).sum())
        ideal = np.sort(g[label_col].to_numpy())[::-1]
        idcg = float((ideal * discounts).sum())
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def aggregated_logloss_from_logits(logits: torch.Tensor, clicks: torch.Tensor, impressions: torch.Tensor):
    loss_per_row = impressions * torch.nn.functional.softplus(logits) - clicks * logits
    return loss_per_row.sum() / impressions.sum().clamp_min(1.0)


def aggregated_logloss_numpy(pred_ctr, clicks, impressions, eps=1e-8):
    p = np.clip(pred_ctr, eps, 1.0 - eps)
    total_loss = -(clicks * np.log(p) + (impressions - clicks) * np.log(1.0 - p)).sum()
    return float(total_loss / np.clip(impressions.sum(), 1.0, None))


def parse_hidden_dims(hidden_dims_str: str) -> List[int]:
    dims = [int(x.strip()) for x in hidden_dims_str.split(",") if x.strip()]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one layer size")
    return dims


def build_vocab(series: pd.Series) -> Dict[str, int]:
    values = series.fillna("__NA__").astype(str)
    unique = pd.Index(values.unique()).sort_values()
    vocab = {"__UNK__": 0}
    vocab.update({value: idx + 1 for idx, value in enumerate(unique)})
    return vocab


def encode_categorical_frame(df: pd.DataFrame, vocabs: Dict[str, Dict[str, int]], features: List[str]) -> np.ndarray:
    arrays = []
    for feat in features:
        values = df[feat].fillna("__NA__").astype(str)
        mapping = vocabs[feat]
        arrays.append(values.map(lambda x: mapping.get(x, 0)).astype(np.int64).to_numpy())
    return np.stack(arrays, axis=1)


def fill_dense_na(df: pd.DataFrame, dense_features: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in dense_features:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for cat_x, dense_x, _, _ in loader:
            cat_x = cat_x.to(device)
            dense_x = dense_x.to(device)
            logits = model(cat_x, dense_x)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_artifact_metadata(
    *,
    model_type: str,
    global_ctr: float,
    max_date: pd.Timestamp,
    valid_days: int,
    hidden_dims: List[int],
    embedding_dim: int,
    dropout: float,
    history_specs: Dict[str, tuple[list[str], str, float]],
    interactions_csv: str | Path,
    users_csv: str | Path,
    banners_csv: str | Path,
    output_dir: str | Path,
    training_config: dict,
    project_root: Path | None = None,
) -> dict:
    return {
        "model_version": RANKER_MODEL_VERSION,
        "model_type": model_type,
        "artifact_dir": normalize_metadata_path(output_dir, project_root=project_root),
        "feature_cols": FEATURE_COLS,
        "cat_features": CAT_FEATURES,
        "dense_features": DENSE_FEATURES,
        "global_ctr": float(global_ctr),
        "latest_event_date": str(max_date.date()),
        "valid_days": valid_days,
        "hidden_dims": hidden_dims,
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "history_specs": {
            name: {"group_cols": group_cols, "feature_name": feature_name, "alpha": alpha}
            for name, (group_cols, feature_name, alpha) in history_specs.items()
        },
        "user_cols": USER_COLS,
        "banner_cols": BANNER_COLS,
        "default_score_mode": "ctr",
        "training_data": {
            "interactions_csv": normalize_metadata_path(interactions_csv, project_root=project_root),
            "users_csv": normalize_metadata_path(users_csv, project_root=project_root),
            "banners_csv": normalize_metadata_path(banners_csv, project_root=project_root),
        },
        "training_config": training_config,
    }


def main():
    args = parse_args()
    set_seed(args.random_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    print("Loading data...")
    df = load_data(args.interactions_csv, args.users_csv, args.banners_csv)
    df = add_base_features(df)
    df, global_ctr, prior_specs = build_training_table(df)
    df = fill_dense_na(df, DENSE_FEATURES)

    max_date = df["event_date"].max()
    valid_start = max_date - pd.Timedelta(days=args.valid_days - 1)
    train_df = df[df["event_date"] < valid_start].copy()
    valid_df = df[df["event_date"] >= valid_start].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"Time split produced an empty dataset. valid_start={valid_start.date()}")

    vocabs = {feat: build_vocab(train_df[feat]) for feat in CAT_FEATURES}
    cat_cardinalities = {feat: len(vocab) for feat, vocab in vocabs.items()}

    scaler = StandardScaler()
    train_dense = scaler.fit_transform(train_df[DENSE_FEATURES].astype(np.float32))
    valid_dense = scaler.transform(valid_df[DENSE_FEATURES].astype(np.float32))
    train_cat = encode_categorical_frame(train_df, vocabs, CAT_FEATURES)
    valid_cat = encode_categorical_frame(valid_df, vocabs, CAT_FEATURES)

    train_ds = TabularDataset(train_cat, train_dense, train_df["clicks"].to_numpy(dtype=np.float32), train_df["impressions"].to_numpy(dtype=np.float32))
    valid_ds = TabularDataset(valid_cat, valid_dense, valid_df["clicks"].to_numpy(dtype=np.float32), valid_df["impressions"].to_numpy(dtype=np.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DeepFM(
        cat_cardinalities=cat_cardinalities,
        dense_dim=len(DENSE_FEATURES),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        emb_dim=args.emb_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_valid_loss = float("inf")
    best_epoch = -1
    patience_left = args.patience

    print(
        f"Training on {len(train_df):,} rows, validating on {len(valid_df):,} rows. "
        f"Train date max={train_df['event_date'].max().date()}, valid start={valid_start.date()}, device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_impr_sum = 0.0
        for cat_x, dense_x, clicks, impressions in train_loader:
            cat_x = cat_x.to(device)
            dense_x = dense_x.to(device)
            clicks = clicks.to(device)
            impressions = impressions.to(device)
            optimizer.zero_grad()
            logits = model(cat_x, dense_x)
            loss = aggregated_logloss_from_logits(logits, clicks, impressions)
            loss.backward()
            optimizer.step()
            batch_impr = impressions.sum().item()
            train_loss_sum += loss.item() * batch_impr
            train_impr_sum += batch_impr
        train_logloss = train_loss_sum / max(train_impr_sum, 1.0)

        model.eval()
        valid_loss_sum = 0.0
        valid_impr_sum = 0.0
        with torch.no_grad():
            for cat_x, dense_x, clicks, impressions in valid_loader:
                cat_x = cat_x.to(device)
                dense_x = dense_x.to(device)
                clicks = clicks.to(device)
                impressions = impressions.to(device)
                logits = model(cat_x, dense_x)
                loss = aggregated_logloss_from_logits(logits, clicks, impressions)
                batch_impr = impressions.sum().item()
                valid_loss_sum += loss.item() * batch_impr
                valid_impr_sum += batch_impr
        valid_logloss = valid_loss_sum / max(valid_impr_sum, 1.0)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_logloss={train_logloss:.6f} | valid_logloss={valid_logloss:.6f}")

        if valid_logloss < best_valid_loss:
            best_valid_loss = valid_logloss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")

    model.load_state_dict(best_state)
    valid_pred = predict_dataset(model, valid_loader, device)
    valid_df = valid_df.copy()
    valid_df["pred_ctr"] = np.clip(valid_pred, 0.0, 1.0)

    metrics = {
        "global_ctr": float(global_ctr),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_start": str(train_df["event_date"].min().date()),
        "train_end": str(train_df["event_date"].max().date()),
        "valid_start": str(valid_df["event_date"].min().date()),
        "valid_end": str(valid_df["event_date"].max().date()),
        "best_epoch": int(best_epoch),
        "weighted_rmse": weighted_rmse(valid_df["target_ctr"].to_numpy(), valid_df["pred_ctr"].to_numpy(), valid_df["impressions"].to_numpy()),
        "rmse_unweighted": float(np.sqrt(mean_squared_error(valid_df["target_ctr"], valid_df["pred_ctr"]))),
        "aggregated_logloss": aggregated_logloss_numpy(valid_df["pred_ctr"].to_numpy(), valid_df["clicks"].to_numpy(), valid_df["impressions"].to_numpy()),
        "ndcg_at_5": ndcg_at_k(valid_df, k=5),
        "mean_pred_ctr": float(valid_df["pred_ctr"].mean()),
        "mean_actual_ctr": float(valid_df["target_ctr"].mean()),
    }

    print("Validation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "cat_features": CAT_FEATURES,
        "dense_features": DENSE_FEATURES,
        "feature_cols": FEATURE_COLS,
        "cat_cardinalities": cat_cardinalities,
        "embedding_dim": model.embedding_dim,
        "vocabs": vocabs,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "global_ctr": float(global_ctr),
    }
    torch.save(checkpoint, output_dir / "deepfm_model.pt")

    history_tables, history_specs = compute_full_history_tables(df, global_ctr)
    for table_name, table_df in history_tables.items():
        table_df.to_csv(output_dir / f"{table_name}_history.csv.gz", index=False, compression="gzip")

    metadata = build_artifact_metadata(
        model_type="deepfm",
        global_ctr=global_ctr,
        max_date=max_date,
        valid_days=args.valid_days,
        hidden_dims=hidden_dims,
        embedding_dim=args.emb_dim,
        dropout=args.dropout,
        history_specs=history_specs,
        interactions_csv=args.interactions_csv,
        users_csv=args.users_csv,
        banners_csv=args.banners_csv,
        output_dir=output_dir,
        training_config={
            "epochs_requested": args.epochs,
            "epochs_completed": epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_dims": hidden_dims,
            "embedding_dim": args.emb_dim,
            "patience": args.patience,
            "random_seed": args.random_seed,
            "device": str(device),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "best_epoch": int(best_epoch),
        },
        project_root=PROJECT_ROOT,
    )
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved model and artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
