#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Make sibling modules importable when the app is launched from another cwd.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.pipeline.train import (  # noqa: E402
    CAT_FEATURES,
    add_base_features,
    build_training_table,
    compute_full_history_tables,
    load_data,
    ndcg_at_k,
    weighted_rmse,
)
from src.pipeline.inference import (  # noqa: E402
    attach_recent_user_banner_history,
    load_history_tables,
    merge_history_features,
)
from src.pipeline.deepfm import train_deepfm as deepfm_pipeline  # noqa: E402
from src.scripts.pytorch_recsys.inference import recommend_top_n  # noqa: E402

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

DEFAULT_INTERACTIONS = "data/db/banner_interactions.csv"
DEFAULT_USERS = "data/db/users.csv"
DEFAULT_BANNERS = "data/db/banners.csv"
DEFAULT_ARTIFACTS = "ctr_artifacts_streamlit"
DEFAULT_DEEPFM_ARTIFACTS = "deepfm_artifacts"
DEFAULT_RETRIEVAL_ARTIFACTS = "artifacts/pytorch_retrieval"
ARTIFACT_PRESETS = {
    "deepfm": DEFAULT_DEEPFM_ARTIFACTS,
    "catboost": DEFAULT_ARTIFACTS,
    "custom": "",
}


def infer_artifact_preset(artifacts_dir: str) -> str:
    if artifacts_dir == DEFAULT_DEEPFM_ARTIFACTS:
        return "deepfm"
    if artifacts_dir == DEFAULT_ARTIFACTS:
        return "catboost"
    return "custom"


def set_active_artifacts_dir(artifacts_dir: str) -> None:
    st.session_state["artifacts_dir_input"] = artifacts_dir
    st.session_state["artifact_preset"] = infer_artifact_preset(artifacts_dir)


def request_active_artifacts_dir(artifacts_dir: str) -> None:
    st.session_state["pending_artifacts_dir"] = artifacts_dir


def apply_pending_artifacts_dir() -> None:
    pending_artifacts_dir = st.session_state.pop("pending_artifacts_dir", None)
    if pending_artifacts_dir is not None:
        set_active_artifacts_dir(pending_artifacts_dir)


def sync_artifact_preset() -> None:
    preset = st.session_state.get("artifact_preset", "deepfm")
    if preset != "custom":
        st.session_state["artifacts_dir_input"] = ARTIFACT_PRESETS[preset]


def sync_artifact_dir() -> None:
    artifacts_dir = st.session_state.get("artifacts_dir_input", DEFAULT_DEEPFM_ARTIFACTS)
    st.session_state["artifact_preset"] = infer_artifact_preset(artifacts_dir)


@st.cache_data(show_spinner=False)
def preview_csv(path: str, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


@st.cache_data(show_spinner=False)
def load_users(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_banners(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["created_at"])


@st.cache_data(show_spinner=False)
def load_metrics(artifacts_dir: str) -> dict:
    path = Path(artifacts_dir) / "metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metadata(artifacts_dir: str) -> dict:
    path = Path(artifacts_dir) / "metadata.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_type(metadata: dict) -> str:
    return str(metadata.get("model_type", "catboost")).lower()


def train_catboost_model(
    interactions_csv: str,
    users_csv: str,
    banners_csv: str,
    output_dir: str,
    valid_days: int,
    iterations: int,
    learning_rate: float,
    depth: int,
    random_seed: int,
) -> tuple[dict, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(interactions_csv, users_csv, banners_csv)
    df = add_base_features(df)
    df, global_ctr, _ = build_training_table(df)

    max_date = df["event_date"].max()
    valid_start = max_date - pd.Timedelta(days=valid_days - 1)

    train_df = df[df["event_date"] < valid_start].copy()
    valid_df = df[df["event_date"] >= valid_start].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError(f"Time split produced an empty dataset. valid_start={valid_start.date()}")

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=random_seed,
        verbose=100,
    )

    model.fit(
        train_df[FEATURE_COLS],
        train_df["target_ctr"],
        cat_features=CAT_FEATURES,
        sample_weight=train_df["impressions"],
        eval_set=(valid_df[FEATURE_COLS], valid_df["target_ctr"]),
        use_best_model=True,
    )

    valid_df = valid_df.copy()
    valid_df["pred_ctr"] = np.clip(model.predict(valid_df[FEATURE_COLS]), 0.0, 1.0)

    metrics = {
        "global_ctr": float(global_ctr),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_start": str(train_df["event_date"].min().date()),
        "train_end": str(train_df["event_date"].max().date()),
        "valid_start": str(valid_df["event_date"].min().date()),
        "valid_end": str(valid_df["event_date"].max().date()),
        "weighted_rmse": weighted_rmse(
            valid_df["target_ctr"].to_numpy(),
            valid_df["pred_ctr"].to_numpy(),
            valid_df["impressions"].to_numpy(),
        ),
        "rmse_unweighted": float(
            np.sqrt(mean_squared_error(valid_df["target_ctr"], valid_df["pred_ctr"]))
        ),
        "ndcg_at_5": ndcg_at_k(valid_df, k=5),
        "mean_pred_ctr": float(valid_df["pred_ctr"].mean()),
        "mean_actual_ctr": float(valid_df["target_ctr"].mean()),
    }

    model_path = output_path / "ctr_model.cbm"
    model.save_model(str(model_path))

    history_tables, history_specs = compute_full_history_tables(df, global_ctr)
    for table_name, table_df in history_tables.items():
        table_df.to_csv(output_path / f"{table_name}_history.csv.gz", index=False, compression="gzip")

    metadata = {
        "feature_cols": FEATURE_COLS,
        "cat_features": CAT_FEATURES,
        "global_ctr": float(global_ctr),
        "latest_event_date": str(max_date.date()),
        "valid_days": valid_days,
        "history_specs": {
            name: {"group_cols": group_cols, "feature_name": feature_name, "alpha": alpha}
            for name, (group_cols, feature_name, alpha) in history_specs.items()
        },
        "default_score_mode": "ctr",
    }

    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    valid_preview = (
        valid_df[["event_date", "user_id", "banner_id", "target_ctr", "pred_ctr", "impressions"]]
        .sort_values("pred_ctr", ascending=False)
        .head(500)
    )
    valid_preview.to_csv(output_path / "validation_preview.csv", index=False)

    return metrics, output_path


def train_deepfm_model(
    interactions_csv: str,
    users_csv: str,
    banners_csv: str,
    output_dir: str,
    valid_days: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    hidden_dims: str,
    emb_dim: int,
    patience: int,
    random_seed: int,
    device_name: str,
) -> tuple[dict, Path]:
    deepfm_pipeline.set_seed(random_seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hidden_dims_list = deepfm_pipeline.parse_hidden_dims(hidden_dims)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    df = deepfm_pipeline.load_data(interactions_csv, users_csv, banners_csv)
    df = deepfm_pipeline.add_base_features(df)
    df, global_ctr, _ = deepfm_pipeline.build_training_table(df)
    df = deepfm_pipeline.fill_dense_na(df, deepfm_pipeline.DENSE_FEATURES)

    max_date = df["event_date"].max()
    valid_start = max_date - pd.Timedelta(days=valid_days - 1)
    train_df = df[df["event_date"] < valid_start].copy()
    valid_df = df[df["event_date"] >= valid_start].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError(f"Time split produced an empty dataset. valid_start={valid_start.date()}")

    vocabs = {feat: deepfm_pipeline.build_vocab(train_df[feat]) for feat in deepfm_pipeline.CAT_FEATURES}
    cat_cardinalities = {feat: len(vocab) for feat, vocab in vocabs.items()}

    scaler = StandardScaler()
    train_dense = scaler.fit_transform(train_df[deepfm_pipeline.DENSE_FEATURES].astype(np.float32))
    valid_dense = scaler.transform(valid_df[deepfm_pipeline.DENSE_FEATURES].astype(np.float32))
    train_cat = deepfm_pipeline.encode_categorical_frame(train_df, vocabs, deepfm_pipeline.CAT_FEATURES)
    valid_cat = deepfm_pipeline.encode_categorical_frame(valid_df, vocabs, deepfm_pipeline.CAT_FEATURES)

    train_ds = deepfm_pipeline.TabularDataset(
        train_cat,
        train_dense,
        train_df["clicks"].to_numpy(dtype=np.float32),
        train_df["impressions"].to_numpy(dtype=np.float32),
    )
    valid_ds = deepfm_pipeline.TabularDataset(
        valid_cat,
        valid_dense,
        valid_df["clicks"].to_numpy(dtype=np.float32),
        valid_df["impressions"].to_numpy(dtype=np.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = deepfm_pipeline.DeepFM(
        cat_cardinalities=cat_cardinalities,
        dense_dim=len(deepfm_pipeline.DENSE_FEATURES),
        hidden_dims=hidden_dims_list,
        dropout=dropout,
        emb_dim=emb_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_valid_loss = float("inf")
    best_epoch = -1
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        for cat_x, dense_x, clicks, impressions in train_loader:
            cat_x = cat_x.to(device)
            dense_x = dense_x.to(device)
            clicks = clicks.to(device)
            impressions = impressions.to(device)
            optimizer.zero_grad()
            logits = model(cat_x, dense_x)
            loss = deepfm_pipeline.aggregated_logloss_from_logits(logits, clicks, impressions)
            loss.backward()
            optimizer.step()

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
                loss = deepfm_pipeline.aggregated_logloss_from_logits(logits, clicks, impressions)
                batch_impr = impressions.sum().item()
                valid_loss_sum += loss.item() * batch_impr
                valid_impr_sum += batch_impr
        valid_logloss = valid_loss_sum / max(valid_impr_sum, 1.0)

        if valid_logloss < best_valid_loss:
            best_valid_loss = valid_logloss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")

    model.load_state_dict(best_state)
    model.to(device)
    valid_pred = deepfm_pipeline.predict_dataset(model, valid_loader, device)
    valid_df = valid_df.copy()
    valid_df["pred_ctr"] = np.clip(valid_pred, 0.0, 1.0)

    metrics = {
        "model_type": "deepfm",
        "global_ctr": float(global_ctr),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_start": str(train_df["event_date"].min().date()),
        "train_end": str(train_df["event_date"].max().date()),
        "valid_start": str(valid_df["event_date"].min().date()),
        "valid_end": str(valid_df["event_date"].max().date()),
        "best_epoch": int(best_epoch),
        "weighted_rmse": deepfm_pipeline.weighted_rmse(
            valid_df["target_ctr"].to_numpy(),
            valid_df["pred_ctr"].to_numpy(),
            valid_df["impressions"].to_numpy(),
        ),
        "rmse_unweighted": float(np.sqrt(mean_squared_error(valid_df["target_ctr"], valid_df["pred_ctr"]))),
        "aggregated_logloss": deepfm_pipeline.aggregated_logloss_numpy(
            valid_df["pred_ctr"].to_numpy(),
            valid_df["clicks"].to_numpy(),
            valid_df["impressions"].to_numpy(),
        ),
        "ndcg_at_5": deepfm_pipeline.ndcg_at_k(valid_df, k=5),
        "mean_pred_ctr": float(valid_df["pred_ctr"].mean()),
        "mean_actual_ctr": float(valid_df["target_ctr"].mean()),
    }

    checkpoint = {
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "cat_features": deepfm_pipeline.CAT_FEATURES,
        "dense_features": deepfm_pipeline.DENSE_FEATURES,
        "feature_cols": deepfm_pipeline.FEATURE_COLS,
        "cat_cardinalities": cat_cardinalities,
        "embedding_dim": model.embedding_dim,
        "vocabs": vocabs,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "hidden_dims": hidden_dims_list,
        "dropout": dropout,
        "global_ctr": float(global_ctr),
    }
    torch.save(checkpoint, output_path / "deepfm_model.pt")

    history_tables, history_specs = deepfm_pipeline.compute_full_history_tables(df, global_ctr)
    for table_name, table_df in history_tables.items():
        table_df.to_csv(output_path / f"{table_name}_history.csv.gz", index=False, compression="gzip")

    metadata = {
        "model_type": "deepfm",
        "feature_cols": deepfm_pipeline.FEATURE_COLS,
        "cat_features": deepfm_pipeline.CAT_FEATURES,
        "dense_features": deepfm_pipeline.DENSE_FEATURES,
        "global_ctr": float(global_ctr),
        "latest_event_date": str(max_date.date()),
        "valid_days": valid_days,
        "hidden_dims": hidden_dims_list,
        "embedding_dim": emb_dim,
        "dropout": dropout,
        "history_specs": {
            name: {"group_cols": group_cols, "feature_name": feature_name, "alpha": alpha}
            for name, (group_cols, feature_name, alpha) in history_specs.items()
        },
        "user_cols": deepfm_pipeline.USER_COLS,
        "banner_cols": deepfm_pipeline.BANNER_COLS,
        "default_score_mode": "ctr",
    }

    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    valid_preview = (
        valid_df[["event_date", "user_id", "banner_id", "target_ctr", "pred_ctr", "impressions"]]
        .sort_values("pred_ctr", ascending=False)
        .head(500)
    )
    valid_preview.to_csv(output_path / "validation_preview.csv", index=False)

    return metrics, output_path


@st.cache_resource(show_spinner=False)
def load_catboost_model(artifacts_dir: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(Path(artifacts_dir) / "ctr_model.cbm"))
    return model


@st.cache_resource(show_spinner=False)
def load_deepfm_bundle(artifacts_dir: str) -> dict:
    checkpoint = torch.load(Path(artifacts_dir) / "deepfm_model.pt", map_location="cpu")
    model = deepfm_pipeline.DeepFM(
        cat_cardinalities=checkpoint["cat_cardinalities"],
        dense_dim=len(checkpoint["dense_features"]),
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
        emb_dim=checkpoint["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {"model": model, "checkpoint": checkpoint}


def predict_with_deepfm(candidates: pd.DataFrame, artifacts_dir: str) -> np.ndarray:
    bundle = load_deepfm_bundle(artifacts_dir)
    checkpoint = bundle["checkpoint"]
    model = bundle["model"]

    dense_features = checkpoint["dense_features"]
    cat_features = checkpoint["cat_features"]
    candidates = deepfm_pipeline.fill_dense_na(candidates, dense_features)
    dense_values = candidates[dense_features].astype(np.float32).to_numpy()

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    dense_scaled = scaler.transform(dense_values).astype(np.float32)

    cat_encoded = deepfm_pipeline.encode_categorical_frame(candidates, checkpoint["vocabs"], cat_features)
    dataset = deepfm_pipeline.TabularDataset(
        cat_encoded,
        dense_scaled,
        np.zeros(len(candidates), dtype=np.float32),
        np.ones(len(candidates), dtype=np.float32),
    )
    loader = DataLoader(dataset, batch_size=8192, shuffle=False, num_workers=0)
    return deepfm_pipeline.predict_dataset(model, loader, torch.device("cpu"))


def recommend_for_user(
    user_id: str,
    users_csv: str,
    banners_csv: str,
    artifacts_dir: str,
    retrieval_artifacts_dir: Optional[str],
    interactions_csv: Optional[str],
    top_k: int,
    retrieval_top_n: int,
    only_active: bool,
    exclude_seen: bool,
    score_mode: str,
    as_of_date: Optional[str],
    candidate_mode: str,
) -> pd.DataFrame:
    metadata = load_metadata(artifacts_dir)
    model_type = resolve_model_type(metadata)
    users = load_users(users_csv)
    banners = load_banners(banners_csv)

    user_df = users[users["user_id"] == user_id].copy()
    if user_df.empty:
        raise ValueError(f"user_id={user_id!r} not found in users file")

    if only_active:
        banners = banners[banners["is_active"] == 1].copy()

    if as_of_date:
        serve_date = pd.Timestamp(as_of_date)
    else:
        serve_date = pd.Timestamp(metadata["latest_event_date"]) + pd.Timedelta(days=1)

    if candidate_mode == "retrieval + ranking":
        if not retrieval_artifacts_dir:
            raise ValueError("Для режима retrieval + ranking укажите папку retrieval-артефактов.")

        retrieved_banner_ids = recommend_top_n(
            artifact_dir=retrieval_artifacts_dir,
            user_id=user_id,
            top_n=retrieval_top_n,
            exclude_seen=exclude_seen,
            interactions_csv=interactions_csv,
        )
        candidates = banners[banners["banner_id"].isin(retrieved_banner_ids)].copy()
        if candidates.empty:
            raise ValueError("Retrieval не вернул кандидатов для ранжирования.")

        retrieval_rank = {
            banner_id: rank for rank, banner_id in enumerate(retrieved_banner_ids, start=1)
        }
        candidates["retrieval_rank"] = candidates["banner_id"].map(retrieval_rank)
        user_row = user_df.iloc[0]
        for column in user_df.columns:
            candidates[column] = user_row[column]
        candidates["event_date"] = serve_date
    else:
        user_df["__k"] = 1
        banners["__k"] = 1
        candidates = banners.merge(user_df, on="__k", how="inner").drop(columns="__k")
        candidates["event_date"] = serve_date

    candidates = add_base_features(candidates)
    history = load_history_tables(artifacts_dir)
    candidates = merge_history_features(candidates, history, metadata)
    candidates = attach_recent_user_banner_history(candidates, interactions_csv, user_id)

    for col in ["served_impressions_total", "served_clicks_total"]:
        if col not in candidates.columns:
            candidates[col] = 0

    if exclude_seen and candidate_mode == "all banners":
        candidates = candidates[candidates["served_impressions_total"] == 0].copy()

    candidates["fatigue_penalty"] = 1.0 / (1.0 + np.log1p(candidates["served_impressions_total"]))
    candidates["repeat_click_bonus"] = np.where(candidates["served_clicks_total"] > 0, 1.05, 1.0)

    feature_cols = metadata["feature_cols"]
    if model_type == "deepfm":
        candidates["pred_ctr"] = np.clip(
            predict_with_deepfm(candidates[feature_cols], artifacts_dir),
            0.0,
            1.0,
        )
    else:
        model = load_catboost_model(artifacts_dir)
        candidates["pred_ctr"] = np.clip(model.predict(candidates[feature_cols]), 0.0, 1.0)

    if score_mode == "ctr":
        candidates["final_score"] = (
            candidates["pred_ctr"]
            * candidates["fatigue_penalty"]
            * candidates["repeat_click_bonus"]
        )
    else:
        candidates["final_score"] = (
            candidates["pred_ctr"]
            * candidates["cpm_bid"]
            * candidates["quality_score"]
            * candidates["fatigue_penalty"]
            * candidates["repeat_click_bonus"]
        )

    result_cols = [
        "banner_id",
        "brand",
        "category",
        "subcategory",
        "banner_format",
        "campaign_goal",
        "pred_ctr",
        "final_score",
        "cpm_bid",
        "quality_score",
        "age_match",
        "gender_match",
        "interest_match_any",
        "interest_match_count",
        "banner_ctr_prior",
        "user_ctr_prior",
        "user_subcategory_ctr_prior",
        "user_banner_ctr_prior",
        "served_impressions_total",
        "served_clicks_total",
        "is_active",
    ]

    result = (
        candidates.sort_values(
            ["final_score", "pred_ctr", "retrieval_rank"] if "retrieval_rank" in candidates.columns else ["final_score", "pred_ctr"],
            ascending=[False, False, True] if "retrieval_rank" in candidates.columns else [False, False],
        )
        .head(top_k)[result_cols]
        .reset_index(drop=True)
    )
    return result


def render_sidebar() -> dict:
    st.sidebar.header("Данные и артефакты")
    interactions_csv = st.sidebar.text_input("banner_interactions.csv", DEFAULT_INTERACTIONS)
    users_csv = st.sidebar.text_input("users.csv", DEFAULT_USERS)
    banners_csv = st.sidebar.text_input("banners.csv", DEFAULT_BANNERS)

    preset_options = ["deepfm", "catboost", "custom"]
    if "artifacts_dir_input" not in st.session_state:
        st.session_state["artifacts_dir_input"] = DEFAULT_DEEPFM_ARTIFACTS
    if "artifact_preset" not in st.session_state:
        st.session_state["artifact_preset"] = infer_artifact_preset(
            st.session_state["artifacts_dir_input"]
        )

    artifact_preset = st.sidebar.selectbox(
        "Пресет артефактов",
        options=preset_options,
        key="artifact_preset",
        on_change=sync_artifact_preset,
        format_func=lambda value: {
            "deepfm": "DeepFM",
            "catboost": "CatBoost",
            "custom": "Custom",
        }[value],
    )

    preset_artifacts_dir = ARTIFACT_PRESETS[artifact_preset]
    artifacts_dir = st.sidebar.text_input(
        "Папка артефактов модели",
        key="artifacts_dir_input",
        on_change=sync_artifact_dir,
    )
    retrieval_artifacts_dir = st.sidebar.text_input(
        "Папка retrieval-артефактов",
        DEFAULT_RETRIEVAL_ARTIFACTS,
    )

    st.sidebar.caption("Можно оставить пути по умолчанию для ваших текущих файлов в /data/db.")
    if artifact_preset != "custom":
        st.sidebar.caption(f"Выбран пресет `{artifact_preset}`: `{preset_artifacts_dir}`")
    metadata_path = Path(artifacts_dir) / "metadata.json"
    if metadata_path.exists():
        try:
            detected_model_type = resolve_model_type(load_metadata(artifacts_dir))
            st.sidebar.caption(f"Тип артефактов: `{detected_model_type}`")
        except Exception:
            st.sidebar.caption("Не удалось прочитать metadata.json")

    return {
        "interactions_csv": interactions_csv,
        "users_csv": users_csv,
        "banners_csv": banners_csv,
        "artifacts_dir": artifacts_dir,
        "retrieval_artifacts_dir": retrieval_artifacts_dir,
    }


def render_previews(cfg: dict) -> None:
    with st.expander("Быстрый просмотр CSV", expanded=False):
        cols = st.columns(3)
        files = [
            ("interactions", cfg["interactions_csv"]),
            ("users", cfg["users_csv"]),
            ("banners", cfg["banners_csv"]),
        ]
        for col, (title, path) in zip(cols, files):
            with col:
                st.markdown(f"**{title}**")
                try:
                    st.dataframe(preview_csv(path), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(str(e))


def train_tab(cfg: dict) -> None:
    st.subheader("Обучение модели")
    train_model_type = st.radio(
        "Архитектура",
        options=["catboost", "deepfm"],
        horizontal=True,
        help="CatBoost для быстрого baseline, DeepFM для нейросеточной ranking-модели.",
    )

    valid_days = st.number_input("Окно validation, дней", min_value=3, max_value=60, value=14)
    random_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42)

    if train_model_type == "catboost":
        c1, c2, c3 = st.columns(3)
        with c1:
            iterations = st.number_input("Итерации CatBoost", min_value=50, max_value=5000, value=400, step=50)
        with c2:
            learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
        with c3:
            depth = st.number_input("Depth", min_value=3, max_value=12, value=8)
        suggested_output_dir = DEFAULT_ARTIFACTS
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=8)
        with c2:
            batch_size = st.number_input("Batch size", min_value=128, max_value=65536, value=4096, step=128)
        with c3:
            learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0005, format="%.4f")
        with c4:
            emb_dim = st.number_input("Embedding dim", min_value=4, max_value=128, value=16, step=4)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=0.1, value=0.000001, step=0.000001, format="%.6f")
        with c6:
            dropout = st.number_input("Dropout", min_value=0.0, max_value=0.9, value=0.1, step=0.05, format="%.2f")
        with c7:
            patience = st.number_input("Patience", min_value=1, max_value=20, value=2)
        with c8:
            device_name = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        hidden_dims = st.text_input("Hidden dims", value="256,128,64")
        suggested_output_dir = DEFAULT_DEEPFM_ARTIFACTS

    output_dir = st.text_input("Папка для сохранения артефактов", value=cfg.get("artifacts_dir") or suggested_output_dir)

    if st.button("Обучить и сохранить артефакты", type="primary"):
        with st.spinner("Идёт обучение модели..."):
            if train_model_type == "catboost":
                metrics, output_path = train_catboost_model(
                    interactions_csv=cfg["interactions_csv"],
                    users_csv=cfg["users_csv"],
                    banners_csv=cfg["banners_csv"],
                    output_dir=output_dir,
                    valid_days=int(valid_days),
                    iterations=int(iterations),
                    learning_rate=float(learning_rate),
                    depth=int(depth),
                    random_seed=int(random_seed),
                )
            else:
                metrics, output_path = train_deepfm_model(
                    interactions_csv=cfg["interactions_csv"],
                    users_csv=cfg["users_csv"],
                    banners_csv=cfg["banners_csv"],
                    output_dir=output_dir,
                    valid_days=int(valid_days),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    dropout=float(dropout),
                    hidden_dims=hidden_dims,
                    emb_dim=int(emb_dim),
                    patience=int(patience),
                    random_seed=int(random_seed),
                    device_name=device_name,
                )
            st.session_state["last_artifacts_dir"] = str(output_path)

        st.success(f"Артефакты сохранены в: {output_path}")
        metric_cols = st.columns(5)
        metric_cols[0].metric("weighted RMSE", f"{metrics['weighted_rmse']:.4f}")
        metric_cols[1].metric("RMSE", f"{metrics['rmse_unweighted']:.4f}")
        metric_cols[2].metric(
            "Logloss" if "aggregated_logloss" in metrics else "NDCG@5",
            f"{metrics.get('aggregated_logloss', metrics['ndcg_at_5']):.4f}",
        )
        metric_cols[3].metric("mean pred CTR", f"{metrics['mean_pred_ctr']:.4f}")
        metric_cols[4].metric("mean actual CTR", f"{metrics['mean_actual_ctr']:.4f}")

        if "aggregated_logloss" in metrics:
            st.caption(f"NDCG@5: {metrics['ndcg_at_5']:.4f}")

        st.json(metrics)

        preview_path = Path(output_path) / "validation_preview.csv"
        if preview_path.exists():
            st.markdown("**Топ validation-предсказания**")
            st.dataframe(pd.read_csv(preview_path).head(50), use_container_width=True, hide_index=True)


def recommend_tab(cfg: dict) -> None:
    st.subheader("Рекомендации баннеров для пользователя")

    quick_switch_cols = st.columns(2)
    with quick_switch_cols[0]:
        if st.button("Использовать DeepFM", use_container_width=True):
            request_active_artifacts_dir(DEFAULT_DEEPFM_ARTIFACTS)
            st.rerun()
    with quick_switch_cols[1]:
        if st.button("Использовать CatBoost", use_container_width=True):
            request_active_artifacts_dir(DEFAULT_ARTIFACTS)
            st.rerun()

    metadata = load_metadata(cfg["artifacts_dir"])
    model_type = resolve_model_type(metadata)
    st.caption(f"Активная модель ранжирования: `{model_type}`")

    users_df = load_users(cfg["users_csv"])
    default_user = str(users_df["user_id"].iloc[0]) if not users_df.empty else ""

    c1, c2, c3 = st.columns(3)
    with c1:
        user_id = st.text_input("user_id", value=default_user)
    with c2:
        top_k = st.number_input("top_k", min_value=1, max_value=100, value=10)
    with c3:
        default_score_mode = metadata.get("default_score_mode", "ctr")
        score_mode = st.selectbox(
            "score_mode",
            options=["ctr", "value"],
            index=0 if default_score_mode == "ctr" else 1,
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        only_active = st.checkbox("Только активные баннеры", value=True)
    with c5:
        exclude_seen = st.checkbox("Исключить уже показанные", value=True)
    with c6:
        as_of_date = st.text_input("Дата показа (YYYY-MM-DD), optional", value="")

    c7, c8 = st.columns(2)
    with c7:
        candidate_mode = st.radio(
            "Режим кандидатов",
            options=["all banners", "retrieval + ranking"],
            horizontal=True,
        )
    with c8:
        retrieval_top_n = st.number_input(
            "retrieval_top_n",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            disabled=candidate_mode != "retrieval + ranking",
        )

    if st.button("Построить рекомендации"):
        with st.spinner("Считаю рекомендации..."):
            recs = recommend_for_user(
                user_id=user_id,
                users_csv=cfg["users_csv"],
                banners_csv=cfg["banners_csv"],
                artifacts_dir=cfg["artifacts_dir"],
                retrieval_artifacts_dir=cfg["retrieval_artifacts_dir"],
                interactions_csv=cfg["interactions_csv"],
                top_k=int(top_k),
                retrieval_top_n=int(retrieval_top_n),
                only_active=only_active,
                exclude_seen=exclude_seen,
                score_mode=score_mode,
                as_of_date=as_of_date or None,
                candidate_mode=candidate_mode,
            )

        st.dataframe(recs, use_container_width=True, hide_index=True)
        csv_bytes = recs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Скачать рекомендации CSV",
            data=csv_bytes,
            file_name=f"recs_{user_id}.csv",
            mime="text/csv",
        )


def artifacts_tab(cfg: dict) -> None:
    st.subheader("Артефакты модели")
    art_dir = Path(cfg["artifacts_dir"])

    if not art_dir.exists():
        st.info("Папка артефактов пока не существует. Сначала обучите модель.")
        return

    files = sorted([p for p in art_dir.glob("*") if p.is_file()])
    if not files:
        st.info("В папке пока нет файлов.")
        return

    rows = []
    for p in files:
        rows.append({
            "file": p.name,
            "size_kb": round(p.stat().st_size / 1024, 2),
            "path": str(p),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    metrics_path = art_dir / "metrics.json"
    if metrics_path.exists():
        st.markdown("**metrics.json**")
        st.json(load_metrics(cfg["artifacts_dir"]))

    metadata_path = art_dir / "metadata.json"
    if metadata_path.exists():
        st.markdown("**metadata.json**")
        metadata = load_metadata(cfg["artifacts_dir"])
        st.json(metadata)
        st.caption(f"Определённый тип модели: `{resolve_model_type(metadata)}`")

    selected = st.selectbox("Посмотреть файл", options=[p.name for p in files])
    selected_path = art_dir / selected
    if selected_path.suffix == ".json":
        st.code(selected_path.read_text(encoding="utf-8"), language="json")
    elif selected_path.suffix in {".csv", ".gz"}:
        try:
            df = pd.read_csv(selected_path)
            st.dataframe(df.head(200), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(str(e))
    else:
        st.caption(str(selected_path))


def main() -> None:
    st.set_page_config(page_title="Banner CTR / Ranking", layout="wide")
    st.title("Banner CTR / Ranking Studio")
    st.caption("Streamlit-приложение для обучения CatBoost/DeepFM ранжирования и выдачи top-K баннеров пользователю.")

    apply_pending_artifacts_dir()
    cfg = render_sidebar()
    render_previews(cfg)

    tabs = st.tabs(["Обучение", "Рекомендации", "Артефакты"]) 
    with tabs[0]:
        train_tab(cfg)
    with tabs[1]:
        recommend_tab(cfg)
    with tabs[2]:
        artifacts_tab(cfg)

    st.markdown("---")
    st.markdown(
        "**Запуск локально:** `streamlit run app_streamlit.py`"
    )


if __name__ == "__main__":
    main()
