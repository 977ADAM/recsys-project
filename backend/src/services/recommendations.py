from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from backend.src.core.config import Settings
from backend.src.core.errors.common import EntityNotFoundError, InvalidRequestError
from backend.src.schemas.recommendations import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from backend.src.services.retrieval import RetrievalService
from src.ranker.artifacts import (
    default_ranker_artifacts_path,
    legacy_ranker_artifacts_paths,
)
from src.ranker.inference import (
    add_base_features,
    attach_recent_user_banner_history,
    load_history_tables,
    merge_history_features,
)
from src.ranker.deepfm import train_deepfm as deepfm_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INTERACTIONS = "data/db/banner_interactions.csv"
DEFAULT_USERS = "data/db/users.csv"
DEFAULT_BANNERS = "data/db/banners.csv"


def _resolve_path(project_root: Path, raw_path: str | None, fallback: str) -> Path:
    path = Path(raw_path or fallback)
    if not path.is_absolute():
        path = project_root / path
    return path


def _file_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def _train_default_deepfm_artifacts(
    *,
    project_root: Path,
    output_dir: Path,
    interactions_csv: Path,
    users_csv: Path,
    banners_csv: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    deepfm_pipeline.set_seed(42)
    df = deepfm_pipeline.load_data(str(interactions_csv), str(users_csv), str(banners_csv))
    df = deepfm_pipeline.add_base_features(df)
    df, global_ctr, _ = deepfm_pipeline.build_training_table(df)
    df = deepfm_pipeline.fill_dense_na(df, deepfm_pipeline.DENSE_FEATURES)

    max_date = df["event_date"].max()
    valid_start = max_date - pd.Timedelta(days=13)
    train_df = df[df["event_date"] < valid_start].copy()
    valid_df = df[df["event_date"] >= valid_start].copy()
    if train_df.empty or valid_df.empty:
        raise InvalidRequestError(
            f"Time split produced an empty dataset while bootstrapping ranking artifacts. valid_start={valid_start.date()}"
        )

    vocabs = {
        feat: deepfm_pipeline.build_vocab(train_df[feat])
        for feat in deepfm_pipeline.CAT_FEATURES
    }
    cat_cardinalities = {feat: len(vocab) for feat, vocab in vocabs.items()}

    scaler = StandardScaler()
    train_dense = scaler.fit_transform(train_df[deepfm_pipeline.DENSE_FEATURES].astype(np.float32))
    valid_dense = scaler.transform(valid_df[deepfm_pipeline.DENSE_FEATURES].astype(np.float32))
    train_cat = deepfm_pipeline.encode_categorical_frame(
        train_df,
        vocabs,
        deepfm_pipeline.CAT_FEATURES,
    )
    valid_cat = deepfm_pipeline.encode_categorical_frame(
        valid_df,
        vocabs,
        deepfm_pipeline.CAT_FEATURES,
    )

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
    train_loader = DataLoader(train_ds, batch_size=8192, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=8192, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    hidden_dims = [128, 64]
    dropout = 0.1
    emb_dim = 16
    model = deepfm_pipeline.DeepFM(
        cat_cardinalities=cat_cardinalities,
        dense_dim=len(deepfm_pipeline.DENSE_FEATURES),
        hidden_dims=hidden_dims,
        dropout=dropout,
        emb_dim=emb_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)

    best_state = None
    best_valid_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, 3):
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
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise InvalidRequestError("Failed to bootstrap DeepFM artifacts.")

    model.load_state_dict(best_state)
    valid_pred = deepfm_pipeline.predict_dataset(model, valid_loader, device)
    valid_df = valid_df.copy()
    valid_df["pred_ctr"] = np.clip(valid_pred, 0.0, 1.0)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "cat_features": deepfm_pipeline.CAT_FEATURES,
        "dense_features": deepfm_pipeline.DENSE_FEATURES,
        "feature_cols": deepfm_pipeline.FEATURE_COLS,
        "cat_cardinalities": cat_cardinalities,
        "embedding_dim": emb_dim,
        "vocabs": vocabs,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "global_ctr": float(global_ctr),
    }
    torch.save(checkpoint, output_dir / "deepfm_model.pt")

    history_tables, history_specs = deepfm_pipeline.compute_full_history_tables(df, global_ctr)
    for table_name, table_df in history_tables.items():
        table_df.to_csv(
            output_dir / f"{table_name}_history.csv.gz",
            index=False,
            compression="gzip",
        )

    metrics = {
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
        "rmse_unweighted": float(
            np.sqrt(mean_squared_error(valid_df["target_ctr"], valid_df["pred_ctr"]))
        ),
        "aggregated_logloss": deepfm_pipeline.aggregated_logloss_numpy(
            valid_df["pred_ctr"].to_numpy(),
            valid_df["clicks"].to_numpy(),
            valid_df["impressions"].to_numpy(),
        ),
        "ndcg_at_5": deepfm_pipeline.ndcg_at_k(valid_df, k=5),
        "mean_pred_ctr": float(valid_df["pred_ctr"].mean()),
        "mean_actual_ctr": float(valid_df["target_ctr"].mean()),
    }
    metadata = deepfm_pipeline.build_artifact_metadata(
        model_type="deepfm",
        global_ctr=global_ctr,
        max_date=max_date,
        valid_days=14,
        hidden_dims=hidden_dims,
        embedding_dim=emb_dim,
        dropout=dropout,
        history_specs=history_specs,
        interactions_csv=interactions_csv,
        users_csv=users_csv,
        banners_csv=banners_csv,
        output_dir=output_dir,
        training_config={
            "epochs_requested": 2,
            "epochs_completed": epoch,
            "batch_size": 8192,
            "learning_rate": 1e-3,
            "weight_decay": 1e-6,
            "dropout": dropout,
            "hidden_dims": hidden_dims,
            "embedding_dim": emb_dim,
            "patience": None,
            "random_seed": 42,
            "device": str(device),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "best_epoch": int(best_epoch),
            "bootstrap_source": "backend_default_bootstrap",
        },
        project_root=project_root,
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    return output_dir


def _resolve_artifacts_path(
    project_root: Path,
    raw_path: str | None,
    interactions_csv: Path,
    users_csv: Path,
    banners_csv: Path,
) -> Path:
    if raw_path is not None:
        return _resolve_path(project_root, raw_path, raw_path)

    default_path = default_ranker_artifacts_path(project_root)
    for candidate in [default_path, *legacy_ranker_artifacts_paths(project_root)]:
        if (candidate / "metadata.json").exists():
            return candidate

    return _train_default_deepfm_artifacts(
        project_root=project_root,
        output_dir=default_path,
        interactions_csv=interactions_csv,
        users_csv=users_csv,
        banners_csv=banners_csv,
    )


@lru_cache(maxsize=8)
def _load_metadata_cached(path_str: str, _mtime_ns: int) -> dict:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _load_metadata(artifacts_dir: str) -> dict:
    path = Path(artifacts_dir) / "metadata.json"
    return _load_metadata_cached(str(path), _file_mtime_ns(path))


def _resolve_model_type(metadata: dict) -> str:
    return str(metadata.get("model_type", "catboost")).lower()


@lru_cache(maxsize=4)
def _load_catboost_model_cached(model_path_str: str, _mtime_ns: int) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(model_path_str)
    return model


def _load_catboost_model(artifacts_dir: str) -> CatBoostRegressor:
    model_path = Path(artifacts_dir) / "ctr_model.cbm"
    return _load_catboost_model_cached(str(model_path), _file_mtime_ns(model_path))


@lru_cache(maxsize=4)
def _load_deepfm_bundle_cached(model_path_str: str, _mtime_ns: int) -> dict:
    checkpoint = torch.load(Path(model_path_str), map_location="cpu")
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


def _load_deepfm_bundle(artifacts_dir: str) -> dict:
    model_path = Path(artifacts_dir) / "deepfm_model.pt"
    return _load_deepfm_bundle_cached(str(model_path), _file_mtime_ns(model_path))


def reset_ranking_caches() -> None:
    _load_metadata_cached.cache_clear()
    _load_catboost_model_cached.cache_clear()
    _load_deepfm_bundle_cached.cache_clear()


def _predict_with_deepfm(candidates: pd.DataFrame, artifacts_dir: str) -> np.ndarray:
    bundle = _load_deepfm_bundle(artifacts_dir)
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


def _build_candidate_pool(
    *,
    request: RecommendationRequest,
    retrieval_service: RetrievalService,
    user_df: pd.DataFrame,
    banners: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    if request.retrieval_artifacts_dir:
        retrieval_result = retrieval_service.get_candidates(
            request=retrieval_service_request(
                user_id=request.user_id,
                top_k=request.retrieval_top_n,
                exclude_seen=request.exclude_seen,
                only_active=request.only_active,
                interactions_csv=request.interactions_csv,
                banners_csv=request.banners_csv,
                artifacts_dir=request.retrieval_artifacts_dir,
            )
        )
        retrieved_banner_ids = [item.banner_id for item in retrieval_result.items]
        retrieval_rank = {item.banner_id: item.retrieval_rank for item in retrieval_result.items}
        retrieval_score = {item.banner_id: item.retrieval_score for item in retrieval_result.items}

        candidates = banners[banners["banner_id"].isin(retrieved_banner_ids)].copy()
        if candidates.empty:
            raise InvalidRequestError("Retrieval returned no candidate banners for reranking.")

        candidates["retrieval_rank"] = candidates["banner_id"].map(retrieval_rank)
        candidates["retrieval_score"] = candidates["banner_id"].map(retrieval_score)
        user_row = user_df.iloc[0]
        for column in user_df.columns:
            candidates[column] = user_row[column]
        candidates["event_date"] = as_of_date
        return candidates, "retrieval + ranking"

    user_df = user_df.copy()
    banners = banners.copy()
    user_df["__k"] = 1
    banners["__k"] = 1
    candidates = banners.merge(user_df, on="__k", how="inner").drop(columns="__k")
    candidates["event_date"] = as_of_date
    return candidates, "all banners"


def retrieval_service_request(
    *,
    user_id: str,
    top_k: int,
    exclude_seen: bool,
    only_active: bool,
    interactions_csv: str | None,
    banners_csv: str | None,
    artifacts_dir: str | None,
):
    from backend.src.schemas.retrieval import RetrievalRequest

    return RetrievalRequest(
        user_id=user_id,
        top_k=top_k,
        exclude_seen=exclude_seen,
        only_active=only_active,
        interactions_csv=interactions_csv,
        banners_csv=banners_csv,
        artifacts_dir=artifacts_dir,
    )


def recommend_banners(
    request: RecommendationRequest,
    settings: Settings,
    retrieval_service: RetrievalService,
) -> RecommendationResponse:
    users_csv = _resolve_path(settings.project_root, request.users_csv, DEFAULT_USERS)
    banners_csv = _resolve_path(settings.project_root, request.banners_csv, DEFAULT_BANNERS)
    interactions_csv = (
        _resolve_path(settings.project_root, request.interactions_csv, DEFAULT_INTERACTIONS)
        if request.exclude_seen or request.interactions_csv is not None
        else None
    )
    training_interactions_csv = _resolve_path(
        settings.project_root,
        request.interactions_csv,
        DEFAULT_INTERACTIONS,
    )
    artifacts_dir = _resolve_artifacts_path(
        settings.project_root,
        request.artifacts_dir,
        training_interactions_csv,
        users_csv,
        banners_csv,
    )

    try:
        metadata = _load_metadata(str(artifacts_dir))
        model_type = _resolve_model_type(metadata)
        users = pd.read_csv(users_csv)
        banners = pd.read_csv(banners_csv, parse_dates=["created_at"])
    except FileNotFoundError as exc:
        raise InvalidRequestError(str(exc)) from exc

    user_df = users[users["user_id"] == request.user_id].copy()
    if user_df.empty:
        raise EntityNotFoundError(f"User with user_id={request.user_id} not found in dataset")

    if request.only_active:
        banners = banners[banners["is_active"] == 1].copy()

    if request.as_of_date is not None:
        as_of_date = pd.Timestamp(request.as_of_date)
    else:
        as_of_date = pd.Timestamp(metadata["latest_event_date"]) + pd.Timedelta(days=1)

    candidates, candidate_mode = _build_candidate_pool(
        request=request,
        retrieval_service=retrieval_service,
        user_df=user_df,
        banners=banners,
        as_of_date=as_of_date,
    )

    candidates = add_base_features(candidates)
    history = load_history_tables(str(artifacts_dir))
    candidates = merge_history_features(candidates, history, metadata)
    candidates = attach_recent_user_banner_history(
        candidates,
        str(interactions_csv) if interactions_csv is not None else None,
        request.user_id,
    )

    for column in ["served_impressions_total", "served_clicks_total"]:
        if column not in candidates.columns:
            candidates[column] = 0

    if request.exclude_seen and candidate_mode == "all banners":
        candidates = candidates[candidates["served_impressions_total"] == 0].copy()

    if candidates.empty:
        raise InvalidRequestError("No candidate banners available for this request.")

    candidates["fatigue_penalty"] = 1.0 / (1.0 + np.log1p(candidates["served_impressions_total"]))
    candidates["repeat_click_bonus"] = np.where(candidates["served_clicks_total"] > 0, 1.05, 1.0)

    feature_cols = metadata["feature_cols"]
    if model_type == "deepfm":
        candidates["pred_ctr"] = np.clip(
            _predict_with_deepfm(candidates[feature_cols], str(artifacts_dir)),
            0.0,
            1.0,
        )
    else:
        model = _load_catboost_model(str(artifacts_dir))
        candidates["pred_ctr"] = np.clip(model.predict(candidates[feature_cols]), 0.0, 1.0)

    if request.score_mode == "ctr":
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

    sort_columns = ["final_score", "pred_ctr"]
    ascending = [False, False]
    if "retrieval_rank" in candidates.columns:
        sort_columns.append("retrieval_rank")
        ascending.append(True)

    result_columns = [
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
    if "landing_page" in candidates.columns:
        result_columns.append("landing_page")
    if "retrieval_rank" in candidates.columns:
        result_columns.append("retrieval_rank")
    if "retrieval_score" in candidates.columns:
        result_columns.append("retrieval_score")

    result = (
        candidates.sort_values(sort_columns, ascending=ascending)
        .head(request.top_k)[result_columns]
        .reset_index(drop=True)
    )

    items = [
        RecommendationItem.model_validate(item)
        for item in result.to_dict(orient="records")
    ]
    retrieval_service.record_served_banners(
        request.user_id,
        [item.banner_id for item in items],
    )
    return RecommendationResponse(
        user_id=request.user_id,
        as_of_date=as_of_date.date(),
        score_mode=request.score_mode,
        candidate_mode=candidate_mode,
        model_type=model_type,
        items=items,
    )
