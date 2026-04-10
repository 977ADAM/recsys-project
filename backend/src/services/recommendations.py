from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from backend.src.core.config import Settings
from backend.src.core.errors.common import EntityNotFoundError, InvalidRequestError
from backend.src.schemas.recommendations import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from src.pipeline.inference import (
    add_base_features,
    attach_recent_user_banner_history,
    load_history_tables,
    merge_history_features,
)
from src.pipeline.deepfm import train_deepfm as deepfm_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "src" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.scripts.pytorch_recsys.inference import recommend_top_n  # noqa: E402

DEFAULT_INTERACTIONS = "data/db/banner_interactions.csv"
DEFAULT_USERS = "data/db/users.csv"
DEFAULT_BANNERS = "data/db/banners.csv"
DEFAULT_CTR_ARTIFACTS = "ctr_artifacts"
DEFAULT_DEEPFM_ARTIFACTS = "deepfm_artifacts"


def _resolve_path(project_root: Path, raw_path: str | None, fallback: str) -> Path:
    path = Path(raw_path or fallback)
    if not path.is_absolute():
        path = project_root / path
    return path


def _resolve_artifacts_path(project_root: Path, raw_path: str | None) -> Path:
    if raw_path is not None:
        return _resolve_path(project_root, raw_path, raw_path)

    deepfm_path = _resolve_path(project_root, None, DEFAULT_DEEPFM_ARTIFACTS)
    if (deepfm_path / "metadata.json").exists():
        return deepfm_path

    ctr_path = _resolve_path(project_root, None, DEFAULT_CTR_ARTIFACTS)
    if (ctr_path / "metadata.json").exists():
        return ctr_path

    raise InvalidRequestError(
        "Model artifacts were not found. Set artifacts_dir explicitly or create metadata.json in deepfm_artifacts or ctr_artifacts."
    )


@lru_cache(maxsize=8)
def _load_metadata(artifacts_dir: str) -> dict:
    path = Path(artifacts_dir) / "metadata.json"
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _resolve_model_type(metadata: dict) -> str:
    return str(metadata.get("model_type", "catboost")).lower()


@lru_cache(maxsize=4)
def _load_catboost_model(artifacts_dir: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(Path(artifacts_dir) / "ctr_model.cbm"))
    return model


@lru_cache(maxsize=4)
def _load_deepfm_bundle(artifacts_dir: str) -> dict:
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
    user_df: pd.DataFrame,
    banners: pd.DataFrame,
    as_of_date: pd.Timestamp,
    interactions_csv: Path | None,
) -> tuple[pd.DataFrame, str]:
    if request.retrieval_artifacts_dir:
        retrieved_banner_ids = recommend_top_n(
            artifact_dir=str(_resolve_path(PROJECT_ROOT, request.retrieval_artifacts_dir, request.retrieval_artifacts_dir)),
            user_id=request.user_id,
            top_n=request.retrieval_top_n,
            exclude_seen=request.exclude_seen,
            interactions_csv=str(interactions_csv) if interactions_csv is not None else None,
        )

        candidates = banners[banners["banner_id"].isin(retrieved_banner_ids)].copy()
        if candidates.empty:
            raise InvalidRequestError("Retrieval returned no candidate banners for reranking.")

        retrieval_rank = {
            banner_id: rank for rank, banner_id in enumerate(retrieved_banner_ids, start=1)
        }
        candidates["retrieval_rank"] = candidates["banner_id"].map(retrieval_rank)
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


def recommend_banners(
    request: RecommendationRequest,
    settings: Settings,
) -> RecommendationResponse:
    artifacts_dir = _resolve_artifacts_path(settings.project_root, request.artifacts_dir)
    users_csv = _resolve_path(settings.project_root, request.users_csv, DEFAULT_USERS)
    banners_csv = _resolve_path(settings.project_root, request.banners_csv, DEFAULT_BANNERS)
    interactions_csv = (
        _resolve_path(settings.project_root, request.interactions_csv, DEFAULT_INTERACTIONS)
        if request.exclude_seen or request.interactions_csv is not None
        else None
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
        user_df=user_df,
        banners=banners,
        as_of_date=as_of_date,
        interactions_csv=interactions_csv,
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

    result = (
        candidates.sort_values(sort_columns, ascending=ascending)
        .head(request.top_k)[result_columns]
        .reset_index(drop=True)
    )

    items = [
        RecommendationItem.model_validate(item)
        for item in result.to_dict(orient="records")
    ]
    return RecommendationResponse(
        user_id=request.user_id,
        as_of_date=as_of_date.date(),
        score_mode=request.score_mode,
        candidate_mode=candidate_mode,
        model_type=model_type,
        items=items,
    )
