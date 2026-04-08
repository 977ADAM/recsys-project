from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from fastapi import FastAPI
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.schemas.recommendation import RecommendationItem, RecommendationRequest, RecommendationResponse  # noqa: E402
from src.pipeline.deepfm import train_deepfm as deepfm_pipeline  # noqa: E402
from src.pipeline.inference import (  # noqa: E402
    attach_recent_user_banner_history,
    load_history_tables,
    merge_history_features,
)
from src.pipeline.train import add_base_features  # noqa: E402
from src.scripts.pytorch_recsys.inference import recommend_top_n  # noqa: E402

INTERACTIONS_CSV = "data/db/banner_interactions.csv"
USERS_CSV = "data/db/users.csv"
BANNERS_CSV = "data/db/banners.csv"
ARTIFACTS_DIR = "deepfm_artifacts"
RETRIEVAL_ARTIFACTS_DIR = "artifacts/pytorch_retrieval"


def _resolve_model_type(metadata: dict) -> str:
    return str(metadata.get("model_type", "catboost")).lower()


@lru_cache(maxsize=8)
def _load_metadata_cached(artifacts_dir: str) -> dict:
    with open(Path(artifacts_dir) / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def _load_users_cached(users_csv: str) -> pd.DataFrame:
    return pd.read_csv(users_csv)


@lru_cache(maxsize=8)
def _load_banners_cached(banners_csv: str) -> pd.DataFrame:
    return pd.read_csv(banners_csv, parse_dates=["created_at"])


@lru_cache(maxsize=8)
def _load_catboost_model(artifacts_dir: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(Path(artifacts_dir) / "ctr_model.cbm"))
    return model


@lru_cache(maxsize=8)
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
    prepared = deepfm_pipeline.fill_dense_na(candidates.copy(), dense_features)
    dense_values = prepared[dense_features].astype(np.float32).to_numpy()

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    dense_scaled = scaler.transform(dense_values).astype(np.float32)

    cat_encoded = deepfm_pipeline.encode_categorical_frame(prepared, checkpoint["vocabs"], cat_features)
    dataset = deepfm_pipeline.TabularDataset(
        cat_encoded,
        dense_scaled,
        np.zeros(len(prepared), dtype=np.float32),
        np.ones(len(prepared), dtype=np.float32),
    )
    loader = DataLoader(dataset, batch_size=8192, shuffle=False, num_workers=0)
    return deepfm_pipeline.predict_dataset(model, loader, torch.device("cpu"))


def build_app() -> FastAPI:
    app = FastAPI(title="Recsys Inference")

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/recommend", response_model=RecommendationResponse, tags=["recommendations"])
    def recommend(payload: RecommendationRequest) -> RecommendationResponse:
        metadata = _load_metadata_cached(ARTIFACTS_DIR)
        model_type = _resolve_model_type(metadata)
        users = _load_users_cached(USERS_CSV)
        banners = _load_banners_cached(BANNERS_CSV)

        user_df = users[users["user_id"] == payload.user_id].copy()
        if user_df.empty:
            raise ValueError(f"user_id={payload.user_id!r} not found in users file")

        if payload.only_active:
            banners = banners[banners["is_active"] == 1].copy()

        if payload.as_of_date:
            serve_date = pd.Timestamp(payload.as_of_date)
        else:
            serve_date = pd.Timestamp(metadata["latest_event_date"]) + pd.Timedelta(days=1)

        if payload.candidate_mode == "retrieval + ranking":
            retrieved_banner_ids = recommend_top_n(
                artifact_dir=RETRIEVAL_ARTIFACTS_DIR,
                user_id=payload.user_id,
                top_n=payload.retrieval_top_n,
                exclude_seen=payload.exclude_seen,
                interactions_csv=INTERACTIONS_CSV,
            )
            candidates = banners[banners["banner_id"].isin(retrieved_banner_ids)].copy()
            if candidates.empty:
                raise ValueError("Retrieval returned no candidate banners for reranking")

            retrieval_rank = {
                banner_id: rank for rank, banner_id in enumerate(retrieved_banner_ids, start=1)
            }
            candidates["retrieval_rank"] = candidates["banner_id"].map(retrieval_rank)
            user_row = user_df.iloc[0]
            for column in user_df.columns:
                candidates[column] = user_row[column]
            candidates["event_date"] = serve_date
        else:
            expanded_user_df = user_df.copy()
            expanded_banners = banners.copy()
            expanded_user_df["__k"] = 1
            expanded_banners["__k"] = 1
            candidates = expanded_banners.merge(expanded_user_df, on="__k", how="inner").drop(columns="__k")
            candidates["event_date"] = serve_date

        candidates = add_base_features(candidates)
        history = load_history_tables(ARTIFACTS_DIR)
        candidates = merge_history_features(candidates, history, metadata)
        candidates = attach_recent_user_banner_history(candidates, INTERACTIONS_CSV, payload.user_id)

        for col in ["served_impressions_total", "served_clicks_total"]:
            if col not in candidates.columns:
                candidates[col] = 0

        if payload.online_banner_stats:
            online_stats_df = pd.DataFrame([item.model_dump() for item in payload.online_banner_stats])
            candidates = candidates.merge(
                online_stats_df.rename(
                    columns={
                        "served_impressions_total": "online_served_impressions_total",
                        "served_clicks_total": "online_served_clicks_total",
                    }
                ),
                on="banner_id",
                how="left",
            )
            candidates["served_impressions_total"] = (
                candidates["served_impressions_total"]
                + candidates["online_served_impressions_total"].fillna(0.0)
            )
            candidates["served_clicks_total"] = (
                candidates["served_clicks_total"]
                + candidates["online_served_clicks_total"].fillna(0.0)
            )
            candidates = candidates.drop(
                columns=["online_served_impressions_total", "online_served_clicks_total"]
            )

        if payload.exclude_seen and payload.candidate_mode == "all banners":
            candidates = candidates[candidates["served_impressions_total"] == 0].copy()
        if payload.exclude_seen and payload.online_seen_banner_ids:
            candidates = candidates[
                ~candidates["banner_id"].astype(str).isin(payload.online_seen_banner_ids)
            ].copy()

        candidates["fatigue_penalty"] = 1.0 / (1.0 + np.log1p(candidates["served_impressions_total"]))
        candidates["repeat_click_bonus"] = np.where(candidates["served_clicks_total"] > 0, 1.05, 1.0)

        feature_cols = metadata["feature_cols"]
        if model_type == "deepfm":
            candidates["pred_ctr"] = np.clip(
                _predict_with_deepfm(candidates[feature_cols], ARTIFACTS_DIR),
                0.0,
                1.0,
            )
        else:
            model = _load_catboost_model(ARTIFACTS_DIR)
            candidates["pred_ctr"] = np.clip(model.predict(candidates[feature_cols]), 0.0, 1.0)

        if payload.score_mode == "ctr":
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

        sort_cols = ["final_score", "pred_ctr", "retrieval_rank"]
        sort_asc = [False, False, True]
        if "retrieval_rank" not in candidates.columns:
            sort_cols = ["final_score", "pred_ctr"]
            sort_asc = [False, False]

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
        recs = (
            candidates.sort_values(sort_cols, ascending=sort_asc)
            .head(payload.top_k)[result_cols]
            .reset_index(drop=True)
        )
        items = [RecommendationItem(**row) for row in recs.to_dict(orient="records")]
        return RecommendationResponse(
            model_type=model_type,
            artifacts_dir=str(Path(ARTIFACTS_DIR).resolve()),
            retrieval_used=payload.candidate_mode == "retrieval + ranking",
            online_state_applied=bool(payload.online_seen_banner_ids or payload.online_banner_stats),
            top_k=payload.top_k,
            items=items,
        )

    return app


app = build_app()
