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
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

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

DEFAULT_INTERACTIONS = "/data/db/banner_interactions.csv"
DEFAULT_USERS = "data/db/users.csv"
DEFAULT_BANNERS = "data/db/banners.csv"
DEFAULT_ARTIFACTS = "data/db/ctr_artifacts_streamlit"


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


def train_model(
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


@st.cache_resource(show_spinner=False)
def load_model(artifacts_dir: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(Path(artifacts_dir) / "ctr_model.cbm"))
    return model


def recommend_for_user(
    user_id: str,
    users_csv: str,
    banners_csv: str,
    artifacts_dir: str,
    interactions_csv: Optional[str],
    top_k: int,
    only_active: bool,
    exclude_seen: bool,
    score_mode: str,
    as_of_date: Optional[str],
) -> pd.DataFrame:
    metadata = load_metadata(artifacts_dir)
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

    if exclude_seen:
        candidates = candidates[candidates["served_impressions_total"] == 0].copy()

    candidates["fatigue_penalty"] = 1.0 / (1.0 + np.log1p(candidates["served_impressions_total"]))
    candidates["repeat_click_bonus"] = np.where(candidates["served_clicks_total"] > 0, 1.05, 1.0)

    model = load_model(artifacts_dir)
    feature_cols = metadata["feature_cols"]
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
        candidates.sort_values(["final_score", "pred_ctr"], ascending=False)
        .head(top_k)[result_cols]
        .reset_index(drop=True)
    )
    return result


def render_sidebar() -> dict:
    st.sidebar.header("Данные и артефакты")
    interactions_csv = st.sidebar.text_input("banner_interactions.csv", DEFAULT_INTERACTIONS)
    users_csv = st.sidebar.text_input("users.csv", DEFAULT_USERS)
    banners_csv = st.sidebar.text_input("banners.csv", DEFAULT_BANNERS)
    artifacts_dir = st.sidebar.text_input("Папка артефактов модели", DEFAULT_ARTIFACTS)

    st.sidebar.caption("Можно оставить пути по умолчанию для ваших текущих файлов в /mnt/data.")

    return {
        "interactions_csv": interactions_csv,
        "users_csv": users_csv,
        "banners_csv": banners_csv,
        "artifacts_dir": artifacts_dir,
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
    st.subheader("Обучение CTR-модели")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        valid_days = st.number_input("Окно validation, дней", min_value=3, max_value=60, value=14)
    with c2:
        iterations = st.number_input("Итерации CatBoost", min_value=50, max_value=5000, value=400, step=50)
    with c3:
        learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    with c4:
        depth = st.number_input("Depth", min_value=3, max_value=12, value=8)

    random_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42)

    if st.button("Обучить и сохранить артефакты", type="primary"):
        with st.spinner("Идёт обучение модели..."):
            metrics, output_path = train_model(
                interactions_csv=cfg["interactions_csv"],
                users_csv=cfg["users_csv"],
                banners_csv=cfg["banners_csv"],
                output_dir=cfg["artifacts_dir"],
                valid_days=int(valid_days),
                iterations=int(iterations),
                learning_rate=float(learning_rate),
                depth=int(depth),
                random_seed=int(random_seed),
            )
            st.session_state["last_artifacts_dir"] = str(output_path)

        st.success(f"Артефакты сохранены в: {output_path}")
        metric_cols = st.columns(5)
        metric_cols[0].metric("weighted RMSE", f"{metrics['weighted_rmse']:.4f}")
        metric_cols[1].metric("RMSE", f"{metrics['rmse_unweighted']:.4f}")
        metric_cols[2].metric("NDCG@5", f"{metrics['ndcg_at_5']:.4f}")
        metric_cols[3].metric("mean pred CTR", f"{metrics['mean_pred_ctr']:.4f}")
        metric_cols[4].metric("mean actual CTR", f"{metrics['mean_actual_ctr']:.4f}")

        st.json(metrics)

        preview_path = Path(output_path) / "validation_preview.csv"
        if preview_path.exists():
            st.markdown("**Топ validation-предсказания**")
            st.dataframe(pd.read_csv(preview_path).head(50), use_container_width=True, hide_index=True)


def recommend_tab(cfg: dict) -> None:
    st.subheader("Рекомендации баннеров для пользователя")

    users_df = load_users(cfg["users_csv"])
    default_user = str(users_df["user_id"].iloc[0]) if not users_df.empty else ""

    c1, c2, c3 = st.columns(3)
    with c1:
        user_id = st.text_input("user_id", value=default_user)
    with c2:
        top_k = st.number_input("top_k", min_value=1, max_value=100, value=10)
    with c3:
        score_mode = st.selectbox("score_mode", options=["ctr", "value"], index=0)

    c4, c5, c6 = st.columns(3)
    with c4:
        only_active = st.checkbox("Только активные баннеры", value=True)
    with c5:
        exclude_seen = st.checkbox("Исключить уже показанные", value=True)
    with c6:
        as_of_date = st.text_input("Дата показа (YYYY-MM-DD), optional", value="")

    if st.button("Построить рекомендации"):
        with st.spinner("Считаю рекомендации..."):
            recs = recommend_for_user(
                user_id=user_id,
                users_csv=cfg["users_csv"],
                banners_csv=cfg["banners_csv"],
                artifacts_dir=cfg["artifacts_dir"],
                interactions_csv=cfg["interactions_csv"],
                top_k=int(top_k),
                only_active=only_active,
                exclude_seen=exclude_seen,
                score_mode=score_mode,
                as_of_date=as_of_date or None,
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
        st.json(load_metadata(cfg["artifacts_dir"]))

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
    st.caption("Streamlit-приложение для обучения персональной CTR-модели и выдачи top-K баннеров пользователю.")

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
