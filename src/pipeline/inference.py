
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend top-K banners for a specific user with a trained CTR model."
    )
    parser.add_argument("--user-id", required=True, default="u_00007")
    parser.add_argument("--users-csv", required=True, default="./data/db/users.csv")
    parser.add_argument("--banners-csv", required=True, default="./data/db/banners.csv")
    parser.add_argument("--artifacts-dir", required=True, default="ctr_artifacts")
    parser.add_argument("--interactions-csv", default="./data/db/banner_interactions.csv",
                        help="Optional. Needed for --exclude-seen or user-banner fatigue features at serving time.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--as-of-date", default=None, help="YYYY-MM-DD. Defaults to one day after the training data max date.")
    parser.add_argument("--score-mode", choices=["ctr", "value"], default="value")
    parser.add_argument("--only-active", action="store_true", default=False)
    parser.add_argument("--exclude-seen", action="store_true", default=False)
    parser.add_argument("--output-csv", default="recs_u_00001.csv")
    return parser.parse_args()


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

    df["banner_age_days"] = (
        (df["event_date"] - df["created_at"]).dt.days.fillna(0).clip(lower=0)
    )
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


def load_history_tables(artifacts_dir):
    history = {}
    for name in [
        "banner",
        "user",
        "subcategory",
        "brand",
        "user_subcategory",
        "user_banner",
    ]:
        path = Path(artifacts_dir) / f"{name}_history.csv.gz"
        history[name] = pd.read_csv(path)
    return history


def merge_history_features(candidates, history, metadata):
    df = candidates.copy()
    global_ctr = metadata["global_ctr"]
    specs = metadata["history_specs"]

    for name, spec in specs.items():
        group_cols = spec["group_cols"]
        feature_name = spec["feature_name"]
        hist = history[name]
        df = df.merge(hist, on=group_cols, how="left")
        df[feature_name] = df[feature_name].fillna(global_ctr)
        df[f"{feature_name}_impr"] = df[f"{feature_name}_impr"].fillna(0.0)
    return df


def attach_recent_user_banner_history(candidates, interactions_csv, user_id):
    if interactions_csv is None:
        return candidates

    interactions = pd.read_csv(interactions_csv)
    user_hist = (
        interactions[interactions["user_id"] == user_id]
        .groupby("banner_id", as_index=False)[["impressions", "clicks"]]
        .sum()
        .rename(
            columns={
                "impressions": "served_impressions_total",
                "clicks": "served_clicks_total",
            }
        )
    )
    out = candidates.merge(user_hist, on="banner_id", how="left")
    out["served_impressions_total"] = out["served_impressions_total"].fillna(0)
    out["served_clicks_total"] = out["served_clicks_total"].fillna(0)
    return out


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    with open(artifacts_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    users = pd.read_csv(args.users_csv)
    banners = pd.read_csv(args.banners_csv, parse_dates=["created_at"])

    user_df = users[users["user_id"] == args.user_id].copy()
    if user_df.empty:
        raise ValueError(f"user_id={args.user_id!r} not found in {args.users_csv}")

    if args.only_active:
        banners = banners[banners["is_active"] == 1].copy()

    if args.as_of_date:
        as_of_date = pd.Timestamp(args.as_of_date)
    else:
        as_of_date = pd.Timestamp(metadata["latest_event_date"]) + pd.Timedelta(days=1)

    # Cross join one user with all candidate banners.
    user_df["__k"] = 1
    banners["__k"] = 1
    candidates = banners.merge(user_df, on="__k", how="inner").drop(columns="__k")
    candidates["event_date"] = as_of_date

    candidates = add_base_features(candidates)

    history = load_history_tables(artifacts_dir)
    candidates = merge_history_features(candidates, history, metadata)
    candidates = attach_recent_user_banner_history(candidates, args.interactions_csv, args.user_id)

    if args.exclude_seen:
        if args.interactions_csv is None:
            raise ValueError("--exclude-seen requires --interactions-csv")
        candidates = candidates[candidates["served_impressions_total"] == 0].copy()

    # Fallback defaults if interactions_csv wasn't provided.
    for col in ["served_impressions_total", "served_clicks_total"]:
        if col not in candidates.columns:
            candidates[col] = 0

    # Use fatigue signals even though the model was not explicitly trained on them.
    candidates["fatigue_penalty"] = 1.0 / (1.0 + np.log1p(candidates["served_impressions_total"]))
    candidates["repeat_click_bonus"] = np.where(candidates["served_clicks_total"] > 0, 1.05, 1.0)

    model = CatBoostRegressor()
    model.load_model(str(artifacts_dir / "ctr_model.cbm"))

    feature_cols = metadata["feature_cols"]
    candidates["pred_ctr"] = np.clip(model.predict(candidates[feature_cols]), 0.0, 1.0)

    if args.score_mode == "ctr":
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
        .head(args.top_k)[result_cols]
        .reset_index(drop=True)
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print(result.to_string(index=False))

    if args.output_csv:
        result.to_csv(args.output_csv, index=False)
        print(f"\nSaved recommendations to: {args.output_csv}")


if __name__ == "__main__":
    main()
