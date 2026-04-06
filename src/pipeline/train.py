
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a personalized CTR model for banner ranking."
    )
    parser.add_argument("--interactions-csv", required=True, default="./data/db/banner_interactions.csv")
    parser.add_argument("--users-csv", required=True, default="./data/db/users.csv")
    parser.add_argument("--banners-csv", required=True, default="./data/db/banners.csv")
    parser.add_argument("--output-dir", required=True, default="ctr_artifacts")
    parser.add_argument("--valid-days", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def load_data(interactions_csv, users_csv, banners_csv):
    interactions = pd.read_csv(interactions_csv, parse_dates=["event_date"])
    users = pd.read_csv(users_csv)
    banners = pd.read_csv(banners_csv, parse_dates=["created_at"])

    banners = banners.copy()
    banners["created_at"] = pd.to_datetime(banners["created_at"], errors="coerce")

    df = interactions.merge(users[USER_COLS], on="user_id", how="left")
    df = df.merge(banners[BANNER_COLS], on="banner_id", how="left")

    # Safe target recomputation.
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


def add_date_prior_feature(df, group_cols, feature_name, alpha, global_ctr):
    """
    Build leakage-safe priors: for each row, use only aggregate clicks/impressions from strictly earlier dates.
    """
    work = (
        df.groupby(group_cols + ["event_date"], as_index=False)[["clicks", "impressions"]]
        .sum()
        .sort_values(group_cols + ["event_date"])
    )

    grp = work.groupby(group_cols, sort=False)
    work["prior_clicks"] = grp["clicks"].cumsum() - work["clicks"]
    work["prior_impressions"] = grp["impressions"].cumsum() - work["impressions"]

    work[feature_name] = (work["prior_clicks"] + alpha * global_ctr) / (
        work["prior_impressions"] + alpha
    )
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
    """
    Tables used at inference time: aggregate all available history up to the latest date in the dataset.
    """
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
        agg[feature_name] = (agg["clicks"] + alpha * global_ctr) / (
            agg["impressions"] + alpha
        )
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(args.interactions_csv, args.users_csv, args.banners_csv)
    df = add_base_features(df)
    df, global_ctr, prior_specs = build_training_table(df)

    max_date = df["event_date"].max()
    valid_start = max_date - pd.Timedelta(days=args.valid_days - 1)

    train_df = df[df["event_date"] < valid_start].copy()
    valid_df = df[df["event_date"] >= valid_start].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError(
            f"Time split produced an empty dataset. valid_start={valid_start.date()}"
        )

    feature_cols = [
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

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=args.random_seed,
        verbose=100,
    )

    print(
        f"Training on {len(train_df):,} rows, validating on {len(valid_df):,} rows. "
        f"Train date max={train_df['event_date'].max().date()}, valid start={valid_start.date()}"
    )
    model.fit(
        train_df[feature_cols],
        train_df["target_ctr"],
        cat_features=CAT_FEATURES,
        sample_weight=train_df["impressions"],
        eval_set=(valid_df[feature_cols], valid_df["target_ctr"]),
        use_best_model=True,
    )

    valid_pred = np.clip(model.predict(valid_df[feature_cols]), 0.0, 1.0)
    valid_df = valid_df.copy()
    valid_df["pred_ctr"] = valid_pred

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

    print("Validation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Save model.
    model_path = output_dir / "ctr_model.cbm"
    model.save_model(str(model_path))

    # Save artifacts for inference.
    history_tables, history_specs = compute_full_history_tables(df, global_ctr)
    for table_name, table_df in history_tables.items():
        table_df.to_csv(output_dir / f"{table_name}_history.csv.gz", index=False, compression="gzip")

    metadata = {
        "feature_cols": feature_cols,
        "cat_features": CAT_FEATURES,
        "global_ctr": float(global_ctr),
        "latest_event_date": str(max_date.date()),
        "valid_days": args.valid_days,
        "history_specs": {
            name: {"group_cols": group_cols, "feature_name": feature_name, "alpha": alpha}
            for name, (group_cols, feature_name, alpha) in history_specs.items()
        },
        "user_cols": USER_COLS,
        "banner_cols": BANNER_COLS,
        "default_score_mode": "ctr",
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved model and artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
