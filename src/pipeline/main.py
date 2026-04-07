from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

SPLIT_DATE = "2026-03-16"
GLOBAL_CTR_ALPHA = {
    "user": 50,
    "banner": 50,
    "user_category": 30,
    "user_brand": 30,
}


def add_hist_ctr_feature(data, keys, prefix, global_ctr, alpha):
    daily = data.groupby(["event_date"] + keys, as_index=False)[["clicks", "impressions"]].sum()
    daily = daily.sort_values(["event_date"] + keys)

    grp = daily.groupby(keys, sort=False)
    daily[f"{prefix}_prev_clicks"] = grp["clicks"].cumsum() - daily["clicks"]
    daily[f"{prefix}_prev_impressions"] = grp["impressions"].cumsum() - daily["impressions"]

    daily[f"{prefix}_hist_ctr"] = (
        daily[f"{prefix}_prev_clicks"] + alpha * global_ctr
    ) / (daily[f"{prefix}_prev_impressions"] + alpha)

    return data.merge(
        daily[
            ["event_date", *keys, f"{prefix}_prev_impressions", f"{prefix}_hist_ctr"]
        ],
        on=["event_date", *keys],
        how="left",
    )

banners = pd.read_csv("./data/db/banners.csv", parse_dates=["created_at"])
interactions = pd.read_csv("./data/db/banner_interactions.csv", parse_dates=["event_date"])
users = pd.read_csv("./data/db/users.csv")

df = interactions.merge(users, on="user_id", how="left")
df = df.merge(banners, on="banner_id", how="left")

df["target_ctr"] = df["clicks"] / df["impressions"]
df = df.sort_values(["event_date", "user_id", "banner_id"]).reset_index(drop=True)

global_ctr = df["clicks"].sum() / df["impressions"].sum()


# Возраст в целевом диапазоне
# Совпадение по возрасту
df["age_match"] = (
    (df["age"] >= df["target_age_min"]) &
    (df["age"] <= df["target_age_max"])
).astype(int)

# Пол в целевом диапазоне
# Совпадение по полу
df["gender_match"] = (
    (df["target_gender"] == "U") |
    (df["gender"] == df["target_gender"]) |
    (df["gender"] == "U")
).astype(int)


# Совподение по интересам
df["interest_match"] = (
    (df["interest_1"] == df["subcategory"]) |
    (df["interest_2"] == df["subcategory"]) |
    (df["interest_3"] == df["subcategory"])
).astype(int)

# Возрастное расстояние до цели
df["age_distance_to_target"] = 0
df.loc[df["age"] < df["target_age_min"], "age_distance_to_target"] = (
    df["target_age_min"] - df["age"]
)
df.loc[df["age"] > df["target_age_max"], "age_distance_to_target"] = (
    df["age"] - df["target_age_max"]
)

df = add_hist_ctr_feature(
    df,
    keys=["user_id"],
    prefix="user",
    global_ctr=global_ctr,
    alpha=GLOBAL_CTR_ALPHA["user"],
)

df = add_hist_ctr_feature(
    df,
    keys=["banner_id"],
    prefix="banner",
    global_ctr=global_ctr,
    alpha=GLOBAL_CTR_ALPHA["banner"],
)

df = add_hist_ctr_feature(
    df,
    keys=["user_id", "category"],
    prefix="user_category",
    global_ctr=global_ctr,
    alpha=GLOBAL_CTR_ALPHA["user_category"],
)

df = add_hist_ctr_feature(
    df,
    keys=["user_id", "brand"],
    prefix="user_brand",
    global_ctr=global_ctr,
    alpha=GLOBAL_CTR_ALPHA["user_brand"],
)








train_df = df[df["event_date"] < SPLIT_DATE].copy()
valid_df = df[df["event_date"] >= SPLIT_DATE].copy()

def run_experiment(feature_list, name):
    exp_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=400,
        learning_rate=0.08,
        depth=6,
        random_seed=42,
        verbose=False
    )

    exp_cat_features = [f for f in cat_features if f in feature_list]

    exp_model.fit(
        train_df[feature_list],
        train_df["target_ctr"],
        cat_features=exp_cat_features,
        sample_weight=train_df["impressions"],
        eval_set=(valid_df[feature_list], valid_df["target_ctr"]),
        use_best_model=True,
        early_stopping_rounds=50
    )

    pred = exp_model.predict(valid_df[feature_list]).clip(0, 1)
    weighted_mse = np.average(
        (valid_df["target_ctr"] - pred) ** 2,
        weights=valid_df["impressions"]
    )
    weighted_rmse = np.sqrt(weighted_mse)

    print(f"{name}: weighted_rmse = {weighted_rmse:.6f}")
    return weighted_rmse

features = [
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
    "age_match",
    "gender_match",
    "interest_match",
    "age_distance_to_target",
    "user_hist_ctr",
    "banner_hist_ctr",
    "user_prev_impressions",
    "banner_prev_impressions",
    "user_category_hist_ctr",
    "user_category_prev_impressions",
    "user_brand_hist_ctr",
    "user_brand_prev_impressions",
]

cat_features = [
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

model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=400,
    learning_rate=0.08,
    depth=6,
    random_seed=42,
    verbose=10
)

model.fit(
    train_df[features],
    train_df["target_ctr"],
    cat_features=cat_features,
    sample_weight=train_df["impressions"],
    eval_set=(valid_df[features], valid_df["target_ctr"]),
    use_best_model=True,
    early_stopping_rounds=50
)

# valid_df["pred"] = model.predict(valid_df[features]).clip(0, 1)

# weighted_mse = np.average(
#     (valid_df["target_ctr"] - valid_df["pred"]) ** 2,
#     weights=valid_df["impressions"]
# )
# weighted_rmse = np.sqrt(weighted_mse)
# print("weighted_rmse =", weighted_rmse)

# importance_df = pd.DataFrame({
#     "feature": features,
#     "importance": model.get_feature_importance()
# }).sort_values("importance", ascending=False)

# print("\nTop feature importances:")
# print(importance_df.head(20).to_string(index=False))

base_features = [
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
    "age_match",
    "gender_match",
    "interest_match",
    "age_distance_to_target",
]

user_banner_history = [
    "user_hist_ctr",
    "banner_hist_ctr",
    "user_prev_impressions",
    "banner_prev_impressions",
]

user_category_history = [
    "user_category_hist_ctr",
    "user_category_prev_impressions",
]

user_brand_history = [
    "user_brand_hist_ctr",
    "user_brand_prev_impressions",
]

experiments = {
    "baseline_only": base_features,
    "baseline_plus_user_banner": base_features + user_banner_history,
    "baseline_plus_user_banner_plus_category": base_features + user_banner_history + user_category_history,
    "full_with_brand": base_features + user_banner_history + user_category_history + user_brand_history,
}

results = {}
for name, feat_list in experiments.items():
    results[name] = run_experiment(feat_list, name)

results_df = pd.DataFrame(
    [{"experiment": k, "weighted_rmse": v} for k, v in results.items()]
).sort_values("weighted_rmse")

print("\nAblation results:")
print(results_df.to_string(index=False))


def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances * discounts)

def ndcg_at_k_for_group(group, pred_col="pred", target_col="target_ctr", k=3):
    g = group.sort_values(pred_col, ascending=False)
    actual = g[target_col].values
    pred_order_dcg = dcg_at_k(actual, k)

    ideal = group.sort_values(target_col, ascending=False)[target_col].values
    ideal_dcg = dcg_at_k(ideal, k)

    if ideal_dcg == 0:
        return np.nan
    return pred_order_dcg / ideal_dcg

def evaluate_ndcg(valid_df, k=3):
    scores = (
        valid_df.groupby(["event_date", "user_id"])
        .apply(lambda g: ndcg_at_k_for_group(g, k=k))
        .dropna()
    )
    return scores.mean()

valid_df["pred"] = model.predict(valid_df[features]).clip(0, 1)
print("ndcg@3 =", evaluate_ndcg(valid_df, k=3))