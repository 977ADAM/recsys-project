from catboost import CatBoostRegressor
import pandas as pd

banners = pd.read_csv(".../data/db/banners.csv", parse_dates=["created_at"])
interactions = pd.read_csv(".../data/db/banner_interactions.csv", parse_dates=["event_date"])
users = pd.read_csv(".../data/db/users.csv")

df = interactions.merge(users, on="user_id", how="left")
df = df.merge(banners, on="banner_id", how="left")

df["target_ctr"] = df["clicks"] / df["impressions"]

train_df = df[df["event_date"] < "2026-03-16"].copy()
valid_df = df[df["event_date"] >= "2026-03-16"].copy()

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
    "is_active",
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
    iterations=500,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=100
)

model.fit(
    train_df[features],
    train_df["target_ctr"],
    cat_features=cat_features,
    sample_weight=train_df["impressions"],
    eval_set=(valid_df[features], valid_df["target_ctr"]),
    use_best_model=True
)