


USER_CATEGORICAL_COLUMNS = [
    "user_id",
    "gender",
    "city_tier",
    "device_os",
    "platform",
    "income_band",
    "activity_segment",
    "interest_1",
    "interest_2",
    "interest_3",
    "country",
]
USER_NUMERIC_COLUMNS = ["age", "signup_days_ago", "is_premium"]

ITEM_CATEGORICAL_COLUMNS = [
    "banner_id",
    "brand",
    "category",
    "subcategory",
    "banner_format",
    "campaign_goal",
    "target_gender",
]
ITEM_NUMERIC_COLUMNS = [
    "target_age_min",
    "target_age_max",
    "cpm_bid",
    "quality_score",
    "is_active",
    "banner_age_days",
]