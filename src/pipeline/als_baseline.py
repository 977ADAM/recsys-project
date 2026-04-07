import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

banners = pd.read_csv("./data/db/banners.csv", parse_dates=["created_at"])
interactions = pd.read_csv("./data/db/banner_interactions.csv", parse_dates=["event_date"])
users = pd.read_csv("./data/db/users.csv")

user_ids = users["user_id"].unique().tolist()
banner_ids = banners["banner_id"].unique().tolist()

user2idx = {u: i for i, u in enumerate(user_ids)}
banner2idx = {b: i for i, b in enumerate(banner_ids)}
idx2banner = {i: b for b, i in banner2idx.items()}

interactions = interactions.copy()
interactions["user_idx"] = interactions["user_id"].map(user2idx)
interactions["banner_idx"] = interactions["banner_id"].map(banner2idx)

interactions["weight"] = interactions["clicks"] * 10 + interactions["impressions"] * 0.2
interactions = interactions[interactions["weight"] > 0]


user_item = csr_matrix(
    (
        interactions["weight"].astype(np.float32),
        (interactions["user_idx"], interactions["banner_idx"])
    ),
    shape=(len(user2idx), len(banner2idx))
)

model = AlternatingLeastSquares(
    factors=32,
    regularization=0.05,
    alpha=10.0,
    iterations=50,
    random_state=42
)

model.fit(user_item)

target_user_id = "u_00001"
target_user_idx = user2idx[target_user_id]

item_idx_rec, scores = model.recommend(
    userid=target_user_idx,
    user_items=user_item[target_user_idx],
    N=5,
    filter_already_liked_items=True
)

rec_banner_ids = [idx2banner[int(i)] for i in item_idx_rec]

print("Рекомендации для user", target_user_id)
for banner_id, score in zip(rec_banner_ids, scores):
    print(f"banner={banner_id}, score={score:.4f}")