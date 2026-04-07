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

clicked_pairs = (
    interactions[interactions["clicks"] > 0]
    .groupby(["user_id", "banner_id"], as_index=False)
    .agg(
        last_event_date=("event_date", "max"),
        total_clicks=("clicks", "sum")
    )
)

# берём только пользователей, у которых хотя бы 2 разных кликнутых баннера
eligible_users = (
    clicked_pairs.groupby("user_id")["banner_id"]
    .nunique()
)
eligible_users = eligible_users[eligible_users >= 2].index

clicked_pairs = clicked_pairs[clicked_pairs["user_id"].isin(eligible_users)].copy()

# последний кликнутый banner -> test
test_pairs = (
    clicked_pairs
    .sort_values(
        ["user_id", "last_event_date", "total_clicks", "banner_id"],
        ascending=[True, True, True, True]
    )
    .groupby("user_id", as_index=False)
    .tail(1)
    [["user_id", "banner_id"]]
    .copy()
)

test_pairs["is_test"] = 1

train_interactions = interactions.merge(
    test_pairs,
    on=["user_id", "banner_id"],
    how="left"
)

train_interactions = (
    train_interactions[train_interactions["is_test"].isna()]
    .drop(columns=["is_test"])
    .copy()
)

train_interactions["user_idx"] = train_interactions["user_id"].map(user2idx)
train_interactions["banner_idx"] = train_interactions["banner_id"].map(banner2idx)

test_pairs["user_idx"] = test_pairs["user_id"].map(user2idx)
test_pairs["banner_idx"] = test_pairs["banner_id"].map(banner2idx)

train_interactions["weight"] = (
    train_interactions["clicks"] * 10.0 +
    train_interactions["impressions"] * 0.2
)


train_interactions = train_interactions[train_interactions["weight"] > 0].copy()



train_user_item = csr_matrix(
    (
        train_interactions["weight"].astype(np.float32),
        (train_interactions["user_idx"], train_interactions["banner_idx"])
    ),
    shape=(len(user2idx), len(banner2idx))
)

test_user_item = csr_matrix(
    (
        np.ones(len(test_pairs), dtype=np.float32),
        (test_pairs["user_idx"], test_pairs["banner_idx"])
    ),
    shape=(len(user2idx), len(banner2idx))
)

model = AlternatingLeastSquares(
    factors=100,
    regularization=0.01,
    alpha=1,
    iterations=15,
    random_state=42,
    calculate_training_loss=True
)

model.fit(train_user_item)

model.save("artifacts/als_model")

k = 5

eval_user_indices = test_pairs["user_idx"].unique().tolist()

hits = []
for uidx in eval_user_indices:
    rec_ids, _ = model.recommend(
        userid=uidx,
        user_items=train_user_item[uidx],
        N=k,
        filter_already_liked_items=True
    )

    true_items = set(test_user_item[uidx].indices.tolist())
    pred_items = set(int(i) for i in rec_ids)

    hit = len(true_items & pred_items) / len(true_items)
    hits.append(hit)

recall_at_k = float(np.mean(hits)) if hits else float("nan")

eval_user_indices = test_pairs["user_idx"].unique().tolist()

precisions = []
recalls = []
average_precisions = []
ndcgs = []

for uidx in eval_user_indices:
    true_items = test_user_item[uidx].indices.tolist()
    if not true_items:
        continue

    rec_ids, _ = model.recommend(
        userid=uidx,
        user_items=train_user_item[uidx],
        N=k,
        filter_already_liked_items=True
    )

    pred = [int(i) for i in rec_ids]
    true_set = set(true_items)

    # hits по позициям
    hits = [1 if item in true_set else 0 for item in pred]
    n_hits = sum(hits)

    # Precision@K
    precision = n_hits / k
    precisions.append(precision)

    # Recall@K
    recall = n_hits / len(true_set)
    recalls.append(recall)

    # AP@K
    running_hits = 0
    ap_sum = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            running_hits += 1
            ap_sum += running_hits / rank
    ap = ap_sum / min(len(true_set), k)
    average_precisions.append(ap)

    # NDCG@K
    dcg = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(true_set), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcgs.append(ndcg)

precision_at_k_manual = float(np.mean(precisions)) if precisions else float("nan")
recall_at_k_manual = float(np.mean(recalls)) if recalls else float("nan")
map_at_k_manual = float(np.mean(average_precisions)) if average_precisions else float("nan")
ndcg_at_k_manual = float(np.mean(ndcgs)) if ndcgs else float("nan")

print(f"eligible users for eval: {len(eval_user_indices)}")
print(f"train nnz: {train_user_item.nnz}")
print(f"test nnz: {test_user_item.nnz}")
print(f"Precision@{k}: {precision_at_k_manual:.6f}")
print(f"Recall@{k}:    {recall_at_k_manual:.6f}")
print(f"MAP@{k}:       {map_at_k_manual:.6f}")
print(f"NDCG@{k}:      {ndcg_at_k_manual:.6f}")

sample_user_idx = eval_user_indices[0]
sample_user_id = user_ids[sample_user_idx]

rec_ids, scores = model.recommend(
    userid=sample_user_idx,
    user_items=train_user_item[sample_user_idx],
    N=k,
    filter_already_liked_items=True
)

true_test_banner_idx = test_user_item[sample_user_idx].indices.tolist()
true_test_banner_ids = [idx2banner[i] for i in true_test_banner_idx]
pred_banner_ids = [idx2banner[int(i)] for i in rec_ids]

print()
print("sample user:", sample_user_id)
print("held-out test banners:", true_test_banner_ids)
print("top-k predictions:")
for bid, score in zip(pred_banner_ids, scores):
    print(f"  {bid}: {score:.4f}")