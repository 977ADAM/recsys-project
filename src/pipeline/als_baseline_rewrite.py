import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # optional
    threadpool_limits = None

from implicit.als import AlternatingLeastSquares


# =========================
# Config
# =========================
FACTORS = 32
REGULARIZATION = 0.05
ALPHA = 10.0
ITERATIONS = 50
TOP_K = 10
TOP_N_EXPORT = 10
RANDOM_STATE = 42
CLICK_WEIGHT = 20.0
IMPRESSION_WEIGHT = 0.1


# =========================
# Path helpers
# =========================
def resolve_project_root() -> Path:
    """
    Assumes script is located at <project_root>/src/pipeline/als_baseline.py
    Falls back to current working directory if needed.
    """
    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parents[2],  # <project_root>
        Path.cwd(),
    ]
    for candidate in candidates:
        data_dir = candidate / "data" / "db"
        if (data_dir / "banners.csv").exists() and (data_dir / "banner_interactions.csv").exists():
            return candidate
    return Path.cwd()


# =========================
# Data loading
# =========================
def load_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = project_root / "data" / "db"
    banners = pd.read_csv(data_dir / "banners.csv", parse_dates=["created_at"])
    interactions = pd.read_csv(data_dir / "banner_interactions.csv", parse_dates=["event_date"])
    users = pd.read_csv(data_dir / "users.csv")
    return banners, interactions, users


# =========================
# Split logic
# =========================
def build_holdout(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-out split by user's last clicked active banner.

    A user is eligible for evaluation if they have clicks on at least 2 unique banners,
    otherwise we can't hold one out and still keep positive signal in train.
    """
    clicked = interactions.loc[interactions["clicks"] > 0, ["user_id", "banner_id", "event_date"]].copy()
    unique_clicked_per_user = clicked.groupby("user_id")["banner_id"].nunique()
    eligible_users = unique_clicked_per_user[unique_clicked_per_user >= 2].index

    holdout = (
        clicked[clicked["user_id"].isin(eligible_users)]
        .sort_values(["user_id", "event_date", "banner_id"])
        .groupby("user_id", as_index=False)
        .tail(1)
        [["user_id", "banner_id"]]
        .drop_duplicates()
        .rename(columns={"banner_id": "target_banner_id"})
        .reset_index(drop=True)
    )
    return holdout


# =========================
# Matrix prep
# =========================
def aggregate_train_interactions(interactions: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    train_df = interactions.merge(
        holdout,
        left_on=["user_id", "banner_id"],
        right_on=["user_id", "target_banner_id"],
        how="left",
    )
    train_df = train_df[train_df["target_banner_id"].isna()].drop(columns=["target_banner_id"])

    train_agg = (
        train_df.groupby(["user_id", "banner_id"], as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
        )
    )

    # Implicit-feedback confidence.
    # Clicks are much stronger than impressions for banner recommendation.
    train_agg["weight"] = (
        train_agg["clicks"] * CLICK_WEIGHT
        + np.log1p(train_agg["impressions"]) * IMPRESSION_WEIGHT
    )
    train_agg = train_agg[train_agg["weight"] > 0].reset_index(drop=True)
    return train_agg



def build_mappings(users: pd.DataFrame, active_banners: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    user_ids = users["user_id"].drop_duplicates().tolist()
    banner_ids = active_banners["banner_id"].drop_duplicates().tolist()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    banner2idx = {b: i for i, b in enumerate(banner_ids)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2banner = {i: b for b, i in banner2idx.items()}
    return user2idx, banner2idx, idx2user, idx2banner



def build_user_item_matrix(
    train_agg: pd.DataFrame,
    user2idx: Dict[str, int],
    banner2idx: Dict[str, int],
) -> csr_matrix:
    df = train_agg.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["banner_idx"] = df["banner_id"].map(banner2idx)
    df = df.dropna(subset=["user_idx", "banner_idx"]).copy()
    df["user_idx"] = df["user_idx"].astype(int)
    df["banner_idx"] = df["banner_idx"].astype(int)

    user_item = csr_matrix(
        (df["weight"].astype(np.float32), (df["user_idx"], df["banner_idx"])),
        shape=(len(user2idx), len(banner2idx)),
        dtype=np.float32,
    )
    return user_item


# =========================
# Model
# =========================
def fit_als(train_user_item: csr_matrix) -> AlternatingLeastSquares:
    """
    The `implicit` package historically expects item-user matrix for fit().
    We train on train_user_item.T and then use train_user_item[user_idx] in recommend().
    """
    model = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        alpha=ALPHA,
        iterations=ITERATIONS,
        random_state=RANDOM_STATE,
    )

    item_user = train_user_item.T.tocsr()
    if threadpool_limits is not None:
        with threadpool_limits(limits=1, user_api="blas"):
            model.fit(item_user)
    else:
        model.fit(item_user)
    return model


# =========================
# Recommendation helpers
# =========================
def compute_popularity_fallback(train_agg: pd.DataFrame, active_banner_ids: List[str]) -> List[str]:
    popularity = (
        train_agg.assign(popularity=train_agg["clicks"] * 1000 + train_agg["impressions"])
        .groupby("banner_id", as_index=False)["popularity"]
        .sum()
        .sort_values(["popularity", "banner_id"], ascending=[False, True])
    )
    popular_ids = popularity["banner_id"].tolist()

    # keep only active banners and preserve order
    active_set = set(active_banner_ids)
    popular_ids = [b for b in popular_ids if b in active_set]

    # append any missing active banners to guarantee full coverage
    missing = [b for b in active_banner_ids if b not in set(popular_ids)]
    return popular_ids + missing



def recommend_for_user(
    model: AlternatingLeastSquares,
    user_id: str,
    train_user_item: csr_matrix,
    user2idx: Dict[str, int],
    idx2banner: Dict[int, str],
    popularity_fallback: List[str],
    n: int = TOP_N_EXPORT,
) -> List[str]:
    user_idx = user2idx[user_id]
    user_vector = train_user_item[user_idx]

    if user_vector.nnz == 0:
        return popularity_fallback[:n]

    item_idx_rec, _ = model.recommend(
        userid=user_idx,
        user_items=user_vector,
        N=n,
        filter_already_liked_items=True,
    )

    recs = []
    for item_idx in item_idx_rec:
        item_idx = int(item_idx)
        banner_id = idx2banner.get(item_idx)
        if banner_id is not None:
            recs.append(banner_id)

    # very defensive fallback in case library/version behavior differs
    if len(recs) < n:
        seen = set(recs)
        for banner_id in popularity_fallback:
            if banner_id not in seen:
                recs.append(banner_id)
                seen.add(banner_id)
            if len(recs) >= n:
                break

    return recs[:n]


# =========================
# Evaluation
# =========================
def evaluate_recall_at_k(
    model: AlternatingLeastSquares,
    holdout: pd.DataFrame,
    train_user_item: csr_matrix,
    user2idx: Dict[str, int],
    idx2banner: Dict[int, str],
    popularity_fallback: List[str],
    k: int = TOP_K,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    hits = 0

    for row in holdout.itertuples(index=False):
        user_id = row.user_id
        target_banner_id = row.target_banner_id

        recs = recommend_for_user(
            model=model,
            user_id=user_id,
            train_user_item=train_user_item,
            user2idx=user2idx,
            idx2banner=idx2banner,
            popularity_fallback=popularity_fallback,
            n=k,
        )
        hit = int(target_banner_id in recs)
        hits += hit

        rows.append(
            {
                "user_id": user_id,
                "target_banner_id": target_banner_id,
                "recommended_banner_ids": ",".join(recs),
                f"hit@{k}": hit,
            }
        )

    eval_df = pd.DataFrame(rows)
    recall_at_k = hits / len(eval_df) if len(eval_df) > 0 else 0.0
    return recall_at_k, eval_df


# =========================
# Main
# =========================
def main() -> None:
    project_root = resolve_project_root()
    output_dir = project_root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    banners, interactions, users = load_data(project_root)

    active_banners = banners[banners["is_active"] == 1].copy()
    active_banner_ids = active_banners["banner_id"].tolist()
    interactions = interactions[interactions["banner_id"].isin(set(active_banner_ids))].copy()

    holdout = build_holdout(interactions)
    train_agg = aggregate_train_interactions(interactions, holdout)

    user2idx, banner2idx, idx2user, idx2banner = build_mappings(users, active_banners)
    train_user_item = build_user_item_matrix(train_agg, user2idx, banner2idx)

    model = fit_als(train_user_item)
    popularity_fallback = compute_popularity_fallback(train_agg, active_banner_ids)

    # Eval
    recall_at_k, eval_df = evaluate_recall_at_k(
        model=model,
        holdout=holdout,
        train_user_item=train_user_item,
        user2idx=user2idx,
        idx2banner=idx2banner,
        popularity_fallback=popularity_fallback,
        k=TOP_K,
    )

    # Recommendations for all users
    all_recs = []
    for user_id in users["user_id"]:
        recs = recommend_for_user(
            model=model,
            user_id=user_id,
            train_user_item=train_user_item,
            user2idx=user2idx,
            idx2banner=idx2banner,
            popularity_fallback=popularity_fallback,
            n=TOP_N_EXPORT,
        )
        all_recs.append(
            {
                "user_id": user_id,
                "recommended_banner_ids": ",".join(recs),
            }
        )
    all_recs_df = pd.DataFrame(all_recs)

    # Save artifacts
    eval_path = output_dir / f"als_eval_recall_at_{TOP_K}.csv"
    recs_path = output_dir / "als_recommendations.csv"
    metrics_path = output_dir / "als_metrics.txt"

    eval_df.to_csv(eval_path, index=False)
    all_recs_df.to_csv(recs_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"users_total={len(users)}\n")
        f.write(f"active_banners_total={len(active_banners)}\n")
        f.write(f"eval_users={len(eval_df)}\n")
        f.write(f"recall@{TOP_K}={recall_at_k:.6f}\n")

    # Print summary
    sample_user_id = users.iloc[0]["user_id"]
    sample_recs = recommend_for_user(
        model=model,
        user_id=sample_user_id,
        train_user_item=train_user_item,
        user2idx=user2idx,
        idx2banner=idx2banner,
        popularity_fallback=popularity_fallback,
        n=5,
    )

    print(f"project_root={project_root}")
    print(f"users_total={len(users)}")
    print(f"active_banners_total={len(active_banners)}")
    print(f"eval_users={len(eval_df)}")
    print(f"Recall@{TOP_K}: {recall_at_k:.4f}")
    print(f"Sample recommendations for {sample_user_id}: {sample_recs}")
    print(f"Saved: {eval_path}")
    print(f"Saved: {recs_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
