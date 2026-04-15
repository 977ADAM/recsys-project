import pandas as pd


def build_mappings(
    train_df: pd.DataFrame,
    banners_csv: str | None = None,
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    if banners_csv is not None:
        banners = pd.read_csv(banners_csv)
        banner_ids = banners["banner_id"].drop_duplicates().astype(str).tolist()
    else:
        banner_ids = train_df["banner_id"].drop_duplicates().astype(str).tolist()

    user_ids = train_df["user_id"].drop_duplicates().astype(str).tolist()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item2idx = {banner_id: idx for idx, banner_id in enumerate(banner_ids)}
    idx2item = {idx: banner_id for banner_id, idx in item2idx.items()}
    return user2idx, item2idx, idx2item
