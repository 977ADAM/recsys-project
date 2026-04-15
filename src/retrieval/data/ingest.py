from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.retrieval.data.validate import (
    validate_banners_frame,
    validate_interactions_frame,
)
from src.retrieval.utils.common import project_root_from

PROJECT_ROOT = project_root_from(__file__)
DATA_DIR = PROJECT_ROOT / "data" / "db"
DEFAULT_INTERACTIONS_CSV = DATA_DIR / "banner_interactions.csv"
DEFAULT_BANNERS_CSV = DATA_DIR / "banners.csv"


def resolve_dataset_path(
    filename: str,
    artifact_dir: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if artifact_dir is not None:
        candidates.append(Path(artifact_dir) / filename)
    candidates.append(DATA_DIR / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not locate {filename}. Looked in: {searched_paths}")


def load_interactions_frame(
    interactions_csv: str | Path,
    *,
    require_event_date: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(interactions_csv)
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)
    if "banner_id" in df.columns:
        df["banner_id"] = df["banner_id"].astype(str)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])

    validate_interactions_frame(df, require_event_date=require_event_date)
    return df


def load_banners_frame(banners_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(banners_csv)
    if "banner_id" in df.columns:
        df["banner_id"] = df["banner_id"].astype(str)

    validate_banners_frame(df)
    return df


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DEFAULT_INTERACTIONS_CSV",
    "DEFAULT_BANNERS_CSV",
    "resolve_dataset_path",
    "load_interactions_frame",
    "load_banners_frame",
]
