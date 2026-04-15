from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
