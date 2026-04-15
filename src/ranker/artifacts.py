from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANKER_MODEL_VERSION = "pytorch_ranker"
DEFAULT_PYTORCH_RANKER_ARTIFACTS = "artifacts/pytorch_ranker"
LEGACY_RANKER_ARTIFACTS = ("deepfm_artifacts", "ctr_artifacts")


def default_ranker_artifacts_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return root / DEFAULT_PYTORCH_RANKER_ARTIFACTS


def legacy_ranker_artifacts_paths(project_root: Path | None = None) -> list[Path]:
    root = project_root or PROJECT_ROOT
    return [root / raw_path for raw_path in LEGACY_RANKER_ARTIFACTS]


def normalize_metadata_path(path: str | Path, project_root: Path | None = None) -> str:
    resolved_path = Path(path).resolve()
    root = (project_root or PROJECT_ROOT).resolve()
    try:
        return str(resolved_path.relative_to(root))
    except ValueError:
        return str(resolved_path)
