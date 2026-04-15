from __future__ import annotations

from pathlib import Path


def project_root_from(source_file: str | Path, depth: int = 3) -> Path:
    return Path(source_file).resolve().parents[depth]


def resolve_project_path(
    project_root: str | Path,
    raw_path: str | Path | None,
    fallback: str | Path,
) -> Path:
    path = Path(raw_path or fallback)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path


def normalize_project_path(
    path: str | Path,
    project_root: str | Path,
) -> str:
    resolved_path = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        return str(resolved_path.relative_to(root))
    except ValueError:
        return str(resolved_path)


__all__ = [
    "project_root_from",
    "resolve_project_path",
    "normalize_project_path",
]
