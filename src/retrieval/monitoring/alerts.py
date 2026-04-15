from __future__ import annotations


def ensure_candidates_available(candidate_count: int) -> None:
    if candidate_count <= 0:
        raise ValueError("No retrieval candidates available for this request.")


__all__ = ["ensure_candidates_available"]
