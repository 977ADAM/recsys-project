from __future__ import annotations

import argparse

from src.scripts.pytorch_recsys.inference import recommend_top_n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate top-N recommendations from trained PyTorch retrieval artifacts."
    )
    parser.add_argument("--artifact-dir", default="artifacts/pytorch_retrieval")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--exclude-seen", action="store_true")
    parser.add_argument("--interactions-csv", default="data/db/banner_interactions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommendations = recommend_top_n(
        artifact_dir=args.artifact_dir,
        user_id=args.user_id,
        top_n=args.top_n,
        exclude_seen=args.exclude_seen,
        interactions_csv=args.interactions_csv if args.exclude_seen else None,
    )
    print("\n".join(recommendations))


if __name__ == "__main__":
    main()
