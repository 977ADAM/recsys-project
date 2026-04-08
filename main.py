#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INTERACTIONS = "./data/db/banner_interactions.csv"
DEFAULT_USERS = "./data/db/users.csv"
DEFAULT_BANNERS = "./data/db/banners.csv"
DEFAULT_CTR_ARTIFACTS = "ctr_artifacts"
DEFAULT_RETRIEVAL_ARTIFACTS = "artifacts/pytorch_retrieval"


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--interactions-csv", default=DEFAULT_INTERACTIONS)
    parser.add_argument("--users-csv", default=DEFAULT_USERS)
    parser.add_argument("--banners-csv", default=DEFAULT_BANNERS)


def run_command(command: list[str]) -> int:
    print("Running:")
    print("  " + " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Banner recommendation project entrypoint. "
            "Use one of the subcommands below to launch the active product flows."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Streamlit application.",
    )
    ui_parser.add_argument("--port", type=int, default=8501)

    ranker_train = subparsers.add_parser(
        "train-ranker",
        help="Train the CatBoost CTR/ranking model.",
    )
    add_common_data_args(ranker_train)
    ranker_train.add_argument("--output-dir", default=DEFAULT_CTR_ARTIFACTS)
    ranker_train.add_argument("--valid-days", type=int, default=14)
    ranker_train.add_argument("--iterations", type=int, default=400)
    ranker_train.add_argument("--learning-rate", type=float, default=0.05)
    ranker_train.add_argument("--depth", type=int, default=8)
    ranker_train.add_argument("--random-seed", type=int, default=42)

    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Run CatBoost inference, optionally with retrieval candidate generation first.",
    )
    add_common_data_args(recommend_parser)
    recommend_parser.add_argument("--user-id", required=True)
    recommend_parser.add_argument("--artifacts-dir", default=DEFAULT_CTR_ARTIFACTS)
    recommend_parser.add_argument("--top-k", type=int, default=10)
    recommend_parser.add_argument("--as-of-date", default=None)
    recommend_parser.add_argument("--score-mode", choices=["ctr", "value"], default="value")
    recommend_parser.add_argument("--only-active", action="store_true", default=False)
    recommend_parser.add_argument("--exclude-seen", action="store_true", default=False)
    recommend_parser.add_argument("--retrieval-artifacts-dir", default=None)
    recommend_parser.add_argument("--retrieval-top-n", type=int, default=100)
    recommend_parser.add_argument("--output-csv", default="recs_u_00001.csv")

    retrieval_train = subparsers.add_parser(
        "train-retrieval",
        help="Train the PyTorch retrieval model and save retrieval artifacts.",
    )
    retrieval_train.add_argument("--epochs", type=int, default=5)
    retrieval_train.add_argument("--batch-size", type=int, default=1024)
    retrieval_train.add_argument("--embedding-dim", type=int, default=64)
    retrieval_train.add_argument("--learning-rate", type=float, default=1e-3)
    retrieval_train.add_argument("--weight-decay", type=float, default=1e-5)
    retrieval_train.add_argument("--k", type=int, default=100)
    retrieval_train.add_argument("--seed", type=int, default=42)
    retrieval_train.add_argument("--output-dir", default=DEFAULT_RETRIEVAL_ARTIFACTS)
    retrieval_train.add_argument("--save-item-embeddings", action="store_true")

    return parser


def print_overview(parser: argparse.ArgumentParser) -> int:
    print("Active product:")
    print("  1. CatBoost ranker: src/pipeline/train.py + src/pipeline/inference.py")
    print("  2. PyTorch retrieval: src/scripts/pytorch_recsys/")
    print("  3. UI: app_streamlit.py")
    print()
    print("Most common commands:")
    print("  python main.py ui")
    print("  python main.py train-ranker")
    print("  python main.py train-retrieval --save-item-embeddings")
    print("  python main.py recommend --user-id u_00007")
    print(
        "  python main.py recommend --user-id u_00007 "
        "--retrieval-artifacts-dir artifacts/pytorch_retrieval"
    )
    print()
    parser.print_help()
    return 0


def main() -> int: 
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        return print_overview(parser)

    if args.command == "ui":
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app_streamlit.py",
            "--server.port",
            str(args.port),
        ]
        return run_command(command)

    if args.command == "train-ranker":
        command = [
            sys.executable,
            "src/pipeline/train.py",
            "--interactions-csv",
            args.interactions_csv,
            "--users-csv",
            args.users_csv,
            "--banners-csv",
            args.banners_csv,
            "--output-dir",
            args.output_dir,
            "--valid-days",
            str(args.valid_days),
            "--iterations",
            str(args.iterations),
            "--learning-rate",
            str(args.learning_rate),
            "--depth",
            str(args.depth),
            "--random-seed",
            str(args.random_seed),
        ]
        return run_command(command)

    if args.command == "recommend":
        command = [
            sys.executable,
            "src/pipeline/inference.py",
            "--user-id",
            args.user_id,
            "--users-csv",
            args.users_csv,
            "--banners-csv",
            args.banners_csv,
            "--artifacts-dir",
            args.artifacts_dir,
            "--interactions-csv",
            args.interactions_csv,
            "--top-k",
            str(args.top_k),
            "--score-mode",
            args.score_mode,
            "--retrieval-top-n",
            str(args.retrieval_top_n),
            "--output-csv",
            args.output_csv,
        ]
        if args.as_of_date:
            command.extend(["--as-of-date", args.as_of_date])
        if args.only_active:
            command.append("--only-active")
        if args.exclude_seen:
            command.append("--exclude-seen")
        if args.retrieval_artifacts_dir:
            command.extend(["--retrieval-artifacts-dir", args.retrieval_artifacts_dir])
        return run_command(command)

    if args.command == "train-retrieval":
        command = [
            sys.executable,
            "src/scripts/pytorch.py",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--embedding-dim",
            str(args.embedding_dim),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--k",
            str(args.k),
            "--seed",
            str(args.seed),
            "--output-dir",
            args.output_dir,
        ]
        if args.save_item_embeddings:
            command.append("--save-item-embeddings")
        return run_command(command)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
