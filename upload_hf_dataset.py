#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    power = min(int(math.log(num_bytes, 1024)), len(units) - 1)
    value = num_bytes / (1024**power)
    return f"{value:.2f} {units[power]}"


def default_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return min(64, max(8, cpu_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload JSONL shards to a Hugging Face dataset repo as fast as possible."
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default="serda-dev/turkish-raw-text-cleaned",
        help="Target Hugging Face repo id.",
    )
    parser.add_argument(
        "--source-dir",
        default="output/filtered",
        help="Directory containing JSONL files to upload.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Number of upload workers. Default uses all local CPU cores up to 64.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch name.",
    )
    parser.add_argument(
        "--no-high-performance",
        action="store_true",
        help="Do not enable HF_XET_HIGH_PERFORMANCE=1.",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        dest="include_patterns",
        default=None,
        help="Optional upload allow-pattern. Can be passed multiple times.",
    )
    return parser.parse_args()


def collect_stats(source_dir: Path, include_patterns: list[str] | None) -> tuple[int, int]:
    patterns = include_patterns or ["*.jsonl"]
    matched: set[Path] = set()
    for pattern in patterns:
        matched.update(source_dir.rglob(pattern))

    files = [path for path in matched if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    return len(files), total_bytes


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()

    if not source_dir.exists():
        print(f"Source directory does not exist: {source_dir}", file=sys.stderr)
        return 1

    if not args.no_high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    include_patterns = args.include_patterns or ["*.jsonl"]
    file_count, total_bytes = collect_stats(source_dir, include_patterns)

    if file_count == 0:
        print(
            f"No files matched in {source_dir} with patterns: {', '.join(include_patterns)}",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        api.whoami()
    except HfHubHTTPError:
        print("Authentication missing. Export HF_TOKEN or run `hf auth login` first.", file=sys.stderr)
        return 1

    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    print(f"Repo        : {args.repo_id} ({args.repo_type})")
    print(f"Source      : {source_dir}")
    print(f"Files       : {file_count}")
    print(f"Total size  : {format_size(total_bytes)}")
    print(f"Workers     : {args.workers}")
    print(f"High perf   : {os.environ.get('HF_XET_HIGH_PERFORMANCE', '0')}")
    print(f"Patterns    : {', '.join(include_patterns)}")
    print("Starting upload...")

    api.upload_large_folder(
        repo_id=args.repo_id,
        folder_path=source_dir,
        repo_type=args.repo_type,
        revision=args.revision,
        allow_patterns=include_patterns,
        num_workers=args.workers,
        print_report=True,
        print_report_every=20,
    )

    print("Upload completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
