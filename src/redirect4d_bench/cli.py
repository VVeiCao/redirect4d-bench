"""Small command-line helper for Redirect4D-Bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SAMPLE_ROOT = Path("data/redirect4d_bench/sample")


def sample_tracks(dataset_root: Path) -> list[str]:
    path = dataset_root / "tracks.jsonl"
    if not path.exists():
        return []
    tracks = []
    for line in path.read_text().splitlines():
        if line.strip():
            tracks.append(json.loads(line)["track"])
    return tracks


def print_quick_view() -> None:
    print(
        """Redirect4D-Bench quick commands

1. Download the two-animal sample:

   hf download vveicao/redirect4d-bench \\
     --repo-type dataset \\
     --include 'sample/**' \\
     --local-dir data/redirect4d_bench

2. Start Viser point-cloud + mask preview:

   python scripts/visualization/serve_pointcloud_viser.py \\
     --dataset-root data/redirect4d_bench/sample \\
     --track bear_NnAlfavy2us_003_001_seq1 \\
     --port 8091

3. Check local sample status:

   redirect4d-bench status
"""
    )


def cmd_status(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root
    print(f"dataset_root: {dataset_root}")
    if not dataset_root.exists():
        print("status: missing")
        print("hint: run `redirect4d-bench` to print the sample download command")
        return 1
    tracks = sample_tracks(dataset_root)
    print(f"status: found")
    print(f"tracks.jsonl: {'yes' if tracks else 'missing or empty'}")
    if tracks:
        print(f"tracks: {len(tracks)}")
        for track in tracks:
            print(f"  - {track}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redirect4d-bench",
        description="Redirect4D-Bench command-line helper.",
    )
    sub = parser.add_subparsers(dest="command")
    status = sub.add_parser("status", help="Check the local sample dataset.")
    status.add_argument("--dataset-root", type=Path, default=SAMPLE_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        return cmd_status(args)
    print_quick_view()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
