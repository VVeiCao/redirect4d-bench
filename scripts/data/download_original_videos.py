#!/usr/bin/env python3
"""Download original videos referenced by Redirect4D-Bench metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from redirect4d_bench.data.metadata import group_tracks_by_video, load_metadata, read_track_list, track_items, youtube_url
from redirect4d_bench.data.youtube import DEFAULT_FORMAT, download_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/redirect4d_bench/metadata.json"))
    parser.add_argument("--track-list", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/original_videos"))
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--cookies", type=Path, default=None)
    parser.add_argument("--cookies-from-browser", default=None)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--rate-limit", default=None)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    selected = read_track_list(args.track_list)
    grouped = group_tracks_by_video(track_items(metadata, selected))

    for video_id, tracks in tqdm(grouped.items(), desc="videos"):
        url = youtube_url(tracks[0][1])
        if args.dry_run:
            print(f"{video_id}\t{url}\t{len(tracks)} tracks")
            continue
        download_video(
            url,
            args.output_dir,
            video_id=video_id,
            fmt=args.format,
            cookies=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
            proxy=args.proxy,
            rate_limit=args.rate_limit,
            retries=args.retries,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
