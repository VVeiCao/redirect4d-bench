#!/usr/bin/env python3
"""Reconstruct Redirect4D-Bench track videos from downloaded original videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from redirect4d_bench.data.metadata import group_tracks_by_video, load_metadata, read_track_list, track_items
from redirect4d_bench.reconstruction.source_tracks import reconstruct_tracks_for_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/redirect4d_bench/metadata.json"))
    parser.add_argument("--track-list", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=Path("data/original_videos"))
    parser.add_argument("--clip-dir", type=Path, default=Path("data/source_clips"))
    parser.add_argument(
        "--require-clips",
        action="store_true",
        help="Fail instead of falling back to full-video frame ids when released clips are unavailable.",
    )
    parser.add_argument(
        "--frame-source",
        choices=("auto", "clip", "full_video"),
        default="auto",
        help=(
            "auto prefers staged released clips when present; clip requires those clips; "
            "full_video skips scene-clip re-encoding and reads frozen video_frame_ids "
            "directly from the original video."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/reconstructed_source_tracks"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    selected = read_track_list(args.track_list)
    grouped = group_tracks_by_video(track_items(metadata, selected))

    for video_id, tracks in tqdm(grouped.items(), desc="videos"):
        video_path = args.video_dir / f"{video_id}.mp4"
        if args.dry_run:
            print(f"{video_path}\t{len(tracks)} tracks")
            for track_name, _ in tracks[:3]:
                print(f"  -> {args.output_root / 'tracks' / track_name / 'input.mp4'}")
            continue
        reconstruct_tracks_for_video(
            video_path,
            tracks,
            args.output_root,
            clip_root=args.clip_dir,
            require_clips=args.require_clips,
            frame_source=args.frame_source,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
