#!/usr/bin/env python3
"""Prepare clean Redirect4D-Bench release metadata from a working dataset.

The working dataset may contain extra internal files. This script treats
<source-root>/tracks/ as the source of truth, filters metadata to those
official tracks, and writes release-friendly metadata plus JSONL manifests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


EXCLUDE_PATTERNS = (
    "excluded_kids/",
    "generated.mp4",
    "generated.seed42_backup.mp4",
    "depth_inferno.mp4",
    "mask.mp4.seed42_backup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/private_full_release"),
        help="Working dataset root. Its tracks/ directory is authoritative.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Metadata JSON to filter. Defaults to <source-root>/metadata.json.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/redirect4d_bench"),
        help="Output staging root for clean metadata and manifests.",
    )
    parser.add_argument("--dataset-name", default="redirect4d_bench")
    parser.add_argument(
        "--write-file-manifest",
        action="store_true",
        help="Write release_files.txt listing files to copy/upload later.",
    )
    return parser.parse_args()


def official_track_names(source_root: Path) -> list[str]:
    tracks_root = source_root / "tracks"
    if not tracks_root.is_dir():
        raise FileNotFoundError(f"missing tracks directory: {tracks_root}")
    return sorted(p.name for p in tracks_root.iterdir() if p.is_dir())


def clean_metadata(metadata_path: Path, tracks: list[str], dataset_name: str) -> dict:
    with metadata_path.open() as f:
        raw = json.load(f)

    raw_tracks = raw.get("tracks", {})
    missing = [t for t in tracks if t not in raw_tracks]
    if missing:
        raise KeyError(f"{len(missing)} official tracks missing from metadata: {missing[:5]}")

    clean_tracks = {name: raw_tracks[name] for name in tracks}
    videos = {m.get("video_id") for m in clean_tracks.values() if m.get("video_id")}
    num_trajectories = sum(len(m.get("trajectories", [])) for m in clean_tracks.values())
    categories = Counter(m.get("category", "unknown") for m in clean_tracks.values())

    clean = dict(raw)
    clean["num_tracks"] = len(clean_tracks)
    clean["num_videos"] = len(videos)
    clean["num_trajectories"] = num_trajectories
    clean["categories"] = dict(categories.most_common())
    clean["tracks"] = clean_tracks
    clean["name"] = dataset_name
    return clean


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_track_rows(metadata: dict) -> list[dict]:
    rows = []
    for track_name, item in sorted(metadata["tracks"].items()):
        row = dict(item)
        row["track"] = track_name
        row["num_trajectories"] = len(item.get("trajectories", []))
        rows.append(row)
    return rows


def build_case_rows(metadata: dict) -> list[dict]:
    rows = []
    for track_name, item in sorted(metadata["tracks"].items()):
        for traj in item.get("trajectories", []):
            row = dict(item)
            row.pop("trajectories", None)
            row["case"] = f"{track_name}__{traj}"
            row["track"] = track_name
            row["trajectory"] = traj
            rows.append(row)
    return rows


def should_release(path: Path) -> bool:
    rel = path.as_posix()
    return not any(pattern in rel for pattern in EXCLUDE_PATTERNS)


def write_file_manifest(source_root: Path, out_root: Path, tracks: list[str]) -> None:
    lines = ["metadata.json", "tracks.jsonl", "cases.jsonl", "official_tracks.txt"]
    for track in tracks:
        track_root = source_root / "tracks" / track
        for path in sorted(p for p in track_root.rglob("*") if p.is_file()):
            rel = path.relative_to(source_root)
            if should_release(rel):
                lines.append(rel.as_posix())
    (out_root / "release_files.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    metadata_path = (args.metadata or (source_root / "metadata.json")).resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tracks = official_track_names(source_root)
    metadata = clean_metadata(metadata_path, tracks, args.dataset_name)

    with (out_root / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    write_jsonl(out_root / "tracks.jsonl", build_track_rows(metadata))
    write_jsonl(out_root / "cases.jsonl", build_case_rows(metadata))
    (out_root / "official_tracks.txt").write_text("\n".join(tracks) + "\n")

    if args.write_file_manifest:
        write_file_manifest(source_root, out_root, tracks)

    print(
        "Prepared clean metadata: "
        f"{metadata['num_tracks']} tracks, "
        f"{metadata['num_trajectories']} cases, "
        f"{metadata['num_videos']} videos -> {out_root}"
    )


if __name__ == "__main__":
    main()
