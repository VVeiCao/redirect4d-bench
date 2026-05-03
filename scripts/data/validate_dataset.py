#!/usr/bin/env python3
"""Validate a local Redirect4D-Bench data installation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN_PUBLIC_NAMES = {
    "video.mp4",
    "generated.mp4",
    "reference.png",
    "original_images.mp4",
    "output_video.mp4",
    "rendered_mask.mp4",
}
FORBIDDEN_PUBLIC_DIRS = {"frames", "raw_images", "videos", "inference"}
SAMPLE_ALLOWED_SOURCE_NAMES = {"video.mp4", "input.mp4", "original_images.mp4"}
SAMPLE_ALLOWED_SOURCE_DIRS = {"frames"}
REQUIRED_TRACK_ITEMS = ("mask_video.mp4", "masks", "pointcloud")
REQUIRED_TRAJ_ITEMS = ("trajectory.json", "prompt.txt", "mask.mp4", "depth.mp4")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_public_dataset(dataset_root: Path, expected_tracks: int, expected_cases: int) -> list[str]:
    errors: list[str] = []
    tracks_path = dataset_root / "tracks.jsonl"
    cases_path = dataset_root / "cases.jsonl"
    if not tracks_path.exists():
        errors.append(f"missing {tracks_path}")
        return errors
    if not cases_path.exists():
        errors.append(f"missing {cases_path}")
        return errors
    tracks = read_jsonl(tracks_path)
    cases = read_jsonl(cases_path)
    if len(tracks) != expected_tracks:
        errors.append(f"expected {expected_tracks} tracks, found {len(tracks)}")
    if len(cases) != expected_cases:
        errors.append(f"expected {expected_cases} cases, found {len(cases)}")

    tracks_root = dataset_root / "tracks"
    for row in tracks:
        track_root = tracks_root / row["track"]
        for item in REQUIRED_TRACK_ITEMS:
            if not (track_root / item).exists():
                errors.append(f"missing {track_root / item}")
    for row in cases:
        traj_root = tracks_root / row["track"] / "redirected" / row["trajectory"]
        for item in REQUIRED_TRAJ_ITEMS:
            if not (traj_root / item).exists():
                errors.append(f"missing {traj_root / item}")

    dataset_is_sample = dataset_root.name == "sample"
    for path in dataset_root.rglob("*"):
        rel = path.relative_to(dataset_root).parts
        if path.is_symlink():
            errors.append(f"public dataset contains symlink: {path}")
        is_sample_path = dataset_is_sample or bool(rel and rel[0] == "sample")
        forbidden_names = FORBIDDEN_PUBLIC_NAMES
        forbidden_dirs = FORBIDDEN_PUBLIC_DIRS
        if is_sample_path:
            forbidden_names = FORBIDDEN_PUBLIC_NAMES - SAMPLE_ALLOWED_SOURCE_NAMES
            forbidden_dirs = FORBIDDEN_PUBLIC_DIRS - SAMPLE_ALLOWED_SOURCE_DIRS
        if path.name in forbidden_names:
            errors.append(f"public dataset contains forbidden source/output file: {path}")
        if any(part in forbidden_dirs for part in rel):
            errors.append(f"public dataset contains forbidden source/output dir: {path}")
    return errors


def validate_restricted_sources(source_root: Path, expected_tracks: int) -> list[str]:
    errors: list[str] = []
    if not source_root.exists():
        return [f"missing restricted source root: {source_root}"]
    tracks = sorted(p for p in (source_root / "tracks").iterdir() if p.is_dir())
    if len(tracks) != expected_tracks:
        errors.append(f"expected {expected_tracks} restricted source tracks, found {len(tracks)}")
    for track in tracks:
        if not (track / "video.mp4").exists():
            errors.append(f"missing {track / 'video.mp4'}")
        frames = list((track / "frames").glob("*.png")) if (track / "frames").exists() else []
        if len(frames) != 45:
            errors.append(f"{track / 'frames'} has {len(frames)} PNG frames")
    return errors


def validate_eval_cache(root: Path, expected_cases: int) -> list[str]:
    errors: list[str] = []
    if not root.exists():
        return [f"missing eval cache root: {root}"]
    mask_root = root / "object_metric_masks" / "seeded_from_pt_box"
    recog_root = root / "object_recognition_cache" / "qwen32b_v12_full"
    if not mask_root.exists():
        errors.append(f"missing object mask cache: {mask_root}")
    else:
        for method_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
            masks = list(method_dir.glob("*.mp4"))
            if len(masks) != expected_cases:
                errors.append(f"{method_dir.name} has {len(masks)} object masks, expected {expected_cases}")
            if any(p.is_symlink() for p in masks):
                errors.append(f"{method_dir.name} contains symlinked object masks")
    if not recog_root.exists():
        errors.append(f"missing recognition cache: {recog_root}")
    elif not any(recog_root.rglob("*")):
        errors.append(f"empty recognition cache: {recog_root}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/redirect4d_bench"))
    parser.add_argument(
        "--restricted-source-root",
        type=Path,
        default=Path("data/reconstructed_source_tracks"),
    )
    parser.add_argument("--eval-cache-root", type=Path, default=Path("outputs/eval_cache"))
    parser.add_argument("--expected-tracks", type=int, default=62)
    parser.add_argument("--expected-cases", type=int, default=83)
    parser.add_argument("--skip-restricted", action="store_true")
    parser.add_argument("--check-eval-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_public_dataset(args.dataset_root, args.expected_tracks, args.expected_cases)
    if not args.skip_restricted:
        errors.extend(validate_restricted_sources(args.restricted_source_root, args.expected_tracks))
    if args.check_eval_cache:
        errors.extend(validate_eval_cache(args.eval_cache_root, args.expected_cases))
    if errors:
        print("[fail] data validation errors:")
        for error in errors[:80]:
            print(f"  - {error}")
        if len(errors) > 80:
            print(f"  ... {len(errors) - 80} more")
        return 1
    print("[ok] Redirect4D-Bench data installation is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
