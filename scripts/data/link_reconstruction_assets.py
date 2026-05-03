#!/usr/bin/env python3
"""Link reconstruction assets into benchmark track folders.

The reconstruction workspace stores point clouds as:
  prepared/<track>/global_background.ply
  prepared/<track>/<frame>/pointcloud/<frame>_foreground_1_view.npz

Evaluation and visualization code expects the public benchmark layout:
  tracks/<track>/video.mp4
  tracks/<track>/camera.json
  tracks/<track>/pointcloud/global_background.ply
  tracks/<track>/pointcloud/<frame>/foreground_1_view.npz

This script creates relative symlinks so the dataset binder is complete without
duplicating large point clouds. If a prompt source dataset is available, it also
links per-trajectory text prompts used by Redirect4D-style generation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


POINTCLOUD_FILES = [
    "foreground_1_view.ply",
    "foreground_1_view.npz",
    "foreground_5_views_aligned_smooth.ply",
    "foreground_5_views_aligned_smooth.npz",
]


def rel_symlink(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if not overwrite:
            return True
        if dst.is_dir() and not dst.is_symlink():
            raise IsADirectoryError(dst)
        dst.unlink()
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel)
    return True


def link_prompts(track_dir: Path, prompt_track_dir: Path, overwrite: bool) -> tuple[int, int]:
    linked = 0
    missing = 0
    redirected = track_dir / "redirected"
    if not redirected.exists():
        return linked, missing
    for traj_dir in sorted(p for p in redirected.iterdir() if p.is_dir()):
        src = prompt_track_dir / "redirected" / traj_dir.name / "prompt.txt"
        dst = traj_dir / "prompt.txt"
        if rel_symlink(src, dst, overwrite):
            linked += 1
        else:
            missing += 1
    return linked, missing


def link_track(
    track_dir: Path,
    prepared_dir: Path,
    processed_dir: Path,
    prompt_track_dir: Path | None,
    overwrite: bool,
) -> tuple[int, int]:
    linked = 0
    missing = 0

    if rel_symlink(processed_dir / "input.mp4", track_dir / "video.mp4", overwrite):
        linked += 1
    else:
        missing += 1

    if rel_symlink(prepared_dir / "global_camera.json", track_dir / "camera.json", overwrite):
        linked += 1
    else:
        missing += 1

    pc_root = track_dir / "pointcloud"
    if rel_symlink(prepared_dir / "global_background.ply", pc_root / "global_background.ply", overwrite):
        linked += 1
    else:
        missing += 1

    for frame_dir in sorted(p for p in prepared_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        src_pc = frame_dir / "pointcloud"
        if not src_pc.exists():
            continue
        for public_name in POINTCLOUD_FILES:
            suffix = public_name
            src = src_pc / f"{frame_dir.name}_{suffix}"
            dst = pc_root / frame_dir.name / public_name
            if rel_symlink(src, dst, overwrite):
                linked += 1
            else:
                missing += 1

    if prompt_track_dir is not None:
        prompt_linked, prompt_missing = link_prompts(track_dir, prompt_track_dir, overwrite)
        linked += prompt_linked
        missing += prompt_missing
    return linked, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/redirect4d_bench/tracks"))
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("reconstruction/redirect4d/outputs/prepared_vipe_lyra_noopt"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/reconstructed_source_tracks/tracks"),
    )
    parser.add_argument(
        "--prompt-source-root",
        type=Path,
        default=None,
        help="Optional source tracks root containing redirected/<traj>/prompt.txt files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    total_tracks = 0
    total_linked = 0
    total_missing = 0
    for track_dir in sorted(p for p in args.dataset_root.iterdir() if p.is_dir()):
        prepared_dir = args.prepared_root / track_dir.name
        processed_dir = args.processed_root / track_dir.name
        if not prepared_dir.exists():
            print(f"[missing prepared] {track_dir.name}")
            total_missing += 1
            continue
        prompt_track_dir = args.prompt_source_root / track_dir.name if args.prompt_source_root else None
        linked, missing = link_track(track_dir, prepared_dir, processed_dir, prompt_track_dir, args.overwrite)
        total_tracks += 1
        total_linked += linked
        total_missing += missing
        print(f"[link] {track_dir.name}: linked={linked} missing={missing}")

    print(f"[ok] tracks={total_tracks} linked={total_linked} missing={total_missing}")
    if total_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
