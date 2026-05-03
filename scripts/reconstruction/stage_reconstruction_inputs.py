#!/usr/bin/env python3
"""Stage processed source frames and masks for bundled reconstruction."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"no image files found in {path}")
    return files


def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; omit --no-overwrite to replace it")
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")


def stage_sequence(src_files: list[Path], dst_dir: Path, mode: str, overwrite: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(src_files):
        dst = dst_dir / f"{idx:05d}{src.suffix.lower()}"
        place_file(src, dst, mode, overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/reconstructed_source_tracks"),
        help="Root containing tracks/<track>/frames from source-video processing.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/redirect4d_bench"),
        help="Dataset root containing tracks/<track>/masks.",
    )
    parser.add_argument(
        "--r4d-root",
        type=Path,
        default=Path("reconstruction/redirect4d"),
        help="Bundled reconstruction root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override staged output root. Defaults to <r4d-root>/data_merged_reprocess.",
    )
    parser.add_argument("--mode", choices=("symlink", "copy", "hardlink"), default="symlink")
    overwrite = parser.add_mutually_exclusive_group()
    overwrite.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Replace staged files if they already exist. This is the default.",
    )
    overwrite.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if staged files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_dir = args.processed_root / "tracks" / args.track / "frames"
    mask_dir = args.dataset_root / "tracks" / args.track / "masks"
    frames = list_images(frame_dir)
    masks = list_images(mask_dir)
    if len(frames) != len(masks):
        raise ValueError(f"frame/mask count mismatch: {len(frames)} frames vs {len(masks)} masks")

    root = args.output_root if args.output_root is not None else args.r4d_root / "data_merged_reprocess"
    out_root = root / args.track
    stage_sequence(frames, out_root / "images", args.mode, args.overwrite)
    stage_sequence(masks, out_root / "masks", args.mode, args.overwrite)
    print(f"staged {len(frames)} frames and masks for {args.track}: {out_root}")


if __name__ == "__main__":
    main()
