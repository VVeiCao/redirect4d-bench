#!/usr/bin/env python3
"""Check Redirect4D-Bench runtime environments."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shutil
import sys


PROFILES = {
    "bench": ["numpy", "cv2", "tqdm", "yt_dlp", "trimesh", "viser", "openai", "redirect4d_bench"],
    "reconstruction": ["torch", "cv2", "open3d", "sgm", "diffsynth", "vipe", "moge"],
    "sam3": ["torch", "cv2", "imageio_ffmpeg", "sam3"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="bench",
        help="Environment profile to check.",
    )
    return parser.parse_args()


def check_modules(modules: list[str]) -> None:
    for module in modules:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "unknown")
        print(f"{module}: {version}")


def main() -> None:
    args = parse_args()
    print(f"python: {sys.executable}")
    print(f"profile: {args.profile}")
    check_modules(PROFILES[args.profile])
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        candidate = Path(sys.executable).resolve().parent / "ffmpeg"
        if candidate.exists():
            ffmpeg = str(candidate)
    if not ffmpeg and args.profile == "sam3":
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    print(f"ffmpeg: {ffmpeg}")


if __name__ == "__main__":
    main()
