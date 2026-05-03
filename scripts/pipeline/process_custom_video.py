#!/usr/bin/env python3
"""Prepare a 45-frame custom Redirect4D-style clip from a user video.

This is the lightweight scale-up entrypoint. It samples frames, resizes them to
832x480, writes `frames/`, `input.mp4`, and masks. For high-quality
reconstruction, pass real foreground masks via `--mask-dir` or `--mask-video`.
Without masks, the script writes all-white placeholder masks so the downstream
folder structure is complete, but reconstruction quality will be poor.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def ffmpeg_bin() -> str:
    candidate = Path(sys.executable).resolve().parent / "ffmpeg"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("ffmpeg not found")


def resize_letterbox(frame, width: int, height: int):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def read_mask_video(path: Path, n: int, width: int, height: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open mask video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), n).astype(int)
    masks = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read mask frame {idx}")
        gray = cv2.cvtColor(resize_letterbox(frame, width, height), cv2.COLOR_BGR2GRAY)
        masks.append((gray > 127).astype(np.uint8) * 255)
    cap.release()
    return masks


def encode(frames_dir: Path, pattern: str, output: Path, fps: float) -> None:
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--case-name", default="custom_case")
    parser.add_argument("--num-frames", type=int, default=45)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--mask-video", type=Path)
    parser.add_argument("--allow-placeholder-mask", action="store_true")
    overwrite = parser.add_mutually_exclusive_group()
    overwrite.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Replace --out-root if it already exists. This is the default.",
    )
    overwrite.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if --out-root already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out_root} already exists; remove it or omit --no-overwrite")
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_root / "frames"
    masks_dir = args.out_root / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {args.input_video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end_frame if args.end_frame is not None else total - 1
    indices = np.linspace(args.start_frame, end, args.num_frames).astype(int)

    for out_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_idx}")
        resized = resize_letterbox(frame, args.width, args.height)
        cv2.imwrite(str(frames_dir / f"{out_idx:05d}.png"), resized)
    cap.release()

    if args.mask_dir:
        mask_paths = sorted(args.mask_dir.glob("*.png"))
        if len(mask_paths) < args.num_frames:
            raise ValueError(f"mask-dir has {len(mask_paths)} PNGs, expected at least {args.num_frames}")
        for i, path in enumerate(mask_paths[: args.num_frames]):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"cannot read mask: {path}")
            mask = cv2.resize(mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(masks_dir / f"{i:05d}.png"), (mask > 127).astype(np.uint8) * 255)
    elif args.mask_video:
        for i, mask in enumerate(read_mask_video(args.mask_video, args.num_frames, args.width, args.height)):
            cv2.imwrite(str(masks_dir / f"{i:05d}.png"), mask)
    elif args.allow_placeholder_mask:
        print("[warn] writing all-white placeholder masks; pass real masks for quality")
        mask = np.full((args.height, args.width), 255, dtype=np.uint8)
        for i in range(args.num_frames):
            cv2.imwrite(str(masks_dir / f"{i:05d}.png"), mask)
    else:
        raise ValueError("pass --mask-dir/--mask-video, or --allow-placeholder-mask for a structural test")

    encode(frames_dir, "%05d.png", args.out_root / "input.mp4", args.fps)
    encode(masks_dir, "%05d.png", args.out_root / "mask_video.mp4", args.fps)
    metadata = {
        "case_name": args.case_name,
        "num_frames": args.num_frames,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "source_video": str(args.input_video),
        "sampled_frame_indices": [int(x) for x in indices],
        "mask_source": str(args.mask_dir or args.mask_video or "placeholder_all_white"),
    }
    (args.out_root / "custom_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[ok] wrote custom clip: {args.out_root}")


if __name__ == "__main__":
    main()
