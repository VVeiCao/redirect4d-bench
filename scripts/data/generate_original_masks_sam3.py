#!/usr/bin/env python3
"""Generate source-view foreground masks from reconstructed clips with SAM3.

This is the metadata-only scale-up bridge: after original videos are downloaded
and cropped into benchmark clips, the category in `metadata.json` is used as the
open-vocabulary SAM3 text prompt. The generated masks are written to the same
layout consumed by the Redirect4D reconstruction stage:

    <dataset-root>/tracks/<track>/masks/00000.png
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM3_SOURCE = ROOT / "third_party" / "sam3"

sys.path.insert(0, str(ROOT / "src"))
from redirect4d_bench.data.metadata import load_metadata, read_track_list, track_items  # noqa: E402


def configure_reproducibility(seed: int, *, deterministic: bool, warn_only: bool, allow_tf32: bool) -> None:
    import random

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def build_predictor(args: argparse.Namespace):
    sam3_source = args.sam3_source.resolve()
    if not (sam3_source / "sam3").is_dir():
        raise FileNotFoundError(
            f"SAM3 source tree not found at {sam3_source}. "
            "Run `git submodule update --init --recursive third_party/sam3`."
        )
    sys.path.insert(0, str(sam3_source))
    from sam3 import build_sam3_predictor

    return build_sam3_predictor(
        version=args.sam3_version,
        compile=False,
        async_loading_frames=False,
    )


def encode_mask_video(mask_dir: Path, out_path: Path, fps: float) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(mask_dir / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def collect_propagation(model, session_id: str) -> dict[int, np.ndarray]:
    import torch

    masks_by_frame: dict[int, np.ndarray] = {}
    for response in model.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
        frame_idx = response.get("frame_index")
        if frame_idx is None:
            continue
        outputs = response.get("outputs", {})
        binary_masks = outputs.get("out_binary_masks")
        if binary_masks is None:
            continue
        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.detach().cpu().numpy()

        union = None
        for mask in binary_masks:
            if mask.ndim == 3:
                mask = mask[0]
            mask = mask.astype(bool)
            union = mask if union is None else (union | mask)
        if union is not None:
            masks_by_frame[int(frame_idx)] = union
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return masks_by_frame


def frame_shape(frames_dir: Path) -> tuple[int, int]:
    first = next(iter(sorted(frames_dir.glob("*.png"))), None)
    if first is None:
        first = next(iter(sorted(frames_dir.glob("*.jpg"))), None)
    if first is None:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    img = cv2.imread(str(first), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read {first}")
    h, w = img.shape[:2]
    return h, w


def write_masks(masks_by_frame: dict[int, np.ndarray], frames_dir: Path, out_dir: Path, *, overwrite: bool) -> int:
    frame_paths = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")))
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = frame_shape(frames_dir)
    empty = np.zeros((h, w), dtype=np.uint8)
    for idx, _path in enumerate(frame_paths):
        mask = masks_by_frame.get(idx)
        if mask is None:
            m8 = empty
        else:
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            m8 = mask.astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), m8)
    return len(frame_paths)


def prompt_for_track(info: dict, template: str) -> str:
    return template.format(category=info.get("category", "object")).strip()


def generate_one(model, track: str, info: dict, args: argparse.Namespace) -> None:
    frames_dir = args.processed_root / "tracks" / track / "frames"
    out_dir = args.dataset_root / "tracks" / track / "masks"
    if out_dir.exists() and not args.overwrite and any(out_dir.glob("*.png")):
        print(f"[skip] masks exist: {out_dir}", flush=True)
        return
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)

    prompt = prompt_for_track(info, args.prompt_template)
    response = model.handle_request({"type": "start_session", "resource_path": str(frames_dir)})
    session_id = response["session_id"]
    try:
        model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": args.prompt_frame,
                "text": prompt,
            }
        )
        masks_by_frame = collect_propagation(model, session_id)
    finally:
        try:
            model.handle_request({"type": "close_session", "session_id": session_id})
        except Exception:
            pass

    n = write_masks(masks_by_frame, frames_dir, out_dir, overwrite=args.overwrite)
    encode_mask_video(out_dir, args.dataset_root / "tracks" / track / "mask_video.mp4", info.get("output_fps", 15))
    print(f"[ok] {track}: prompt={prompt!r}, frames={n}, nonempty={len(masks_by_frame)} -> {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/redirect4d_bench/metadata.json"))
    parser.add_argument("--track-list", type=Path)
    parser.add_argument("--track", action="append", default=[])
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/reconstructed_source_tracks"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/redirect4d_bench"))
    parser.add_argument("--prompt-template", default="{category}")
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--sam3-source", type=Path, default=DEFAULT_SAM3_SOURCE)
    parser.add_argument("--sam3-version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--deterministic-strict", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    selected = list(args.track)
    if args.track_list:
        selected.extend(read_track_list(args.track_list) or [])
    tracks = track_items(metadata, selected or None)
    print(f"[tracks] {len(tracks)}", flush=True)
    if args.dry_run:
        for track, info in tracks:
            print(f"{track}: {prompt_for_track(info, args.prompt_template)}")
        return

    configure_reproducibility(
        args.seed,
        deterministic=not args.no_deterministic,
        warn_only=not args.deterministic_strict,
        allow_tf32=args.allow_tf32,
    )
    model = build_predictor(args)
    for track, info in tracks:
        generate_one(model, track, info, args)


if __name__ == "__main__":
    main()
