#!/usr/bin/env python3
"""Refine rough point-cloud-rendered masks in Redirect4D-Bench using
SAM 3 base (Sam3TrackerPredictor).

Method: use frame 0 of the rough mask (`redirected/<view>/rendered_mask.mp4`) as a
single mask prompt, then forward + reverse propagate through the 45-frame
generated video. One mask prompt per view; no keyframes elsewhere. This was
empirically the cleanest of several strategies tried.

Output: `redirected/<view>/mask.mp4` (overwrites).

Usage:
    python refine_masks_sam3.py --limit 1
    python refine_masks_sam3.py --tracks bear,dancer_5
    python refine_masks_sam3.py --shard 0/4           # for parallel dispatch
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HF_HOME = Path(os.environ.get("REDIRECT4D_HF_HOME", Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_model

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

DATASET_ROOT = Path(os.environ.get("REDIRECT4D_DATASET_ROOT", REPO_ROOT / "data" / "redirect4d_bench"))
TRACKS_ROOT = DATASET_ROOT / "tracks"
OUT_NAME = "mask.mp4"
MASK_THRESHOLD = 127


def read_mask_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames, fps


def extract_frames_to_dir(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    return n


def write_mask_mp4(masks, out_path: Path, fps: float):
    h, w = masks[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, m in enumerate(masks):
            m8 = (m.astype(np.uint8) * 255) if m.dtype == bool else m.astype(np.uint8)
            if m8.max() <= 1:
                m8 = m8 * 255
            m3 = np.stack([m8, m8, m8], axis=-1)
            cv2.imwrite(str(tmp_dir / f"{i:05d}.png"), m3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-framerate", f"{fps:g}",
            "-i", str(tmp_dir / "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-vf", f"scale={w}:{h}",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)


def collect_views():
    for td in sorted(TRACKS_ROOT.iterdir()):
        if not td.is_dir():
            continue
        red = td / "redirected"
        if not red.is_dir():
            continue
        for vd in sorted(red.iterdir()):
            if not vd.is_dir():
                continue
            if (vd / "generated.mp4").exists() and (vd / "rendered_mask.mp4").exists():
                yield td.name, vd.name, vd


def propagate_bidirectional(predictor, state, n, prior_shape):
    """Forward then reverse propagate, average logits per frame."""
    fwd = {}
    for frame_idx, _, _, video_res_masks, _ in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=n,
            reverse=False, propagate_preflight=True):
        fwd[frame_idx] = video_res_masks[0, 0].detach().float().cpu().numpy()

    rev = {}
    for frame_idx, _, _, video_res_masks, _ in predictor.propagate_in_video(
            state, start_frame_idx=n - 1, max_frame_num_to_track=n,
            reverse=True, propagate_preflight=False):
        rev[frame_idx] = video_res_masks[0, 0].detach().float().cpu().numpy()

    out = []
    for i in range(n):
        f, r = fwd.get(i), rev.get(i)
        if f is None and r is None:
            out.append(np.zeros(prior_shape, dtype=bool))
        elif f is None:
            out.append(r > 0)
        elif r is None:
            out.append(f > 0)
        else:
            out.append(((f + r) / 2.0) > 0)
    return out


def refine_one_view(predictor, generated_path: Path, rough_mask_path: Path,
                    out_path: Path):
    rough_frames, fps = read_mask_frames(rough_mask_path)
    n = len(rough_frames)
    rough0 = rough_frames[0] > MASK_THRESHOLD
    H, W = rough0.shape

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp) / "frames"
        n_frames = extract_frames_to_dir(generated_path, frames_dir)
        if n_frames != n:
            n = min(n, n_frames)

        state = predictor.init_state(video_path=str(frames_dir))
        predictor.clear_all_points_in_video(state)

        if rough0.any():
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                mask=torch.from_numpy(rough0).to(torch.bool),
            )

        refined = propagate_bidirectional(predictor, state, n, (H, W))

    write_mask_mp4(refined, out_path, fps)
    return n


def build_model(device):
    sam3_model = build_sam3_video_model(device=str(device))
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    return predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="",
                    help="comma-separated substring filter on track names")
    ap.add_argument("--view-regex", default="",
                    help="regex filter for view folder names (e.g. 'yaw_-110')")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only first N matching pairs")
    ap.add_argument("--shard", default="",
                    help='shard spec "I/N" for parallel workers (e.g. "0/4")')
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    track_filters = [s for s in args.tracks.split(",") if s]
    view_re = re.compile(args.view_regex) if args.view_regex else None
    shard_i, shard_n = 0, 1
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    targets = []
    idx = 0
    for tname, vname, vdir in collect_views():
        if track_filters and not any(f in tname for f in track_filters):
            continue
        if view_re and not view_re.search(vname):
            continue
        if idx % shard_n != shard_i:
            idx += 1
            continue
        idx += 1
        targets.append((tname, vname, vdir))
        if args.limit and len(targets) >= args.limit:
            break

    print(f"Matched {len(targets)} (track, view) pairs")
    if args.dry_run:
        for t, v, _ in targets:
            print(f"  {t} / {v}")
        return

    print("Loading SAM 3 base video model...")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    predictor = build_model(device)
    print("Model loaded.\n")

    total_t0 = time.time()
    for i, (tname, vname, vdir) in enumerate(targets, 1):
        out = vdir / OUT_NAME
        t0 = time.time()
        print(f"[{i}/{len(targets)}] {tname} / {vname} ...", flush=True)
        try:
            n = refine_one_view(
                predictor,
                vdir / "generated.mp4",
                vdir / "rendered_mask.mp4",
                out,
            )
            print(f"  -> {out.name}  ({n} frames, {time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print(f"\nDone. Total {time.time()-total_t0:.1f}s for {len(targets)} views.")


if __name__ == "__main__":
    main()
