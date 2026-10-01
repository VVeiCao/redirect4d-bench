#!/usr/bin/env python3
"""SAM3 tracker (mask-seed) propagation using SOURCE video's frame-0 mask
as identity anchor.

Why not the target pseudo-GT `mask.mp4` frame 0?
  - Source mask is the pristine hand/AIM-curated mask of the intended
    foreground subject.
  - R4D's refined mask went through SAM3 once already → adds boundary
    variance AND creates circular dependency when we later compare the
    generated-video mask to the released target pseudo-GT mask.

Usage:
    python extract_propagated.py --shard 0/8

Output:
    masks/seeded_from_source/<cfg>/<key>.mp4   (binary mask video, 15fps)
    masks/seeded_from_source/<cfg>/<key>.json  (per-frame area + seed area)

This pipeline CANNOT detect vanishing — the tracker will hallucinate a
mask even if the subject is absent. Downstream evaluator gates by text
mode's per-frame detection signal.
"""
import argparse
import json
import os
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
import numpy as np
import torch

# Reuse the SAM3 helper functions shipped with this repository.
sys.path.insert(0, str(REPO_ROOT / "metrics" / "sam3_refine" / "scripts"))
from refine_masks_sam3 import (
    build_model, extract_frames_to_dir, read_mask_frames,
    propagate_bidirectional, write_mask_mp4,
)

PREDICTIONS_ROOT = REPO_ROOT / "predictions"
DATASET_ROOT = REPO_ROOT / "data" / "redirect4d_bench" / "tracks"
SWEEP_ROOT_BASE = REPO_ROOT / "outputs" / "object_metric_masks"

CONFIGS_DEFAULT: list[str] = []
MASK_THRESHOLD = 127

# Available seed sources:
#   src: source video's frame-0 mask, tracks/<track>/masks/00000.png
#        (pristine, original camera view, frame 0 of redirected ≈ source since
#         redirected camera starts at source pose)
#   gt:  target pseudo-GT mask at the redirected view,
#        redirected/<traj>/mask.mp4 frame 0
#        (already SAM3-processed, identity matches the downstream GT)
SEED_SOURCES = {"src", "gt"}

# Prompt-mode defines how the seed mask is converted into a SAM3 prompt:
#   mask        — pass binary mask directly (add_new_mask); tracker stores it as-is
#   pt_erosion  — erode mask, sample 3 points inside; SAM3 re-segments generated f0
#   pt_box      — mask centroid (pos click) + mask bbox (box prompt); SAM3 re-segments
PROMPT_MODES = {"mask", "pt_erosion", "pt_box"}


def seed_path_for(seed_source: str, track: str, traj: str) -> Path:
    if seed_source == "src":
        return DATASET_ROOT / track / "masks" / "00000.png"
    if seed_source == "gt":
        return DATASET_ROOT / track / "redirected" / traj / "mask.mp4"
    raise ValueError(f"unknown seed_source: {seed_source!r}")


def read_seed_mask(path: Path):
    """Load a binary seed mask from either a PNG or an MP4 (frame 0)."""
    if path.suffix == ".mp4":
        frames, _fps = read_mask_frames(path)
        if not frames:
            raise RuntimeError(f"empty mask video {path}")
        gray = frames[0]
    elif path.suffix == ".png":
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"cannot read seed {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"unsupported seed format: {path.suffix}")
    return gray > MASK_THRESHOLD


def enumerate_cases(configs, seed_source: str):
    configs = list(configs)
    if not configs:
        configs = sorted(p.name for p in PREDICTIONS_ROOT.iterdir() if (p / "videos").is_dir())
    ref = PREDICTIONS_ROOT / configs[0] / "videos"
    keys = sorted(p.stem for p in ref.glob("*.mp4"))
    out = []
    for key in keys:
        track, traj = key.split("__", 1)
        seed = seed_path_for(seed_source, track, traj)
        if not seed.exists():
            print(f"WARN: missing {seed_source} seed for {key}", file=sys.stderr)
            continue
        for cfg in configs:
            video = PREDICTIONS_ROOT / cfg / "videos" / f"{key}.mp4"
            if video.exists():
                out.append((cfg, video, seed, key))
    return out


def derive_point_prompts(seed_mask: np.ndarray, prompt_mode: str,
                         rng: np.random.Generator):
    """Return dict {points, labels, box} (coords normalized to [0,1])."""
    H, W = seed_mask.shape
    ys, xs = np.where(seed_mask)
    cx, cy = xs.mean() / W, ys.mean() / H

    if prompt_mode == "pt_erosion":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        eroded = cv2.erode(seed_mask.astype(np.uint8), kernel).astype(bool)
        if eroded.sum() < 3:
            eroded = seed_mask
        ey, ex = np.where(eroded)
        idx = rng.choice(len(ey), size=min(3, len(ey)), replace=False)
        pts = [[ex[i]/W, ey[i]/H] for i in idx]
        return {"points": pts, "labels": [1]*len(pts), "box": None}

    if prompt_mode == "pt_box":
        y0, y1 = ys.min()/H, ys.max()/H
        x0, x1 = xs.min()/W, xs.max()/W
        return {"points": [[cx, cy]], "labels": [1],
                "box": [x0, y0, x1, y1]}

    raise ValueError(f"unknown point prompt_mode: {prompt_mode}")


def extract_one(predictor, generated_path: Path, seed_path: Path,
                prompt_mode: str, out_mp4: Path, out_json: Path,
                rng: np.random.Generator):
    seed = read_seed_mask(seed_path)
    H, W = seed.shape

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp) / "frames"
        n = extract_frames_to_dir(generated_path, frames_dir)

        state = predictor.init_state(video_path=str(frames_dir))
        predictor.clear_all_points_in_video(state)

        if not seed.any():
            raise RuntimeError(f"seed mask is empty: {seed_path}")

        if prompt_mode == "mask":
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0, obj_id=1,
                mask=torch.from_numpy(seed).to(torch.bool),
            )
            prompt_meta = {"mode": "mask", "seed_area": int(seed.sum())}
        else:
            prompts = derive_point_prompts(seed, prompt_mode, rng)
            pts = torch.tensor(prompts["points"], dtype=torch.float32)
            lbs = torch.tensor(prompts["labels"], dtype=torch.int32)
            box = (torch.tensor(prompts["box"], dtype=torch.float32)
                   if prompts["box"] is not None else None)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0, obj_id=1,
                points=pts, labels=lbs, box=box,
                rel_coordinates=True,
            )
            prompt_meta = {"mode": prompt_mode, **prompts,
                           "seed_area": int(seed.sum())}

        masks = propagate_bidirectional(predictor, state, n, (H, W))

    # metadata
    rows = []
    for i, mm in enumerate(masks):
        area = int(mm.sum())
        rows.append({"frame": i, "mask_area": area})
    detected = sum(1 for r in rows if r["mask_area"] > 0)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    write_mask_mp4(masks, out_mp4, fps=15.0)
    with open(out_json, "w") as f:
        json.dump({
            "video": str(generated_path),
            "seed": str(seed_path),
            "prompt": prompt_meta,
            "seed_area": int(seed.sum()),
            "n_frames": n, "H": H, "W": W,
            "detected_frames": detected,
            "detection_rate": detected / n if n else 0.0,
            "note": ("detection_rate from mask-seed propagation is NOT a "
                     "reliable vanishing signal; tracker may hallucinate. "
                     "Gate via text-mode detection downstream."),
            "per_frame": rows,
        }, f)
    return n, detected


def derive_output_dir(prompt_mode: str, seed_source: str) -> str:
    """Output subdir name under masks/, grouped by how the prompt is constructed."""
    if prompt_mode == "mask":
        return f"seeded_from_{'source' if seed_source == 'src' else 'gt'}"
    suffix = "" if seed_source == "src" else "_gt"
    return f"seeded_from_{prompt_mode}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed-source", required=True, choices=sorted(SEED_SOURCES),
                    help="src = tracks/<track>/masks/00000.png, "
                         "gt = redirected/<traj>/mask.mp4 frame 0")
    ap.add_argument("--prompt-mode", default="mask", choices=sorted(PROMPT_MODES),
                    help="How to pass the seed to SAM3: 'mask' = add_new_mask "
                         "(hard-store); 'pt_erosion' = 3 points inside eroded "
                         "mask; 'pt_box' = centroid click + mask bbox.")
    ap.add_argument("--configs", default=",".join(CONFIGS_DEFAULT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="rerun even if output exists")
    ap.add_argument("--seed-rng", type=int, default=7,
                    help="random seed for erosion point sampling")
    args = ap.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    requested_configs = [c for c in args.configs.split(",") if c]
    configs = (
        [c for c in requested_configs if c in CONFIGS_DEFAULT]
        if CONFIGS_DEFAULT
        else requested_configs
    )

    sweep_dir_name = derive_output_dir(args.prompt_mode, args.seed_source)
    out_root = SWEEP_ROOT_BASE / sweep_dir_name
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed_rng)

    all_cases = enumerate_cases(configs, args.seed_source)
    my_cases = [c for i, c in enumerate(all_cases) if i % shard_n == shard_i]
    if args.limit:
        my_cases = my_cases[:args.limit]

    tag = f"[shard {shard_i}/{shard_n} seed={args.seed_source} mode={args.prompt_mode}]"
    print(f"{tag} total={len(all_cases)} my={len(my_cases)} -> {out_root}", flush=True)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    predictor = build_model(torch.device("cuda"))
    print(f"{tag} model loaded in {time.time()-t0:.1f}s", flush=True)

    t_start = time.time()
    done = skipped = failed = 0
    for idx, (cfg, video, seed_p, key) in enumerate(my_cases, 1):
        out_mp4 = out_root / cfg / f"{key}.mp4"
        out_json = out_root / cfg / f"{key}.json"
        if (not args.force) and out_mp4.exists() and out_json.exists():
            skipped += 1
            continue
        t1 = time.time()
        try:
            n, det = extract_one(predictor, video, seed_p,
                                 args.prompt_mode, out_mp4, out_json, rng)
            done += 1
            if idx % 10 == 0 or idx == 1:
                print(f"{tag} {idx}/{len(my_cases)} {cfg}/{key[:40]} "
                      f"det={det}/{n} ({time.time()-t1:.1f}s)", flush=True)
        except Exception as e:
            failed += 1
            print(f"{tag} FAIL {cfg}/{key}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            import traceback; traceback.print_exc()

    print(f"{tag} DONE done={done} skipped={skipped} failed={failed} "
          f"in {(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
