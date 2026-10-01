"""Localization metric (Layer 3 of Object Rendering Quality cascade).

For each frame where both GT and pred have a mask, compute MaskIoU and
BBoxIoU. Cascade-conditioning on plausibility is done downstream by
overall/build_cascade.py.

Output:
  per_frame.csv : case, frame, gt_has, pred_has, present, mask_iou, bbox_iou
                  (iou fields empty if present==0)
  per_video.csv : case, n_present, cond_mask_iou, cond_bbox_iou
  summary.json

Usage:
    python eval_localization.py \
        --pred_dir outputs/object_metric_masks/seeded_from_pt_box/<method> \
        --out_dir  outputs/object_metrics/results/<method>/_localization
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT_DEFAULT = str(Path(__file__).resolve().parents[3] / "data" / "redirect4d_bench")
AREA_THRESH_DEFAULT = 32


def read_mask_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127)
    cap.release()
    if not frames: return np.zeros((0, 1, 1), dtype=bool)
    return np.stack(frames, axis=0)


def resize_mask_to(pred, h, w):
    if pred.shape[1] == h and pred.shape[2] == w: return pred
    out = np.zeros((pred.shape[0], h, w), dtype=bool)
    for i in range(pred.shape[0]):
        r = cv2.resize(pred[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        out[i] = r.astype(bool)
    return out


def mask_iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def bbox_of(mask):
    ys, xs = np.where(mask)
    if xs.size == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_iou(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw = max(0, ix2 - ix1 + 1); ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    a1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    a2 = (X2 - X1 + 1) * (Y2 - Y1 + 1)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def gt_path_for_key(key, dataset_root):
    track, traj = key.split("__", 1)
    traj_root = dataset_root / "tracks" / track / "redirected" / traj
    return traj_root / "mask.mp4"


def evaluate_case(pred_path, gt_path, area_thresh):
    pred = read_mask_video(pred_path)
    gt = read_mask_video(gt_path)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return None
    pred = resize_mask_to(pred, gt.shape[1], gt.shape[2])
    n = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:n], gt[:n]

    gt_has = np.array([int(m.sum()) > area_thresh for m in gt])
    pred_has = np.array([int(m.sum()) > area_thresh for m in pred])
    rows = []
    for i in range(n):
        r = {"frame": i,
             "gt_has": int(gt_has[i]),
             "pred_has": int(pred_has[i]),
             "present": int(bool(gt_has[i] and pred_has[i])),
             "mask_iou": "", "bbox_iou": ""}
        if r["present"]:
            mi = mask_iou(gt[i], pred[i])
            p_bb = bbox_of(pred[i]); g_bb = bbox_of(gt[i])
            bi = bbox_iou(g_bb, p_bb) if (p_bb and g_bb) else 0.0
            r["mask_iou"] = f"{mi:.4f}"
            r["bbox_iou"] = f"{bi:.4f}"
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_root", default=DATASET_ROOT_DEFAULT)
    ap.add_argument("--area_thresh", type=int, default=AREA_THRESH_DEFAULT)
    ap.add_argument("--glob", default="*.mp4")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob(args.glob))
    print(f"[localization] pred_dir={pred_dir}  n={len(pred_files)}", flush=True)

    all_frames, per_video = [], []
    t0 = time.time()
    for idx, p in enumerate(pred_files, 1):
        key = p.stem
        gt = gt_path_for_key(key, Path(args.dataset_root))
        if not gt.exists():
            print(f"[skip] {key}: no GT"); continue
        rows = evaluate_case(str(p), str(gt), args.area_thresh)
        if rows is None: continue
        mious = [float(r["mask_iou"]) for r in rows if r["mask_iou"] != ""]
        bious = [float(r["bbox_iou"]) for r in rows if r["bbox_iou"] != ""]
        for r in rows:
            r["case"] = key
            all_frames.append(r)
        per_video.append({
            "case": key,
            "n_present": len(mious),
            "cond_mask_iou": f"{np.mean(mious):.4f}" if mious else "0.0000",
            "cond_bbox_iou": f"{np.mean(bious):.4f}" if bious else "0.0000",
        })
        print(f"[{idx:3d}/{len(pred_files)}] {key[:58]:58s} "
              f"n_present={len(mious):2d}  cM={np.mean(mious) if mious else 0:.3f}  "
              f"cB={np.mean(bious) if bious else 0:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    with open(out_dir / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "frame", "gt_has", "pred_has", "present",
                                          "mask_iou", "bbox_iou"])
        w.writeheader(); w.writerows(all_frames)

    with open(out_dir / "per_video.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "n_present", "cond_mask_iou", "cond_bbox_iou"])
        w.writeheader(); w.writerows(per_video)

    cMs = [float(r["cond_mask_iou"]) for r in per_video]
    cBs = [float(r["cond_bbox_iou"]) for r in per_video]
    summary = {
        "metric": "localization",
        "n_videos": len(per_video),
        "cond_mask_iou_mean": float(np.mean(cMs)) if cMs else 0,
        "cond_bbox_iou_mean": float(np.mean(cBs)) if cBs else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
