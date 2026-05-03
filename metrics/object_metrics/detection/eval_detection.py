"""Detection metric (Layer 1 of Object Rendering Quality cascade).

For each frame of each method output, does the predicted mask exist?
Output:
  per_frame.csv : case, frame, gt_has, pred_has, present (1 iff both)
  per_video.csv : case, n_frames, n_gt_has, n_pred_has, n_both, presence
  summary.json  : per-method aggregate

Usage:
    python eval_detection.py \
        --pred_dir outputs/object_metric_masks/seeded_from_pt_box/<method> \
        --out_dir  outputs/object_metrics/results/<method>/_detection

GT is read from tracks/<track>/redirected/<traj>/mask.mp4.
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


def read_mask_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray > 127)
    cap.release()
    if not frames:
        return np.zeros((0, 1, 1), dtype=bool)
    return np.stack(frames, axis=0)


def resize_mask_to(pred: np.ndarray, h: int, w: int) -> np.ndarray:
    if pred.shape[1] == h and pred.shape[2] == w:
        return pred
    out = np.zeros((pred.shape[0], h, w), dtype=bool)
    for i in range(pred.shape[0]):
        r = cv2.resize(pred[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        out[i] = r.astype(bool)
    return out


def gt_path_for_key(key: str, dataset_root: Path) -> Path:
    track, traj = key.split("__", 1)
    traj_root = dataset_root / "tracks" / track / "redirected" / traj
    return traj_root / "mask.mp4"


def evaluate_case(pred_path: str, gt_path: str, area_thresh: int):
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
        rows.append({
            "frame": i,
            "gt_has": int(gt_has[i]),
            "pred_has": int(pred_has[i]),
            "present": int(bool(gt_has[i] and pred_has[i])),
        })
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
    dataset_root = Path(args.dataset_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob(args.glob))
    print(f"[detection] pred_dir={pred_dir}  n={len(pred_files)}", flush=True)

    per_frame_path = out_dir / "per_frame.csv"
    per_video_path = out_dir / "per_video.csv"

    all_frames, per_video_rows = [], []
    t0 = time.time()
    for idx, p in enumerate(pred_files, 1):
        key = p.stem
        gt = gt_path_for_key(key, dataset_root)
        if not gt.exists():
            print(f"[skip] {key}: no GT"); continue
        rows = evaluate_case(str(p), str(gt), args.area_thresh)
        if rows is None:
            continue
        n_gt = sum(r["gt_has"] for r in rows)
        n_pred = sum(r["pred_has"] for r in rows)
        n_both = sum(r["present"] for r in rows)
        presence = n_both / n_gt if n_gt else 0.0
        for r in rows:
            r["case"] = key
            all_frames.append(r)
        per_video_rows.append({
            "case": key, "n_frames": len(rows),
            "n_gt_has": n_gt, "n_pred_has": n_pred, "n_both": n_both,
            "presence": presence,
        })
        print(f"[{idx:3d}/{len(pred_files)}] {key[:58]:58s} "
              f"presence={presence:.3f} ({time.time()-t0:.1f}s)", flush=True)

    with open(per_frame_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "frame", "gt_has", "pred_has", "present"])
        w.writeheader()
        w.writerows(all_frames)

    with open(per_video_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "n_frames", "n_gt_has", "n_pred_has",
                                          "n_both", "presence"])
        w.writeheader()
        for r in per_video_rows:
            r["presence"] = f"{r['presence']:.4f}"
            w.writerow(r)

    presences = [float(r["presence"]) for r in per_video_rows]
    summary = {
        "metric": "detection",
        "n_videos": len(per_video_rows),
        "presence_mean": float(np.mean(presences)) if presences else 0,
        "presence_std": float(np.std(presences)) if presences else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] {per_frame_path}  |  {per_video_path}  |  summary.json")


if __name__ == "__main__":
    sys.exit(main())
