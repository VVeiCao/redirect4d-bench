"""Cascade aggregator - combines detection, optional recognition, and localization.

Reads per-frame CSVs from detection and localization. If a VLM recognition
cache is supplied, it also joins recognition verdicts on (case, method, frame).
Without a recognition cache, present frames are treated as plausible so the
aggregation is fully mask/IoU based.

  **Object fidelity** (does the subject come out right?)
    D = n_present / n_gt       Detection      (mask exists on both GT+pred)
    R = n_plausible / n_gt     Recognition    (present AND V12 says 'plausible')

  **Localization** (when it does come out, is it in the right place?)
    cond_mask_iou = mean MaskIoU over (present ∧ plausible) frames
    cond_bbox_iou = mean BBoxIoU over (present ∧ plausible) frames
    r_mask_iou    = R · cond_mask_iou   (= Σ MaskIoU / n_gt, missing/broken=0)
    r_bbox_iou    = R · cond_bbox_iou   (= Σ BBoxIoU / n_gt, missing/broken=0)

r_{mask,bbox}_iou is the Panoptic-Quality-style unconditional score — it
penalizes methods that abstain on frames where the pseudo-GT object is
visible, because missing and broken frames count as 0 in that average.
Frames without a pseudo-GT object mask are not included in the denominator.
cond_* reveals the "quality ceiling when it works".

Inputs:
  detection/<method>/per_frame.csv     (case, frame, gt_has, pred_has, present)
  localization/<method>/per_frame.csv  (case, frame, ..., mask_iou, bbox_iou)
  optional recognition cache: scores_<case>.json

Outputs (under --out_root, default ../results):
  leaderboard.csv               one row per method
  <method>/summary.json
  <method>/per_video.csv      rows x metric columns plus frame counts
  <method>/per_frame.csv      joined per-frame state + IoUs
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_detection(path):
    out = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["case"]][int(r["frame"])] = (
                int(r["gt_has"]), int(r["pred_has"]), int(r["present"])
            )
    return out


def load_localization(path):
    out = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            mi = float(r["mask_iou"]) if r["mask_iou"] else None
            bi = float(r["bbox_iou"]) if r["bbox_iou"] else None
            out[r["case"]][int(r["frame"])] = (mi, bi)
    return out


def load_recognition(cache_dir, methods):
    out = {m: defaultdict(dict) for m in methods}
    if not cache_dir:
        return out
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return out
    for p in sorted(Path(cache_dir).glob("scores_*.json")):
        d = json.loads(p.read_text())
        case = d["case"]
        for m, mr in d["methods"].items():
            if m not in out: continue
            for r in mr["frames"]:
                dft = r.get("defect")
                if dft is True: v = 1
                elif dft is False: v = 0
                else: v = None
                out[m][case][int(r["frame"])] = v
    return out


def process_method(method, det_dir, loc_dir, rec_by_method, out_dir, recognition_enabled):
    det = load_detection(det_dir / "per_frame.csv")
    loc = load_localization(loc_dir / "per_frame.csv")
    rec = rec_by_method[method]

    out_dir.mkdir(parents=True, exist_ok=True)

    per_frame_rows = []
    per_video_rows = []

    cases = sorted(det.keys())
    for case in cases:
        case_frames = det[case]
        n_frames = len(case_frames)

        n_gt_has = 0
        n_present = 0
        n_plausible = 0
        mask_ious_plausible = []
        bbox_ious_plausible = []

        for frame, (gt_has, pred_has, present) in sorted(case_frames.items()):
            mask_iou, bbox_iou = loc.get(case, {}).get(frame, (None, None))
            defect = rec.get(case, {}).get(frame)  # 0=OK, 1=DEF, None=missing/parse-fail

            if gt_has:
                n_gt_has += 1
            if not recognition_enabled and present:
                defect = 0
            is_plausible = bool(present and defect == 0)
            if present:
                n_present += 1
            if is_plausible:
                n_plausible += 1
                if mask_iou is not None:
                    mask_ious_plausible.append(mask_iou)
                if bbox_iou is not None:
                    bbox_ious_plausible.append(bbox_iou)

            if not present:
                state = "MISSING"
            elif defect == 1:
                state = "BROKEN"
            elif defect == 0:
                state = "ALIGNED" if (bbox_iou is not None and bbox_iou >= 0.5) else "MISLOCATED"
            else:
                state = "NOJUDGE"

            per_frame_rows.append({
                "case": case, "frame": frame, "method": method,
                "present": present,
                "defect": defect if defect is not None else "",
                "plausible": int(is_plausible),
                "mask_iou": f"{mask_iou:.4f}" if mask_iou is not None else "",
                "bbox_iou": f"{bbox_iou:.4f}" if bbox_iou is not None else "",
                "state": state,
            })

        denom = n_gt_has if n_gt_has else n_frames
        D = n_present / denom if denom else 0.0
        R = n_plausible / denom if denom else 0.0
        cond_mask = float(np.mean(mask_ious_plausible)) if mask_ious_plausible else 0.0
        cond_bbox = float(np.mean(bbox_ious_plausible)) if bbox_ious_plausible else 0.0
        r_mask = R * cond_mask
        r_bbox = R * cond_bbox

        per_video_rows.append({
            "case": case,
            "n_frames": n_frames,
            "n_gt_has": n_gt_has,
            "n_present": n_present,
            "n_plausible": n_plausible,
            "detection": f"{D:.4f}",
            "recognition": f"{R:.4f}",
            "cond_mask_iou": f"{cond_mask:.4f}",
            "cond_bbox_iou": f"{cond_bbox:.4f}",
            "r_mask_iou": f"{r_mask:.4f}",
            "r_bbox_iou": f"{r_bbox:.4f}",
        })

    with open(out_dir / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "frame", "method", "present", "defect",
                                          "plausible", "mask_iou", "bbox_iou", "state"])
        w.writeheader(); w.writerows(per_frame_rows)

    with open(out_dir / "per_video.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "n_frames", "n_gt_has", "n_present", "n_plausible",
                                          "detection", "recognition",
                                          "cond_mask_iou", "cond_bbox_iou",
                                          "r_mask_iou", "r_bbox_iou"])
        w.writeheader(); w.writerows(per_video_rows)

    Ds = [float(r["detection"]) for r in per_video_rows]
    Rs = [float(r["recognition"]) for r in per_video_rows]
    cMs = [float(r["cond_mask_iou"]) for r in per_video_rows]
    cBs = [float(r["cond_bbox_iou"]) for r in per_video_rows]
    rMs = [float(r["r_mask_iou"]) for r in per_video_rows]
    rBs = [float(r["r_bbox_iou"]) for r in per_video_rows]

    summary = {
        "method": method,
        "n_videos": len(per_video_rows),
        "n_frames": sum(r["n_frames"] for r in per_video_rows),
        "n_gt_frames": sum(r["n_gt_has"] for r in per_video_rows),
        "detection_mean":     float(np.mean(Ds)),
        "recognition_mean":   float(np.mean(Rs)),
        "cond_mask_iou_mean": float(np.mean(cMs)),
        "cond_bbox_iou_mean": float(np.mean(cBs)),
        "r_mask_iou_mean":    float(np.mean(rMs)),
        "r_bbox_iou_mean":    float(np.mean(rBs)),
        "recognition_enabled": recognition_enabled,
        "recipe": {
            "D": "fraction of frames where both GT and pred have a mask",
            "D_denominator": "frames where the pseudo-GT object mask exists; GT-empty frames are ignored",
            "R": "fraction of GT-visible frames that are present and, when a recognition cache is supplied, VLM-plausible",
            "cond_mask_iou": "mean MaskIoU over frames that are present AND plausible (strict conditional)",
            "cond_bbox_iou": "mean BBoxIoU over frames that are present AND plausible",
            "r_mask_iou": "R * cond_mask_iou - unconditional mean MaskIoU over GT-visible frames with missing/broken counted as 0",
            "r_bbox_iou": "R * cond_bbox_iou - unconditional mean BBoxIoU over GT-visible frames with missing/broken counted as 0",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[{method}]  "
          f"D={summary['detection_mean']*100:5.1f}%  "
          f"R={summary['recognition_mean']*100:5.1f}%  "
          f"cMaskIoU={summary['cond_mask_iou_mean']*100:5.1f}%  "
          f"cBBoxIoU={summary['cond_bbox_iou_mean']*100:5.1f}%  "
          f"R*MaskIoU={summary['r_mask_iou_mean']*100:5.1f}%  "
          f"R*BBoxIoU={summary['r_bbox_iou_mean']*100:5.1f}%  "
          f"-> {out_dir}")
    return summary


def write_leaderboard(summaries, path):
    # Sort by the combined bbox score as the default headline.
    summaries = sorted(summaries, key=lambda s: -s["r_bbox_iou_mean"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "method", "n_videos", "n_frames",
                    "detection", "recognition",
                    "cond_mask_iou", "cond_bbox_iou",
                    "r_mask_iou", "r_bbox_iou"])
        for i, s in enumerate(summaries, 1):
            w.writerow([i, s["method"], s["n_videos"], s["n_frames"],
                        f"{s['detection_mean']:.4f}",
                        f"{s['recognition_mean']:.4f}",
                        f"{s['cond_mask_iou_mean']:.4f}",
                        f"{s['cond_bbox_iou_mean']:.4f}",
                        f"{s['r_mask_iou_mean']:.4f}",
                        f"{s['r_bbox_iou_mean']:.4f}"])
    print(f"\n[leaderboard] {path}")


def main():
    here = Path(__file__).parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection_root", default=str(here / "detection"))
    ap.add_argument("--localization_root", default=str(here / "localization"))
    ap.add_argument("--recognition_cache", default="")
    ap.add_argument("--out_root", default=str(here / "results"))
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()

    det_root = Path(args.detection_root)
    loc_root = Path(args.localization_root)
    methods = args.methods or sorted(
        p.name for p in det_root.iterdir() if (p / "per_frame.csv").exists()
    )
    rec_cache = Path(args.recognition_cache) if args.recognition_cache else None
    recognition_enabled = bool(rec_cache and rec_cache.exists())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if recognition_enabled:
        print(f"[cascade] reading recognition cache from {rec_cache}")
    else:
        print("[cascade] no recognition cache supplied; using mask presence as plausibility")
    rec_by_method = load_recognition(rec_cache, methods)
    print(f"  loaded recognition: {sum(len(rec_by_method[m]) for m in methods)} case-method pairs")

    summaries = []
    for method in methods:
        det_dir = det_root / method
        loc_dir = loc_root / method
        if not (det_dir / "per_frame.csv").exists():
            print(f"[skip] {method}: no detection data at {det_dir}")
            continue
        if not (loc_dir / "per_frame.csv").exists():
            print(f"[skip] {method}: no localization data at {loc_dir}")
            continue
        out_dir = out_root / method
        summaries.append(
            process_method(method, det_dir, loc_dir, rec_by_method, out_dir, recognition_enabled)
        )

    write_leaderboard(summaries, out_root / "leaderboard.csv")


if __name__ == "__main__":
    main()
