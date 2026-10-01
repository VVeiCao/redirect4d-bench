"""Merge per-case scores_*.json into a flat per-frame CSV for easy analysis.

Each row = (case, class, method, frame, defect, what, raw_snippet, crop_path).
"""
import argparse
import csv
import json
from pathlib import Path


def merge(out_dir, out_csv):
    rows = []
    for p in sorted(Path(out_dir).glob("scores_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        case = d["case"]
        cls = d.get("class", "")
        for m, mr in d["methods"].items():
            for fr in mr["frames"]:
                rows.append(dict(
                    case=case,
                    cls=cls,
                    method=m,
                    frame=fr["frame"],
                    defect=("" if fr["defect"] is None else int(bool(fr["defect"]))),
                    what=(fr.get("what") or "").replace("\n", " ").strip(),
                    raw=(fr.get("raw") or "").replace("\n", " ").strip()[:300],
                    crop=fr.get("crop", ""),
                ))
    rows.sort(key=lambda r: (r["case"], r["method"], r["frame"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "cls", "method", "frame",
                                          "defect", "what", "raw", "crop"])
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out_csv}  ({len(rows)} rows from "
          f"{len(set(r['case'] for r in rows))} cases, "
          f"{len(set(r['method'] for r in rows))} methods)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="Directory with scores_*.json files")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <dir>/all_frames.csv)")
    args = ap.parse_args()
    d = Path(args.dir)
    out = Path(args.out) if args.out else d / "all_frames.csv"
    merge(d, out)
