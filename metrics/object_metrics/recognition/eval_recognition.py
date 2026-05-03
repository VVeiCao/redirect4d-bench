"""Optional VLM recognition judge for object fidelity.

This script reads method-generated videos and the corresponding propagated
masks, crops the visible object, and asks a VLM whether the object is
structurally plausible. The resulting cache can be passed to
`object_metrics/overall/build_cascade.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eval_utils import build_crop, build_grid, load_vlm, read_mask, read_video, ask as qwen_ask
from prompts_v12 import PROMPTS, class_label, parse_answer


ROOT = Path(__file__).resolve().parents[3]


def load_cases(path: Path) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "track" in row and "trajectory" in row:
            cases.append((row["track"], row["trajectory"]))
        elif "case" in row:
            track, trajectory = row["case"].split("__", 1)
            cases.append((track, trajectory))
    return cases


def infer_methods(video_root: Path) -> list[str]:
    return sorted(p.name for p in video_root.iterdir() if (p / "videos").is_dir())


def run_case(
    track: str,
    trajectory: str,
    *,
    methods: list[str],
    video_root: Path,
    mask_root: Path,
    n_frames: int,
    model,
    proc,
    save_dir: Path,
    prompt_tpl,
    version: str,
    crop_mode: str,
    model_id: str,
) -> dict:
    key = f"{track}__{trajectory}"
    cls = class_label(track)
    prompt = prompt_tpl.format(cls=cls)
    print(f"\n[qwen-{version}] === {key} class={cls!r} ===", flush=True)

    out = {
        "case": key,
        "class": cls,
        "model": model_id,
        "prompt_version": f"qwen-{version}",
        "methods": {},
    }
    crops_dir = save_dir / "_crops" / key
    crops_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        video = video_root / method / "videos" / f"{key}.mp4"
        mask = mask_root / method / f"{key}.mp4"
        if not video.exists() or not mask.exists():
            print(f"  [skip] {method}: missing video or mask", flush=True)
            continue

        frames = read_video(video)
        masks = read_mask(mask)
        n = min(len(frames), len(masks))
        if n < 2:
            continue
        idxs = np.linspace(0, n - 1, n_frames).astype(int).tolist()

        results = []
        for i in idxs:
            frame, mask_frame = frames[i], masks[i]
            if frame.shape[:2] != mask_frame.shape:
                mask_frame = cv2.resize(
                    mask_frame,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            crop = build_crop(frame, mask_frame, mode=crop_mode)
            if crop is None or crop.size == 0:
                continue
            pil = Image.fromarray(crop, mode="RGB")
            crop_path = crops_dir / f"{method}_f{i:03d}.png"
            pil.save(crop_path)
            try:
                resp = qwen_ask(model, proc, pil, prompt, max_new=80, fewshot=None)
                defect, what = parse_answer(resp)
            except Exception as exc:
                resp = f"[ERR] {exc}"
                defect, what = None, str(exc)[:80]
            results.append(
                {
                    "frame": int(i),
                    "defect": defect,
                    "what": what,
                    "raw": resp,
                    "crop": str(crop_path),
                }
            )
            label = "DEFECT" if defect is True else ("OK" if defect is False else "NA")
            print(f"  {method:>16} f{i:03d} {label:7s} {what[:60]}", flush=True)

        if results:
            parsed = [r for r in results if r["defect"] is not None]
            n_ok = sum(1 for r in parsed if r["defect"] is False)
            n_def = sum(1 for r in parsed if r["defect"] is True)
            out["methods"][method] = {
                "frames": results,
                "n_sampled": len(results),
                "n_parsed": len(parsed),
                "n_ok": n_ok,
                "n_defect": n_def,
                "plausible_rate": n_ok / len(parsed) if parsed else None,
            }

    (save_dir / f"scores_{key}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    build_grid(out, save_dir / f"grid_{key}.png")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v12", choices=list(PROMPTS.keys()))
    ap.add_argument("--model-id", "--model_id", dest="model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--crop-mode", "--crop_mode", dest="crop_mode", default="bbox", choices=["bbox", "masked_black", "masked_gray", "masked_white"])
    ap.add_argument("--load-in-4bit", "--load_in_4bit", dest="load_in_4bit", action="store_true")
    ap.add_argument("--n-frames", "--n_frames", dest="n_frames", type=int, default=45)
    ap.add_argument("--case-slice", "--case_slice", dest="case_slice", default=None)
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--video-root", type=Path, default=ROOT / "predictions")
    ap.add_argument("--mask-root", type=Path, default=ROOT / "outputs" / "object_metric_masks" / "seeded_from_pt_box")
    ap.add_argument("--cases", type=Path, default=ROOT / "data" / "redirect4d_bench" / "cases.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "object_recognition_cache" / "qwen32b_v12_full")
    args = ap.parse_args()

    methods = args.methods or infer_methods(args.video_root)
    cases = load_cases(args.cases)
    if args.case_slice:
        start, end = args.case_slice.split(":")
        cases = cases[int(start):int(end)]
        print(f"[shard] case_slice={args.case_slice} running {len(cases)} cases", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, proc = load_vlm(args.model_id, load_in_4bit=args.load_in_4bit)
    prompt_tpl = PROMPTS[args.version]

    rows = []
    t0 = time.time()
    for track, trajectory in cases:
        result = run_case(
            track,
            trajectory,
            methods=methods,
            video_root=args.video_root,
            mask_root=args.mask_root,
            n_frames=args.n_frames,
            model=model,
            proc=proc,
            save_dir=args.out_dir,
            prompt_tpl=prompt_tpl,
            version=args.version,
            crop_mode=args.crop_mode,
            model_id=args.model_id,
        )
        for method, mr in result["methods"].items():
            rows.append(
                [
                    result["case"],
                    method,
                    mr["n_sampled"],
                    mr["n_parsed"],
                    mr["n_ok"],
                    mr["n_defect"],
                    f"{mr['plausible_rate']:.3f}" if mr["plausible_rate"] is not None else "",
                ]
            )

    suffix = f"_{args.case_slice.replace(':', '-')}" if args.case_slice else ""
    with (args.out_dir / f"summary{suffix}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "method", "n_sampled", "n_parsed", "n_ok", "n_defect", "plausible_rate"])
        writer.writerows(rows)
    print(f"[ok] {len(cases)} cases in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
