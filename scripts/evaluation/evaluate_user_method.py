#!/usr/bin/env python3
"""Evaluate user-generated videos with Redirect4D-Bench metrics.

User prediction layout:

    <pred-root>/<method>/videos/<track>_<trajectory>.mp4

or pass a single video folder directly:

    --video-dir <path-to-videos>

This script stages predictions into the common layout expected by
`run_final_metrics_case.py`, extracts generated-video masks when needed, and
evaluates each requested case.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import resolve_python  # noqa: E402


def public_case_name(case: str) -> str:
    return case.replace("__", "_", 1)


def load_cases(path: Path, limit: int | None = None) -> list[str]:
    cases: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("{"):
            cases.append(json.loads(line)["case"])
        else:
            cases.append(line.strip())
        if limit and len(cases) >= limit:
            break
    return cases


def load_case_name_maps(dataset_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    cases_path = dataset_root / "cases.jsonl"
    if not cases_path.exists():
        return {}, {}
    internal_to_public: dict[str, str] = {}
    public_to_internal: dict[str, str] = {}
    for case in load_cases(cases_path):
        public = public_case_name(case)
        internal_to_public[case] = public
        public_to_internal[public] = case
        public_to_internal[case] = case
    return internal_to_public, public_to_internal


def load_cases_from_video_dir(
    video_dir: Path,
    public_to_internal: dict[str, str],
    limit: int | None = None,
) -> list[str]:
    cases = []
    unknown = []
    for path in sorted(video_dir.glob("*.mp4")):
        case = public_to_internal.get(path.stem)
        if case is None:
            unknown.append(path.name)
            continue
        cases.append(case)
    if limit:
        cases = cases[:limit]
    if not cases:
        raise FileNotFoundError(f"no .mp4 files found in {video_dir}")
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(
            f"could not match video file name(s) to dataset cases: {preview}. "
            "Use '<track>_<trajectory>.mp4', for example "
            "bear_NnAlfavy2us_003_001_seq1_yaw_120_pitch_0_roll_0_scale_1.mp4."
        )
    return cases


def link(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        raise FileExistsError(
            f"{dst} already exists in the evaluation output directory. "
            "Each run writes to a timestamped output folder by default; "
            "delete this output folder or pass a new --out-dir if you want to rerun it. "
            "Your input prediction videos are not modified."
        )
    dst.symlink_to(src.resolve())


def run(cmd: list[str], *, cwd: Path = ROOT, dry_run: bool = False) -> None:
    if dry_run:
        print("$ " + " ".join(str(x) for x in cmd))
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_changed(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and file_sha256(src) == file_sha256(dst):
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def infer_methods_from_pred_root(pred_root: Path) -> list[str]:
    methods = []
    for path in sorted(pred_root.iterdir() if pred_root.exists() else []):
        if not path.is_dir():
            continue
        videos = path / "videos"
        if videos.is_dir() and any(videos.glob("*.mp4")):
            methods.append(path.name)
    if not methods:
        raise FileNotFoundError(f"no method folders with videos/*.mp4 found under {pred_root}")
    return methods


def default_out_dir(methods: list[str]) -> Path:
    method_name = methods[0] if len(methods) == 1 else "_".join(methods)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("outputs") / method_name
    out_dir = root / timestamp
    suffix = 2
    while out_dir.exists():
        out_dir = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return out_dir


def write_prompt_file(dataset_root: Path, cases: list[str], out: Path) -> None:
    prompts: dict[str, str] = {}
    for case in cases:
        if "__" not in case:
            raise ValueError(f"internal case key must contain '<track>__<trajectory>': {case}")
        track, trajectory = case.split("__", 1)
        prompt_path = dataset_root / "tracks" / track / "redirected" / trajectory / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(prompt_path)
        prompts[case] = prompt_path.read_text().strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n")


def default_method_from_video_dir(video_dir: Path) -> str:
    if video_dir.name.lower() == "videos" and video_dir.parent.name:
        return video_dir.parent.name
    return video_dir.name


def method_root(args: argparse.Namespace, method: str) -> Path:
    if args.video_dir is not None:
        if args.video_dir.name.lower() == "videos":
            return args.video_dir.parent
        return args.video_dir
    return args.pred_root / method


def public_mask_dir(args: argparse.Namespace, method: str) -> Path:
    return method_root(args, method) / "mask"


def track_from_case(case: str) -> str:
    if "__" not in case:
        raise ValueError(f"internal case key must contain '<track>__<trajectory>': {case}")
    return case.split("__", 1)[0]


def candidate_video_names(case: str, internal_to_public: dict[str, str]) -> list[str]:
    names = [f"{internal_to_public.get(case, public_case_name(case))}.mp4"]
    legacy = f"{case}.mp4"
    if legacy not in names:
        names.append(legacy)
    return names


def prediction_video(args: argparse.Namespace, method: str, case: str) -> Path:
    roots = [args.video_dir] if args.video_dir is not None else [args.pred_root / method / "videos"]
    for root in roots:
        for name in candidate_video_names(case, args.internal_to_public):
            path = root / name
            if path.exists():
                return path
    return roots[0] / candidate_video_names(case, args.internal_to_public)[0]


def stage_predictions(args: argparse.Namespace, cases: list[str], staged: Path) -> None:
    for case in cases:
        for method in args.methods:
            src = prediction_video(args, method, case)
            if not args.dry_run:
                link(src, staged / method / "videos" / f"{case}.mp4")


def extract_object_masks(args: argparse.Namespace, staged: Path) -> Path:
    out_base = args.out_dir / "object_metric_masks"
    cmd = [
        args.object_python,
        str(ROOT / "scripts" / "metrics" / "extract_propagated_masks_sam3.py"),
        "--metrics-root",
        str(args.metrics_root),
        "--pred-root",
        str(staged),
        "--dataset-root",
        str(args.dataset_root),
        "--out-base",
        str(out_base),
        "--methods",
        *args.methods,
    ]
    run(cmd, dry_run=args.dry_run)
    return out_base / "seeded_from_pt_box"


def mirror_object_masks(args: argparse.Namespace, mask_root: Path, cases: list[str]) -> None:
    for method in args.methods:
        dst_dir = public_mask_dir(args, method)
        for case in cases:
            src = mask_root / method / f"{case}.mp4"
            public_name = args.internal_to_public.get(case, public_case_name(case))
            dst = dst_dir / f"{public_name}.mp4"
            if args.dry_run:
                print(f"# generated-video mask: {src} -> {dst}")
            else:
                copy_if_changed(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/redirect4d_bench"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video-dir", type=Path, help="Folder containing <track>_<trajectory>.mp4 files.")
    inputs.add_argument("--pred-root", type=Path, help="Root containing <method>/videos/<case>.mp4.")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--cases", type=Path, default=Path("data/redirect4d_bench/cases.jsonl"))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("metrics"),
        help="Path to this repository's metrics directory.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["object", "camera_pose"],
        choices=["object", "camera_pose"],
    )
    parser.add_argument(
        "--object-python",
        help=(
            "Python executable for SAM3 generated-video mask extraction. "
            "Defaults to OBJECT_METRICS_PY, then conda env redirect4d-sam3, then current Python."
        ),
    )
    parser.add_argument(
        "--reconstruction-python",
        help=(
            "Python executable for camera-pose reconstruction. "
            "Defaults to R4D_RECON_PY, then conda env redirect4d-recon, then current Python."
        ),
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        help=(
            "Optional precomputed generated-video mask root. If omitted for "
            "object metrics, masks are extracted from the submitted RGB videos."
        ),
    )
    parser.add_argument("--recognition-cache", type=Path)
    parser.add_argument("--camera-modes", default="noopt")
    parser.add_argument("--metric-seed", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.video_dir is not None:
        if args.methods is None:
            args.methods = [default_method_from_video_dir(args.video_dir)]
        elif len(args.methods) != 1:
            parser.error("--video-dir supports one --method")
    elif args.methods is None:
        args.methods = infer_methods_from_pred_root(args.pred_root)
    if args.out_dir is None:
        args.out_dir = default_out_dir(args.methods)
    return args


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    if args.pred_root is not None:
        args.pred_root = args.pred_root.resolve()
    if args.video_dir is not None:
        args.video_dir = args.video_dir.resolve()
    args.cases = args.cases.resolve()
    args.metrics_root = args.metrics_root.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.mask_root:
        args.mask_root = args.mask_root.resolve()
    if args.recognition_cache:
        args.recognition_cache = args.recognition_cache.resolve()
    args.object_python = resolve_python(
        args.object_python,
        env_var="OBJECT_METRICS_PY",
        conda_env="redirect4d-sam3",
        fallback=sys.executable,
    )
    args.reconstruction_python = resolve_python(
        args.reconstruction_python,
        env_var="R4D_RECON_PY",
        conda_env="redirect4d-recon",
        fallback=sys.executable,
    )
    args.internal_to_public, args.public_to_internal = load_case_name_maps(args.dataset_root)
    if args.case:
        cases = [args.public_to_internal.get(case, case) for case in args.case]
    elif args.video_dir is not None:
        cases = load_cases_from_video_dir(args.video_dir, args.public_to_internal, args.limit)
    else:
        cases = load_cases(args.cases, args.limit)
    staged = args.out_dir / "_staged_predictions"
    stage_predictions(args, cases, staged)
    if "object" in args.suites and args.mask_root is None:
        print(f"[env] object metrics Python: {args.object_python}")
        args.mask_root = extract_object_masks(args, staged)
        mirror_object_masks(args, args.mask_root, cases)
    if "camera_pose" in args.suites:
        print(f"[env] camera-pose Python: {args.reconstruction_python}")

    for case in cases:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluation" / "run_final_metrics_case.py"),
            "--case",
            case,
            "--dataset-root",
            str(args.dataset_root),
            "--prediction-root",
            str(staged),
            "--metrics-root",
            str(args.metrics_root),
            "--out-dir",
            str(args.out_dir / "metrics"),
            "--methods",
            *args.methods,
            "--suites",
            *args.suites,
            "--object-python",
            args.object_python,
            "--reconstruction-python",
            args.reconstruction_python,
            "--camera-modes",
            args.camera_modes,
            "--metric-seed",
            args.metric_seed,
        ]
        if args.mask_root:
            cmd.extend(["--mask-root", str(args.mask_root)])
        if args.recognition_cache:
            cmd.extend(["--recognition-cache", str(args.recognition_cache)])

        if args.dry_run:
            run(cmd, dry_run=True)
        else:
            public_case = args.internal_to_public.get(case, case)
            print(f"[eval] {public_case} -> {args.out_dir / 'metrics'}")
            run(cmd)


if __name__ == "__main__":
    main()
