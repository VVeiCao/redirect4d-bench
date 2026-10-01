#!/usr/bin/env python3
"""Run Redirect4D-Bench metrics for one staged case.

This is the low-level wrapper used by evaluate_user_method.py. It intentionally
only runs the Redirect4D-Bench metrics shipped in this repository:

  - Object fidelity / localization from generated-video masks.
  - Camera-pose accuracy from VIPE-LyRA reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import resolve_python  # noqa: E402

VOLATILE_METRIC_KEYS = {"pred_video", "wall_time_sec"}


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def link_or_keep(src: Path, dst: Path) -> None:
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
            "Use a new --out-dir or delete the existing output folder before rerunning. "
            "Your input prediction videos are not modified."
        )
    dst.symlink_to(src.resolve())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_metric_json(obj):
    if isinstance(obj, dict):
        return {
            key: normalize_metric_json(value)
            for key, value in sorted(obj.items())
            if key not in VOLATILE_METRIC_KEYS
        }
    if isinstance(obj, list):
        return [normalize_metric_json(value) for value in obj]
    return obj


def write_metric_scores_copy(summary_path: Path) -> None:
    if not summary_path.exists():
        return
    payload = normalize_metric_json(json.loads(summary_path.read_text()))
    scores_path = summary_path.with_name("summary_scores.json")
    scores_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def split_case(case: str) -> tuple[str, str]:
    if "__" not in case:
        raise ValueError(f"case must be '<track>__<trajectory>': {case}")
    return case.split("__", 1)


def target_mask_path(dataset_root: Path, case: str) -> Path:
    track, trajectory = split_case(case)
    traj_root = dataset_root / "tracks" / track / "redirected" / trajectory
    path = traj_root / "mask.mp4"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"target pseudo-GT mask not found; expected {path}"
    )


def object_gt_root(args: argparse.Namespace) -> Path:
    """Stage the target pseudo-GT mask expected by the object metric scripts."""

    track, trajectory = split_case(args.case)
    src = target_mask_path(args.dataset_root, args.case)
    root = args.out_dir / "_object_gt_compat"
    compat_dir = root / "tracks" / track / "redirected" / trajectory
    link_or_keep(src, compat_dir / "mask.mp4")
    return root


def stage_inputs(args: argparse.Namespace) -> None:
    video_name = f"{args.case}.mp4"
    manifest = {"case": args.case, "videos": {}}
    if args.vipe_source_root:
        manifest["vipe_source_root"] = str(args.vipe_source_root)

    for method in args.methods:
        src = args.prediction_root / method / "videos" / video_name
        dst = args.out_dir / "inputs" / method / "videos" / video_name
        link_or_keep(src, dst)
        manifest["videos"][method] = {
            "source": str(src.resolve()),
            "staged": str(dst),
            "sha256": file_sha256(src),
            "bytes": src.stat().st_size,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "inputs_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def run_object(args: argparse.Namespace) -> None:
    detect = args.metrics_root / "object_metrics" / "detection" / "eval_detection.py"
    locate = args.metrics_root / "object_metrics" / "localization" / "eval_localization.py"
    cascade = args.metrics_root / "object_metrics" / "overall" / "build_cascade.py"
    object_dataset_root = object_gt_root(args)

    for method in args.methods:
        run(
            [
                args.object_python,
                str(detect),
                "--pred_dir",
                str(args.mask_root / method),
                "--out_dir",
                str(args.out_dir / "object_metrics" / "detection" / method),
                "--dataset_root",
                str(object_dataset_root),
                "--glob",
                f"{args.case}.mp4",
            ],
            cwd=detect.parent,
        )
        run(
            [
                args.object_python,
                str(locate),
                "--pred_dir",
                str(args.mask_root / method),
                "--out_dir",
                str(args.out_dir / "object_metrics" / "localization" / method),
                "--dataset_root",
                str(object_dataset_root),
                "--glob",
                f"{args.case}.mp4",
            ],
            cwd=locate.parent,
        )

    cmd = [
        args.object_python,
        str(cascade),
        "--detection_root",
        str(args.out_dir / "object_metrics" / "detection"),
        "--localization_root",
        str(args.out_dir / "object_metrics" / "localization"),
        "--out_root",
        str(args.out_dir / "object_metrics" / "cascade"),
        "--methods",
        *args.methods,
    ]
    if args.recognition_cache:
        cmd.extend(["--recognition_cache", str(args.recognition_cache)])
    run(cmd, cwd=cascade.parent)


def run_camera_pose(args: argparse.Namespace) -> None:
    mode_to_script = {
        "noopt": args.metrics_root / "camera_pose_metrics" / "eval_camera_pose.py",
        "opt": args.metrics_root / "camera_pose_metrics" / "eval_camera_pose.py",
        "gt_intr": args.metrics_root / "camera_pose_metrics" / "eval_camera_pose_gt.py",
    }
    for mode in args.camera_modes.split(","):
        mode = mode.strip()
        if not mode:
            continue
        if mode not in mode_to_script:
            raise ValueError(f"unknown camera mode: {mode}")
        script = mode_to_script[mode]
        for method in args.methods:
            env = os.environ.copy()
            if mode == "noopt":
                env["VIPE_NO_OPT_INTR"] = "1"
            else:
                env.pop("VIPE_NO_OPT_INTR", None)
            env.setdefault("REDIRECT4D_METRIC_SEED", args.metric_seed)
            env.setdefault("PYTHONHASHSEED", args.metric_seed)
            if args.vipe_source_root:
                env.setdefault("REDIRECT4D_VIPE_ROOT", str(args.vipe_source_root))
            run(
                [
                    args.reconstruction_python,
                    str(Path(__file__).with_name("run_seeded_metric.py")),
                    str(script),
                    "--pred_video",
                    str(args.out_dir / "inputs" / method / "videos" / f"{args.case}.mp4"),
                    "--dataset_root",
                    str(args.dataset_root),
                    "--output_dir",
                    str(args.out_dir / "camera_pose_metrics" / mode / method),
                    "--keep_vipe_outputs",
                ],
                cwd=script.parent,
                env=env,
            )
            write_metric_scores_copy(
                args.out_dir / "camera_pose_metrics" / mode / method / "summary.json"
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="Internal case key: <track>__<trajectory>")
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--prediction-root", required=True, type=Path)
    ap.add_argument("--metrics-root", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/final_metrics_case"))
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument(
        "--suites",
        nargs="+",
        default=["object", "camera_pose"],
        choices=["object", "camera_pose"],
    )
    ap.add_argument("--mask-root", type=Path)
    ap.add_argument(
        "--recognition-cache",
        type=Path,
        help="Optional VLM recognition cache. If omitted, object aggregation uses mask presence/IoU only.",
    )
    ap.add_argument("--camera-modes", default="noopt")
    ap.add_argument("--object-python")
    ap.add_argument("--reconstruction-python")
    ap.add_argument("--metric-seed", default=os.environ.get("REDIRECT4D_METRIC_SEED", "0"))
    ap.add_argument(
        "--vipe-source-root",
        type=Path,
        default=Path(os.environ.get("REDIRECT4D_VIPE_ROOT", ROOT / "third_party" / "vipe")),
        help="Pinned VIPE source tree for Camera Accuracy. Defaults to third_party/vipe.",
    )
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()

    args.dataset_root = args.dataset_root.resolve()
    args.prediction_root = args.prediction_root.resolve()
    args.metrics_root = args.metrics_root.resolve()
    args.out_dir = (args.out_dir / args.case).resolve()
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
    if args.vipe_source_root:
        args.vipe_source_root = args.vipe_source_root.resolve()
        if not (args.vipe_source_root / "vipe").is_dir():
            raise FileNotFoundError(
                f"VIPE source tree not found at {args.vipe_source_root}. "
                "Run `git submodule update --init --recursive third_party/vipe`."
            )
    if "object" in args.suites and not args.mask_root:
        raise ValueError("--mask-root is required for object metrics")
    return args


def main() -> None:
    args = parse_args()
    stage_inputs(args)
    if args.stage_only:
        print(f"[ok] staged inputs under {args.out_dir}")
        return
    if "object" in args.suites:
        run_object(args)
    if "camera_pose" in args.suites:
        run_camera_pose(args)
    print(f"\n[ok] Redirect4D-Bench metric outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
