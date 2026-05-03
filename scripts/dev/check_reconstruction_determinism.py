#!/usr/bin/env python
"""Check that VIPE/LyRA reconstruction is deterministic for one prepared case.

The input must be a prepared Redirect4D case containing per-frame images, masks,
and foreground_5_views point clouds. The script creates two lightweight copies,
runs the VIPE/LyRA background stage twice with the same seed, and compares the
artifacts that drive pseudo-GT rendering and mask generation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R4D_ROOT = ROOT / "reconstruction" / "redirect4d"
DEFAULT_VIPE_SOURCE = ROOT / "third_party" / "vipe"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_ids(case_dir: Path, limit: int | None = None) -> list[str]:
    ids = sorted(p.name for p in case_dir.iterdir() if p.is_dir() and p.name.isdigit())
    return ids[:limit] if limit else ids


def symlink_or_replace(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve())


def stage_case(src_case: Path, dst_case: Path, limit: int | None) -> None:
    if dst_case.exists():
        shutil.rmtree(dst_case)
    for fid in frame_ids(src_case, limit):
        src_frame = src_case / fid
        dst_frame = dst_case / fid
        for subdir in ("images", "masks"):
            (dst_frame / subdir).mkdir(parents=True, exist_ok=True)
            for item in (src_frame / subdir).iterdir():
                symlink_or_replace(item, dst_frame / subdir / item.name)
        pc_src = src_frame / "pointcloud" / f"{fid}_foreground_5_views.npz"
        if not pc_src.is_file():
            raise FileNotFoundError(pc_src)
        pc_dst = dst_frame / "pointcloud"
        pc_dst.mkdir(parents=True, exist_ok=True)
        symlink_or_replace(pc_src, pc_dst / pc_src.name)


def run_one(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(R4D_ROOT))
    from core.vipe_background import VIPeBackgroundGenerator
    from utils.reproducibility import configure_reproducibility

    configure_reproducibility(
        seed=args.seed,
        deterministic=not args.no_deterministic,
        warn_only=not args.deterministic_strict,
    )
    generator = VIPeBackgroundGenerator(
        vipe_root=str(args.vipe_source),
        vipe_pipeline=args.vipe_pipeline,
    )
    generator.generate_global_background(str(args.case_dir))
    return 0


def launch_run(
    case_dir: Path,
    args: argparse.Namespace,
    gpu: str,
    log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VIPE_NO_OPT_INTR"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONHASHSEED"] = str(args.seed)
    if not args.no_deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        "--case-dir",
        str(case_dir),
        "--vipe-source",
        str(args.vipe_source),
        "--vipe-pipeline",
        args.vipe_pipeline,
        "--seed",
        str(args.seed),
    ]
    if args.no_deterministic:
        cmd.append("--no-deterministic")
    if args.deterministic_strict:
        cmd.append("--deterministic-strict")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def compare_npz(a: Path, b: Path) -> tuple[bool, str]:
    import numpy as np

    if sha256(a) == sha256(b):
        return True, "file_hash_equal=True"
    za = np.load(a)
    zb = np.load(b)
    diffs: list[str] = ["file_hash_equal=False"]
    ok = True
    for key in za.files:
        av = za[key]
        bv = zb[key]
        exact = np.array_equal(av, bv)
        if not exact:
            ok = False
        diffs.append(f"{key}: exact={exact}")
    return ok, "; ".join(diffs)


def decoded_depth_exact(a_zip: Path, b_zip: Path) -> bool:
    import numpy as np

    sys.path.insert(0, str(R4D_ROOT))
    from core.vipe_background import _read_exr_from_bytes

    with zipfile.ZipFile(a_zip) as za, zipfile.ZipFile(b_zip) as zb:
        if za.namelist() != zb.namelist():
            return False
        for name in za.namelist():
            da = _read_exr_from_bytes(za.read(name))
            db = _read_exr_from_bytes(zb.read(name))
            if not np.array_equal(da, db):
                return False
    return True


def compare_runs(case_a: Path, case_b: Path) -> list[str]:
    lines: list[str] = []
    for rel in ("global_camera.json", "global_background.ply"):
        same = sha256(case_a / rel) == sha256(case_b / rel)
        lines.append(f"{rel}: exact={same}")

    for rel in (
        "_vipe_output/pose/_vipe_frames.npz",
        "_vipe_output/intrinsics/_vipe_frames.npz",
    ):
        same = sha256(case_a / rel) == sha256(case_b / rel)
        lines.append(f"{rel}: exact={same}")

    depth_same = decoded_depth_exact(
        case_a / "_vipe_output/depth/_vipe_frames.zip",
        case_b / "_vipe_output/depth/_vipe_frames.zip",
    )
    lines.append(f"_vipe_output/depth/_vipe_frames.zip decoded_depth_exact={depth_same}")

    exact_frames = 0
    ids = frame_ids(case_a)
    for fid in ids:
        ok, detail = compare_npz(
            case_a / fid / "pointcloud" / f"{fid}_foreground_1_view.npz",
            case_b / fid / "pointcloud" / f"{fid}_foreground_1_view.npz",
        )
        exact_frames += int(ok)
        if not ok:
            lines.append(f"{fid}_foreground_1_view.npz: {detail}")
            break
    lines.append(f"foreground_1_view exact frames: {exact_frames}/{len(ids)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-case", type=Path, help="Prepared case to test.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "debug" / "reconstruction_determinism")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--gpu-a", default="0")
    parser.add_argument("--gpu-b", default="0")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--vipe-source", type=Path, default=DEFAULT_VIPE_SOURCE)
    parser.add_argument("--vipe-pipeline", default="lyra")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--deterministic-strict", action="store_true")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        if args.case_dir is None:
            parser.error("--case-dir is required with --run-one")
        return run_one(args)

    if args.prepared_case is None:
        parser.error("--prepared-case is required")

    case_a = args.output_dir / "run_a" / args.prepared_case.name
    case_b = args.output_dir / "run_b" / args.prepared_case.name
    stage_case(args.prepared_case, case_a, args.frame_limit)
    stage_case(args.prepared_case, case_b, args.frame_limit)

    proc_a = launch_run(case_a, args, args.gpu_a, args.output_dir / "run_a.log")
    if args.parallel:
        proc_b = launch_run(case_b, args, args.gpu_b, args.output_dir / "run_b.log")
        rc_a = proc_a.wait()
        rc_b = proc_b.wait()
    else:
        rc_a = proc_a.wait()
        proc_b = launch_run(case_b, args, args.gpu_b, args.output_dir / "run_b.log")
        rc_b = proc_b.wait()

    if rc_a or rc_b:
        print(f"run_a rc={rc_a}, run_b rc={rc_b}")
        print(f"logs: {args.output_dir / 'run_a.log'} and {args.output_dir / 'run_b.log'}")
        return 1

    report = compare_runs(case_a, case_b)
    report_path = args.output_dir / "determinism_report.txt"
    report_path.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"report: {report_path}")
    return 0 if all(("exact=False" not in line and "decoded_depth_exact=False" not in line) for line in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
