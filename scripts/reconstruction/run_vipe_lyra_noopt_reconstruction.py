#!/usr/bin/env python
"""Run the canonical Redirect4D reconstruction entrypoints.

This wrapper delegates the heavyweight reconstruction work to the bundled
`reconstruction/redirect4d` tree by default. Large third-party code is pinned as
submodules and model weights are downloaded into ignored local directories.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_R4D_ROOT = Path(__file__).resolve().parents[2] / "reconstruction" / "redirect4d"


def read_scenes(path: Path | None, inline: list[str]) -> list[str]:
    scenes: list[str] = []
    if path is not None:
        scenes.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    scenes.extend(inline)
    seen: set[str] = set()
    unique: list[str] = []
    for scene in scenes:
        if scene not in seen:
            unique.append(scene)
            seen.add(scene)
    return unique


def build_command(args: argparse.Namespace, scenes_file: Path | None) -> list[str]:
    r4d_root = args.r4d_root.resolve()
    script_name = "batch_run_pipeline.py" if args.mode == "full" else "vipe_batch_run.py"
    script = r4d_root / script_name
    if not script.is_file():
        raise FileNotFoundError(f"missing upstream reconstruction script: {script}")

    cmd = [
        str(args.python),
        str(script),
        "--worker-id",
        str(args.worker_id),
        "--total-workers",
        str(args.total_workers),
        "--vipe-pipeline",
        args.vipe_pipeline,
        "--intrinsics-mode",
        args.intrinsics_mode,
    ]
    if scenes_file is not None:
        cmd.extend(["--scenes-file", str(scenes_file)])
    if args.retry_fail:
        cmd.append("--retry-fail")
    if args.force:
        cmd.append("--force")
    if args.prepared_root:
        cmd.extend(["--prepared-root", args.prepared_root])
    if args.data_root:
        cmd.extend(["--data-root", args.data_root])
    if args.stamps_subdir:
        cmd.extend(["--stamps-subdir", args.stamps_subdir])
    cmd.extend(["--seed", str(args.seed)])
    if args.no_deterministic:
        cmd.append("--no-deterministic")
    if args.deterministic_strict:
        cmd.append("--deterministic-strict")
    if args.mode == "vipe-only" and args.dpg_root:
        cmd.extend(["--dpg-root", args.dpg_root])
    return cmd


def make_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.intrinsics_mode == "noopt":
        env["VIPE_NO_OPT_INTR"] = "1"
    else:
        env.pop("VIPE_NO_OPT_INTR", None)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    if not args.no_deterministic:
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    python_bin = Path(args.python).resolve().parent
    env["PATH"] = f"{python_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Redirect4D VIPE+LyRA reconstruction from a public wrapper."
    )
    parser.add_argument(
        "--r4d-root",
        type=Path,
        default=DEFAULT_R4D_ROOT,
        help="Redirect4D reconstruction root. Defaults to the bundled reconstruction/redirect4d tree.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for the upstream env.")
    parser.add_argument(
        "--mode",
        choices=("full", "vipe-only"),
        default="full",
        help=(
            "full runs Stage 0 + VGGT foreground + VIPE background/alignment; "
            "vipe-only reuses existing foreground outputs and reruns VIPE."
        ),
    )
    parser.add_argument("--scene", action="append", default=[], help="Scene/track name to process.")
    parser.add_argument("--scenes-file", type=Path, help="Text file containing scene names.")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--gpu", help="Value for CUDA_VISIBLE_DEVICES, for example 0.")
    parser.add_argument("--retry-fail", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore done stamps and rebuild selected scenes.")
    parser.add_argument("--vipe-pipeline", default="lyra")
    parser.add_argument("--intrinsics-mode", choices=("noopt", "opt"), default="noopt")
    parser.add_argument(
        "--prepared-root",
        "--vipe-root",
        dest="prepared_root",
        default="outputs/prepared_vipe_lyra_noopt",
        help=(
            "Upstream prepared reconstruction output root. "
            "--vipe-root is kept as a deprecated alias and does not point to "
            "the VIPE source submodule."
        ),
    )
    parser.add_argument(
        "--data-root",
        help="Staged reconstruction input root containing <scene>/images and <scene>/masks.",
    )
    parser.add_argument("--dpg-root", help="Only used by --mode vipe-only.")
    parser.add_argument("--stamps-subdir", help="Stamp directory name under upstream batch_logs/.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--deterministic-strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it.")
    args = parser.parse_args()

    scenes = read_scenes(args.scenes_file, args.scene)
    temp_file = None
    scenes_file = args.scenes_file
    try:
        if scenes and args.scenes_file is None:
            temp_file = tempfile.NamedTemporaryFile(
                "w", prefix="redirect4d_scenes_", suffix=".txt", delete=False
            )
            temp_file.write("\n".join(scenes) + "\n")
            temp_file.close()
            scenes_file = Path(temp_file.name)

        cmd = build_command(args, scenes_file)
        env = make_env(args)
        print(f"intrinsics_mode={args.intrinsics_mode}")
        print(f"VIPE_NO_OPT_INTR={env.get('VIPE_NO_OPT_INTR', '<unset>')}")
        print(f"PYTHONHASHSEED={env.get('PYTHONHASHSEED')}")
        if "CUBLAS_WORKSPACE_CONFIG" in env:
            print(f"CUBLAS_WORKSPACE_CONFIG={env['CUBLAS_WORKSPACE_CONFIG']}")
        if args.gpu is not None:
            print(f"CUDA_VISIBLE_DEVICES={args.gpu}")
        print(" ".join(cmd))
        if args.dry_run:
            return 0

        completed = subprocess.run(cmd, cwd=args.r4d_root, env=env, check=False)
        return completed.returncode
    finally:
        if temp_file is not None:
            Path(temp_file.name).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
