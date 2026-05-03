#!/usr/bin/env python3
"""Render a prepared Redirect4D case with the upstream Stage 1.4 script.

This wrapper is intentionally thin: it calls the upstream `scripts/1_4_rendering.py`
from a Redirect4D reconstruction checkout, while making sure the active
environment's `ffmpeg` is visible so the rendered PNG sequences are packaged
into MP4 files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_R4D_ROOT = Path(__file__).resolve().parents[2] / "reconstruction" / "redirect4d"


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--r4d-root",
        default=DEFAULT_R4D_ROOT,
        type=Path,
        help="Redirect4D reconstruction root. Defaults to the bundled reconstruction/redirect4d tree.",
    )
    ap.add_argument("--data-dir", required=True, type=Path, help="Prepared track directory.")
    ap.add_argument("--trajectory-json", required=True, type=Path)
    ap.add_argument(
        "--trajectory-label",
        help=(
            "Name used for the upstream render output directory. Defaults to the "
            "trajectory JSON stem, or the parent directory name when the JSON is named trajectory.json."
        ),
    )
    ap.add_argument("--python", default=os.environ.get("R4D_RECON_PY", sys.executable))
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--point-radius-px")
    ap.add_argument("--image-height")
    ap.add_argument("--image-width")
    ap.add_argument("--fps")
    ap.add_argument("--output-root", type=Path, help="Output root for this track; trajectory subdir is created below it.")
    return ap.parse_args()


def prepare_trajectory_json(path: Path, *, r4d_root: Path, label: str | None) -> Path:
    resolved = path.resolve()
    output_label = label or resolved.stem
    if output_label == "trajectory":
        output_label = resolved.parent.name
    if output_label == resolved.stem:
        return resolved

    cache_dir = r4d_root / ".render_trajectories"
    cache_dir.mkdir(parents=True, exist_ok=True)
    copied = cache_dir / f"{output_label}.json"
    shutil.copyfile(resolved, copied)
    return copied


def main() -> None:
    args = parse_args()
    args.r4d_root = args.r4d_root.resolve()
    script = args.r4d_root / "scripts" / "1_4_rendering.py"
    if not script.exists():
        raise FileNotFoundError(script)
    trajectory_json = prepare_trajectory_json(
        args.trajectory_json,
        r4d_root=args.r4d_root,
        label=args.trajectory_label,
    )
    cmd = [
        args.python,
        str(script),
        "--data_dir",
        str(args.data_dir.resolve()),
        "--trajectory_json",
        str(trajectory_json),
    ]
    for flag in ("point_radius_px", "image_height", "image_width", "fps"):
        value = getattr(args, flag)
        if value is not None:
            cmd.extend(["--" + flag.replace("_", "-"), str(value)])
    if args.output_root is not None:
        cmd.extend(["--output-rendering-base", str(args.output_root.resolve())])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    python_bin = Path(args.python).resolve().parent
    env["PATH"] = str(python_bin) + os.pathsep + env.get("PATH", "")
    run(cmd, cwd=args.r4d_root, env=env)


if __name__ == "__main__":
    main()
