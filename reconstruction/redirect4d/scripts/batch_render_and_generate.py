#!/usr/bin/env python3
"""Batch run Stage 1.4 (render) + Stage 2.0 (Wan2.2 video gen) for every
non-global_camera trajectory in prepared_vipe_lyra_noopt/.

Parallelism: one job per GPU (round-robin via CUDA_VISIBLE_DEVICES).

Two phases:
    PHASE 1: render every trajectory (quick, ~5 min each)
    PHASE 2: Wan2.2 video-generate every rendered trajectory (slow, ~25 min each)

Resume-safe:
    - render skips if videos/rendered_depths.mp4 already exists
    - generate skips if inference/output_video.mp4 already exists
    - per-job logs in batch_logs/render_*.log and batch_logs/generate_*.log

Usage:
    python scripts/batch_render_and_generate.py
    python scripts/batch_render_and_generate.py --phase render     # just render
    python scripts/batch_render_and_generate.py --phase generate   # just generate
    python scripts/batch_render_and_generate.py --dry-run          # print plan only
"""

import argparse
import glob
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "outputs" / "prepared_vipe_lyra_noopt"
PREPARED_ROOT = DEFAULT_PREPARED_ROOT  # reassigned from CLI at main()
RENDERING_ROOT = PROJECT_ROOT / "outputs" / "rendering"
LOG_DIR = PROJECT_ROOT / "batch_logs"

CONDA_PY = os.environ.get("R4D_RECON_PY", sys.executable)


def collect_jobs() -> List[Tuple[str, str]]:
    """Return list of (track_name, trajectory_filename) for every non-global_camera trajectory."""
    jobs = []
    for track in sorted(os.listdir(PREPARED_ROOT)):
        track_dir = PREPARED_ROOT / track
        if not track_dir.is_dir():
            continue
        if not (track_dir / "global_background.ply").exists():
            continue
        for traj_path in sorted(track_dir.glob("*.json")):
            if traj_path.name == "global_camera.json":
                continue
            jobs.append((track, traj_path.name))
    return jobs


def render_output_dir(track: str, traj_filename: str) -> Path:
    traj_name = os.path.splitext(traj_filename)[0]
    return RENDERING_ROOT / track / traj_name


def render_is_done(track: str, traj_filename: str) -> bool:
    # Rendered RGB video is the canonical "done" signal.
    return (render_output_dir(track, traj_filename) / "videos" / "rendered_images.mp4").exists()


def generate_is_done(track: str, traj_filename: str) -> bool:
    return (render_output_dir(track, traj_filename) / "inference" / "output_video.mp4").exists()


def run_cmd(cmd: List[str], log_path: Path, env: dict) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write(f"# CMD: {' '.join(cmd)}\n")
        logf.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        logf.write(f"# START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                cwd=str(PROJECT_ROOT))
        rc = proc.wait()
        logf.write(f"\n# END: {time.strftime('%Y-%m-%d %H:%M:%S')}  rc={rc}\n")
    return rc


def make_render_cmd(track: str, traj_filename: str) -> List[str]:
    track_dir = PREPARED_ROOT / track
    traj_path = track_dir / traj_filename
    return [
        CONDA_PY,
        "scripts/1_4_rendering.py",
        "--data_dir", str(track_dir),
        "--trajectory_json", str(traj_path),
    ]


def make_generate_cmd(track: str, traj_filename: str) -> List[str]:
    out_dir = render_output_dir(track, traj_filename)
    return [
        CONDA_PY,
        "scripts/2_0_Wan2.2-VACE-Fun-A14B.py",
        "--data_dir", str(out_dir),
        "--use_saved_prompt",  # skip caption if already saved
    ]


def worker(gpu_queue: queue.Queue, cmd: List[str], log_path: Path, label: str) -> Tuple[str, int]:
    gpu = gpu_queue.get()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # 1_4_rendering invokes `ffmpeg` via subprocess — make sure the conda env's
        # ffmpeg (not absent from the default PATH) is found.
        conda_bin = os.path.dirname(CONDA_PY)
        env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
        t0 = time.time()
        rc = run_cmd(cmd, log_path, env)
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[{status:10s}] gpu{gpu} {label}  ({dt/60:.1f}m)", flush=True)
        return (label, rc)
    finally:
        gpu_queue.put(gpu)


def run_phase(phase: str, jobs: List[Tuple[str, str]], num_gpus: int, dry_run: bool):
    """phase ∈ {'render', 'generate'}"""
    is_done = render_is_done if phase == "render" else generate_is_done
    make_cmd = make_render_cmd if phase == "render" else make_generate_cmd

    todo = [(t, f) for (t, f) in jobs if not is_done(t, f)]
    skipped = len(jobs) - len(todo)
    print(f"\n=== PHASE: {phase} ===")
    print(f"  total={len(jobs)}  todo={len(todo)}  skipped(done)={skipped}")

    if dry_run:
        for t, f in todo[:30]:
            print(f"  would run: {phase}  {t}/{f}")
        if len(todo) > 30:
            print(f"  ... +{len(todo) - 30} more")
        return

    if not todo:
        print(f"  [{phase}] nothing to do.")
        return

    gpu_queue: queue.Queue = queue.Queue()
    for g in range(num_gpus):
        gpu_queue.put(g)

    with ThreadPoolExecutor(max_workers=num_gpus) as ex:
        futures = []
        for track, traj_filename in todo:
            traj_name = os.path.splitext(traj_filename)[0]
            label = f"{track}/{traj_name}"
            log_path = LOG_DIR / f"{phase}_{track}__{traj_name}.log"
            cmd = make_cmd(track, traj_filename)
            futures.append(ex.submit(worker, gpu_queue, cmd, log_path, label))

        fails = []
        for fut in as_completed(futures):
            lbl, rc = fut.result()
            if rc != 0:
                fails.append(lbl)

    print(f"\n=== {phase} done: {len(todo) - len(fails)}/{len(todo)} ok, {len(fails)} failed ===")
    if fails:
        print("Failed jobs (see batch_logs/):")
        for f in fails[:20]:
            print(f"  {f}")
        if len(fails) > 20:
            print(f"  ... +{len(fails) - 20} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["both", "render", "generate"], default="both")
    ap.add_argument("--num_gpus", type=int, default=None,
                    help="Parallel workers (default=all visible GPUs)")
    ap.add_argument("--root", type=str, default=None,
                    help="Prepared-data root directory (default: outputs/prepared_vipe_lyra_noopt)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global PREPARED_ROOT
    if args.root:
        PREPARED_ROOT = Path(args.root).resolve()
    print(f"Using prepared root: {PREPARED_ROOT}")

    if args.num_gpus is None:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:
            n = 1
        args.num_gpus = n
    print(f"Using {args.num_gpus} parallel GPU worker(s)")

    jobs = collect_jobs()
    print(f"Collected {len(jobs)} trajectories across "
          f"{len(set(t for t, _ in jobs))} tracks")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("both", "render"):
        run_phase("render", jobs, args.num_gpus, args.dry_run)
    if args.phase in ("both", "generate"):
        run_phase("generate", jobs, args.num_gpus, args.dry_run)

    print("\nAll done.")


if __name__ == "__main__":
    main()
