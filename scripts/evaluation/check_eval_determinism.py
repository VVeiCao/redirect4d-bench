#!/usr/bin/env python3
"""Run a small deterministic evaluation check twice and compare outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import resolve_python  # noqa: E402

VOLATILE_JSON_KEYS = {"pred_video", "source", "staged", "wall_time_sec"}


def normalize_json(obj):
    if isinstance(obj, dict):
        return {
            key: normalize_json(value)
            for key, value in sorted(obj.items())
            if key not in VOLATILE_JSON_KEYS
        }
    if isinstance(obj, list):
        return [normalize_json(value) for value in obj]
    return obj


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        if "inputs" in parts or "vipe_tmp" in parts:
            continue
        if path.suffix.lower() not in {".json", ".csv"}:
            continue
        digest.update(rel.encode("utf-8") + b"\0")
        if path.suffix.lower() == ".json":
            payload = json.dumps(
                normalize_json(json.loads(path.read_text())),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        else:
            payload = path.read_bytes()
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def run_once(args: argparse.Namespace, out_dir: Path) -> str:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluation" / "run_final_metrics_case.py"),
        "--case",
        args.case,
        "--dataset-root",
        str(args.dataset_root),
        "--prediction-root",
        str(args.prediction_root),
        "--metrics-root",
        str(args.metrics_root),
        "--out-dir",
        str(out_dir),
        "--methods",
        *args.methods,
        "--suites",
        *args.suites,
        "--camera-modes",
        args.camera_modes,
        "--reconstruction-python",
        args.reconstruction_python,
        "--object-python",
        args.object_python,
        "--metric-seed",
        args.metric_seed,
    ]
    if args.mask_root:
        cmd.extend(["--mask-root", str(args.mask_root)])
    if args.recognition_cache:
        cmd.extend(["--recognition-cache", str(args.recognition_cache)])
    print("$ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return tree_digest(out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/redirect4d_bench"))
    parser.add_argument("--prediction-root", type=Path, default=Path("predictions"))
    parser.add_argument("--metrics-root", type=Path, default=Path("metrics"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/determinism_check"))
    parser.add_argument("--methods", nargs="+", default=["my_method"])
    parser.add_argument("--suites", nargs="+", default=["camera_pose"])
    parser.add_argument("--camera-modes", default="noopt")
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--recognition-cache", type=Path)
    parser.add_argument("--reconstruction-python")
    parser.add_argument("--object-python")
    parser.add_argument("--metric-seed", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.prediction_root = args.prediction_root.resolve()
    args.metrics_root = args.metrics_root.resolve()
    args.out_root = args.out_root.resolve()
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
    digest_a = run_once(args, args.out_root / "run_a")
    digest_b = run_once(args, args.out_root / "run_b")
    print(f"run_a sha256: {digest_a}")
    print(f"run_b sha256: {digest_b}")
    if digest_a != digest_b:
        print("[fail] evaluation outputs are not deterministic")
        return 1
    print("[ok] evaluation outputs are deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
