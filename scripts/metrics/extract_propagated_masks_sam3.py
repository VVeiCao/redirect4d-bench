#!/usr/bin/env python3
"""Extract SAM3 propagated masks for generated videos before object metrics.

User methods provide generated RGB videos only. Object Fidelity / Localization
first converts each generated video into a foreground-mask video with the
benchmark SAM3 propagation method, then compares that mask against the
dataset's target pseudo-GT mask.

Object Fidelity / Localization consume binary mask videos under:

    <out-root>/<method>/<case>.mp4

This wrapper calls the benchmark mask extractor from the metrics checkout, but
keeps dataset and prediction paths controlled by this repo.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINAL_METHODS: tuple[str, ...] = ()


def load_extractor(metrics_root: Path):
    extractor = metrics_root / "mask_extraction" / "extract" / "extract_propagated.py"
    if not extractor.exists():
        raise FileNotFoundError(
            f"SAM3 propagated-mask extractor not found: {extractor}. "
            "Pass --metrics-root pointing to the Redirect4D metrics checkout."
        )
    for path in (ROOT / "third_party" / "sam3", metrics_root / "sam3_refine" / "sam3"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    sys.path.insert(0, str(extractor.parent))
    spec = importlib.util.spec_from_file_location("redirect4d_extract_propagated", extractor)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {extractor}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-root", type=Path, required=True)
    ap.add_argument("--pred-root", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "redirect4d_bench")
    ap.add_argument(
        "--out-base",
        type=Path,
        default=ROOT / "outputs" / "object_metric_masks",
        help="Base directory. The extractor writes <out-base>/<seed_prompt>/<method>/<case>.mp4.",
    )
    ap.add_argument("--methods", nargs="+", default=list(FINAL_METHODS))
    ap.add_argument("--seed-source", choices=("src", "gt"), default="src")
    ap.add_argument("--prompt-mode", choices=("mask", "pt_erosion", "pt_box"), default="pt_box")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed-rng", type=int, default=7)
    return ap.parse_args()


def install_public_dataset_adapters(module, dataset_tracks: Path) -> None:
    """Adapt the legacy extractor to the public dataset layout.

    The public benchmark supports arbitrary user method names and stores the
    final target pseudo-GT mask as `mask.mp4`.
    """

    def seed_path_for(seed_source: str, track: str, traj: str) -> Path:
        if seed_source == "src":
            return dataset_tracks / track / "masks" / "00000.png"
        if seed_source == "gt":
            new_name = dataset_tracks / track / "redirected" / traj / "mask.mp4"
            return new_name
        raise ValueError(f"unknown seed_source: {seed_source!r}")

    def enumerate_cases(configs, seed_source: str):
        configs = list(configs)
        if not configs:
            configs = sorted(
                p.name for p in module.PREDICTIONS_ROOT.iterdir() if (p / "videos").is_dir()
            )
        keys = sorted(
            {
                path.stem
                for cfg in configs
                for path in (module.PREDICTIONS_ROOT / cfg / "videos").glob("*.mp4")
            }
        )
        out = []
        for key in keys:
            if "__" not in key:
                print(
                    f"WARN: skipping non-internal case filename for mask extraction: {key}",
                    file=sys.stderr,
                )
                continue
            track, traj = key.split("__", 1)
            seed = seed_path_for(seed_source, track, traj)
            if not seed.exists():
                print(f"WARN: missing {seed_source} seed for {key}: {seed}", file=sys.stderr)
                continue
            for cfg in configs:
                video = module.PREDICTIONS_ROOT / cfg / "videos" / f"{key}.mp4"
                if video.exists():
                    out.append((cfg, video, seed, key))
        return out

    module.seed_path_for = seed_path_for
    module.enumerate_cases = enumerate_cases


def main() -> int:
    args = parse_args()
    metrics_root = args.metrics_root.resolve()
    pred_root = args.pred_root.resolve()
    dataset_root = args.dataset_root.resolve()
    dataset_tracks = dataset_root / "tracks" if (dataset_root / "tracks").is_dir() else dataset_root
    out_base = args.out_base.resolve()

    module = load_extractor(metrics_root)
    module.PREDICTIONS_ROOT = pred_root
    module.DATASET_ROOT = dataset_tracks
    module.SWEEP_ROOT_BASE = out_base
    module.CONFIGS_DEFAULT = list(args.methods)
    install_public_dataset_adapters(module, dataset_tracks)

    # The upstream extractor writes under
    # SWEEP_ROOT_BASE/<derived_seed_prompt_name>/<method>/<case>.mp4.
    derived = module.derive_output_dir(args.prompt_mode, args.seed_source)
    expected = out_base / derived
    print(f"[info] masks will be written to: {expected}")
    print(f"[info] pass this as --mask-root for object metrics: {expected}")

    sys.argv = [
        sys.argv[0],
        "--configs",
        ",".join(args.methods),
        "--seed-source",
        args.seed_source,
        "--prompt-mode",
        args.prompt_mode,
        "--shard",
        args.shard,
        "--seed-rng",
        str(args.seed_rng),
    ]
    if args.limit:
        sys.argv.extend(["--limit", str(args.limit)])
    if args.force:
        sys.argv.append("--force")
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
