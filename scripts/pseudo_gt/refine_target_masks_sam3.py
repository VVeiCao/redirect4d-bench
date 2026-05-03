#!/usr/bin/env python3
"""Refine target-view pseudo-GT masks with SAM3 video propagation.

The Redirect4D-Bench mask refinement step uses the first frame of the rough
rendered target-view mask as a SAM3 mask prompt, then propagates that prompt
forward and backward through the post-WAN target RGB sequence. The final mask
for each frame is the thresholded average of forward and reverse logits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM3_SOURCE = ROOT / "third_party" / "sam3"


def configure_reproducibility(seed: int, deterministic: bool, warn_only: bool, allow_tf32: bool) -> None:
    import random

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


@dataclass(frozen=True)
class RefineTarget:
    case: str
    rgb_video: Path
    rough_mask: Path
    out_mask: Path


def read_mask_frames(path: Path, threshold: int) -> tuple[list[np.ndarray], float]:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open mask video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray > threshold)
    cap.release()
    if not frames:
        raise RuntimeError(f"mask video has no frames: {path}")
    return frames, fps


def extract_video_frames(video_path: Path, out_dir: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open RGB video: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError(f"RGB video has no frames: {video_path}")
    return n


def write_mask_mp4(masks: list[np.ndarray], out_path: Path, fps: float) -> None:
    import cv2
    import imageio_ffmpeg
    import numpy as np

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    h, w = masks[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, mask in enumerate(masks):
            m8 = mask.astype(np.uint8) * 255
            m3 = np.repeat(m8[:, :, None], 3, axis=2)
            cv2.imwrite(str(tmp_dir / f"{i:05d}.png"), m3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(tmp_dir / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-vf",
            f"scale={w}:{h}",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)


def write_refinement_metadata(
    out_mask: Path,
    *,
    rgb_video: Path,
    rough_mask: Path,
    n_frames: int,
    fps: float | None,
    seed: int,
    threshold: int,
) -> None:
    meta = {
        "method": "rough_frame0_post_wan_maskrefine_bidirectional",
        "rgb_video": str(rgb_video),
        "rough_mask_video": str(rough_mask),
        "seed_frame": 0,
        "n_frames": n_frames,
        "fps": fps,
        "seed": seed,
        "mask_threshold": threshold,
    }
    out_mask.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def build_predictor(device: str, sam3_source: Path):
    import torch

    sam3_source = sam3_source.resolve()
    if not (sam3_source / "sam3").is_dir():
        raise FileNotFoundError(
            f"SAM3 source tree not found at {sam3_source}. "
            "Run `git submodule update --init --recursive third_party/sam3` from the repo root."
        )
    sys.path.insert(0, str(sam3_source))
    loaded_sam3 = sys.modules.get("sam3")
    if loaded_sam3 is not None:
        loaded_path = Path(getattr(loaded_sam3, "__file__", "")).resolve()
        if sam3_source not in loaded_path.parents:
            for name in list(sys.modules):
                if name == "sam3" or name.startswith("sam3."):
                    del sys.modules[name]

    try:
        from sam3.model_builder import build_sam3_video_model
    except ImportError as exc:
        raise ImportError(
            "SAM3 is not importable. Install the submodule first, for example:\n"
            "  scripts/env/create_sam3_env.sh\n"
            "  conda activate redirect4d-sam3\n"
            "  python scripts/dev/check_environment.py --profile sam3"
        ) from exc

    sam3_model = build_sam3_video_model(device=device)
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    return predictor


def propagate_bidirectional(predictor, state, n: int, shape: tuple[int, int]) -> list[np.ndarray]:
    fwd = {}
    for frame_idx, _obj_ids, _scores, masks, _extra in predictor.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=n,
        reverse=False,
        propagate_preflight=True,
    ):
        fwd[frame_idx] = masks[0, 0].detach().float().cpu().numpy()

    rev = {}
    for frame_idx, _obj_ids, _scores, masks, _extra in predictor.propagate_in_video(
        state,
        start_frame_idx=n - 1,
        max_frame_num_to_track=n,
        reverse=True,
        propagate_preflight=False,
    ):
        rev[frame_idx] = masks[0, 0].detach().float().cpu().numpy()

    out: list[np.ndarray] = []
    for i in range(n):
        f = fwd.get(i)
        r = rev.get(i)
        if f is None and r is None:
            out.append(np.zeros(shape, dtype=bool))
        elif f is None:
            out.append(r > 0)
        elif r is None:
            out.append(f > 0)
        else:
            out.append(((f + r) / 2.0) > 0)
    return out


def refine_one(
    predictor,
    *,
    rgb_video: Path,
    rough_mask: Path,
    out_mask: Path,
    threshold: int,
) -> int:
    import torch

    rough_frames, fps = read_mask_frames(rough_mask, threshold)
    rough0 = rough_frames[0]
    h, w = rough0.shape[:2]

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp) / "frames"
        n_rgb = extract_video_frames(rgb_video, frames_dir)
        n = min(len(rough_frames), n_rgb)

        state = predictor.init_state(video_path=str(frames_dir))
        predictor.clear_all_points_in_video(state)

        if rough0.any():
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                mask=torch.from_numpy(rough0).to(torch.bool),
            )
        refined = propagate_bidirectional(predictor, state, n, (h, w))

    write_mask_mp4(refined, out_mask, fps)
    return n


def parse_case(case: str) -> tuple[str, str]:
    if "__" not in case:
        raise ValueError(f"case must be '<track>__<trajectory>': {case}")
    return case.split("__", 1)


def read_cases_jsonl(path: Path) -> list[str]:
    cases = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(__import__("json").loads(line)["case"] if line.startswith("{") else line)
    return cases


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def collect_dataset_targets(args: argparse.Namespace) -> list[RefineTarget]:
    if args.dataset_root is None:
        return []
    cases = list(args.case)
    if args.cases_jsonl:
        cases.extend(read_cases_jsonl(args.cases_jsonl))
    if not cases:
        default_cases = args.dataset_root / "cases.jsonl"
        if default_cases.exists():
            cases.extend(read_cases_jsonl(default_cases))
    if args.track:
        cases = [case for case in cases if parse_case(case)[0] == args.track]

    view_re = re.compile(args.view_regex) if args.view_regex else None
    targets: list[RefineTarget] = []
    for case in cases:
        track, trajectory = parse_case(case)
        if view_re and not view_re.search(trajectory):
            continue
        traj_dir = first_existing(
            [
                args.dataset_root / "tracks" / track / "redirected" / trajectory,
                args.dataset_root / "tracks" / track / "trajectories" / trajectory,
            ]
        )
        if traj_dir is None:
            print(f"[skip] no trajectory directory for {case}", file=sys.stderr)
            continue
        rgb_video = first_existing(
            [
                traj_dir / "inference" / "output_video.mp4",
                traj_dir / "videos" / "generated.mp4",
                traj_dir / "generated.mp4",
                traj_dir / "videos" / "rendered_rgb.mp4",
                traj_dir / "videos" / "rendered_images.mp4",
            ]
        )
        if rgb_video is None:
            print(f"[skip] no target RGB video for {case}", file=sys.stderr)
            continue
        rough_mask = first_existing(
            [
                traj_dir / "rendered_mask.mp4",
                traj_dir / "videos" / "rendered_mask.mp4",
                traj_dir / "inference" / "rendered_mask.mp4",
            ]
        )
        if rough_mask is None or not rough_mask.exists():
            print(f"[skip] no rough target mask for {case}", file=sys.stderr)
            continue
        targets.append(RefineTarget(case, rgb_video, rough_mask, traj_dir / args.output_name))
        if args.limit and len(targets) >= args.limit:
            break
    return targets


def parse_shard(value: str) -> tuple[int, int]:
    if not value:
        return 0, 1
    left, right = value.split("/", 1)
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-video", type=Path, help="Target-view RGB video to propagate through.")
    parser.add_argument("--rough-mask-video", type=Path, help="Rendered rough target-view mask scaffold.")
    parser.add_argument("--out-mask", type=Path, help="Output refined pseudo-GT mask video.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Dataset root with tracks/<track>/redirected/<trajectory>.",
    )
    parser.add_argument("--case", action="append", default=[], help="Case key: <track>__<trajectory>.")
    parser.add_argument("--cases-jsonl", type=Path)
    parser.add_argument("--track")
    parser.add_argument("--view-regex", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", default="", help='Shard spec "I/N" for parallel dispatch.')
    parser.add_argument("--output-name", default="mask.mp4")
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sam3-source", type=Path, default=DEFAULT_SAM3_SOURCE)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--deterministic-strict", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = collect_dataset_targets(args)
    if args.rgb_video or args.rough_mask_video or args.out_mask:
        if not (args.rgb_video and args.rough_mask_video and args.out_mask):
            raise ValueError("--rgb-video, --rough-mask-video, and --out-mask must be provided together")
        targets.append(RefineTarget("single", args.rgb_video, args.rough_mask_video, args.out_mask))
    if not targets:
        raise ValueError("no refinement targets selected")

    shard_i, shard_n = parse_shard(args.shard)
    targets = [target for i, target in enumerate(targets) if i % shard_n == shard_i]
    print(f"[targets] {len(targets)} mask refinement jobs")
    if args.dry_run:
        for target in targets:
            print(f"{target.case}: {target.rgb_video} + {target.rough_mask} -> {target.out_mask}")
        return

    configure_reproducibility(
        seed=args.seed,
        deterministic=not args.no_deterministic,
        warn_only=not args.deterministic_strict,
        allow_tf32=args.allow_tf32,
    )
    if args.device.startswith("cuda"):
        import torch

        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if args.allow_tf32 and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    predictor = build_predictor(args.device, args.sam3_source)

    for i, target in enumerate(targets, 1):
        if args.skip_existing and target.out_mask.exists():
            print(f"[skip] existing {target.out_mask}")
            continue
        print(f"[{i}/{len(targets)}] {target.case}")
        n = refine_one(
            predictor,
            rgb_video=target.rgb_video,
            rough_mask=target.rough_mask,
            out_mask=target.out_mask,
            threshold=args.mask_threshold,
        )
        _rough_frames, fps = read_mask_frames(target.rough_mask, args.mask_threshold)
        write_refinement_metadata(
            target.out_mask,
            rgb_video=target.rgb_video,
            rough_mask=target.rough_mask,
            n_frames=n,
            fps=fps,
            seed=args.seed,
            threshold=args.mask_threshold,
        )
        print(f"[ok] wrote {target.out_mask} ({n} frames)")


if __name__ == "__main__":
    main()
