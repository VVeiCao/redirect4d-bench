#!/usr/bin/env python3
"""Run the Redirect4D-Bench scale-up pipeline for one custom case."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGES = ("prepare", "stage", "reconstruct", "render", "wan", "refine", "package")
PROTECTED_DATA_ROOTS = (
    ROOT / "data" / "redirect4d_bench",
    ROOT / "data" / "redirect4d_bench_restricted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    mask = parser.add_mutually_exclusive_group(required=True)
    mask.add_argument("--mask-video", type=Path)
    mask.add_argument("--mask-dir", type=Path)
    parser.add_argument("--trajectory-json", type=Path)
    parser.add_argument("--yaw", type=float, help="Target trajectory yaw in degrees.")
    parser.add_argument("--pitch", type=float, default=0.0, help="Target trajectory pitch in degrees.")
    parser.add_argument("--roll", type=float, default=0.0, help="Target trajectory roll in degrees.")
    parser.add_argument("--scale", type=float, default=1.0, help="Target trajectory radius scale.")
    parser.add_argument(
        "--trajectory-label",
        help="Output trajectory name. Defaults to the trajectory JSON parent name or stem.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help=(
            "Optional frozen prompt to reuse. If omitted, Wan generates "
            "generated_prompt.txt from the source/original video during the run."
        ),
    )
    parser.add_argument("--workspace", type=Path, help="Default: outputs/scale_up/<case-name>.")
    parser.add_argument("--release-root", type=Path, help="Default: <workspace>/release.")
    parser.add_argument("--num-frames", type=int, default=45)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--intrinsics-mode", choices=("noopt", "opt"), default="noopt")
    parser.add_argument("--bench-env", default="redirect4d-bench")
    parser.add_argument("--recon-env", default="redirect4d-recon")
    parser.add_argument("--sam3-env", default="redirect4d-sam3")
    parser.add_argument("--conda", default="conda")
    parser.add_argument(
        "--no-conda-run",
        action="store_true",
        help="Use the active Python for every stage instead of conda run.",
    )
    parser.add_argument("--package-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--include-source-rgb", action="store_true")
    parser.add_argument("--start-at", choices=STAGES, default="prepare")
    parser.add_argument("--stop-after", choices=STAGES, default="package")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    return parser.parse_args()


def trajectory_label(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    if path.name == "trajectory.json":
        return path.parent.name
    return path.stem


def format_trajectory_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def trajectory_label_from_params(args: argparse.Namespace) -> str:
    return (
        f"yaw_{format_trajectory_number(args.yaw)}"
        f"_pitch_{format_trajectory_number(args.pitch)}"
        f"_roll_{format_trajectory_number(args.roll)}"
        f"_scale_{format_trajectory_number(args.scale)}"
    )


def validate_trajectory_args(args: argparse.Namespace) -> None:
    if args.trajectory_json and args.yaw is not None:
        raise SystemExit(
            "error: use either --trajectory-json or --yaw/--pitch/--roll/--scale, not both"
        )
    if not args.trajectory_json and args.yaw is None:
        raise SystemExit("error: provide --yaw, or pass a custom --trajectory-json")
    if not args.trajectory_json and args.trajectory_label:
        raise SystemExit("error: --trajectory-label is only supported with --trajectory-json")


def selected_stages(start: str, stop: str) -> set[str]:
    start_idx = STAGES.index(start)
    stop_idx = STAGES.index(stop)
    if stop_idx < start_idx:
        raise ValueError("--stop-after must not come before --start-at")
    return set(STAGES[start_idx : stop_idx + 1])


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_output_is_not_dataset(path: Path, name: str) -> None:
    for root in PROTECTED_DATA_ROOTS:
        root = root.resolve()
        if path == root or is_relative_to(path, root):
            raise SystemExit(
                f"error: {name} must not be inside the released dataset folder: {root}\n"
                "Write scale-up outputs to outputs/scale_up/<case-name> or another "
                "separate folder."
            )


def py_cmd(args: argparse.Namespace, env_name: str) -> list[str]:
    if args.no_conda_run:
        return [sys.executable]
    return [args.conda, "run", "--no-capture-output", "-n", env_name, "python"]


def run(stage: str, cmd: list[str], *, dry_run: bool) -> None:
    print(f"\n[{stage}] " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


def remove_existing(path: Path, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"{path} already exists; omit --no-overwrite to replace it")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def place(src: Path, dst: Path, *, mode: str, overwrite: bool, required: bool = True) -> None:
    if not src.exists():
        if required:
            raise FileNotFoundError(src)
        return
    remove_existing(dst, overwrite)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def install_prompt(render_dir: Path, prompt_file: Path | None, overwrite: bool) -> bool:
    if prompt_file is None:
        return False
    dst = render_dir / "inference" / "generated_prompt.txt"
    place(prompt_file, dst, mode="copy", overwrite=overwrite)
    return True


def write_trajectory_metadata(path: Path, *, case: str, label: str, num_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "tracks": {
            case: {
                "track": case,
                "num_frames": num_frames,
                "trajectories": [label],
            }
        }
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def package_release(
    *,
    args: argparse.Namespace,
    workspace: Path,
    release_root: Path,
    label: str,
    render_dir: Path,
) -> None:
    case = args.case_name
    processed_track = workspace / "processed" / "tracks" / case
    prepared_track = workspace / "reconstruction" / "prepared_vipe_lyra_noopt" / case
    track_out = release_root / "tracks" / case
    if args.overwrite:
        remove_existing(track_out, True)
    track_out.mkdir(parents=True, exist_ok=True)

    place(
        processed_track / "mask_video.mp4",
        track_out / "mask_video.mp4",
        mode=args.package_mode,
        overwrite=args.overwrite,
    )
    place(
        processed_track / "masks",
        track_out / "masks",
        mode=args.package_mode,
        overwrite=args.overwrite,
    )
    place(
        processed_track / "custom_metadata.json",
        track_out / "custom_metadata.json",
        mode=args.package_mode,
        overwrite=args.overwrite,
        required=False,
    )

    if args.include_source_rgb:
        place(
            processed_track / "input.mp4",
            track_out / "video.mp4",
            mode=args.package_mode,
            overwrite=args.overwrite,
        )
        place(
            processed_track / "frames",
            track_out / "frames",
            mode=args.package_mode,
            overwrite=args.overwrite,
        )

    place(
        prepared_track / "global_camera.json",
        track_out / "camera.json",
        mode=args.package_mode,
        overwrite=args.overwrite,
        required=False,
    )
    pc_out = track_out / "pointcloud"
    place(
        prepared_track / "global_background.ply",
        pc_out / "global_background.ply",
        mode=args.package_mode,
        overwrite=args.overwrite,
    )
    for frame_dir in sorted(p for p in prepared_track.iterdir() if p.is_dir() and p.name.isdigit()):
        src_pc = frame_dir / "pointcloud"
        if src_pc.exists():
            place(src_pc, pc_out / frame_dir.name, mode=args.package_mode, overwrite=args.overwrite)

    traj_out = track_out / "redirected" / label
    place(
        args.trajectory_json,
        traj_out / "trajectory.json",
        mode=args.package_mode,
        overwrite=args.overwrite,
    )
    place(
        render_dir / "inference" / "rendered_depths.mp4",
        traj_out / "depth.mp4",
        mode=args.package_mode,
        overwrite=args.overwrite,
    )
    place(
        render_dir / "inference" / "mask.mp4",
        traj_out / "mask.mp4",
        mode=args.package_mode,
        overwrite=args.overwrite,
    )
    place(
        render_dir / "inference" / "generated_prompt.txt",
        traj_out / "prompt.txt",
        mode=args.package_mode,
        overwrite=args.overwrite,
    )
    print(f"[package] wrote release-style case: {track_out}", flush=True)


def main() -> None:
    args = parse_args()
    validate_trajectory_args(args)
    args.input_video = args.input_video.resolve()
    if args.mask_video:
        args.mask_video = args.mask_video.resolve()
    if args.mask_dir:
        args.mask_dir = args.mask_dir.resolve()
    if args.trajectory_json:
        args.trajectory_json = args.trajectory_json.resolve()
    if args.prompt_file:
        args.prompt_file = args.prompt_file.resolve()

    workspace = (args.workspace or Path("outputs") / "scale_up" / args.case_name).resolve()
    release_root = (args.release_root or workspace / "release").resolve()
    ensure_output_is_not_dataset(workspace, "--workspace")
    ensure_output_is_not_dataset(release_root, "--release-root")
    generated_trajectory = args.trajectory_json is None
    if generated_trajectory:
        label = trajectory_label_from_params(args)
        args.trajectory_json = (
            workspace
            / "trajectory_specs"
            / "tracks"
            / args.case_name
            / "redirected"
            / label
            / "trajectory.json"
        )
    else:
        label = trajectory_label(args.trajectory_json, args.trajectory_label)
    render_base = workspace / "rendering" / args.case_name
    render_dir = render_base / label
    selected = selected_stages(args.start_at, args.stop_after)

    if "prepare" in selected:
        cmd = [
            *py_cmd(args, args.bench_env),
            ROOT / "scripts" / "pipeline" / "process_custom_video.py",
            "--input-video",
            args.input_video,
            "--out-root",
            workspace / "processed" / "tracks" / args.case_name,
            "--case-name",
            args.case_name,
            "--num-frames",
            args.num_frames,
            "--fps",
            args.fps,
            "--width",
            args.width,
            "--height",
            args.height,
        ]
        if args.mask_video:
            cmd.extend(["--mask-video", args.mask_video])
        else:
            cmd.extend(["--mask-dir", args.mask_dir])
        if args.overwrite:
            cmd.append("--overwrite")
        else:
            cmd.append("--no-overwrite")
        run("prepare", cmd, dry_run=args.dry_run)

    if "stage" in selected:
        cmd = [
            *py_cmd(args, args.recon_env),
            ROOT / "scripts" / "reconstruction" / "stage_reconstruction_inputs.py",
            "--track",
            args.case_name,
            "--processed-root",
            workspace / "processed",
            "--dataset-root",
            workspace / "processed",
            "--output-root",
            workspace / "reconstruction" / "data_merged_reprocess",
            "--mode",
            "symlink",
        ]
        cmd.append("--overwrite" if args.overwrite else "--no-overwrite")
        run("stage", cmd, dry_run=args.dry_run)

    if "reconstruct" in selected:
        cmd = [
            *py_cmd(args, args.recon_env),
            ROOT / "scripts" / "reconstruction" / "run_vipe_lyra_noopt_reconstruction.py",
            "--mode",
            "full",
            "--scene",
            args.case_name,
            "--data-root",
            workspace / "reconstruction" / "data_merged_reprocess",
            "--prepared-root",
            workspace / "reconstruction" / "prepared_vipe_lyra_noopt",
            "--intrinsics-mode",
            args.intrinsics_mode,
            "--gpu",
            args.gpu,
            "--seed",
            args.seed,
        ]
        if args.overwrite:
            cmd.append("--force")
        run("reconstruct", cmd, dry_run=args.dry_run)

    if "render" in selected:
        if generated_trajectory:
            metadata = workspace / "trajectory_specs" / "metadata.json"
            if not args.dry_run:
                write_trajectory_metadata(
                    metadata,
                    case=args.case_name,
                    label=label,
                    num_frames=args.num_frames,
                )
            cmd = [
                *py_cmd(args, args.recon_env),
                ROOT / "scripts" / "reconstruction" / "generate_arc_trajectories.py",
                "--metadata",
                metadata,
                "--track",
                args.case_name,
                "--prepared-root",
                workspace / "reconstruction" / "prepared_vipe_lyra_noopt",
                "--dataset-root",
                workspace / "trajectory_specs",
                "--overwrite",
            ]
            run("trajectory", cmd, dry_run=args.dry_run)
        cmd = [
            *py_cmd(args, args.recon_env),
            ROOT / "scripts" / "reconstruction" / "render_prepared_case.py",
            "--data-dir",
            workspace / "reconstruction" / "prepared_vipe_lyra_noopt" / args.case_name,
            "--trajectory-json",
            args.trajectory_json,
            "--trajectory-label",
            label,
            "--output-root",
            render_base,
            "--gpu",
            args.gpu,
            "--image-height",
            args.height,
            "--image-width",
            args.width,
            "--fps",
            int(args.fps),
        ]
        run("render", cmd, dry_run=args.dry_run)

    if "wan" in selected:
        has_prompt = (
            False
            if args.dry_run
            else install_prompt(render_dir, args.prompt_file, args.overwrite)
        )
        cmd = [
            *py_cmd(args, args.recon_env),
            ROOT
            / "reconstruction"
            / "redirect4d"
            / "scripts"
            / "2_0_Wan2.2-VACE-Fun-A14B.py",
            "--data_dir",
            render_dir,
            "--seed",
            args.seed,
        ]
        if args.prompt_file or has_prompt:
            cmd.append("--use_saved_prompt")
        run("wan", cmd, dry_run=args.dry_run)

    if "refine" in selected:
        cmd = [
            *py_cmd(args, args.sam3_env),
            ROOT / "scripts" / "pseudo_gt" / "refine_target_masks_sam3.py",
            "--rgb-video",
            render_dir / "inference" / "output_video.mp4",
            "--rough-mask-video",
            render_dir / "inference" / "rendered_mask.mp4",
            "--out-mask",
            render_dir / "inference" / "mask.mp4",
            "--device",
            "cuda",
            "--seed",
            args.seed,
        ]
        run("refine", cmd, dry_run=args.dry_run)

    if "package" in selected:
        print(f"\n[package] release root: {release_root}", flush=True)
        if not args.dry_run:
            package_release(
                args=args,
                workspace=workspace,
                release_root=release_root,
                label=label,
                render_dir=render_dir,
            )


if __name__ == "__main__":
    main()
