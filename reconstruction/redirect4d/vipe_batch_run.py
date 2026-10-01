"""Per-worker batch runner for VIPE-only rerun.

Assumes scenes already have VGGT foreground outputs in
outputs/prepared_vipe_lyra_noopt/<scene>/ (from batch_run_pipeline.py).
Runs Stage1APointCloudPipeline(resume_from="background") with:
  - method=vipe
  - pipeline=lyra (default, overridable)
  - explicit intrinsics mode: noopt or opt

Writes global_background.ply, global_camera.json, and *_aligned* files into
the same directory (in-place, self-contained).

Each finished scene drops a stamp at batch_logs/stamps_vipe_lyra_noopt/<scene>.done
so this script is safe to re-launch for resume.

Usage (8 workers on 8 GPUs):
    CUDA_VISIBLE_DEVICES=0 python vipe_batch_run.py --worker-id 0 --total-workers 8 &
    ...
"""
import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Cap thread counts BEFORE importing torch/numpy to prevent oversubscription.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "page-4d"))

DPG_PREPARED = REPO / "outputs" / "prepared"
VIPE_PREPARED = REPO / "outputs" / "prepared_vipe_lyra_noopt"
RERUN_LIST = REPO / "rerun_vipe.txt"
LOGS = REPO / "batch_logs"
STAMPS = LOGS / "stamps_vipe_lyra_noopt"

DEFAULT_VIPE_PIPELINE = "lyra"


def set_paths(dpg_root: str = None, prepared_root: str = None, stamps_subdir: str = None):
    """Override module-level paths (used by overnight orchestrator).

    Uses REPO-relative paths (no resolve()) to preserve symlinks — required
    because output roots are typically local scratch directories or symlinks.
    """
    global DPG_PREPARED, VIPE_PREPARED, STAMPS

    def _path(p):
        p = Path(p)
        return p if p.is_absolute() else REPO / p

    if dpg_root:
        DPG_PREPARED = _path(dpg_root)
    if prepared_root:
        VIPE_PREPARED = _path(prepared_root)
    if stamps_subdir:
        STAMPS = LOGS / stamps_subdir
        STAMPS.mkdir(parents=True, exist_ok=True)

REUSE_FILES_IN_POINTCLOUD = (
    "{fid}_foreground_5_views.ply",
    "{fid}_foreground_5_views.npz",
)


def load_scenes(scenes_file: str = None) -> list[str]:
    """Load the list of seq names to rerun with VIPE."""
    src = Path(scenes_file) if scenes_file else RERUN_LIST
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; run apply_review_decisions.py first")
    lines = [ln.strip() for ln in src.read_text().splitlines()]
    return sorted(ln for ln in lines if ln)


def configure_intrinsics_mode(mode: str) -> None:
    if mode == "noopt":
        os.environ["VIPE_NO_OPT_INTR"] = "1"
    elif mode == "opt":
        os.environ.pop("VIPE_NO_OPT_INTR", None)
    else:
        raise ValueError(f"unknown intrinsics mode: {mode}")


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def symlink_or_replace(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve())


def setup_vipe_scene(scene: str) -> Path:
    """Create outputs/prepared_vipe/<scene>/ as a symlink mirror of the DPG-prepared dir.

    - images/ and masks/ are whole-dir symlinks (read-only for VIPE)
    - pointcloud/ is a real directory so VIPE can write foreground_1_view*
    - pointcloud/<fid>_foreground_5_views*  are symlinked individually (reused)
    """
    src = DPG_PREPARED / scene
    dst = VIPE_PREPARED / scene
    if not src.is_dir():
        raise FileNotFoundError(f"DPG-prepared source missing: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    frame_ids = sorted(d.name for d in src.iterdir()
                       if d.is_dir() and d.name.isdigit())
    if not frame_ids:
        raise ValueError(f"no frame dirs in {src}")

    for fid in frame_ids:
        frame_src = src / fid
        frame_dst = dst / fid
        frame_dst.mkdir(parents=True, exist_ok=True)

        for subdir in ("images", "masks"):
            sub_src = frame_src / subdir
            if sub_src.exists():
                symlink_or_replace(sub_src, frame_dst / subdir)

        pc_src = frame_src / "pointcloud"
        pc_dst = frame_dst / "pointcloud"
        pc_dst.mkdir(parents=True, exist_ok=True)
        for tmpl in REUSE_FILES_IN_POINTCLOUD:
            fname = tmpl.format(fid=fid)
            f_src = pc_src / fname
            if f_src.exists():
                symlink_or_replace(f_src, pc_dst / fname)

    return dst


def build_config(scene: str, prepared_vipe_dir: Path):
    from utils.config import Config

    cfg = Config()  # configs/default.yaml
    cfg.update("project.name", scene)
    try:
        rel = str(prepared_vipe_dir.relative_to(REPO))
    except ValueError:
        rel = str(prepared_vipe_dir)
    cfg.update("project.output_prepared", rel)
    cfg.update("stage_1.background.method", "vipe")
    if "VIPE_PIPELINE_OVERRIDE" in globals() and VIPE_PIPELINE_OVERRIDE:
        cfg.update("stage_1.background.vipe_pipeline", VIPE_PIPELINE_OVERRIDE)
    # VIPE post-processing overrides — much faster than defaults, quality
    # difference is negligible on VIPE-density point clouds.
    cfg.update("stage_1.background.voxel_size", 0.005)
    cfg.update("stage_1.background.outlier_nb_neighbors", 100)
    # VIPE-specific alignment overrides:
    # - Stricter depth edge filter: VIPE has many flying points from DA per-frame depth
    # - Larger erosion: removes more boundary uncertainty
    # - Kalman Q/R scaled ~100x: VIPE is in metric (m) scale, ~10x larger than DPG's
    #   arbitrary scale → Q and R should scale by k² for identical smoothing behavior.
    cfg.update("stage_1.alignment.depth_edge_rtol", 0.01)
    cfg.update("stage_1.alignment.erosion_kernel_size", 20)
    cfg.update("stage_1.alignment.erosion_iterations", 1)
    # Fallback (Level 2) for frames where strict Level 1 fails (small/thin foregrounds)
    cfg.update("stage_1.alignment.fallback_enabled", True)
    cfg.update("stage_1.alignment.fallback_erosion_kernel_size", 5)
    cfg.update("stage_1.alignment.fallback_erosion_iterations", 1)
    cfg.update("stage_1.alignment.fallback_depth_edge_rtol", 0.02)
    cfg.update("stage_1.alignment.kalman_process_noise", 1.0e-7)
    cfg.update("stage_1.alignment.kalman_measurement_noise", 20.0)
    cfg.update("stage_1.alignment.kalman_z_penalty", 0.01)
    cfg._auto_generate_paths()
    return cfg


def run_scene(scene: str) -> tuple[float, float, float]:
    from pipeline.stage_1a_pointcloud_pipeline import Stage1APointCloudPipeline

    t0 = time.time()
    dst = setup_vipe_scene(scene)
    t_setup = time.time() - t0

    cfg = build_config(scene, dst)

    t0 = time.time()
    Stage1APointCloudPipeline(cfg).run(resume_from="background")
    t_s1a = time.time() - t0

    return t_setup, t_s1a, t_setup + t_s1a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, required=True)
    parser.add_argument("--retry-fail", action="store_true",
                        help="Also retry scenes previously marked .fail")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing stamps and rebuild each assigned VIPE scene.")
    parser.add_argument("--scenes-file", type=str, default=None,
                        help="Load scene list from file instead of rerun_vipe.txt")
    parser.add_argument("--dpg-root", type=str, default=None,
                        help="Override DPG prepared root")
    parser.add_argument("--prepared-root", "--vipe-root", dest="prepared_root", type=str,
                        default=None,
                        help="Override prepared reconstruction output root. "
                             "--vipe-root is kept as a deprecated alias.")
    parser.add_argument("--stamps-subdir", type=str, default=None,
                        help="Override stamp dir under batch_logs/")
    parser.add_argument("--vipe-pipeline", type=str, default=None,
                        help="VIPE pipeline config name: dav3 (default) / lyra / default")
    parser.add_argument("--intrinsics-mode", choices=("noopt", "opt"), default="noopt",
                        help="noopt sets VIPE_NO_OPT_INTR=1; opt clears it and lets VIPE optimize intrinsics.")
    parser.add_argument("--seed", type=int, default=23,
                        help="Seed for Python/NumPy/Torch reconstruction components.")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="Disable deterministic CUDA settings.")
    parser.add_argument("--deterministic-strict", action="store_true",
                        help="Abort on PyTorch non-deterministic CUDA ops instead of warning.")
    args = parser.parse_args()

    set_paths(args.dpg_root, args.prepared_root, args.stamps_subdir)
    # Propagate pipeline override through global (read in build_config)
    global VIPE_PIPELINE_OVERRIDE
    VIPE_PIPELINE_OVERRIDE = args.vipe_pipeline or DEFAULT_VIPE_PIPELINE
    configure_intrinsics_mode(args.intrinsics_mode)

    from utils.reproducibility import configure_reproducibility

    configure_reproducibility(
        seed=args.seed,
        deterministic=not args.no_deterministic,
        warn_only=not args.deterministic_strict,
    )

    STAMPS.mkdir(parents=True, exist_ok=True)
    VIPE_PREPARED.mkdir(parents=True, exist_ok=True)

    scenes_all = load_scenes(args.scenes_file)
    scenes = scenes_all[args.worker_id::args.total_workers]

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    tag = f"[vipe-w{args.worker_id:02d}|gpu{gpu}]"
    print(f"{tag} {len(scenes)} scenes to process (from {len(scenes_all)} total)",
          flush=True)
    print(f"{tag} seed={args.seed} deterministic={not args.no_deterministic} "
          f"warn_only={not args.deterministic_strict}", flush=True)
    print(
        f"{tag} vipe_pipeline={VIPE_PIPELINE_OVERRIDE} intrinsics_mode={args.intrinsics_mode} "
        f"VIPE_NO_OPT_INTR={os.environ.get('VIPE_NO_OPT_INTR', '<unset>')}",
        flush=True,
    )

    for i, scene in enumerate(scenes, 1):
        done_stamp = STAMPS / f"{scene}.done"
        fail_stamp = STAMPS / f"{scene}.fail"

        if args.force:
            if done_stamp.exists():
                done_stamp.unlink()
            if fail_stamp.exists():
                fail_stamp.unlink()
            prepared_scene = VIPE_PREPARED / scene
            if prepared_scene.exists() or prepared_scene.is_symlink():
                print(f"{tag} ({i}/{len(scenes)}) force rebuild removes: {prepared_scene}", flush=True)
                remove_path(prepared_scene)

        if done_stamp.exists():
            print(f"{tag} ({i}/{len(scenes)}) skip done: {scene}", flush=True)
            continue
        if fail_stamp.exists() and not args.retry_fail:
            print(f"{tag} ({i}/{len(scenes)}) skip fail: {scene}", flush=True)
            continue

        print(f"{tag} ({i}/{len(scenes)}) start: {scene}", flush=True)
        t0 = time.time()
        try:
            t_setup, t_s1a, t_total = run_scene(scene)
            done_stamp.write_text(
                f"worker={args.worker_id}\n"
                f"setup={t_setup:.1f}s\n"
                f"stage_1a={t_s1a:.1f}s\n"
                f"vipe_pipeline={VIPE_PIPELINE_OVERRIDE}\n"
                f"intrinsics_mode={args.intrinsics_mode}\n"
                f"total={t_total:.1f}s\n"
            )
            if fail_stamp.exists():
                fail_stamp.unlink()
            total = time.time() - t0
            print(f"{tag} ({i}/{len(scenes)}) done: {scene} "
                  f"setup={t_setup:.0f}s s1a={t_s1a:.0f}s total={total:.0f}s",
                  flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            fail_stamp.write_text(f"worker={args.worker_id}\n{tb}\n")
            print(f"{tag} ({i}/{len(scenes)}) FAIL: {scene}: {e}", flush=True)
            print(tb, flush=True)

    print(f"{tag} worker finished", flush=True)


if __name__ == "__main__":
    main()
