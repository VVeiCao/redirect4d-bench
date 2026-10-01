"""Per-worker batch runner: Stage 0 (SV4D multiview)
+ VGGT foreground + VIPE+LyRA bg/align — per scene.

Each scene writes directly into outputs/prepared_vipe_lyra_noopt/<scene>/
(self-contained, no intermediate DPG prepared/ to clean up).

Stamps:
  batch_logs/stamps_pipeline_lyra_noopt/{scene}.done / .fail

Usage:
    CUDA_VISIBLE_DEVICES=0 python batch_run_pipeline.py \
        --worker-id 0 --total-workers 8 --scenes-file <list.txt>
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "page-4d"))


def patch_transformers_bert_head_mask() -> None:
    """Restore the BertModel.get_head_mask helper removed in newer transformers.

    VIPE's bundled GroundingDINO wrapper was written against transformers 4.x,
    where BertModel inherited get_head_mask from PreTrainedModel. Newer releases
    removed that public method, but GroundingDINO only needs the standard helper
    behavior: None expands to one None per layer, and explicit masks are
    reshaped to the 5D attention-mask format.
    """

    try:
        from transformers import BertModel
    except Exception:
        return
    original_get_extended_attention_mask = BertModel.get_extended_attention_mask

    def get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
        # GroundingDINO calls the transformers 4.x signature:
        #   get_extended_attention_mask(mask, input_shape, device)
        # Newer transformers interpret the third positional argument as dtype.
        try:
            import torch

            if isinstance(device, torch.device):
                device = None
            if dtype is None and isinstance(device, torch.dtype):
                dtype = device
            if dtype is None:
                dtype = getattr(self, "dtype", None)
        except Exception:
            pass
        try:
            return original_get_extended_attention_mask(
                self,
                attention_mask,
                input_shape,
                dtype=dtype,
            )
        except TypeError:
            return original_get_extended_attention_mask(
                self,
                attention_mask,
                input_shape,
                device,
            )

    BertModel.get_extended_attention_mask = get_extended_attention_mask

    if hasattr(BertModel, "get_head_mask"):
        return

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if hasattr(self, "_convert_head_mask_to_5d"):
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        # Minimal fallback matching the old transformers helper.
        import torch

        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=self.dtype if hasattr(self, "dtype") else torch.float32)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    BertModel.get_head_mask = get_head_mask


patch_transformers_bert_head_mask()

DATA_DIR = REPO / "data_merged_reprocess"
MANIFEST = DATA_DIR / "manifest.json"
LOGS = REPO / "batch_logs"
STAMPS = LOGS / "stamps_pipeline_lyra_noopt"

# Single output dir for the whole pipeline (VGGT + VIPE both write here).
VIPE_PREPARED = REPO / "outputs" / "prepared_vipe_lyra_noopt"

DEFAULT_VIPE_PIPELINE = "lyra"

REUSE_FILES_IN_POINTCLOUD = (
    "{fid}_foreground_5_views.ply",
    "{fid}_foreground_5_views.npz",
)


def has_vggt_foreground(scene: str) -> bool:
    prepared = VIPE_PREPARED / scene
    if not prepared.is_dir():
        return False
    frame_dirs = sorted(p for p in prepared.iterdir() if p.is_dir() and p.name.isdigit())
    if not frame_dirs:
        return False
    for frame_dir in frame_dirs:
        fid = frame_dir.name
        if not (frame_dir / "images" / f"{fid}_original.png").exists():
            return False
        if not (
            (frame_dir / "masks" / f"{fid}_mask.png").exists()
            or (frame_dir / "masks" / f"{fid}_original_mask.png").exists()
        ):
            return False
        pc_dir = frame_dir / "pointcloud"
        if not all((pc_dir / tmpl.format(fid=fid)).exists() for tmpl in REUSE_FILES_IN_POINTCLOUD):
            return False
    return True


def load_scenes(scenes_file: str = None) -> list[str]:
    if scenes_file:
        lines = Path(scenes_file).read_text().splitlines()
        return sorted(ln.strip() for ln in lines if ln.strip())
    m = json.loads(MANIFEST.read_text())
    return sorted(s["out_name"] for s in m["sequences"] if s["seq_index"] != 1)


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


# ── Stage 0 + VGGT foreground ────────────────────────────────────────────

def build_config(scene: str):
    """Write VGGT foreground directly into prepared_vipe_lyra_noopt/<scene>/
    so no intermediate prepared/<scene> is ever created.
    """
    from utils.config import Config
    cfg = Config()
    cfg.update("project.name", scene)
    cfg.update("project.input_images", str(DATA_DIR / scene / "images"))
    cfg.update("project.input_masks", str(DATA_DIR / scene / "masks"))
    # Route DPG output to the canonical VIPE dir — same target as VIPE config
    dst = VIPE_PREPARED / scene
    try:
        rel = str(dst.relative_to(REPO))
    except ValueError:
        rel = str(dst)
    cfg.update("project.output_prepared", rel)
    cfg._auto_generate_paths()
    cfg.update("project.output_prepared", rel)  # re-apply in case auto_paths overrode
    return cfg


def run_vggt_foreground(scene: str) -> tuple[float, float]:
    from pipeline.stage_0_pipeline import Stage0Pipeline
    from pipeline.stage_1a_pointcloud_pipeline import Stage1APointCloudPipeline

    if has_vggt_foreground(scene):
        print(f"[resume] Reusing existing Stage 0 + VGGT foreground for {scene}", flush=True)
        return 0.0, 0.0

    cfg = build_config(scene)

    t0 = time.time()
    Stage0Pipeline(cfg).run()
    t_s0 = time.time() - t0

    t0 = time.time()
    # Only run VGGT foreground — skip DPG's background + align (VIPE will redo them).
    Stage1APointCloudPipeline(cfg).run(stop_at="foreground")
    t_s1a = time.time() - t0

    return t_s0, t_s1a


# ── VIPE ─────────────────────────────────────────────────────────────────

def setup_vipe_scene(scene: str) -> Path:
    """No-op: DPG already wrote images/masks/5-view-pointcloud directly into
    VIPE_PREPARED/<scene>/. Just verify and return the path."""
    dst = VIPE_PREPARED / scene
    if not dst.is_dir():
        raise FileNotFoundError(f"VIPE prepared dir missing (did VGGT foreground run?): {dst}")
    return dst


def build_vipe_config(scene: str, prepared_vipe_dir: Path):
    from utils.config import Config
    cfg = Config()
    cfg.update("project.name", scene)
    try:
        rel = str(prepared_vipe_dir.relative_to(REPO))
    except ValueError:
        rel = str(prepared_vipe_dir)
    cfg.update("project.output_prepared", rel)
    cfg.update("stage_1.background.method", "vipe")
    cfg.update("stage_1.background.voxel_size", 0.005)
    cfg.update("stage_1.background.outlier_nb_neighbors", 100)
    if "VIPE_PIPELINE_OVERRIDE" in globals() and VIPE_PIPELINE_OVERRIDE:
        cfg.update("stage_1.background.vipe_pipeline", VIPE_PIPELINE_OVERRIDE)
    cfg._auto_generate_paths()
    return cfg


def run_vipe(scene: str) -> tuple[float, float]:
    from pipeline.stage_1a_pointcloud_pipeline import Stage1APointCloudPipeline

    t0 = time.time()
    dst = setup_vipe_scene(scene)
    t_setup = time.time() - t0

    cfg = build_vipe_config(scene, dst)

    t0 = time.time()
    Stage1APointCloudPipeline(cfg).run(resume_from="background")
    t_vipe = time.time() - t0

    return t_setup, t_vipe


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, required=True)
    parser.add_argument("--retry-fail", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing stamps and rebuild each assigned scene from Stage 0.")
    parser.add_argument("--scenes-file", type=str, default=None)
    parser.add_argument("--prepared-root", "--vipe-root", dest="prepared_root", type=str,
                        default=None,
                        help="Override prepared reconstruction output dir "
                             "(default: outputs/prepared_vipe_lyra_noopt). "
                             "--vipe-root is kept as a deprecated alias.")
    parser.add_argument("--stamps-subdir", type=str, default=None,
                        help="Override stamps dir name (default: stamps_pipeline_lyra_noopt)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override staged input root containing <scene>/images and <scene>/masks.")
    parser.add_argument("--vipe-pipeline", type=str, default=None,
                        help="VIPE pipeline config: dav3 / lyra / default (default: lyra)")
    parser.add_argument("--intrinsics-mode", choices=("noopt", "opt"), default="noopt",
                        help="noopt sets VIPE_NO_OPT_INTR=1; opt clears it and lets VIPE optimize intrinsics.")
    parser.add_argument("--seed", type=int, default=23,
                        help="Seed for Python/NumPy/Torch reconstruction components.")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="Disable deterministic CUDA settings.")
    parser.add_argument("--deterministic-strict", action="store_true",
                        help="Abort on PyTorch non-deterministic CUDA ops instead of warning.")
    args = parser.parse_args()

    global DATA_DIR, VIPE_PREPARED, STAMPS, VIPE_PIPELINE_OVERRIDE
    if args.data_root:
        p = Path(args.data_root)
        DATA_DIR = p if p.is_absolute() else (REPO / p)
    if args.prepared_root:
        p = Path(args.prepared_root)
        VIPE_PREPARED = p if p.is_absolute() else (REPO / p)
    if args.stamps_subdir:
        STAMPS = LOGS / args.stamps_subdir
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
    tag = f"[pipeline-w{args.worker_id:02d}|gpu{gpu}]"
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
        t_total_start = time.time()
        try:
            # Stage 0 + VGGT foreground
            t_s0, t_s1a = run_vggt_foreground(scene)
            print(f"{tag} ({i}/{len(scenes)}) VGGT done: {scene} "
                  f"s0={t_s0:.0f}s s1a={t_s1a:.0f}s", flush=True)

            # VIPE
            t_setup, t_vipe = run_vipe(scene)
            print(f"{tag} ({i}/{len(scenes)}) VIPE done: {scene} "
                  f"setup={t_setup:.0f}s vipe={t_vipe:.0f}s", flush=True)

            total = time.time() - t_total_start
            done_stamp.write_text(
                f"worker={args.worker_id}\n"
                f"stage_0={t_s0:.1f}s\n"
                f"stage_1a_foreground={t_s1a:.1f}s\n"
                f"vipe_setup={t_setup:.1f}s\n"
                f"vipe={t_vipe:.1f}s\n"
                f"vipe_pipeline={VIPE_PIPELINE_OVERRIDE}\n"
                f"intrinsics_mode={args.intrinsics_mode}\n"
                f"total={total:.1f}s\n"
            )
            if fail_stamp.exists():
                fail_stamp.unlink()
            print(f"{tag} ({i}/{len(scenes)}) DONE: {scene} total={total:.0f}s",
                  flush=True)

        except Exception as e:
            tb = traceback.format_exc()
            fail_stamp.write_text(f"worker={args.worker_id}\n{tb}\n")
            print(f"{tag} ({i}/{len(scenes)}) FAIL: {scene}: {e}", flush=True)
            print(tb, flush=True)

    print(f"{tag} worker finished", flush=True)


if __name__ == "__main__":
    main()
