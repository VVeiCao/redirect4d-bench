"""Camera-pose evaluation with GT intrinsics forced into VIPE (NoOpt style).

Rationale
---------
The default VIPE-LyRA-NoOpt pipeline initialises intrinsics from GeoCalib,
which is biased per-video (different artefacts → different fx estimates,
ranging from -13% to +13% vs the true intrinsic on our data). This biases
the per-method comparison.

Here we force VIPE to use the ground-truth intrinsic stored in
`trajectory.json[camera_path][0][intrinsic]` (the exact same K that R4D
renders against). Concretely we replace the `GeoCalibIntrinsicsProcessor`
with a stub that returns the GT fov_y without running the neural network.

Usage mirrors eval_camera_pose.py:
    python eval_camera_pose_gt.py \
        --pred_video <method>/videos/<case>.mp4 \
        --output_dir  results/gt_intr/<case>/<method>/ \
        --keep_vipe_outputs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "TORCH_HOME",
    os.environ.get("REDIRECT4D_TORCH_HOME", str(Path.home() / ".cache" / "torch")),
)

R4D_VIPE_ROOT = os.environ.get("REDIRECT4D_VIPE_ROOT", str(REPO_ROOT / "third_party" / "vipe"))
sys.path.insert(0, R4D_VIPE_ROOT)

from eval_camera_pose import (  # noqa: E402  (reuse math + target loader)
    load_and_interp_target_trajectory,
    compute_trajectory_errors,
)


# ---- Read GT intrinsics from trajectory.json ---------------------------------
def load_gt_intrinsics(trajectory_json: Path) -> dict:
    """Return dict with fx, fy, cx, cy, fov_y (radians). Uses camera_path[0]."""
    tj = json.loads(Path(trajectory_json).read_text())
    if "camera_path" not in tj or not tj["camera_path"]:
        raise ValueError(f"trajectory.json has no camera_path: {trajectory_json}")
    K = np.array(tj["camera_path"][0]["intrinsic"], dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    H = 2.0 * cy  # cy = H/2 by convention in R4D's camera.json
    # VIPE uses vertical FOV (radians): fx = H / (2 tan(fov_y/2))
    fov_y = 2.0 * math.atan(H / (2.0 * fx))
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "fov_y": fov_y, "H_hint": H}


# ---- Monkey-patch: swap GeoCalibIntrinsicsProcessor --------------------------
def install_gt_intrinsics_patch(fov_y: float) -> None:
    """Replace VIPE's GeoCalibIntrinsicsProcessor with a stub returning GT fov_y.

    IntrinsicEstimationProcessor.__call__ computes fx/fy from self.fov_y at
    per-frame time, so we just need to set self.fov_y and skip the heavy
    GeoCalib neural-net init.
    """
    import vipe.pipeline.default as _default
    import vipe.pipeline.processors as _procs
    from vipe.utils.cameras import CameraType

    _FOV_Y_RAD = float(fov_y)

    class _ForcedIntrinsicsProcessor(_procs.IntrinsicEstimationProcessor):
        def __init__(self, video_stream, gap_sec=1.0, camera_type=CameraType.PINHOLE):
            _procs.IntrinsicEstimationProcessor.__init__(self, video_stream, gap_sec)
            self.fov_y = _FOV_Y_RAD
            self.camera_type = camera_type
            self.distortion = []

    # Replace in the namespace default.py imports from
    _default.GeoCalibIntrinsicsProcessor = _ForcedIntrinsicsProcessor
    _procs.GeoCalibIntrinsicsProcessor = _ForcedIntrinsicsProcessor


# ---- VIPE runner (identical to eval_camera_pose, except it patches first) ----
def run_vipe_on_video(pred_video: Path,
                      output_dir: Path,
                      fov_y: float,
                      pipeline: str = "lyra") -> tuple:
    """Patch + run VIPE, return (c2w_poses, intrinsics)."""
    # Apply the monkey-patch BEFORE VIPE's pipeline builder resolves
    install_gt_intrinsics_patch(fov_y)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from vipe.streams.base import StreamList
    from vipe.pipeline import make_pipeline

    if not OmegaConf.has_resolver("eq"):
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    if not OmegaConf.has_resolver("neq"):
        OmegaConf.register_new_resolver("neq", lambda a, b: a != b)

    output_dir.mkdir(parents=True, exist_ok=True)
    vipe_config_dir = str((Path(R4D_VIPE_ROOT) / "configs").resolve())

    overrides = [
        f"pipeline={pipeline}",
        "streams=raw_mp4_stream",
        f"streams.base_path={pred_video}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
        f"pipeline.output.path={output_dir}",
        # GT-intr mode is conceptually NoOpt: we have the "true" K and don't
        # want SLAM to wander off from it.
        "pipeline.slam.optimize_intrinsics=false",
    ]

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=vipe_config_dir, version_base=None):
        cfg = compose(config_name="default", overrides=overrides)

    pipeline_cfg = OmegaConf.create(OmegaConf.to_container(cfg.pipeline, resolve=True))
    stream_cfg = OmegaConf.create(OmegaConf.to_container(cfg.streams, resolve=True))

    stream_list = StreamList.make(stream_cfg)
    pipe = make_pipeline(pipeline_cfg)
    for stream_idx in range(len(stream_list)):
        pipe.run(stream_list[stream_idx])

    case = pred_video.stem
    pose_file = output_dir / "pose" / f"{case}.npz"
    intr_file = output_dir / "intrinsics" / f"{case}.npz"
    if not pose_file.exists():
        raise FileNotFoundError(f"VIPE produced no pose file: {pose_file}")
    c2w = np.load(str(pose_file))["data"]
    intr = np.load(str(intr_file))["data"]
    return c2w, intr


# ---- Main --------------------------------------------------------------------
def parse_video_key(pred_video: Path) -> tuple:
    stem = pred_video.stem
    track, traj = stem.split("__", 1)
    return track, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_video", required=True, type=Path)
    ap.add_argument("--dataset_root", type=Path,
                    default=REPO_ROOT / "data" / "redirect4d_bench")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep_vipe_outputs", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    track, traj = parse_video_key(args.pred_video)
    traj_json = args.dataset_root / "tracks" / track / "redirected" / traj / "trajectory.json"
    if not traj_json.exists():
        raise FileNotFoundError(f"Target trajectory.json not found: {traj_json}")

    gt = load_gt_intrinsics(traj_json)
    print(f"[gt_intr] fx={gt['fx']:.2f}  fy={gt['fy']:.2f}  cx={gt['cx']:.1f}  cy={gt['cy']:.1f}  "
          f"fov_y={gt['fov_y']:.5f} rad ({math.degrees(gt['fov_y']):.2f}°)")

    vipe_out = args.output_dir / "vipe_tmp"
    t0 = time.time()
    print(f"[info] pred_video: {args.pred_video}")
    print(f"[info] target:     {traj_json}")
    print(f"[info] vipe out:   {vipe_out}")

    c2w_gen, intr = run_vipe_on_video(args.pred_video, vipe_out, fov_y=gt["fov_y"])
    print(f"[vipe] done in {time.time()-t0:.1f}s, {c2w_gen.shape[0]} frames")

    # Verify: VIPE output intrinsics should match GT (since we patched it)
    print(f"[vipe] intr[0] = {intr[0].tolist()}  (expect fx≈{gt['fx']:.1f})")

    c2w_target = load_and_interp_target_trajectory(traj_json, num_frames=c2w_gen.shape[0])
    res = compute_trajectory_errors(c2w_gen, c2w_target)
    res["track"] = track
    res["trajectory"] = traj
    res["pred_video"] = str(args.pred_video)
    res["pipeline"] = "lyra"
    res["mode"] = "gt_intr"
    res["gt_intrinsics_fov_y_rad"] = gt["fov_y"]
    res["gt_intrinsics_fx"] = gt["fx"]
    res["intrinsics_first_frame"] = intr[0].tolist() if intr.size else None
    res["wall_time_sec"] = float(time.time() - t0)

    (args.output_dir / "summary.json").write_text(json.dumps(res, indent=2))
    if not args.keep_vipe_outputs:
        shutil.rmtree(vipe_out, ignore_errors=True)

    print("\n=== Result ===")
    print(f"  track:     {track}")
    print(f"  traj:      {traj}")
    print(f"  n_frames:  {res['n_frames']}")
    print(f"  RotErr:    {res['rot_err_deg_mean']:.3f}° mean  ({math.degrees(res['rot_err_rad_sum']):.2f}° sum)")
    print(f"  TransErr:  {res['trans_err_m_mean']:.4f} m mean ({res['trans_err_m_sum']:.4f} m sum)")
    print(f"  wall:      {res['wall_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
