"""Camera-trajectory accuracy: RotErr / TransErr vs. target trajectory.

Approach
--------
1. Run VIPE-LyRA-NoOpt on the method-generated video to recover a c2w pose
   sequence (metric scale, anchored at frame 0).
2. Interpolate the target `trajectory.json` keyframes onto the same per-frame
   timesteps (SLERP rotation, linear translation).
3. Convert both to relative-to-frame-0 poses and compare.

We use relative-to-frame-0 poses so that any residual world-frame offset
between VIPE's run-on-source and VIPE's run-on-generated (both anchored near
identity, but not exactly identical) cancels out.

Metrics (CameraCtrl formula, inherited by ReCamMaster):
    RotErr_per_frame   = arccos( (tr(R_gen · R_target^T) - 1) / 2 )     [rad]
    TransErr_per_frame = || t_gen - t_target ||_2                        [m]

Usage:
    VIPE_NO_OPT_INTR=1 python eval_camera_pose.py \\
        --pred_video <method>/videos/<key>.mp4 \\
        --dataset_root data/redirect4d_bench \\
        --output_dir results/<cfg>/<key>/ \\
        --device cuda
"""
import argparse
import json
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
# VIPE_NO_OPT_INTR is honoured but not set by default — the caller chooses.

R4D_VIPE_ROOT = os.environ.get("REDIRECT4D_VIPE_ROOT", str(REPO_ROOT / "third_party" / "vipe"))
sys.path.insert(0, R4D_VIPE_ROOT)  # so `import vipe` resolves


# ---------- VIPE runner (mirrors core/vipe_background.py) ----------
def run_vipe_on_video(pred_video: Path, output_dir: Path, pipeline: str = "lyra") -> tuple:
    """Run VIPE on a single mp4 → return (c2w_poses (N,4,4), intrinsics (N,4))."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from vipe.streams.base import StreamList
    from vipe.pipeline import make_pipeline

    # Register custom OmegaConf resolvers (VIPE's config uses eq/neq)
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
    ]
    if os.environ.get("VIPE_NO_OPT_INTR"):
        overrides.append("pipeline.slam.optimize_intrinsics=false")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=vipe_config_dir, version_base=None):
        cfg = compose(config_name="default", overrides=overrides)

    pipeline_cfg = OmegaConf.create(OmegaConf.to_container(cfg.pipeline, resolve=True))
    stream_cfg = OmegaConf.create(OmegaConf.to_container(cfg.streams, resolve=True))

    stream_list = StreamList.make(stream_cfg)
    pipe = make_pipeline(pipeline_cfg)
    for stream_idx in range(len(stream_list)):
        pipe.run(stream_list[stream_idx])

    # Find case name (it's pred_video.stem by default)
    case = pred_video.stem
    pose_file = output_dir / "pose" / f"{case}.npz"
    intr_file = output_dir / "intrinsics" / f"{case}.npz"
    if not pose_file.exists():
        raise FileNotFoundError(f"VIPE did not produce pose file: {pose_file}")
    c2w = np.load(str(pose_file))["data"]  # (N, 4, 4)
    intr = np.load(str(intr_file))["data"]  # (N, 4) [fx, fy, cx, cy]
    return c2w, intr


# ---------- Target trajectory interpolation ----------
def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two wxyz quaternions."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta0 = np.arccos(np.clip(d, -1.0, 1.0))
    sin_theta0 = np.sin(theta0)
    s0 = np.sin((1.0 - t) * theta0) / sin_theta0
    s1 = np.sin(t * theta0) / sin_theta0
    return s0 * q0 + s1 * q1


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def load_and_interp_target_trajectory(trajectory_json: Path, num_frames: int) -> np.ndarray:
    """Return (num_frames, 4, 4) c2w poses — R4D's actual rendering trajectory.

    trajectory.json ships with a pre-computed `camera_path` field (the dense
    per-frame w2c extrinsics R4D renders against). We use that directly.
    Fallback to Kochanek-Bartels spline interpolation of `keyframes` if
    `camera_path` is absent (matching R4D's renderer, not naive SLERP+linear).

    NOTE re kf0 "pollution": trajectory.json sometimes has two keyframes at
    timestep=0 — the raw source-camera pose (slightly off origin) and a
    synthetic origin anchor. The dense camera_path therefore begins at the
    polluted pose and has a small spline bend in frames 1–3. **This is fine
    for the metric** because `compute_trajectory_errors` applies
    `relative_to_frame0` to BOTH pred and target — the common anchor cancels
    the pollution. It only matters for visualization, where there is no
    anchor cancellation (see render_trajectories.py for the viz fix).
    """
    tj = json.loads(Path(trajectory_json).read_text())

    # Preferred: use the dense camera_path written by R4D's trajectory editor.
    if "camera_path" in tj and tj["camera_path"]:
        cp = tj["camera_path"]
        out = np.zeros((num_frames, 4, 4), dtype=np.float64)
        for i, entry in enumerate(cp[:num_frames]):
            ext = np.array(entry["extrinsic"], dtype=np.float64)  # 3x4 w2c
            T_w2c = np.eye(4); T_w2c[:3, :] = ext
            T_c2w = np.linalg.inv(T_w2c)
            out[i] = T_c2w
        # Pad if dense path is shorter than requested
        if len(cp) < num_frames:
            out[len(cp):] = out[len(cp) - 1]
        return out

    # Fallback: densify from keyframes
    kfs = tj["keyframes"]
    # If multiple keyframes share a timestep, take the LAST one at that timestep
    # (that's what normalized target values look like — e.g. position=[0,0,0]).
    by_t = {}
    for k in kfs:
        by_t[int(k["timestep"])] = k
    times = sorted(by_t.keys())
    if times[0] != 0:
        # Prepend identity if trajectory doesn't start at t=0
        by_t[0] = {"timestep": 0, "position": [0, 0, 0], "wxyz": [1, 0, 0, 0]}
        times = [0] + times

    positions = np.array([by_t[t]["position"] for t in times], dtype=np.float64)  # (K, 3)
    quats = np.array([by_t[t]["wxyz"] for t in times], dtype=np.float64)           # (K, 4), wxyz
    # Normalize
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)

    out = np.zeros((num_frames, 4, 4), dtype=np.float64)
    for f in range(num_frames):
        # Find bracket [t_lo, t_hi]
        if f <= times[0]:
            p, q = positions[0], quats[0]
        elif f >= times[-1]:
            p, q = positions[-1], quats[-1]
        else:
            for i in range(len(times) - 1):
                if times[i] <= f <= times[i + 1]:
                    t_lo, t_hi = times[i], times[i + 1]
                    alpha = (f - t_lo) / (t_hi - t_lo)
                    p = (1 - alpha) * positions[i] + alpha * positions[i + 1]
                    q = _quat_slerp(quats[i], quats[i + 1], alpha)
                    break
        R = _quat_wxyz_to_R(q)
        out[f, :3, :3] = R
        out[f, :3, 3] = p
        out[f, 3, 3] = 1.0
    return out


# ---------- Metrics ----------
def relative_to_frame0(poses: np.ndarray) -> np.ndarray:
    """poses: (N,4,4) c2w → return (N,4,4) where result[t] = inv(poses[0]) @ poses[t]."""
    inv0 = np.linalg.inv(poses[0])
    return np.einsum("ij,njk->nik", inv0, poses)


def rotation_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angular distance (radians) between two 3x3 rotation matrices."""
    R = R_a @ R_b.T
    cos = (np.trace(R) - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.arccos(cos))


def compute_trajectory_errors(c2w_gen: np.ndarray, c2w_target: np.ndarray) -> dict:
    """Compute per-frame + summary RotErr/TransErr after relative-to-frame-0."""
    n = min(c2w_gen.shape[0], c2w_target.shape[0])
    rel_g = relative_to_frame0(c2w_gen[:n])
    rel_t = relative_to_frame0(c2w_target[:n])

    rot_errs = np.array([
        rotation_angle(rel_g[f, :3, :3], rel_t[f, :3, :3]) for f in range(n)
    ])  # (N,) radians
    trans_errs = np.linalg.norm(rel_g[:, :3, 3] - rel_t[:, :3, 3], axis=-1)  # (N,) meters

    return {
        "n_frames": int(n),
        "rot_err_rad_per_frame": rot_errs.tolist(),
        "trans_err_m_per_frame": trans_errs.tolist(),
        "rot_err_rad_mean": float(np.mean(rot_errs)),
        "rot_err_deg_mean": float(np.degrees(np.mean(rot_errs))),
        "rot_err_rad_sum": float(np.sum(rot_errs)),
        "trans_err_m_mean": float(np.mean(trans_errs)),
        "trans_err_m_sum": float(np.sum(trans_errs)),
    }


# ---------- Main ----------
def parse_video_key(pred_video: Path) -> tuple:
    """pred_video = <...>/<track>__<traj>.mp4 → (track, traj)."""
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
    ap.add_argument("--keep_vipe_outputs", action="store_true",
                    help="Do NOT delete the intermediate vipe_tmp/ folder.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    track, traj = parse_video_key(args.pred_video)
    traj_json = args.dataset_root / "tracks" / track / "redirected" / traj / "trajectory.json"
    if not traj_json.exists():
        raise FileNotFoundError(f"Target trajectory.json not found: {traj_json}")

    vipe_out = args.output_dir / "vipe_tmp"
    t0 = time.time()
    print(f"[info] pred_video: {args.pred_video}")
    print(f"[info] target:     {traj_json}")
    print(f"[info] vipe out:   {vipe_out}")

    c2w_gen, intr = run_vipe_on_video(args.pred_video, vipe_out, pipeline="lyra")
    print(f"[vipe] done in {time.time()-t0:.1f}s, {c2w_gen.shape[0]} frames")

    c2w_target = load_and_interp_target_trajectory(traj_json, num_frames=c2w_gen.shape[0])
    print(f"[target] interpolated to {c2w_target.shape[0]} frames")

    res = compute_trajectory_errors(c2w_gen, c2w_target)
    res["track"] = track
    res["trajectory"] = traj
    res["pred_video"] = str(args.pred_video)
    res["pipeline"] = "lyra"
    res["no_opt_intrinsics"] = bool(os.environ.get("VIPE_NO_OPT_INTR"))
    res["intrinsics_first_frame"] = intr[0].tolist() if intr.size else None
    res["wall_time_sec"] = float(time.time() - t0)

    (args.output_dir / "summary.json").write_text(json.dumps(res, indent=2))
    if not args.keep_vipe_outputs:
        shutil.rmtree(vipe_out, ignore_errors=True)

    print("\n=== Result ===")
    print(f"  track:     {track}")
    print(f"  traj:      {traj}")
    print(f"  n_frames:  {res['n_frames']}")
    print(f"  RotErr:    {res['rot_err_deg_mean']:.3f}° mean  ({res['rot_err_rad_sum']:.4f} rad sum)")
    print(f"  TransErr:  {res['trans_err_m_mean']:.4f} m mean ({res['trans_err_m_sum']:.4f} m sum)")
    print(f"  wall:      {res['wall_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
