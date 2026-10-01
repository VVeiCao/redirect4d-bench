#!/usr/bin/env python3
"""Generate Redirect4D arc trajectory JSONs from metadata trajectory names.

Trajectory labels in the public metadata use the convention
`yaw_<deg>_pitch_<deg>_roll_<deg>_scale_<scale>`. This script reconstructs the
corresponding camera path from a prepared Redirect4D case and writes
`tracks/<track>/redirected/<trajectory>/trajectory.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from redirect4d_bench.data.metadata import load_metadata, read_track_list, track_items  # noqa: E402


TRAJ_RE = re.compile(
    r"yaw_(?P<yaw>-?\d+(?:p\d+)?)_pitch_(?P<pitch>-?\d+(?:p\d+)?)_roll_(?P<roll>-?\d+(?:p\d+)?)_scale_(?P<scale>-?\d+(?:p\d+)?)"
)


def parse_number(value: str) -> float:
    return float(value.replace("p", "."))


def parse_trajectory(label: str) -> tuple[float, float, float, float]:
    m = TRAJ_RE.fullmatch(label)
    if not m:
        raise ValueError(f"unsupported trajectory label: {label}")
    return tuple(parse_number(m.group(k)) for k in ("yaw", "pitch", "roll", "scale"))  # type: ignore[return-value]


def transform_to_z_up(points: np.ndarray) -> np.ndarray:
    points_flat = points.reshape(-1, 3)
    return np.column_stack([points_flat[:, 0], points_flat[:, 2], -points_flat[:, 1]]).reshape(points.shape)


def transform_rotation_to_z_up(r: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64) @ r


def transform_from_z_up(points: np.ndarray) -> np.ndarray:
    points_flat = points.reshape(-1, 3)
    return np.column_stack([points_flat[:, 0], -points_flat[:, 2], points_flat[:, 1]]).reshape(points.shape)


def transform_rotation_from_z_up(r: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64) @ r


def load_frame0_camera(prepared: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    with (prepared / "global_camera.json").open() as f:
        camera_data = json.load(f)
    image_size = camera_data.get("image_size", [480, 832])
    cam0 = camera_data["00000"]
    t_cam_world = np.vstack([np.asarray(cam0["extrinsic"], dtype=np.float64), [0, 0, 0, 1]])
    t_world_cam = np.linalg.inv(t_cam_world)
    pos_orig = t_world_cam[:3, 3]
    r_orig = t_world_cam[:3, :3]
    pos_zup = transform_to_z_up(pos_orig.reshape(1, 3)).flatten()
    r_zup = transform_rotation_to_z_up(r_orig)
    return pos_zup, r_zup, np.asarray(cam0["intrinsic"], dtype=np.float64), image_size


def center_depth_y(prepared: Path) -> float:
    pointmap_path = prepared / "00000" / "pointcloud" / "00000_foreground_1_view.npz"
    if pointmap_path.exists():
        data = np.load(pointmap_path)
        pointmap = data["foreground_pointmap"]
        h, w = pointmap.shape[:2]
        value = float(pointmap[h // 2, w // 2, 2])
        if math.isfinite(value) and abs(value) > 1e-8:
            return value
    bg = prepared / "global_background.ply"
    if bg.exists():
        import trimesh

        mesh = trimesh.load(bg)
        pts = transform_to_z_up(np.asarray(mesh.vertices))
        if len(pts):
            return float(np.min(pts[:, 1]))
    return -1.0


def zup_pose_to_original_extrinsic(position_zup: np.ndarray, r_zup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_orig = transform_from_z_up(np.asarray(position_zup).reshape(1, 3)).flatten()
    r_orig = transform_rotation_from_z_up(r_zup)
    t_world_cam = np.eye(4)
    t_world_cam[:3, :3] = r_orig
    t_world_cam[:3, 3] = pos_orig
    t_cam_world = np.linalg.inv(t_world_cam)
    return t_cam_world[:3, :], pos_orig


def generate_path(
    prepared: Path,
    trajectory: str,
    *,
    num_frames: int,
    num_keyframes: int,
) -> dict:
    yaw_deg, pitch_deg, roll_deg, scale = parse_trajectory(trajectory)
    ref_pos, ref_r, intrinsic, image_size = load_frame0_camera(prepared)
    depth_y = center_depth_y(prepared)

    base_center = np.array([0.0, depth_y, 0.0], dtype=np.float64)
    base_direction = base_center / (np.linalg.norm(base_center) or 1.0)
    center = base_direction * (abs(depth_y) * scale if abs(depth_y) > 1e-8 else scale)
    initial_vec = -center

    key_times = np.linspace(0, num_frames - 1, num_keyframes)
    key_positions = []
    key_rotations = []
    for i in range(num_keyframes):
        t = i / max(num_keyframes - 1, 1)
        yaw = np.deg2rad(yaw_deg * t)
        pitch = np.deg2rad(pitch_deg * t)
        roll = np.deg2rad(roll_deg * t)
        r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        r_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        r_roll = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]])
        r_comp = r_yaw @ r_pitch @ r_roll
        key_positions.append(center + r_comp @ initial_vec)
        key_rotations.append(r_comp @ ref_r)

    slerp = Slerp(key_times, Rotation.from_matrix(np.stack(key_rotations)))
    sample_times = np.arange(num_frames)
    sample_rot = slerp(sample_times).as_matrix()
    key_positions = np.asarray(key_positions)
    sample_pos = np.empty((num_frames, 3), dtype=np.float64)
    for axis in range(3):
        sample_pos[:, axis] = np.interp(sample_times, key_times, key_positions[:, axis])

    fy = intrinsic[1, 1]
    fov = float(2 * np.arctan(float(image_size[0]) / (2 * fy))) if fy else float(np.deg2rad(60.0))
    aspect = float(image_size[1] / image_size[0])

    out = {
        "keyframes": [],
        "camera_path": [],
        "metadata": {
            "image_size": image_size,
            "num_keyframes": num_keyframes,
            "timestep_range": [0, num_frames - 1],
            "coordinate_system": "y-down (original)",
            "total_output_frames": num_frames,
            "source": "metadata_trajectory_label",
            "trajectory": trajectory,
        },
    }

    for t, pos, rot in zip(key_times, key_positions, key_rotations):
        extr, pos_orig = zup_pose_to_original_extrinsic(pos, rot)
        wxyz = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
        out["keyframes"].append(
            {
                "timestep": int(round(t)),
                "position": pos_orig.tolist(),
                "wxyz": wxyz.tolist(),
                "extrinsic": extr.tolist(),
                "intrinsic": intrinsic.tolist(),
                "fov": fov,
                "aspect": aspect,
            }
        )

    for i, (pos, rot) in enumerate(zip(sample_pos, sample_rot)):
        extr, pos_orig = zup_pose_to_original_extrinsic(pos, rot)
        wxyz = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
        out["camera_path"].append(
            {
                "output_frame": i,
                "video_time": i,
                "extrinsic": extr.tolist(),
                "intrinsic": intrinsic.tolist(),
                "position": pos_orig.tolist(),
                "wxyz": wxyz.tolist(),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/redirect4d_bench/metadata.json"))
    parser.add_argument("--track-list", type=Path)
    parser.add_argument("--track", action="append", default=[])
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--num-keyframes", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    selected = list(args.track)
    if args.track_list:
        selected.extend(read_track_list(args.track_list) or [])
    for track, info in track_items(metadata, selected or None):
        prepared = args.prepared_root / track
        if not prepared.is_dir():
            raise FileNotFoundError(prepared)
        for trajectory in info.get("trajectories", []):
            out_path = args.dataset_root / "tracks" / track / "redirected" / trajectory / "trajectory.json"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {out_path}", flush=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = generate_path(
                prepared,
                trajectory,
                num_frames=int(info.get("num_frames", 45)),
                num_keyframes=args.num_keyframes,
            )
            out_path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"[ok] {track} {trajectory} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
