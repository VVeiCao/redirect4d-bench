#!/usr/bin/env python3
"""Serve Redirect4D-Bench point clouds with Viser.

The script reads the public dataset layout directly:

    tracks/<track>/pointcloud/global_background.ply
    tracks/<track>/pointcloud/<frame>/foreground_*.ply

It also accepts the normalized layout used by some development exports:

    tracks/<track>/reconstruction/pointclouds/frames/<frame>/<frame>_foreground_*.ply
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
import viser


SMOOTH_NAME = "foreground_5_views_aligned_smooth.ply"
MASK_PANEL_WIDTH = 320
TRAJECTORY_PALETTE = (
    (40, 145, 255),
    (255, 132, 40),
    (96, 210, 128),
    (210, 95, 255),
)
ORIGINAL_TRAJECTORY_COLOR = (235, 90, 105)
ORIGINAL_CAMERA_COLOR = (235, 90, 105)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_ply(path: Path, stride: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    points = np.asarray(mesh.vertices, dtype=np.float32)
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    if stride > 1 and len(points) > stride:
        points = points[::stride]
        colors = colors[::stride]
    return points, colors


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized [w, x, y, z] quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / max(float(np.linalg.norm(quat)), 1e-8)


def extrinsic_to_camera_pose(extrinsic: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_cam_world = np.eye(4, dtype=np.float64)
    t_cam_world[:3, :] = np.asarray(extrinsic, dtype=np.float64)
    t_world_cam = np.linalg.inv(t_cam_world)
    return (
        t_world_cam[:3, 3].astype(np.float32),
        matrix_to_wxyz(t_world_cam[:3, :3]),
    )


def fov_aspect_from_intrinsic(
    intrinsic: list | np.ndarray | None,
    image_size: tuple[int, int] | None = None,
    default_fov: float = 0.55,
    default_aspect: float = 16.0 / 9.0,
) -> tuple[float, float]:
    if intrinsic is None:
        return default_fov, default_aspect
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if image_size is None:
        width = max(float(intrinsic[0, 2]) * 2.0, 1.0)
        height = max(float(intrinsic[1, 2]) * 2.0, 1.0)
    else:
        height, width = image_size
    fy = max(float(intrinsic[1, 1]), 1e-8)
    return float(2.0 * np.arctan(float(height) / (2.0 * fy))), float(width) / float(height)


def pointcloud_roots(track_dir: Path) -> tuple[Path, Path, str]:
    preserved = track_dir / "pointcloud"
    if preserved.exists():
        return preserved, preserved, "preserved"
    normalized = track_dir / "reconstruction" / "pointclouds"
    return normalized, normalized / "frames", "normalized"


def frame_ply(frame_dir: Path, frame: str, name: str, style: str) -> Path:
    if style == "preserved":
        return frame_dir / name
    return frame_dir / f"{frame}_{name}"


def resize_preview(
    image: np.ndarray,
    width: int = MASK_PANEL_WIDTH,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=interpolation)


def colorize_mask(mask: np.ndarray, color: tuple[int, int, int] = (255, 132, 40)) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    active = mask > 8
    out = np.full((*mask.shape[:2], 3), 24, dtype=np.uint8)
    out[active] = np.array(color, dtype=np.uint8)
    return resize_preview(out)


def missing_preview(label: str) -> np.ndarray:
    image = np.full((180, MASK_PANEL_WIDTH, 3), 38, dtype=np.uint8)
    cv2.putText(
        image,
        label,
        (16, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return image


def load_mask_video(path: Path, color: tuple[int, int, int]) -> list[np.ndarray]:
    if not path.exists():
        return []
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(colorize_mask(gray, color=color))
    cap.release()
    return frames


def preview_rgb(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return resize_preview(rgb, interpolation=cv2.INTER_AREA)


def load_rgb_video(path: Path) -> list[np.ndarray]:
    if not path.exists():
        return []
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(preview_rgb(frame))
    cap.release()
    return frames


def load_source_video_frames(
    track_dir: Path,
    frame_ids: list[str],
) -> dict[str, np.ndarray]:
    video_candidates = [
        track_dir / "video.mp4",
        track_dir / "input.mp4",
        track_dir / "source_video.mp4",
        track_dir / "original_images.mp4",
    ]
    for path in video_candidates:
        video_frames = load_rgb_video(path)
        if video_frames:
            return {
                frame: video_frames[min(idx, len(video_frames) - 1)]
                for idx, frame in enumerate(frame_ids)
            }

    frame_dir_candidates = [track_dir / "frames"]
    for frame_dir in frame_dir_candidates:
        if not frame_dir.exists():
            continue
        frames = {}
        for frame in frame_ids:
            path = frame_dir / f"{frame}.png"
            if not path.exists():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                frames[frame] = preview_rgb(image)
        if frames:
            return frames
    return {}


def load_source_video_masks(track_dir: Path, frame_ids: list[str]) -> dict[str, np.ndarray]:
    video_frames = load_mask_video(track_dir / "mask_video.mp4", color=(255, 132, 40))
    if video_frames:
        return {
            frame: video_frames[min(idx, len(video_frames) - 1)]
            for idx, frame in enumerate(frame_ids)
        }

    masks = {}
    mask_dir = track_dir / "masks"
    for frame in frame_ids:
        path = mask_dir / f"{frame}.png"
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks[frame] = colorize_mask(mask, color=(255, 132, 40))
    return masks


def load_target_video_masks(track_dir: Path) -> dict[str, list[np.ndarray]]:
    redirected = track_dir / "redirected"
    if not redirected.exists():
        redirected = track_dir / "trajectories"
    if not redirected.exists():
        return {}
    masks = {}
    for traj_dir in sorted(p for p in redirected.iterdir() if p.is_dir()):
        frames = load_mask_video(traj_dir / "mask.mp4", color=(80, 185, 255))
        if frames:
            masks[traj_dir.name] = frames
    return masks


def load_target_depths(track_dir: Path) -> dict[str, list[np.ndarray]]:
    redirected = track_dir / "redirected"
    if not redirected.exists():
        redirected = track_dir / "trajectories"
    if not redirected.exists():
        return {}
    depths = {}
    for traj_dir in sorted(p for p in redirected.iterdir() if p.is_dir()):
        frames = load_rgb_video(traj_dir / "depth.mp4")
        if frames:
            depths[traj_dir.name] = frames
    return depths


def load_trajectories(track_dir: Path) -> dict[str, dict]:
    redirected = track_dir / "redirected"
    if not redirected.exists():
        redirected = track_dir / "trajectories"
    if not redirected.exists():
        return {}

    trajectories = {}
    for traj_dir in sorted(p for p in redirected.iterdir() if p.is_dir()):
        path = traj_dir / "trajectory.json"
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[warn] failed to parse {path}: {exc}")
            continue
        camera_path = raw.get("camera_path") or raw.get("keyframes") or []
        points = []
        wxyz = []
        fovs = []
        aspects = []
        for item in camera_path:
            if "extrinsic" in item:
                position, rotation = extrinsic_to_camera_pose(item["extrinsic"])
            elif "position" in item:
                position = np.asarray(item["position"], dtype=np.float32)
                rotation = np.asarray(item.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            else:
                continue
            fov, aspect = fov_aspect_from_intrinsic(item.get("intrinsic"))
            points.append(position)
            wxyz.append(rotation)
            fovs.append(float(item.get("fov", fov)))
            aspects.append(float(item.get("aspect", aspect)))
        if not points:
            continue
        trajectories[traj_dir.name] = {
            "points": np.asarray(points, dtype=np.float32),
            "wxyz": np.asarray(wxyz, dtype=np.float32),
            "fov": fovs,
            "aspect": aspects,
        }
    return trajectories


def load_original_camera_trajectory(track_dir: Path) -> dict | None:
    path = track_dir / "camera.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[warn] failed to parse {path}: {exc}")
        return None

    image_size = raw.get("image_size")
    if isinstance(image_size, list) and len(image_size) >= 2:
        image_size = (int(image_size[0]), int(image_size[1]))
    else:
        image_size = None

    points = []
    wxyz = []
    fovs = []
    aspects = []
    for frame_id in sorted(k for k in raw.keys() if k.isdigit()):
        item = raw.get(frame_id)
        if not isinstance(item, dict) or "extrinsic" not in item:
            continue
        position, rotation = extrinsic_to_camera_pose(item["extrinsic"])
        fov, aspect = fov_aspect_from_intrinsic(item.get("intrinsic"), image_size=image_size)
        points.append(position)
        wxyz.append(rotation)
        fovs.append(fov)
        aspects.append(aspect)

    if not points:
        return None
    return {
        "points": np.asarray(points, dtype=np.float32),
        "wxyz": np.asarray(wxyz, dtype=np.float32),
        "fov": fovs,
        "aspect": aspects,
    }


def list_available_tracks(dataset_root: Path) -> list[str]:
    tracks: list[str] = []
    seen: set[str] = set()

    def add(track: str) -> None:
        if track and track not in seen:
            tracks.append(track)
            seen.add(track)

    for row in read_jsonl(dataset_root / "tracks.jsonl"):
        if isinstance(row, dict) and "track" in row:
            add(str(row["track"]))

    tracks_dir = dataset_root / "tracks"
    if tracks_dir.exists():
        for track_dir in sorted(p for p in tracks_dir.iterdir() if p.is_dir()):
            add(track_dir.name)
    return tracks


def resolve_track_dir(dataset_root: Path, track: str) -> Path:
    track_dir = dataset_root / "tracks" / track
    if track_dir.is_dir():
        return track_dir
    raise FileNotFoundError(f"track folder not found for {track} under {dataset_root}")


def choose_track(dataset_root: Path, track: str | None) -> str:
    tracks = list_available_tracks(dataset_root)
    if not tracks:
        raise FileNotFoundError(
            f"dataset root has no tracks: {dataset_root}\n"
            "If you are trying the quick sample, download it first:\n\n"
            "  hf download vveicao/redirect4d-bench \\\n"
            "    --repo-type dataset \\\n"
            "    --include 'sample/**' \\\n"
            "    --local-dir data/redirect4d_bench\n\n"
            "Then run with --dataset-root data/redirect4d_bench/sample.\n"
        )
    if track:
        if track not in tracks:
            available = "\n".join(f"  - {item}" for item in tracks[:20])
            extra = "" if len(tracks) <= 20 else f"\n  ... {len(tracks) - 20} more"
            raise KeyError(
                f"track not found in {dataset_root}: {track}\n"
                f"Available tracks:\n{available}{extra}"
            )
        return track
    return tracks[0]


def load_track(args: argparse.Namespace, track: str | None = None) -> dict:
    track = choose_track(args.dataset_root, track if track is not None else args.track)
    track_dir = resolve_track_dir(args.dataset_root, track)
    pc_root, frame_root, style = pointcloud_roots(track_dir)
    if not pc_root.exists():
        raise FileNotFoundError(f"pointcloud folder not found: {pc_root}")

    bg_path = pc_root / "global_background.ply"
    bg = load_ply(bg_path, args.bg_subsample) if bg_path.exists() else None

    frame_dirs = sorted(p for p in frame_root.iterdir() if p.is_dir() and p.name.isdigit())
    if args.frame_step > 1:
        frame_dirs = frame_dirs[:: args.frame_step]
    if args.max_frames:
        frame_dirs = frame_dirs[: args.max_frames]

    frames = []
    for frame_dir in frame_dirs:
        frame = frame_dir.name
        smooth_path = frame_ply(frame_dir, frame, SMOOTH_NAME, style)
        smooth = load_ply(smooth_path, args.fg_subsample) if smooth_path.exists() else None
        if smooth is not None:
            frames.append({"frame": frame, "smooth": smooth})

    frame_ids = [item["frame"] for item in frames]
    source_video_frames = load_source_video_frames(
        track_dir,
        frame_ids,
    )
    source_video_masks = load_source_video_masks(track_dir, frame_ids)
    target_video_masks = load_target_video_masks(track_dir)
    target_depths = load_target_depths(track_dir)
    trajectories = load_trajectories(track_dir)
    original_camera = load_original_camera_trajectory(track_dir)

    if bg is None and not frames:
        raise RuntimeError(f"no point clouds found under {pc_root}")
    return {
        "track": track,
        "background": bg,
        "frames": frames,
        "source_video_frames": source_video_frames,
        "source_video_masks": source_video_masks,
        "target_video_masks": target_video_masks,
        "target_depths": target_depths,
        "trajectories": trajectories,
        "original_camera": original_camera,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/redirect4d_bench"))
    parser.add_argument("--track")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--bg-subsample", type=int, default=12)
    parser.add_argument("--fg-subsample", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--point-size", type=float, default=0.007)
    parser.add_argument("--fg-point-size", type=float, default=0.004)
    parser.add_argument("--trajectory-camera-scale", type=float, default=0.08)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    args.dataset_root = args.dataset_root.resolve()
    track_names = list_available_tracks(args.dataset_root)
    if not track_names:
        raise RuntimeError(f"no tracks found in {args.dataset_root}")
    initial_track = choose_track(args.dataset_root, args.track)
    current_track_index = track_names.index(initial_track)
    data = load_track(args, initial_track)
    frames = data["frames"]

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.world_axes.visible = True

    background_handle = None
    smooth_handles = []
    trajectory_handles = {}
    trajectory_colors = {}
    current_camera_handle = None
    original_camera_path_handle = None
    original_camera_points_handle = None
    original_camera_handle = None

    def remove_handle(handle) -> None:
        if handle is not None:
            handle.remove()

    def clear_scene_handles() -> None:
        nonlocal background_handle
        nonlocal smooth_handles
        nonlocal trajectory_handles
        nonlocal trajectory_colors
        nonlocal current_camera_handle
        nonlocal original_camera_path_handle
        nonlocal original_camera_points_handle
        nonlocal original_camera_handle

        remove_handle(background_handle)
        for handle in smooth_handles:
            remove_handle(handle)
        for handle in trajectory_handles.values():
            remove_handle(handle)
        remove_handle(current_camera_handle)
        remove_handle(original_camera_path_handle)
        remove_handle(original_camera_points_handle)
        remove_handle(original_camera_handle)

        background_handle = None
        smooth_handles = []
        trajectory_handles = {}
        trajectory_colors = {}
        current_camera_handle = None
        original_camera_path_handle = None
        original_camera_points_handle = None
        original_camera_handle = None

    def build_scene_handles() -> None:
        nonlocal background_handle
        nonlocal smooth_handles
        nonlocal trajectory_handles
        nonlocal trajectory_colors
        nonlocal current_camera_handle
        nonlocal original_camera_path_handle
        nonlocal original_camera_points_handle
        nonlocal original_camera_handle

        if data["background"] is not None:
            points, colors = data["background"]
            background_handle = server.scene.add_point_cloud(
                "/background",
                points=points,
                colors=colors,
                point_size=args.point_size,
            )

        smooth_handles = []
        for idx, item in enumerate(frames):
            if item["smooth"] is not None:
                points, colors = item["smooth"]
                handle = server.scene.add_point_cloud(
                    f"/foreground_smooth/{item['frame']}",
                    points=points,
                    colors=colors,
                    point_size=args.fg_point_size,
                    visible=(idx == 0),
                )
                smooth_handles.append(handle)
            else:
                smooth_handles.append(None)

        trajectory_handles = {}
        trajectory_colors = {}
        for color_idx, (name, trajectory) in enumerate(data["trajectories"].items()):
            color = TRAJECTORY_PALETTE[color_idx % len(TRAJECTORY_PALETTE)]
            trajectory_colors[name] = color
            points = trajectory["points"]
            if len(points) < 2:
                continue
            trajectory_handles[name] = server.scene.add_spline_catmull_rom(
                f"/target_trajectory/{name}",
                points=points,
                line_width=3.0,
                color=color,
                visible=False,
            )

        first_trajectory = next(iter(data["trajectories"]), None)
        if first_trajectory is not None:
            trajectory = data["trajectories"][first_trajectory]
            color = trajectory_colors.get(first_trajectory, TRAJECTORY_PALETTE[0])
            current_camera_handle = server.scene.add_camera_frustum(
                "/target_camera/current",
                fov=trajectory["fov"][0],
                aspect=trajectory["aspect"][0],
                scale=args.trajectory_camera_scale,
                line_width=2.0,
                color=color,
                wxyz=trajectory["wxyz"][0],
                position=trajectory["points"][0],
                visible=True,
            )

        if data["original_camera"] is not None:
            original_camera = data["original_camera"]
            if len(original_camera["points"]) >= 2:
                original_camera_path_handle = server.scene.add_spline_catmull_rom(
                    "/original_camera/path",
                    points=original_camera["points"],
                    line_width=5.0,
                    color=ORIGINAL_TRAJECTORY_COLOR,
                    visible=False,
                )
                colors = np.tile(
                    np.array(ORIGINAL_TRAJECTORY_COLOR, dtype=np.uint8),
                    (len(original_camera["points"]), 1),
                )
                original_camera_points_handle = server.scene.add_point_cloud(
                    "/original_camera/positions",
                    points=original_camera["points"],
                    colors=colors,
                    point_size=0.025,
                    visible=False,
                )
            original_camera_handle = server.scene.add_camera_frustum(
                "/original_camera/current",
                fov=original_camera["fov"][0],
                aspect=original_camera["aspect"][0],
                scale=args.trajectory_camera_scale * 1.35,
                line_width=2.0,
                color=ORIGINAL_CAMERA_COLOR,
                wxyz=original_camera["wxyz"][0],
                position=original_camera["points"][0],
                visible=False,
            )

    build_scene_handles()

    with server.gui.add_folder("Track"):
        track_counter_text = server.gui.add_text(
            "Index",
            initial_value=f"{current_track_index + 1} / {len(track_names)}",
            disabled=True,
        )
        track_select = server.gui.add_dropdown(
            "Active track",
            options=tuple(track_names),
            initial_value=data["track"],
        )
        previous_track_button = server.gui.add_button(
            "Previous",
            icon=viser.Icon.ARROW_LEFT,
        )
        next_track_button = server.gui.add_button(
            "Next",
            icon=viser.Icon.ARROW_RIGHT,
        )
        frames_loaded_text = server.gui.add_text(
            "Frames loaded",
            initial_value=str(len(frames)),
            disabled=True,
        )
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, len(frames) - 1),
            step=1,
            initial_value=0,
        )
        frame_text = server.gui.add_text(
            "Frame ID",
            initial_value=frames[0]["frame"] if frames else "background only",
            disabled=True,
        )
        play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
        fps = server.gui.add_slider("FPS", min=1.0, max=30.0, step=0.5, initial_value=args.fps)
    with server.gui.add_folder("Display"):
        show_background = server.gui.add_checkbox("Background", True)
        show_smooth = server.gui.add_checkbox("Final foreground", True)
    with server.gui.add_folder("Trajectory"):
        trajectory_names = tuple(data["trajectories"].keys()) or ("(none)",)
        target_count_text = server.gui.add_text(
            "Target count",
            initial_value=str(len(data["trajectories"])),
            disabled=True,
        )
        trajectory_select = server.gui.add_dropdown(
            "Active target trajectory",
            options=trajectory_names,
            initial_value=trajectory_names[0],
        )
        show_target = server.gui.add_checkbox("Target camera & trajectory", True)
        show_original_camera = server.gui.add_checkbox("Original camera & trajectory", False)
    with server.gui.add_folder("Videos / Masks"):
        source_video_image = server.gui.add_image(
            missing_preview("source video missing"),
            label="Source video",
            format="png",
        )
        source_mask_image = server.gui.add_image(
            missing_preview("source video mask missing"),
            label="Source video mask",
            format="png",
        )
        target_mask_image = server.gui.add_image(
            missing_preview("target video mask missing"),
            label="Target video mask",
            format="png",
            visible=trajectory_select.value in data["target_video_masks"],
        )
        target_depth_image = server.gui.add_image(
            missing_preview("target depth missing"),
            label="Target depth",
            format="png",
            visible=trajectory_select.value in data["target_depths"],
        )

    is_playing = False
    switching_track = False

    def update_visibility() -> None:
        current = min(int(frame_slider.value), max(0, len(frames) - 1))
        if background_handle is not None:
            background_handle.visible = bool(show_background.value)
        for idx, handle in enumerate(smooth_handles):
            if handle is not None:
                handle.visible = bool(show_smooth.value) and idx == current
        if frames:
            frame_id = frames[current]["frame"]
            frame_text.value = frame_id
            source_video_image.image = data["source_video_frames"].get(
                frame_id,
                missing_preview(f"source video {frame_id} missing"),
            )
            source_mask_image.image = data["source_video_masks"].get(
                frame_id,
                missing_preview(f"source video mask {frame_id} missing"),
            )
            traj = trajectory_select.value
            target_frames = data["target_video_masks"].get(traj, [])
            if target_frames:
                target_mask_image.visible = True
                target_mask_image.image = target_frames[min(current, len(target_frames) - 1)]
            else:
                target_mask_image.visible = False
            target_depth_frames = data["target_depths"].get(traj, [])
            if target_depth_frames:
                target_depth_image.visible = True
                target_depth_image.image = target_depth_frames[
                    min(current, len(target_depth_frames) - 1)
                ]
            else:
                target_depth_image.visible = False
            for name, handle in trajectory_handles.items():
                handle.visible = bool(show_target.value) and name == traj
            trajectory = data["trajectories"].get(traj)
            if current_camera_handle is not None and trajectory is not None:
                camera_idx = min(current, len(trajectory["points"]) - 1)
                current_camera_handle.visible = bool(show_target.value)
                current_camera_handle.color = trajectory_colors.get(traj, TRAJECTORY_PALETTE[0])
                current_camera_handle.position = trajectory["points"][camera_idx]
                current_camera_handle.wxyz = trajectory["wxyz"][camera_idx]
                current_camera_handle.fov = trajectory["fov"][camera_idx]
                current_camera_handle.aspect = trajectory["aspect"][camera_idx]
            elif current_camera_handle is not None:
                current_camera_handle.visible = False
            if data["original_camera"] is not None:
                original_camera = data["original_camera"]
                original_idx = min(current, len(original_camera["points"]) - 1)
                if original_camera_path_handle is not None:
                    original_camera_path_handle.visible = bool(show_original_camera.value)
                if original_camera_points_handle is not None:
                    original_camera_points_handle.visible = bool(show_original_camera.value)
                if original_camera_handle is not None:
                    original_camera_handle.visible = bool(show_original_camera.value)
                    original_camera_handle.position = original_camera["points"][original_idx]
                    original_camera_handle.wxyz = original_camera["wxyz"][original_idx]
                    original_camera_handle.fov = original_camera["fov"][original_idx]
                    original_camera_handle.aspect = original_camera["aspect"][original_idx]
        else:
            frame_text.value = "background only"
            source_video_image.image = missing_preview("source video missing")
            source_mask_image.image = missing_preview("source video mask missing")
            target_mask_image.visible = False
            target_depth_image.visible = False

    def switch_track(target_index: int) -> None:
        nonlocal current_track_index
        nonlocal data
        nonlocal frames
        nonlocal is_playing
        nonlocal switching_track

        if not track_names:
            return
        target_index %= len(track_names)
        if target_index == current_track_index and data["track"] == track_names[target_index]:
            return

        switching_track = True
        is_playing = False
        play_button.visible = True
        pause_button.visible = False
        frame_slider.disabled = False

        clear_scene_handles()
        current_track_index = target_index
        data = load_track(args, track_names[current_track_index])
        frames = data["frames"]
        build_scene_handles()

        track_counter_text.value = f"{current_track_index + 1} / {len(track_names)}"
        track_select.value = data["track"]
        frames_loaded_text.value = str(len(frames))
        frame_slider.max = max(0, len(frames) - 1)
        frame_slider.value = 0
        frame_text.value = frames[0]["frame"] if frames else "background only"

        trajectory_names = tuple(data["trajectories"].keys()) or ("(none)",)
        trajectory_select.options = trajectory_names
        trajectory_select.value = trajectory_names[0]
        target_count_text.value = str(len(data["trajectories"]))

        switching_track = False
        update_visibility()
        print(
            f"Switched to {data['track']} ({current_track_index + 1}/{len(track_names)})",
            flush=True,
        )

    @frame_slider.on_update
    def _(_) -> None:
        update_visibility()

    @play_button.on_click
    def _(_) -> None:
        nonlocal is_playing
        is_playing = True
        play_button.visible = False
        pause_button.visible = True
        frame_slider.disabled = True

    @pause_button.on_click
    def _(_) -> None:
        nonlocal is_playing
        is_playing = False
        play_button.visible = True
        pause_button.visible = False
        frame_slider.disabled = False

    @show_background.on_update
    def _(_) -> None:
        update_visibility()

    @show_smooth.on_update
    def _(_) -> None:
        update_visibility()

    @show_target.on_update
    def _(_) -> None:
        update_visibility()

    @show_original_camera.on_update
    def _(_) -> None:
        update_visibility()

    @trajectory_select.on_update
    def _(_) -> None:
        update_visibility()

    @track_select.on_update
    def _(_) -> None:
        if switching_track:
            return
        if track_select.value in track_names:
            switch_track(track_names.index(track_select.value))

    @previous_track_button.on_click
    def _(_) -> None:
        switch_track(current_track_index - 1)

    @next_track_button.on_click
    def _(_) -> None:
        switch_track(current_track_index + 1)

    update_visibility()

    print(
        f"Viser is serving {len(track_names)} tracks on http://localhost:{args.port}; "
        f"initial track: {data['track']}"
    )
    print("Use VS Code Ports to forward the port, then open the forwarded URL.")
    while True:
        if is_playing and frames:
            frame_slider.value = (int(frame_slider.value) + 1) % len(frames)
            update_visibility()
        time.sleep(1.0 / max(float(fps.value), 1.0))


if __name__ == "__main__":
    main()
