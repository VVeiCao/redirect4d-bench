#!/usr/bin/env python3
"""Per-track camera trajectory editor.

Combines the trajectory editing features of 1_3_cam_traj.py with the multi-track
navigation of 1_3_quick_check.py, so you can browse tracks across backends and
create/save/load trajectories for each one without restarting.

Usage:
    python scripts/1_3_cam_traj_tracks.py --port 8082
"""

import os
import re
import glob
import time
import shutil
import argparse
import json
import colorsys
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

try:
    import splines
    import splines.quaternion
    HAS_SPLINES = True
except ImportError:
    print("Warning: splines library not found. Install: pip install splines")
    HAS_SPLINES = False


@dataclass
class Keyframe:
    timestep: int
    position: np.ndarray
    wxyz: np.ndarray
    fov: float
    aspect: float


def transform_to_z_up(points: np.ndarray) -> np.ndarray:
    """y-down -> z-up: (x,y,z) -> (x,z,-y)"""
    points_flat = points.reshape(-1, 3)
    return np.column_stack([points_flat[:, 0], points_flat[:, 2], -points_flat[:, 1]]).reshape(points.shape)


def transform_rotation_to_z_up(R: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64) @ R


def transform_from_z_up(points: np.ndarray) -> np.ndarray:
    """z-up -> y-down: (x,y,z) -> (x,-z,y)"""
    points_flat = points.reshape(-1, 3)
    return np.column_stack([points_flat[:, 0], -points_flat[:, 2], points_flat[:, 1]]).reshape(points.shape)


def transform_rotation_from_z_up(R: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64) @ R


def parse_video(seq_name: str) -> str:
    m = re.match(r"^(.+)_\d+_\d+_seq\d+$", seq_name)
    if m:
        return m.group(1)
    parts = seq_name.rsplit("_", 1)
    if len(parts) == 2 and re.match(r"^seq\d+$", parts[1]):
        return parts[0]
    return seq_name


BACKEND_VIPE_LYRA = "VIPE+LyRA"
BACKEND_VIPE_LYRA_NOOPT = "VIPE+LyRA+NoOpt"
BACKEND_VIPE_DEFAULT_NOOPT = "VIPE+Default+NoOpt"
VIDEO_ALL = "(all)"

DECISION_PENDING = "Pending"
DECISION_KEEP = "Keep"
DECISION_REMOVE = "Remove"


def load_decisions(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] failed to read {path}: {e}; starting fresh")
        return {}


def save_decisions_atomic(path: str, data: Dict[str, Dict]):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def decision_label(entry: Optional[Dict]) -> str:
    if entry is None:
        return DECISION_PENDING
    if entry.get("remove"):
        return DECISION_REMOVE
    if entry.get("keep"):
        return DECISION_KEEP
    return DECISION_PENDING


def parse_args():
    parser = argparse.ArgumentParser(description="Per-track camera trajectory editor")
    parser.add_argument("--vipe_lyra_root", type=str, default="outputs/prepared_vipe_lyra")
    parser.add_argument("--vipe_lyra_noopt_root", type=str, default="outputs/prepared_vipe_lyra_noopt")
    parser.add_argument("--vipe_default_noopt_root", type=str, default="outputs/prepared_vipe_default_noopt")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--point_size", type=float, default=0.002)
    parser.add_argument("--foreground_point_size", type=float, default=0.001)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--subsample", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--show_bbox", action="store_true")
    parser.add_argument("--decisions", type=str, default="track_trajectory_decisions.json",
                       help="Path to per-track review decisions JSON (auto-saved on every change)")
    return parser.parse_args()


class PerTrackCameraTrajectoryEditor:
    """Multi-track camera trajectory editor."""

    def __init__(self, vipe_lyra_root: str, vipe_lyra_noopt_root: str, vipe_default_noopt_root: str,
                 port: int, point_size: float, foreground_point_size: float,
                 fps: float, subsample: int, num_frames: Optional[int], show_bbox: bool,
                 decisions_path: str):
        self.roots = {
            BACKEND_VIPE_LYRA: vipe_lyra_root,
            BACKEND_VIPE_LYRA_NOOPT: vipe_lyra_noopt_root,
            BACKEND_VIPE_DEFAULT_NOOPT: vipe_default_noopt_root,
        }
        self.port = port
        self.point_size = point_size
        self.foreground_point_size = foreground_point_size
        self.fps = fps
        self.subsample = subsample
        self.num_frames_limit = num_frames
        self.show_bbox = show_bbox

        self.decisions_path = decisions_path
        self.decisions: Dict[str, Dict] = load_decisions(decisions_path)
        print(f"[decisions] loaded {len(self.decisions)} entries from {decisions_path}")

        # Scan tracks across all backends
        self.seq_backends: Dict[str, List[str]] = defaultdict(list)
        for backend, root in self.roots.items():
            if root and os.path.isdir(root):
                seqs = self._scan_one_root(root)
                for s in seqs:
                    self.seq_backends[s].append(backend)
                print(f"[scan] {backend}: {len(seqs)} seqs  ({root})")
            else:
                print(f"[scan] {backend}: (root missing, skipped)")

        canonical_seqs = (
            {s for s, bs in self.seq_backends.items() if BACKEND_VIPE_LYRA_NOOPT in bs}
            if any(BACKEND_VIPE_LYRA_NOOPT in bs for bs in self.seq_backends.values())
            else set(self.seq_backends.keys())
        )
        self.all_seqs: List[str] = sorted(canonical_seqs)
        if not self.all_seqs:
            raise ValueError("No prepared sequences found in any backend")

        self.videos: List[str] = sorted(set(parse_video(s) for s in self.all_seqs))
        self.filtered_seqs: List[str] = list(self.all_seqs)
        self.current_video: str = VIDEO_ALL
        self.current_backend: str = BACKEND_VIPE_LYRA_NOOPT
        self.current_seq: Optional[str] = None
        self.data_dir: Optional[str] = None
        self.image_width, self.image_height = 640, 480
        self._suppress_cb = False

        # Per-track reset state
        self.background_data: Optional[Dict] = None
        self.camera_data: Optional[Dict] = None
        self.intrinsics_frame0: Optional[np.ndarray] = None
        self.foreground_frames: Dict[str, List[Dict]] = {'single_view': [], 'partial': [], 'aligned': [], 'smooth': []}
        self.num_frames = 0
        self.unified_fov = np.deg2rad(60.0)
        self.unified_aspect = self.image_width / self.image_height

        self.keyframes: Dict[int, Keyframe] = {}
        self.keyframe_counter = 0
        self.keyframe_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.trajectory_spline_position = None
        self.trajectory_spline_orientation = None
        self.trajectory_spline_timestep = None
        self.trajectory_handles: List[viser.SceneNodeHandle] = []
        self.trajectory_camera_handle: Optional[viser.CameraFrustumHandle] = None
        self.scene_center = None

        self.arc_preview_handles: List[viser.SceneNodeHandle] = []
        self.arc_params = {
            'yaw': {'center': None, 'base_radius': None, 'start_angle': None},
            'pitch': {'center': None, 'base_radius': None, 'start_angle': None},
            'roll': {'center': None, 'base_radius': None, 'start_angle': None}
        }

        self.available_trajectories: Dict[str, Dict] = {}
        # All widgets in the "Saved Trajectories" section (markdown + checkboxes + buttons).
        # Rebuilt from scratch on every track switch.
        self.trajectory_gui_widgets: List = []
        self.trajectory_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
            (76, 114, 176), (85, 168, 104), (255, 165, 0), (128, 0, 128),
        ]
        self.trajectory_color_order: List[str] = []

        # Scene handles (cleared between tracks)
        self.background_handle = None
        self.foreground_handles = {'single_view': [], 'partial': [], 'aligned': [], 'smooth': []}
        self.camera_handles: List[Dict] = []
        self.camera_points_handle = None
        self.original_trajectory_handle = None
        self.axes_handle = None
        self.bbox_handle = None

        self.solid_colors = {
            'single_view': np.array([255, 165, 0], dtype=np.uint8),
            'partial': np.array([255, 200, 50], dtype=np.uint8),
            'aligned': np.array([100, 255, 100], dtype=np.uint8),
            'smooth': np.array([100, 150, 255], dtype=np.uint8)
        }

        self.is_playing = False

        self.server = viser.ViserServer(host="0.0.0.0", port=self.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self._create_gui(self.server)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self._set_client_camera(client)

        # Load first track
        self._load_seq(self.filtered_seqs[0])

    # ========================================================================
    # Track scanning
    # ========================================================================

    def _scan_one_root(self, root: str) -> List[str]:
        seqs = []
        for name in sorted(os.listdir(root)):
            seq_dir = os.path.join(root, name)
            if not os.path.isdir(seq_dir):
                continue
            if not (os.path.exists(os.path.join(seq_dir, "global_background.ply")) and
                    os.path.exists(os.path.join(seq_dir, "global_camera.json"))):
                continue
            seqs.append(name)
        return seqs

    def _available_backends_for(self, seq_name: str) -> List[str]:
        return self.seq_backends.get(seq_name, [])

    def _apply_video_filter(self):
        if self.current_video == VIDEO_ALL:
            self.filtered_seqs = list(self.all_seqs)
        else:
            self.filtered_seqs = [s for s in self.all_seqs if parse_video(s) == self.current_video]

    # ========================================================================
    # Data loading
    # ========================================================================

    def _load_global_background(self) -> Optional[Dict]:
        bg_path = os.path.join(self.data_dir, "global_background.ply")
        if not os.path.exists(bg_path):
            return None
        try:
            mesh = trimesh.load(bg_path)
            points = np.array(mesh.vertices)
            colors = (np.array(mesh.visual.vertex_colors)[:, :3] if hasattr(mesh, 'visual')
                     and hasattr(mesh.visual, 'vertex_colors')
                     else np.ones((len(points), 3), dtype=np.uint8) * 150)
            if self.subsample > 1:
                indices = np.arange(0, len(points), self.subsample)
                points, colors = points[indices], colors[indices]
            points = transform_to_z_up(points)
            return {'points': points, 'colors': colors}
        except Exception as e:
            print(f"Failed to load background: {e}")
            return None

    def _load_camera_parameters(self) -> Optional[Dict]:
        camera_json_path = os.path.join(self.data_dir, "global_camera.json")
        if not os.path.exists(camera_json_path):
            return None
        try:
            with open(camera_json_path, 'r') as f:
                camera_data = json.load(f)
            if "image_size" in camera_data:
                self.image_height, self.image_width = camera_data["image_size"]
            camera_data_processed = {}
            for frame_id, cam_info in camera_data.items():
                if not isinstance(cam_info, dict) or "extrinsic" not in cam_info:
                    continue
                T_cam_world = np.vstack([np.array(cam_info['extrinsic']), [0, 0, 0, 1]])
                T_world_cam = np.linalg.inv(T_cam_world)
                position_old, R_old = T_world_cam[:3, 3], T_world_cam[:3, :3]
                position_new = transform_to_z_up(position_old.reshape(1, 3)).flatten()
                R_new = transform_rotation_to_z_up(R_old)
                T_world_cam_new = np.eye(4)
                T_world_cam_new[:3, :3] = R_new
                T_world_cam_new[:3, 3] = position_new
                camera_data_processed[frame_id] = {
                    'extrinsic': np.linalg.inv(T_world_cam_new)[:3, :],
                    'intrinsic': np.array(cam_info['intrinsic']),
                    'position': position_new,
                    'R_world_cam': R_new
                }
            return camera_data_processed
        except Exception as e:
            print(f"Failed to load camera parameters: {e}")
            return None

    def _extract_frame0_intrinsics(self) -> Optional[np.ndarray]:
        if not self.camera_data or '00000' not in self.camera_data:
            self.unified_fov = np.deg2rad(60.0)
            self.unified_aspect = self.image_width / self.image_height
            return None
        intrinsics = self.camera_data['00000']['intrinsic']
        self.unified_fov = 2 * np.arctan(self.image_height / (2 * intrinsics[1, 1]))
        self.unified_aspect = self.image_width / self.image_height
        return intrinsics

    def _load_aligned_foreground_frames(self) -> Dict[str, List[Dict]]:
        subdirs = sorted([d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()])
        if self.num_frames_limit:
            subdirs = subdirs[:self.num_frames_limit]

        foreground_dict = {'single_view': [], 'partial': [], 'aligned': [], 'smooth': []}
        patterns = {
            'single_view': '{frame_id}_foreground_1_view.ply',
            'partial': '{frame_id}_foreground_5_views.ply',
            'aligned': '{frame_id}_foreground_5_views_aligned.ply',
            'smooth': '{frame_id}_foreground_5_views_aligned_smooth.ply'
        }

        for frame_id in subdirs:
            for version, pattern in patterns.items():
                fg_path = os.path.join(self.data_dir, frame_id, 'pointcloud',
                                      pattern.format(frame_id=frame_id))
                if os.path.exists(fg_path):
                    try:
                        mesh = trimesh.load(fg_path)
                        points = np.array(mesh.vertices)
                        colors = (np.array(mesh.visual.vertex_colors)[:, :3] if hasattr(mesh, 'visual')
                                 and hasattr(mesh.visual, 'vertex_colors')
                                 else np.ones((len(points), 3), dtype=np.uint8) * 150)
                        if self.subsample > 1:
                            indices = np.arange(0, len(points), self.subsample)
                            points, colors = points[indices], colors[indices]
                        points = transform_to_z_up(points)
                        solid_colors = np.tile(self.solid_colors[version], (len(points), 1))
                        foreground_dict[version].append({
                            'frame_id': frame_id,
                            'points': points,
                            'colors': colors,
                            'solid_colors': solid_colors
                        })
                    except Exception:
                        pass

        return foreground_dict

    def _scan_trajectory_files(self):
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        self.available_trajectories.clear()
        self.trajectory_color_order.clear()
        if not json_files:
            return
        for traj_file in sorted(json_files):
            filename = os.path.basename(traj_file)
            try:
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                if filename == "global_camera.json":
                    converted_data = self._convert_global_camera_to_trajectory(traj_data)
                    if converted_data is None:
                        continue
                    traj_data = converted_data
                    num_frames = len(traj_data['camera_path'])
                else:
                    if 'camera_path' not in traj_data and 'keyframes' not in traj_data:
                        continue
                    if 'camera_path' in traj_data:
                        num_frames = len(traj_data['camera_path'])
                    elif 'keyframes' in traj_data:
                        num_frames = len(traj_data['keyframes'])
                    else:
                        num_frames = 0
                self.available_trajectories[filename] = {
                    'data': traj_data, 'handles': [], 'checkbox': None, 'num_frames': num_frames, 'folder': None
                }
            except Exception:
                pass

    def _convert_global_camera_to_trajectory(self, camera_data: Dict) -> Optional[Dict]:
        try:
            camera_path = []
            image_size = camera_data.get("image_size", [self.image_height, self.image_width])
            frame_ids = sorted([k for k in camera_data.keys() if k.isdigit() or k.replace('_', '').isdigit()])
            for idx, frame_id in enumerate(frame_ids):
                frame_data = camera_data[frame_id]
                if 'extrinsic' in frame_data and 'intrinsic' in frame_data:
                    camera_path.append({
                        'timestep': idx, 'extrinsic': frame_data['extrinsic'],
                        'intrinsic': frame_data['intrinsic']
                    })
            if len(camera_path) == 0:
                return None
            return {'camera_path': camera_path, 'image_size': image_size}
        except Exception:
            return None

    # ========================================================================
    # GUI
    # ========================================================================

    def _create_gui(self, server: viser.ViserServer):
        first_seq = self.filtered_seqs[0]

        # --- Track navigation ---
        with server.gui.add_folder("Track"):
            self.gui_video = server.gui.add_dropdown(
                "Video",
                options=(VIDEO_ALL,) + tuple(self.videos),
                initial_value=VIDEO_ALL,
                hint="Filter tracks by source video",
            )
            self.gui_seq = server.gui.add_dropdown(
                "Track", options=tuple(self.filtered_seqs), initial_value=first_seq,
                hint="Pick which track to edit",
            )
            self.gui_prev = server.gui.add_button("<- Prev Track", color="cyan")
            self.gui_next = server.gui.add_button("Next Track ->", color="cyan")
            self.gui_prev_video = server.gui.add_button("<- Prev Video", color="yellow")
            self.gui_next_video = server.gui.add_button("Next Video ->", color="yellow")
            self.gui_totals = server.gui.add_text(
                "Totals", initial_value=f"{len(self.videos)} videos / {len(self.all_seqs)} tracks",
                disabled=True,
            )
            self.gui_info = server.gui.add_text("Info", initial_value="", disabled=True)
            self.gui_stats = server.gui.add_markdown("")

        # --- Review ---
        with server.gui.add_folder("Review"):
            self.gui_status = server.gui.add_text("Status", initial_value=DECISION_PENDING, disabled=True)
            self.gui_btn_pending = server.gui.add_button("Pending", icon=viser.Icon.CIRCLE)
            self.gui_btn_keep = server.gui.add_button("Keep", icon=viser.Icon.CHECK, color="green")
            self.gui_btn_remove = server.gui.add_button("Remove", icon=viser.Icon.TRASH, color="red")
            self.gui_btn_next_pending = server.gui.add_button(
                "Next Unreviewed ->", color="orange", icon=viser.Icon.ARROW_BIG_RIGHT,
            )
            self.gui_btn_save_decisions = server.gui.add_button(
                "Save Decisions JSON", icon=viser.Icon.DEVICE_FLOPPY,
            )
            self.gui_btn_clean = server.gui.add_button(
                "Clean Data (move Remove-marked tracks)", color="red", icon=viser.Icon.TRASH_X,
                hint="Move every track marked Remove into <outputs>/removed/",
            )

        # --- Time Control ---
        with server.gui.add_folder("Time Control"):
            # Create with max=1 (not 0) so the initial slider has a non-degenerate range;
            # _load_seq rewrites max to num_frames-1 once data is loaded.
            self.gui_timestep = server.gui.add_slider(
                "Frame", min=0, max=1, step=1, initial_value=0,
            )
            self.gui_frame_id_label = server.gui.add_text("Frame ID", initial_value="N/A", disabled=True)
            self.gui_play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
            self.gui_pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
            self.gui_framerate = server.gui.add_slider("FPS", min=1, max=30, step=0.5, initial_value=self.fps)
            self.gui_view_mode = server.gui.add_button_group(
                "Playback View", ("First Person", "Third Person"),
                hint="First Person: follow camera | Third Person: observe camera"
            )
            self.gui_view_mode.value = "First Person"

        self.gui_reset_camera = server.gui.add_button("Reset View", icon=viser.Icon.VIEWFINDER)

        # --- Arc generator ---
        with server.gui.add_folder("Arc Trajectory Generator"):
            self.gui_arc_radius_scale = server.gui.add_slider(
                "Radius Scale", min=0.3, max=3.0, step=0.1, initial_value=1.0,
            )
            self.gui_arc_yaw_enabled = server.gui.add_checkbox("Yaw (Horizontal Orbit)", initial_value=True)
            self.gui_arc_yaw_angle = server.gui.add_slider(
                "Yaw Angle", min=-360, max=360, step=10.0, initial_value=-120,
            )
            self.gui_arc_pitch_enabled = server.gui.add_checkbox("Pitch (Vertical Tilt)", initial_value=False)
            self.gui_arc_pitch_angle = server.gui.add_slider(
                "Pitch Angle", min=-360, max=360, step=10.0, initial_value=0,
            )
            self.gui_arc_roll_enabled = server.gui.add_checkbox("Roll (Lateral Roll)", initial_value=False)
            self.gui_arc_roll_angle = server.gui.add_slider(
                "Roll Angle", min=-360, max=360, step=10.0, initial_value=0,
            )
            self.gui_arc_num_keyframes = server.gui.add_slider(
                "Keyframes", min=4, max=24, step=1.0, initial_value=8
            )
            self.gui_arc_generate = server.gui.add_button("Generate Trajectory", color="green", icon=viser.Icon.CIRCLES)

        # --- Save ---
        with server.gui.add_folder("Save Trajectory"):
            self.gui_trajectory_filename = server.gui.add_text("Filename", initial_value="camera_trajectory.json")
            self.gui_save_trajectory = server.gui.add_button("Save", color="blue", icon=viser.Icon.FILE_EXPORT)
            self.gui_show_trajectory = server.gui.add_checkbox("Show Trajectory", initial_value=True)

        # --- Display Control ---
        self.display_control_folder = server.gui.add_folder("Display Control")
        with self.display_control_folder:
            self.gui_show_background = server.gui.add_checkbox("Background", True)
            self.gui_show_foreground = server.gui.add_checkbox("Foreground (5-view)", True)
            self.gui_show_single_view_fg = server.gui.add_checkbox("Single-View Foreground", False)
            self.gui_show_camera = server.gui.add_checkbox("Original Camera", True)
            self.gui_show_axes = server.gui.add_checkbox("Axes", True)

        # Saved trajectories list (outside Display Control; dynamically populated)
        self.gui_trajectory_markdown = None
        self.gui_clear_all_trajectories = None
        self.gui_delete_trajectory = None

        self.gui_show_original_trajectory = type('obj', (object,), {'value': False})()
        self.gui_rainbow_trajectories = type('obj', (object,), {'value': False})()
        self.server = server

        self._bind_events()

    def _bind_events(self):
        # Track nav
        @self.gui_video.on_update
        def _(_):
            if self._suppress_cb:
                return
            new_cat = self.gui_video.value
            if new_cat == self.current_video:
                return
            self.current_video = new_cat
            self._apply_video_filter()
            if not self.filtered_seqs:
                return
            self._suppress_cb = True
            self.gui_seq.options = tuple(self.filtered_seqs)
            self.gui_seq.value = self.filtered_seqs[0]
            self._suppress_cb = False
            self._load_seq(self.filtered_seqs[0])

        @self.gui_seq.on_update
        def _(_):
            if self._suppress_cb:
                return
            target = self.gui_seq.value
            if target and target != self.current_seq:
                self._load_seq(target)

        @self.gui_prev.on_click
        def _(_):
            idx = self.filtered_seqs.index(self.current_seq) if self.current_seq in self.filtered_seqs else 0
            new_idx = (idx - 1) % len(self.filtered_seqs)
            self.gui_seq.value = self.filtered_seqs[new_idx]

        @self.gui_next.on_click
        def _(_):
            idx = self.filtered_seqs.index(self.current_seq) if self.current_seq in self.filtered_seqs else 0
            new_idx = (idx + 1) % len(self.filtered_seqs)
            self.gui_seq.value = self.filtered_seqs[new_idx]

        def _jump_video(direction: int):
            if not self.videos:
                return
            cur = parse_video(self.current_seq) if self.current_seq else None
            try:
                idx = self.videos.index(cur)
            except (ValueError, TypeError):
                idx = 0
            new_idx = (idx + direction) % len(self.videos)
            target_video = self.videos[new_idx]
            if self.gui_video.value != target_video:
                self.gui_video.value = target_video
            else:
                first = next((s for s in self.all_seqs if parse_video(s) == target_video), None)
                if first:
                    self.gui_seq.value = first

        @self.gui_prev_video.on_click
        def _(_):
            _jump_video(-1)

        @self.gui_next_video.on_click
        def _(_):
            _jump_video(+1)

        # Time / display
        @self.gui_timestep.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_play_button.on_click
        def _(_) -> None:
            self.is_playing = True
            self.gui_play_button.visible = False
            self.gui_pause_button.visible = True
            self.gui_timestep.disabled = True

        @self.gui_pause_button.on_click
        def _(_) -> None:
            self.is_playing = False
            self.gui_play_button.visible = True
            self.gui_pause_button.visible = False
            self.gui_timestep.disabled = False

        for cb in [self.gui_show_background, self.gui_show_foreground,
                   self.gui_show_single_view_fg, self.gui_show_camera, self.gui_show_axes]:
            @cb.on_update
            def _(_):
                self._update_display()

        @self.gui_reset_camera.on_click
        def _(event: viser.GuiEvent) -> None:
            self._reset_camera_callback(event)

        @self.gui_view_mode.on_click
        def _(_) -> None:
            if self.is_playing and len(self.keyframes) >= 2:
                self._update_camera_along_trajectory(self.gui_timestep.value)

        @self.gui_save_trajectory.on_click
        def _(event: viser.GuiEvent) -> None:
            self._save_trajectory_callback(event)

        @self.gui_show_trajectory.on_update
        def _(_) -> None:
            self._toggle_trajectory_visibility()

        for gui_elem in [self.gui_arc_radius_scale, self.gui_arc_yaw_enabled, self.gui_arc_yaw_angle,
                         self.gui_arc_pitch_enabled, self.gui_arc_pitch_angle,
                         self.gui_arc_roll_enabled, self.gui_arc_roll_angle]:
            @gui_elem.on_update
            def _(_) -> None:
                self._update_arc_info()

        @self.gui_arc_generate.on_click
        def _(event: viser.GuiEvent) -> None:
            self._generate_arc_trajectory_callback(event)

        # --- Review callbacks ---
        def _decide_and_advance(label: str):
            self._set_decision(label)
            idx = self.filtered_seqs.index(self.current_seq) if self.current_seq in self.filtered_seqs else 0
            new_idx = (idx + 1) % len(self.filtered_seqs)
            self.gui_seq.value = self.filtered_seqs[new_idx]

        @self.gui_btn_pending.on_click
        def _(_):
            _decide_and_advance(DECISION_PENDING)

        @self.gui_btn_keep.on_click
        def _(_):
            _decide_and_advance(DECISION_KEEP)

        @self.gui_btn_remove.on_click
        def _(_):
            _decide_and_advance(DECISION_REMOVE)

        @self.gui_btn_next_pending.on_click
        def _(_):
            target = self._find_next_pending()
            if target is None:
                print("[review] no pending tracks left in current filter")
                return
            self.gui_seq.value = target

        @self.gui_btn_save_decisions.on_click
        def _(_):
            try:
                save_decisions_atomic(self.decisions_path, self.decisions)
                print(f"[save] manually saved {len(self.decisions)} entries to {self.decisions_path}")
            except Exception as e:
                print(f"[!] manual save failed: {e}")

        @self.gui_btn_clean.on_click
        def _(event: viser.GuiEvent):
            self._prompt_clean_data(event.client)

    # ========================================================================
    # Track load / clear
    # ========================================================================

    def _clear_all_scene(self):
        """Remove every scene node and reset per-track state."""
        with self.server.atomic():
            # Background / axes / original trajectory
            for h in [self.background_handle, self.axes_handle, self.bbox_handle,
                      self.camera_points_handle, self.original_trajectory_handle]:
                if h is not None:
                    try: h.remove()
                    except Exception: pass
            self.background_handle = None
            self.axes_handle = None
            self.bbox_handle = None
            self.camera_points_handle = None
            self.original_trajectory_handle = None

            # Foreground clouds
            for version in ['single_view', 'partial', 'aligned', 'smooth']:
                for h in self.foreground_handles[version]:
                    try: h.remove()
                    except Exception: pass
                self.foreground_handles[version] = []

            # Original cameras
            for ch in self.camera_handles:
                try: ch['handle'].remove()
                except Exception: pass
            self.camera_handles = []

            # Keyframe frustums
            for h in self.keyframe_handles.values():
                try: h.remove()
                except Exception: pass
            self.keyframe_handles.clear()
            self.keyframes.clear()
            self.keyframe_counter = 0

            # Generated trajectory
            for h in self.trajectory_handles:
                try: h.remove()
                except Exception: pass
            self.trajectory_handles.clear()
            if self.trajectory_camera_handle is not None:
                try: self.trajectory_camera_handle.remove()
                except Exception: pass
                self.trajectory_camera_handle = None

            # Arc preview (circle center etc.)
            for h in self.arc_preview_handles:
                try: h.remove()
                except Exception: pass
            self.arc_preview_handles.clear()

            # Saved trajectory visualizations
            for filename, traj_info in list(self.available_trajectories.items()):
                for h in traj_info.get('handles', []):
                    try: h.remove()
                    except Exception: pass

        # Spline state
        self.trajectory_spline_position = None
        self.trajectory_spline_orientation = None
        self.trajectory_spline_timestep = None

        # Remove all saved-trajectories GUI widgets (tracked via single list).
        for w in self.trajectory_gui_widgets:
            try: w.remove()
            except Exception: pass
        self.trajectory_gui_widgets = []
        self.gui_trajectory_markdown = None
        self.gui_delete_trajectory = None
        self.gui_clear_all_trajectories = None

    def _load_seq(self, seq_name: str):
        avail = self._available_backends_for(seq_name)
        if not avail:
            print(f"[!] {seq_name} has no prepared data in any backend")
            self.gui_info.value = f"ERROR: no data for {seq_name}"
            return
        if self.current_backend not in avail:
            self.current_backend = avail[0]

        self._clear_all_scene()

        self.current_seq = seq_name
        self.data_dir = os.path.join(self.roots[self.current_backend], seq_name)

        t0 = time.time()
        self.background_data = self._load_global_background()
        self.camera_data = self._load_camera_parameters()
        self.intrinsics_frame0 = self._extract_frame0_intrinsics()
        self.foreground_frames = self._load_aligned_foreground_frames()
        self.num_frames = max([len(v) for v in self.foreground_frames.values()]) if self.foreground_frames else 0

        if self.num_frames == 0 and self.background_data is None:
            print(f"[!] {seq_name}: no valid point cloud files")
            return

        self._scan_trajectory_files()

        self._create_scene()
        self._setup_initial_camera_for_track()
        self._add_initial_keyframe()
        self._compute_arc_parameters()
        self._rebuild_trajectory_list_gui()

        # Refresh time slider — keep max >= 1 so the slider is never degenerate
        self._suppress_cb = True
        self.gui_timestep.max = int(max(self.num_frames - 1, 1))
        self.gui_timestep.value = 0
        self._suppress_cb = False

        first_frame_id = "N/A"
        for version in ['smooth', 'aligned', 'single_view']:
            if len(self.foreground_frames[version]) > 0:
                first_frame_id = self.foreground_frames[version][0]['frame_id']
                break
        self.gui_frame_id_label.value = first_frame_id

        idx_str = ""
        if self.current_seq in self.filtered_seqs:
            idx = self.filtered_seqs.index(self.current_seq)
            idx_str = f" {idx + 1}/{len(self.filtered_seqs)}"
        self.gui_info.value = (
            f"bg={len(self.background_data['points']) if self.background_data else 0} | "
            f"fg={self.num_frames} | cams={len(self.camera_data) if self.camera_data else 0}"
            f"{idx_str}"
        )

        self._update_display()
        self._recenter_client_camera()
        self._populate_decision_gui(seq_name)
        elapsed = time.time() - t0
        print(f"[load] {seq_name} ({self.current_backend})  ({elapsed:.2f}s)")

    # ========================================================================
    # Review / stats
    # ========================================================================

    def _count_track_trajectories(self, seq_name: str) -> int:
        """Return the number of saved non-global_camera.json trajectories under the
        canonical track directory (uses VIPE+LyRA+NoOpt by convention)."""
        root = self.roots.get(BACKEND_VIPE_LYRA_NOOPT) or next(iter(self.roots.values()))
        if not root:
            return 0
        seq_dir = os.path.join(root, seq_name)
        if not os.path.isdir(seq_dir):
            return 0
        return len([f for f in glob.glob(os.path.join(seq_dir, "*.json"))
                    if os.path.basename(f) != "global_camera.json"])

    def _find_next_pending(self) -> Optional[str]:
        if not self.filtered_seqs:
            return None
        start = self.filtered_seqs.index(self.current_seq) if self.current_seq in self.filtered_seqs else 0
        n = len(self.filtered_seqs)
        for step in range(1, n + 1):
            cand = self.filtered_seqs[(start + step) % n]
            if cand not in self.decisions:
                return cand
        return None

    def _populate_decision_gui(self, seq_name: str):
        entry = self.decisions.get(seq_name)
        self.gui_status.value = decision_label(entry)
        self._refresh_stats()

    def _set_decision(self, label: str):
        if not self.current_seq:
            return
        if label == DECISION_PENDING:
            if self.current_seq in self.decisions:
                del self.decisions[self.current_seq]
        else:
            self.decisions[self.current_seq] = {
                "keep": label == DECISION_KEEP,
                "remove": label == DECISION_REMOVE,
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            }
        self.gui_status.value = label
        try:
            save_decisions_atomic(self.decisions_path, self.decisions)
            print(f"[review] {self.current_seq} -> {label}")
        except Exception as e:
            print(f"[!] failed to save decisions: {e}")
        self._refresh_stats()

    def _prompt_clean_data(self, client):
        """Show a confirmation modal listing the tracks to be moved."""
        targets = sorted([t for t, dec in self.decisions.items() if dec.get("remove")])
        if not targets:
            if client is not None:
                with client.gui.add_modal("Nothing to clean") as modal:
                    client.gui.add_markdown("No tracks are marked **Remove**.")
                    close_button = client.gui.add_button("OK")
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()
            return
        canonical = os.path.basename(self.roots[BACKEND_VIPE_LYRA_NOOPT].rstrip("/"))
        preview = "\n".join(f"- `{t}`" for t in targets[:20])
        if len(targets) > 20:
            preview += f"\n- ...  (+{len(targets) - 20} more)"
        if client is None:
            self._run_clean_data()
            return
        with client.gui.add_modal("Confirm Clean Data") as modal:
            client.gui.add_markdown(
                f"**Move {len(targets)} track(s) to `removed/{canonical}/`**\n\n"
                f"{preview}\n\n"
                "This only touches `" + canonical + "/<track>/`. Source data and other "
                "backends are untouched. Proceed?"
            )
            btn_cancel = client.gui.add_button("Cancel")
            btn_ok = client.gui.add_button("Move", color="red", icon=viser.Icon.TRASH_X)
            @btn_cancel.on_click
            def _(_):
                modal.close()
            @btn_ok.on_click
            def _(_):
                modal.close()
                self._run_clean_data()

    def _run_clean_data(self):
        """Move every track marked Remove to `<outputs>/removed/<canonical>/<track>/`.
        Only the canonical backend (VIPE+LyRA+NoOpt) is touched."""
        canonical_root = self.roots.get(BACKEND_VIPE_LYRA_NOOPT)
        if not canonical_root:
            print("[clean] no canonical root configured; aborting")
            return
        outputs_root = os.path.dirname(os.path.abspath(canonical_root))
        canonical_name = os.path.basename(canonical_root.rstrip("/"))

        moved = errors = missing = 0
        moved_seqs: List[str] = []
        for t, dec in list(self.decisions.items()):
            if not dec.get("remove"):
                continue
            src = os.path.join(outputs_root, canonical_name, t)
            dst = os.path.join(outputs_root, "removed", canonical_name, t)
            if not os.path.lexists(src):
                missing += 1
                continue
            if os.path.lexists(dst):
                print(f"  [!] dst already exists, skipping: {dst}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(src, dst)
                moved += 1
                moved_seqs.append(t)
            except Exception as e:
                errors += 1
                print(f"  [!] move failed: {src} -> {e}")

        print(f"[clean] done: {moved} moved, {missing} not found, {errors} errors")

        # Refresh the track list so removed tracks disappear from the GUI.
        self._rescan_tracks_after_clean(moved_seqs)

    def _rescan_tracks_after_clean(self, moved_seqs: List[str]):
        """Re-scan roots, refresh dropdowns + stats, and if the current track was
        moved, jump to the next available one."""
        self.seq_backends.clear()
        for backend, root in self.roots.items():
            if root and os.path.isdir(root):
                for s in self._scan_one_root(root):
                    self.seq_backends[s].append(backend)
        canonical_seqs = (
            {s for s, bs in self.seq_backends.items() if BACKEND_VIPE_LYRA_NOOPT in bs}
            if any(BACKEND_VIPE_LYRA_NOOPT in bs for bs in self.seq_backends.values())
            else set(self.seq_backends.keys())
        )
        self.all_seqs = sorted(canonical_seqs)
        self.videos = sorted(set(parse_video(s) for s in self.all_seqs))
        self._apply_video_filter()

        self._suppress_cb = True
        self.gui_video.options = (VIDEO_ALL,) + tuple(self.videos)
        if self.current_video != VIDEO_ALL and self.current_video not in self.videos:
            self.current_video = VIDEO_ALL
            self.gui_video.value = VIDEO_ALL
            self._apply_video_filter()
        if self.filtered_seqs:
            self.gui_seq.options = tuple(self.filtered_seqs)
            if self.current_seq not in self.filtered_seqs:
                self.gui_seq.value = self.filtered_seqs[0]
        else:
            self.gui_seq.options = ("",)
        self.gui_totals.value = f"{len(self.videos)} videos / {len(self.all_seqs)} tracks"
        self._suppress_cb = False

        # If current track was moved, load a new one.
        if self.current_seq in moved_seqs and self.filtered_seqs:
            self._load_seq(self.filtered_seqs[0])
        else:
            self._refresh_stats()

    def _refresh_stats(self):
        """Render a per-video breakdown of tracks, each with a K/R/. marker and
        trajectory-count (T=N). Highlights the currently-selected track with *."""
        seqs = self.filtered_seqs
        n = len(seqs)
        n_keep = n_remove = 0
        for s in seqs:
            d = self.decisions.get(s, {})
            if d.get("remove"):
                n_remove += 1
            elif d.get("keep"):
                n_keep += 1
        n_pending = n - n_keep - n_remove
        lines = [f"all: K={n_keep} R={n_remove} .={n_pending} ({n})"]

        if self.current_video == VIDEO_ALL:
            # Bird's-eye: one line per video with K/R/. counts
            video_groups: Dict[str, List[str]] = {}
            for s in self.filtered_seqs:
                video_groups.setdefault(parse_video(s), []).append(s)
            cur_video = parse_video(self.current_seq) if self.current_seq else None
            lines.append("")
            lines.append(f"videos ({len(video_groups)}):")
            for v in sorted(video_groups):
                tracks = video_groups[v]
                vk = vr = 0
                for t in tracks:
                    d = self.decisions.get(t, {})
                    if d.get("remove"): vr += 1
                    elif d.get("keep"): vk += 1
                vp = len(tracks) - vk - vr
                cur = " *" if v == cur_video else ""
                lines.append(f" {v}  K={vk} R={vr} .={vp}{cur}")
        elif self.current_seq:
            # Detail: tracks within the currently-selected video
            cv = parse_video(self.current_seq)
            sibs = sorted(s for s in self.all_seqs if parse_video(s) == cv)
            if sibs:
                lines.append("")
                lines.append(f"{cv} ({len(sibs)})")
                for s in sibs:
                    lab = decision_label(self.decisions.get(s))
                    sym = {"Keep": "K", "Remove": "R"}.get(lab, ".")
                    ntraj = self._count_track_trajectories(s)
                    suf = s[len(cv):].lstrip("_") or s
                    cur = " *" if s == self.current_seq else ""
                    lines.append(f" [{sym}] T={ntraj} {suf}{cur}")
        self.gui_stats.content = "```\n" + "\n".join(lines) + "\n```"

    def _setup_initial_camera_for_track(self):
        all_points = []
        if self.background_data:
            all_points.append(self.background_data['points'])
        for version in ['single_view', 'aligned', 'smooth']:
            for fg_frame in self.foreground_frames[version]:
                all_points.append(fg_frame['points'])
        if not all_points:
            self.scene_center = np.array([0.0, 0.0, 0.0])
            return
        all_points_combined = np.concatenate(all_points)
        bbox_min, bbox_max = np.min(all_points_combined, axis=0), np.max(all_points_combined, axis=0)
        self.scene_center = (bbox_min + bbox_max) / 2.0

    def _set_client_camera(self, client: viser.ClientHandle):
        if self.scene_center is None:
            return
        if self.camera_data and '00000' in self.camera_data:
            init_pos = self.camera_data['00000']['position']
        else:
            init_pos = self.scene_center + np.array([0.0, -1.0, 0.5])
        try:
            client.camera.position = init_pos
            client.camera.look_at = self.scene_center
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])
        except Exception:
            pass

    def _recenter_client_camera(self):
        try:
            clients = self.server.get_clients()
        except Exception:
            clients = {}
        for client in clients.values():
            self._set_client_camera(client)

    # ========================================================================
    # Scene construction (background, foreground, cameras, axes)
    # ========================================================================

    def _create_scene(self):
        server = self.server
        self.background_handle = None
        if self.background_data:
            self.background_handle = server.scene.add_point_cloud(
                name="/pointcloud/global_background",
                points=self.background_data['points'],
                colors=self.background_data['colors'],
                point_size=self.point_size, point_shape="circle",
            )

        self.foreground_handles = {'single_view': [], 'partial': [], 'aligned': [], 'smooth': []}
        for version in ['single_view', 'aligned', 'smooth']:
            for pc_data in self.foreground_frames[version]:
                frame_id = pc_data['frame_id']
                handle = server.scene.add_point_cloud(
                    name=f"/pointcloud/foreground_{version}_{frame_id}",
                    points=pc_data['points'], colors=pc_data['colors'],
                    point_size=self.foreground_point_size, point_shape="circle",
                )
                self.foreground_handles[version].append(handle)

        self.camera_handles = []
        self.camera_points_handle = None
        if self.camera_data:
            self._create_cameras(server)

        self.axes_handle = None
        self._create_coordinate_axes(server)

    def _create_coordinate_axes(self, server: viser.ViserServer):
        all_points = []
        if self.background_data:
            all_points.append(self.background_data['points'])
        for version in ['single_view', 'aligned', 'smooth']:
            all_points.extend([fg['points'] for fg in self.foreground_frames[version]])
        axis_scale = 0.5 if not all_points else np.max(np.max(np.concatenate(all_points), axis=0) -
                                                       np.min(np.concatenate(all_points), axis=0)) * 0.1
        origin = np.array([0.0, 0.0, 0.0])
        axis_points, axis_colors = [], []
        for axis_vec, color in [([axis_scale, 0, 0], [255, 0, 0]),
                                ([0, axis_scale, 0], [0, 255, 0]),
                                ([0, 0, axis_scale], [0, 0, 255])]:
            for t in np.linspace(0, 1, 50):
                axis_points.append(origin + t * np.array(axis_vec))
                axis_colors.append(color)
        axis_points.append(origin)
        axis_colors.append([255, 255, 255])
        self.axes_handle = server.scene.add_point_cloud(
            "/reference/coordinate_axes", points=np.array(axis_points), colors=np.array(axis_colors),
            point_size=self.point_size * 3.0, point_shape="circle")

    def _create_cameras(self, server: viser.ViserServer):
        if not self.camera_data:
            return
        camera_color = (196, 78, 82)
        frame_ids_sorted = sorted(self.camera_data.keys())
        camera_positions = []
        for frame_id in frame_ids_sorted:
            try:
                cam_info = self.camera_data[frame_id]
                wxyz = Rotation.from_matrix(cam_info['R_world_cam']).as_quat()[[3, 0, 1, 2]]
                fov_y = 2 * np.arctan(self.image_height / (2 * cam_info['intrinsic'][1, 1]))
                aspect = self.image_width / self.image_height
                handle = server.scene.add_camera_frustum(
                    f"/camera/original_frame_{frame_id}", fov=fov_y, aspect=aspect, scale=0.018,
                    wxyz=wxyz, position=cam_info['position'], color=camera_color)
                self.camera_handles.append({'handle': handle, 'frame_id': frame_id})
                camera_positions.append(cam_info['position'])
            except Exception:
                pass
        if len(camera_positions) > 0:
            camera_positions = np.array(camera_positions)
            camera_colors = np.tile(camera_color, (len(camera_positions), 1))
            self.camera_points_handle = server.scene.add_point_cloud(
                "/camera/original_positions", points=camera_positions,
                colors=camera_colors, point_size=0.018, point_shape="circle"
            )
            self.original_trajectory_handle = None
            if len(camera_positions) >= 2:
                try:
                    self.original_trajectory_handle = server.scene.add_spline_catmull_rom(
                        "/camera/original_trajectory", positions=camera_positions,
                        color=camera_color, line_width=2.0, segments=len(camera_positions) * 2
                    )
                    self.original_trajectory_handle.visible = False
                except Exception:
                    pass

    def _update_display(self):
        # Guard against viser sending NaN/float during drag on degenerate sliders.
        raw_ts = self.gui_timestep.value
        try:
            current_timestep = int(raw_ts)
        except (TypeError, ValueError):
            current_timestep = 0
        show_background = self.gui_show_background.value
        show_foreground = self.gui_show_foreground.value
        show_single_view_fg = self.gui_show_single_view_fg.value
        show_camera = self.gui_show_camera.value
        show_axes = self.gui_show_axes.value

        current_frame_id = None
        if current_timestep < len(self.foreground_frames['smooth']):
            current_frame_id = self.foreground_frames['smooth'][current_timestep]['frame_id']

        if current_frame_id:
            self.gui_frame_id_label.value = current_frame_id

        with self.server.atomic():
            if self.background_handle:
                self.background_handle.visible = show_background
            for i, handle in enumerate(self.foreground_handles['single_view']):
                handle.visible = (show_single_view_fg and i == current_timestep)
            for handle in self.foreground_handles['aligned']:
                handle.visible = False
            for i, handle in enumerate(self.foreground_handles['smooth']):
                handle.visible = (show_foreground and i == current_timestep)
            if self.camera_handles and current_frame_id:
                for cam_info in self.camera_handles:
                    if not show_camera:
                        cam_info['handle'].visible = False
                    else:
                        cam_info['handle'].visible = (cam_info['frame_id'] == current_frame_id)
            if self.camera_points_handle:
                self.camera_points_handle.visible = show_camera
            if self.axes_handle:
                self.axes_handle.visible = show_axes

    # ========================================================================
    # Keyframe visualization
    # ========================================================================

    def _add_initial_keyframe(self):
        if not self.camera_data or '00000' not in self.camera_data:
            return
        try:
            cam_info = self.camera_data['00000']
            wxyz = Rotation.from_matrix(cam_info['R_world_cam']).as_quat()[[3, 0, 1, 2]]
            fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
            aspect = self.unified_aspect if self.unified_aspect is not None else (self.image_width / self.image_height)
            keyframe = Keyframe(timestep=0, position=cam_info['position'], wxyz=wxyz, fov=fov, aspect=aspect)
            self.keyframes[0] = keyframe
            self.keyframe_counter = 1
            self._visualize_keyframe(0, keyframe)
        except Exception:
            pass

    def _visualize_keyframe(self, keyframe_id: int, keyframe: Keyframe):
        handle = self.server.scene.add_camera_frustum(
            name=f"/keyframe/keyframe_{keyframe_id}",
            fov=keyframe.fov, aspect=keyframe.aspect, scale=0.05,
            wxyz=keyframe.wxyz, position=keyframe.position, color=(200, 10, 30),
        )
        self.keyframe_handles[keyframe_id] = handle

    def _reset_camera_callback(self, event: viser.GuiEvent):
        if not event.client or self.scene_center is None:
            return
        event.client.camera.look_at = self.scene_center
        event.client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def _clear_all_keyframes(self):
        frame0_keyframe = next((kf for kf in self.keyframes.values() if kf.timestep == 0), None)
        for handle in self.keyframe_handles.values():
            try: handle.remove()
            except Exception: pass
        self.keyframes.clear()
        self.keyframe_handles.clear()
        if frame0_keyframe:
            self.keyframe_counter = 0
            self.keyframes[0] = frame0_keyframe
            self.keyframe_counter = 1
            self._visualize_keyframe(0, frame0_keyframe)
        else:
            self.keyframe_counter = 0
        self._clear_trajectory()

    # ========================================================================
    # Trajectory interpolation and visualization
    # ========================================================================

    def _update_trajectory(self):
        if not HAS_SPLINES:
            return
        if len(self.keyframes) < 2:
            self._clear_trajectory()
            return
        keyframes_sorted = sorted(self.keyframes.values(), key=lambda x: x.timestep)
        positions = np.array([kf.position for kf in keyframes_sorted])
        wxyzs = np.array([kf.wxyz for kf in keyframes_sorted])
        timesteps = np.array([kf.timestep for kf in keyframes_sorted])
        tension = 0.0
        try:
            self.trajectory_spline_position = splines.KochanekBartels(
                positions.tolist(), tcb=(tension, 0.0, 0.0), endconditions="natural")
            quaternions = [splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(wxyz, -1)) for wxyz in wxyzs]
            self.trajectory_spline_orientation = splines.quaternion.KochanekBartels(
                quaternions, tcb=(tension, 0.0, 0.0), endconditions="natural")
            self.trajectory_spline_timestep = splines.KochanekBartels(
                timesteps.tolist(), tcb=(tension, 0.0, 0.0), endconditions="natural")
            self._visualize_trajectory()
        except Exception:
            pass

    def _toggle_trajectory_visibility(self):
        show = self.gui_show_trajectory.value
        for handle in self.trajectory_handles:
            try: handle.visible = show
            except Exception: pass
        for handle in self.keyframe_handles.values():
            try: handle.visible = show
            except Exception: pass

    def _visualize_trajectory(self):
        if not self.trajectory_spline_position:
            return
        self._clear_trajectory_visualization()
        num_samples = 100
        t_values = np.linspace(0, len(self.keyframes) - 1, num_samples)
        trajectory_positions = self.trajectory_spline_position.evaluate(t_values)
        colors_array = np.array([colorsys.hls_to_rgb(h, 0.5, 1.0)
                                for h in np.linspace(0.0, 1.0, len(trajectory_positions))])
        try:
            self.trajectory_handles.append(self.server.scene.add_spline_catmull_rom(
                "/trajectory/path", positions=trajectory_positions,
                color=(220, 220, 220), line_width=0.5, segments=num_samples + 1))
        except Exception:
            pass
        self.trajectory_handles.append(self.server.scene.add_point_cloud(
            "/trajectory/points", points=trajectory_positions, colors=colors_array,
            point_size=0.02, point_shape="circle"))

    def _clear_trajectory(self):
        self.trajectory_spline_position = None
        self.trajectory_spline_orientation = None
        self.trajectory_spline_timestep = None
        self._clear_trajectory_visualization()
        if self.trajectory_camera_handle:
            try: self.trajectory_camera_handle.remove()
            except Exception: pass
            self.trajectory_camera_handle = None

    def _clear_trajectory_visualization(self):
        for handle in self.trajectory_handles:
            try: handle.remove()
            except Exception: pass
        self.trajectory_handles.clear()

    # ========================================================================
    # Save trajectory
    # ========================================================================

    def _save_trajectory_callback(self, event: viser.GuiEvent):
        if len(self.keyframes) < 2:
            if event.client is not None:
                with event.client.gui.add_modal("Error") as modal:
                    event.client.gui.add_markdown("Need at least 2 keyframes to save trajectory")
                    close_button = event.client.gui.add_button("OK")
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()
            return
        if self.trajectory_spline_position is None or self.trajectory_spline_timestep is None:
            if event.client is not None:
                with event.client.gui.add_modal("Error") as modal:
                    event.client.gui.add_markdown("Trajectory not generated, please add keyframes first")
                    close_button = event.client.gui.add_button("OK")
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()
            return
        try:
            filename = self.gui_trajectory_filename.value
            if not filename.endswith('.json'):
                filename += '.json'
            output_path = os.path.join(self.data_dir, filename)
            file_exists = os.path.exists(output_path)
            self._save_trajectory_to_json()

            if not file_exists:
                self._scan_trajectory_files()
                self._rebuild_trajectory_list_gui()
                self._refresh_stats()
        except Exception as e:
            print(f"Failed to save trajectory: {e}")
            import traceback
            traceback.print_exc()
            if event.client is not None:
                with event.client.gui.add_modal("Error") as modal:
                    event.client.gui.add_markdown(f"Save failed:\n`{str(e)}`")
                    close_button = event.client.gui.add_button("OK")
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()

    def _save_trajectory_to_json(self):
        keyframes_sorted = sorted(self.keyframes.values(), key=lambda x: x.timestep)
        min_timestep = keyframes_sorted[0].timestep
        max_timestep = keyframes_sorted[-1].timestep

        if self.intrinsics_frame0 is None:
            raise ValueError("Frame 0 intrinsics not found")

        intrinsics_original = self.intrinsics_frame0
        original_frames = int(max_timestep - min_timestep + 1)

        trajectory_data = {
            "keyframes": [],
            "camera_path": [],
            "metadata": {
                "image_size": [self.image_height, self.image_width],
                "num_keyframes": len(self.keyframes),
                "timestep_range": [int(min_timestep), int(max_timestep)],
                "coordinate_system": "y-down (original)",
                "total_output_frames": original_frames,
            }
        }

        for keyframe_id, keyframe in sorted(self.keyframes.items(), key=lambda x: x[1].timestep):
            position_original = transform_from_z_up(keyframe.position.reshape(1, 3)).flatten()
            R_world_cam_zup = Rotation.from_quat(keyframe.wxyz[[1, 2, 3, 0]]).as_matrix()
            R_world_cam_original = transform_rotation_from_z_up(R_world_cam_zup)
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = R_world_cam_original
            T_world_cam[:3, 3] = position_original
            T_cam_world = np.linalg.inv(T_world_cam)
            extrinsic_3x4 = T_cam_world[:3, :]
            trajectory_data["keyframes"].append({
                "timestep": int(keyframe.timestep),
                "position": position_original.tolist(),
                "wxyz": keyframe.wxyz.tolist(),
                "extrinsic": extrinsic_3x4.tolist(),
                "intrinsic": intrinsics_original.tolist(),
                "fov": float(keyframe.fov),
                "aspect": float(keyframe.aspect)
            })

        def zup_to_extrinsic(pos_zup, wxyz_zup):
            pos_original = transform_from_z_up(np.array(pos_zup).reshape(1, 3)).flatten()
            R_zup = Rotation.from_quat(np.array(wxyz_zup)[[1, 2, 3, 0]]).as_matrix()
            R_original = transform_rotation_from_z_up(R_zup)
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = R_original
            T_world_cam[:3, 3] = pos_original
            T_cam_world = np.linalg.inv(T_world_cam)
            return T_cam_world[:3, :], pos_original

        num_samples = 100
        t_values = np.linspace(0, len(self.keyframes) - 1, num_samples)
        timestep_samples = self.trajectory_spline_timestep.evaluate(t_values)

        for output_frame_idx, video_time in enumerate(range(int(min_timestep), int(max_timestep) + 1)):
            idx = np.argmin(np.abs(timestep_samples - video_time))
            t = t_values[idx]
            position_zup = self.trajectory_spline_position.evaluate(t)
            quat = self.trajectory_spline_orientation.evaluate(t)
            wxyz_zup = np.array([quat.scalar, *quat.vector])
            extrinsic_3x4, position_original = zup_to_extrinsic(position_zup, wxyz_zup)
            trajectory_data["camera_path"].append({
                "output_frame": output_frame_idx,
                "video_time": video_time,
                "extrinsic": extrinsic_3x4.tolist(),
                "intrinsic": intrinsics_original.tolist(),
                "position": position_original.tolist(),
                "wxyz": wxyz_zup.tolist()
            })

        filename = self.gui_trajectory_filename.value
        if not filename.endswith('.json'):
            filename += '.json'
        output_path = os.path.join(self.data_dir, filename)
        os.makedirs(self.data_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        print(f"Trajectory saved: {output_path} ({len(trajectory_data['camera_path'])} frames)")

    def _update_camera_along_trajectory(self, current_timestep: int):
        if self.trajectory_spline_position is None or self.trajectory_spline_timestep is None:
            return
        if len(self.keyframes) < 2:
            return
        try:
            keyframes_sorted = sorted(self.keyframes.values(), key=lambda x: x.timestep)
            min_timestep = keyframes_sorted[0].timestep
            max_timestep = keyframes_sorted[-1].timestep

            if min_timestep <= current_timestep <= max_timestep:
                num_samples = 100
                t_values = np.linspace(0, len(self.keyframes) - 1, num_samples)
                timestep_samples = self.trajectory_spline_timestep.evaluate(t_values)
                idx = np.argmin(np.abs(timestep_samples - current_timestep))
                t = t_values[idx]
                position = self.trajectory_spline_position.evaluate(t)
                quat = self.trajectory_spline_orientation.evaluate(t)
                wxyz = np.array([quat.scalar, *quat.vector])
                view_mode = self.gui_view_mode.value

                if view_mode == "First Person":
                    fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
                    for client in self.server.get_clients().values():
                        with client.atomic():
                            client.camera.position = position
                            client.camera.wxyz = wxyz
                            client.camera.fov = fov
                    if self.trajectory_camera_handle is not None:
                        self.trajectory_camera_handle.visible = False
                else:
                    show_trajectory = self.gui_show_trajectory.value
                    if self.trajectory_camera_handle is None:
                        fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
                        aspect = self.unified_aspect if self.unified_aspect is not None else (self.image_width / self.image_height)
                        self.trajectory_camera_handle = self.server.scene.add_camera_frustum(
                            name="/trajectory/current_camera",
                            fov=fov, aspect=aspect, scale=0.05,
                            wxyz=wxyz, position=position, color=(10, 200, 30), visible=show_trajectory
                        )
                    else:
                        self.trajectory_camera_handle.position = position
                        self.trajectory_camera_handle.wxyz = wxyz
                        self.trajectory_camera_handle.visible = show_trajectory
        except Exception:
            pass

    # ========================================================================
    # Arc trajectory generator
    # ========================================================================

    def _compute_arc_parameters(self):
        center_depth_y = None

        first_frame_id = None
        for version in ['smooth', 'aligned', 'single_view']:
            if len(self.foreground_frames[version]) > 0:
                first_frame_id = self.foreground_frames[version][0]['frame_id']
                break

        if first_frame_id is not None:
            pointmap_path = os.path.join(self.data_dir, first_frame_id, 'pointcloud',
                                        f'{first_frame_id}_foreground_1_view.npz')
            try:
                pointmap_data = np.load(pointmap_path)
                pointmap = pointmap_data['foreground_pointmap']
                H, W = pointmap.shape[:2]
                center_h, center_w = H // 2, W // 2
                center_depth_y = pointmap[center_h, center_w, 2]
            except Exception:
                pass

        if center_depth_y is None:
            first_frame = None
            for version in ['smooth', 'aligned', 'single_view']:
                if len(self.foreground_frames[version]) > 0:
                    first_frame = self.foreground_frames[version][0]
                    break
            if first_frame is None:
                return
            points = first_frame['points']
            if len(points) == 0:
                return
            center_depth_y = np.min(points[:, 1])

        start_point = np.array([0.0, 0.0, 0.0])

        for arc_type in ['yaw', 'pitch', 'roll']:
            center = np.array([0.0, center_depth_y, 0.0])
            radius = abs(center_depth_y) if center_depth_y != 0 else 1.0
            vec_to_start = start_point - center
            if arc_type == 'yaw':
                start_angle = np.arctan2(vec_to_start[1], vec_to_start[0])
            elif arc_type == 'pitch':
                start_angle = np.arctan2(vec_to_start[2], vec_to_start[1])
            else:
                start_angle = np.arctan2(vec_to_start[2], vec_to_start[0])
            self.arc_params[arc_type] = {
                'center': center, 'base_radius': radius, 'start_angle': start_angle
            }

        self._update_arc_info()

    def _update_arc_info(self):
        if 'yaw' not in self.arc_params or self.arc_params['yaw']['center'] is None:
            return

        def _safe_int(v, default=0):
            try:
                f = float(v)
                if f != f:  # NaN check
                    return default
                return int(f)
            except (TypeError, ValueError):
                return default

        yaw = _safe_int(self.gui_arc_yaw_angle.value) if self.gui_arc_yaw_enabled.value else 0
        pitch = _safe_int(self.gui_arc_pitch_angle.value) if self.gui_arc_pitch_enabled.value else 0
        roll = _safe_int(self.gui_arc_roll_angle.value) if self.gui_arc_roll_enabled.value else 0
        try:
            scale = float(self.gui_arc_radius_scale.value)
            if scale != scale:
                scale = 1.0
        except (TypeError, ValueError):
            scale = 1.0
        scale_str = f"{scale:g}".replace('.', 'p')
        auto_filename = f"yaw_{yaw}_pitch_{pitch}_roll_{roll}_scale_{scale_str}.json"
        self.gui_trajectory_filename.value = auto_filename

    def _generate_arc_trajectory_callback(self, event: viser.GuiEvent):
        if 'yaw' not in self.arc_params or self.arc_params['yaw']['center'] is None:
            if event.client is not None:
                with event.client.gui.add_modal("Error") as modal:
                    event.client.gui.add_markdown("**Arc parameters not computed**\n\nMake sure foreground point cloud is loaded")
                    close_button = event.client.gui.add_button("OK")
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()
            return

        if not (self.gui_arc_yaw_enabled.value or self.gui_arc_pitch_enabled.value or self.gui_arc_roll_enabled.value):
            if event.client is not None:
                with event.client.gui.add_modal("Info") as modal:
                    event.client.gui.add_markdown("**Please enable at least one rotation axis**")
                    close_button = event.client.gui.add_button("OK")
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()
            return

        arc_center = self.arc_params['yaw']['center']
        arc_base_radius = self.arc_params['yaw']['base_radius']
        radius_scale = self.gui_arc_radius_scale.value
        arc_radius = arc_base_radius * radius_scale

        try:
            num_keyframes = int(self.gui_arc_num_keyframes.value)
            if np.isnan(num_keyframes):
                num_keyframes = 8
        except (ValueError, TypeError):
            num_keyframes = 8

        yaw_angle_deg = self.gui_arc_yaw_angle.value if self.gui_arc_yaw_enabled.value else 0.0
        pitch_angle_deg = self.gui_arc_pitch_angle.value if self.gui_arc_pitch_enabled.value else 0.0
        roll_angle_deg = self.gui_arc_roll_angle.value if self.gui_arc_roll_enabled.value else 0.0

        keyframes_data = self._generate_composite_arc_keyframes(
            arc_center, arc_radius,
            yaw_angle_deg, pitch_angle_deg, roll_angle_deg,
            num_keyframes
        )

        self._clear_all_keyframes()

        for i, kf_data in enumerate(keyframes_data):
            fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
            aspect = self.unified_aspect if self.unified_aspect is not None else (self.image_width / self.image_height)
            keyframe = Keyframe(timestep=kf_data['timestep'], position=kf_data['position'],
                               wxyz=kf_data['wxyz'], fov=fov, aspect=aspect)
            keyframe_id = self.keyframe_counter
            self.keyframes[keyframe_id] = keyframe
            self.keyframe_counter += 1
            self._visualize_keyframe(keyframe_id, keyframe)

        self._update_trajectory()
        self._visualize_center()

    def _generate_composite_arc_keyframes(self, arc_center: np.ndarray, arc_radius: float,
                                          yaw_angle_deg: float, pitch_angle_deg: float, roll_angle_deg: float,
                                          num_keyframes: int) -> List[Dict]:
        keyframes_data = []
        reference_R = None
        if self.camera_data and '00000' in self.camera_data:
            reference_R = self.camera_data['00000']['R_world_cam']

        start_point = np.array([0.0, 0.0, 0.0])
        base_direction = arc_center / np.linalg.norm(arc_center)
        adjusted_center = base_direction * arc_radius
        initial_vec = start_point - adjusted_center

        for i in range(num_keyframes):
            t = i / max(num_keyframes - 1, 1)

            yaw_current = np.deg2rad(yaw_angle_deg * t)
            pitch_current = np.deg2rad(pitch_angle_deg * t)
            roll_current = np.deg2rad(roll_angle_deg * t)

            cos_yaw, sin_yaw = np.cos(yaw_current), np.sin(yaw_current)
            R_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

            cos_pitch, sin_pitch = np.cos(pitch_current), np.sin(pitch_current)
            R_pitch = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])

            cos_roll, sin_roll = np.cos(roll_current), np.sin(roll_current)
            R_roll = np.array([[cos_roll, 0, sin_roll], [0, 1, 0], [-sin_roll, 0, cos_roll]])

            R_composite = R_yaw @ R_pitch @ R_roll
            rotated_vec = R_composite @ initial_vec
            position = adjusted_center + rotated_vec

            if reference_R is not None:
                R_world_cam = R_composite @ reference_R
            else:
                forward = adjusted_center - position
                forward = forward / np.linalg.norm(forward)
                world_up = np.array([0.0, 0.0, 1.0])
                right = np.cross(forward, world_up)
                if np.linalg.norm(right) < 0.01:
                    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                up = up / np.linalg.norm(up)
                R_world_cam = np.column_stack([right, up, -forward])

            wxyz = Rotation.from_matrix(R_world_cam).as_quat()[[3, 0, 1, 2]]

            if self.num_frames > 0:
                timestep = int(i * (self.num_frames - 1) / max(num_keyframes - 1, 1))
            else:
                timestep = i

            keyframes_data.append({'position': position, 'wxyz': wxyz, 'timestep': timestep})

        return keyframes_data

    def _visualize_center(self):
        self._clear_arc_preview()
        if 'yaw' not in self.arc_params or self.arc_params['yaw']['center'] is None:
            return
        arc_center = self.arc_params['yaw']['center']
        self.arc_preview_handles.append(
            self.server.scene.add_point_cloud(
                "/arc_preview/center", points=arc_center.reshape(1, 3),
                colors=np.array([[255, 0, 255]]), point_size=0.015, point_shape="circle"
            )
        )

    def _clear_arc_preview(self):
        for handle in self.arc_preview_handles:
            try: handle.remove()
            except Exception: pass
        self.arc_preview_handles.clear()

    # ========================================================================
    # Saved trajectories list (per-track)
    # ========================================================================

    def _update_trajectory_display(self, filename: str):
        if filename not in self.available_trajectories:
            return
        traj_info = self.available_trajectories[filename]
        if traj_info['checkbox'] is None:
            return
        is_visible = traj_info['checkbox'].value
        if is_visible:
            if traj_info['color'] is None:
                color_idx = len(self.trajectory_color_order) % len(self.trajectory_colors)
                traj_info['color'] = self.trajectory_colors[color_idx]
                self.trajectory_color_order.append(filename)
            if len(traj_info['handles']) == 0:
                self._visualize_saved_trajectory(filename)
            else:
                for handle in traj_info['handles']:
                    try: handle.visible = True
                    except Exception: pass
        else:
            for handle in traj_info['handles']:
                try: handle.visible = False
                except Exception: pass

    def _visualize_saved_trajectory(self, filename: str):
        if filename not in self.available_trajectories:
            return
        traj_info = self.available_trajectories[filename]
        traj_data = traj_info['data']
        color = traj_info['color']

        try:
            for handle in traj_info['handles']:
                try: handle.remove()
                except Exception: pass
            traj_info['handles'].clear()

            if 'camera_path' in traj_data and len(traj_data['camera_path']) > 0:
                camera_path = traj_data['camera_path']
            elif 'keyframes' in traj_data and len(traj_data['keyframes']) > 0:
                camera_path = traj_data['keyframes']
            else:
                return

            positions_original = []
            rotations_data = []
            for frame in camera_path:
                pos = None
                wxyz = None
                if 'position' in frame:
                    pos = np.array(frame['position'])
                if 'extrinsic' in frame:
                    extrinsic = np.array(frame['extrinsic'])
                    T_cam_world = np.vstack([extrinsic, [0, 0, 0, 1]])
                    T_world_cam = np.linalg.inv(T_cam_world)
                    R_world_cam = T_world_cam[:3, :3]
                    if pos is None:
                        pos = T_world_cam[:3, 3]
                    R_world_cam_zup = transform_rotation_to_z_up(R_world_cam)
                    wxyz = Rotation.from_matrix(R_world_cam_zup).as_quat()[[3, 0, 1, 2]]
                elif 'wxyz' in frame:
                    wxyz = np.array(frame['wxyz'])
                if pos is not None:
                    positions_original.append(pos)
                    rotations_data.append(wxyz)

            if len(positions_original) == 0:
                return

            positions_original = np.array(positions_original)
            positions_zup = transform_to_z_up(positions_original)

            try:
                handle = self.server.scene.add_spline_catmull_rom(
                    f"/saved_trajectory/{filename}/path", positions=positions_zup,
                    color=color, line_width=2.0, segments=len(positions_zup) * 2
                )
                traj_info['handles'].append(handle)
            except Exception:
                pass

            positions_zup_skip_first = positions_zup[1:] if len(positions_zup) > 1 else positions_zup
            point_colors = np.tile(color, (len(positions_zup_skip_first), 1))

            handle = self.server.scene.add_point_cloud(
                f"/saved_trajectory/{filename}/points", points=positions_zup_skip_first,
                colors=point_colors, point_size=0.01, point_shape="circle"
            )
            traj_info['handles'].append(handle)

            fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
            aspect = self.unified_aspect if self.unified_aspect is not None else (self.image_width / self.image_height)

            n_frames = len(positions_zup)
            if n_frames >= 3:
                display_indices = [0, n_frames // 2, n_frames - 1]
            elif n_frames == 2:
                display_indices = [0, 1]
            else:
                display_indices = [0] if n_frames > 0 else []

            for i, idx in enumerate(display_indices):
                pos = positions_zup[idx]
                wxyz = rotations_data[idx] if idx < len(rotations_data) and rotations_data[idx] is not None else None
                if wxyz is not None:
                    camera_handle = self.server.scene.add_camera_frustum(
                        name=f"/saved_trajectory/{filename}/camera_{i}",
                        fov=fov, aspect=aspect, scale=0.035,
                        wxyz=wxyz, position=pos, color=color,
                    )
                    traj_info['handles'].append(camera_handle)

        except Exception as e:
            print(f"Failed to load trajectory {filename}: {e}")

    def _delete_selected_trajectory(self, client):
        selected_filename = None
        for filename, traj_info in self.available_trajectories.items():
            if traj_info['checkbox'] is not None and traj_info['checkbox'].value:
                selected_filename = filename
                break

        if selected_filename is None:
            if client is not None:
                with client.gui.add_modal("Info") as modal:
                    client.gui.add_markdown("**Please select a trajectory first**")
                    close_button = client.gui.add_button("OK")
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()
            return

        if selected_filename == "global_camera.json":
            if client is not None:
                with client.gui.add_modal("Cannot Delete") as modal:
                    client.gui.add_markdown("**`global_camera.json` cannot be deleted**")
                    close_button = client.gui.add_button("OK")
                    @close_button.on_click
                    def _(_) -> None:
                        modal.close()
            return

        try:
            traj_info = self.available_trajectories[selected_filename]

            traj_path = os.path.join(self.data_dir, selected_filename)
            if os.path.exists(traj_path):
                os.remove(traj_path)
                print(f"Deleted trajectory file: {traj_path}")

            for handle in traj_info['handles']:
                try: handle.remove()
                except Exception: pass
            traj_info['handles'].clear()

            del self.available_trajectories[selected_filename]

            # Rebuild the whole saved-trajectories UI from scratch so the list stays consistent.
            self._rebuild_trajectory_list_gui()
            self._refresh_stats()

        except Exception as e:
            print(f"Error deleting trajectory: {e}")

    def _rebuild_trajectory_list_gui(self):
        """Build the saved-trajectories UI (markdown, checkboxes, delete/clear buttons)
        from scratch for the current track. All widgets are tracked in
        self.trajectory_gui_widgets so they can be wiped together on track switch."""
        # Remove EVERY previously-added widget in this section.
        for w in self.trajectory_gui_widgets:
            try: w.remove()
            except Exception: pass
        self.trajectory_gui_widgets = []
        self.gui_trajectory_markdown = None
        self.gui_delete_trajectory = None
        self.gui_clear_all_trajectories = None
        for filename, traj_info in self.available_trajectories.items():
            traj_info['checkbox'] = None

        if len(self.available_trajectories) == 0:
            return

        self.gui_trajectory_markdown = self.server.gui.add_markdown("---\n**Saved Trajectories**")
        self.trajectory_gui_widgets.append(self.gui_trajectory_markdown)

        for filename, traj_info in sorted(self.available_trajectories.items()):
            checkbox = self.server.gui.add_checkbox(filename, initial_value=False)
            traj_info['checkbox'] = checkbox
            traj_info['folder'] = None
            traj_info['color'] = None
            self.trajectory_gui_widgets.append(checkbox)
            @checkbox.on_update
            def _(event, fn=filename):
                self._update_trajectory_display(fn)

        self.gui_delete_trajectory = self.server.gui.add_button(
            "Delete Selected", color="red"
        )
        self.trajectory_gui_widgets.append(self.gui_delete_trajectory)
        @self.gui_delete_trajectory.on_click
        def _(event) -> None:
            self._delete_selected_trajectory(event.client)

        self.gui_clear_all_trajectories = self.server.gui.add_button(
            "Clear All Trajectories", icon=viser.Icon.EYE_OFF
        )
        self.trajectory_gui_widgets.append(self.gui_clear_all_trajectories)
        @self.gui_clear_all_trajectories.on_click
        def _(event) -> None:
            self._clear_all_trajectory_displays()

    def _clear_all_trajectory_displays(self):
        for filename, traj_info in self.available_trajectories.items():
            if traj_info['checkbox'] is not None:
                traj_info['checkbox'].value = False
            traj_info['color'] = None
            for handle in traj_info['handles']:
                try: handle.visible = False
                except Exception: pass
        self.trajectory_color_order.clear()

    # ========================================================================
    # Main loop
    # ========================================================================

    def run(self):
        print(f"Per-Track Camera Trajectory Editor")
        print(f"  URL: http://localhost:{self.port}")
        print(f"  Tracks: {len(self.all_seqs)} / Videos: {len(self.videos)}")
        print(f"  Coordinate system: Z-up")

        prev_timestep = self.gui_timestep.value
        while True:
            if self.is_playing and self.num_frames > 0:
                next_timestep = (self.gui_timestep.value + 1) % self.num_frames
                self.gui_timestep.value = next_timestep
                if len(self.keyframes) >= 2 and self.trajectory_spline_position:
                    self._update_camera_along_trajectory(next_timestep)
            if self.gui_timestep.value != prev_timestep:
                self._update_display()
                prev_timestep = self.gui_timestep.value
            time.sleep(1.0 / self.gui_framerate.value)


def main():
    args = parse_args()
    if not HAS_SPLINES:
        print(f"Error: splines library not installed. Run: pip install splines")
        return
    roots = [args.vipe_lyra_root, args.vipe_lyra_noopt_root, args.vipe_default_noopt_root]
    if not any(os.path.isdir(r) for r in roots if r):
        print(f"Error: none of the roots exist: {roots}")
        return
    try:
        viewer = PerTrackCameraTrajectoryEditor(
            vipe_lyra_root=args.vipe_lyra_root,
            vipe_lyra_noopt_root=args.vipe_lyra_noopt_root,
            vipe_default_noopt_root=args.vipe_default_noopt_root,
            port=args.port, point_size=args.point_size,
            foreground_point_size=args.foreground_point_size,
            fps=args.fps, subsample=args.subsample,
            num_frames=args.num_frames, show_bbox=args.show_bbox,
            decisions_path=os.path.abspath(args.decisions),
        )
        viewer.run()
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
