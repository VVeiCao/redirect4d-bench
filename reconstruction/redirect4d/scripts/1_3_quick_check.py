#!/usr/bin/env python3
"""Quick-check viewer for prepared sequences — per-track review mode.

Backends: VIPE+LyRA, VIPE+LyRA+NoOpt (canonical), VIPE+Default+NoOpt.
Review actions: Pending / Keep / Remove.
Clean Data moves only prepared_vipe_lyra_noopt/<track> to removed/.

Data layout (after 0410+0411 merge):
    data_merged_reprocess/   — 1724 source tracks
    outputs/                 — all pipeline outputs (merged)

Usage:
    python scripts/1_3_quick_check.py --port 8081

    # With separate decisions file:
    python scripts/1_3_quick_check.py --port 8081 \\
        --decisions track_decisions_merged.json
"""


import os
import re
import time
import json
import shutil
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation


def transform_to_z_up(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 3)
    return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]]).reshape(points.shape)


def transform_rotation_to_z_up(R: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64) @ R


def load_ply(path: str, subsample: int):
    mesh = trimesh.load(path)
    pts = np.array(mesh.vertices)
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        colors = np.array(mesh.visual.vertex_colors)[:, :3]
    else:
        colors = np.ones((len(pts), 3), dtype=np.uint8) * 150
    if subsample > 1 and len(pts) > subsample:
        idx = np.arange(0, len(pts), subsample)
        pts, colors = pts[idx], colors[idx]
    return transform_to_z_up(pts), colors


def load_sequence(seq_dir: str, bg_subsample: int, fg_subsample: int,
                  num_frames_limit: Optional[int] = None) -> Dict:
    bg_path = os.path.join(seq_dir, "global_background.ply")
    cam_path = os.path.join(seq_dir, "global_camera.json")

    bg_points, bg_colors = load_ply(bg_path, bg_subsample)

    with open(cam_path, "r") as f:
        cam_raw = json.load(f)
    image_size = cam_raw.get("image_size", [480, 640])
    image_h, image_w = image_size[0], image_size[1]

    cameras: Dict[str, Dict] = {}
    for frame_id, info in cam_raw.items():
        if not isinstance(info, dict) or "extrinsic" not in info:
            continue
        T_cw = np.vstack([np.array(info["extrinsic"]), [0, 0, 0, 1]])
        T_wc = np.linalg.inv(T_cw)
        pos = transform_to_z_up(T_wc[:3, 3].reshape(1, 3)).flatten()
        R = transform_rotation_to_z_up(T_wc[:3, :3])
        cameras[frame_id] = {
            "position": pos,
            "R": R,
            "intrinsic": np.array(info["intrinsic"]),
        }

    frame_ids = sorted(
        d for d in os.listdir(seq_dir)
        if d.isdigit() and os.path.isdir(os.path.join(seq_dir, d))
    )
    if num_frames_limit:
        frame_ids = frame_ids[:num_frames_limit]

    smooth_candidates = [
        "{fid}_foreground_5_views_aligned_smooth.ply",
        "{fid}_foreground_5_views_aligned.ply",
        "{fid}_foreground_5_views.ply",
    ]
    single_view_pattern = "{fid}_foreground_1_view.ply"

    fg_smooth: List[Dict] = []
    fg_single: List[Dict] = []
    for fid in frame_ids:
        pc_dir = os.path.join(seq_dir, fid, "pointcloud")

        for tmpl in smooth_candidates:
            p = os.path.join(pc_dir, tmpl.format(fid=fid))
            if os.path.exists(p):
                try:
                    pts, cols = load_ply(p, fg_subsample)
                    fg_smooth.append({"frame_id": fid, "points": pts, "colors": cols})
                except Exception as e:
                    print(f"  [!] failed to load {os.path.basename(p)}: {e}")
                break

        p = os.path.join(pc_dir, single_view_pattern.format(fid=fid))
        if os.path.exists(p):
            try:
                pts, cols = load_ply(p, fg_subsample)
                fg_single.append({"frame_id": fid, "points": pts, "colors": cols})
            except Exception as e:
                print(f"  [!] failed to load {os.path.basename(p)}: {e}")

    return {
        "bg": {"points": bg_points, "colors": bg_colors},
        "cameras": cameras,
        "fg_smooth": fg_smooth,
        "fg_single": fg_single,
        "image_size": (image_w, image_h),
    }


# --- Decision constants ---
DECISION_PENDING = "Pending"
DECISION_KEEP = "Keep"
DECISION_REMOVE = "Remove"
DECISION_OPTIONS = (DECISION_PENDING, DECISION_KEEP, DECISION_REMOVE)

VIDEO_ALL = "(all)"


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
    # Legacy entries (keep_vipe / keep_dpg) all map to generic Keep now
    if entry.get("keep_vipe") or entry.get("keep_dpg"):
        return DECISION_KEEP
    return DECISION_PENDING


def parse_video(seq_name: str) -> str:
    """Extract video id (category + YouTube ID) by stripping trailing
    '_<clip>_<inst>_seq<n>' from a track name.
    E.g. 'bear_NnAlfavy2us_003_001_seq1' -> 'bear_NnAlfavy2us'.
    Falls back to stripping just '_seqN' if the longer pattern doesn't match.
    """
    m = re.match(r"^(.+)_\d+_\d+_seq\d+$", seq_name)
    if m:
        return m.group(1)
    parts = seq_name.rsplit("_", 1)
    if len(parts) == 2 and re.match(r"^seq\d+$", parts[1]):
        return parts[0]
    return seq_name


BACKEND_DPG = "DPG"
BACKEND_VIPE_LYRA = "VIPE+LyRA"
BACKEND_VIPE_LYRA_NOOPT = "VIPE+LyRA+NoOpt"
BACKEND_VIPE_DA3_NOOPT = "VIPE+DA3+NoOpt"
BACKEND_VIPE_DEFAULT_NOOPT = "VIPE+Default+NoOpt"


class QuickViewer:
    def __init__(self, dpg_root: str, port: int,
                 bg_subsample: int, fg_subsample: int,
                 point_size: float, fg_point_size: float, fps: float,
                 num_frames_limit: Optional[int], decisions_path: str,
                 vipe_lyra_root: str = "",
                 vipe_lyra_noopt_root: str = "",
                 vipe_default_noopt_root: str = ""):
        # dpg_root kept as anchor for outputs_root derivation only (not selectable)
        self._dpg_root = dpg_root
        self.roots: Dict[str, str] = {BACKEND_VIPE_LYRA: vipe_lyra_root,
                                      BACKEND_VIPE_LYRA_NOOPT: vipe_lyra_noopt_root,
                                      BACKEND_VIPE_DEFAULT_NOOPT: vipe_default_noopt_root}
        self.bg_subsample = bg_subsample
        self.fg_subsample = fg_subsample
        self.point_size = point_size
        self.fg_point_size = fg_point_size
        self.fps_default = fps
        self.num_frames_limit = num_frames_limit

        self.decisions_path = decisions_path
        self.decisions: Dict[str, Dict] = load_decisions(decisions_path)
        print(f"[decisions] loaded {len(self.decisions)} entries from {decisions_path}")

        # Scan all roots; later restrict the visible track list to those that
        # exist in the canonical Clean-Data target (prepared_vipe_lyra_noopt),
        # so Remove + Clean Data actually makes the track disappear.
        self.seq_backends: Dict[str, List[str]] = defaultdict(list)  # seq_name -> [backends]
        for backend, root in self.roots.items():
            if root and os.path.isdir(root):
                seqs = self._scan_one_root(root)
                for s in seqs:
                    self.seq_backends[s].append(backend)
                print(f"[scan] {backend:4s}: {len(seqs)} seqs  ({root})")
            else:
                print(f"[scan] {backend:4s}: (root missing, skipped)")

        # Only show tracks that exist in the canonical clean-data target.
        canonical_seqs = (
            {s for s, bs in self.seq_backends.items() if BACKEND_VIPE_LYRA_NOOPT in bs}
            if any(BACKEND_VIPE_LYRA_NOOPT in bs for bs in self.seq_backends.values())
            else set(self.seq_backends.keys())
        )
        self.all_seqs: List[str] = sorted(canonical_seqs)
        if not self.all_seqs:
            raise ValueError("No prepared sequences found in any backend")
        n_skipped = len(self.seq_backends) - len(self.all_seqs)
        if n_skipped:
            print(f"[scan] hiding {n_skipped} tracks not in canonical "
                  f"{BACKEND_VIPE_LYRA_NOOPT} (already removed elsewhere)")

        # Build category list
        self.videos: List[str] = sorted(set(parse_video(s) for s in self.all_seqs))
        self.filtered_seqs: List[str] = list(self.all_seqs)
        self.current_video: str = VIDEO_ALL

        # Optional pre-computed background flatness lookup
        # {<track>: {<backend>: pca_thin/wide ratio}}
        # Try dataset-local first (outputs_XXXX/bg_flatness.json), then project root.
        self.bg_flatness: Dict[str, Dict[str, float]] = {}
        outputs_root = os.path.dirname(os.path.abspath(self._dpg_root))
        candidates = [
            os.path.join(outputs_root, "bg_flatness.json"),
            os.path.normpath(os.path.join(outputs_root, "..", "bg_flatness.json")),
        ]
        for flat_path in candidates:
            if os.path.exists(flat_path):
                try:
                    with open(flat_path) as f:
                        self.bg_flatness = json.load(f)
                    print(f"[flatness] loaded {len(self.bg_flatness)} entries from {flat_path}")
                    break
                except Exception as e:
                    print(f"[!] failed to load {flat_path}: {e}")

        self.current_backend = BACKEND_VIPE_LYRA_NOOPT
        self.current_seq: Optional[str] = None
        self._suppress_cb = False
        self.data: Optional[Dict] = None
        self.bg_handle = None
        self.fg_smooth_handles: List = []
        self.fg_single_handles: List = []
        self.camera_handles: List[Dict] = []
        self.cam_points_handle = None
        self.traj_handle = None
        self.axes_handle = None
        self.is_playing = False

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self._build_gui()
        self._load_seq(self.filtered_seqs[0])

    def _scan_one_root(self, root: str) -> List[str]:
        """Return list of seq names under `root` that have global_background.ply + global_camera.json."""
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

    def _apply_video_filter(self):
        if self.current_video == VIDEO_ALL:
            self.filtered_seqs = list(self.all_seqs)
        else:
            self.filtered_seqs = [s for s in self.all_seqs
                                  if parse_video(s) == self.current_video]

    def _sync_backend_rows(self, backend: str):
        """Row widgets are read-only info about which backend is active.
        Single-button rows always show their one label; nothing to sync."""
        pass

    def _update_bg_warning(self, seq_name: str):
        """Show flatness across backends for current track. Highlight if current backend collapsed."""
        per_backend = self.bg_flatness.get(seq_name, {})
        if not per_backend:
            self.gui_bg_warning.value = "(no flatness data)"
            return
        cur_val = per_backend.get(self.current_backend)
        # Compact summary of all backends
        parts = []
        for k in self.roots:
            v = per_backend.get(k)
            if v is None:
                parts.append(f"{k}=?")
            else:
                tag = "X" if v < 0.05 else ("!" if v < 0.10 else "ok")
                parts.append(f"{k}={v:.3f}[{tag}]")
        prefix = ""
        if cur_val is not None:
            if cur_val < 0.05:
                prefix = "⚠ COLLAPSED — "
            elif cur_val < 0.10:
                prefix = "⚠ flat — "
        self.gui_bg_warning.value = prefix + "  ".join(parts)

    def _apply_point_sizes(self):
        """Update live point_size on already-rendered handles without reload."""
        sf = getattr(self, "current_scale_factor", 1.0)
        pt = self.point_size * sf
        fg = self.fg_point_size * sf
        if getattr(self, "bg_handle", None) is not None:
            try: self.bg_handle.point_size = pt
            except Exception: pass
        for h in self.fg_smooth_handles:
            try: h.point_size = fg
            except Exception: pass
        for h in self.fg_single_handles:
            try: h.point_size = fg
            except Exception: pass

    def _available_backends_for(self, seq_name: str) -> List[str]:
        return self.seq_backends.get(seq_name, [])

    def _build_gui(self):
        s = self.server

        first_seq = self.filtered_seqs[0]
        backends_here = self._available_backends_for(first_seq)
        if backends_here and self.current_backend not in backends_here:
            self.current_backend = backends_here[0]

        with s.gui.add_folder("Track"):
            self.gui_video = s.gui.add_dropdown(
                "Video",
                options=(VIDEO_ALL,) + tuple(self.videos),
                initial_value=VIDEO_ALL,
                hint="Filter tracks by source video (all seq* of the same video grouped)",
            )
            self.gui_seq = s.gui.add_dropdown(
                "Track", options=tuple(self.filtered_seqs), initial_value=first_seq,
                hint="Pick which track (sequence) to visualize and review",
            )
            # One backend per row — labels are long and overflow when grouped
            self.gui_backend_row1 = s.gui.add_button_group(
                "Backend", (BACKEND_VIPE_LYRA,),
                hint="Switch which reconstruction backend to visualize.",
            )
            self.gui_backend_row2 = s.gui.add_button_group(
                " ", (BACKEND_VIPE_LYRA_NOOPT,),
            )
            self.gui_backend_row3 = s.gui.add_button_group(
                "  ", (BACKEND_VIPE_DEFAULT_NOOPT,),
            )
            self.gui_backend_row4 = None
            self._suppress_cb = True
            self.gui_backend_row1.value = BACKEND_VIPE_LYRA
            self.gui_backend_row2.value = BACKEND_VIPE_LYRA_NOOPT
            self.gui_backend_row3.value = BACKEND_VIPE_DEFAULT_NOOPT
            self._suppress_cb = False
            self.gui_avail = s.gui.add_text("Available", initial_value="", disabled=True)
            # Track-level nav (cyan)
            self.gui_prev = s.gui.add_button("<- Prev Track", color="cyan")
            self.gui_next = s.gui.add_button("Next Track ->", color="cyan")
            # Video-level nav (yellow)
            self.gui_prev_video = s.gui.add_button("<- Prev Video", color="yellow")
            self.gui_next_video = s.gui.add_button("Next Video ->", color="yellow")
            # Unreviewed (orange)
            self.gui_next_pending = s.gui.add_button(
                "Next Unreviewed Track ->", color="orange",
                icon=viser.Icon.ARROW_BIG_RIGHT,
            )
            self.gui_info = s.gui.add_text("Info", initial_value="", disabled=True)
            self.gui_index = s.gui.add_text("Index", initial_value="", disabled=True)
            self.gui_stats = s.gui.add_markdown("")
            self.gui_bg_warning = s.gui.add_text(
                "BG Quality", initial_value="", disabled=True,
                hint="Per-track per-backend background flatness (PCA thin/wide). "
                     "<0.05 = collapsed (very flat sheet), 0.05-0.10 = suspicious.",
            )

        with s.gui.add_folder("Review"):
            self.gui_status = s.gui.add_text("Status", initial_value=DECISION_PENDING, disabled=True)
            self.gui_btn_pending = s.gui.add_button("Pending", icon=viser.Icon.CIRCLE)
            self.gui_btn_keep = s.gui.add_button("Keep", icon=viser.Icon.CHECK, color="green")
            self.gui_btn_remove = s.gui.add_button("Remove", icon=viser.Icon.TRASH, color="red")
            self.gui_btn_save = s.gui.add_button("Save JSON", icon=viser.Icon.DEVICE_FLOPPY)
            self.gui_btn_clean = s.gui.add_button("Clean Data", icon=viser.Icon.TRASH_X,
                                                   hint="Apply decisions: move unchosen outputs to removed/")

        with s.gui.add_folder("Time"):
            self.gui_time = s.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0)
            self.gui_fid = s.gui.add_text("Frame ID", initial_value="", disabled=True)
            self.gui_play = s.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
            self.gui_pause = s.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
            self.gui_fps = s.gui.add_slider("FPS", min=1, max=30, step=0.5, initial_value=self.fps_default)

        with s.gui.add_folder("Display"):
            self.gui_show_bg = s.gui.add_checkbox("Background", True)
            self.gui_show_fg_smooth = s.gui.add_checkbox("Smooth Foreground (5-view)", True)
            self.gui_show_fg_single = s.gui.add_checkbox("Single-View Foreground", False)
            self.gui_show_cam = s.gui.add_checkbox("Cameras", True)
            self.gui_show_traj = s.gui.add_checkbox("Camera Trajectory", True)
            self.gui_show_axes = s.gui.add_checkbox("Axes", True)
            self.gui_bg_pt_size = s.gui.add_slider(
                "BG Point Size", min=0.0005, max=0.05, step=0.0005,
                initial_value=self.point_size,
            )
            self.gui_fg_pt_size = s.gui.add_slider(
                "FG Point Size", min=0.0005, max=0.05, step=0.0005,
                initial_value=self.fg_point_size,
            )

        self.gui_reset_view = s.gui.add_button("Reset View", icon=viser.Icon.VIEWFINDER)

        # --- Callbacks ---

        @self.gui_video.on_update
        def _(_):
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

        def _handle_backend_click(new_backend: str):
            if self._suppress_cb:
                return
            if new_backend == self.current_backend:
                return
            if new_backend not in self._available_backends_for(self.current_seq):
                print(f"[!] {self.current_seq}: no {new_backend} data; staying on {self.current_backend}")
                self._sync_backend_rows(self.current_backend)
                return
            self.current_backend = new_backend
            self._sync_backend_rows(new_backend)
            self._load_seq(self.current_seq)

        @self.gui_backend_row1.on_click
        def _(_):
            _handle_backend_click(self.gui_backend_row1.value)

        @self.gui_backend_row2.on_click
        def _(_):
            _handle_backend_click(self.gui_backend_row2.value)

        @self.gui_backend_row3.on_click
        def _(_):
            _handle_backend_click(self.gui_backend_row3.value)

        if self.gui_backend_row4 is not None:
            @self.gui_backend_row4.on_click
            def _(_):
                _handle_backend_click(self.gui_backend_row4.value)

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
            """direction=+1 next video, -1 prev video. Iterates over ALL videos
            regardless of current Video filter. Switches Video filter to the
            target video (forces specific selection), landing on its first track.
            """
            if not self.videos:
                return
            cur = parse_video(self.current_seq) if self.current_seq else None
            try:
                idx = self.videos.index(cur)
            except (ValueError, TypeError):
                idx = 0
            new_idx = (idx + direction) % len(self.videos)
            target_video = self.videos[new_idx]
            # Drive the Video dropdown — its on_update will filter and load first track
            if self.gui_video.value != target_video:
                self.gui_video.value = target_video
            else:
                # Already filtered to this video — explicitly load first track
                first = next((s for s in self.all_seqs if parse_video(s) == target_video), None)
                if first:
                    self.gui_seq.value = first

        @self.gui_prev_video.on_click
        def _(_):
            _jump_video(-1)

        @self.gui_next_video.on_click
        def _(_):
            _jump_video(+1)

        @self.gui_next_pending.on_click
        def _(_):
            target = self._find_next_pending()
            if target is None:
                print("[review] no pending tracks left in current filter")
                return
            self.gui_seq.value = target

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

        @self.gui_btn_save.on_click
        def _(_):
            try:
                save_decisions_atomic(self.decisions_path, self.decisions)
                n = len(self.decisions)
                print(f"[save] manually saved {n} entries to {self.decisions_path}")
            except Exception as e:
                print(f"[!] manual save failed: {e}")

        @self.gui_btn_clean.on_click
        def _(_):
            self._run_clean_data()

        @self.gui_time.on_update
        def _(_):
            self._update_frame()

        @self.gui_play.on_click
        def _(_):
            self.is_playing = True
            self.gui_play.visible = False
            self.gui_pause.visible = True
            self.gui_time.disabled = True

        @self.gui_pause.on_click
        def _(_):
            self.is_playing = False
            self.gui_play.visible = True
            self.gui_pause.visible = False
            self.gui_time.disabled = False

        for cb in [self.gui_show_bg, self.gui_show_fg_smooth, self.gui_show_fg_single,
                   self.gui_show_cam, self.gui_show_traj, self.gui_show_axes]:
            @cb.on_update
            def _(_):
                self._apply_visibility()

        @self.gui_bg_pt_size.on_update
        def _(_):
            self.point_size = float(self.gui_bg_pt_size.value)
            self._apply_point_sizes()

        @self.gui_fg_pt_size.on_update
        def _(_):
            self.fg_point_size = float(self.gui_fg_pt_size.value)
            self._apply_point_sizes()

        @self.gui_reset_view.on_click
        def _(_):
            self._recenter_client_camera()

        @s.on_client_connect
        def _(client: viser.ClientHandle):
            self._set_client_camera(client)

    def _clear_scene(self):
        with self.server.atomic():
            for h in [self.bg_handle, self.cam_points_handle, self.traj_handle, self.axes_handle]:
                if h is not None:
                    try:
                        h.remove()
                    except Exception:
                        pass
            self.bg_handle = None
            self.cam_points_handle = None
            self.traj_handle = None
            self.axes_handle = None
            for h in self.fg_smooth_handles + self.fg_single_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self.fg_smooth_handles.clear()
            self.fg_single_handles.clear()
            for ch in self.camera_handles:
                try:
                    ch["handle"].remove()
                except Exception:
                    pass
            self.camera_handles.clear()

    def _load_seq(self, seq_name: str):
        """Load a single prepared sequence into the 3D scene."""
        avail = self._available_backends_for(seq_name)
        if not avail:
            print(f"[!] {seq_name} has no prepared data in any backend")
            self.gui_info.value = f"ERROR: no data for {seq_name}"
            return

        if self.current_backend not in avail:
            new_backend = avail[0]
            print(f"[backend] {seq_name}: {self.current_backend} unavailable, switching to {new_backend}")
            self.current_backend = new_backend

        self._sync_backend_rows(self.current_backend)

        self.gui_avail.value = "+".join(avail) if len(avail) > 1 else f"{avail[0]} only"

        t0 = time.time()
        root = self.roots[self.current_backend]
        seq_dir = os.path.join(root, seq_name)
        try:
            data = load_sequence(seq_dir, self.bg_subsample, self.fg_subsample,
                                 self.num_frames_limit)
        except Exception as e:
            print(f"[!] failed to load {seq_name} ({self.current_backend}): {e}")
            self.gui_info.value = f"ERROR: {e}"
            return

        self.current_seq = seq_name
        self.data = data
        self._clear_scene()
        self._build_scene(data)

        n_frames = max(len(data["fg_smooth"]), len(data["fg_single"]))
        self.gui_time.max = max(n_frames - 1, 0)
        self.gui_time.value = 0

        if self.current_seq in self.filtered_seqs:
            idx = self.filtered_seqs.index(self.current_seq)
            self.gui_index.value = f"{idx + 1} / {len(self.filtered_seqs)}"
        else:
            self.gui_index.value = ""

        self.gui_info.value = (
            f"[{self.current_backend}] {len(data['bg']['points'])} bg pts | "
            f"smooth {len(data['fg_smooth'])} / single {len(data['fg_single'])} | "
            f"{len(data['cameras'])} cams"
        )
        self._update_bg_warning(seq_name)

        self._populate_decision_gui(seq_name)
        self._apply_visibility()
        self._update_frame()
        self._recenter_client_camera()
        elapsed = time.time() - t0
        print(f"[load] {seq_name} (showing {self.current_backend})  ({elapsed:.2f}s)")

    def _build_scene(self, data: Dict):
        s = self.server
        bg = data["bg"]

        # Auto-scale point size based on scene extent so DPG and VIPE look similar
        extent = np.max(bg["points"].max(0) - bg["points"].min(0))
        self.current_scale_factor = max(extent / 1.5, 1.0)  # DPG is ~1.4 extent, use as reference
        pt_size = self.point_size * self.current_scale_factor
        fg_pt_size = self.fg_point_size * self.current_scale_factor

        self.bg_handle = s.scene.add_point_cloud(
            name="/pc/background",
            points=bg["points"], colors=bg["colors"],
            point_size=pt_size, point_shape="circle",
        )

        for fr in data["fg_smooth"]:
            h = s.scene.add_point_cloud(
                name=f"/pc/fg_smooth_{fr['frame_id']}",
                points=fr["points"], colors=fr["colors"],
                point_size=fg_pt_size, point_shape="circle",
            )
            h.visible = False
            self.fg_smooth_handles.append(h)

        for fr in data["fg_single"]:
            h = s.scene.add_point_cloud(
                name=f"/pc/fg_single_{fr['frame_id']}",
                points=fr["points"], colors=fr["colors"],
                point_size=fg_pt_size, point_shape="circle",
            )
            h.visible = False
            self.fg_single_handles.append(h)

        image_w, image_h = data["image_size"]
        cam_positions: List[np.ndarray] = []
        for frame_id in sorted(data["cameras"].keys()):
            cam = data["cameras"][frame_id]
            try:
                wxyz = Rotation.from_matrix(cam["R"]).as_quat()[[3, 0, 1, 2]]
                fov_y = 2 * np.arctan(image_h / (2 * cam["intrinsic"][1, 1]))
                aspect = image_w / image_h
                h = s.scene.add_camera_frustum(
                    f"/cam/frustum_{frame_id}",
                    fov=fov_y, aspect=aspect, scale=0.018 * self.current_scale_factor,
                    wxyz=wxyz, position=cam["position"], color=(196, 78, 82),
                )
                self.camera_handles.append({"handle": h, "frame_id": frame_id})
                cam_positions.append(cam["position"])
            except Exception:
                pass

        if cam_positions:
            cam_positions_arr = np.array(cam_positions)
            cam_colors = np.tile([196, 78, 82], (len(cam_positions_arr), 1))
            self.cam_points_handle = s.scene.add_point_cloud(
                "/cam/positions", points=cam_positions_arr, colors=cam_colors,
                point_size=0.018 * self.current_scale_factor, point_shape="circle",
            )
            if len(cam_positions_arr) >= 2:
                try:
                    self.traj_handle = s.scene.add_spline_catmull_rom(
                        "/cam/trajectory", positions=cam_positions_arr,
                        color=(196, 78, 82), line_width=2.0,
                        segments=len(cam_positions_arr) * 2,
                    )
                except Exception:
                    pass

        all_pts = ([bg["points"]]
                   + [f["points"] for f in data["fg_smooth"]]
                   + [f["points"] for f in data["fg_single"]])
        merged = np.concatenate(all_pts)
        scale = float(np.max(merged.max(0) - merged.min(0)) * 0.1) or 0.5
        axis_pts, axis_cols = [], []
        for vec, col in [([scale, 0, 0], [255, 0, 0]),
                         ([0, scale, 0], [0, 255, 0]),
                         ([0, 0, scale], [0, 0, 255])]:
            for t in np.linspace(0, 1, 50):
                axis_pts.append([t * vec[0], t * vec[1], t * vec[2]])
                axis_cols.append(col)
        self.axes_handle = s.scene.add_point_cloud(
            "/reference/axes",
            points=np.array(axis_pts), colors=np.array(axis_cols),
            point_size=self.point_size * 3.0, point_shape="circle",
        )

    def _apply_visibility(self):
        if self.bg_handle is not None:
            self.bg_handle.visible = self.gui_show_bg.value
        if self.cam_points_handle is not None:
            self.cam_points_handle.visible = self.gui_show_cam.value
        if self.traj_handle is not None:
            self.traj_handle.visible = self.gui_show_traj.value
        if self.axes_handle is not None:
            self.axes_handle.visible = self.gui_show_axes.value
        self._update_frame()

    def _update_frame(self):
        if not self.data:
            return
        t = int(self.gui_time.value)
        smooth_frames = self.data["fg_smooth"]
        single_frames = self.data["fg_single"]
        current_fid = ""
        if t < len(smooth_frames):
            current_fid = smooth_frames[t]["frame_id"]
        elif t < len(single_frames):
            current_fid = single_frames[t]["frame_id"]
        self.gui_fid.value = current_fid

        show_smooth = self.gui_show_fg_smooth.value
        show_single = self.gui_show_fg_single.value
        show_cam = self.gui_show_cam.value
        with self.server.atomic():
            for i, h in enumerate(self.fg_smooth_handles):
                h.visible = show_smooth and (i == t)
            for i, h in enumerate(self.fg_single_handles):
                h.visible = show_single and (i == t)
            for ch in self.camera_handles:
                if not show_cam:
                    ch["handle"].visible = False
                else:
                    ch["handle"].visible = (ch["frame_id"] == current_fid)

    def _scene_center_and_extent(self):
        if not self.data:
            return None, None
        all_pts = ([self.data["bg"]["points"]]
                   + [f["points"] for f in self.data["fg_smooth"]]
                   + [f["points"] for f in self.data["fg_single"]])
        merged = np.concatenate(all_pts)
        center = (merged.min(0) + merged.max(0)) / 2.0
        extent = float(np.max(merged.max(0) - merged.min(0)))
        return center, extent

    def _set_client_camera(self, client: viser.ClientHandle):
        center, extent = self._scene_center_and_extent()
        if center is None:
            return
        if self.data and "00000" in self.data["cameras"]:
            init_pos = self.data["cameras"]["00000"]["position"]
        else:
            init_pos = center + np.array([0.0, -extent * 1.5, extent * 0.5])
        try:
            client.camera.position = init_pos
            client.camera.look_at = center
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])
        except Exception:
            pass

    def _find_next_pending(self) -> Optional[str]:
        if not self.filtered_seqs:
            return None
        if self.current_seq not in self.filtered_seqs:
            start = 0
        else:
            start = self.filtered_seqs.index(self.current_seq)
        n = len(self.filtered_seqs)
        for step in range(1, n + 1):
            cand = self.filtered_seqs[(start + step) % n]
            if cand not in self.decisions:
                return cand
        return None

    def _refresh_stats(self):
        # Global summary across the filtered set
        seqs = self.filtered_seqs
        n = len(seqs)
        n_keep = n_remove = 0
        for s in seqs:
            d = self.decisions.get(s, {})
            if d.get("remove"):
                n_remove += 1
            elif d.get("keep") or d.get("keep_vipe") or d.get("keep_dpg"):
                n_keep += 1
        n_pending = n - n_keep - n_remove
        lines = [f"all: K={n_keep} R={n_remove} .={n_pending} ({n})"]

        if self.current_video == VIDEO_ALL:
            # Bird's-eye view: one line per video with status counts
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
                    elif d.get("keep") or d.get("keep_vipe") or d.get("keep_dpg"): vk += 1
                vp = len(tracks) - vk - vr
                cur = " *" if v == cur_video else ""
                lines.append(f" {v}  K={vk} R={vr} .={vp}{cur}")
        elif self.current_seq:
            # Detail view: tracks within currently-selected video
            cv = parse_video(self.current_seq)
            sibs = sorted(s for s in self.all_seqs if parse_video(s) == cv)
            if sibs:
                lines.append("")
                lines.append(f"{cv} ({len(sibs)})")
                for s in sibs:
                    lab = decision_label(self.decisions.get(s))
                    sym = {"Keep": "K", "Remove": "R"}.get(lab, ".")
                    suf = s[len(cv):].lstrip("_") or s
                    cur = " *" if s == self.current_seq else ""
                    lines.append(f" [{sym}] {suf}{cur}")
        # Wrap in code fence so markdown preserves spacing/alignment
        self.gui_stats.content = "```\n" + "\n".join(lines) + "\n```"

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

    def _run_clean_data(self):
        """Apply decisions: for each 'Remove'-marked track, move ONLY the canonical
        prepared_vipe_lyra_noopt/<track> output to removed/prepared_vipe_lyra_noopt/.
        Other variants and source data (data_merged_reprocess_*/) are NEVER touched.
        """
        outputs_root = os.path.dirname(os.path.abspath(self._dpg_root))
        canonical = "prepared_vipe_lyra_noopt"

        moved, errors, missing = 0, 0, 0

        def safe_move(src, dst):
            nonlocal moved, errors, missing
            if not os.path.lexists(src):
                missing += 1; return
            if os.path.lexists(dst):
                return
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                errors += 1
                print(f"  [!] move failed: {src} -> {e}")

        for t, dec in self.decisions.items():
            if not dec.get("remove"):
                continue
            src = os.path.join(outputs_root, canonical, t)
            dst = os.path.join(outputs_root, "removed", canonical, t)
            safe_move(src, dst)

        print(f"[clean] done: {moved} tracks moved (only {canonical}), "
              f"{missing} not found, {errors} errors")

        # Refresh the viewer's sequence list
        self.seq_backends.clear()
        for backend, root in self.roots.items():
            if root and os.path.isdir(root):
                seqs = self._scan_one_root(root)
                for s in seqs:
                    self.seq_backends[s].append(backend)
        canonical_seqs = (
            {s for s, bs in self.seq_backends.items() if BACKEND_VIPE_LYRA_NOOPT in bs}
            if any(BACKEND_VIPE_LYRA_NOOPT in bs for bs in self.seq_backends.values())
            else set(self.seq_backends.keys())
        )
        self.all_seqs = sorted(canonical_seqs)
        self.videos = sorted(set(parse_video(s) for s in self.all_seqs))
        self._apply_video_filter()

        # Update GUI dropdowns
        self._suppress_cb = True
        self.gui_video.options = (VIDEO_ALL,) + tuple(self.videos)
        self.gui_seq.options = tuple(self.filtered_seqs)
        if self.filtered_seqs:
            if self.current_seq not in self.filtered_seqs:
                self.gui_seq.value = self.filtered_seqs[0]
        self._suppress_cb = False

        # Update index display
        if self.current_seq and self.current_seq in self.filtered_seqs:
            idx = self.filtered_seqs.index(self.current_seq)
            self.gui_index.value = f"{idx + 1} / {len(self.filtered_seqs)}"
        elif self.filtered_seqs:
            self.gui_index.value = f"? / {len(self.filtered_seqs)}"

        self._refresh_stats()
        n_videos = len(set(parse_video(s) for s in self.all_seqs))
        print(f"[clean] viewer refreshed: {len(self.all_seqs)} tracks / "
              f"{n_videos} videos remaining")

    def _recenter_client_camera(self):
        try:
            clients = self.server.get_clients()
        except Exception:
            clients = {}
        for client in clients.values():
            self._set_client_camera(client)

    def run(self):
        print(f"[serve] http://<host>:{self.server.get_port() if hasattr(self.server, 'get_port') else '<port>'}")
        while True:
            if self.is_playing and self.data:
                n = max(len(self.data["fg_smooth"]), len(self.data["fg_single"]), 1)
                next_t = (int(self.gui_time.value) + 1) % n
                self.gui_time.value = next_t
            time.sleep(1.0 / max(float(self.gui_fps.value), 1e-3))


def parse_args():
    p = argparse.ArgumentParser(description="Quick-check viewer — per-track review mode")
    p.add_argument("--dpg_root", type=str, default="outputs/prepared_vipe_lyra_noopt",
                   help="Anchor dir (only used to derive outputs/ root; DPG itself is gone)")
    p.add_argument("--vipe_lyra_root", type=str, default="outputs/prepared_vipe_lyra",
                   help="Directory of VIPE+LyRA (opt=on) sequences")
    p.add_argument("--vipe_lyra_noopt_root", type=str, default="outputs/prepared_vipe_lyra_noopt",
                   help="Directory of VIPE+LyRA+NoOpt (opt=off) sequences")
    p.add_argument("--vipe_default_noopt_root", type=str, default="outputs/prepared_vipe_default_noopt",
                   help="Directory of VIPE+Default+NoOpt (UniDepth-L, opt=off) sequences")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--bg_subsample", type=int, default=8,
                   help="Background point cloud subsample stride (larger=sparser=faster)")
    p.add_argument("--fg_subsample", type=int, default=4,
                   help="Foreground point cloud subsample stride")
    p.add_argument("--point_size", type=float, default=0.003)
    p.add_argument("--fg_point_size", type=float, default=0.002)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--num_frames", type=int, default=None,
                   help="Cap loaded frames per sequence (for even faster switching)")
    p.add_argument("--decisions", type=str, default="track_decisions.json",
                   help="Path to track-level review decisions JSON (auto-saved on every change)")
    return p.parse_args()


def main():
    args = parse_args()
    roots = [args.dpg_root, args.vipe_lyra_root, args.vipe_lyra_noopt_root,
             args.vipe_default_noopt_root]
    if not any(os.path.isdir(r) for r in roots):
        print(f"Error: none of the roots exist: {roots}")
        return
    try:
        viewer = QuickViewer(
            dpg_root=args.dpg_root, port=args.port,
            bg_subsample=args.bg_subsample, fg_subsample=args.fg_subsample,
            point_size=args.point_size, fg_point_size=args.fg_point_size,
            fps=args.fps, num_frames_limit=args.num_frames,
            decisions_path=os.path.abspath(args.decisions),
            vipe_lyra_root=args.vipe_lyra_root,
            vipe_lyra_noopt_root=args.vipe_lyra_noopt_root,
            vipe_default_noopt_root=args.vipe_default_noopt_root,
        )
        viewer.run()
    except KeyboardInterrupt:
        print("\n[exit] interrupted")
    except Exception as e:
        print(f"[error] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
