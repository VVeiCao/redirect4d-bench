#!/usr/bin/env python3
"""Playback viewer: compare DPG-via-hybrid, VIPE foreground_1_view, and VIPE final output.

Loads every frame of a track and serves a viser instance with:
  - Time slider + Play/Pause
  - RED    = DPG foreground_1_view → C (via global hybrid s, t)
  - BLUE   = VIPE foreground_1_view (alignment INPUT/target, raw backproject)
  - GREEN  = VIPE foreground_5_views_aligned_smooth (alignment OUTPUT, what viewer "VIPE" backend shows)
  - GRAY   = VIPE background (static)

Usage:
    python scripts/viz_hybrid_player.py --track dancer_2VTWn9TA8Qw_010_001_seq2
    python scripts/viz_hybrid_player.py --track <name> --port 8083 --no-bg
"""
import argparse
import time
import threading
from pathlib import Path

import numpy as np
import trimesh
import viser


COLOR_DPG          = np.array([255,  80,  80], dtype=np.uint8)   # red
COLOR_VIPE_F1V     = np.array([ 80, 160, 255], dtype=np.uint8)   # blue
COLOR_VIPE_OUT     = np.array([ 80, 220, 100], dtype=np.uint8)   # green (VIPE alignment output)
COLOR_VIPE_F1V_PLY = np.array([255, 200,  60], dtype=np.uint8)   # yellow (raw .ply, real RGB option)
COLOR_BG           = np.array([120, 120, 120], dtype=np.uint8)   # gray


def to_z_up(pts: np.ndarray) -> np.ndarray:
    """Convert from Y-down (model) to Z-up (viewer)."""
    return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])


def load_frame_points(npz_path: Path):
    """Returns foreground pixel 3D points from foreground_1_view.npz."""
    d = np.load(str(npz_path))
    pmap = d['foreground_pointmap']
    mask = d['foreground_mask']
    return pmap[mask]


def load_aligned_smooth(npz_path: Path):
    """Load points from foreground_5_views_aligned_smooth.npz."""
    d = np.load(str(npz_path))
    return d['points']


def load_ply_with_colors(ply_path: Path):
    """Load PLY file with its original RGB colors. Returns (points, colors_uint8)."""
    m = trimesh.load(str(ply_path))
    pts = np.array(m.vertices)
    try:
        cols = np.array(m.visual.vertex_colors)[:, :3].astype(np.uint8)
    except Exception:
        cols = np.tile(np.array([150, 150, 150], dtype=np.uint8), (len(pts), 1))
    return pts, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', required=True)
    ap.add_argument('--dpg-root', default='outputs_0410/prepared')
    ap.add_argument('--vipe-root', default='outputs_0410/prepared_vipe')
    ap.add_argument('--hybrid-root', default='outputs_0410/prepared_hybrid')
    ap.add_argument('--port', type=int, default=8083)
    ap.add_argument('--no-bg', action='store_true', help='Skip background')
    ap.add_argument('--bg-subsample', type=int, default=20)
    ap.add_argument('--fg-subsample', type=int, default=2)
    ap.add_argument('--point-size', type=float, default=0.01)
    ap.add_argument('--fps', type=float, default=8.0)
    args = ap.parse_args()

    track = args.track
    dpg_dir = Path(args.dpg_root) / track
    vipe_dir = Path(args.vipe_root) / track
    hybrid_dir = Path(args.hybrid_root) / track

    # Load global s, t from hybrid output
    h_npz = next(hybrid_dir.glob('*/pointcloud/*foreground_5_views_aligned_smooth.npz'), None)
    if h_npz is None:
        print(f"Error: no hybrid output at {hybrid_dir}")
        return
    h = np.load(str(h_npz))
    s = h['global_s']
    t = h['global_t']
    print(f"Track: {track}")
    print(f"Global transform: s={s}, t={t}")

    # Find frames present in both
    dpg_frames = sorted(d.name for d in dpg_dir.iterdir()
                       if d.is_dir() and d.name.isdigit())
    frame_ids = []
    for fid in dpg_frames:
        dpg_npz = dpg_dir / fid / 'pointcloud' / f'{fid}_foreground_1_view.npz'
        vipe_npz = vipe_dir / fid / 'pointcloud' / f'{fid}_foreground_1_view.npz'
        if dpg_npz.exists() and vipe_npz.exists():
            frame_ids.append(fid)

    print(f"Loading {len(frame_ids)} frames...")

    # Pre-load all frames
    dpg_frames_pts = []
    vipe_f1v_frames_pts = []
    vipe_out_frames_pts = []
    vipe_ply_frames_pts = []
    vipe_ply_frames_cols = []
    for fid in frame_ids:
        pts_dpg = load_frame_points(dpg_dir / fid / 'pointcloud' / f'{fid}_foreground_1_view.npz')
        pts_vipe_f1v = load_frame_points(vipe_dir / fid / 'pointcloud' / f'{fid}_foreground_1_view.npz')

        # VIPE alignment OUTPUT (foreground_5_views_aligned_smooth.npz)
        vipe_out_path = vipe_dir / fid / 'pointcloud' / f'{fid}_foreground_5_views_aligned_smooth.npz'
        if vipe_out_path.exists():
            pts_vipe_out = load_aligned_smooth(vipe_out_path)
        else:
            pts_vipe_out = np.zeros((0, 3))

        # VIPE foreground_1_view.PLY directly (real RGB colors)
        vipe_ply_path = vipe_dir / fid / 'pointcloud' / f'{fid}_foreground_1_view.ply'
        if vipe_ply_path.exists():
            pts_ply, cols_ply = load_ply_with_colors(vipe_ply_path)
        else:
            pts_ply = np.zeros((0, 3))
            cols_ply = np.zeros((0, 3), dtype=np.uint8)

        # Transform DPG to C
        pts_dpg_in_C = s * pts_dpg + t

        # Subsample
        if args.fg_subsample > 1:
            pts_dpg_in_C = pts_dpg_in_C[::args.fg_subsample]
            pts_vipe_f1v = pts_vipe_f1v[::args.fg_subsample]
            pts_vipe_out = pts_vipe_out[::args.fg_subsample]
            pts_ply = pts_ply[::args.fg_subsample]
            cols_ply = cols_ply[::args.fg_subsample]

        dpg_frames_pts.append(to_z_up(pts_dpg_in_C))
        vipe_f1v_frames_pts.append(to_z_up(pts_vipe_f1v))
        vipe_out_frames_pts.append(to_z_up(pts_vipe_out))
        vipe_ply_frames_pts.append(to_z_up(pts_ply))
        vipe_ply_frames_cols.append(cols_ply)

    print(f"Loaded {len(frame_ids)} frames")

    # Background (static)
    bg_pts = bg_cols = None
    if not args.no_bg:
        bg_ply = vipe_dir / 'global_background.ply'
        if bg_ply.exists():
            bg = trimesh.load(str(bg_ply))
            bg_pts_raw = np.array(bg.vertices)[::args.bg_subsample]
            bg_pts = to_z_up(bg_pts_raw)
            bg_cols = np.tile(COLOR_BG, (len(bg_pts), 1))
            print(f"Background: {len(bg_pts)} pts (subsample={args.bg_subsample})")

    # Start viser
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout='collapsible')

    # Background (static, added once)
    if bg_pts is not None:
        server.scene.add_point_cloud('bg', bg_pts, bg_cols, point_size=args.point_size * 0.7)

    # Initial frame
    dpg_handle = server.scene.add_point_cloud(
        'fg/dpg_in_C',
        dpg_frames_pts[0],
        np.tile(COLOR_DPG, (len(dpg_frames_pts[0]), 1)),
        point_size=args.point_size,
    )
    vipe_f1v_handle = server.scene.add_point_cloud(
        'fg/vipe_f1v',
        vipe_f1v_frames_pts[0],
        np.tile(COLOR_VIPE_F1V, (len(vipe_f1v_frames_pts[0]), 1)),
        point_size=args.point_size,
    )
    vipe_out_handle = server.scene.add_point_cloud(
        'fg/vipe_aligned_smooth',
        vipe_out_frames_pts[0],
        np.tile(COLOR_VIPE_OUT, (len(vipe_out_frames_pts[0]), 1)),
        point_size=args.point_size,
    )
    vipe_ply_handle = server.scene.add_point_cloud(
        'fg/vipe_f1v_ply',
        vipe_ply_frames_pts[0],
        vipe_ply_frames_cols[0],  # real RGB by default
        point_size=args.point_size,
        visible=False,  # off by default — RGB conflicts visually with overlays
    )

    # GUI
    with server.gui.add_folder('Playback'):
        gui_frame = server.gui.add_slider('Frame', min=0, max=len(frame_ids)-1, step=1,
                                          initial_value=0)
        gui_fid = server.gui.add_text('Frame ID', initial_value=frame_ids[0], disabled=True)
        gui_play = server.gui.add_button('Play', icon=viser.Icon.PLAYER_PLAY)
        gui_pause = server.gui.add_button('Pause', icon=viser.Icon.PLAYER_PAUSE, visible=False)
        gui_fps = server.gui.add_slider('FPS', min=1, max=30, step=0.5, initial_value=args.fps)

    with server.gui.add_folder('Display'):
        gui_show_dpg = server.gui.add_checkbox('DPG → C (red)', True)
        gui_show_vipe_f1v = server.gui.add_checkbox('VIPE foreground_1_view (blue)', True)
        gui_show_vipe_out = server.gui.add_checkbox('VIPE aligned_smooth (green)', True)
        gui_show_vipe_ply = server.gui.add_checkbox('VIPE foreground_1_view.ply (real RGB)', False)
        gui_ply_use_yellow = server.gui.add_checkbox('  ↳ override RGB → yellow', False)

    is_playing = [False]

    def update_frame(idx):
        idx = int(idx) % len(frame_ids)
        gui_fid.value = frame_ids[idx]
        # Update positions
        dpg_pts = dpg_frames_pts[idx]
        vipe_f1v_pts = vipe_f1v_frames_pts[idx]
        vipe_out_pts = vipe_out_frames_pts[idx]
        vipe_ply_pts = vipe_ply_frames_pts[idx]
        vipe_ply_cols = vipe_ply_frames_cols[idx]
        dpg_handle.points = dpg_pts
        dpg_handle.colors = np.tile(COLOR_DPG, (len(dpg_pts), 1))
        vipe_f1v_handle.points = vipe_f1v_pts
        vipe_f1v_handle.colors = np.tile(COLOR_VIPE_F1V, (len(vipe_f1v_pts), 1))
        vipe_out_handle.points = vipe_out_pts
        vipe_out_handle.colors = np.tile(COLOR_VIPE_OUT, (len(vipe_out_pts), 1))
        vipe_ply_handle.points = vipe_ply_pts
        if gui_ply_use_yellow.value:
            vipe_ply_handle.colors = np.tile(COLOR_VIPE_F1V_PLY, (len(vipe_ply_pts), 1))
        else:
            vipe_ply_handle.colors = vipe_ply_cols

    @gui_frame.on_update
    def _(_):
        update_frame(gui_frame.value)

    @gui_play.on_click
    def _(_):
        is_playing[0] = True
        gui_play.visible = False
        gui_pause.visible = True

    @gui_pause.on_click
    def _(_):
        is_playing[0] = False
        gui_play.visible = True
        gui_pause.visible = False

    @gui_show_dpg.on_update
    def _(_):
        dpg_handle.visible = gui_show_dpg.value

    @gui_show_vipe_f1v.on_update
    def _(_):
        vipe_f1v_handle.visible = gui_show_vipe_f1v.value

    @gui_show_vipe_out.on_update
    def _(_):
        vipe_out_handle.visible = gui_show_vipe_out.value

    @gui_show_vipe_ply.on_update
    def _(_):
        vipe_ply_handle.visible = gui_show_vipe_ply.value

    @gui_ply_use_yellow.on_update
    def _(_):
        update_frame(gui_frame.value)

    print(f"\nViser running on port {args.port}")
    print("  RED    = DPG foreground_1_view → C (via hybrid 4-DOF)")
    print("  BLUE   = VIPE foreground_1_view (loaded from .npz, alignment INPUT)")
    print("  GREEN  = VIPE foreground_5_views_aligned_smooth (alignment OUTPUT)")
    print("  RGB    = VIPE foreground_1_view.PLY (toggle on, real RGB; or yellow)")
    if bg_pts is not None:
        print("  GRAY   = VIPE background")
    print("\nPress Ctrl+C to exit")

    # Playback loop
    try:
        while True:
            if is_playing[0]:
                next_idx = (int(gui_frame.value) + 1) % len(frame_ids)
                gui_frame.value = next_idx
            time.sleep(1.0 / max(float(gui_fps.value), 1e-3))
    except KeyboardInterrupt:
        print("\nExiting")


if __name__ == '__main__':
    main()
