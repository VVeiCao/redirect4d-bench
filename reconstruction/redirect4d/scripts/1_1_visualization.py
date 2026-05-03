#!/usr/bin/env python3
"""Background + foreground point cloud visualization with Viser."""

import os
import time
import argparse
from typing import List, Dict, Optional
import numpy as np
import trimesh
import viser


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Step 1.1: Background + foreground point cloud visualization")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory (step 1.1 output)")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--point_size", type=float, default=0.0005, help="Point size")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback frame rate")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample rate")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to load")
    parser.add_argument("--show_bbox", action="store_true", help="Show AABB bounding box")
    return parser.parse_args()


class BackgroundForegroundViewer:
    """Viewer for global background + per-frame foreground point clouds."""

    def __init__(
        self,
        data_dir: str,
        port: int = 8080,
        point_size: float = 0.003,
        fps: float = 5.0,
        subsample: int = 1,
        num_frames: int = None,
        show_bbox: bool = False
    ):
        self.data_dir = data_dir
        self.port = port
        self.point_size = point_size
        self.fps = fps
        self.subsample = subsample
        self.num_frames_limit = num_frames
        self.show_bbox = show_bbox

        self.background_data = self._load_global_background()
        self.foreground_frames = self._load_foreground_frames()
        self.num_frames = len(self.foreground_frames)

        if self.num_frames == 0 and self.background_data is None:
            raise ValueError("No valid point cloud files found")

        self.server = self._setup_viser()

    def _load_global_background(self) -> Optional[Dict]:
        """Load global background point cloud."""
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

            return {'points': points, 'colors': colors}
        except Exception as e:
            print(f"Error loading background: {e}")
            return None

    def _load_foreground_frames(self) -> List[Dict]:
        """Load per-frame foreground point clouds."""
        subdirs = sorted([d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()])
        if self.num_frames_limit:
            subdirs = subdirs[:self.num_frames_limit]

        foreground_list = []
        for frame_id in subdirs:
            fg_path = os.path.join(self.data_dir, frame_id, 'pointcloud', f"{frame_id}_foreground_1_view.ply")
            if not os.path.exists(fg_path):
                continue

            try:
                mesh = trimesh.load(fg_path)
                points = np.array(mesh.vertices)
                colors = (np.array(mesh.visual.vertex_colors)[:, :3] if hasattr(mesh, 'visual')
                         and hasattr(mesh.visual, 'vertex_colors')
                         else np.ones((len(points), 3), dtype=np.uint8) * [0, 255, 0])

                if self.subsample > 1:
                    indices = np.arange(0, len(points), self.subsample)
                    points, colors = points[indices], colors[indices]

                foreground_list.append({'frame_id': frame_id, 'points': points, 'colors': colors})
            except Exception as e:
                print(f"Error loading frame {frame_id}: {e}")

        return foreground_list

    def _setup_viser(self) -> viser.ViserServer:
        """Set up Viser server and create GUI."""
        server = viser.ViserServer(host="0.0.0.0", port=self.port)
        server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self._create_gui(server)
        self._create_scene(server)

        return server

    def _create_gui(self, server: viser.ViserServer):
        """Create GUI control panel."""
        with server.gui.add_folder("Time Control"):
            self.gui_timestep = server.gui.add_slider(
                "Frame Index",
                min=0,
                max=max(self.num_frames - 1, 0),
                step=1,
                initial_value=0,
            )

            self.gui_frame_id_label = server.gui.add_text(
                "Current Frame ID",
                initial_value=self.foreground_frames[0]['frame_id'] if self.foreground_frames else "N/A",
                disabled=True,
            )

            self.gui_playing = server.gui.add_checkbox("Play", False)
            self.gui_framerate = server.gui.add_slider("FPS", min=1, max=30, step=0.5, initial_value=self.fps)

        with server.gui.add_folder("Display Control"):
            self.gui_show_background = server.gui.add_checkbox("Show Background", True)
            self.gui_show_foreground = server.gui.add_checkbox("Show Foreground", True)

            self.gui_show_mode = server.gui.add_button_group(
                "Display Mode",
                ("4D (Current Frame)", "3D (All Frames)")
            )

            self.gui_point_size = server.gui.add_slider(
                "Point Size",
                min=0.0001,
                max=0.01,
                step=0.0001,
                initial_value=self.point_size
            )

        with server.gui.add_folder("Statistics"):
            bg_points = len(self.background_data['points']) if self.background_data else 0
            fg_points = sum(len(pc['points']) for pc in self.foreground_frames)

            info_text = f"Step: 1.1 Background point cloud generation\n"
            info_text += f"Background: {bg_points:,} points\n"
            info_text += f"Foreground frames: {self.num_frames}\n"
            info_text += f"Foreground total: {fg_points:,} points\n"
            info_text += f"Total: {bg_points + fg_points:,} points"

            self.gui_stats = server.gui.add_text(
                "Stats",
                initial_value=info_text,
                disabled=True,
            )

        self.server = server

    def _create_scene(self, server: viser.ViserServer):
        """Create 3D scene with background and foreground point clouds."""
        self.background_handle = None
        if self.background_data:
            self.background_handle = server.scene.add_point_cloud(
                name="/pointcloud/global_background",
                points=self.background_data['points'],
                colors=self.background_data['colors'],
                point_size=self.point_size,
                point_shape="circle",
            )

        self.foreground_handles = []
        for pc_data in self.foreground_frames:
            frame_id = pc_data['frame_id']

            handle = server.scene.add_point_cloud(
                name=f"/pointcloud/foreground_{frame_id}",
                points=pc_data['points'],
                colors=pc_data['colors'],
                point_size=self.point_size,
                point_shape="circle",
            )

            self.foreground_handles.append(handle)

        self.bbox_handle = None
        if self.show_bbox and len(self.foreground_frames) > 0:
            first_fg = self.foreground_frames[0]
            self._create_bbox(server, first_fg['points'])

        self._update_display()
        self._bind_events()

    def _create_bbox(self, server: viser.ViserServer, points: np.ndarray):
        """Create AABB bounding box (yellow wireframe)."""
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)

        def create_bbox_lines(bbox_min, bbox_max, color, num_points_per_edge=50):
            x_min, y_min, z_min = bbox_min
            x_max, y_max, z_max = bbox_max

            vertices = np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max],
            ])

            edges = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7),
            ]

            line_points = []
            for v1_idx, v2_idx in edges:
                v1, v2 = vertices[v1_idx], vertices[v2_idx]
                for t in np.linspace(0, 1, num_points_per_edge):
                    point = v1 + t * (v2 - v1)
                    line_points.append(point)

            return np.array(line_points), np.tile(color, (len(line_points), 1))

        bbox_points, bbox_colors = create_bbox_lines(bbox_min, bbox_max, [255, 255, 0])

        self.bbox_handle = server.scene.add_point_cloud(
            name="/reference/frame_0_bbox",
            points=bbox_points,
            colors=bbox_colors,
            point_size=self.point_size * 5.0,
            point_shape="circle",
        )

    def _bind_events(self):
        """Bind GUI event callbacks."""
        @self.gui_timestep.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = self.gui_playing.value

        @self.gui_show_mode.on_click
        def _(_) -> None:
            self._update_display()

        @self.gui_show_background.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_show_foreground.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_point_size.on_update
        def _(_) -> None:
            self._update_point_size()

    def _update_point_size(self):
        """Update point size for all point clouds."""
        new_size = self.gui_point_size.value
        with self.server.atomic():
            if self.background_handle:
                self.background_handle.point_size = new_size

            for handle in self.foreground_handles:
                handle.point_size = new_size

            if self.bbox_handle is not None:
                self.bbox_handle.point_size = new_size * 5.0

    def _update_display(self):
        """Update display state (frame switching, 4D/3D mode)."""
        current_timestep = self.gui_timestep.value
        show_mode = self.gui_show_mode.value
        show_background = self.gui_show_background.value
        show_foreground = self.gui_show_foreground.value

        if current_timestep < len(self.foreground_frames):
            current_frame = self.foreground_frames[current_timestep]
            self.gui_frame_id_label.value = current_frame['frame_id']

        with self.server.atomic():
            if self.background_handle:
                self.background_handle.visible = show_background

            for i, handle in enumerate(self.foreground_handles):
                if not show_foreground:
                    handle.visible = False
                elif show_mode == "4D (Current Frame)":
                    handle.visible = (i == current_timestep)
                else:
                    handle.visible = True

    def run(self):
        """Start the viewer and enter main loop."""
        print(f"Step 1.1 viewer started at http://localhost:{self.port}")
        print(f"Data directory: {self.data_dir}, {self.num_frames} foreground frames")

        prev_timestep = self.gui_timestep.value
        while True:
            if self.gui_playing.value and self.num_frames > 0:
                next_timestep = (self.gui_timestep.value + 1) % self.num_frames
                self.gui_timestep.value = next_timestep

            if self.gui_timestep.value != prev_timestep:
                self._update_display()
                prev_timestep = self.gui_timestep.value

            time.sleep(1.0 / self.gui_framerate.value)


def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: data directory does not exist: {args.data_dir}")
        return

    try:
        viewer = BackgroundForegroundViewer(
            data_dir=args.data_dir,
            port=args.port,
            point_size=args.point_size,
            fps=args.fps,
            subsample=args.subsample,
            num_frames=args.num_frames,
            show_bbox=args.show_bbox
        )

        viewer.run()

    except ValueError as e:
        print(f"Error: {e}")
        return
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
