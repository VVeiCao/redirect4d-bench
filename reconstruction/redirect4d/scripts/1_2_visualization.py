#!/usr/bin/env python3
"""Aligned point cloud visualization with Viser (global background + aligned foreground)."""

import os
import sys
import glob
import time
import argparse
import json
from typing import List, Dict, Optional
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation
import matplotlib


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1.2: Aligned point cloud visualization")
    parser.add_argument("--data_dir", type=str, default="outputs/prepared/camel", help="Data directory")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--point_size", type=float, default=0.001, help="Point size")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback frame rate")
    parser.add_argument("--subsample", type=int, default=4, help="Subsample rate")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to load")
    parser.add_argument("--show_bbox", action="store_true", help="Show AABB bounding boxes")
    return parser.parse_args()


class AlignedPointcloudViewer:
    """Viewer for aligned 5-view foreground point clouds in background space."""

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
        self.camera_data = self._load_camera_parameters()
        self.foreground_frames = self._load_aligned_foreground_frames()
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

    def _load_camera_parameters(self) -> Optional[Dict]:
        """Load global camera parameters."""
        camera_json_path = os.path.join(self.data_dir, "global_camera.json")
        if not os.path.exists(camera_json_path):
            return None

        try:
            with open(camera_json_path, 'r') as f:
                camera_data = json.load(f)
            camera_data_processed = {
                frame_id: {'extrinsic': np.array(cam['extrinsic']), 'intrinsic': np.array(cam['intrinsic'])}
                for frame_id, cam in camera_data.items() if frame_id.isdigit()
            }
            return camera_data_processed
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            return None

    def _load_aligned_foreground_frames(self) -> List[Dict]:
        """Load per-frame aligned foreground point clouds (original, aligned, and smoothed versions)."""
        subdirs = sorted([d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()])

        if self.num_frames_limit:
            subdirs = subdirs[:self.num_frames_limit]

        foreground_list = []
        num_with_smooth = 0
        num_with_fg1 = 0

        for frame_id in subdirs:
            fg_path = os.path.join(self.data_dir, frame_id, 'pointcloud', f"{frame_id}_foreground_5_views_aligned.ply")
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

                frame_data = {
                    'frame_id': frame_id,
                    'points': points,
                    'colors': colors,
                    'points_smooth': None,
                    'colors_smooth': None,
                    'points_fg1': None,
                    'colors_fg1': None
                }

                fg_path_fg1 = os.path.join(self.data_dir, frame_id, 'pointcloud', f"{frame_id}_foreground_1_view.ply")
                if os.path.exists(fg_path_fg1):
                    try:
                        mesh_fg1 = trimesh.load(fg_path_fg1)
                        points_fg1 = np.array(mesh_fg1.vertices)
                        colors_fg1 = (np.array(mesh_fg1.visual.vertex_colors)[:, :3] if hasattr(mesh_fg1, 'visual')
                                     and hasattr(mesh_fg1.visual, 'vertex_colors')
                                     else np.ones((len(points_fg1), 3), dtype=np.uint8) * [255, 0, 255])

                        if self.subsample > 1:
                            indices_fg1 = np.arange(0, len(points_fg1), self.subsample)
                            points_fg1, colors_fg1 = points_fg1[indices_fg1], colors_fg1[indices_fg1]

                        frame_data['points_fg1'] = points_fg1
                        frame_data['colors_fg1'] = colors_fg1
                        num_with_fg1 += 1
                    except Exception:
                        pass

                fg_path_smooth = os.path.join(self.data_dir, frame_id, 'pointcloud', f"{frame_id}_foreground_5_views_aligned_smooth.ply")
                if os.path.exists(fg_path_smooth):
                    try:
                        mesh_smooth = trimesh.load(fg_path_smooth)
                        points_smooth = np.array(mesh_smooth.vertices)
                        colors_smooth = (np.array(mesh_smooth.visual.vertex_colors)[:, :3] if hasattr(mesh_smooth, 'visual')
                                        and hasattr(mesh_smooth.visual, 'vertex_colors')
                                        else np.ones((len(points_smooth), 3), dtype=np.uint8) * [0, 255, 0])

                        if self.subsample > 1:
                            indices_smooth = np.arange(0, len(points_smooth), self.subsample)
                            points_smooth, colors_smooth = points_smooth[indices_smooth], colors_smooth[indices_smooth]

                        frame_data['points_smooth'] = points_smooth
                        frame_data['colors_smooth'] = colors_smooth
                        num_with_smooth += 1
                    except Exception:
                        pass

                foreground_list.append(frame_data)

            except Exception as e:
                print(f"Error loading frame {frame_id}: {e}")

        print(f"Loaded {len(foreground_list)} frames, {num_with_fg1} with fg1, {num_with_smooth} with smoothed version")
        self.has_smooth_data = num_with_smooth > 0
        self.has_fg1_data = num_with_fg1 > 0

        return foreground_list

    def _setup_viser(self) -> viser.ViserServer:
        """Set up Viser server."""
        server = viser.ViserServer(host="0.0.0.0", port=self.port)
        server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self._create_gui(server)
        self._create_scene(server)

        return server

    def _create_gui(self, server: viser.ViserServer):
        """Create GUI controls."""
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
            self.gui_show_camera = server.gui.add_checkbox("Show Camera", True)
            self.gui_show_axes = server.gui.add_checkbox("Show Axes", True)
            self.gui_show_bbox = server.gui.add_checkbox("Show Foreground Bbox", True)

            server.gui.add_markdown("**Foreground Version** (multi-select)")

            if self.has_fg1_data:
                self.gui_show_fg1 = server.gui.add_checkbox("Show Original FG1", False)
            else:
                self.gui_show_fg1 = None

            self.gui_show_aligned = server.gui.add_checkbox("Show Aligned", True)

            if self.has_smooth_data:
                self.gui_show_smooth = server.gui.add_checkbox("Show Smoothed", False)
            else:
                self.gui_show_smooth = None

            server.gui.add_text(
                "Coordinate System",
                initial_value="Axes: Y-up (standard 3D)\nCamera: Y-down (OpenCV)",
                disabled=True,
            )

            self.gui_show_mode = server.gui.add_button_group(
                "Display Mode",
                ("4D (Current Frame)", "3D (All Frames)")
            )

            self.gui_point_size = server.gui.add_slider(
                "Point Size",
                min=0.001,
                max=0.01,
                step=0.0001,
                initial_value=self.point_size
            )

        with server.gui.add_folder("Statistics"):
            bg_points = len(self.background_data['points']) if self.background_data else 0
            fg_points = sum(len(pc['points']) for pc in self.foreground_frames)
            num_with_fg1 = sum(1 for pc in self.foreground_frames if pc['points_fg1'] is not None)
            num_with_smooth = sum(1 for pc in self.foreground_frames if pc['points_smooth'] is not None)

            info_text = f"Script: Step 1.2 (aligned to background space)\n"
            info_text += f"Background points: {bg_points:,}\n"
            info_text += f"Foreground frames: {self.num_frames}\n"
            if self.has_fg1_data:
                info_text += f"FG1 original frames: {num_with_fg1}\n"
            if self.has_smooth_data:
                info_text += f"Smoothed frames: {num_with_smooth}\n"
            info_text += f"Foreground total points (aligned): {fg_points:,}\n"
            info_text += f"Total points: {bg_points + fg_points:,}"

            self.gui_stats = server.gui.add_text(
                "Stats",
                initial_value=info_text,
                disabled=True,
            )

        self.server = server

    def _create_scene(self, server: viser.ViserServer):
        """Create 3D scene."""
        self.background_handle = None
        if self.background_data:
            self.background_handle = server.scene.add_point_cloud(
                name="/pointcloud/global_background",
                points=self.background_data['points'],
                colors=self.background_data['colors'],
                point_size=self.point_size,
                point_shape="circle",
            )

        self.foreground_handles_fg1 = []
        self.foreground_handles = []
        self.foreground_handles_smooth = []

        for pc_data in self.foreground_frames:
            frame_id = pc_data['frame_id']

            if pc_data['points_fg1'] is not None:
                handle_fg1 = server.scene.add_point_cloud(
                    name=f"/pointcloud/foreground_fg1_{frame_id}",
                    points=pc_data['points_fg1'],
                    colors=pc_data['colors_fg1'],
                    point_size=self.point_size,
                    point_shape="circle",
                )
                self.foreground_handles_fg1.append(handle_fg1)
            else:
                self.foreground_handles_fg1.append(None)

            handle = server.scene.add_point_cloud(
                name=f"/pointcloud/foreground_aligned_{frame_id}",
                points=pc_data['points'],
                colors=pc_data['colors'],
                point_size=self.point_size,
                point_shape="circle",
            )
            self.foreground_handles.append(handle)

            if pc_data['points_smooth'] is not None:
                handle_smooth = server.scene.add_point_cloud(
                    name=f"/pointcloud/foreground_aligned_smooth_{frame_id}",
                    points=pc_data['points_smooth'],
                    colors=pc_data['colors_smooth'],
                    point_size=self.point_size,
                    point_shape="circle",
                )
                self.foreground_handles_smooth.append(handle_smooth)
            else:
                self.foreground_handles_smooth.append(None)

        self.camera_handles = []
        if self.camera_data:
            self._create_cameras(server)

        self.axes_handle = None
        self._create_coordinate_axes(server)

        self.bbox_handles_fg1 = []
        self.bbox_handles = []
        self.bbox_handles_smooth = []
        self._create_all_bboxes(server)

        self._update_display()
        self._bind_events()

    def _create_all_bboxes(self, server: viser.ViserServer):
        """Create AABB bounding boxes for all foreground versions."""
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

        for i, fg_data in enumerate(self.foreground_frames):
            frame_id = fg_data['frame_id']

            # FG1 original bbox (magenta)
            points_fg1 = fg_data['points_fg1']
            if points_fg1 is not None and len(points_fg1) > 0:
                bbox_min_fg1 = np.min(points_fg1, axis=0)
                bbox_max_fg1 = np.max(points_fg1, axis=0)

                bbox_points_fg1, bbox_colors_fg1 = create_bbox_lines(bbox_min_fg1, bbox_max_fg1, [255, 0, 255])

                handle_fg1 = server.scene.add_point_cloud(
                    name=f"/reference/foreground_bbox_fg1_{frame_id}",
                    points=bbox_points_fg1,
                    colors=bbox_colors_fg1,
                    point_size=self.point_size * 5.0,
                    point_shape="circle",
                )
                self.bbox_handles_fg1.append(handle_fg1)
            else:
                self.bbox_handles_fg1.append(None)

            # Aligned bbox (yellow)
            points = fg_data['points']
            if len(points) > 0:
                bbox_min = np.min(points, axis=0)
                bbox_max = np.max(points, axis=0)

                bbox_points, bbox_colors = create_bbox_lines(bbox_min, bbox_max, [255, 255, 0])

                handle = server.scene.add_point_cloud(
                    name=f"/reference/foreground_bbox_{frame_id}",
                    points=bbox_points,
                    colors=bbox_colors,
                    point_size=self.point_size * 5.0,
                    point_shape="circle",
                )
                self.bbox_handles.append(handle)
            else:
                self.bbox_handles.append(None)

            # Smoothed bbox (cyan)
            points_smooth = fg_data['points_smooth']
            if points_smooth is not None and len(points_smooth) > 0:
                bbox_min_smooth = np.min(points_smooth, axis=0)
                bbox_max_smooth = np.max(points_smooth, axis=0)

                bbox_points_smooth, bbox_colors_smooth = create_bbox_lines(bbox_min_smooth, bbox_max_smooth, [0, 255, 255])

                handle_smooth = server.scene.add_point_cloud(
                    name=f"/reference/foreground_bbox_smooth_{frame_id}",
                    points=bbox_points_smooth,
                    colors=bbox_colors_smooth,
                    point_size=self.point_size * 5.0,
                    point_shape="circle",
                )
                self.bbox_handles_smooth.append(handle_smooth)
            else:
                self.bbox_handles_smooth.append(None)

    def _create_coordinate_axes(self, server: viser.ViserServer):
        """Create coordinate axes visualization.

        Cameras use OpenCV convention (y-down), but axes use standard 3D (y-up).
        """
        all_points = []
        if self.background_data:
            all_points.append(self.background_data['points'])
        for fg_frame in self.foreground_frames:
            all_points.append(fg_frame['points'])

        if len(all_points) == 0:
            axis_scale = 0.5
        else:
            all_points_combined = np.concatenate(all_points, axis=0)
            bbox_min = np.min(all_points_combined, axis=0)
            bbox_max = np.max(all_points_combined, axis=0)
            bbox_size = bbox_max - bbox_min
            max_size = np.max(bbox_size)
            axis_scale = max_size * 0.1

        origin = np.array([0.0, 0.0, 0.0])

        num_points_per_axis = 50
        axis_points = []
        axis_colors = []

        # X-axis: red (right)
        for t in np.linspace(0, 1, num_points_per_axis):
            point = origin + t * np.array([axis_scale, 0, 0])
            axis_points.append(point)
            axis_colors.append([255, 0, 0])

        # Y-axis: green (up)
        for t in np.linspace(0, 1, num_points_per_axis):
            point = origin + t * np.array([0, axis_scale, 0])
            axis_points.append(point)
            axis_colors.append([0, 255, 0])

        # Z-axis: blue (forward)
        for t in np.linspace(0, 1, num_points_per_axis):
            point = origin + t * np.array([0, 0, axis_scale])
            axis_points.append(point)
            axis_colors.append([0, 0, 255])

        # Origin (white)
        axis_points.append(origin)
        axis_colors.append([255, 255, 255])

        axis_points = np.array(axis_points)
        axis_colors = np.array(axis_colors)

        self.axes_handle = server.scene.add_point_cloud(
            name="/reference/coordinate_axes",
            points=axis_points,
            colors=axis_colors,
            point_size=self.point_size * 3.0,
            point_shape="circle",
        )

    def _create_cameras(self, server: viser.ViserServer):
        """Create camera frustum visualization."""
        if not self.camera_data:
            return

        try:
            cmap = matplotlib.colormaps['gist_rainbow']
        except (AttributeError, KeyError):
            from matplotlib.cm import get_cmap
            cmap = get_cmap('gist_rainbow')

        image_height = 480
        image_width = 640

        frame_ids_sorted = sorted(self.camera_data.keys())

        for i, frame_id in enumerate(frame_ids_sorted):
            if frame_id not in self.camera_data:
                continue

            try:
                cam_info = self.camera_data[frame_id]
                extrinsic_3x4 = cam_info['extrinsic']
                intrinsic_3x3 = cam_info['intrinsic']

                T_cam_world = np.vstack([extrinsic_3x4, [0, 0, 0, 1]])
                T_world_cam = np.linalg.inv(T_cam_world)

                position = T_world_cam[:3, 3]
                R_world_cam = T_world_cam[:3, :3]
                wxyz = Rotation.from_matrix(R_world_cam).as_quat()[[3, 0, 1, 2]]

                fx = intrinsic_3x3[0, 0]
                fy = intrinsic_3x3[1, 1]
                fov_y = 2 * np.arctan(image_height / (2 * fy))
                aspect = image_width / image_height

                rgba_color = cmap(i / max(len(frame_ids_sorted) - 1, 1))
                camera_color = tuple(int(255 * x) for x in rgba_color[:3])

                handle = server.scene.add_camera_frustum(
                    name=f"/camera/frame_{frame_id}",
                    fov=fov_y,
                    aspect=aspect,
                    scale=0.05,
                    wxyz=wxyz,
                    position=position,
                    color=camera_color,
                )

                self.camera_handles.append({
                    'handle': handle,
                    'frame_id': frame_id
                })

            except Exception as e:
                print(f"Error creating camera for frame {frame_id}: {e}")
                continue

    def _bind_events(self):
        """Bind GUI events."""
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

        @self.gui_show_camera.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_show_axes.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_show_bbox.on_update
        def _(_) -> None:
            self._update_display()

        if self.gui_show_fg1 is not None:
            @self.gui_show_fg1.on_update
            def _(_) -> None:
                self._update_display()

        @self.gui_show_aligned.on_update
        def _(_) -> None:
            self._update_display()

        if self.gui_show_smooth is not None:
            @self.gui_show_smooth.on_update
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

            for handle in self.foreground_handles_fg1:
                if handle is not None:
                    handle.point_size = new_size

            for handle in self.foreground_handles:
                handle.point_size = new_size

            for handle in self.foreground_handles_smooth:
                if handle is not None:
                    handle.point_size = new_size

            for handle in self.bbox_handles_fg1:
                if handle is not None:
                    handle.point_size = new_size * 5.0

            for handle in self.bbox_handles:
                if handle is not None:
                    handle.point_size = new_size * 5.0

            for handle in self.bbox_handles_smooth:
                if handle is not None:
                    handle.point_size = new_size * 5.0

            if self.axes_handle is not None:
                self.axes_handle.point_size = new_size * 3.0

    def _update_display(self):
        """Update display state (supports multi-version simultaneous display)."""
        current_timestep = self.gui_timestep.value
        show_mode = self.gui_show_mode.value
        show_background = self.gui_show_background.value
        show_foreground = self.gui_show_foreground.value
        show_camera = self.gui_show_camera.value
        show_axes = self.gui_show_axes.value
        show_bbox = self.gui_show_bbox.value

        show_fg1 = self.gui_show_fg1.value if self.gui_show_fg1 is not None else False
        show_aligned = self.gui_show_aligned.value
        show_smooth = self.gui_show_smooth.value if self.gui_show_smooth is not None else False

        if current_timestep < len(self.foreground_frames):
            current_frame = self.foreground_frames[current_timestep]
            self.gui_frame_id_label.value = current_frame['frame_id']

        with self.server.atomic():
            if self.background_handle:
                self.background_handle.visible = show_background

            for i, handle in enumerate(self.foreground_handles_fg1):
                if handle is None:
                    continue
                if not show_foreground or not show_fg1:
                    handle.visible = False
                elif show_mode == "4D (Current Frame)":
                    handle.visible = (i == current_timestep)
                else:
                    handle.visible = True

            for i, handle in enumerate(self.foreground_handles):
                if not show_foreground or not show_aligned:
                    handle.visible = False
                elif show_mode == "4D (Current Frame)":
                    handle.visible = (i == current_timestep)
                else:
                    handle.visible = True

            for i, handle in enumerate(self.foreground_handles_smooth):
                if handle is None:
                    continue
                if not show_foreground or not show_smooth:
                    handle.visible = False
                elif show_mode == "4D (Current Frame)":
                    handle.visible = (i == current_timestep)
                else:
                    handle.visible = True

            for i, handle in enumerate(self.bbox_handles_fg1):
                if handle is None:
                    continue
                if not show_bbox or not show_fg1:
                    handle.visible = False
                elif show_mode == "4D (Current Frame)":
                    handle.visible = (i == current_timestep)
                else:
                    handle.visible = True

            for i, handle in enumerate(self.bbox_handles):
                if handle is None:
                    continue
                if not show_bbox or not show_aligned:
                    handle.visible = False
                elif show_mode == "4D (Current Frame)":
                    handle.visible = (i == current_timestep)
                else:
                    handle.visible = True

            for i, handle in enumerate(self.bbox_handles_smooth):
                if handle is None:
                    continue
                if not show_bbox or not show_smooth:
                    handle.visible = False
                elif show_mode == "4D (Current Frame)":
                    handle.visible = (i == current_timestep)
                else:
                    handle.visible = True

            if self.camera_handles:
                current_frame_id = None
                if current_timestep < len(self.foreground_frames):
                    current_frame_id = self.foreground_frames[current_timestep]['frame_id']

                for cam_info in self.camera_handles:
                    if not show_camera:
                        cam_info['handle'].visible = False
                    elif show_mode == "4D (Current Frame)":
                        cam_info['handle'].visible = (cam_info['frame_id'] == current_frame_id)
                    else:
                        cam_info['handle'].visible = True

            if self.axes_handle:
                self.axes_handle.visible = show_axes

    def run(self):
        """Run the viewer."""
        print(f"Step 1.2 aligned point cloud viewer started at http://localhost:{self.port}")
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
        viewer = AlignedPointcloudViewer(
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
