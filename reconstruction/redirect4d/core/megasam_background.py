"""Background point cloud generator using MegaSaM (Depth-Anything + DROID v2).

Reads a precomputed MegaSaM cvd_opt output cache (`{scene}_sgd_cvd_hr.npz`)
that contains:
  - images: (T, H, W, 3) uint8
  - depths: (T, H, W) float (metric)
  - intrinsic: (3, 3)
  - cam_c2w: (T, 4, 4)

Converts to the same format as VIPeBackgroundGenerator so that alignment +
rendering code needs zero changes.

Workflow:
  1. Resize MegaSaM output to the target image resolution (480x832 by default).
  2. For each frame, backproject depth + camera → world points.
  3. Use VGGT foreground mask (from `foreground_5_views.npz`) to split fg/bg.
  4. Save:
     - `{frame_id}_foreground_1_view.npz/.ply`  (per-frame backprojection)
     - `global_background.ply`                  (aggregated background)
     - `global_camera.json`                     (per-frame camera params)

The MegaSaM cache is produced separately by `scripts/megasam_run_batch.py`
(which runs in the `mega_sam` conda env on a GPU).
"""

import os
import cv2
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from utils.file_io import find_frame_dirs, ensure_dir
from utils.logging import setup_logger

logger = setup_logger('megasam_background')


def _depth_edge_mask(depth: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    """Detect depth discontinuity edges via relative gradient thresholding."""
    grad_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    grad_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
    safe_d = np.maximum(np.abs(depth), 1e-8)
    return (grad_x / safe_d > rtol) | (grad_y / safe_d > rtol)


class MegaSaMBackgroundGenerator:
    """Background point cloud generator using MegaSaM precomputed output."""

    def __init__(self,
                 cache_dir: str,
                 pad_pixels: int = 5,
                 voxel_size: float = 0.001,
                 outlier_nb_neighbors: int = 500,
                 outlier_std_ratio: float = 1.5,
                 confidence_threshold: float = 1.0,
                 depth_edge_rtol: float = 0.01,
                 depth_min: float = 0.05,
                 depth_max: float = 100.0):
        self.cache_dir = str(cache_dir)
        self.pad_pixels = pad_pixels
        self.voxel_size = voxel_size
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.confidence_threshold = confidence_threshold
        self.depth_edge_rtol = depth_edge_rtol
        self.depth_min = depth_min
        self.depth_max = depth_max

    # ------------------------------------------------------------------
    # MegaSaM cache loading
    # ------------------------------------------------------------------

    def _load_cache(self, scene_name: str) -> dict:
        """Load MegaSaM cvd_opt output for a scene."""
        path = Path(self.cache_dir) / f'{scene_name}_sgd_cvd_hr.npz'
        if not path.exists():
            raise FileNotFoundError(
                f"MegaSaM cache not found: {path}. "
                f"Run scripts/megasam_run_batch.py first."
            )
        d = np.load(str(path))
        return {
            'images': d['images'],          # (T, H, W, 3) uint8
            'depths': np.float32(d['depths']),  # (T, H, W) float
            'intrinsic': np.float32(d['intrinsic']),  # (3, 3)
            'cam_c2w': np.float32(d['cam_c2w']),  # (T, 4, 4)
        }

    @staticmethod
    def _backproject_depth(depth, c2w, fx, fy, cx, cy):
        """Backproject (H, W) depth + camera to (H, W, 3) world points."""
        H, W = depth.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        Z = depth.astype(np.float32)
        X = (xx - cx) * Z / fx
        Y = (yy - cy) * Z / fy
        pts_cam = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        # Apply c2w
        pts_cam_h = np.concatenate([pts_cam, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
        pts_world_h = pts_cam_h @ c2w.T
        return pts_world_h[..., :3]

    @staticmethod
    def _c2w_to_w2c_3x4(c2w: np.ndarray) -> np.ndarray:
        c2w_4 = np.eye(4, dtype=np.float32)
        if c2w.shape == (4, 4):
            c2w_4 = c2w
        else:
            c2w_4[:3] = c2w
        w2c = np.linalg.inv(c2w_4)
        return w2c[:3].astype(np.float32)

    def _compute_padded_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        """Dilate foreground mask by `pad_pixels` to exclude foreground neighborhood."""
        if self.pad_pixels <= 0:
            return fg_mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * self.pad_pixels + 1, 2 * self.pad_pixels + 1))
        padded = cv2.dilate(fg_mask.astype(np.uint8) * 255, kernel, iterations=1)
        return padded > 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_global_background(self, data_dir: str):
        """Generate global background + per-frame foreground_1_view from MegaSaM cache.

        Output files match VIPeBackgroundGenerator format exactly.

        `data_dir` is the per-track prepared dir, e.g.,
        `outputs_0410/prepared_megasam/<track>`. The scene_name is derived from
        the directory name.
        """
        data_dir = Path(data_dir)
        scene_name = data_dir.name

        cache = self._load_cache(scene_name)
        cache_images = cache['images']  # (T, H_ms, W_ms, 3)
        cache_depths = cache['depths']  # (T, H_ms, W_ms)
        K_ms = cache['intrinsic']
        c2w_ms = cache['cam_c2w']  # (T, 4, 4)

        T_ms = cache_depths.shape[0]
        H_ms, W_ms = cache_depths.shape[1:3]
        logger.info(f"Loaded MegaSaM cache for {scene_name}: T={T_ms}, "
                    f"resolution={H_ms}x{W_ms}")

        frame_ids = find_frame_dirs(str(data_dir))
        if not frame_ids:
            raise ValueError(f"No frames found: {data_dir}")
        if len(frame_ids) != T_ms:
            logger.warning(f"Frame count mismatch: dir has {len(frame_ids)}, "
                           f"cache has {T_ms}. Using min.")
        n_use = min(len(frame_ids), T_ms)

        bg_points_all, bg_colors_all, bg_confs_all = [], [], []
        camera_data = {}
        image_size = None

        for frame_idx in tqdm(range(n_use), desc='MegaSaM background', unit='frame'):
            frame_id = frame_ids[frame_idx]

            ms_image = cache_images[frame_idx]  # uint8
            ms_depth = cache_depths[frame_idx]
            c2w = c2w_ms[frame_idx]

            # Load original-resolution image for output (preserve high-res RGB)
            img_path = data_dir / frame_id / 'images' / f'{frame_id}_original.png'
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning(f"Cannot read {img_path}, falling back to MegaSaM RGB")
                img = ms_image.astype(np.float32) / 255.0
            else:
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            H_orig, W_orig = img.shape[:2]

            if image_size is None:
                image_size = (H_orig, W_orig)

            # Resize MegaSaM depth to original resolution if needed
            fx, fy, cx, cy = K_ms[0, 0], K_ms[1, 1], K_ms[0, 2], K_ms[1, 2]
            if (H_ms, W_ms) != (H_orig, W_orig):
                scale_x = W_orig / W_ms
                scale_y = H_orig / H_ms
                ms_depth = cv2.resize(ms_depth, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
                fx = float(fx) * scale_x
                fy = float(fy) * scale_y
                cx = float(cx) * scale_x
                cy = float(cy) * scale_y

            # Load foreground mask from VGGT (pixel-exact, same convention as VIPE)
            vggt_path = data_dir / frame_id / 'pointcloud' / f'{frame_id}_foreground_5_views.npz'
            if vggt_path.exists():
                fg_mask = np.load(str(vggt_path))['foreground_masks'][0]
            else:
                mask_path = data_dir / frame_id / 'masks' / f'{frame_id}_foreground_mask.png'
                fg_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
                logger.warning(f"VGGT output not found for {frame_id}, using raw mask")

            # Backproject depth + camera → world coords
            pts_world = self._backproject_depth(ms_depth, c2w, fx, fy, cx, cy)

            # Validity + edge filter
            valid = (ms_depth > self.depth_min) & (ms_depth < self.depth_max) & np.isfinite(ms_depth)
            valid = valid & ~_depth_edge_mask(ms_depth, rtol=self.depth_edge_rtol)

            confidence = np.zeros((H_orig, W_orig), dtype=np.float32)
            confidence[valid] = 1.0

            padded_mask = self._compute_padded_mask(fg_mask)

            # Background pixels (excluding foreground + padded neighborhood)
            bg_mask = ~padded_mask & valid
            bg_points_all.append(pts_world[bg_mask])
            bg_colors_all.append(img[bg_mask])
            bg_confs_all.append(confidence[bg_mask])

            # Save per-frame foreground_1_view (exact VIPE/DPG format)
            w2c_3x4 = self._c2w_to_w2c_3x4(c2w)
            K_per_frame = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            camera_data[frame_id] = {
                'extrinsic': w2c_3x4.tolist(),
                'intrinsic': K_per_frame.tolist(),
            }

            pc_dir = data_dir / frame_id / 'pointcloud'
            ensure_dir(pc_dir)

            fg_points = pts_world[fg_mask]
            fg_colors = img[fg_mask]
            if len(fg_points) > 0:
                fg_pcd = o3d.geometry.PointCloud()
                fg_pcd.points = o3d.utility.Vector3dVector(fg_points)
                fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors)
                o3d.io.write_point_cloud(str(pc_dir / f'{frame_id}_foreground_1_view.ply'), fg_pcd)

            np.savez(str(pc_dir / f'{frame_id}_foreground_1_view.npz'),
                     foreground_pointmap=pts_world,
                     foreground_mask=fg_mask,
                     padded_mask=padded_mask,
                     color=img,
                     confidence=confidence,
                     pad_pixels=self.pad_pixels,
                     original_size=list(image_size),
                     intrinsic=K_per_frame,
                     extrinsic=w2c_3x4)

        # Aggregate background
        global_bg_points = np.concatenate(bg_points_all)
        global_bg_colors = np.concatenate(bg_colors_all)
        global_bg_confs = np.concatenate(bg_confs_all)

        if len(global_bg_points) > 0:
            if self.confidence_threshold > 0:
                threshold_value = np.percentile(global_bg_confs, self.confidence_threshold)
            else:
                threshold_value = 0.0
            conf_mask = (global_bg_confs >= threshold_value) & (global_bg_confs > 1e-5)
            bg_pts = global_bg_points[conf_mask]
            bg_cls = global_bg_colors[conf_mask]

            if len(bg_pts) == 0:
                logger.warning("No background points after confidence filtering")
                return

            bg_pcd = o3d.geometry.PointCloud()
            bg_pcd.points = o3d.utility.Vector3dVector(bg_pts)
            bg_pcd.colors = o3d.utility.Vector3dVector(np.clip(bg_cls, 0, 1))

            if self.voxel_size:
                bg_pcd = bg_pcd.voxel_down_sample(voxel_size=self.voxel_size)

            nb = min(self.outlier_nb_neighbors, len(bg_pcd.points))
            if nb > 1:
                bg_pcd, _ = bg_pcd.remove_statistical_outlier(
                    nb_neighbors=nb, std_ratio=self.outlier_std_ratio)

            bg_ply_path = data_dir / 'global_background.ply'
            o3d.io.write_point_cloud(str(bg_ply_path), bg_pcd)
            logger.info(f"Background saved: {bg_ply_path} "
                        f"({len(global_bg_points):,} -> {len(bg_pcd.points):,} points)")
        else:
            logger.warning("Background point cloud is empty")

        # Save camera params
        camera_json = {
            'image_size': list(image_size) if image_size else [],
            'coordinate_system': 'y-down (original)',
        }
        camera_json.update(camera_data)

        camera_json_path = data_dir / 'global_camera.json'
        with open(camera_json_path, 'w') as f:
            json.dump(camera_json, f, indent=4)
        logger.info(f"Camera params saved: {camera_json_path} ({len(camera_data)} frames)")

        logger.info("MegaSaM background generation complete")

    @classmethod
    def from_config(cls, config):
        """Create generator from config."""
        from utils.config import Config
        if not isinstance(config, Config):
            config = Config(config)

        project_root = Path(__file__).parent.parent
        cache_dir = config.get('stage_1.background.megasam_cache_dir',
                                str(project_root.parent / 'mega-sam' / 'outputs_cvd'))
        if not os.path.isabs(cache_dir):
            cache_dir = str(project_root / cache_dir)

        return cls(
            cache_dir=cache_dir,
            pad_pixels=config.get('stage_1.background.pad_pixels', 5),
            voxel_size=config.get('stage_1.background.voxel_size', 0.001),
            outlier_nb_neighbors=config.get('stage_1.background.outlier_nb_neighbors', 500),
            outlier_std_ratio=config.get('stage_1.background.outlier_std_ratio', 1.5),
            confidence_threshold=config.get('stage_1.background.confidence_threshold', 1.0),
            depth_edge_rtol=config.get('stage_1.background.depth_edge_rtol', 0.01),
            depth_min=config.get('stage_1.background.depth_min', 0.05),
            depth_max=config.get('stage_1.background.depth_max', 100.0),
        )
