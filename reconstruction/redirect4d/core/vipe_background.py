"""Background point cloud generator using vipe (camera poses + depth maps).

vipe is installed in the same environment (pip install -e vipe --no-deps).
This module calls vipe's pipeline API directly — no subprocess needed.
"""

import os
import cv2
import json
import shutil
import sys
import zipfile
import tempfile
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

from utils.file_io import find_frame_dirs, ensure_dir
from utils.logging import setup_logger

logger = setup_logger('vipe_background')


def _depth_edge_mask(depth: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    """Detect depth discontinuity edges via relative gradient thresholding.

    Args:
        depth: (H, W) depth map.
        rtol: Relative tolerance threshold.

    Returns:
        (H, W) bool mask where True = edge pixel (to be excluded).
    """
    grad_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    grad_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
    safe_d = np.maximum(np.abs(depth), 1e-8)
    return (grad_x / safe_d > rtol) | (grad_y / safe_d > rtol)


def _read_exr_from_bytes(exr_bytes: bytes) -> np.ndarray:
    """Read a single-channel EXR depth map from raw bytes.

    Uses OpenEXR because cv2 cannot decode the 'Z' channel name used by vipe.

    Returns:
        (H, W) float32 depth array.
    """
    import OpenEXR
    import Imath

    tmp = tempfile.NamedTemporaryFile(suffix='.exr', delete=False)
    tmp.write(exr_bytes)
    tmp.close()
    try:
        exr = OpenEXR.InputFile(tmp.name)
        dw = exr.header()['dataWindow']
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        raw = exr.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
    finally:
        os.unlink(tmp.name)


class VIPeBackgroundGenerator:
    """Background point cloud generator using vipe depth + camera estimation.

    Produces the exact same output format as BackgroundPointCloudGenerator (DPG),
    so that alignment and rendering code need zero changes.
    """

    def __init__(self,
                 vipe_root: str,
                 vipe_pipeline: str = 'dav3',
                 pad_pixels: int = 5,
                 voxel_size: float = 0.001,
                 outlier_nb_neighbors: int = 500,
                 outlier_std_ratio: float = 1.5,
                 confidence_threshold: float = 5.0,
                 depth_edge_rtol: float = 0.03,
                 depth_min: float = 0.1,
                 depth_max: float = 100.0,
                 mvd_window_size: int = 48):
        self.vipe_root = str(vipe_root)
        self.vipe_pipeline = vipe_pipeline
        self.pad_pixels = pad_pixels
        self.voxel_size = voxel_size
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.confidence_threshold = confidence_threshold
        self.depth_edge_rtol = depth_edge_rtol
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.mvd_window_size = mvd_window_size

    # ------------------------------------------------------------------
    # vipe invocation (direct Python API)
    # ------------------------------------------------------------------

    def _prepare_vipe_input(self, data_dir: str, frame_ids: List[str]) -> str:
        """Create a flat symlink directory for vipe's frame_dir_stream."""
        frames_dir = Path(data_dir) / '_vipe_frames'
        ensure_dir(frames_dir)

        for frame_id in frame_ids:
            src = Path(data_dir) / frame_id / 'images' / f'{frame_id}_original.png'
            dst = frames_dir / f'{frame_id}.png'
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if src.exists():
                dst.symlink_to(src.resolve())
            else:
                raise FileNotFoundError(f"Original image not found: {src}")

        return str(frames_dir)

    def _run_vipe(self, frames_dir: str, output_dir: str):
        """Run vipe pipeline directly via Python API."""
        vipe_root_path = Path(self.vipe_root).resolve()
        if not (vipe_root_path / "vipe").is_dir():
            raise FileNotFoundError(
                f"VIPE source tree not found at {vipe_root_path}. "
                "Run `git submodule update --init --recursive` from the repo root."
            )
        sys.path.insert(0, str(vipe_root_path))
        loaded_vipe = sys.modules.get("vipe")
        if loaded_vipe is not None:
            loaded_path = Path(getattr(loaded_vipe, "__file__", "")).resolve()
            if vipe_root_path not in loaded_path.parents:
                for name in list(sys.modules):
                    if name == "vipe" or name.startswith("vipe."):
                        del sys.modules[name]

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
        from vipe.streams.base import StreamList
        from vipe.pipeline import make_pipeline

        # Register vipe's custom OmegaConf resolvers
        if not OmegaConf.has_resolver("eq"):
            OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
        if not OmegaConf.has_resolver("neq"):
            OmegaConf.register_new_resolver("neq", lambda a, b: a != b)

        ensure_dir(output_dir)

        # Use Hydra compose to properly resolve defaults and interpolations
        vipe_config_dir = str((Path(self.vipe_root) / 'configs').resolve())

        extra_overrides = []
        if os.environ.get("VIPE_NO_OPT_INTR"):
            extra_overrides.append("pipeline.slam.optimize_intrinsics=false")

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=vipe_config_dir, version_base=None):
            cfg = compose(config_name="default",
                          overrides=[
                              f"pipeline={self.vipe_pipeline}",
                              "streams=frame_dir_stream",
                              f"streams.base_path={frames_dir}",
                              "pipeline.output.save_artifacts=true",
                              "pipeline.output.save_viz=false",
                              f"pipeline.output.path={output_dir}",
                          ] + extra_overrides)

        # Resolve all interpolations eagerly and convert to plain container
        # so that downstream code doesn't hit lazy resolution issues
        pipeline_cfg = OmegaConf.to_container(cfg.pipeline, resolve=True)
        pipeline_cfg = OmegaConf.create(pipeline_cfg)
        stream_cfg = OmegaConf.to_container(cfg.streams, resolve=True)
        stream_cfg = OmegaConf.create(stream_cfg)

        # Patch MultiviewDepthProcessor window_size default (vipe is read-only)
        from vipe.pipeline.processors import MultiviewDepthProcessor
        _orig_defaults = MultiviewDepthProcessor.__init__.__defaults__
        # defaults: (model, window_size, overlap_size, secondary_keyframe)
        MultiviewDepthProcessor.__init__.__defaults__ = (
            _orig_defaults[0], self.mvd_window_size, *_orig_defaults[2:]
        )

        logger.info(f"Running vipe pipeline on {frames_dir} (mvd_window_size={self.mvd_window_size})")
        stream_list = StreamList.make(stream_cfg)
        pipeline = make_pipeline(pipeline_cfg)

        try:
            for stream_idx in range(len(stream_list)):
                video_stream = stream_list[stream_idx]
                logger.info(f"  Processing: {video_stream.name()} ({stream_idx + 1}/{len(stream_list)})")
                pipeline.run(video_stream)
        finally:
            MultiviewDepthProcessor.__init__.__defaults__ = _orig_defaults

        logger.info("vipe pipeline completed.")

    # ------------------------------------------------------------------
    # Load vipe outputs
    # ------------------------------------------------------------------

    def _find_case_name(self, vipe_output_dir: str) -> str:
        """Find the case name from vipe depth output directory."""
        depth_dir = Path(vipe_output_dir) / 'depth'
        zips = list(depth_dir.glob('*.zip'))
        if not zips:
            raise FileNotFoundError(f"No depth zip found in {depth_dir}")
        return zips[0].stem

    def _load_depths(self, vipe_output_dir: str, case_name: str, num_frames: int) -> List[np.ndarray]:
        """Load depth maps from vipe's EXR zip archive."""
        zip_path = Path(vipe_output_dir) / 'depth' / f'{case_name}.zip'
        depths = []

        with zipfile.ZipFile(str(zip_path), 'r') as z:
            names = sorted(z.namelist())
            for name in names[:num_frames]:
                exr_bytes = z.read(name)
                depth = _read_exr_from_bytes(exr_bytes)
                depths.append(depth)

        if len(depths) != num_frames:
            logger.warning(f"vipe produced {len(depths)} depth maps, expected {num_frames}")

        return depths

    def _load_poses(self, vipe_output_dir: str, case_name: str) -> np.ndarray:
        """Load c2w poses. Returns (N, 4, 4)."""
        pose_path = Path(vipe_output_dir) / 'pose' / f'{case_name}.npz'
        return np.load(str(pose_path))['data']

    def _load_intrinsics(self, vipe_output_dir: str, case_name: str) -> np.ndarray:
        """Load intrinsics. Returns (N, 4) [fx, fy, cx, cy]."""
        intr_path = Path(vipe_output_dir) / 'intrinsics' / f'{case_name}.npz'
        return np.load(str(intr_path))['data']

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _c2w_to_w2c_3x4(c2w: np.ndarray) -> np.ndarray:
        """Convert 4x4 c2w to 3x4 W2C extrinsic matrix."""
        w2c = np.linalg.inv(c2w.astype(np.float64))
        return w2c[:3, :].astype(np.float32)

    @staticmethod
    def _fxfycxcy_to_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        """Convert [fx, fy, cx, cy] to 3x3 intrinsic matrix."""
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    @staticmethod
    def _backproject_depth(depth: np.ndarray, c2w: np.ndarray,
                           fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        """Back-project depth map to world-space point map. Returns (H, W, 3)."""
        H, W = depth.shape
        ys, xs = np.mgrid[0:H, 0:W]

        d = depth.astype(np.float64)
        x_cam = (xs.astype(np.float64) - cx) * d / fx
        y_cam = (ys.astype(np.float64) - cy) * d / fy
        pts_cam = np.stack([x_cam, y_cam, d], axis=-1)

        R = c2w[:3, :3].astype(np.float64)
        t = c2w[:3, 3].astype(np.float64)
        pts_world = (pts_cam.reshape(-1, 3) @ R.T + t).reshape(H, W, 3)

        return pts_world.astype(np.float32)

    def _compute_padded_mask(self, foreground_mask: np.ndarray) -> np.ndarray:
        """Dilate foreground mask morphologically (same as DPG)."""
        if not np.any(foreground_mask):
            return np.zeros_like(foreground_mask, dtype=bool)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.pad_pixels * 2 + 1, self.pad_pixels * 2 + 1)
        )
        padded = cv2.dilate(
            foreground_mask.astype(np.uint8) * 255, kernel, iterations=1
        )
        return padded > 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_global_background(self, data_dir: str):
        """Generate global background point cloud using vipe.

        Produces the same output files as BackgroundPointCloudGenerator:
        - global_background.ply
        - global_camera.json
        - Per-frame {frame_id}_foreground_1_view.npz / .ply
        """
        frame_ids = find_frame_dirs(data_dir)
        if not frame_ids:
            raise ValueError(f"No frames found: {data_dir}")

        num_frames = len(frame_ids)
        logger.info(f"vipe background: {num_frames} frames in {data_dir}")

        # Step 1-2: Run vipe (skip if outputs already exist)
        vipe_output_dir = str(Path(data_dir) / '_vipe_output')
        depth_dir = Path(vipe_output_dir) / 'depth'
        if depth_dir.exists() and list(depth_dir.glob('*.zip')):
            logger.info(f"vipe outputs already exist, skipping vipe run.")
        else:
            frames_dir = self._prepare_vipe_input(data_dir, frame_ids)
            self._run_vipe(frames_dir, vipe_output_dir)

        # Step 3: Load vipe outputs
        case_name = self._find_case_name(vipe_output_dir)
        depths = self._load_depths(vipe_output_dir, case_name, num_frames)
        poses = self._load_poses(vipe_output_dir, case_name)
        intrinsics = self._load_intrinsics(vipe_output_dir, case_name)

        # Step 4: Per-frame processing
        bg_points_all, bg_colors_all, bg_confs_all = [], [], []
        camera_data = {}
        image_size = None

        for frame_idx in tqdm(range(num_frames), desc="vipe background", unit="frame"):
            frame_id = frame_ids[frame_idx]
            depth = depths[frame_idx]
            c2w = poses[frame_idx]
            fx, fy, cx, cy = intrinsics[frame_idx]

            # Load original image
            img_path = Path(data_dir) / frame_id / 'images' / f'{frame_id}_original.png'
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning(f"Cannot read {img_path}, skipping")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            H_orig, W_orig = img.shape[:2]

            if image_size is None:
                image_size = (H_orig, W_orig)

            # Resize depth if resolution mismatch
            H_d, W_d = depth.shape
            if (H_d, W_d) != (H_orig, W_orig):
                scale_x, scale_y = W_orig / W_d, H_orig / H_d
                depth = cv2.resize(depth, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
                fx, fy, cx, cy = fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y

            # Load foreground mask from VGGT output (pixel-exact for alignment)
            vggt_path = Path(data_dir) / frame_id / 'pointcloud' / f'{frame_id}_foreground_5_views.npz'
            if vggt_path.exists():
                fg_mask = np.load(str(vggt_path))['foreground_masks'][0]
            else:
                mask_path = Path(data_dir) / frame_id / 'masks' / f'{frame_id}_foreground_mask.png'
                fg_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
                logger.warning(f"VGGT output not found for {frame_id}, using raw mask")

            # Back-project depth to world coords
            pts_world = self._backproject_depth(depth, c2w, fx, fy, cx, cy)

            # Validity and confidence
            valid = (depth > self.depth_min) & (depth < self.depth_max) & np.isfinite(depth)
            valid = valid & ~_depth_edge_mask(depth, rtol=self.depth_edge_rtol)

            confidence = np.zeros((H_orig, W_orig), dtype=np.float32)
            confidence[valid] = 1.0

            # Padded (dilated) foreground mask
            padded_mask = self._compute_padded_mask(fg_mask)

            # Separate foreground / background
            bg_mask = ~padded_mask & valid
            bg_points_all.append(pts_world[bg_mask])
            bg_colors_all.append(img[bg_mask])
            bg_confs_all.append(confidence[bg_mask])

            # Convert camera params to DPG format
            w2c_3x4 = self._c2w_to_w2c_3x4(c2w)
            K = self._fxfycxcy_to_K(fx, fy, cx, cy)

            camera_data[frame_id] = {
                'extrinsic': w2c_3x4.tolist(),
                'intrinsic': K.tolist(),
            }

            # Save per-frame foreground (exact DPG format)
            pc_dir = Path(data_dir) / frame_id / 'pointcloud'
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
                     intrinsic=K,
                     extrinsic=w2c_3x4)

        # Step 5: Assemble global background
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

            bg_ply_path = Path(data_dir) / 'global_background.ply'
            o3d.io.write_point_cloud(str(bg_ply_path), bg_pcd)
            logger.info(f"Background saved: {bg_ply_path} "
                        f"({len(global_bg_points):,} -> {len(bg_pcd.points):,} points)")
        else:
            logger.warning("Background point cloud is empty.")

        # Step 6: Save camera params
        camera_json = {
            'image_size': list(image_size) if image_size else [],
            'coordinate_system': 'y-down (original)',
        }
        camera_json.update(camera_data)

        camera_json_path = Path(data_dir) / 'global_camera.json'
        with open(camera_json_path, 'w') as f:
            json.dump(camera_json, f, indent=4)
        logger.info(f"Camera params saved: {camera_json_path} ({len(camera_data)} frames)")

        # Cleanup
        frames_dir_path = Path(data_dir) / '_vipe_frames'
        if frames_dir_path.exists():
            shutil.rmtree(str(frames_dir_path))

        logger.info("vipe background generation complete.")

    @classmethod
    def from_config(cls, config):
        """Create generator from config."""
        from utils.config import Config
        if not isinstance(config, Config):
            config = Config(config)

        project_root = Path(__file__).parent.parent
        vipe_root = config.get('stage_1.background.vipe_root', '../../third_party/vipe')
        vipe_root_path = _resolve_vipe_root(project_root, vipe_root)

        return cls(
            vipe_root=str(vipe_root_path),
            vipe_pipeline=config.get('stage_1.background.vipe_pipeline', 'dav3'),
            pad_pixels=config.get('stage_1.background.pad_pixels', 5),
            voxel_size=config.get('stage_1.background.voxel_size', 0.001),
            outlier_nb_neighbors=config.get('stage_1.background.outlier_nb_neighbors', 500),
            outlier_std_ratio=config.get('stage_1.background.outlier_std_ratio', 1.5),
            confidence_threshold=config.get('stage_1.background.confidence_threshold', 5.0),
            depth_edge_rtol=config.get('stage_1.background.depth_edge_rtol', 0.03),
            depth_min=config.get('stage_1.background.depth_min', 0.1),
            depth_max=config.get('stage_1.background.depth_max', 100.0),
            mvd_window_size=config.get('stage_1.background.mvd_window_size', 48),
        )


def _resolve_vipe_root(project_root: Path, configured: str) -> Path:
    """Resolve the VIPE source tree from packaged and debug layouts.

    In the public repo, Redirect4D lives at reconstruction/redirect4d and the
    VIPE submodule lives at third_party/vipe, so ../../third_party/vipe works.
    Debug/clone simulations often copy reconstruction/redirect4d to a temporary
    r4d folder and attach vipe as an adjacent symlink. Try both layouts.
    """
    env_root = os.environ.get("REDIRECT4D_VIPE_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    configured_path = Path(configured)
    candidates.append(configured_path if configured_path.is_absolute() else project_root / configured_path)
    candidates.extend(
        [
            project_root / "vipe",
            project_root.parent / "third_party" / "vipe",
            project_root.parent.parent / "third_party" / "vipe",
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "vipe").is_dir() and (resolved / "configs").is_dir():
            return resolved

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "VIPE source tree not found. Run `git submodule update --init --recursive` "
        "from the repo root, or set REDIRECT4D_VIPE_ROOT.\n"
        f"Tried:\n{tried}"
    )
