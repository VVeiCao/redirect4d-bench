"""Reconstruct benchmark source tracks from original videos and metadata.

This mirrors the canonical released-data path:

* clip-frame ids are frozen in metadata and are not recomputed from FPS;
* when the released scene clip is available, frames are decoded from that clip with
  MoviePy at the frozen clip FPS;
* AIM's crop helper zero-pads then uses OpenCV's default linear resize;
* the exported source RGB was written as JPEG quality 95, so regenerated
  frames pass through the same JPEG round-trip before being stored as PNGs;
* the track mp4 is encoded at the benchmark playback FPS, normally 15 fps.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def _ffmpeg_bin() -> str:
    env_ffmpeg = Path(sys.executable).resolve().parent / "ffmpeg"
    if env_ffmpeg.exists():
        return str(env_ffmpeg)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError("ffmpeg not found. Install it in the conda environment.")


def crop_and_resize(frame, box: list[int], output_resolution: tuple[int, int]):
    import cv2
    import numpy as np

    x_min, y_min, x_max, y_max = [int(v) for v in box]
    out_w, out_h = [int(v) for v in output_resolution]
    crop_height, crop_width = y_max - y_min, x_max - x_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    left = int(x_center - crop_width / 2)
    top = int(y_center - crop_height / 2)
    right = int(x_center + crop_width / 2)
    bottom = int(y_center + crop_height / 2)

    pad_left, pad_top = max(0, -left), max(0, -top)
    left, top = max(0, left), max(0, top)
    right, bottom = min(frame.shape[1], right), min(frame.shape[0], bottom)
    cropped = frame[top:bottom, left:right]

    channels = frame.shape[2] if frame.ndim == 3 else 1
    padded = np.zeros((crop_height, crop_width, channels), dtype=frame.dtype).squeeze()
    padded[pad_top : pad_top + cropped.shape[0], pad_left : pad_left + cropped.shape[1]] = cropped
    return cv2.resize(padded, (out_w, out_h))


RELEASE_JPEG_QUALITY = 95


def release_rgb_roundtrip(frame):
    """Match released source RGB export: cv2 JPEG quality 95, then decode."""

    import cv2

    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, RELEASE_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed while matching released source RGB")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed while matching released source RGB")
    return decoded


def read_moviepy_clip_frames(clip_path: Path, frame_ids: list[int], fps: float | None = None):
    """Read RGB frames by clip-frame index using the release MoviePy path."""

    wanted = {int(frame_id) for frame_id in frame_ids}
    if not wanted:
        return {}
    max_frame_id = max(wanted)
    try:
        from moviepy.editor import VideoFileClip
    except ImportError as exc:
        raise ImportError(
            "moviepy is required for exact clip-frame reconstruction. "
            "Install the package dependencies from pyproject.toml."
        ) from exc

    clip = VideoFileClip(str(clip_path))
    frames = {}
    try:
        iter_fps = float(fps or clip.fps)
        for frame_idx, (_frame_time, frame) in enumerate(
            clip.iter_frames(fps=iter_fps, with_times=True, dtype="uint8")
        ):
            if frame_idx in wanted:
                frames[frame_idx] = frame
                if len(frames) == len(wanted):
                    break
            if frame_idx > max_frame_id:
                break
    finally:
        clip.reader.close()
        if clip.audio is not None:
            clip.audio.reader.close_proc()
    missing = sorted(wanted - frames.keys())
    if missing:
        raise RuntimeError(f"{clip_path} is missing requested clip frames: {missing[:10]}")
    return frames


def encode_video(frames_dir: Path, output_path: Path, fps: float) -> None:
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        str(output_path),
    ]
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc != 0 or not output_path.exists():
        raise RuntimeError(f"ffmpeg encode failed for {output_path}")


def reconstruct_track(
    video_path: str | Path,
    track_info: dict,
    output_path: str | Path,
    *,
    clip_path: str | Path | None = None,
    require_clip: bool = False,
    frame_source: str = "auto",
    fps: float | None = None,
    overwrite: bool = False,
) -> Path:
    """Crop selected source frames into one benchmark track video."""

    video_path = Path(video_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for source-track reconstruction. "
            "Install with `pip install opencv-python`."
        ) from exc

    # These are frozen frame ids from the release metadata. Do not recompute
    # them from clip_start_time/FPS, because scene-clip extraction used decoder
    # frame choices that differ from a naive round() formula on some videos.
    if frame_source not in {"auto", "clip", "full_video"}:
        raise ValueError(f"frame_source must be one of auto/clip/full_video, got {frame_source!r}")
    clip_path = Path(clip_path) if clip_path is not None else None
    clip_available = (
        clip_path is not None
        and clip_path.exists()
        and bool(track_info.get("clip_frame_ids"))
    )
    if frame_source == "full_video":
        use_clip = False
    else:
        use_clip = clip_available
    if (require_clip or frame_source == "clip") and not clip_available:
        raise FileNotFoundError(
            f"clip-frame reconstruction requested, but clip is unavailable for "
            f"{track_info.get('video_id')} {track_info.get('clip_id')}: {clip_path}"
        )

    if use_clip:
        frame_ids = track_info.get("clip_frame_ids")
        boxes = track_info.get("crop_boxes_xyxy")
    else:
        frame_ids = track_info.get("full_video_frame_ids") or track_info.get("video_frame_ids")
        boxes = track_info.get("full_video_crop_boxes_xyxy") or track_info.get("crop_boxes_xyxy")
    if not frame_ids or not boxes:
        raise ValueError("track metadata requires frame ids and crop_boxes_xyxy")
    if len(frame_ids) != len(boxes):
        raise ValueError(
            f"frame_ids/boxes length mismatch: {len(frame_ids)} frame ids vs {len(boxes)} boxes"
        )

    width, height = track_info.get("output_resolution", [832, 480])
    out_fps = float(fps or track_info.get("output_fps") or 15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if use_clip:
        clip_frame_ids = [int(frame_id) for frame_id in frame_ids]
        decode_fps = track_info.get("clip_decode_fps", track_info.get("clip_fps"))
        try:
            clip_frames = read_moviepy_clip_frames(
                clip_path,
                clip_frame_ids,
                fps=float(decode_fps) if decode_fps is not None else None,
            )
        except RuntimeError:
            if decode_fps is not None or not track_info.get("clip_fps"):
                raise
            clip_frames = read_moviepy_clip_frames(
                clip_path,
                clip_frame_ids,
                fps=float(track_info["clip_fps"]),
            )
        for idx, (frame_id, box) in enumerate(zip(frame_ids, boxes)):
            frame_rgb = clip_frames[int(frame_id)]
            cropped_rgb = crop_and_resize(frame_rgb, box, (int(width), int(height)))
            cropped = release_rgb_roundtrip(cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(frames_dir / f"{idx:05d}.png"), cropped)
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")

        cached_frame_id = None
        cached_frame = None
        try:
            for idx, (frame_id, box) in enumerate(zip(frame_ids, boxes)):
                frame_id = int(frame_id)
                if frame_id == cached_frame_id:
                    frame = cached_frame
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"failed to read frame {frame_id} from {video_path}")
                    cached_frame_id = frame_id
                    cached_frame = frame
                cropped = release_rgb_roundtrip(crop_and_resize(frame, box, (int(width), int(height))))
                cv2.imwrite(str(frames_dir / f"{idx:05d}.png"), cropped)
        finally:
            cap.release()
    encode_video(frames_dir, output_path, out_fps)

    return output_path


def reconstruct_tracks_for_video(
    video_path: str | Path,
    tracks: list[tuple[str, dict]],
    output_root: str | Path,
    *,
    clip_root: str | Path | None = None,
    require_clips: bool = False,
    frame_source: str = "auto",
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    outputs = []
    for track_name, info in tracks:
        output_path = Path(output_root) / "tracks" / track_name / "input.mp4"
        outputs.append(output_path)
        if not dry_run:
            clip_path = None
            if clip_root is not None:
                clip_id = info.get("clip_id")
                if clip_id is None and info.get("video_id") is not None and info.get("clip") is not None:
                    clip_id = f"{info['video_id']}_{int(info['clip']):03d}"
                if clip_id is not None:
                    clip_path = Path(clip_root) / f"{clip_id}.mp4"
            reconstruct_track(
                video_path,
                info,
                output_path,
                clip_path=clip_path,
                require_clip=require_clips,
                frame_source=frame_source,
                overwrite=overwrite,
            )
    return outputs
