"""Download source videos from YouTube metadata."""

from __future__ import annotations

from pathlib import Path
import sys


DEFAULT_FORMAT = (
    "bestvideo[vcodec^=avc1][height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/"
    "best[ext=mp4]"
)


def download_video(
    url: str,
    output_dir: str | Path,
    *,
    video_id: str | None = None,
    fmt: str = DEFAULT_FORMAT,
    cookies: str | Path | None = None,
    cookies_from_browser: str | None = None,
    proxy: str | None = None,
    rate_limit: str | None = None,
    retries: int = 10,
    overwrite: bool = False,
) -> Path:
    """Download one video and return the expected mp4 path.

    The file is named by YouTube id whenever available. This mirrors the
    dataset metadata and keeps reconstructed tracks reproducible.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    if video_id:
        expected = output_dir / f"{video_id}.mp4"
        if expected.exists() and not overwrite:
            return expected

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise ImportError("yt-dlp is required for video download. Install with `pip install yt-dlp`.") from exc

    opts: dict = {
        "format": fmt,
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "retries": retries,
        "continuedl": True,
        "overwrites": overwrite,
        "noplaylist": True,
        "quiet": False,
    }
    env_ffmpeg = Path(sys.executable).resolve().parent / "ffmpeg"
    if env_ffmpeg.exists():
        opts["ffmpeg_location"] = str(env_ffmpeg)
    if cookies:
        opts["cookiefile"] = str(cookies)
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if proxy:
        opts["proxy"] = proxy
    if rate_limit:
        opts["ratelimit"] = rate_limit

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    downloaded_id = str(info.get("id") or video_id)
    if not downloaded_id:
        raise RuntimeError(f"could not determine video id for {url}")
    return output_dir / f"{downloaded_id}.mp4"
