"""Metadata helpers for Redirect4D-Bench."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_metadata(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        metadata = json.load(f)
    if "tracks" in metadata:
        return metadata

    tracks_path = path.with_name("tracks.jsonl")
    if not tracks_path.exists() and path.parent.name == "metadata":
        tracks_path = path.parent.parent / "tracks.jsonl"
    if not tracks_path.exists():
        return metadata

    rows = _read_jsonl(tracks_path)
    metadata = dict(metadata)
    metadata["tracks"] = {row["track"]: dict(row) for row in rows}
    metadata.setdefault("num_tracks", len(rows))
    metadata.setdefault("num_trajectories", sum(len(row.get("trajectories", [])) for row in rows))
    metadata.setdefault("num_videos", len({row.get("video_id") for row in rows if row.get("video_id")}))
    return metadata


def track_items(metadata: dict, tracks: Iterable[str] | None = None) -> list[tuple[str, dict]]:
    all_tracks = metadata.get("tracks", {})
    if tracks is None:
        names = sorted(all_tracks)
    else:
        names = sorted(tracks)

    missing = [name for name in names if name not in all_tracks]
    if missing:
        raise KeyError(f"{len(missing)} tracks are missing from metadata: {missing[:5]}")
    return [(name, all_tracks[name]) for name in names]


def read_track_list(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def group_tracks_by_video(items: Iterable[tuple[str, dict]]) -> dict[str, list[tuple[str, dict]]]:
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for track_name, info in items:
        video_id = info.get("video_id")
        if not video_id:
            raise ValueError(f"track {track_name} is missing video_id")
        grouped[video_id].append((track_name, info))
    return dict(sorted(grouped.items()))


def youtube_url(info: dict) -> str:
    url = info.get("youtube_url")
    if url:
        return url
    video_id = info.get("video_id")
    if not video_id:
        raise ValueError("metadata item is missing youtube_url and video_id")
    return f"https://youtu.be/{video_id}"
