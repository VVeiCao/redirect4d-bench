#!/usr/bin/env python3
"""Build a static browser for a Redirect4D-Bench dataset folder.

The browser is intentionally read-only. It creates a small output folder with
an HTML file and symlinks back to the dataset, so large videos and point clouds
are not copied.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


SOURCE_RGB_NAMES = ("input.mp4", "video.mp4", "original_images.mp4", "rgb.mp4")
ORIGINAL_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_link(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def url_for(path: Path | None, root: Path, link_name: str) -> str | None:
    if path is None or not path.exists():
        return None
    return f"{link_name}/{path.relative_to(root).as_posix()}"


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def pointcloud_layout(track_dir: Path) -> tuple[Path, Path, str]:
    """Return global pointcloud root, frame root, and filename style."""
    preserved = track_dir / "pointcloud"
    if preserved.exists():
        return preserved, preserved, "preserved"
    normalized = track_dir / "reconstruction" / "pointclouds"
    return normalized, normalized / "frames", "normalized"


def frame_pointcloud_path(frame_dir: Path, frame: str, suffix: str, style: str) -> Path:
    if style == "preserved":
        return frame_dir / suffix
    return frame_dir / f"{frame}_{suffix}"


def find_original_video(track_row: dict, original_root: Path | None) -> Path | None:
    if original_root is None:
        return None
    track = track_row["track"]
    video_id = track_row.get("video_id", "")
    names = []
    for stem in (track, video_id):
        if not stem:
            continue
        names.extend([stem + suffix for suffix in ORIGINAL_VIDEO_SUFFIXES])
    return first_existing([original_root / name for name in names])


def build_track_item(
    dataset_root: Path,
    dataset_link: str,
    track_row: dict,
    cases_by_track: dict[str, list[dict]],
    processed_root: Path | None,
    processed_link: str,
    original_video_root: Path | None,
    original_video_link: str,
) -> dict:
    track = track_row["track"]
    track_dir = dataset_root / "tracks" / track
    pointcloud_root, pc_root, pc_style = pointcloud_layout(track_dir)

    frame_items = []
    if pc_root.exists():
        for frame_dir in sorted(p for p in pc_root.iterdir() if p.is_dir()):
            frame = frame_dir.name
            frame_items.append(
                {
                    "frame": frame,
                    "foreground": url_for(
                        frame_pointcloud_path(
                            frame_dir,
                            frame,
                            "foreground_5_views_aligned_smooth.ply",
                            pc_style,
                        ),
                        dataset_root,
                        dataset_link,
                    ),
                    "single_view": url_for(
                        frame_pointcloud_path(frame_dir, frame, "foreground_1_view.ply", pc_style),
                        dataset_root,
                        dataset_link,
                    ),
                }
            )

    source_rgb_candidates = [track_dir / name for name in SOURCE_RGB_NAMES]
    if processed_root is not None:
        processed_track = processed_root / "tracks" / track
        processed_candidates = [processed_track / name for name in SOURCE_RGB_NAMES]
    else:
        processed_candidates = []
    source_rgb_path = first_existing(source_rgb_candidates)
    source_rgb_url = url_for(source_rgb_path, dataset_root, dataset_link)
    if source_rgb_url is None and processed_root is not None:
        processed_rgb_path = first_existing(processed_candidates)
        source_rgb_url = url_for(processed_rgb_path, processed_root, processed_link)

    original_video_path = find_original_video(track_row, original_video_root)
    original_video_url = (
        url_for(original_video_path, original_video_root, original_video_link)
        if original_video_root is not None
        else None
    )

    trajectory_items = []
    for case in cases_by_track.get(track, []):
        traj = case["trajectory"]
        traj_dir = first_existing([
            track_dir / "redirected" / traj,
            track_dir / "trajectories" / traj,
        ]) or (track_dir / "redirected" / traj)
        trajectory_items.append(
            {
                "case": case.get("case", f"{track}__{traj}"),
                "trajectory": traj,
                "trajectory_json": url_for(traj_dir / "trajectory.json", dataset_root, dataset_link),
                "mask": url_for(traj_dir / "mask.mp4", dataset_root, dataset_link),
                "depth": url_for(traj_dir / "depth.mp4", dataset_root, dataset_link),
            }
        )

    return {
        "track": track,
        "category": track_row.get("category", ""),
        "video_id": track_row.get("video_id", ""),
        "youtube_url": track_row.get("youtube_url", ""),
        "num_frames": track_row.get("num_frames"),
        "frames": frame_items,
        "trajectories": trajectory_items,
        "global_background": url_for(
            first_existing(
                [
                    pointcloud_root / "global_background.ply",
                    track_dir / "reconstruction" / "global_background.ply",
                ]
            ),
            dataset_root,
            dataset_link,
        ),
        "global_camera": url_for(
            first_existing(
                [
                    pointcloud_root / "global_camera.json",
                    track_dir / "camera.json",
                    track_dir / "reconstruction" / "global_camera.json",
                ]
            ),
            dataset_root,
            dataset_link,
        ),
        "source_mask": url_for(track_dir / "mask_video.mp4", dataset_root, dataset_link),
        "source_rgb": source_rgb_url,
        "original_video": original_video_url,
    }


def collect_dataset(
    dataset_root: Path,
    dataset_link: str,
    processed_root: Path | None,
    processed_link: str,
    original_video_root: Path | None,
    original_video_link: str,
    limit: int | None,
) -> dict:
    tracks = read_jsonl(dataset_root / "tracks.jsonl")
    cases = read_jsonl(dataset_root / "cases.jsonl")
    if limit is not None:
        tracks = tracks[:limit]
        keep = {row["track"] for row in tracks}
        cases = [row for row in cases if row.get("track") in keep]

    cases_by_track: dict[str, list[dict]] = {}
    for case in cases:
        cases_by_track.setdefault(case["track"], []).append(case)

    items = [
        build_track_item(
            dataset_root=dataset_root,
            dataset_link=dataset_link,
            track_row=row,
            cases_by_track=cases_by_track,
            processed_root=processed_root,
            processed_link=processed_link,
            original_video_root=original_video_root,
            original_video_link=original_video_link,
        )
        for row in tracks
    ]

    return {
        "dataset_root": dataset_root.name,
        "num_tracks": len(items),
        "num_cases": sum(len(item["trajectories"]) for item in items),
        "tracks": items,
    }


def render_html(data: dict, title: str) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    escaped_title = html.escape(title)
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f4;
      --panel: #ffffff;
      --ink: #1e2328;
      --muted: #66707a;
      --line: #d8ded6;
      --accent: #0b6f6a;
      --soft: #e7f1ef;
      --warn-bg: #fff7e8;
      --warn: #8a4a00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header.topbar {
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(245, 246, 244, 0.96);
      backdrop-filter: blur(10px);
    }
    h1 {
      margin: 0 0 10px;
      font-size: 21px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(220px, 0.9fr) minmax(160px, 0.55fr);
      gap: 10px;
      align-items: center;
    }
    input, select, button {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      min-height: calc(100vh - 92px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfbfa;
      overflow: auto;
      max-height: calc(100vh - 92px);
      padding: 12px;
    }
    .track-row {
      width: 100%;
      display: block;
      text-align: left;
      margin: 0 0 8px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
    }
    .track-row.is-active {
      border-color: var(--accent);
      background: var(--soft);
    }
    .track-row.is-hidden { display: none; }
    .track-name {
      font-size: 13px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .track-meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .workspace {
      padding: 14px;
      min-width: 0;
    }
    .case-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }
    h2 {
      margin: 0;
      font-size: 17px;
      line-height: 1.35;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }
    .subtle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 3px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      margin-bottom: 12px;
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .panel-title {
      font-size: 13px;
      font-weight: 650;
    }
    .toggles {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .toggles label {
      display: inline-flex;
      gap: 5px;
      align-items: center;
    }
    #viewer {
      width: 100%;
      height: min(58vh, 620px);
      min-height: 360px;
      background: #101214;
    }
    .media-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      padding: 12px;
    }
    .media-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    video, .missing-box, canvas#trajectoryCanvas {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #111;
    }
    .missing-box {
      display: grid;
      place-items: center;
      color: var(--warn);
      background: var(--warn-bg);
      font-size: 12px;
      text-align: center;
      padding: 10px;
    }
    .trajectory-body {
      display: grid;
      grid-template-columns: minmax(240px, 420px) 1fr;
      gap: 12px;
      padding: 12px;
      align-items: start;
    }
    #trajectorySummary {
      margin: 0;
      white-space: pre-wrap;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    a {
      color: var(--accent);
      text-decoration: none;
    }
    @media (max-width: 980px) {
      main, .trajectory-body, .case-head, .controls { grid-template-columns: 1fr; }
      aside { max-height: 280px; border-right: 0; border-bottom: 1px solid var(--line); }
      .workspace { padding: 12px; }
      #viewer { height: 52vh; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <h1>__TITLE__</h1>
    <div class="controls">
      <input id="search" type="search" placeholder="Search track, category, trajectory, or YouTube id">
      <select id="trajectorySelect"></select>
      <select id="frameSelect"></select>
    </div>
  </header>
  <main>
    <aside id="trackList"></aside>
    <section class="workspace">
      <div class="case-head">
        <div>
          <h2 id="trackTitle"></h2>
          <div id="trackMeta" class="subtle"></div>
        </div>
        <button id="loadPointClouds" class="primary" type="button">Load point clouds</button>
      </div>

      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">Point clouds</div>
          <div class="toggles">
            <label><input id="showForeground" type="checkbox" checked> final foreground</label>
            <label><input id="showSingleView" type="checkbox"> 1-view foreground</label>
            <label><input id="showBackground" type="checkbox"> background</label>
            <label>point size <input id="pointSize" type="range" min="0.002" max="0.04" step="0.002" value="0.01"></label>
          </div>
        </div>
        <div id="viewer"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">Videos</div>
        </div>
        <div class="media-grid" id="mediaGrid"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">Trajectory</div>
          <a id="trajectoryLink" href="#" target="_blank" rel="noreferrer">open trajectory.json</a>
        </div>
        <div class="trajectory-body">
          <canvas id="trajectoryCanvas" width="640" height="360"></canvas>
          <pre id="trajectorySummary"></pre>
        </div>
      </section>
    </section>
  </main>

  <script type="importmap">
    {"imports": {"three": "https://unpkg.com/three@0.160.0/build/three.module.js"}}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { PLYLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js';

    const DATA = __DATA__;
    let activeIndex = 0;
    let loaded = { foreground: null, singleView: null, background: null, trajectory: null };
    const trackList = document.getElementById('trackList');
    const search = document.getElementById('search');
    const frameSelect = document.getElementById('frameSelect');
    const trajectorySelect = document.getElementById('trajectorySelect');
    const mediaGrid = document.getElementById('mediaGrid');
    const trajectoryLink = document.getElementById('trajectoryLink');
    const trajectorySummary = document.getElementById('trajectorySummary');
    const trajectoryCanvas = document.getElementById('trajectoryCanvas');
    const showForeground = document.getElementById('showForeground');
    const showSingleView = document.getElementById('showSingleView');
    const showBackground = document.getElementById('showBackground');
    const pointSize = document.getElementById('pointSize');

    const viewer = document.getElementById('viewer');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);
    const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 10000);
    camera.position.set(0, 0, 3);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    viewer.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    function resize() {
      const w = viewer.clientWidth;
      const h = viewer.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);
    resize();

    function activeTrack() {
      return DATA.tracks[activeIndex];
    }

    function activeFrame() {
      const track = activeTrack();
      return track.frames.find(frame => frame.frame === frameSelect.value) || track.frames[0] || null;
    }

    function activeTrajectory() {
      const track = activeTrack();
      return track.trajectories.find(item => item.trajectory === trajectorySelect.value) || track.trajectories[0] || null;
    }

    function renderTrackList() {
      trackList.innerHTML = '';
      DATA.tracks.forEach((track, index) => {
        const button = document.createElement('button');
        button.className = 'track-row';
        button.type = 'button';
        button.dataset.haystack = [track.track, track.category, track.video_id, ...track.trajectories.map(t => t.trajectory)].join(' ').toLowerCase();
        button.innerHTML = `<div class="track-name">${escapeHtml(track.track)}</div>
          <div class="track-meta">${escapeHtml(track.category || 'unknown')} · ${track.frames.length} frames · ${track.trajectories.length} trajectories</div>`;
        button.addEventListener('click', () => {
          activeIndex = index;
          updateForTrack();
        });
        trackList.appendChild(button);
      });
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }

    function updateTrackButtons() {
      Array.from(document.querySelectorAll('.track-row')).forEach((button, index) => {
        button.classList.toggle('is-active', index === activeIndex);
      });
    }

    function updateSelectors() {
      const track = activeTrack();
      trajectorySelect.innerHTML = track.trajectories.map(item =>
        `<option value="${escapeHtml(item.trajectory)}">${escapeHtml(item.trajectory)}</option>`
      ).join('');
      frameSelect.innerHTML = track.frames.map(item =>
        `<option value="${escapeHtml(item.frame)}">frame ${escapeHtml(item.frame)}</option>`
      ).join('');
    }

    function setVideo(label, url, missingText) {
      const cell = document.createElement('section');
      cell.className = 'media-cell';
      const title = document.createElement('div');
      title.className = 'media-label';
      title.textContent = label;
      cell.appendChild(title);
      if (url) {
        const video = document.createElement('video');
        video.controls = true;
        video.muted = true;
        video.preload = 'metadata';
        video.src = url;
        cell.appendChild(video);
      } else {
        const box = document.createElement('div');
        box.className = 'missing-box';
        box.textContent = missingText;
        cell.appendChild(box);
      }
      mediaGrid.appendChild(cell);
    }

    function updateMedia() {
      const track = activeTrack();
      const traj = activeTrajectory();
      mediaGrid.innerHTML = '';
      setVideo('Original video', track.original_video, 'not downloaded');
      setVideo('Source RGB', track.source_rgb, 'not included in public dataset');
      setVideo('Source mask', track.source_mask, 'missing');
      setVideo('Target mask', traj?.mask, 'missing');
      setVideo('Target depth', traj?.depth, 'missing');
    }

    function clearObject(name) {
      if (!loaded[name]) return;
      scene.remove(loaded[name]);
      loaded[name].geometry?.dispose?.();
      loaded[name].material?.dispose?.();
      loaded[name] = null;
    }

    function pointMaterial(color, useVertexColors) {
      return new THREE.PointsMaterial({
        size: Number(pointSize.value),
        color,
        vertexColors: useVertexColors,
        sizeAttenuation: true
      });
    }

    function frameGeometry(geometry) {
      geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      box.getCenter(center);
      box.getSize(size);
      const radius = Math.max(size.x, size.y, size.z) || 1;
      camera.position.set(center.x, center.y, center.z + radius * 2.4);
      camera.near = Math.max(radius / 2000, 0.001);
      camera.far = radius * 2000;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
    }

    function loadPly(url, name, color) {
      if (!url) {
        clearObject(name);
        return Promise.resolve(null);
      }
      return new Promise((resolve, reject) => {
        new PLYLoader().load(url, geometry => {
          clearObject(name);
          const hasColor = geometry.hasAttribute('color');
          const points = new THREE.Points(geometry, pointMaterial(color, hasColor));
          scene.add(points);
          loaded[name] = points;
          frameGeometry(geometry);
          resolve(points);
        }, undefined, reject);
      });
    }

    function updateVisibility() {
      if (loaded.foreground) loaded.foreground.visible = showForeground.checked;
      if (loaded.singleView) loaded.singleView.visible = showSingleView.checked;
      if (loaded.background) loaded.background.visible = showBackground.checked;
      for (const key of ['foreground', 'singleView', 'background']) {
        if (loaded[key]?.material) loaded[key].material.size = Number(pointSize.value);
      }
    }

    async function loadPointClouds() {
      const track = activeTrack();
      const frame = activeFrame();
      if (!frame) return;
      await loadPly(showForeground.checked ? frame.foreground : null, 'foreground', 0x40c9a2).catch(console.warn);
      await loadPly(showSingleView.checked ? frame.single_view : null, 'singleView', 0xf2b84b).catch(console.warn);
      await loadPly(showBackground.checked ? track.global_background : null, 'background', 0x9aa3ad).catch(console.warn);
      updateVisibility();
    }

    function drawTrajectory2d(points) {
      const canvas = trajectoryCanvas;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!points.length) return;
      const xs = points.map(p => p[0]);
      const zs = points.map(p => p[2] ?? p[1] ?? 0);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minZ = Math.min(...zs), maxZ = Math.max(...zs);
      const pad = 34;
      const sx = (canvas.width - pad * 2) / Math.max(maxX - minX, 1e-6);
      const sz = (canvas.height - pad * 2) / Math.max(maxZ - minZ, 1e-6);
      const scale = Math.min(sx, sz);
      function map(p) {
        const x = pad + (p[0] - minX) * scale;
        const z = canvas.height - pad - ((p[2] ?? p[1] ?? 0) - minZ) * scale;
        return [x, z];
      }
      ctx.strokeStyle = '#40c9a2';
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((p, i) => {
        const [x, z] = map(p);
        if (i === 0) ctx.moveTo(x, z);
        else ctx.lineTo(x, z);
      });
      ctx.stroke();
      points.forEach((p, i) => {
        const [x, z] = map(p);
        ctx.fillStyle = i === 0 ? '#ffffff' : (i === points.length - 1 ? '#f2b84b' : '#40c9a2');
        ctx.beginPath();
        ctx.arc(x, z, i === 0 || i === points.length - 1 ? 5 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function drawTrajectory3d(points) {
      if (loaded.trajectory) {
        scene.remove(loaded.trajectory);
        loaded.trajectory.geometry.dispose();
        loaded.trajectory.material.dispose();
        loaded.trajectory = null;
      }
      if (!points.length) return;
      const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
      const material = new THREE.LineBasicMaterial({ color: 0xf2b84b });
      loaded.trajectory = new THREE.Line(geometry, material);
      scene.add(loaded.trajectory);
    }

    async function loadTrajectory() {
      const traj = activeTrajectory();
      if (!traj?.trajectory_json) {
        trajectorySummary.textContent = 'No trajectory selected.';
        trajectoryLink.removeAttribute('href');
        drawTrajectory2d([]);
        drawTrajectory3d([]);
        return;
      }
      trajectoryLink.href = traj.trajectory_json;
      const data = await fetch(traj.trajectory_json).then(r => r.json());
      const path = data.camera_path || [];
      const points = path.map(item => item.position).filter(Boolean);
      drawTrajectory2d(points);
      drawTrajectory3d(points);
      const meta = data.metadata || {};
      trajectorySummary.textContent = [
        `case: ${traj.case}`,
        `trajectory: ${traj.trajectory}`,
        `frames: ${path.length}`,
        `keyframes: ${(data.keyframes || []).length}`,
        `image_size: ${JSON.stringify(meta.image_size || '')}`,
        `source: ${meta.source || ''}`
      ].join('\n');
    }

    function updateForTrack() {
      const track = activeTrack();
      updateTrackButtons();
      updateSelectors();
      document.getElementById('trackTitle').textContent = track.track;
      document.getElementById('trackMeta').textContent =
        `${track.category || 'unknown'} · YouTube ${track.video_id || 'n/a'} · ${track.frames.length} frames · ${track.trajectories.length} trajectories`;
      clearObject('foreground');
      clearObject('singleView');
      clearObject('background');
      updateMedia();
      loadTrajectory().catch(console.warn);
      loadPointClouds().catch(console.warn);
    }

    function applySearch() {
      const q = search.value.trim().toLowerCase();
      Array.from(document.querySelectorAll('.track-row')).forEach(button => {
        button.classList.toggle('is-hidden', q && !button.dataset.haystack.includes(q));
      });
    }

    document.getElementById('loadPointClouds').addEventListener('click', () => loadPointClouds());
    frameSelect.addEventListener('change', () => loadPointClouds());
    trajectorySelect.addEventListener('change', () => {
      updateMedia();
      loadTrajectory().catch(console.warn);
    });
    search.addEventListener('input', applySearch);
    for (const el of [showForeground, showSingleView, showBackground]) {
      el.addEventListener('change', () => loadPointClouds());
    }
    pointSize.addEventListener('input', updateVisibility);

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    renderTrackList();
    if (DATA.tracks.length) updateForTrack();
    animate();
  </script>
</body>
</html>
"""
    return template.replace("__TITLE__", escaped_title).replace("__DATA__", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static Redirect4D dataset browser.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/redirect4d_bench"))
    parser.add_argument("--processed-root", type=Path, help="Optional source-RGB folder regenerated from original videos.")
    parser.add_argument("--original-video-root", type=Path, help="Optional folder containing downloaded original videos.")
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_browser/index.html"))
    parser.add_argument("--title", default="Redirect4D Dataset Browser")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not (dataset_root / "tracks.jsonl").exists():
        raise FileNotFoundError(f"tracks.jsonl not found under {dataset_root}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_link = "dataset"
    safe_link(dataset_root, output.parent / dataset_link, overwrite=True)

    processed_root = args.processed_root.resolve() if args.processed_root else None
    processed_link = "processed"
    if processed_root is not None and processed_root.exists():
        safe_link(processed_root, output.parent / processed_link, overwrite=True)

    original_video_root = args.original_video_root.resolve() if args.original_video_root else None
    original_video_link = "original_videos"
    if original_video_root is not None and original_video_root.exists():
        safe_link(original_video_root, output.parent / original_video_link, overwrite=True)

    data = collect_dataset(
        dataset_root=dataset_root,
        dataset_link=dataset_link,
        processed_root=processed_root if processed_root and processed_root.exists() else None,
        processed_link=processed_link,
        original_video_root=original_video_root if original_video_root and original_video_root.exists() else None,
        original_video_link=original_video_link,
        limit=args.limit,
    )
    output.write_text(render_html(data, args.title))
    print(f"[ok] wrote {output}")
    print(f"[ok] tracks={data['num_tracks']} cases={data['num_cases']}")
    print(f"[serve] python -m http.server 8090 -d {output.parent}")
    print(f"[open]  http://localhost:8090/{output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
