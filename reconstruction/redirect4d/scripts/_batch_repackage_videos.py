#!/usr/bin/env python3
"""One-shot helper: for every outputs/rendering/*/*/ where PNG frames exist but
.mp4 videos don't, run ffmpeg to package them + re-run organize_inference_structure.

Use after installing ffmpeg on a batch that silently failed at the video step.
"""
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.rendering import generate_videos_from_subdirs, organize_inference_structure

RENDERING_ROOT = PROJECT_ROOT / "outputs" / "rendering"
FPS = 10  # matches default used by 1_4_rendering.py

def main():
    fixed = 0
    skipped = 0
    missing = 0
    for track_dir in sorted(RENDERING_ROOT.iterdir()):
        if not track_dir.is_dir(): continue
        for traj_dir in sorted(track_dir.iterdir()):
            if not traj_dir.is_dir(): continue
            raw = traj_dir / "raw_images"
            if not raw.is_dir():
                missing += 1; continue
            subdirs = {
                'original_images': str(raw / "original_images"),
                'rendered_images': str(raw / "rendered_images"),
                'rendered_depths': str(raw / "rendered_depths"),
                'rendered_masks':  str(raw / "rendered_masks"),
            }
            # Need rendered_images/ non-empty
            rimg = Path(subdirs['rendered_images'])
            if not rimg.is_dir() or not any(rimg.iterdir()):
                missing += 1; continue
            # Skip if videos already exist
            if (traj_dir / "videos" / "rendered_images.mp4").exists():
                skipped += 1; continue
            try:
                generate_videos_from_subdirs(subdirs, FPS, str(traj_dir))
                organize_inference_structure(str(traj_dir), subdirs)
                fixed += 1
                print(f"[OK] {track_dir.name}/{traj_dir.name}")
            except Exception as e:
                print(f"[FAIL] {track_dir.name}/{traj_dir.name}: {e}")
    print(f"\nDone: {fixed} repackaged, {skipped} already-ok, {missing} missing PNGs")

if __name__ == "__main__":
    main()
