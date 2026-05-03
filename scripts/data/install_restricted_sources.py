#!/usr/bin/env python3
"""Install restricted source videos/frames into the benchmark data workspace.

The public Hugging Face dataset intentionally omits source RGB videos and
frames. Users who are granted access to the restricted source package can place
it with this script.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite")
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=False)


def extract_archive(archive: Path, out: Path, overwrite: bool) -> None:
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes)
    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive) as tf:
            tf.extractall(out)
        return
    if suffixes.endswith(".zip"):
        subprocess.run(["unzip", "-q", str(archive), "-d", str(out)], check=True)
        return
    raise ValueError(f"unsupported archive type: {archive}")


def normalize_layout(root: Path) -> Path:
    """Return the folder that contains tracks/<track>/video.mp4."""
    if (root / "tracks").exists():
        return root
    children = [p for p in root.iterdir() if p.is_dir()]
    for child in children:
        if (child / "tracks").exists():
            return child
    raise FileNotFoundError(f"could not find tracks/ under {root}")


def validate_sources(root: Path, expected_tracks: int = 62) -> None:
    tracks = sorted(p for p in (root / "tracks").iterdir() if p.is_dir())
    if len(tracks) != expected_tracks:
        raise RuntimeError(f"expected {expected_tracks} source tracks, found {len(tracks)}")
    missing = []
    for track in tracks:
        if not (track / "video.mp4").exists():
            missing.append(str(track / "video.mp4"))
        frame_count = len(list((track / "frames").glob("*.png"))) if (track / "frames").exists() else 0
        if frame_count != 45:
            missing.append(f"{track / 'frames'} has {frame_count} PNG frames")
    if missing:
        raise RuntimeError("invalid restricted source package:\n  - " + "\n  - ".join(missing[:20]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", type=Path, help="Downloaded .tar/.tar.gz/.tgz/.zip package.")
    src.add_argument("--source-dir", type=Path, help="Unpacked package directory.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reconstructed_source_tracks"),
    )
    parser.add_argument("--expected-tracks", type=int, default=62)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.archive:
        tmp = args.out.parent / f".{args.out.name}.extracting"
        if tmp.exists():
            shutil.rmtree(tmp)
        extract_archive(args.archive.resolve(), tmp, overwrite=True)
        source_root = normalize_layout(tmp)
        copy_tree(source_root, args.out, overwrite=args.overwrite)
        shutil.rmtree(tmp)
    else:
        source_root = normalize_layout(args.source_dir.resolve())
        copy_tree(source_root, args.out, overwrite=args.overwrite)

    validate_sources(args.out, expected_tracks=args.expected_tracks)
    print(f"[ok] installed restricted sources -> {args.out}")


if __name__ == "__main__":
    main()
