#!/usr/bin/env python3
"""
Migrate existing trajectory data to the sidecar-file layout.

Old layout (one big JSON per episode):
  episode_<id>/
    episode_data.json     # contains hex-encoded image bytes inline

New layout (sidecar files):
  episode_<id>/
    episode_data.json     # rgb_images/depth_images values are relative paths
    rgb/<camera>/<sha1>.jpg
    depth/<camera>/<sha1>.npz

Each unique image is written once and deduped by content hash, so a frame that
appears as the closest match for several synchronized data points only takes
disk space once.

Usage:
    python compress_existing_trajectories.py --data-dir /path/to/rl_training_data
    python compress_existing_trajectories.py --data-dir /path --dry-run
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tyro
from tqdm import tqdm

# Must match aera_semi_autonomous.data.trajectory_data_collector.DEPTH_PNG_SCALE
DEPTH_PNG_SCALE = 10000.0


def _looks_like_hex(value: str) -> bool:
    # Sidecar paths contain '/' and a file suffix; hex strings don't.
    return "/" not in value and "." not in value


def _migrate_episode(episode_dir: Path, dry_run: bool) -> tuple[bool, int, int]:
    """Migrate one episode.

    Returns: (changed, bytes_in_old_json, bytes_in_new_artifacts_estimate).
    """
    json_path = episode_dir / "episode_data.json"
    if not json_path.exists():
        return False, 0, 0

    old_size = json_path.stat().st_size

    with open(json_path) as f:
        data = json.load(f)

    trajectory = data.get("trajectory_data", [])
    if not trajectory:
        return False, old_size, old_size

    metadata = data.get("metadata", {})
    height = metadata.get("image_height")
    width = metadata.get("image_width")

    rgb_files: dict[str, bytes] = {}
    depth_files: dict[str, np.ndarray] = {}
    npz_to_remove: set[Path] = set()
    changed = False

    for step in trajectory:
        rgb_images = step.get("observations", {}).get("rgb_images", {})
        for cam, ref in list(rgb_images.items()):
            if not isinstance(ref, str) or not _looks_like_hex(ref):
                continue
            digest = hashlib.sha1(ref.encode("ascii")).hexdigest()[:16]
            rel_path = f"rgb/{cam}/{digest}.jpg"
            if rel_path not in rgb_files:
                rgb_files[rel_path] = bytes.fromhex(ref)
            rgb_images[cam] = rel_path
            changed = True

        depth_images = step.get("observations", {}).get("depth_images", {})
        for cam, ref in list(depth_images.items()):
            if not isinstance(ref, str):
                continue
            if ref.endswith(".png"):
                continue  # already in target format
            if ref.endswith(".npz"):
                npz_path = episode_dir / ref
                rel_path = ref[:-4] + ".png"
                if rel_path not in depth_files and npz_path.exists():
                    depth_files[rel_path] = np.load(npz_path)["depth"]
                npz_to_remove.add(npz_path)
                depth_images[cam] = rel_path
                changed = True
            elif _looks_like_hex(ref):
                if height is None or width is None:
                    raise RuntimeError(
                        f"{episode_dir.name}: metadata missing image_height/image_width; "
                        "cannot reshape legacy depth bytes."
                    )
                digest = hashlib.sha1(ref.encode("ascii")).hexdigest()[:16]
                rel_path = f"depth/{cam}/{digest}.png"
                if rel_path not in depth_files:
                    arr = np.frombuffer(bytes.fromhex(ref), dtype=np.float32).reshape(
                        (height, width)
                    )
                    depth_files[rel_path] = arr
                depth_images[cam] = rel_path
                changed = True

    if not changed:
        return False, old_size, old_size

    if dry_run:
        return True, old_size, 0

    for rel_path, blob in rgb_files.items():
        full = episode_dir / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(blob)
    for rel_path, arr in depth_files.items():
        full = episode_dir / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        scaled = np.clip(arr * DEPTH_PNG_SCALE, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(full), scaled)
    if depth_files:
        data.setdefault("metadata", {})["depth_png_scale"] = DEPTH_PNG_SCALE
    for npz_path in npz_to_remove:
        if npz_path.exists():
            npz_path.unlink()

    # Atomic JSON rewrite: write to temp in same dir, then rename.
    fd, tmp_path = tempfile.mkstemp(
        prefix="episode_data.", suffix=".json.tmp", dir=str(episode_dir)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, json_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    new_size = json_path.stat().st_size
    sidecar_size = sum(
        (episode_dir / p).stat().st_size for p in list(rgb_files) + list(depth_files)
    )
    return True, old_size, new_size + sidecar_size


def main(data_dir: str, dry_run: bool = False):
    """
    Args:
        data_dir: Root directory containing episode_<id> subdirectories.
        dry_run: Scan only; do not write any files.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    episode_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not episode_dirs:
        print(f"No episode directories found in {root}")
        return

    migrated = 0
    skipped = 0
    total_old = 0
    total_new = 0
    failures: list[str] = []
    for episode_dir in tqdm(episode_dirs, desc="Migrating", unit="ep"):
        try:
            changed, old_size, new_size = _migrate_episode(episode_dir, dry_run)
        except Exception as e:
            failures.append(f"{episode_dir.name}: {e}")
            continue
        if changed:
            migrated += 1
            total_old += old_size
            total_new += new_size
        else:
            skipped += 1

    print(f"\nMigrated:    {migrated}")
    print(f"Skipped:     {skipped}  (already migrated or empty)")
    print(f"Failures:    {len(failures)}")
    if total_old:
        ratio = total_new / total_old if total_old else 0
        delta_gb = (total_old - total_new) / (1024**3)
        print(f"Old size:    {total_old / (1024**3):.2f} GiB")
        if not dry_run:
            print(f"New size:    {total_new / (1024**3):.2f} GiB  ({ratio:.2%})")
            print(f"Reclaimed:   {delta_gb:.2f} GiB")
    if failures:
        print("\nFailures:")
        for line in failures:
            print(f"  - {line}")
    if dry_run:
        print("\n(dry-run: no files written)")


if __name__ == "__main__":
    tyro.cli(main)
