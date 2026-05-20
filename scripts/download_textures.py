"""Bulk-download CC0 albedo textures from ambientCG to expand domain-rand variety.

Usage:
    python scripts/download_textures.py            # download everything missing
    python scripts/download_textures.py --dry-run  # show what would be downloaded
    python scripts/download_textures.py --limit 5  # cap per-category for a quick smoke test
    python scripts/download_textures.py --force    # re-download even if file already exists
    python scripts/download_textures.py --resolution 4K  # 1K | 2K | 4K (default 2K)

Pulls PNG asset zips, extracts only the *_Color.png (MuJoCo's renderer ignores
normal/roughness/metalness maps), and drops them into the sim textures dir using
the lowercased ambientCG asset ID as the filename (e.g. "Wood050" -> "wood050.png").

After a successful run, the script prints the XML <texture> lines to paste into
ar4_mk3/scene.xml and the Python entries to add to AVAILABLE_TEXTURES and
_TEXTURE_CLASSES, so wiring the new textures into the domain randomizer is a
copy-paste step.

CC0 source: https://ambientcg.com (no attribution required, free for any use).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TEXTURES_DIR = (
    Path(__file__).resolve().parent.parent
    / "aera"
    / "autonomous"
    / "simulation"
    / "textures"
)

# (ambientCG category, target class in _TEXTURE_CLASSES, n_assets).
# Broad coverage of natural materials. Color variety is also injected at
# runtime via per-class rgba tints in domain_rand_config_generator.py — that's
# why we don't try to source "red wood" / "blue plaster" assets here (they
# barely exist as CC0 photographs and the tint pass produces them on the fly).
# Categories below are picked for surface-type and pattern diversity.
CATEGORY_PLAN: list[tuple[str, str, int]] = [
    ("Wood",            "wood",           8),
    ("WoodFloor",       "wood_varnished", 4),
    ("Metal",           "metal_brushed",  6),
    ("MetalPlates",     "metal_brushed",  2),
    ("Plaster",         "plaster",        4),
    ("PaintedPlaster",  "plaster",        3),
    ("Wallpaper",       "plaster",        4),  # patterned painted surfaces
    ("Bricks",          "matte_rough",    5),
    ("Concrete",        "matte_rough",    4),
    ("Asphalt",         "matte_rough",    2),
    ("Rock",            "matte_rough",    3),
    ("Ground",          "matte_rough",    3),  # soil / grass / gravel
    ("Tiles",           "tile",           6),
    ("Fabric",          "fabric",         5),
    ("Leather",         "fabric",         4),  # softer/glossier than Fabric
    ("Marble",          "ceramic",        3),
    ("Plastic",         "plastic",        5),
    ("Rubber",          "plastic",        3),  # often brightly colored
    ("Cardboard",       "printed",        3),  # warm-tone packaging look
]

API_URL = "https://ambientcg.com/api/v2/full_json"
ZIP_URL_TEMPLATE = "https://ambientcg.com/get?file={asset_id}_{resolution}-PNG.zip"
VALID_RESOLUTIONS = ("1K", "2K", "4K")
USER_AGENT = "aera-texture-fetch/1.0 (+research; mujoco sim2real)"
REQUEST_TIMEOUT = 60


@dataclass
class FetchedAsset:
    asset_id: str       # e.g. "Wood050"
    filename: str       # e.g. "wood050"
    klass: str          # e.g. "wood"


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read()


def list_category_assets(category: str, limit: int) -> list[str]:
    """Return ambientCG asset IDs in a category, sorted by popularity."""
    query = urllib.parse.urlencode({
        "category": category,
        "type": "PhotoTexturePBR",
        "sort": "Popular",
        "limit": str(limit),
    })
    raw = _http_get(f"{API_URL}?{query}")
    data = json.loads(raw)
    assets = data.get("foundAssets") or []
    ids = [a["assetId"] for a in assets if "assetId" in a]
    if not ids:
        print(f"  ! no assets returned for category={category}", file=sys.stderr)
    return ids


def _extract_color_png(zip_bytes: bytes) -> bytes | None:
    """Pull the *_Color.png entry out of an ambientCG PBR zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Prefer Color, fall back to Albedo or Diffuse if the asset uses an
        # older naming convention.
        for suffix in ("_Color.png", "_Albedo.png", "_Diffuse.png"):
            for name in zf.namelist():
                if name.endswith(suffix):
                    return zf.read(name)
    return None


def download_one(asset_id: str, dest: Path, resolution: str) -> bool:
    url = ZIP_URL_TEMPLATE.format(asset_id=asset_id, resolution=resolution)
    try:
        zip_bytes = _http_get(url)
    except urllib.error.HTTPError as e:
        print(f"  ! {asset_id}: HTTP {e.code} on zip download", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"  ! {asset_id}: network error: {e}", file=sys.stderr)
        return False

    color = _extract_color_png(zip_bytes)
    if color is None:
        print(f"  ! {asset_id}: no Color/Albedo/Diffuse PNG inside zip", file=sys.stderr)
        return False

    dest.write_bytes(color)
    return True


def plan_downloads(per_category_limit: int | None) -> Iterable[tuple[str, str, str]]:
    """Yield (asset_id, target_filename_stem, class) for everything in the plan."""
    for category, klass, n in CATEGORY_PLAN:
        want = n if per_category_limit is None else min(n, per_category_limit)
        try:
            ids = list_category_assets(category, want)
        except Exception as e:
            print(f"  ! category {category} list failed: {e}", file=sys.stderr)
            continue
        for asset_id in ids[:want]:
            yield asset_id, asset_id.lower(), klass


def run(
    dry_run: bool,
    per_category_limit: int | None,
    resolution: str,
    force: bool,
) -> list[FetchedAsset]:
    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fetched: list[FetchedAsset] = []
    seen_names: set[str] = set()

    for asset_id, stem, klass in plan_downloads(per_category_limit):
        if stem in seen_names:
            continue
        seen_names.add(stem)

        dest = TEXTURES_DIR / f"{stem}.png"
        if dest.exists() and not force:
            print(f"  = {asset_id} -> {dest.name} (exists, skipping)")
            fetched.append(FetchedAsset(asset_id, stem, klass))
            continue

        if dry_run:
            tag = "re-download" if dest.exists() else "dry-run"
            print(f"  + {asset_id} -> {dest.name} [{klass}] ({tag}, {resolution})")
            fetched.append(FetchedAsset(asset_id, stem, klass))
            continue

        print(f"  + {asset_id} -> {dest.name} [{klass}] ({resolution})")
        if download_one(asset_id, dest, resolution):
            fetched.append(FetchedAsset(asset_id, stem, klass))
            # Polite spacing — ambientCG is generous but no need to hammer.
            time.sleep(0.25)

    return fetched


def print_snippets(fetched: list[FetchedAsset]) -> None:
    if not fetched:
        return
    print("\n" + "=" * 72)
    print("Paste into ar4_mk3/scene.xml <asset> block:")
    print("=" * 72)
    for f in fetched:
        print(
            f'    <texture type="2d" name="{f.filename}" '
            f'file="../../textures/{f.filename}.png"/>'
        )

    print("\n" + "=" * 72)
    print("Append to AVAILABLE_TEXTURES in aera/autonomous/envs/ar4_mk3_config.py:")
    print("=" * 72)
    for f in fetched:
        print(f'    "{f.filename}",')

    print("\n" + "=" * 72)
    print("Append to _TEXTURE_CLASSES in domain_rand_config_generator.py:")
    print("=" * 72)
    for f in fetched:
        print(f'    "{f.filename}": "{f.klass}",')

    manifest_path = TEXTURES_DIR / "manifest_new.json"
    manifest_path.write_text(
        json.dumps(
            {f.filename: {"asset_id": f.asset_id, "class": f.klass} for f in fetched},
            indent=2,
        )
    )
    print(f"\nManifest with provenance written to {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without writing files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap assets per category (smoke test).",
    )
    parser.add_argument(
        "--resolution",
        choices=VALID_RESOLUTIONS,
        default="2K",
        help="ambientCG texture resolution (default 2K — better for close-up wrist cam).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists (use after bumping --resolution).",
    )
    args = parser.parse_args()

    print(f"Texture dir: {TEXTURES_DIR}")
    print(f"Resolution: {args.resolution}  Force: {args.force}")
    fetched = run(
        dry_run=args.dry_run,
        per_category_limit=args.limit,
        resolution=args.resolution,
        force=args.force,
    )
    print(f"\n{len(fetched)} assets ready ({'dry run' if args.dry_run else 'downloaded'}).")
    print_snippets(fetched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
