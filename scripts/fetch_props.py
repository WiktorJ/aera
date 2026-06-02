"""Download background/clutter prop assets (YCB + robocasa) for the sim scene.

Usage:
    python scripts/fetch_props.py                  # download everything missing
    python scripts/fetch_props.py --dry-run        # show what would be downloaded
    python scripts/fetch_props.py --source ycb     # only YCB
    python scripts/fetch_props.py --source robocasa
    python scripts/fetch_props.py --force          # re-download even if present

Props are static clutter (contype=0, conaffinity=0) — they exist only to make
the scene look like a workshop / lab / kitchen / office rather than an empty
room. They never interact with the arm.

YCB (Yale-CMU-Berkeley) provides the lab/workshop look: hand tools, kitchen
cans/boxes, fruits. License: BSD-style, free for any use.
Source: http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/

robocasa is sourced via NVIDIA's PhysicalAI MJCF dataset on HuggingFace — each
object ships as a self-contained .zip with a model.xml MJCF snippet plus
visual/collision meshes and textures, so they're plug-and-play for MuJoCo.
License: CC-BY-4.0.
Source: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF

Outputs go to aera/autonomous/simulation/props/{ycb,robocasa}/<name>/ and are
gitignored — re-run this script after cloning.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROPS_DIR = (
    Path(__file__).resolve().parent.parent
    / "aera"
    / "autonomous"
    / "simulation"
    / "props"
)

# YCB curated subset — chosen for workshop / lab / kitchen visual variety. The
# canonical YCB object list lives at the URL below; names here use the standard
# `NNN_name` prefix. Each entry downloads as `<name>_google_16k.tgz`, ~1-5MB.
YCB_OBJECTS: list[str] = [
    # Hand tools — the workshop/lab signal.
    "035_power_drill",
    "037_scissors",
    "040_large_marker",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "051_large_clamp",
    # Cans / boxes — generic shelf clutter.
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "008_pudding_box",
    "010_potted_meat_can",
    "019_pitcher_base",
    "021_bleach_cleanser",
    # Kitchenware.
    "024_bowl",
    "025_mug",
    "026_sponge",
    "029_plate",
    # Misc desk/table props.
    "036_wood_block",
    "061_foam_brick",
    "056_tennis_ball",
    "072-a_toy_airplane",
    "073-a_lego_duplo",
    "077_rubiks_cube",
]

YCB_BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google"

# Robocasa subset (nvidia/PhysicalAI MJCF set) — curated for visual variety
# without blowing the disk budget. Each entry is one .zip under
# objects_lightwheel/. Picks lean toward distinctive silhouettes (kettle,
# plant, stool, dish rack) and away from the largest assets (>30MB) that
# don't add proportional realism.
ROBOCASA_OBJECTS: list[str] = [
    # Kitchen vessels / cookware.
    "basket",
    "blender_jug",
    "colander",
    "kettle",
    "pot",
    "saucepan",
    "pitcher",
    "tray",
    "tupperware",
    # Tools / utensils.
    "dish_brush",
    "measuring_cup",
    "peeler",
    "tongs",
    "whisk",
    "wooden_spoon",
    # Bottles / containers (good label variety).
    "flour_bag",
    "glass_cup",
    "honey_bottle",
    "mustard",
    "oil_and_vinegar_bottle",
    "soap_dispenser",
    "spray",
    "syrup_bottle",
    # Furniture / decor — strongest scene-filling silhouettes.
    "dish_rack",
    "mug_tree",
    "paper_towel_holder",
    "plant",
    "stool",
    # Floor-scale furniture variety (added to diversify the floor zone beyond
    # stools — see FLOOR_MIN_DIM in domain_rand_config_generator.py).
    "tiered_basket",
    "tiered_shelf",
    "utensil_rack",
    # Extra shelf/table diversity. Empty vases (not plants), distinctive
    # vertical knife blocks, fruit bowls, generic jars.
    "flower_vase",
    "fruit_bowl",
    "jar",
    "knife_block",
]

ROBOCASA_BASE_URL = (
    "https://huggingface.co/datasets/nvidia/"
    "PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF/resolve/main/"
    "objects_lightwheel"
)

USER_AGENT = "aera-prop-fetch/1.0 (+research; mujoco sim2real)"
REQUEST_TIMEOUT = 120


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read()


def _fetch_ycb_one(name: str, dest_dir: Path) -> bool:
    """Download and extract a single YCB object's google_16k mesh + texture."""
    url = f"{YCB_BASE_URL}/{name}_google_16k.tgz"
    try:
        blob = _http_get(url)
    except urllib.error.HTTPError as e:
        print(f"  ! {name}: HTTP {e.code}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"  ! {name}: network error: {e}", file=sys.stderr)
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(dest_dir)
    return True


def _fetch_robocasa_one(name: str, dest_dir: Path) -> bool:
    """Download and unzip a single robocasa MJCF object bundle."""
    url = f"{ROBOCASA_BASE_URL}/{name}.zip"
    try:
        blob = _http_get(url)
    except urllib.error.HTTPError as e:
        print(f"  ! {name}: HTTP {e.code}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"  ! {name}: network error: {e}", file=sys.stderr)
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dest_dir)
    return True


def _run_source(
    label: str,
    objects: list[str],
    root: Path,
    fetch_fn,
    dry_run: bool,
    force: bool,
) -> tuple[int, int]:
    """Returns (newly_downloaded, total_present)."""
    print(f"\n=== {label} ({len(objects)} curated) ===")
    root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    present = 0
    for name in objects:
        # YCB tarballs extract into a top-level dir named after the object;
        # robocasa zips extract into a similar top-level dir. Use that dir's
        # existence as the idempotency check.
        dest = root / name
        if dest.exists() and not force:
            print(f"  = {name} (exists, skipping)")
            present += 1
            continue
        if dry_run:
            print(f"  + {name} (dry-run)")
            continue
        print(f"  + {name}")
        if fetch_fn(name, root):
            downloaded += 1
            present += 1
            # Be polite to the source servers.
            time.sleep(0.2)
    return downloaded, present


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without writing files.",
    )
    parser.add_argument(
        "--source",
        choices=("all", "ycb", "robocasa"),
        default="all",
        help="Restrict to one asset source.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target dir already exists.",
    )
    args = parser.parse_args()

    print(f"Props dir: {PROPS_DIR}")
    total_new = 0
    total_present = 0

    if args.source in ("all", "ycb"):
        n, p = _run_source(
            "YCB",
            YCB_OBJECTS,
            PROPS_DIR / "ycb",
            _fetch_ycb_one,
            args.dry_run,
            args.force,
        )
        total_new += n
        total_present += p

    if args.source in ("all", "robocasa"):
        n, p = _run_source(
            "robocasa",
            ROBOCASA_OBJECTS,
            PROPS_DIR / "robocasa",
            _fetch_robocasa_one,
            args.dry_run,
            args.force,
        )
        total_new += n
        total_present += p

    print(
        f"\nDone. {total_new} newly downloaded, {total_present} present total"
        f"{' (dry run)' if args.dry_run else ''}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
