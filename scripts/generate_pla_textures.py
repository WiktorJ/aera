"""Generate tileable grayscale FDM layer-line textures for the task objects.

The pick-and-place policy is deployed onto real 3D-PLA-printed blocks and
target plates, which carry the characteristic FDM striation pattern (horizontal
layer lines). A flat one-color surface reads as "CAD / sim", so this script
synthesizes that striation as grayscale textures the domain randomizer can
tint + shade into plausible printed parts.

Why procedural instead of scanned photos: it's reproducible (seeded), tiles
cleanly so texrepeat>1 doesn't seam, and is pure grayscale so the runtime's
rgba tint-multiply recovers any filament color from one texture.

Usage:
    python scripts/generate_pla_textures.py            # write the 3 variants
    python scripts/generate_pla_textures.py --force    # overwrite existing

Output: pla_lines_{fine,medium,coarse}.png in the sim textures dir. After a
run, add the names to PLA_LINE_TEXTURES in domain_rand_config_generator.py and
declare them in scene.xml (the script prints both snippets).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TEXTURES_DIR = (
    Path(__file__).resolve().parent.parent
    / "aera"
    / "autonomous"
    / "simulation"
    / "textures"
)

SIZE = 512  # px; plenty for ~2-5cm parts even on the close wrist cam.

# (name, n_lines, ridge_sharpness, contrast) per layer-height class. n_lines is
# the cycle count across the image height; combined with the sampler's texrepeat
# (1-2) it lands at an intentionally exaggerated-but-visible density on a ~24mm
# cube (fine layers would otherwise vanish at camera distance). ridge_sharpness
# shapes the bead profile (higher = crisper valley between layers); contrast is
# the peak-to-valley brightness swing.
VARIANTS = (
    ("pla_lines_fine",   48, 3.0, 0.13),  # ~0.1mm layers — subtle
    ("pla_lines_medium", 32, 2.2, 0.18),  # ~0.2mm layers — typical
    ("pla_lines_coarse", 20, 1.6, 0.24),  # ~0.3mm layers — pronounced
)

# Mean brightness kept high so the runtime rgba tint-multiply yields a vivid
# filament color rather than a muddy dark one (texture * rgba in MuJoCo). The
# generator (_PLA_TEXTURE_MEAN in domain_rand_config_generator.py) divides the
# tint by this so the final color matches the intended filament value; keep the
# two in sync if you change it.
BASE_LEVEL = 0.88


def _layer_profile(v: np.ndarray, n_lines: int, sharpness: float) -> np.ndarray:
    """Rounded-bead layer ridges as a function of the tiling vertical coord v.

    A raised cosine raised to a power gives the convex bead look of an extruded
    FDM bead (bright ridge crown, darker shadowed valley) instead of a flat
    sine. Period is exactly 1/n_lines so the pattern tiles seamlessly in v.
    """
    phase = np.cos(2.0 * np.pi * n_lines * v)
    bead = ((phase + 1.0) * 0.5) ** sharpness  # in [0, 1], peaked at ridges
    return bead - bead.mean()  # zero-mean so BASE_LEVEL stays the mean


def generate(name: str, n_lines: int, sharpness: float, contrast: float,
             rng: np.random.Generator) -> Image.Image:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    v = yy / SIZE  # vertical tiling coordinate; lines run horizontally

    # Nozzle wobble: low-freq horizontal waviness shifts each layer slightly in
    # v. Built from integer-frequency sines so it stays periodic (tileable) in x.
    wobble = np.zeros((SIZE, SIZE))
    for freq in (1, 2, 3):
        amp = rng.uniform(0.1, 0.4) / n_lines
        ph = rng.uniform(0, 2 * np.pi)
        wobble += amp * np.sin(2.0 * np.pi * freq * xx / SIZE + ph)

    ridges = _layer_profile(v + wobble, n_lines, sharpness)
    # Normalize the zero-mean ridge field to the requested peak-to-valley swing.
    ridges = ridges / (np.abs(ridges).max() + 1e-9) * contrast

    # Per-layer brightness drift: each printed layer reflects a touch differently.
    band = np.floor((v + wobble) * n_lines).astype(int) % n_lines
    layer_gain = rng.normal(0.0, 0.02, size=n_lines)[band]

    # Fine surface micro-roughness (kept small so texrepeat seams stay invisible).
    grain = rng.normal(0.0, 0.012, size=(SIZE, SIZE))

    img = BASE_LEVEL + ridges + layer_gain + grain
    img = np.clip(img, 0.0, 1.0)
    arr = (img * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite textures that already exist.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible patterns.")
    args = parser.parse_args()

    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    written = []
    for name, n_lines, sharpness, contrast in VARIANTS:
        dest = TEXTURES_DIR / f"{name}.png"
        if dest.exists() and not args.force:
            print(f"  = {dest.name} (exists, skipping)")
            written.append(name)
            continue
        img = generate(name, n_lines, sharpness, contrast, rng)
        img.save(dest)
        print(f"  + {dest.name}")
        written.append(name)

    print("\n" + "=" * 64)
    print("Declare in ar4_mk3/scene.xml <asset> block:")
    print("=" * 64)
    for name in written:
        print(f'    <texture type="2d" name="{name}" '
              f'file="../../textures/{name}.png"/>')
    print("\n" + "=" * 64)
    print("PLA_LINE_TEXTURES in domain_rand_config_generator.py:")
    print("=" * 64)
    print("PLA_LINE_TEXTURES = (")
    for name in written:
        print(f'    "{name}",')
    print(")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
