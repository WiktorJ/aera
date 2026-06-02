"""Download CC0 painting images from the Art Institute of Chicago for use as
wall-art textures in the AR4 MK3 scene.

Usage:
    python scripts/download_paintings.py                # default: 15 images
    python scripts/download_paintings.py --num 25       # request more
    python scripts/download_paintings.py --force        # re-download

Source: https://api.artic.edu/api/v1 (public REST API). Public-domain works are
released under CC0; see the `license_text` in each API response.

Images land in aera/autonomous/simulation/props/_paintings/<id>.png, resized so
the long edge is <=512px (consistent with the prop-texture pipeline — paintings
are wall background and never closer than ~1.5m to the render camera).
PNG is required because MuJoCo's built-in texture loader only handles PNG;
JPEGs trip "Non-PNG texture, assuming custom binary file format".
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
from pathlib import Path

from PIL import Image

PAINTINGS_DIR = (
    Path(__file__).resolve().parent.parent
    / "aera" / "autonomous" / "simulation" / "props" / "_paintings"
)

# Search seeds — broad enough to surface paintings rather than drawings, prints,
# or photographs (which the AIC collection has lots of). We page through hits
# and reject anything that isn't a CC0 painting; the seed words just shape the
# visual variety. No people-portraits intentionally — wall portraits often read
# as "judging eyes" in robot-camera frames and add false context.
SEARCH_TERMS = (
    "landscape",
    "still life",
    "abstract",
    "impressionist",
    "color field",
    "garden",
    "seascape",
)

API_BASE = "https://api.artic.edu/api/v1/artworks/search"
IIIF_BASE = "https://www.artic.edu/iiif/2"
# 843px is the standard AIC IIIF size for an artwork — small file, plenty of
# detail before we downsample again.
IIIF_VARIANT = "843,"
MAX_DIM = 512

USER_AGENT = "aera-painting-fetch/1.0 (+research; mujoco sim2real)"
# AIC's IIIF image server rejects requests without their custom UA header. The
# regular User-Agent is sent too so the search-API hits look the same as any
# other client.
AIC_UA = "aera-paintings/1.0 (research; sim2real contact wiktor@aera.local)"
TIMEOUT = 60


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "AIC-User-Agent": AIC_UA,
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def _search_one_term(term: str, want: int) -> list[dict]:
    """Return up to `want` CC0 painting records for one search term."""
    out: list[dict] = []
    page = 1
    while len(out) < want and page <= 5:  # cap pages — don't drift forever
        url = (
            f"{API_BASE}?q={urllib.parse.quote(term)}"
            f"&limit=50&page={page}"
            "&fields=id,title,image_id,is_public_domain,classification_title"
        )
        try:
            blob = _http_get(url)
        except urllib.error.URLError as e:
            print(f"  ! search '{term}' p{page}: {e}", file=sys.stderr)
            return out
        data = json.loads(blob)["data"]
        if not data:
            break
        for hit in data:
            if not hit.get("is_public_domain"):
                continue
            if hit.get("classification_title") != "painting":
                continue
            if not hit.get("image_id"):
                continue
            out.append(hit)
            if len(out) >= want:
                break
        page += 1
    return out


def _save_image(image_id: str, dest: Path) -> bool:
    url = f"{IIIF_BASE}/{image_id}/full/{IIIF_VARIANT}/0/default.jpg"
    try:
        blob = _http_get(url)
    except urllib.error.URLError as e:
        print(f"  ! image {image_id}: {e}", file=sys.stderr)
        return False
    im = Image.open(io.BytesIO(blob))
    long_edge = max(im.size)
    if long_edge > MAX_DIM:
        scale = MAX_DIM / long_edge
        new_size = (max(1, round(im.size[0] * scale)),
                    max(1, round(im.size[1] * scale)))
        im = im.resize(new_size, Image.LANCZOS)
    # MuJoCo expects 8-bit RGB; convert to drop alpha / palette modes.
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(dest, format="PNG", optimize=True)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num", type=int, default=15,
                   help="Target number of paintings to download.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if file exists.")
    args = p.parse_args()

    PAINTINGS_DIR.mkdir(parents=True, exist_ok=True)
    seen_ids: set[int] = set()
    picks: list[dict] = []
    per_term = max(3, args.num // len(SEARCH_TERMS) + 1)
    for term in SEARCH_TERMS:
        if len(picks) >= args.num:
            break
        print(f"=== search: {term} ===")
        hits = _search_one_term(term, per_term)
        for h in hits:
            if h["id"] in seen_ids:
                continue
            seen_ids.add(h["id"])
            picks.append(h)
            if len(picks) >= args.num:
                break

    print(f"\n=== downloading {len(picks)} ===")
    n_saved = 0
    for h in picks:
        dest = PAINTINGS_DIR / f"painting_{h['id']}.png"
        if dest.exists() and not args.force:
            print(f"  = painting_{h['id']} ({h['title'][:50]}) exists, skipping")
            n_saved += 1
            continue
        print(f"  + painting_{h['id']} ({h['title'][:50]})")
        if _save_image(h["image_id"], dest):
            n_saved += 1
            time.sleep(0.2)
    print(f"\nDone. {n_saved} paintings present in {PAINTINGS_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
