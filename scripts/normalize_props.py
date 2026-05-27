"""Normalize downloaded YCB + robocasa prop assets into a flat manifest.

Usage:
    python scripts/normalize_props.py            # process everything, write manifest
    python scripts/normalize_props.py --dry-run  # report what would be done
    python scripts/normalize_props.py --force    # rebuild merged meshes from scratch

For each prop asset under aera/autonomous/simulation/props/{ycb,robocasa}/:
  - Drop YCB collision-only files (nontextured.{ply,stl}) — props are visual-only.
  - For robocasa variants with multiple visual meshes, concatenate the OBJ vertex
    + face streams into one merged mesh in props/_merged/<id>.obj. All robocasa
    visual meshes within a variant share the same texture file, so merging only
    loses per-part specular/shininess (which the DR pipeline overrides anyway).
  - Compute the asset's axis-aligned bounding box from the (merged) mesh.
  - Emit aera/autonomous/simulation/props/_manifest.json — the single source of
    truth that scene XML generation (step 3) and the runtime sampler (step 4)
    both consume.

Manifest entries look like:

    {
      "id": "ycb_037_scissors",
      "source": "ycb",
      "mesh_rel": "ycb/037_scissors/google_16k/textured.obj",
      "texture_rel": "ycb/037_scissors/google_16k/texture_map.png",
      "aabb_min": [...], "aabb_max": [...], "size": [w, d, h],
      "themes": ["workshop", "office"]
    }

`*_rel` paths are relative to props/, so scene.xml refers to them as
`../../props/<rel>` and a Python caller does `PROPS_DIR / rel`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

PROPS_DIR = (
    Path(__file__).resolve().parent.parent
    / "aera"
    / "autonomous"
    / "simulation"
    / "props"
)
MANIFEST_PATH = PROPS_DIR / "_manifest.json"
MERGED_DIR = PROPS_DIR / "_merged"
TEXTURES_DIR = PROPS_DIR / "_textures"

# YCB textures ship as 4K (16MB uncompressed RGB → 64MB RGBA in MuJoCo's
# tex_data buffer). With ~28 YCB props alone that exceeds the 2GB int32 limit
# on ntexdata and the model fails to compile. 512 is more than enough for
# background clutter that's never closer than ~50cm to the camera.
DEFAULT_MAX_TEX_DIM = 512

# Themes guide per-scene sampling: the runtime sampler picks one theme per
# episode and draws props whose themes list includes it. An asset can belong to
# multiple themes — a mug is both kitchen and office, scissors are both
# workshop and office. Hand-curated, easy to edit.
YCB_THEMES: dict[str, list[str]] = {
    "035_power_drill":         ["workshop"],
    "037_scissors":            ["workshop", "office"],
    "040_large_marker":        ["workshop", "office"],
    "042_adjustable_wrench":   ["workshop"],
    "043_phillips_screwdriver":["workshop"],
    "044_flat_screwdriver":    ["workshop"],
    "048_hammer":              ["workshop"],
    "050_medium_clamp":        ["workshop"],
    "051_large_clamp":         ["workshop"],
    "002_master_chef_can":     ["kitchen", "lab"],
    "003_cracker_box":         ["kitchen"],
    "004_sugar_box":           ["kitchen"],
    "005_tomato_soup_can":     ["kitchen"],
    "006_mustard_bottle":      ["kitchen"],
    "008_pudding_box":         ["kitchen"],
    "010_potted_meat_can":     ["kitchen"],
    "019_pitcher_base":        ["kitchen", "lab"],
    "021_bleach_cleanser":     ["lab", "workshop"],
    "024_bowl":                ["kitchen", "lab"],
    "025_mug":                 ["kitchen", "office"],
    "026_sponge":              ["kitchen", "workshop"],
    "029_plate":               ["kitchen"],
    "036_wood_block":          ["workshop"],
    "061_foam_brick":          ["workshop", "lab"],
    "056_tennis_ball":         ["office"],
    "072-a_toy_airplane":      ["office"],
    "073-a_lego_duplo":        ["office"],
    "077_rubiks_cube":         ["office"],
}

ROBOCASA_THEMES: dict[str, list[str]] = {
    "basket":                  ["kitchen"],
    "blender_jug":             ["kitchen"],
    "colander":                ["kitchen"],
    "kettle":                  ["kitchen"],
    "pot":                     ["kitchen"],
    "saucepan":                ["kitchen"],
    "pitcher":                 ["kitchen"],
    "tray":                    ["kitchen"],
    "tupperware":              ["kitchen", "office"],
    "dish_brush":              ["kitchen"],
    "measuring_cup":           ["kitchen", "lab"],
    "peeler":                  ["kitchen"],
    "tongs":                   ["kitchen"],
    "whisk":                   ["kitchen"],
    "wooden_spoon":            ["kitchen"],
    "flour_bag":               ["kitchen"],
    "glass_cup":               ["kitchen", "lab", "office"],
    "honey_bottle":            ["kitchen"],
    "mustard":                 ["kitchen"],
    "oil_and_vinegar_bottle":  ["kitchen"],
    "soap_dispenser":          ["kitchen", "lab"],
    "spray":                   ["kitchen", "lab", "workshop"],
    "syrup_bottle":            ["kitchen"],
    "dish_rack":               ["kitchen"],
    "mug_tree":                ["kitchen"],
    "paper_towel_holder":      ["kitchen", "office"],
    "plant":                   ["kitchen", "office"],
    "stool":                   ["kitchen", "office", "workshop"],
}

YCB_COLLISION_FILES = ("nontextured.ply", "nontextured.stl", "kinbody.xml")


def _normalize_texture(src: Path, max_dim: int, dry_run: bool) -> Path:
    """Resize + dedup a source texture into _textures/<sha1>.png.

    Many robocasa variants ship the same texture file (T_BC001.png) — hashing
    the source bytes means we register it once in MuJoCo's asset block and
    every prop that references it shares the same texid. That alone roughly
    halves ntexdata.

    Images are downsampled to fit max_dim on the long edge using LANCZOS
    (preserves edge crispness for label text on cans/boxes). Original aspect
    is preserved.
    """
    raw = src.read_bytes()
    digest = hashlib.sha1(raw).hexdigest()[:16]
    dest = TEXTURES_DIR / f"{digest}.png"
    if dest.exists():
        return dest
    if dry_run:
        return dest  # path is correct; just don't write
    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    im = Image.open(src)
    long_edge = max(im.size)
    if long_edge > max_dim:
        scale = max_dim / long_edge
        new_size = (max(1, round(im.size[0] * scale)),
                    max(1, round(im.size[1] * scale)))
        im = im.resize(new_size, Image.LANCZOS)
    # Convert palette / 16-bit / etc. to plain RGB — MuJoCo expects 8-bit
    # RGB/RGBA and silently fails on exotic encodings.
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    im.save(dest, format="PNG", optimize=True)
    return dest


@dataclass
class PropEntry:
    id: str
    source: str
    category: str  # ycb name (e.g. "037_scissors") or robocasa cat (e.g. "whisk")
    mesh_rel: str
    texture_rel: str | None
    aabb_min: list[float]
    aabb_max: list[float]
    size: list[float]
    themes: list[str]


def _parse_obj_vertices(obj_path: Path) -> list[tuple[float, float, float]]:
    """Return just the `v x y z` lines as float triples. Cheap streaming parse —
    we only need vertex positions for AABB / merge counts, not faces/normals."""
    verts: list[tuple[float, float, float]] = []
    with obj_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                # Accept v with optional w; only take first 3.
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return verts


def _aabb(verts: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]:
    if not verts:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


_FACE_TOKEN = re.compile(r"(-?\d+)(?:/(-?\d+)?(?:/(-?\d+))?)?")


def _rewrite_face_indices(line: str, v_off: int, vt_off: int, vn_off: int) -> str:
    """Shift 1-based v/vt/vn indices in an `f` line by the running offsets so
    we can concatenate multiple OBJ files into one mesh without index aliasing.
    Handles `f v`, `f v/vt`, `f v//vn`, `f v/vt/vn`. Ignores negative (relative)
    indices — robocasa meshes don't use them; if any showed up we'd warn loudly."""
    tokens = line.split()
    out = [tokens[0]]  # "f"
    for tok in tokens[1:]:
        m = _FACE_TOKEN.match(tok)
        if not m:
            out.append(tok)
            continue
        v, vt, vn = m.group(1), m.group(2), m.group(3)
        if int(v) < 0:
            raise ValueError(f"relative face index not supported: {tok}")
        parts = [str(int(v) + v_off)]
        if vt is not None:
            parts.append(str(int(vt) + vt_off) if vt else "")
            if vn is not None:
                parts.append(str(int(vn) + vn_off) if vn else "")
        elif vn is not None:
            # `f v//vn` form
            parts.append("")
            parts.append(str(int(vn) + vn_off))
        out.append("/".join(parts))
    return " ".join(out) + "\n"


def _merge_objs(obj_paths: list[Path], out_path: Path) -> None:
    """Concatenate multiple OBJ visual meshes into one mesh file.

    Faces are re-indexed against the merged vertex/uv/normal streams so the
    combined file is a single self-contained mesh. We deliberately strip
    `usemtl`/`mtllib` lines because the merged result is rendered with one
    material assigned by the DR pipeline."""
    v_total = vt_total = vn_total = 0
    with out_path.open("w") as out:
        out.write(f"# merged from {len(obj_paths)} robocasa visual meshes\n")
        for src in obj_paths:
            v_in = vt_in = vn_in = 0
            with src.open("r") as f:
                for line in f:
                    if line.startswith(("mtllib", "usemtl", "o ", "g ")):
                        continue
                    if line.startswith("v "):
                        v_in += 1
                        out.write(line)
                    elif line.startswith("vt "):
                        vt_in += 1
                        out.write(line)
                    elif line.startswith("vn "):
                        vn_in += 1
                        out.write(line)
                    elif line.startswith("f "):
                        out.write(_rewrite_face_indices(
                            line.rstrip("\n"), v_total, vt_total, vn_total
                        ))
                    # Drop everything else (s smoothing groups, comments).
            v_total += v_in
            vt_total += vt_in
            vn_total += vn_in


def _parse_robocasa_model(model_xml: Path) -> tuple[list[str], str | None]:
    """Return (visual_mesh_filenames, texture_filename) extracted from a
    robocasa model.xml. Visual meshes are the ones used by geoms with
    class="visual". Texture is the file referenced by the first material those
    geoms point at — robocasa always uses one shared texture per variant."""
    tree = ET.parse(model_xml)
    root = tree.getroot()

    mesh_files = {m.get("name"): m.get("file") for m in root.iter("mesh")}
    texture_files = {t.get("name"): t.get("file") for t in root.iter("texture")}
    material_to_texture = {
        m.get("name"): m.get("texture") for m in root.iter("material")
    }

    visual_mesh_names: list[str] = []
    visual_material_names: list[str] = []
    for geom in root.iter("geom"):
        if geom.get("class") != "visual":
            continue
        mesh = geom.get("mesh")
        mat = geom.get("material")
        if mesh and mesh in mesh_files:
            visual_mesh_names.append(mesh)
        if mat and mat in material_to_texture:
            visual_material_names.append(mat)

    visual_objs = [mesh_files[n] for n in visual_mesh_names if mesh_files.get(n)]
    texture_file = None
    for mat_name in visual_material_names:
        tex_name = material_to_texture.get(mat_name)
        if tex_name and tex_name in texture_files:
            texture_file = texture_files[tex_name]
            break
    return visual_objs, texture_file


def process_ycb(force: bool, dry_run: bool, max_tex_dim: int) -> list[PropEntry]:
    ycb_root = PROPS_DIR / "ycb"
    if not ycb_root.exists():
        print("  (no ycb/ — run scripts/fetch_props.py first)")
        return []
    entries: list[PropEntry] = []
    for obj_dir in sorted(p for p in ycb_root.iterdir() if p.is_dir()):
        name = obj_dir.name
        themes = YCB_THEMES.get(name)
        if not themes:
            print(f"  ? ycb/{name}: no theme tag, skipping")
            continue
        google = obj_dir / "google_16k"
        mesh = google / "textured.obj"
        tex = google / "texture_map.png"
        if not mesh.exists() or not tex.exists():
            print(f"  ! ycb/{name}: missing textured.obj or texture_map.png")
            continue
        if not dry_run:
            for cf in YCB_COLLISION_FILES:
                stale = google / cf
                if stale.exists():
                    stale.unlink()
        verts = _parse_obj_vertices(mesh)
        lo, hi = _aabb(verts)
        size = [hi[i] - lo[i] for i in range(3)]
        tex_norm = _normalize_texture(tex, max_tex_dim, dry_run)
        entries.append(PropEntry(
            id=f"ycb_{name}",
            source="ycb",
            category=name,
            mesh_rel=str(mesh.relative_to(PROPS_DIR)),
            texture_rel=str(tex_norm.relative_to(PROPS_DIR)),
            aabb_min=lo, aabb_max=hi, size=size,
            themes=themes,
        ))
        print(f"  + ycb_{name} ({len(verts)} verts, size={[round(s,3) for s in size]})")
    return entries


def process_robocasa(force: bool, dry_run: bool, max_tex_dim: int) -> list[PropEntry]:
    rc_root = PROPS_DIR / "robocasa"
    if not rc_root.exists():
        print("  (no robocasa/ — run scripts/fetch_props.py first)")
        return []
    if not dry_run:
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[PropEntry] = []
    for category_dir in sorted(p for p in rc_root.iterdir() if p.is_dir()):
        category = category_dir.name
        themes = ROBOCASA_THEMES.get(category)
        if not themes:
            print(f"  ? robocasa/{category}: no theme tag, skipping")
            continue
        for variant_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            variant = variant_dir.name
            model_xml = variant_dir / "model.xml"
            if not model_xml.exists():
                continue
            try:
                obj_names, tex_name = _parse_robocasa_model(model_xml)
            except ET.ParseError as e:
                print(f"  ! robocasa/{category}/{variant}: XML parse failed: {e}")
                continue
            if not obj_names:
                print(f"  ! robocasa/{category}/{variant}: no visual meshes found")
                continue
            obj_paths = [variant_dir / o for o in obj_names if (variant_dir / o).exists()]
            if not obj_paths:
                continue
            asset_id = f"robocasa_{category}_{variant}"
            if len(obj_paths) == 1:
                # No merge needed — reference the visual OBJ in place.
                mesh_path = obj_paths[0]
                mesh_rel = str(mesh_path.relative_to(PROPS_DIR))
            else:
                merged_path = MERGED_DIR / f"{asset_id}.obj"
                if dry_run:
                    print(f"  + {asset_id} (would merge {len(obj_paths)} meshes)")
                    # Use the first source for AABB estimation in dry run.
                    mesh_path = obj_paths[0]
                else:
                    if force or not merged_path.exists():
                        _merge_objs(obj_paths, merged_path)
                    mesh_path = merged_path
                mesh_rel = str(merged_path.relative_to(PROPS_DIR))
            tex_path = variant_dir / tex_name if tex_name else None
            if tex_path and tex_path.exists():
                tex_norm = _normalize_texture(tex_path, max_tex_dim, dry_run)
                tex_rel = str(tex_norm.relative_to(PROPS_DIR))
            else:
                tex_rel = None
            verts = _parse_obj_vertices(mesh_path)
            lo, hi = _aabb(verts)
            size = [hi[i] - lo[i] for i in range(3)]
            entries.append(PropEntry(
                id=asset_id,
                source="robocasa",
                category=category,
                mesh_rel=mesh_rel,
                texture_rel=tex_rel,
                aabb_min=lo, aabb_max=hi, size=size,
                themes=themes,
            ))
            tag = "" if len(obj_paths) == 1 else f" (merged x{len(obj_paths)})"
            print(
                f"  + {asset_id}{tag} "
                f"({len(verts)} verts, size={[round(s,3) for s in size]})"
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-merge robocasa meshes even if the cached file exists.",
    )
    parser.add_argument(
        "--clean-merged", action="store_true",
        help="Delete props/_merged/ before running (useful after editing the merger).",
    )
    parser.add_argument(
        "--clean-textures", action="store_true",
        help="Delete props/_textures/ before running (use after bumping --max-tex-dim).",
    )
    parser.add_argument(
        "--max-tex-dim", type=int, default=DEFAULT_MAX_TEX_DIM,
        help="Downsample textures so the longer edge is at most this many px.",
    )
    args = parser.parse_args()

    if args.clean_merged and MERGED_DIR.exists() and not args.dry_run:
        shutil.rmtree(MERGED_DIR)
        print(f"Cleaned {MERGED_DIR}")
    if args.clean_textures and TEXTURES_DIR.exists() and not args.dry_run:
        shutil.rmtree(TEXTURES_DIR)
        print(f"Cleaned {TEXTURES_DIR}")

    print(f"Props dir: {PROPS_DIR}  max_tex_dim={args.max_tex_dim}")
    print("\n=== YCB ===")
    ycb = process_ycb(force=args.force, dry_run=args.dry_run, max_tex_dim=args.max_tex_dim)
    print("\n=== Robocasa ===")
    rc = process_robocasa(force=args.force, dry_run=args.dry_run, max_tex_dim=args.max_tex_dim)

    all_entries = ycb + rc
    print(
        f"\n{len(all_entries)} entries total "
        f"({len(ycb)} ycb + {len(rc)} robocasa)"
    )
    if args.dry_run:
        print("(dry run — manifest not written)")
        return 0

    manifest = {
        "version": 1,
        "props_dir": str(PROPS_DIR.relative_to(PROPS_DIR.parent.parent.parent.parent)),
        "props": [asdict(e) for e in all_entries],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {MANIFEST_PATH.relative_to(PROPS_DIR.parent.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
