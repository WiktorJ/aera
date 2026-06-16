"""Generate chamfered / filleted cube OBJ variants for the pickable PLA blocks.

A perfect sharp-edged cube is a strong "CAD / sim" tell — real printed parts
have beveled or rounded edges, and different prints differ. This emits a small
library of edge-treatment variants; the domain randomizer shows one per block
per episode (visual only — collision stays the box geom, so grasp physics is
unchanged).

Two families:
  * chamfer — a flat 45deg bevel. FLAT per-face normals so each facet shades
    uniformly (averaged normals would Gouraud-interpolate a diagonal seam across
    the flat faces).
  * fillet  — a rounded edge (convex hull of spheres swept to the box corners).
    SMOOTH per-vertex normals so the rounded part reads as a curve, not facets.

All variants keep their flat faces at the unit half-extent (+/-1), so every one
matches the collision box after scaling — only the edge treatment differs.

Each carries texture coordinates (vt): a UV-less MuJoCo mesh shows no texture,
which would drop the FDM layer lines. UVs are per-face planar projections with
the V axis on world z so the texture's horizontal lines run around the sides.

Usage:
    python scripts/generate_pla_block_mesh.py          # write all variants
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ASSETS_DIR = (
    Path(__file__).resolve().parent.parent
    / "aera" / "autonomous" / "simulation" / "mujoco" / "ar4_mk3" / "assets"
)

# (mesh_name_suffix, kind, amount). Mesh files are pla_block_<suffix>.obj and the
# domain randomizer references these suffixes (PLA_BLOCK_VARIANTS in
# ar4_mk3_config.py must match this list). amount is the bevel/round size as a
# fraction of the half-edge.
VARIANTS = (
    ("chamfer_s", "chamfer", 0.09),
    ("chamfer_m", "chamfer", 0.16),
    ("chamfer_l", "chamfer", 0.24),
    ("fillet_s",  "fillet",  0.12),
    ("fillet_m",  "fillet",  0.20),
    ("fillet_l",  "fillet",  0.28),
)


def _chamfered_cube_vertices(chamfer: float) -> np.ndarray:
    """24 vertices of a chamfered unit cube: each corner (+/-1,+/-1,+/-1) splits
    into 3 vertices, each pulled inward along one axis by `chamfer`."""
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                for k in range(3):
                    p = [float(sx), float(sy), float(sz)]
                    p[k] *= (1.0 - chamfer)
                    pts.append(p)
    return np.array(pts)


def _sphere_dirs(nlat: int, nlong: int) -> np.ndarray:
    """Unit directions on a lat/long grid. nlat even + nlong %4==0 guarantees the
    six axis poles (+/-x,+/-y,+/-z) are present, so the rounded box keeps exact
    flat faces at the axis extents."""
    dirs = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    for i in range(1, nlat):
        theta = np.pi * i / nlat
        for j in range(nlong):
            phi = 2.0 * np.pi * j / nlong
            dirs.append((np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)))
    return np.array(dirs)


def _filleted_cube_vertices(radius: float, nlat: int = 6,
                            nlong: int = 12) -> np.ndarray:
    """Vertices of a rounded cube = convex hull of 8 spheres (radius `radius`)
    centered at the inner corners +/-(1-radius). Flat faces land at +/-1."""
    inner = 1.0 - radius
    dirs = _sphere_dirs(nlat, nlong)
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                center = np.array([sx * inner, sy * inner, sz * inner])
                pts.extend((center + radius * dirs).tolist())
    return np.array(pts)


def _face_uv(v: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Planar UV for a vertex on a face with the given normal. Drops the
    dominant axis; for side faces the V axis maps to world z so the texture's
    horizontal layer lines render horizontally (as a real print)."""
    ax = int(np.argmax(np.abs(normal)))
    if ax == 2:
        u, w = v[0], v[1]
    elif ax == 0:
        u, w = v[1], v[2]
    else:
        u, w = v[0], v[2]
    return (u + 1.0) * 0.5, (w + 1.0) * 0.5


def build_obj(pts: np.ndarray, smooth: bool) -> str:
    hull = ConvexHull(pts)
    # Oriented outward face normals (centroid is the origin by symmetry).
    tris = []
    for simplex in hull.simplices:
        order = list(simplex)
        tri = pts[order]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n = n / np.linalg.norm(n)
        if np.dot(n, tri.mean(axis=0)) < 0:
            order = order[::-1]
            n = -n
        tris.append((order, n))

    # Smooth shading: per-vertex normal = normalized sum of incident face
    # normals. Used for fillets so the rounded part reads as a curve.
    vert_normals = None
    if smooth:
        acc = np.zeros_like(pts)
        for order, n in tris:
            for vi in order:
                acc[vi] += n
        norms = np.linalg.norm(acc, axis=1, keepdims=True)
        vert_normals = acc / np.where(norms < 1e-12, 1.0, norms)

    lines = ["# PLA block variant (generate_pla_block_mesh.py)"]
    for p in pts:
        lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    vt_lines, vn_lines, face_lines = [], [], []
    for order, n in tris:
        corners = []
        for vi in order:
            u, w = _face_uv(pts[vi], n)
            vt_lines.append(f"vt {u:.6f} {w:.6f}")
            nrm = vert_normals[vi] if smooth else n
            vn_lines.append(f"vn {nrm[0]:.6f} {nrm[1]:.6f} {nrm[2]:.6f}")
            corners.append((vi + 1, len(vt_lines), len(vn_lines)))
        face_lines.append(
            "f " + " ".join(f"{v}/{t}/{nn}" for v, t, nn in corners)
        )
    return "\n".join(lines + vt_lines + vn_lines + face_lines) + "\n"


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kind, amount in VARIANTS:
        if kind == "chamfer":
            pts = _chamfered_cube_vertices(amount)
            obj = build_obj(pts, smooth=False)
        else:
            pts = _filleted_cube_vertices(amount)
            obj = build_obj(pts, smooth=True)
        dest = ASSETS_DIR / f"pla_block_{suffix}.obj"
        dest.write_text(obj)
        print(f"wrote {dest.name} ({kind} {amount})")

    print("\nThese 6 edge OBJs are referenced (at several scales) by the "
          "PLA_BLOCK_PRESETS in ar4_mk3_config.py. The per-preset <mesh> + "
          "<geom> XML for ar4_mk3.xml is derived from that table — regenerate "
          "those if you change the edge suffixes or the preset list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
