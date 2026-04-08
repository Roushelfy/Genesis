"""Load triangle mesh from a USD file using OpenUSD (pxr).

Returns (vertices, faces) numpy arrays that can be fed directly
into ``uipc.geometry.trimesh()``.

Typical USD asset layout (e.g. Pan025.usd)::

    /root/<Name>/Visuals/<Name>        <- visual mesh  (use this)
    /root/<Name>/Collisions/<Name>     <- collision mesh (duplicate)
    /root/<Name>/Sites/...             <- auxiliary surfaces

By default only meshes under ``Visuals`` are loaded.  Pass
``prim_filter=None`` to load everything (old behaviour).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_usd_mesh(
    usd_path: str | Path,
    prim_filter: str | Sequence[str] | None = "Visuals",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract mesh geometry from a USD stage.

    Args:
        usd_path: Path to ``.usd`` / ``.usda`` file.
        prim_filter: Only include mesh prims whose ancestor path contains
            one of these tokens.  Set to ``None`` to include all meshes.
            Default ``"Visuals"`` — skips Collisions / Sites duplicates.

    Returns:
        vertices: (N, 3) float64 array
        faces:    (M, 3) int32 array  (triangulated)
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))

    if isinstance(prim_filter, str):
        prim_filter = [prim_filter]

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        prim_path = str(prim.GetPath())

        if prim_filter is not None:
            if not any(tok in prim_path for tok in prim_filter):
                continue

        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

        # Fan-triangulate each polygon
        idx = 0
        tris = []
        for c in counts:
            v0 = indices[idx]
            for k in range(1, c - 1):
                tris.append([v0, indices[idx + k], indices[idx + k + 1]])
            idx += c

        if len(tris) == 0:
            continue

        # Apply any xform on the prim
        xformable = UsdGeom.Xformable(prim)
        xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        mat = np.array(xf, dtype=np.float64)
        if not np.allclose(mat, np.eye(4)):
            ones = np.ones((len(points), 1), dtype=np.float64)
            pts4 = np.hstack([points, ones])
            points = (pts4 @ mat.T)[:, :3]

        print(f"[usd]   prim {prim_path}: {len(points)} verts, {len(tris)} tris")
        all_verts.append(points)
        all_faces.append(np.array(tris, dtype=np.int32) + vert_offset)
        vert_offset += len(points)

    if not all_verts:
        raise RuntimeError(f"No mesh prims found in {usd_path} "
                           f"(filter={prim_filter})")

    vertices = np.vstack(all_verts).astype(np.float64)
    faces = np.vstack(all_faces).astype(np.int32)

    print(f"[usd] {Path(usd_path).name}: {len(vertices)} verts, {len(faces)} tris total")
    return vertices, faces


def usd_to_obj(usd_path: str | Path, obj_path: str | Path,
               prim_filter: str | Sequence[str] | None = "Visuals") -> None:
    """Convert a USD file to Wavefront OBJ."""
    verts, faces = load_usd_mesh(usd_path, prim_filter=prim_filter)
    with open(obj_path, "w") as f:
        f.write(f"# Converted from {Path(usd_path).name}\n")
        f.write(f"# {len(verts)} vertices, {len(faces)} triangles\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"[usd] Saved {obj_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python usd_mesh_loader.py <file.usd> [output.obj]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".usd", ".obj")
    usd_to_obj(src, dst)
