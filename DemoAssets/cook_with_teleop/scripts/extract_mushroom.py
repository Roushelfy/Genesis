"""Extract a single mushroom slice from the USDZ, normalize to 5cm diag,
tetrahedralize with TetGen, and save as .npz for FEM simulation.

Outputs:
  ../mushroom_slice.obj  – surface mesh (for polyscope preview)
  ../mushroom_slice.npz  – tet mesh (vertices, tetrahedra arrays for UIPC)
"""

from pathlib import Path
import sys

import numpy as np
import tetgen

_HERE = Path(__file__).resolve().parent
_OUT_DIR = _HERE.parent
sys.path.insert(0, str(_HERE))
from usd_mesh_loader import load_usd_mesh

MUSHROOM_USDZ = _OUT_DIR / "Mushrooms_sliced_champignons.usdz"
TARGET_DIAG = 0.05  # 5 cm


def _write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def main():
    print("[mushroom] Loading mushroom slice from USDZ ...")
    verts, faces = load_usd_mesh(MUSHROOM_USDZ, prim_filter="/RootNode/Plane/")
    print(f"  raw surface: {len(verts)} verts, {len(faces)} tris")

    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts = verts - center
    diag = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    if diag > 0:
        verts *= TARGET_DIAG / diag
    print(f"  centered & normalized to {TARGET_DIAG*100:.0f}cm diag")

    obj_path = _OUT_DIR / "mushroom_slice.obj"
    _write_obj(obj_path, verts, faces)
    print(f"  saved surface OBJ: {obj_path}")

    print("[mushroom] Tetrahedralizing with TetGen ...")
    tg = tetgen.TetGen(verts, faces)
    tg.tetrahedralize(order=1, mindihedral=10, quality=True)
    tet_verts = tg.node
    tet_elems = tg.elem
    print(f"  tet mesh: {len(tet_verts)} nodes, {len(tet_elems)} tetrahedra")

    npz_path = _OUT_DIR / "mushroom_slice.npz"
    np.savez(npz_path, vertices=tet_verts, tetrahedra=tet_elems)
    print(f"  saved tet NPZ: {npz_path}")

    print("[mushroom] Done.")


if __name__ == "__main__":
    main()
