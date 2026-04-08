"""Generate a procedural tomato slice surface mesh and tetrahedralize it.

Outputs:
  ../tomato_slice.obj   – surface mesh (for polyscope preview)
  ../tomato_slice.npz   – tet mesh (Vs, Ts arrays for UIPC tetmesh)
"""

from pathlib import Path

import numpy as np
import tetgen

_HERE = Path(__file__).resolve().parent
_OUT_DIR = _HERE.parent

RADIUS = 0.04          # 4 cm radius
THICKNESS = 0.008      # 8 mm thick
N_RADIAL = 24          # circumferential resolution
N_RINGS = 6            # concentric rings (excluding center)
DOME_HEIGHT = 0.001    # slight dome on top/bottom


def _make_tomato_slice_surface():
    """Generate a watertight surface mesh for a tomato slice (thick disc with domed faces)."""
    verts = []
    faces = []

    def _add_vert(x, y, z):
        verts.append([x, y, z])
        return len(verts) - 1

    half_t = THICKNESS / 2.0
    center_top = _add_vert(0, 0, half_t + DOME_HEIGHT)
    center_bot = _add_vert(0, 0, -(half_t + DOME_HEIGHT))

    ring_top = []
    ring_bot = []

    for ri in range(1, N_RINGS + 1):
        frac = ri / N_RINGS
        r = RADIUS * frac
        dome = DOME_HEIGHT * (1.0 - frac * frac)
        row_top = []
        row_bot = []
        for ai in range(N_RADIAL):
            theta = 2.0 * np.pi * ai / N_RADIAL
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            row_top.append(_add_vert(x, y, half_t + dome))
            row_bot.append(_add_vert(x, y, -(half_t + dome)))
        ring_top.append(row_top)
        ring_bot.append(row_bot)

    for ai in range(N_RADIAL):
        a1 = (ai + 1) % N_RADIAL
        faces.append([center_top, ring_top[0][ai], ring_top[0][a1]])
        faces.append([center_bot, ring_bot[0][a1], ring_bot[0][ai]])

    for ri in range(len(ring_top) - 1):
        for ai in range(N_RADIAL):
            a1 = (ai + 1) % N_RADIAL
            t0, t1 = ring_top[ri][ai], ring_top[ri][a1]
            t2, t3 = ring_top[ri + 1][ai], ring_top[ri + 1][a1]
            faces.append([t0, t2, t3])
            faces.append([t0, t3, t1])

            b0, b1 = ring_bot[ri][ai], ring_bot[ri][a1]
            b2, b3 = ring_bot[ri + 1][ai], ring_bot[ri + 1][a1]
            faces.append([b0, b3, b2])
            faces.append([b0, b1, b3])

    outer_top = ring_top[-1]
    outer_bot = ring_bot[-1]
    for ai in range(N_RADIAL):
        a1 = (ai + 1) % N_RADIAL
        faces.append([outer_top[ai], outer_bot[ai], outer_bot[a1]])
        faces.append([outer_top[ai], outer_bot[a1], outer_top[a1]])

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def _write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def main():
    print("[tomato] Generating procedural tomato slice surface mesh ...")
    verts, faces = _make_tomato_slice_surface()
    print(f"  surface: {len(verts)} verts, {len(faces)} tris")

    obj_path = _OUT_DIR / "tomato_slice.obj"
    _write_obj(obj_path, verts, faces)
    print(f"  saved surface OBJ: {obj_path}")

    print("[tomato] Tetrahedralizing with TetGen ...")
    tg = tetgen.TetGen(verts, faces)
    tg.tetrahedralize(order=1, mindihedral=10, quality=True)
    tet_verts = tg.node
    tet_elems = tg.elem
    print(f"  tet mesh: {len(tet_verts)} nodes, {len(tet_elems)} tetrahedra")

    npz_path = _OUT_DIR / "tomato_slice.npz"
    np.savez(npz_path, vertices=tet_verts, tetrahedra=tet_elems)
    print(f"  saved tet NPZ: {npz_path}")

    print("[tomato] Done.")


if __name__ == "__main__":
    main()
