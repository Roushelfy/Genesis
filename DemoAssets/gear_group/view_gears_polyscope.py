"""
Load Gears_Mechanism.usdz, extract every mesh part with its world transform,
and display them in Polyscope with per-part colouring.

Usage:  python view_gears_polyscope.py
"""

from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf
import polyscope as ps

USDZ_PATH = Path(__file__).parent / "Gears_Mechanism.usdz"

PART_COLORS = [
    (0.85, 0.32, 0.30),  # red
    (0.30, 0.68, 0.85),  # blue
    (0.40, 0.82, 0.45),  # green
    (0.92, 0.72, 0.20),  # gold
    (0.65, 0.40, 0.85),  # purple
    (0.90, 0.55, 0.25),  # orange
    (0.35, 0.80, 0.75),  # teal
    (0.80, 0.45, 0.65),  # pink
    (0.55, 0.55, 0.55),  # grey
]

FRIENDLY_NAMES = {
    "pCube1":       "base_plate",
    "pTorus2":      "gear_ring_A",
    "pCylinder8":   "lever_arm",
    "pCylinder9":   "axle_pin_1",
    "pCylinder10":  "axle_pin_2",
    "pCylinder11":  "axle_pin_3",
    "polySurface5": "gear_body",
    "pTorus4":      "gear_ring_B",
    "pSphere20":    "lock_ball",
}


def gf_matrix_to_np(m: Gf.Matrix4d) -> np.ndarray:
    """Convert a pxr Gf.Matrix4d to a 4x4 numpy array (row-major)."""
    return np.array([[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)


def triangulate_ngons(face_vertex_counts, face_vertex_indices):
    """Fan-triangulate arbitrary n-gon face lists into (N, 3) triangle array."""
    tris = []
    idx = 0
    for n in face_vertex_counts:
        v0 = face_vertex_indices[idx]
        for j in range(1, n - 1):
            tris.append([v0, face_vertex_indices[idx + j], face_vertex_indices[idx + j + 1]])
        idx += n
    return np.array(tris, dtype=np.int32)


def extract_meshes(stage):
    """Walk the stage and return a list of (name, vertices_world, triangles)."""
    parts = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        fvc_attr = mesh.GetFaceVertexCountsAttr().Get()
        fvi_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts_attr or not fvc_attr or not fvi_attr:
            continue

        pts = np.array(pts_attr, dtype=np.float64)
        fvc = list(fvc_attr)
        fvi = list(fvi_attr)
        tris = triangulate_ngons(fvc, fvi)

        world_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        mat = gf_matrix_to_np(world_xf)

        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_world = (mat @ pts_h.T).T[:, :3]

        # Walk up ancestors to find a recognisable part name
        label = prim.GetPath().name
        p = prim.GetPath().GetParentPath()
        while str(p) != "/":
            if p.name in FRIENDLY_NAMES:
                label = FRIENDLY_NAMES[p.name]
                break
            p = p.GetParentPath()
        parts.append((label, pts_world, tris))

    return parts


def main():
    stage = Usd.Stage.Open(str(USDZ_PATH))
    up = UsdGeom.GetStageUpAxis(stage)
    meters = UsdGeom.GetStageMetersPerUnit(stage)

    print(f"Stage up-axis: {up},  metersPerUnit: {meters}")
    print(f"Extracting meshes ...")

    parts = extract_meshes(stage)

    print(f"\nFound {len(parts)} mesh parts:")
    print(f"{'#':<4} {'Name':<20} {'Verts':>8} {'Tris':>8}")
    print("-" * 44)
    for i, (name, verts, tris) in enumerate(parts):
        print(f"{i:<4} {name:<20} {len(verts):>8} {len(tris):>8}")

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1600, 1000)

    for i, (name, verts, tris) in enumerate(parts):
        color = PART_COLORS[i % len(PART_COLORS)]
        sm = ps.register_surface_mesh(name, verts, tris, enabled=True)
        sm.set_color(color)
        sm.set_smooth_shade(True)
        sm.set_edge_width(0.5)

    ps.reset_camera_to_home_view()
    print("\nPolyscope window is open. Close the window to exit.")
    ps.show()


if __name__ == "__main__":
    main()
