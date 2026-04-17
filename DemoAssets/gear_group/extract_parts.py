"""
Extract each mesh part from Gears_Mechanism.usdz as-is (raw USD vertices).

Since all parts share the same trivial world transform (uniform ~10x scale),
the vertex positions in USD already define the assembly layout.  We export
the raw vertices directly — no centering, no transform baking.

Output:
  parts/<name>.obj        — raw USD vertices, triangulated
  parts/transforms.json   — stage metadata + per-part stats

Usage:  python extract_parts.py
"""

from pathlib import Path
import json
import numpy as np
from pxr import Usd, UsdGeom, Gf

USDZ_PATH = Path(__file__).parent / "Gears_Mechanism.usdz"
OUT_DIR = Path(__file__).parent / "assets"

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


def triangulate_ngons(fvc, fvi):
    tris = []
    idx = 0
    for n in fvc:
        v0 = fvi[idx]
        for j in range(1, n - 1):
            tris.append((v0, fvi[idx + j], fvi[idx + j + 1]))
        idx += n
    return tris


def write_obj(filepath: Path, verts: np.ndarray, tris: list,
              normals: np.ndarray = None):
    with open(filepath, "w") as f:
        f.write(f"# {filepath.name}\n")
        f.write(f"# {len(verts)} vertices, {len(tris)} triangles\n\n")
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        if normals is not None and len(normals) == len(verts):
            f.write("\n")
            for n in normals:
                f.write(f"vn {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")
        f.write("\n")
        for t in tris:
            if normals is not None and len(normals) == len(verts):
                f.write(f"f {t[0]+1}//{t[0]+1} {t[1]+1}//{t[1]+1} {t[2]+1}//{t[2]+1}\n")
            else:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def resolve_name(prim) -> str:
    p = prim.GetPath().GetParentPath()
    while str(p) != "/":
        if p.name in FRIENDLY_NAMES:
            return FRIENDLY_NAMES[p.name]
        p = p.GetParentPath()
    return prim.GetPath().name


def gf_matrix_to_list(m: Gf.Matrix4d) -> list:
    return [[m[r][c] for c in range(4)] for r in range(4)]


def main():
    stage = Usd.Stage.Open(str(USDZ_PATH))
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"Stage  up_axis={up_axis}  metersPerUnit={meters_per_unit}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    transforms = {
        "_meta": {
            "source": USDZ_PATH.name,
            "up_axis": up_axis,
            "meters_per_unit": meters_per_unit,
            "note": "All parts share the same uniform scale (~10x). "
                    "Raw USD vertices already encode assembly positions.",
        }
    }

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        pts_raw = mesh.GetPointsAttr().Get()
        fvc = list(mesh.GetFaceVertexCountsAttr().Get())
        fvi = list(mesh.GetFaceVertexIndicesAttr().Get())
        if not pts_raw:
            continue

        normals_raw = mesh.GetNormalsAttr().Get()
        name = resolve_name(prim)
        pts = np.array(pts_raw, dtype=np.float64)
        tris = triangulate_ngons(fvc, fvi)

        normals = None
        if normals_raw and len(normals_raw) == len(pts):
            normals = np.array(normals_raw, dtype=np.float64)

        obj_path = OUT_DIR / f"{name}.obj"
        write_obj(obj_path, pts, tris, normals)

        world_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        bbox_min = pts.min(axis=0).tolist()
        bbox_max = pts.max(axis=0).tolist()

        transforms[name] = {
            "usd_prim_path": str(prim.GetPath()),
            "world_transform_4x4": gf_matrix_to_list(world_xf),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "vertices": len(pts),
            "triangles": len(tris),
            "obj": f"{name}.obj",
        }
        print(f"  [OK] {name:<20s}  {len(pts):>6} verts  {len(tris):>6} tris")

    json_path = OUT_DIR / "transforms.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2, ensure_ascii=False)

    # --- merged scene OBJ (pure v/f, no normals to avoid index mismatch) ---
    scene_path = OUT_DIR / "scene.obj"
    with open(scene_path, "w") as sf:
        sf.write("# scene.obj — all parts merged for reference\n\n")
        vert_offset = 0
        for name in (k for k in transforms if k != "_meta"):
            info = transforms[name]
            obj_file = OUT_DIR / info["obj"]
            verts, faces = [], []
            with open(obj_file) as pf:
                for line in pf:
                    if line.startswith("v "):
                        verts.append(line)
                    elif line.startswith("f "):
                        faces.append(line)
            sf.write(f"g {name}\n")
            for v in verts:
                sf.write(v)
            for fline in faces:
                toks = fline.split()
                # extract only vertex index (strip //vn or /vt/vn)
                vi = [int(tok.split("/")[0]) + vert_offset for tok in toks[1:]]
                sf.write(f"f {vi[0]} {vi[1]} {vi[2]}\n")
            vert_offset += len(verts)
    print(f"  [OK] scene.obj  ({vert_offset} verts total)")

    print(f"\nDone. {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
