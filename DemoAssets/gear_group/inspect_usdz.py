"""
Inspect a USDZ file: print the full prim hierarchy, mesh stats, and materials.
Usage:  python inspect_usdz.py
"""

from pathlib import Path
from pxr import Usd, UsdGeom, UsdShade

USDZ_PATH = Path(__file__).parent / "Gears_Mechanism.usdz"


def walk(prim, depth=0):
    indent = "  " * depth
    type_name = prim.GetTypeName() or "(no type)"
    print(f"{indent}{prim.GetPath()}  [{type_name}]", end="")

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        n_pts = len(pts) if pts else 0
        n_faces = len(fvc) if fvc else 0
        print(f"  -> {n_pts} verts, {n_faces} faces", end="")

    if prim.IsA(UsdGeom.Xform):
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        if ops:
            print(f"  xformOps={[str(op.GetOpName()) for op in ops]}", end="")

    mat_binding = UsdShade.MaterialBindingAPI(prim)
    bound = mat_binding.GetDirectBinding()
    mat_path = bound.GetMaterialPath()
    if mat_path and str(mat_path) != "":
        print(f"  mat={mat_path}", end="")

    print()

    for child in prim.GetChildren():
        walk(child, depth + 1)


def main():
    stage = Usd.Stage.Open(str(USDZ_PATH))
    print(f"=== USDZ: {USDZ_PATH.name} ===")
    print(f"Up axis : {UsdGeom.GetStageUpAxis(stage)}")
    print(f"Meters/unit: {UsdGeom.GetStageMetersPerUnit(stage)}")
    print()

    root = stage.GetPseudoRoot()
    for child in root.GetChildren():
        walk(child)

    print("\n=== Mesh summary ===")
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    total_v, total_f = 0, 0
    for p in meshes:
        m = UsdGeom.Mesh(p)
        nv = len(m.GetPointsAttr().Get() or [])
        nf = len(m.GetFaceVertexCountsAttr().Get() or [])
        total_v += nv
        total_f += nf
        print(f"  {p.GetPath().name:40s}  {nv:>7} verts  {nf:>7} faces")
    print(f"  {'TOTAL':40s}  {total_v:>7} verts  {total_f:>7} faces")

    print(f"\n=== Materials ===")
    materials = [p for p in stage.Traverse() if p.IsA(UsdShade.Material)]
    for mat_prim in materials:
        mat = UsdShade.Material(mat_prim)
        print(f"  {mat_prim.GetPath()}")
        surface = mat.GetSurfaceOutput()
        if surface:
            conn = surface.GetConnectedSource()
            if conn:
                print(f"    surface shader -> {conn[0].GetPath()}")


if __name__ == "__main__":
    main()
