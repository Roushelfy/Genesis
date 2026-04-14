"""
Decimate collision meshes referenced in marvin_sharpa.urdf.

Rule: collision meshes whose filename does NOT contain 'visual'.
- Rename original -> xxxx_v0.STL (backup)
- Decimate to ~12% face count and save as the original filename
- Skip meshes that already have a _v0.STL backup
"""

import os
import shutil
import trimesh
import fast_simplification
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DECIMATE_RATIO = 0.12  # keep ~12% of faces (range: 10-15%)

# All collision meshes to decimate (no 'visual' in filename, excluding left_PP_mod)
COLLISION_MESHES = [
    # ---- left hand ----
    "hands/sharpa/left/meshes/wrist_collision.STL",
    "hands/sharpa/left/meshes/left_hand_C_MC.STL",
    "hands/sharpa/left/meshes/left_thumb_MC.STL",
    "hands/sharpa/left/meshes/left_thumb_PP.STL",
    "hands/sharpa/left/meshes/DP_HB1_TH.STL",
    "hands/sharpa/left/meshes/elastomer_HB1_TH.STL",
    "hands/sharpa/left/meshes/MCP_VL.STL",
    "hands/sharpa/left/meshes/left_PP.STL",
    "hands/sharpa/left/meshes/left_MP.STL",
    "hands/sharpa/left/meshes/DP_HB1_4F.STL",
    "hands/sharpa/left/meshes/elastomer_HB1_4F.STL",
    "hands/sharpa/left/meshes/left_pinky_MC.STL",
    # ---- right hand ----
    "hands/sharpa/right/meshes/wrist_collision.STL",
    "hands/sharpa/right/meshes/right_hand_C_MC.STL",
    "hands/sharpa/right/meshes/right_thumb_MC.STL",
    "hands/sharpa/right/meshes/right_thumb_PP.STL",
    "hands/sharpa/right/meshes/DP_HB1_TH.STL",
    "hands/sharpa/right/meshes/elastomer_HB1_TH.STL",
    "hands/sharpa/right/meshes/MCP_VL.STL",
    "hands/sharpa/right/meshes/right_PP.STL",
    "hands/sharpa/right/meshes/right_MP.STL",
    "hands/sharpa/right/meshes/DP_HB1_4F.STL",
    "hands/sharpa/right/meshes/elastomer_HB1_4F.STL",
    "hands/sharpa/right/meshes/right_pinky_MC.STL",
]


def get_full_path(relative_mesh_path):
    return os.path.join(BASE_DIR, relative_mesh_path.replace("/", os.sep))


def get_v0_path(mesh_path):
    """e.g. foo.STL -> foo_v0.STL"""
    base, ext = os.path.splitext(mesh_path)
    return f"{base}_v0{ext}"


def list_all_targets():
    """List collision meshes and their processing status."""
    print(f"Collision meshes to decimate ({len(COLLISION_MESHES)} total):\n")

    to_process = []
    already_done = []

    for m in COLLISION_MESHES:
        full = get_full_path(m)
        v0 = get_v0_path(full)
        exists = os.path.exists(full)
        has_v0 = os.path.exists(v0)

        if has_v0:
            already_done.append(m)
            print(f"  [SKIP] {m}  (already has _v0 backup)")
        elif not exists:
            print(f"  [MISS] {m}  (file not found!)")
        else:
            face_count = "?"
            try:
                mesh = trimesh.load(full)
                face_count = len(mesh.faces)
            except Exception:
                pass
            to_process.append(m)
            print(f"  [TODO] {m}  ({face_count} faces)")

    print(f"\nSummary: {len(to_process)} to process, {len(already_done)} already done")
    return to_process


def decimate_one(relative_mesh_path, ratio=DECIMATE_RATIO):
    """Decimate a single mesh: backup original as _v0, save decimated as original."""
    full_path = get_full_path(relative_mesh_path)
    v0_path = get_v0_path(full_path)

    if os.path.exists(v0_path):
        print(f"SKIP (v0 exists): {relative_mesh_path}")
        return

    if not os.path.exists(full_path):
        print(f"SKIP (not found): {relative_mesh_path}")
        return

    mesh = trimesh.load(full_path)
    original_faces = len(mesh.faces)
    original_verts = len(mesh.vertices)
    target_faces = max(int(original_faces * ratio), 4)

    print(f"\nProcessing: {relative_mesh_path}")
    print(f"  Original: {original_faces} faces, {original_verts} vertices")
    print(f"  Target:   {target_faces} faces ({ratio*100:.0f}%)")

    shutil.copy2(full_path, v0_path)
    print(f"  Backed up to: {os.path.basename(v0_path)}")

    verts_out, faces_out = fast_simplification.simplify(
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int32),
        target_count=target_faces,
    )
    decimated = trimesh.Trimesh(vertices=verts_out, faces=faces_out)
    print(f"  Result:   {len(decimated.faces)} faces, {len(decimated.vertices)} vertices")
    print(f"  Reduction: {len(decimated.faces)/original_faces*100:.1f}% of original")

    decimated.export(full_path, file_type="stl")
    print(f"  Saved to: {os.path.basename(full_path)}")

    return {
        "path": relative_mesh_path,
        "original_faces": original_faces,
        "decimated_faces": len(decimated.faces),
        "ratio": len(decimated.faces) / original_faces,
    }


def decimate_all(ratio=DECIMATE_RATIO):
    """Decimate all collision meshes that haven't been processed yet."""
    targets = list_all_targets()
    print(f"\n{'='*60}")
    print(f"Decimating {len(targets)} meshes at {ratio*100:.0f}% ratio...")
    print(f"{'='*60}")
    results = []
    for m in targets:
        r = decimate_one(m, ratio)
        if r:
            results.append(r)
    print(f"\nDone! Processed {len(results)} meshes.")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        decimate_all()
    elif len(sys.argv) > 1 and sys.argv[1] == "--one":
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        targets = list_all_targets()
        if idx < len(targets):
            print(f"\n{'='*60}")
            decimate_one(targets[idx])
        else:
            print(f"Index {idx} out of range (max {len(targets)-1})")
    else:
        list_all_targets()
