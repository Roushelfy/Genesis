"""
Verify that an OBJ and GLB file produce the same mesh centroid in Genesis,
and optionally convert an OBJ to a GLB in pure Python (no Blender required).

Genesis loads OBJ via trimesh (no coordinate transform) and GLB via pygltflib
(raw Y-up vertices + node transforms + Y_UP_TRANSFORM applied).  This script
replicates that exact pipeline so the centroid comparison is authoritative.

The centroid is the key metric: if OBJ and GLB have the same centroid in
Genesis internal space, _align_link shifts both link origins by the same
amount and trajectory positions apply correctly.

Usage
-----
    # Verify all standard gear pairs:
    python verify_mesh_pair.py --all

    # Verify one pair:
    python verify_mesh_pair.py sun_gear_handle.obj sun_gear_handle.glb

    # Convert OBJ → GLB (geometry only, no Blender):
    python verify_mesh_pair.py --convert sun_gear_handle.obj out.glb
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# From genesis/utils/mesh.py — row-major, translation on bottom row.
# Genesis applies as: verts_out = verts_in @ Y_UP_TRANSFORM  (row-vector)
# equivalent to trimesh.apply_transform(Y_UP_TRANSFORM.T)   (column-vector)
Y_UP_TRANSFORM = np.array(
    [[1.0,  0.0, 0.0, 0.0],
     [0.0,  0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0,  0.0, 0.0, 1.0]],
    dtype=np.float64,
)

_ASSETS = Path(__file__).resolve().parents[2] / "DemoAssets" / "planetary_gear" / "assets"

GEAR_PAIRS = [
    # (obj_name, glb_name, file_meshes_are_zup)
    # file_meshes_are_zup must match what the replay/debug scripts pass to Genesis.
    ("sun_gear_handle.obj",  "sun_gear_handle_recentered.glb", False),
    ("carrier.obj",          "carrier_recentered.glb",          False),
    ("ring_gear.obj",        "ring_gear.glb",                   True),  # loaded with zup=True in replay
    ("planet_gear.obj",      "planet_gear.glb",                 False),
]


# ── OBJ loading (trimesh, same as Genesis parse_mesh_trimesh) ─────────────────

def load_obj_verts(path: str) -> np.ndarray:
    import trimesh
    scene = trimesh.load(str(path), force="scene", process=False)
    parts = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
    verts = np.concatenate([np.array(m.vertices) for m in parts if hasattr(m, "vertices")])
    return verts.astype(np.float64)


# ── GLB loading (pygltflib + node transforms + Y_UP_TRANSFORM) ────────────────
# Replicates genesis/utils/gltf.py  parse_glb_tree + parse_mesh_glb pipeline.

def _parse_node_transform(node) -> np.ndarray:
    """Return 4×4 row-major transform (translation on bottom) for a GLTF node.
    Replicates genesis/utils/gltf.py parse_glb_tree exactly."""
    if node.matrix is not None:
        # GLTF column-major; numpy reshape reads row-by-row → effectively transposes
        # so the result is already row-major with translation on bottom. No extra .T.
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4)
    M = np.eye(4, dtype=np.float64)
    if node.translation is not None:
        M[:3, 3] = node.translation
    if node.rotation is not None:
        q = np.array(node.rotation, dtype=np.float64)  # xyzw
        x, y, z, w = q
        M[:3, :3] = np.array([
            [1 - 2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x+z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x+y*y)],
        ])
    if node.scale is not None:
        M[:3, :3] *= np.array(node.scale)
    return M.T  # → translation on bottom row (Genesis convention)


def _walk_tree(glb, node_index: int, parent_T: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Recursively collect (mesh_index, accumulated_transform) pairs."""
    node = glb.nodes[node_index]
    local_T = _parse_node_transform(node)
    # Accumulate: child @ parent  (row-vector convention, same as genesis mesh_transform @= transform)
    T = local_T @ parent_T if not np.all(parent_T == np.eye(4)) else local_T
    results = []
    for child_idx in (node.children or []):
        results.extend(_walk_tree(glb, child_idx, T))
    if node.mesh is not None:
        results.append((node.mesh, T))
    return results


def _get_accessor(glb, accessor_index: int) -> np.ndarray:
    import pygltflib
    accessor   = glb.accessors[accessor_index]
    buf_view   = glb.bufferViews[accessor.bufferView]
    buf        = glb.buffers[buf_view.buffer]
    # decode base64 data URI or read binary
    import base64, io
    uri = buf.uri or ""
    if uri.startswith("data:"):
        b64 = uri.split(",", 1)[1]
        raw = base64.b64decode(b64)
    else:
        raw = glb._glb_data or b""  # binary GLB blob

    type_counts  = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
    ctype_dtypes = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
                    5125: np.uint32, 5126: np.float32}
    n     = type_counts[accessor.type]
    dtype = ctype_dtypes[accessor.componentType]
    count = accessor.count
    off   = (buf_view.byteOffset or 0) + (accessor.byteOffset or 0)
    stride = buf_view.byteStride
    item  = np.dtype(dtype).itemsize * n

    if not stride or stride == item:
        data = np.frombuffer(raw[off: off + count * item], dtype=dtype)
    else:
        data = np.zeros(count * n, dtype=dtype)
        for i in range(count):
            chunk = raw[off + i*stride: off + i*stride + item]
            data[i*n:(i+1)*n] = np.frombuffer(chunk, dtype=dtype)
    return data.reshape(count, n)


def load_glb_verts_genesis(path: str, is_mesh_zup: bool = False) -> np.ndarray:
    """
    Extract vertex positions from a GLB exactly as Genesis does:
      pygltflib raw vertices → apply node transforms → apply Y_UP_TRANSFORM
      (only if is_mesh_zup=False, matching file_meshes_are_zup in the morph).
    """
    import pygltflib
    glb = pygltflib.GLTF2().load(str(path))
    scene_idx = glb.scene if glb.scene is not None else 0
    scene     = glb.scenes[scene_idx]

    identity = np.eye(4, dtype=np.float64)
    mesh_list = []
    for ni in scene.nodes:
        mesh_list.extend(_walk_tree(glb, ni, identity))

    all_verts = []
    for mesh_idx, mesh_T in mesh_list:
        for primitive in glb.meshes[mesh_idx].primitives:
            pts = _get_accessor(glb, primitive.attributes.POSITION).astype(np.float64)
            # Apply node transform (row-vector: pts @ T)
            pts_h = np.column_stack([pts, np.ones(len(pts))])
            pts   = (pts_h @ mesh_T)[:, :3]
            all_verts.append(pts)

    if not all_verts:
        return np.zeros((0, 3))
    all_verts = np.concatenate(all_verts)

    if not is_mesh_zup:
        # Apply Genesis Y_UP_TRANSFORM (row-vector: verts @ Y_UP_TRANSFORM)
        verts_h = np.column_stack([all_verts, np.ones(len(all_verts))])
        all_verts = (verts_h @ Y_UP_TRANSFORM)[:, :3]

    return all_verts


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(obj_path: str, glb_path: str, is_mesh_zup: bool = False, tol: float = 1e-4) -> bool:
    print(f"OBJ : {Path(obj_path).name}")
    print(f"GLB : {Path(glb_path).name}  (file_meshes_are_zup={is_mesh_zup})")

    obj_v = load_obj_verts(obj_path)
    glb_v = load_glb_verts_genesis(glb_path, is_mesh_zup=is_mesh_zup)

    obj_c = obj_v.mean(0)
    glb_c = glb_v.mean(0)
    delta = obj_c - glb_c
    dist  = float(np.linalg.norm(delta))

    print(f"  OBJ  verts={len(obj_v):6d}  centroid={np.round(obj_c, 6)}")
    print(f"  GLB  verts={len(glb_v):6d}  centroid={np.round(glb_c, 6)}")
    print(f"  Δ centroid (OBJ-GLB) = {np.round(delta, 6)}   |Δ|={dist:.6f}")

    ok = dist < tol
    if ok:
        print("  ✓  Centroids match — Genesis will produce the same link origin")
    else:
        print("  ✗  Centroids differ — Genesis will offset the link origin")
        print("     → re-export GLB from OBJ geometry, or run --convert")

    obj_bb = np.array([obj_v.min(0), obj_v.max(0)])
    glb_bb = np.array([glb_v.min(0), glb_v.max(0)])
    bb_delta = float(np.abs(obj_bb - glb_bb).max())
    bb_ok = bb_delta < tol
    print(f"  {'✓' if bb_ok else '✗'}  Bounding-box max diff: {bb_delta:.6f}")

    if len(obj_v) != len(glb_v):
        print(f"  ·  Vertex counts differ ({len(obj_v)} vs {len(glb_v)}) — normal across formats")

    print()
    return ok


# ── OBJ → GLB conversion ──────────────────────────────────────────────────────

def convert_obj_to_glb(obj_path: str, out_path: str) -> None:
    """
    Convert OBJ → GLB in pure Python using trimesh.

    Trimesh applies Y_UP_TRANSFORM (Z-up→Y-up) when exporting to GLB/GLTF, and
    Genesis applies it again when loading (file_meshes_are_zup=False). To avoid
    the double-transform, pre-apply diag([1,-1,-1]) before export so the stored
    Y-up vertices are correct for Genesis to recover: v @ diag(1,-1,-1) @ Y_UP = v @ Y_UP.T
    Materials from the OBJ's .mtl are preserved; Blender UV maps are NOT.
    """
    import trimesh
    print(f"Loading  {obj_path}")
    scene = trimesh.load(str(obj_path), force="scene", process=False)
    print(f"  {sum(len(m.vertices) for m in scene.geometry.values())} total verts")
    # trimesh.apply_transform(T) stores v @ T.T (column convention).
    # We want stored vertices = v @ Y_UP_TRANSFORM.T so that Genesis's
    # load-time @ Y_UP_TRANSFORM recovers the original coords: v @ Y_UP.T @ Y_UP = v.
    # Passing Y_UP_TRANSFORM achieves this: T.T = Y_UP_TRANSFORM.T → stored = v @ Y_UP.T.
    for m in scene.geometry.values():
        m.apply_transform(Y_UP_TRANSFORM)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    print(f"Exported {out}\n")
    print("Round-trip verification:")
    compare(obj_path, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or convert OBJ↔GLB mesh pairs.")
    parser.add_argument("obj", nargs="?", help="OBJ file path")
    parser.add_argument("glb", nargs="?", help="GLB file path (verify) or output path (--convert)")
    parser.add_argument("--convert", action="store_true",
                        help="Convert OBJ → GLB instead of verifying")
    parser.add_argument("--all", action="store_true",
                        help=f"Verify all standard gear pairs in {_ASSETS}")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Centroid match tolerance in mesh units (default 1e-4)")
    args = parser.parse_args()

    if args.all:
        results = []
        for obj_name, glb_name, zup in GEAR_PAIRS:
            obj_p = str(_ASSETS / obj_name)
            glb_p = str(_ASSETS / glb_name)
            if not Path(obj_p).exists() or not Path(glb_p).exists():
                print(f"SKIP  {obj_name} / {glb_name}  (file not found)\n")
                continue
            results.append(compare(obj_p, glb_p, is_mesh_zup=zup, tol=args.tol))
        print("=" * 50)
        print(f"Result: {sum(results)}/{len(results)} pairs pass centroid check")
        sys.exit(0 if all(results) else 1)

    if not args.obj or not args.glb:
        parser.print_help()
        sys.exit(1)

    if args.convert:
        convert_obj_to_glb(args.obj, args.glb)
    else:
        ok = compare(args.obj, args.glb, tol=args.tol)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
