"""Convert UIPC-convention sequence to Genesis convention for GLB replay.

UIPC convention (input):
  pos = OBJ-origin world position
  q   = visual rotation (R_body_local baked in for sun gear)

Genesis convention (output):
  pos = COM world position (for free bodies)
  q   = physics body quat (no R_body_local)

This is the REVERSE of convert_trajectory.py.

Usage:
  python convert_to_genesis.py [--input full_sequence.npz] [--output gs_full_sequence.npz]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

MESH_SCALE = 0.0012
SUN_TEETH = 12
PLANET_TEETH = 9

_HERE = Path(__file__).resolve().parent
_GEAR = _HERE.parent / "planetary_gear"
_ASSETS = _GEAR / "assets"

# Per-gear Genesis euler (degrees) — must match convert_trajectory.py
def _gear_euler_deg(key: str) -> float:
    if key.startswith("rigid_planet_gear_"):
        idx = int(key.split("_")[-1])
        return idx * 360.0 * SUN_TEETH / PLANET_TEETH + 270.0
    return 270.0

# Sun gear q_phys at build time (from Genesis probe with v2 mesh)
_SUN_Q_BUILD_WXYZ = np.array([-0.49349102, -0.08041826, 0.1392924, 0.85474974])

ALL_GEAR_KEYS = [
    "rigid_sun_gear", "rigid_carrier", "rigid_ring_gear",
    "rigid_planet_gear_0", "rigid_planet_gear_1", "rigid_planet_gear_2",
]

# OBJ files for centroid computation
_FREE_BODY_OBJ = {
    "rigid_sun_gear":      _ASSETS / "sun_gear_handle_v2.obj",
    "rigid_carrier":       _ASSETS / "carrier.obj",
    "rigid_planet_gear_0": _ASSETS / "planet_gear_v2.obj",
    "rigid_planet_gear_1": _ASSETS / "planet_gear_v2.obj",
    "rigid_planet_gear_2": _ASSETS / "planet_gear_v2.obj",
}


def _load_obj_verts_faces(path):
    verts, faces = [], []
    for line in open(path, errors='replace'):
        if line.startswith('v '):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            if len(idx) == 3: faces.append(idx)
            elif len(idx) == 4: faces.append(idx[:3]); faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, np.float64), np.array(faces, np.int32)


def _volume_centroid(V, F):
    total_vol = 0.0; c = np.zeros(3, np.float64)
    for tri in F:
        v0, v1, v2 = V[tri[0]], V[tri[1]], V[tri[2]]
        vol = np.dot(v0, np.cross(v1, v2)) / 6.0
        total_vol += vol; c += vol * (v0 + v1 + v2) / 4.0
    return c / total_vol


def _Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]], np.float64)


def convert_to_genesis(data: dict) -> dict:
    """Reverse of convert_trajectory.py: UIPC convention -> Genesis convention."""
    out = dict(data)

    # Compute centroids
    centroids = {}
    for key, obj_path in _FREE_BODY_OBJ.items():
        V, F = _load_obj_verts_faces(str(obj_path))
        centroids[key] = _volume_centroid(V, F) * MESH_SCALE

    for key in ALL_GEAR_KEYS:
        if key not in data:
            continue
        poses = data[key].copy().astype(np.float64)
        n = len(poses)
        c_scaled = centroids.get(key, np.zeros(3))
        is_free = key in _FREE_BODY_OBJ

        R_euler = _Rz(_gear_euler_deg(key))

        if key == "rigid_sun_gear":
            R_q0 = Rotation.from_quat([_SUN_Q_BUILD_WXYZ[1], _SUN_Q_BUILD_WXYZ[2],
                                        _SUN_Q_BUILD_WXYZ[3], _SUN_Q_BUILD_WXYZ[0]])
            R_body_local = R_q0.inv().as_matrix() @ R_euler
            R_body_local_inv = np.linalg.inv(R_body_local)
            needs_q_conversion = True
        else:
            R_body_local_inv = np.eye(3)
            needs_q_conversion = False

        for i in range(n):
            qw, qx, qy, qz = poses[i, 3:]
            R_vis = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

            if needs_q_conversion:
                R_phys = R_vis @ R_body_local_inv
            else:
                R_phys = R_vis

            r_phys = Rotation.from_matrix(R_phys)
            sx, sy, sz, sw = r_phys.as_quat()
            poses[i, 3:] = [sw, sx, sy, sz]

            # OBJ-origin -> COM
            if is_free:
                poses[i, :3] += R_vis @ c_scaled

        out[key] = poses.astype(np.float32)
        tag = "q+pos" if needs_q_conversion else "pos" if is_free else "pass"
        print(f"  [{tag:5s}] {key}")

    return out


def main():
    default_in = str(_HERE / "full_sequence.npz")
    default_out = str(_HERE / "gs_full_sequence.npz")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=default_in)
    parser.add_argument("--output", default=default_out)
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    data = dict(np.load(args.input, allow_pickle=True))
    n = len(data.get("sim_time", []))
    print(f"  {n} frames, keys: {sorted(data.keys())}")

    print("Converting UIPC -> Genesis convention:")
    out = convert_to_genesis(data)

    np.savez_compressed(args.output, **out)
    print(f"\nSaved: {args.output}  ({Path(args.output).stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
