"""Convert Genesis trajectory to UIPC-compatible convention.

Genesis internal rendering (verified via entity.get_verts SVD):
  V_world = R_z(euler_per_gear) @ V_scaled + placement_pos   (at t=0)

At any frame t:
  V_world = R(q_phys_t) @ R_body_local @ V_scaled + pos_effective
  where R_body_local = R(q_phys_build)^-1 @ R_z(euler)

For SYMMETRIC gears (ring, carrier, planets):
  q_phys_build = R_z(euler)  =>  R_body_local = I
  => q stays unchanged, only pos needs COM -> OBJ-origin

For SUN GEAR (asymmetric centroid):
  q_phys_build != R_z(euler) due to off-axis COM
  => R_body_local != I, both q and pos need conversion

q_phys_build for sun gear is obtained from the Genesis probe script
(entity.get_quat right after scene.build, before any physics step).

Usage:
  python convert_trajectory.py [--input IN.npz] [--output OUT.npz]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

MESH_SCALE = 0.0012
SUN_TEETH = 12
PLANET_TEETH = 9

# Per-gear Genesis euler (degrees)
def _gear_euler_deg(key: str) -> float:
    if key.startswith("rigid_planet_gear_"):
        idx = int(key.split("_")[-1])
        return idx * 360.0 * SUN_TEETH / PLANET_TEETH + 270.0
    return 270.0

# OBJ files used by Genesis to record the trajectory (must match!)
_ASSETS = Path(__file__).resolve().parent / "assets"
_FREE_BODY_OBJ = {
    "rigid_sun_gear":      _ASSETS / "sun_gear_handle_v2.obj",
    "rigid_carrier":       _ASSETS / "carrier.obj",
    "rigid_planet_gear_0": _ASSETS / "planet_gear_v2.obj",
    "rigid_planet_gear_1": _ASSETS / "planet_gear_v2.obj",
    "rigid_planet_gear_2": _ASSETS / "planet_gear_v2.obj",
}


def _load_obj(path):
    verts, faces = [], []
    for line in open(path, errors='replace'):
        if line.startswith('v '):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) == 4:
                faces.append(idx[:3]); faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, np.float64), np.array(faces, np.int32)


def _volume_centroid(V, F):
    """Signed-tetrahedra volume centroid (same method Genesis uses)."""
    total_vol = 0.0
    c = np.zeros(3, dtype=np.float64)
    for tri in F:
        v0, v1, v2 = V[tri[0]], V[tri[1]], V[tri[2]]
        vol = np.dot(v0, np.cross(v1, v2)) / 6.0
        total_vol += vol
        c += vol * (v0 + v1 + v2) / 4.0
    return c / total_vol


def _compute_centroids():
    """Compute volume centroids from the actual OBJ files used by Genesis."""
    result = {}
    for key, obj_path in _FREE_BODY_OBJ.items():
        V, F = _load_obj(str(obj_path))
        c_mm = _volume_centroid(V, F)
        result[key] = c_mm * MESH_SCALE
        print(f"  centroid {key}: {np.round(c_mm, 4)} mm")
    return result


FREE_BODY_CENTROID_SCALED: dict[str, np.ndarray] = {}

ALL_GEAR_KEYS = [
    "rigid_sun_gear", "rigid_carrier", "rigid_ring_gear",
    "rigid_planet_gear_0", "rigid_planet_gear_1", "rigid_planet_gear_2",
]

# Sun gear q_phys at build time (from Genesis probe with sun_gear_handle_v2.obj).
# This differs from R_z(270) due to asymmetric centroid (handle).
_SUN_Q_BUILD_WXYZ = np.array([-0.49349102, -0.08041826, 0.1392924, 0.85474974])


def _Rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.],
                     [np.sin(a),  np.cos(a), 0.],
                     [0., 0., 1.]], dtype=np.float64)


def convert(data: dict) -> dict:
    out = dict(data)

    # Compute centroids from actual OBJ files (not hardcoded)
    if not FREE_BODY_CENTROID_SCALED:
        print("Computing volume centroids from OBJ files:")
        FREE_BODY_CENTROID_SCALED.update(_compute_centroids())

    for key in ALL_GEAR_KEYS:
        if key not in data:
            continue
        poses = data[key].copy().astype(np.float64)
        n = len(poses)

        R_euler = _Rz(_gear_euler_deg(key))
        c_scaled = FREE_BODY_CENTROID_SCALED.get(key, np.zeros(3))
        is_free = key in FREE_BODY_CENTROID_SCALED

        # Determine R_body_local
        if key == "rigid_sun_gear":
            # Asymmetric: use probe q_build
            R_q0 = Rotation.from_quat([_SUN_Q_BUILD_WXYZ[1], _SUN_Q_BUILD_WXYZ[2],
                                        _SUN_Q_BUILD_WXYZ[3], _SUN_Q_BUILD_WXYZ[0]])
            R_body_local = R_q0.inv().as_matrix() @ R_euler
            needs_q_conversion = True
        else:
            # Symmetric: q_build = R_z(euler), so R_body_local = I
            R_body_local = np.eye(3)
            needs_q_conversion = False

        for i in range(n):
            qw, qx, qy, qz = poses[i, 3:]
            R_phys = Rotation.from_quat([qx, qy, qz, qw])

            if needs_q_conversion:
                R_vis = R_phys.as_matrix() @ R_body_local
            else:
                R_vis = R_phys.as_matrix()

            # Convert q
            r_vis = Rotation.from_matrix(R_vis)
            sx, sy, sz, sw = r_vis.as_quat()
            poses[i, 3:] = [sw, sx, sy, sz]

            # Convert pos: COM -> OBJ-origin
            if is_free:
                poses[i, :3] -= R_vis @ c_scaled

        out[key] = poses.astype(np.float32)
        tag = "FREE " if is_free else "FIXED"
        q_tag = " (q converted)" if needs_q_conversion else ""
        print(f"  [{tag}] {key}{q_tag}")

    return out


def main():
    here = Path(__file__).resolve().parent
    default_in = here / "trajectory_gear_sharpa.npz"
    default_out = here / "trajectory_gear_sharpa_objorigin.npz"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  type=str, default=str(default_in))
    parser.add_argument("--output", type=str, default=str(default_out))
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    raw = dict(np.load(args.input, allow_pickle=True))
    print(f"  frames: {len(raw.get('sim_time', []))}")

    print("Converting Genesis -> UIPC convention:")
    out = convert(raw)

    np.savez_compressed(args.output, **out)
    print(f"\nSaved: {args.output}")

    # Verify
    p2_path = here.parent / "planetary_with_teleop" / "ik_sequence_50.npz"
    p2 = dict(np.load(str(p2_path))) if p2_path.exists() else {}
    uipc_main = here.parent / "planetary_with_teleop" / "ik_sequence.npz"
    if not p2 and uipc_main.exists():
        p2 = dict(np.load(str(uipc_main)))

    print("\nVerification (frame 0 -> should match UIPC initial state BEFORE settle):")
    for k in ALL_GEAR_KEYS:
        if k not in out:
            continue
        q = out[k][0, 3:]
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        z = r.apply([0, 0, 1])
        pos = out[k][0, :3]
        line = f"  {k:30s}  pos_Z={pos[2]:.5f}  Z_up={z[2]:.3f}"
        if k in p2:
            p2_pos = p2[k][0, :3]
            line += f"  UIPC_Z={p2_pos[2]:.5f}  dZ={pos[2]-p2_pos[2]:.5f}"
        print(line)


if __name__ == "__main__":
    main()
