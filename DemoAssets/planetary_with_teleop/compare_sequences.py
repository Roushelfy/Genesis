"""Side-by-side comparison of two gear sequences in Polyscope.

Left  = Sequence A (e.g. Genesis converted)
Right = Sequence B (e.g. UIPC output)

Each has its own timeline scrubber. Gears are offset in X for visual separation.

Usage:
  python compare_sequences.py [--seq-a A.npz] [--seq-b B.npz] [--offset 0.15]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polyscope as ps
from polyscope import imgui

_HERE = Path(__file__).resolve().parent
_GEAR = _HERE.parent / "planetary_gear"
_GEAR_ASSETS = _GEAR / "assets"

MESH_SCALE = 0.0012

_GEAR_KEYS = [
    "rigid_sun_gear", "rigid_carrier", "rigid_ring_gear",
    "rigid_planet_gear_0", "rigid_planet_gear_1", "rigid_planet_gear_2",
]
_GEAR_OBJ = {
    "rigid_sun_gear":      "sun_gear_handle_v2.obj",
    "rigid_carrier":       "carrier.obj",
    "rigid_ring_gear":     "ring_gear.obj",
    "rigid_planet_gear_0": "planet_gear_v2.obj",
    "rigid_planet_gear_1": "planet_gear_v2.obj",
    "rigid_planet_gear_2": "planet_gear_v2.obj",
}
_COLOR_A = {
    "rigid_sun_gear":      (0.8, 0.3, 0.3),
    "rigid_carrier":       (0.7, 0.4, 0.3),
    "rigid_ring_gear":     (0.6, 0.3, 0.3),
    "rigid_planet_gear_0": (0.7, 0.35, 0.35),
    "rigid_planet_gear_1": (0.7, 0.35, 0.35),
    "rigid_planet_gear_2": (0.7, 0.35, 0.35),
}
_COLOR_B = {
    "rigid_sun_gear":      (0.3, 0.3, 0.8),
    "rigid_carrier":       (0.3, 0.4, 0.7),
    "rigid_ring_gear":     (0.3, 0.3, 0.6),
    "rigid_planet_gear_0": (0.35, 0.35, 0.7),
    "rigid_planet_gear_1": (0.35, 0.35, 0.7),
    "rigid_planet_gear_2": (0.35, 0.35, 0.7),
}


def _load_obj(path):
    verts, faces = [], []
    for line in open(path, errors="replace"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) == 4:
                faces.append([idx[0], idx[1], idx[2]])
                faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, np.float64), np.array(faces, np.int32)


def _quat_wxyz_to_mat3(qw, qx, qy, qz):
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _apply_transform(V_scaled, pose7):
    px, py, pz = pose7[:3].astype(np.float64)
    R = _quat_wxyz_to_mat3(*pose7[3:].astype(np.float64))
    return (R @ V_scaled.T).T + np.array([px, py, pz])


def main():
    default_a = str(_GEAR / "trajectory_gear_sharpa_objorigin.npz")
    default_b = str(_HERE / "ik_sequence.npz")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-a", type=str, default=default_a, help="Sequence A (left, red)")
    parser.add_argument("--seq-b", type=str, default=default_b, help="Sequence B (right, blue)")
    args = parser.parse_args()

    print(f"Loading A: {args.seq_a}")
    da = dict(np.load(args.seq_a, allow_pickle=True))
    na = len(da["sim_time"]) if "sim_time" in da else max(len(da[k]) for k in _GEAR_KEYS if k in da)
    print(f"  {na} frames")

    print(f"Loading B: {args.seq_b}")
    db = dict(np.load(args.seq_b, allow_pickle=True))
    nb = len(db["sim_time"]) if "sim_time" in db else max(len(db[k]) for k in _GEAR_KEYS if k in db)
    print(f"  {nb} frames")

    # Load meshes
    mesh_V: dict[str, np.ndarray] = {}
    mesh_F: dict[str, np.ndarray] = {}
    for k, obj in _GEAR_OBJ.items():
        p = _GEAR_ASSETS / obj
        if not p.exists():
            continue
        V, F = _load_obj(str(p))
        mesh_V[k] = (V * MESH_SCALE).astype(np.float64)
        mesh_F[k] = F

    # Also load support pin
    pin_path = _GEAR_ASSETS / "support_pin.obj"
    if pin_path.exists():
        V_pin, F_pin = _load_obj(str(pin_path))
        V_pin_s = (V_pin * MESH_SCALE).astype(np.float64)
    else:
        V_pin_s = F_pin = None

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1800, 900)

    # Register meshes for A and B
    ps_a: dict[str, ps.SurfaceMesh] = {}
    ps_b: dict[str, ps.SurfaceMesh] = {}

    for k in mesh_V:
        V0 = mesh_V[k]
        F = mesh_F[k]
        if k in da:
            sm = ps.register_surface_mesh(f"A/{k}", _apply_transform(V0, da[k][0]), F)
            sm.set_color(_COLOR_A.get(k, (0.7, 0.3, 0.3)))
            sm.set_smooth_shade(True)
            sm.set_transparency(0.4)
            ps_a[k] = sm
        if k in db:
            sm = ps.register_surface_mesh(f"B/{k}", _apply_transform(V0, db[k][0]), F)
            sm.set_color(_COLOR_B.get(k, (0.3, 0.3, 0.7)))
            sm.set_smooth_shade(True)
            sm.set_transparency(0.4)
            ps_b[k] = sm

    # Pin — each side loads independently
    if V_pin_s is not None:
        if "rigid_support_pin" in da:
            V_w = _apply_transform(V_pin_s, da["rigid_support_pin"][0])
            sm = ps.register_surface_mesh("A/support_pin", V_w, F_pin)
            sm.set_color((0.6, 0.4, 0.4))
            sm.set_smooth_shade(True)
            sm.set_transparency(0.4)
        if "rigid_support_pin" in db:
            V_w = _apply_transform(V_pin_s, db["rigid_support_pin"][0])
            sm = ps.register_surface_mesh("B/support_pin", V_w, F_pin)
            sm.set_color((0.4, 0.4, 0.6))
            sm.set_smooth_shade(True)
            sm.set_transparency(0.4)

    st = {"frame_a": 0, "frame_b": 0}

    def _update_a(f):
        f = max(0, min(f, na - 1))
        st["frame_a"] = f
        for k, sm in ps_a.items():
            if k in da and f < len(da[k]):
                sm.update_vertex_positions(_apply_transform(mesh_V[k], da[k][f]))

    def _update_b(f):
        f = max(0, min(f, nb - 1))
        st["frame_b"] = f
        for k, sm in ps_b.items():
            if k in db and f < len(db[k]):
                sm.update_vertex_positions(_apply_transform(mesh_V[k], db[k][f]))

    def gui_callback():
        imgui.Text("=== Sequence Comparator ===")
        imgui.Text("Left (red) = A    Right (blue) = B")
        imgui.Separator()

        imgui.Text("Sequence A: %d frames" % na)
        changed, val = imgui.SliderInt("Frame A", st["frame_a"], 0, na - 1)
        if changed:
            _update_a(val)

        imgui.Separator()
        imgui.Text("Sequence B: %d frames" % nb)
        changed, val = imgui.SliderInt("Frame B", st["frame_b"], 0, nb - 1)
        if changed:
            _update_b(val)

        imgui.Separator()

        # Show position comparison at current frames
        imgui.Text("--- Position comparison (current frames) ---")
        for k in _GEAR_KEYS:
            if k not in da or k not in db:
                continue
            fa = min(st["frame_a"], len(da[k]) - 1)
            fb = min(st["frame_b"], len(db[k]) - 1)
            pa = da[k][fa, :3]
            pb = db[k][fb, :3]
            d = pb - pa
            short = k.replace("rigid_", "")
            imgui.Text("  %s  dZ=%.4f" % (short, d[2]))

    ps.set_user_callback(gui_callback)
    _update_a(0)
    _update_b(0)
    ps.show()


if __name__ == "__main__":
    main()
