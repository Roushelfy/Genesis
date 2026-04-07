"""Pure UIPC scene: replay pan & spatula trajectories with contact disabled.

Reads the cooking teleop trajectory (pos + quat per frame) and drives
pan / spatula as kinematic affine bodies via SoftTransformConstraint.

Usage:
    python replay_uipc.py [--traj PATH] [--speed 1.0] [--broccoli]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Animation, Logger, Timer, builtin, view
from uipc import Engine, Scene, SceneIO, World
from uipc.constitution import AffineBodyConstitution, SoftTransformConstraint
from uipc.geometry import SimplicialComplexIO, ground, label_surface
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

from asset_dir import AssetDir

_HERE = Path(__file__).resolve().parent
_ASSET_ROOT = _HERE.parent
_COOK_ROOT = _ASSET_ROOT.parent / "cook"

PAN_OBJ = _ASSET_ROOT / "pan.obj"
SPATULA_OBJ = _ASSET_ROOT / "spatula.obj"
BROCCOLI_OBJ = _ASSET_ROOT / "broccoli.obj"
DEFAULT_TRAJ = _COOK_ROOT / "trajectories" / "cooking_demo.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def quat_pos_to_4x4(quat, pos):
    """[w, x, y, z] quaternion + [x, y, z] position -> 4x4 homogeneous matrix."""
    w, x, y, z = quat
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = 1 - 2 * (y * y + z * z)
    M[0, 1] = 2 * (x * y - w * z)
    M[0, 2] = 2 * (x * z + w * y)
    M[1, 0] = 2 * (x * y + w * z)
    M[1, 1] = 1 - 2 * (x * x + z * z)
    M[1, 2] = 2 * (y * z - w * x)
    M[2, 0] = 2 * (x * z - w * y)
    M[2, 1] = 2 * (y * z + w * x)
    M[2, 2] = 1 - 2 * (x * x + y * y)
    M[0, 3] = pos[0]
    M[1, 3] = pos[1]
    M[2, 3] = pos[2]
    return M


def generate_pan_obj(filepath: Path, radius=0.12, rim_height=0.025, segments=16):
    """Generate a low-poly pan mesh (disc + rim) and write to OBJ."""
    lines = [
        f"# Generated pan mesh: {2*segments+1} verts, {3*segments} tris",
        "v 0.000000 0.000000 0.000000",
    ]
    for i in range(segments):
        a = 2 * np.pi * i / segments
        lines.append(f"v {radius*np.cos(a):.6f} {radius*np.sin(a):.6f} 0.000000")
    for i in range(segments):
        a = 2 * np.pi * i / segments
        lines.append(f"v {radius*np.cos(a):.6f} {radius*np.sin(a):.6f} {rim_height:.6f}")
    for i in range(segments):
        c, n = i + 2, (i + 1) % segments + 2
        lines.append(f"f 1 {c} {n}")
    for i in range(segments):
        bc, bn = i + 2, (i + 1) % segments + 2
        rc, rn = bc + segments, bn + segments
        lines.append(f"f {bc} {rc} {rn}")
        lines.append(f"f {bc} {rn} {bn}")
    filepath.write_text("\n".join(lines) + "\n")
    print(f"[gen] pan.obj: {2*segments+1} verts, {3*segments} tris")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="UIPC kinematic replay of cooking teleop")
    parser.add_argument("--traj", type=str, default=str(DEFAULT_TRAJ),
                        help="Path to trajectory JSON")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (adjusts dt)")
    parser.add_argument("--broccoli", action="store_true",
                        help="Load broccoli mesh as a fixed object")
    args = parser.parse_args()

    # ---- Load trajectory ----
    with open(args.traj) as f:
        traj_data = json.load(f)
    frames = traj_data["frames"]
    n_frames = len(frames)
    print(f"Loaded {n_frames} frames from {args.traj}")
    if frames:
        t0, t1 = frames[0].get("sim_time", 0), frames[-1].get("sim_time", 0)
        print(f"  sim_time range: {t0:.3f} – {t1:.3f} s")

    # ---- Generate pan.obj if missing ----
    if not PAN_OBJ.exists():
        generate_pan_obj(PAN_OBJ)

    # ---- UIPC engine / scene ----
    Logger.set_level(Logger.Level.Warn)
    Timer.enable_all()

    workspace = AssetDir.output_path(__file__)
    engine = Engine("cuda", workspace)
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = (1.0 / 60.0) / args.speed
    config["gravity"] = [[0.0], [0.0], [-9.8]]
    config["contact"]["enable"] = False
    config["contact"]["friction"]["enable"] = False
    config["newton"]["velocity_tol"] = 1
    config["newton"]["max_iter"] = 8
    config["linear_system"]["tol_rate"] = 1e-4
    scene = Scene(config)
    scene.animator().substep(1)

    abd = AffineBodyConstitution()
    stc = SoftTransformConstraint()

    scene.contact_tabular().default_model(0.0, 1.0 * GPa, False)
    default_element = scene.contact_tabular().default_element()

    io = SimplicialComplexIO()

    # ---- Pan ----
    pan_mesh = io.read(str(PAN_OBJ))
    label_surface(pan_mesh)
    abd.apply_to(pan_mesh, 100.0 * MPa)
    stc.apply_to(pan_mesh, np.array([1.0, 1.0]))
    default_element.apply_to(pan_mesh)
    pan_obj = scene.objects().create("pan")
    pan_obj.geometries().create(pan_mesh)

    # ---- Spatula ----
    spatula_mesh = io.read(str(SPATULA_OBJ))
    label_surface(spatula_mesh)
    abd.apply_to(spatula_mesh, 100.0 * MPa)
    stc.apply_to(spatula_mesh, np.array([1.0, 1.0]))
    default_element.apply_to(spatula_mesh)
    spatula_obj = scene.objects().create("spatula")
    spatula_obj.geometries().create(spatula_mesh)

    # ---- Broccoli (optional, fixed) ----
    broccoli_obj = None
    if args.broccoli and BROCCOLI_OBJ.exists():
        broccoli_mesh = io.read(str(BROCCOLI_OBJ))
        label_surface(broccoli_mesh)
        abd.apply_to(broccoli_mesh, 100.0 * MPa)
        default_element.apply_to(broccoli_mesh)
        is_fixed = broccoli_mesh.instances().find(builtin.is_fixed)
        view(is_fixed)[0] = 1
        broccoli_obj = scene.objects().create("broccoli")
        broccoli_obj.geometries().create(broccoli_mesh)
        print("[scene] broccoli loaded (fixed)")
    elif args.broccoli:
        print(f"[warn] broccoli.obj not found at {BROCCOLI_OBJ}")

    # ---- Ground ----
    ground_obj = scene.objects().create("ground")
    ground_obj.geometries().create(ground(0.0))

    # ---- Animator callbacks ----
    def _make_replay_cb(entity_name: str):
        """Return an animator callback that drives one entity from the trajectory."""

        def _cb(info: Animation.UpdateInfo):
            idx = info.frame()
            if idx >= n_frames:
                return
            frame = frames[idx]
            data = frame.get(entity_name)
            if data is None or "pos" not in data or "quat" not in data:
                return

            geo = info.geo_slots()[0].geometry()
            view(geo.instances().find(builtin.is_constrained))[0] = 1
            mat = quat_pos_to_4x4(data["quat"], data["pos"])
            view(geo.instances().find(builtin.aim_transform))[0] = mat

        return _cb

    animator = scene.animator()
    animator.insert(pan_obj, _make_replay_cb("pan"))
    animator.insert(spatula_obj, _make_replay_cb("spatula"))

    # ---- Init ----
    world.init(scene)

    # ---- GUI ----
    ps.init()
    sgui = SceneGUI(scene, "merge")
    sio = SceneIO(scene)
    sgui.register()

    run = False

    def on_update():
        nonlocal run

        if imgui.Button("Run / Stop"):
            run = not run

        cur = world.frame()
        imgui.Text(f"Frame: {cur} / {n_frames}")
        if cur < n_frames and frames:
            imgui.Text(f"sim_time: {frames[min(cur, n_frames-1)].get('sim_time', 0):.3f} s")

        if run and cur < n_frames:
            world.advance()
            world.retrieve()
            sgui.update()

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()
