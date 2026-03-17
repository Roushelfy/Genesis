"""
Standalone libuipc test: load the 227-bike-chain fixture WITHOUT Genesis.

This isolates whether the CUDA misaligned address crash is a libuipc issue
or a Genesis integration issue.

Usage:
    python examples/IPC_demo/test_libuipc_bike_chain.py [--frames N] [--no-gui]
"""

import argparse
import json
import time

import numpy as np

import uipc
import uipc.builtin as builtin
from uipc import AngleAxis, Quaternion, Timer, Transform, Vector3, view
from uipc.constitution import AffineBodyConstitution, RotatingMotor
from uipc.core import Animation, Engine, Scene, World
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
)
from uipc.unit import GPa, MPa

RIGID_IPC_ROOT = "/home/zhehuan/Desktop/hz/rigid-ipc"
MESH_ROOT = f"{RIGID_IPC_ROOT}/meshes"
BIKE_CHAIN_JSON = f"{RIGID_IPC_ROOT}/fixtures/3D/mechanisms/507-movements/227-bike-chain.json"


def process_surface(sc):
    """Prepare mesh for contact. For surface-only meshes (.obj), skip
    label_triangle_orient which requires tetmesh."""
    label_surface(sc)
    # label_triangle_orient only works on tetmesh (3D simplicial complex);
    # .obj files are surface-only, so skip it and flip_inward_triangles.
    return sc


def euler_xyz_to_quat(rx_deg, ry_deg, rz_deg):
    """Rigid-IPC Euler angles (degrees, intrinsic ZYX) -> uipc Quaternion."""
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)
    q = AngleAxis(rz, Vector3.UnitZ()) * AngleAxis(ry, Vector3.UnitY()) * AngleAxis(rx, Vector3.UnitX())
    return q


def main():
    parser = argparse.ArgumentParser(description="Standalone libuipc bike chain test")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    with open(BIKE_CHAIN_JSON) as f:
        data = json.load(f)

    rb_problem = data["rigid_body_problem"]
    dt = data.get("timestep", 0.01)
    json_bodies = rb_problem["rigid_bodies"]

    print(f"Loading {len(json_bodies)} bodies from {BIKE_CHAIN_JSON}")
    print(f"dt = {dt}")

    # -- Engine, World, Scene --
    Timer.enable_all()
    engine = Engine("cuda", "/tmp/libuipc_bike_chain_test")
    world = World(engine)

    config = Scene.default_config()
    config["dt"] = dt
    # Y-up (matching rigid-ipc convention)
    config["gravity"] = [[0.0], [-9.81], [0.0]]
    config["contact"]["d_hat"] = 2e-4
    config["contact"]["friction"]["enable"] = True
    config["newton"]["semi_implicit"] = {"enable": True}
    config["linear_system"]["tol_rate"] = 1e-4
    scene = Scene(config)

    # Contact model
    scene.contact_tabular().default_model(0.5, 1.0 * GPa)
    default_contact = scene.contact_tabular().default_element()

    # Constitutions
    abd = AffineBodyConstitution()
    motor = RotatingMotor()

    io = SimplicialComplexIO()

    # Cache loaded & scaled meshes by (path, scale)
    mesh_cache = {}

    def load_mesh(mesh_rel, scale):
        cache_key = (mesh_rel, scale)
        if cache_key not in mesh_cache:
            mesh_path = f"{MESH_ROOT}/{mesh_rel}"
            mesh = io.read(mesh_path)
            # Apply scale directly to vertex positions (like Genesis does)
            if scale != 1.0:
                pos = view(mesh.positions())
                pos[:] = pos * scale
            mesh = process_surface(mesh)
            abd.apply_to(mesh, 100 * MPa)
            default_contact.apply_to(mesh)
            mesh_cache[cache_key] = mesh
        return mesh_cache[cache_key]

    # Track which objects are kinematic sprockets (for RotatingMotor)
    kinematic_sprockets = []

    # Create objects per body
    for i, body in enumerate(json_bodies):
        mesh_rel = body["mesh"]
        scale = body.get("scale", 1.0)
        base_mesh = load_mesh(mesh_rel, scale)
        this_mesh = base_mesh.copy()

        pos = body.get("position", [0, 0, 0])
        rot_deg = body.get("rotation", [0, 0, 0])
        body_type = body.get("type", "dynamic")
        is_dof_fixed = body.get("is_dof_fixed", False)

        # Build transform (Y-up, same as rigid-ipc)
        # Scale already applied to vertex positions
        t = Transform.Identity()
        position = Vector3.Zero()
        position[0] = pos[0]
        position[1] = pos[1]
        position[2] = pos[2]
        t.translate(position)

        if rot_deg != [0, 0, 0]:
            q = euler_xyz_to_quat(*rot_deg)
            t.rotate(q)

        view(this_mesh.transforms())[0] = t.matrix()

        # Handle fixed
        if isinstance(is_dof_fixed, bool):
            is_fixed_val = 1 if (is_dof_fixed or body_type == "static") else 0
        elif isinstance(is_dof_fixed, list):
            is_fixed_val = 1 if all(is_dof_fixed) else 0
        else:
            is_fixed_val = 1 if body_type == "static" else 0

        is_fixed_attr = this_mesh.instances().find("is_fixed")
        view(is_fixed_attr)[0] = is_fixed_val

        obj_name = f"body_{i}"
        obj = scene.objects().create(obj_name)

        # Apply RotatingMotor to kinematic sprocket
        ang_vel = body.get("angular_velocity", None)
        if body_type == "kinematic" and ang_vel is not None:
            omega_deg = ang_vel
            # Y-up: rotation around Z axis
            omega_rad = np.radians(abs(omega_deg[2]))
            if omega_rad > 1e-10:
                motor.apply_to(
                    this_mesh,
                    100.0,
                    Vector3.UnitZ(),
                    -omega_rad,
                )
                kinematic_sprockets.append((obj, obj_name))

        obj.geometries().create(this_mesh)

        if i % 20 == 0 or i == len(json_bodies) - 1:
            print(f"  [{i}] {mesh_rel} type={body_type} fixed={is_fixed_val}")

    print(f"\nKinematic sprockets with RotatingMotor: {[name for _, name in kinematic_sprockets]}")

    # Animator for kinematic sprockets
    animator = scene.animator()
    for obj, name in kinematic_sprockets:

        def _make_anim(obj_ref):
            def _anim(info: Animation.UpdateInfo):
                geo = info.geo_slots()[0].geometry()
                is_constrained = geo.instances().find(builtin.is_constrained)
                view(is_constrained)[0] = 1
                RotatingMotor.animate(geo, info.dt())

            return _anim

        animator.insert(obj, _make_anim(obj))
        print(f"  Animator registered for {name}")

    # Init
    print("\nInitializing world...")
    world.init(scene)
    print("World initialized successfully")

    if not args.no_gui:
        try:
            import polyscope as ps
            from polyscope import imgui
            from uipc.gui import SceneGUI

            sgui = SceneGUI(scene)
            ps.init()
            tri_surf, _, _ = sgui.register()
            tri_surf.set_edge_width(1)

            run = False
            frame = [0]

            def on_update():
                nonlocal run
                if imgui.Button("run & stop"):
                    run = not run

                if run and frame[0] < args.frames:
                    t0 = time.perf_counter()
                    world.advance()
                    world.retrieve()
                    dt_ms = (time.perf_counter() - t0) * 1000
                    frame[0] += 1
                    print(f"Frame {frame[0]:4d}: {dt_ms:.0f}ms")
                    sgui.update()

            ps.set_user_callback(on_update)
            ps.show()
            return
        except Exception as e:
            print(f"GUI unavailable ({e}), running headless")

    # Headless mode
    print(f"\nRunning {args.frames} frames headless...")
    for i in range(args.frames):
        t0 = time.perf_counter()
        world.advance()
        world.retrieve()
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"Frame {i + 1:4d}: {dt_ms:.0f}ms")

    print("Done!")


if __name__ == "__main__":
    main()
