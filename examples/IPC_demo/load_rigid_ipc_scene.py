"""
Load rigid-ipc JSON fixture files into Genesis.

Rigid-IPC format:
  - Y-up coordinate system (gravity = [0, -9.81, 0])
  - Rotation: [rx, ry, rz] in degrees, applied as Rz * Ry * Rx (intrinsic ZYX)
  - Mesh paths relative to meshes/ directory
  - Body types: "static", "kinematic", "dynamic" (default)
  - is_dof_fixed: bool or [tx, ty, tz, rx, ry, rz]

Genesis uses Z-up, so we apply a Y-up to Z-up transform:
  x_gs =  x_ipc
  y_gs = -z_ipc
  z_gs =  y_ipc
"""

import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ASSETS_ROOT = REPO_ROOT / "DemoAssets"
RIGID_IPC_ROOT = DEMO_ASSETS_ROOT / "track_bike" / "rigid_ipc_src"
MESH_ROOT = RIGID_IPC_ROOT


def euler_xyz_deg_to_quat(rx_deg, ry_deg, rz_deg):
    """
    Convert rigid-ipc Euler angles (degrees, intrinsic ZYX / extrinsic XYZ)
    to a quaternion [w, x, y, z].
    """
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)

    # Quaternion for Rz * Ry * Rx
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return np.array([w, x, y, z])


def yup_to_zup_position(pos):
    """Convert Y-up position [x, y, z] to Z-up [x, -z, y]."""
    return (pos[0], -pos[2], pos[1])


def yup_to_zup_quat(q_wxyz):
    """
    Convert a Y-up quaternion to Z-up by pre-multiplying with the
    90-degree rotation around X that maps Y-up to Z-up.

    q_coord = quat(90 deg around X) = [cos(45), sin(45), 0, 0]
              = [sqrt(2)/2, sqrt(2)/2, 0, 0]
    """
    c = np.sqrt(2) / 2
    q_coord = np.array([c, c, 0, 0])
    return quat_mul(q_coord, q_wxyz)


def quat_mul(a, b):
    """Multiply two quaternions [w, x, y, z]."""
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return np.array([w, x, y, z])


def load_rigid_ipc_scene(
    json_path,
    show_viewer=True,
    no_gravity=False,
    vis_mode="visual",
    coacd_threshold=0.1,
    ipc=False,
    two_way_body_indices=None,
):
    """Load a rigid-ipc JSON fixture into a Genesis scene.

    Returns (scene, entities, bodies, data) without building or running.
    The caller should call scene.build() and scene.step() as needed.
    """
    with open(json_path) as f:
        data = json.load(f)

    rb_problem = data["rigid_body_problem"]
    dt = data.get("timestep", 0.01)
    json_bodies = rb_problem["rigid_bodies"]

    gs.init(backend=gs.gpu)

    # Camera: rigid-ipc default is position=[0,0,3] gaze=[0,0,-1] in Y-up
    # (top-down view of XZ ground plane).
    # Y-up→Z-up: [x,y,z] → (x, -z, y), so camera at (0, -3, 0) looking at origin.
    # Small Z offset avoids polyscope lookAt degeneracy with Z-up vector.
    scene_kwargs = dict(
        show_viewer=show_viewer,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -2, 0.06),
            camera_lookat=(0, 0, 0),
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            # IPC handles contacts; disable redundant rigid solver collision
            enable_collision=not ipc,
            gravity=(0, 0, 0) if no_gravity else (0, 0, -9.81),
        ),
    )
    if ipc:
        ipc_opts = {}
        # libuipc sample overrides JSON d_hat (1e-5) to 2e-4 for performance
        ipc_opts["contact_d_hat"] = 2e-4
        ipc_opts["newton_semi_implicit_enable"] = False
        ipc_opts["linear_system_tolerance"] = 1e-4
        scene_kwargs["coupler_options"] = gs.options.IPCCouplerOptions(**ipc_opts)

    scene = gs.Scene(**scene_kwargs)

    entities = []
    bodies = []
    for i, body in enumerate(json_bodies):
        mesh_path = str(MESH_ROOT / body["mesh"])
        scale = body.get("scale", 1.0)
        pos_ipc = body.get("position", [0, 0, 0])
        rot_deg = body.get("rotation", [0, 0, 0])
        body_type = body.get("type", "dynamic")

        # Convert rotation
        q_yup = euler_xyz_deg_to_quat(*rot_deg)
        q_zup = yup_to_zup_quat(q_yup)
        pos_zup = yup_to_zup_position(pos_ipc)

        # Determine if fixed
        is_dof_fixed = body.get("is_dof_fixed", False)
        if isinstance(is_dof_fixed, bool):
            fixed = is_dof_fixed or body_type == "static"
        elif isinstance(is_dof_fixed, list):
            fixed = all(is_dof_fixed)
        else:
            fixed = body_type == "static"

        morph_kwargs = dict(
            file=mesh_path,
            pos=pos_zup,
            quat=q_zup,
            scale=scale,
            fixed=fixed,
        )
        if ipc:
            # IPC uses raw triangle meshes; skip convex decomposition
            morph_kwargs["convexify"] = False
        else:
            morph_kwargs["decompose_object_error_threshold"] = 0.0
            morph_kwargs["coacd_options"] = gs.options.CoacdOptions(threshold=coacd_threshold)

        morph = gs.morphs.Mesh(**morph_kwargs)
        entity_kwargs = dict(vis_mode=vis_mode)
        if ipc and not fixed:
            if two_way_body_indices is not None and i in two_way_body_indices:
                entity_kwargs["material"] = gs.materials.Rigid(coup_type="two_way_soft_constraint")
            else:
                # All non-fixed bodies use ipc_only: IPC handles dynamics (gravity + contact).
                entity_kwargs["material"] = gs.materials.Rigid(coup_type="ipc_only")
        entity = scene.add_entity(morph, **entity_kwargs)
        entities.append(entity)
        bodies.append(body)

        print(f"  [{i}] {body['mesh']} type={body_type} fixed={fixed} pos={pos_zup}")

    return scene, entities, bodies, data


def print_scene_info(scene):
    """Print summary of a built scene."""
    rigid_solver = scene.sim.rigid_solver
    entities = rigid_solver.entities
    n_fixed = sum(1 for e in entities if getattr(e.morph, "fixed", False))
    print("\nScene info:")
    print(f"  entities: {len(entities)} ({n_fixed} fixed, {len(entities) - n_fixed} dynamic)")
    print(f"  links:  {len(rigid_solver.links)}")
    print(f"  geoms:  {rigid_solver.n_geoms}")
    print(f"  vgeoms: {rigid_solver.n_vgeoms}")
    print(f"  verts:  {rigid_solver.n_verts}")
    if rigid_solver.collider is not None:
        print(f"  collision-eligible pairs: {rigid_solver.collider._n_possible_pairs}")


def main():
    parser = argparse.ArgumentParser(description="Load rigid-ipc JSON fixtures into Genesis")
    parser.add_argument("json_file", help="Path to rigid-ipc JSON fixture file")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-gravity", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--vis", default="visual", choices=["visual", "collision"])
    parser.add_argument(
        "--coacd-threshold",
        type=float,
        default=0.1,
        help="CoACD decomposition threshold (lower = tighter collision mesh, default 0.1)",
    )
    parser.add_argument("--ipc", action="store_true", help="Use IPCCoupler (convexify=False, reads d_hat from JSON)")
    args = parser.parse_args()

    print(f"Loading: {args.json_file}")
    scene, entities, bodies, data = load_rigid_ipc_scene(
        args.json_file,
        show_viewer=not args.no_viewer,
        no_gravity=args.no_gravity,
        vis_mode=args.vis,
        coacd_threshold=args.coacd_threshold,
        ipc=args.ipc,
    )

    scene.build()
    print_scene_info(scene)

    dt = data.get("timestep", 0.01)
    max_time = data.get("max_time", 5.0)
    n_steps = int(max_time / dt)
    print(f"\nSimulation: dt={dt}, max_time={max_time}, n_steps={n_steps}")
    if not args.no_sim:
        while True:
            scene.step()
    else:
        while True:
            scene._visualizer.update()


if __name__ == "__main__":
    main()
