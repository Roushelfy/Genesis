"""
Bike chain scene (rigid-ipc #227) with Franka robot driving the active sprocket.

Fixed cylinder axles through sprocket holes prevent translation. A Franka Panda
grips the active sprocket and rotates it via wrist control.

Usage:
    LD_PRELOAD=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12 \
        python examples/debug/ipc_bike_chain.py [--no-viewer] [--use-motor]
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu

import uipc
from uipc.constitution import RotatingMotor
from uipc.core import Animation

from load_rigid_ipc_scene import (
    MESH_ROOT,
    load_rigid_ipc_scene,
    print_scene_info,
    yup_to_zup_position,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BIKE_CHAIN_JSON = REPO_ROOT / "DemoAssets" / "track_bike" / "rigid_ipc_src" / "227-bike-chain.json"

# Cylinder height (along Y-axis in Z-up, the rotation axis)
CYLINDER_HEIGHT = 0.1


def find_sprockets(entities, bodies):
    """Find bodies with partial is_dof_fixed (sprockets).

    Returns list of dicts with center, omega, axis, hole_radius, etc.
    """
    sprockets = []
    for entity, body in zip(entities, bodies):
        is_dof_fixed = body.get("is_dof_fixed", False)
        if not isinstance(is_dof_fixed, list):
            continue

        # Compute sprocket center from mesh centroid
        mesh_path = str(MESH_ROOT / body["mesh"])
        scale = body.get("scale", 1.0)
        mesh = trimesh.load(mesh_path)
        center_ipc = mesh.centroid * scale
        center_zup = np.array(yup_to_zup_position(center_ipc), dtype=np.float32)

        # Estimate hole radius from vertices closest to the centroid (inner ring)
        # In Y-up, the rotation axis is Y; project onto XZ plane to find radial distance
        verts = mesh.vertices * scale
        centroid = mesh.centroid * scale
        dxz = verts[:, [0, 2]] - centroid[[0, 2]]
        radial_dists = np.linalg.norm(dxz, axis=1)
        # The hole's inner ring = the smallest radial distances from center
        hole_radius = float(np.min(radial_dists))

        # Angular velocity (Y-up deg/s -> Z-up rad/s)
        ang_vel_yup = np.radians(body.get("angular_velocity", [0, 0, 0]))
        ang_vel_zup = np.array([ang_vel_yup[0], -ang_vel_yup[2], ang_vel_yup[1]])
        omega = float(np.linalg.norm(ang_vel_zup))

        if omega > 1e-10:
            axis = (ang_vel_zup / omega).reshape(3, 1)
        else:
            # Default: Y-axis in Z-up (rotation axis for this scene)
            axis = np.array([[0], [1], [0]], dtype=np.float32)

        sprockets.append(
            {
                "entity": entity,
                "body": body,
                "center": center_zup,
                "hole_radius": hole_radius,
                "omega": omega,
                "axis": axis,
                "is_kinematic": body.get("type", "dynamic") == "kinematic",
            }
        )
        print(
            f"  sprocket entity {entity.idx}: center={center_zup}, "
            f"hole_r={hole_radius:.4f}, omega={omega:.4f} rad/s, "
            f"kinematic={sprockets[-1]['is_kinematic']}"
        )

    return sprockets


def add_cylinder_axles(scene, sprockets):
    """Add fixed cylinder entities through each sprocket hole.

    The cylinders are proper Genesis entities added via scene.add_entity(),
    so they go through the full Genesis pipeline and get IPC coupling.
    """
    cylinders = []
    radii = [0.05, 0.025]
    for i, info in enumerate(sprockets):
        c = info["center"]
        cylinder = scene.add_entity(
            gs.morphs.Cylinder(
                pos=(float(c[0]), float(c[1]), float(c[2])),
                # Default cylinder axis is Z; rotate 90 deg around X to align with Y
                euler=(90, 0, 0),
                height=CYLINDER_HEIGHT,
                radius=radii[i],
                fixed=True,
            ),
            material=gs.materials.Rigid(coup_type="ipc_only"),
            surface=gs.surfaces.Plastic(color=(0.0, 0.0, 0.0)),
        )
        cylinders.append(cylinder)
        print(f"  Added fixed cylinder at {c}, radius={radii[i]:.4f}, height={CYLINDER_HEIGHT}")
    return cylinders


def add_rotating_motor(coupler, sprockets):
    """Add RotatingMotor to kinematic sprockets via IPC pre-finalize hook.

    RotatingMotor is a native IPC constitution with no Genesis entity equivalent,
    so it must be injected via the coupler's pre-finalize hook.
    """
    motor = RotatingMotor()
    coupler._ipc_constitution_tabular.insert(motor)

    for info in sprockets:
        if not info["is_kinematic"] or info["omega"] < 1e-10:
            continue
        entity = info["entity"]
        link = entity.links[0]
        abd_data = coupler._abd_data_by_link.get(link)
        if abd_data is None:
            continue

        for slot in abd_data.slots:
            geom = slot.geometry()
            motor.apply_to(
                geom,
                strength=100.0,
                motor_axis=info["axis"],
                motor_rot_vel=-info["omega"],
            )

        # Register animator to advance the motor each frame
        for env_idx in range(coupler._B):
            obj_name = f"rigid_link_{link.idx}_{env_idx}"
            abd_objs = coupler._ipc_objects.find(obj_name)
            if not abd_objs:
                continue

            def _make_anim_cb():
                def _anim(anim_info: Animation.UpdateInfo):
                    geo = anim_info.geo_slots()[0].geometry()
                    uipc.view(geo.instances().find(uipc.builtin.is_constrained))[0] = 1
                    RotatingMotor.animate(geo, anim_info.dt())

                return _anim

            coupler._ipc_animator.insert(abd_objs[0], _make_anim_cb())
        print(f"  Applied RotatingMotor to entity {entity.idx}")


def add_franka(scene, sprockets):
    """Add a Franka Panda robot positioned to grip the active sprocket."""
    # Find the active (kinematic) sprocket
    active = None
    for info in sprockets:
        if info["is_kinematic"]:
            active = info
            break
    if active is None:
        active = sprockets[0]

    gear_center = active["center"]
    print(f"  Active sprocket center: {gear_center}")

    # Position the Franka so its end effector can reach the gear.
    # Gear is at ~[0, 0, 0]; place the robot base behind it along +X.
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
            pos=(0.55, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(
            coup_type="external_articulation",
            coup_friction=0.8,
        ),
    )
    return franka, gear_center


def main():
    parser = argparse.ArgumentParser(description="Bike chain with Franka robot driving sprocket")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--use-motor", action="store_true", help="Use RotatingMotor instead of Franka robot")
    parser.add_argument(
        "--two-way", action="store_true", help="Use two_way_soft_constraint for sprockets (instead of ipc_only)"
    )
    args = parser.parse_args()

    # Sprocket body indices in the JSON fixture (last two bodies)
    two_way_indices = {136, 137} if args.two_way else None

    print(f"Loading: {BIKE_CHAIN_JSON}")
    scene, entities, bodies, data = load_rigid_ipc_scene(
        str(BIKE_CHAIN_JSON),
        show_viewer=not args.no_viewer,
        ipc=True,
        two_way_body_indices=two_way_indices,
    )

    # Find sprockets
    sprockets = find_sprockets(entities, bodies)

    # Add fixed cylinder axles as proper Genesis entities
    if sprockets:
        add_cylinder_axles(scene, sprockets)

    if args.use_motor:
        if args.two_way:
            # Two-way: drive sprockets from Genesis side with velocity control
            scene.build()
            print_scene_info(scene)

            for info in sprockets:
                if not info["is_kinematic"] or info["omega"] < 1e-10:
                    continue
                entity = info["entity"]
                # Single-link mesh entity with a free joint (6 DOFs: tx,ty,tz,rx,ry,rz)
                # Rotation axis is Y in Y-up → Z in Z-up → DOF index 5 (rz)
                motor_dof = [5]
                entity.set_dofs_kp(0.0, dofs_idx_local=motor_dof)
                entity.set_dofs_kv(10.0, dofs_idx_local=motor_dof)
                entity.control_dofs_velocity(info["omega"], dofs_idx_local=motor_dof)
                print(f"  Two-way velocity control on entity {entity.idx}: omega={info['omega']:.4f} on DOF 5 (rz)")

            print(f"\nSimulation (two-way motor mode): dt={data.get('timestep', 0.01)}")
            while True:
                scene.step()
        else:
            # RotatingMotor via IPC pre-finalize hook
            from genesis.engine.couplers.ipc_coupler import IPCCoupler

            coupler = scene.sim.coupler
            if isinstance(coupler, IPCCoupler):
                coupler._pre_finalize_hooks.append(lambda c: add_rotating_motor(c, sprockets))
            scene.build()
            print_scene_info(scene)

            print(f"\nSimulation (motor mode): dt={data.get('timestep', 0.01)}")
            while True:
                scene.step()
    else:
        # Franka robot grips and rotates the gear
        franka, gear_center = add_franka(scene, sprockets)
        scene.build()
        print_scene_info(scene)

        # DOF indices
        motor_dofs = np.arange(7)
        finger_dofs = np.arange(7, 9)
        ee_link = franka.get_link("hand")

        # Hand orientation: palm facing down toward gear
        ee_quat = gu.xyz_to_quat(np.array([0.0, 180.0, 0.0], dtype=gs.np_float), degrees=True)

        # Phase 1: Move to pre-grasp position above the gear
        pre_grasp_pos = np.array(
            [
                float(gear_center[0]),
                float(gear_center[1]),
                float(gear_center[2]) + 0.15,
            ],
            dtype=gs.np_float,
        )
        qpos = franka.inverse_kinematics(
            link=ee_link,
            pos=pre_grasp_pos,
            quat=ee_quat,
            dofs_idx_local=motor_dofs,
        )
        franka.set_qpos(qpos)
        franka.control_dofs_position(qpos[motor_dofs], motor_dofs)
        # Open fingers
        franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs)

        franka.set_dofs_kp(500.0, dofs_idx_local=finger_dofs)
        franka.set_dofs_kv(50.0, dofs_idx_local=finger_dofs)

        print("\n=== Phase 1: Settle at pre-grasp ===")
        for _ in range(100):
            scene.step()

        # Phase 2: Lower to grasp height
        grasp_pos = np.array(
            [
                float(gear_center[0]),
                float(gear_center[1]),
                float(gear_center[2]) + 0.06,
            ],
            dtype=gs.np_float,
        )

        print("=== Phase 2: Lower to grasp ===")
        n_lower = 200
        for i in range(n_lower):
            t = (i + 1) / n_lower
            pos = pre_grasp_pos * (1 - t) + grasp_pos * t
            qpos = franka.inverse_kinematics(
                link=ee_link,
                pos=pos,
                quat=ee_quat,
                dofs_idx_local=motor_dofs,
            )
            franka.control_dofs_position(qpos[motor_dofs], motor_dofs)
            franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs)
            scene.step()

        # Phase 3: Close fingers to grip the gear
        print("=== Phase 3: Close fingers ===")
        for _ in range(200):
            franka.control_dofs_position(0.0, dofs_idx_local=finger_dofs)
            scene.step()

        # Phase 4: Rotate the wrist (joint7) to spin the gear
        print("=== Phase 4: Rotate wrist to drive gear ===")
        wrist_dof = 6
        current_qpos = franka.get_qpos().cpu().numpy().flatten()

        while True:
            # Velocity control on wrist, position hold on other joints
            franka.control_dofs_position(current_qpos[:6], np.arange(6))
            franka.control_dofs_velocity(np.array([1.0]), np.array([wrist_dof]))
            franka.control_dofs_position(0.0, dofs_idx_local=finger_dofs)
            scene.step()


if __name__ == "__main__":
    main()
