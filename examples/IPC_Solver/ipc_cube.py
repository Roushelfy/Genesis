import genesis as gs
import logging
import argparse

import numpy as np


def main():
    gs.init(backend=gs.gpu, logging_level=logging.DEBUG, performance_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    dt = 1e-3

    coupler_options = (
        gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(100, 100),  # (translation, rotation) strength ratios,
            disable_ipc_ground_contact=False,
            disable_ipc_logging=False,
            IPC_self_contact=True,
            contact_friction_mu=0.8,
            enable_ipc_gui=args.vis_ipc,
            newton_transrate_tol=0.1,
            sync_dof_enable=False,
            # contact_constitution="al-ipc",
        )
        if args.ipc
        else None
    )
    args.vis = args.vis or args.vis_ipc
    rigid_options = gs.options.RigidOptions(
        enable_collision=False,  # Disable rigid collision when using IPC
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        rigid_options=rigid_options,
        coupler_options=coupler_options,
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    material = (
        gs.materials.FEM.Elastic(E=5.0e3, nu=0.45, rho=1000.0, model="stable_neohookean")
        if args.ipc
        else gs.materials.Rigid()
    )
    material = gs.materials.Rigid()
    cube1 = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.65, 0.0, 0.1), size=(0.05, 0.05, 0.05)),
        material=material,
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )
    cube2 = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.65, 0.0, 0.3), size=(0.05, 0.05, 0.05)),
        material=material,
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=cube1,
        coupling_type="ipc_only",
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=cube2,
        coupling_type="ipc_only",
    )
    scene.build()
    print("Scene built successfully!")

    for i in range(int(10 / dt)):

        scene.step()


if __name__ == "__main__":
    main()
