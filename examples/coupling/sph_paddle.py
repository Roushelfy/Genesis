import argparse
import os

import numpy as np
import torch

import genesis as gs

# A paddle on a hinge stirs viscous water in a tank, driven by a velocity controller. The fluid and the paddle are
# two-way coupled: the implicit viscosity and the pressure of the fluid act on the paddle within each substep, and the
# torque of the controller balances the fluid torque that SPHSolver.get_links_fluid_wrench reads out.

PADDLE_MJCF = """
<mujoco>
    <worldbody>
        <body name="arm" pos="0 0 0.12">
            <joint name="hinge" type="hinge" axis="0 0 1" armature="0.001"/>
            <geom type="box" pos="0.035 0 -0.06" size="0.015 0.004 0.025" density="1000"/>
        </body>
    </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="info")

    half, depth, particle_size = 0.06, 0.09, 0.005
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            particle_volume_factor=1.0,
            lower_bound=(-half - 0.02, -half - 0.02, -0.02),
            upper_bound=(half + 0.02, half + 0.02, depth + 0.06),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.3, 0.3),
            camera_lookat=(0.0, 0.0, 0.05),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )
    # tank: floor and four walls
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, -0.005), size=(2 * half + 0.02, 2 * half + 0.02, 0.01), fixed=True),
    )
    wall = half + 0.005
    for x, y, sx, sy in ((wall, 0.0, 0.01, 2 * half), (-wall, 0.0, 0.01, 2 * half)):
        scene.add_entity(morph=gs.morphs.Box(pos=(x, y, 0.07), size=(sx, sy, 0.14), fixed=True))
    for x, y, sx, sy in ((0.0, wall, 2 * half, 0.01), (0.0, -wall, 2 * half, 0.01)):
        scene.add_entity(morph=gs.morphs.Box(pos=(x, y, 0.07), size=(sx, sy, 0.14), fixed=True))
    paddle = scene.add_entity(
        morph=gs.morphs.MJCF(file=PADDLE_MJCF, default_armature=0.0),
    )
    water = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, depth / 2), size=(2 * half - particle_size, 2 * half - particle_size, depth)
        ),
        # kinematic viscosity in m^2/s, a thousand times that of water
        material=gs.materials.SPH.Liquid(sampler="regular", viscosity=1e-3, surface_tension=0.0),
        surface=gs.surfaces.Default(color=(0.5, 0.7, 0.9, 1.0)),
    )
    scene.build()

    # remove the water that the paddle occupies
    pos = water.get_particles_pos()
    center = torch.tensor((0.035, 0.0, 0.06), device=gs.device)
    paddle_half = torch.tensor((0.015, 0.004, 0.025), device=gs.device) + 0.5 * particle_size
    water.set_particles_active(~((pos - center).abs() < paddle_half).all(dim=1))

    sph = scene.sph_solver
    arm = paddle.get_link("arm")
    hinge = np.array((0.0, 0.0, 0.12))
    paddle.set_dofs_kv((1.0,))

    horizon = 500 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        omega = 3.0 * min(i * scene.dt / 0.2, 1.0)
        paddle.control_dofs_velocity((omega,))
        scene.step()
        if i % 50 == 49 or i == horizon - 1:
            # fluid torque about the hinge axis, from the fluid force and torque on the paddle about its center of mass
            force, torque = sph.get_links_fluid_wrench()
            com = paddle.get_links_pos(ref=gs.link_ref_frame.link_COM)[arm.idx_local].cpu().numpy()
            fluid_torque = torque[arm.idx, 2].item() + np.cross(com - hinge, force[arm.idx].cpu().numpy())[2]
            stats = sph.solve_stats
            print(
                f"t {(i + 1) * scene.dt:.2f} s: paddle speed {paddle.get_dofs_velocity()[0].item():.3f} rad/s, "
                f"motor torque {paddle.get_dofs_control_force()[0].item():.3e} N m, "
                f"fluid torque {fluid_torque:.3e} N m, density iterations {stats.n_density_iters}, "
                f"coupling passes {stats.n_coupling_passes}, viscous dissipation {sph.get_viscous_dissipation().item():.3e} J"
            )


if __name__ == "__main__":
    main()
