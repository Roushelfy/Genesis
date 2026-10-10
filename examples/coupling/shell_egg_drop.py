import argparse
import os

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU")
    parser.add_argument("-b", "--num-envs", type=int, default=16, help="Number of parallel environments, one drop each")
    parser.add_argument("-t", "--seconds", type=float, default=0.5, help="Duration of the drops, in s")
    parser.add_argument("--substeps", type=int, default=10, help="Number of substeps per control step")
    parser.add_argument("--precision", type=str, default="32", help="Floating-point precision, 32 or 64")
    parser.add_argument("--pcg-tolerance", type=float, default=1e-4, help="Relative tolerance of the linear solves")
    parser.add_argument(
        "--pcg-velocity-tolerance", type=float, default=1e-4, help="Velocity tolerance of the linear solves, in m/s"
    )
    parser.add_argument("--pcg-max-iterations", type=int, default=1000, help="Iteration limit of every linear solve")
    parser.add_argument("--contact-stiffness", type=float, default=25.0, help="Stiffness ratio of the ground contacts")
    args = parser.parse_args()
    if "PYTEST_VERSION" in os.environ:
        args.seconds = 0.02

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision=args.precision, logging_level="warning")

    # A low-resolution hollow egg: an icosphere whose cross-section tapers from the blunt to the pointed end
    EGG_LENGTH, EGG_WIDTH, GRAVITY, HEIGHT = 0.06, 0.045, 9.81, 0.08
    sphere = trimesh.creation.icosphere(subdivisions=2)
    egg_verts = sphere.vertices * 0.5 * np.array([EGG_LENGTH, EGG_WIDTH, EGG_WIDTH])
    egg_verts[:, 1:] *= 1.0 - 0.12 * sphere.vertices[:, :1]
    os.makedirs("out", exist_ok=True)
    egg_path = os.path.join("out", "shell_egg_drop_egg.obj")
    trimesh.Trimesh(egg_verts, sphere.faces, process=False).export(egg_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=args.substeps,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        shell_options=gs.options.ShellOptions(
            pcg_tolerance=args.pcg_tolerance,
            pcg_velocity_tolerance=args.pcg_velocity_tolerance,
            pcg_max_iterations=args.pcg_max_iterations,
            contact_stiffness=args.contact_stiffness,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    egg = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=egg_path,
        ),
        material=gs.materials.Shell(
            rho=2e4,
            E=1e10,
            nu=0.3,
            thickness=4e-4,
            damping=1e-4,
            tensile_strength=1e7,
            fracture=False,
        ),
    )
    n_envs = args.num_envs
    scene.build(n_envs=n_envs)

    # Every environment drops the egg at its own orientation, its lowest point from a height growing from a quarter of
    # HEIGHT to HEIGHT, so that the impact speeds span a factor of two
    rng = np.random.default_rng(1)
    axes = rng.normal(size=(n_envs, 3))
    angles = rng.uniform(0.0, np.pi, size=n_envs)
    verts = egg.init_verts @ gu.axis_angle_to_R(axes, angles).transpose(0, 2, 1)
    heights = HEIGHT * np.linspace(0.25, 1.0, n_envs)
    verts[..., 2] += (heights - verts[..., 2].min(axis=-1))[:, None]
    egg.set_verts_pos(verts)

    contact_force_max = np.zeros(n_envs)
    for _ in range(int(args.seconds / scene.dt)):
        scene.step()
        contact_force = np.linalg.norm(tensor_to_array(egg.get_verts_contact_force()).sum(axis=-2), axis=-1)
        contact_force_max = np.maximum(contact_force_max, contact_force)

    print("impact speed (m/s)", np.array2string(np.sqrt(2.0 * GRAVITY * heights), precision=3))
    print("largest contact force on the egg (N)", np.array2string(contact_force_max, precision=1))
    print("peak damage index", np.array2string(tensor_to_array(egg.get_peak_damage()), precision=5))
    print("linear solver failures", tensor_to_array(scene.sim.shell_solver.get_envs_solver_failure()))


if __name__ == "__main__":
    main()
