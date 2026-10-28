"""Complete Franka grasp with automatic-radius stress recovery through the public options API."""

import argparse
from pathlib import Path

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("volume", "shell"), default="volume")
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=730)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.envs < 1 or args.steps < 1:
        parser.error("Use positive environment and step counts.")

    assets = Path(__file__).parent / "assets" / "hollow_egg" / "level1"
    if args.backend == "shell":
        options = gs.options.RigidStressOptions.dpr(
            mesh=assets / "shell.npz",
            backend="shell",
            shell_thickness=0.0005,
            shell_loading="planar",
            loading_subdivisions=32,
            quadrature=10,
            scatter_tasks_per_env=128,
            box_edge_policy="planar",
            output_mode=args.output_mode,
        )
    else:
        options = gs.options.RigidStressOptions.dpr(
            mesh=assets / "elastic.npz", box_edge_policy="planar", output_mode=args.output_mode
        )

    gs.init(backend=gs.gpu, precision="64")
    grasp = FrankaEgg(n_envs=args.envs, viewer=args.viewer, stress_options=options)
    print(
        f"Stress configuration: {options.backend}, {options.radius_model} radius, "
        f"{options.pressure_profile} pressure, {options.pressure_quadrature}/q{options.quadrature}, "
        f"{options.output_mode} output."
    )
    for _ in range(args.steps):
        grasp.step()
    grasp.scene.rigid_solver.check_errno()
    peak = grasp.link.get_max_stress()
    print("Maximum sampled von Mises stress [Pa], shape:", tuple(peak.shape))
    print("Maximum across environments [Pa]:", peak.max().item())
    if args.output_mode == "full":
        tensor, von_mises = grasp.link.get_stress_field()
        print("Stress tensor / von Mises [Pa] shapes:", tuple(tensor.shape), tuple(von_mises.shape))
        if args.backend == "shell":
            membrane, bending, shear = grasp.link.get_shell_resultants()
            print("N [N/m], M [N], Q [N/m] shapes:", tuple(membrane.shape), tuple(bending.shape), tuple(shear.shape))


if __name__ == "__main__":
    main()
