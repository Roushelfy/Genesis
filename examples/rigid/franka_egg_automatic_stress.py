"""Grasp an egg or custom mesh, reporting maximum stress or showing the recovered field in the viewer."""

import argparse
import math
from pathlib import Path

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.rigid.stress_field_view import StressFieldSurface
from examples.rigid.stress_mesh_asset import prepare_stress_mesh, write_rigid_asset
from genesis.utils.geom import trans_quat_to_T
from genesis.utils.misc import get_cache_dir, tensor_to_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("volume", "shell"), default="volume")
    parser.add_argument(
        "--thickness", type=float, default=0.0005, help="Shell thickness in metres, independent of --scale."
    )
    parser.add_argument("--mesh", type=Path, help="Surface file or elastic NPZ; shell surfaces are midsurfaces.")
    parser.add_argument("--stress-mesh", type=Path, help="Matching volume or Q4 shell NPZ in the same authored frame.")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Convert custom surface and elastic coordinates to metres."
    )
    parser.add_argument("--max-volume", type=float, help="Maximum tetrahedron volume in m^3 for automatic meshing.")
    parser.add_argument("--cache-dir", type=Path, help="Directory for content-addressed prepared meshes.")
    parser.add_argument("--output-mode", choices=("max", "full"), help="Defaults to full with a viewer, max otherwise.")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=730)
    parser.add_argument(
        "--substeps",
        type=int,
        help="Physical substeps: defaults to 4 for custom shell material collision, 1 otherwise.",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--color-max", type=float, help="Fixed color maximum in Pa; omitted uses a running peak scale.")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--log-every", type=int, default=60)
    args = parser.parse_args()
    if args.envs < 1 or args.steps < 1 or args.log_every < 1:
        parser.error("Use positive environment, step and logging counts.")
    if args.substeps is not None and args.substeps < 1:
        parser.error("--substeps must be positive.")
    if args.viewer and args.envs != 1:
        parser.error("The stress viewer shows one environment; omit --viewer for batched maximum-only runs.")
    if args.viewer and args.output_mode == "max":
        parser.error("Stress coloring requires full output; omit --output-mode when using --viewer.")
    if args.color_max is not None and (not math.isfinite(args.color_max) or args.color_max <= 0):
        parser.error("--color-max must be finite and positive, in Pa.")
    output_mode = args.output_mode or ("full" if args.viewer else "max")

    assets = Path(__file__).parent / "assets" / "hollow_egg" / "level1"
    gs.init(backend=gs.cpu if args.device == "cpu" else gs.gpu, precision="64")
    mesh = assets / ("shell.npz" if args.backend == "shell" else "elastic.npz")
    rigid_mesh = None
    prepared = None
    if args.mesh is not None or args.stress_mesh is not None or args.scale != 1.0 or args.max_volume is not None:
        elastic = args.stress_mesh
        if args.mesh is None and elastic is None:
            elastic = mesh
        prepared = prepare_stress_mesh(
            source=args.mesh or assets / "egg_outer.obj",
            elastic=elastic,
            backend=args.backend,
            scale=args.scale,
            max_volume=args.max_volume,
            cache=args.cache_dir or Path(get_cache_dir()) / "rigid_stress_meshes",
            thickness=args.thickness,
        )
        mesh = prepared.elastic
    if args.backend == "shell":
        options = gs.options.RigidStressOptions.dpr(
            mesh=mesh,
            backend="shell",
            shell_thickness=args.thickness,
            shell_loading="planar",
            loading_subdivisions=32,
            quadrature=10,
            scatter_tasks_per_env=128,
            box_edge_policy="planar",
            output_mode=output_mode,
        )
    else:
        options = gs.options.RigidStressOptions.dpr(
            mesh=mesh,
            box_edge_policy="planar",
            output_mode=output_mode,
        )

    surface = (
        StressFieldSurface(mesh, options.backend, options.shell_thickness)
        if args.viewer or prepared is not None
        else None
    )
    if prepared is not None:
        rigid_file = write_rigid_asset(prepared.collision, surface.material_mesh, options.density)
        rigid_mesh = gs.morphs.URDF(
            pos=(0.65, 0, 0.001 - prepared.minimum_z),
            visualization=not args.viewer,
            file=rigid_file,
            align=False,
            convexify=args.backend != "shell",
            decimate=False,
            watertighten=None if args.backend == "shell" else 5,
            preserve_mesh_precision=args.backend == "shell",
        )
    viewer_options = None
    if args.viewer:
        viewer_options = gs.options.ViewerOptions(
            camera_pos=(0.80, -0.23, 0.07),
            camera_lookat=(0.65, 0, 0.04),
        )
    # A material shell has an inner boundary that a coarser rigid contact step can cross.
    substeps = args.substeps or (4 if prepared is not None and args.backend == "shell" else 1)
    grasp = FrankaEgg(
        n_envs=args.envs,
        substeps=substeps,
        viewer=args.viewer,
        stress_options=options,
        rigid_mesh=rigid_mesh,
        stress_visualization=args.viewer,
        viewer_options=viewer_options,
    )
    print(
        f"Stress configuration: {options.backend}, {options.radius_model} radius, "
        f"{options.pressure_profile} pressure, {options.pressure_quadrature}/q{options.quadrature}, "
        f"{options.output_mode} output, {substeps} physical substeps."
    )
    field = surface if args.viewer else None
    overlay = None
    color_limit = args.color_max or 1.0
    if args.viewer:
        print("Surface von Mises colors: blue=0, red=color limit. The rigid shape stays undeformed.")
    for i_step in range(args.steps):
        if args.viewer and not grasp.scene.viewer.is_alive():
            break
        grasp.step()
        if field is not None:
            grasp.scene.rigid_solver.check_errno()
            _, von_mises = grasp.link.get_stress_field(envs_idx=[0], copy=False)
            if args.color_max is None:
                color_limit = max(color_limit, grasp.link.get_max_stress(copy=False).max().item())
            colored = field.colored_mesh(tensor_to_array(von_mises)[0], color_limit)
            transform = trans_quat_to_T(grasp.link.get_pos(envs_idx=[0])[0], grasp.link.get_quat(envs_idx=[0])[0])
            with grasp.scene.visualizer.viewer_lock:
                if overlay is not None:
                    grasp.scene.clear_debug_object(overlay)
                overlay = grasp.scene.draw_debug_mesh(colored, T=transform)
                # The material surface must occlude normally, unlike x-ray debug markers
                overlay.is_marker = False
            grasp.scene.visualizer.update(force=False)
        if (i_step + 1) % args.log_every == 0:
            grasp.scene.rigid_solver.check_errno()
            print(f"Step {i_step + 1}: max stress = {grasp.link.get_max_stress(copy=False).max().item():.6g} Pa")
            if field is not None:
                print(f"Color limit = {color_limit:.6g} Pa")
    grasp.scene.rigid_solver.check_errno()
    peak = grasp.link.get_max_stress()
    print("Maximum sampled von Mises stress [Pa], shape:", tuple(peak.shape))
    print("Maximum across environments [Pa]:", peak.max().item())
    if args.envs <= 16:
        print("Per-environment maxima [Pa]:", tensor_to_array(peak))
    if output_mode == "full":
        tensor, von_mises = grasp.link.get_stress_field()
        print("Stress tensor / von Mises [Pa] shapes:", tuple(tensor.shape), tuple(von_mises.shape))
        if options.backend == "shell":
            membrane, bending, shear = grasp.link.get_shell_resultants()
            print("N [N/m], M [N], Q [N/m] shapes:", tuple(membrane.shape), tuple(bending.shape), tuple(shear.shape))


if __name__ == "__main__":
    main()
