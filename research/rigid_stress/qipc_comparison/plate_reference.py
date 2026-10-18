"""Measure force-controlled hollow-solid QIPC compression without a robot or egg constraints."""

import json
from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch

import qipc
import qipc._src.native.solver.qipc_solver_ext as extension
from qipc import tetmesh
from qipc._src.native.solver import GlobalVertexManager

from .artifacts import file_id, runtime_identity, write_json
from .plates import equilibrium, force_history, parser, plate_xml
from .reference import ReferenceConfig, SolidObserver, preserve_contact_failure, solver_statistics


def main():
    cli = parser()
    cli.add_argument("--velocity-tolerance", type=float, default=1e-8)
    cli.add_argument("--linear-tolerance", type=float, default=1e-4)
    cli.add_argument("--d-hat", type=float, default=2e-5)
    cli.add_argument("--resistance", type=float, default=1e10)
    cli.add_argument("--friction", type=float, default=0.0)
    cli.add_argument("--linear-solver", choices=("linear_pcg", "partition_pcg"), default="linear_pcg")
    cli.add_argument("--mas-precision", choices=("float", "double"), default="float")
    cli.add_argument("--cost-steps", type=int, default=0)
    cli.add_argument("--cost-stride", type=int, default=100)
    cli.add_argument("--cost-repeats", type=int, default=3)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    config = ReferenceConfig(
        dt=args.dt,
        velocity_tolerance=args.velocity_tolerance,
        linear_tolerance=args.linear_tolerance,
        linear_solver=args.linear_solver,
        mas_precision=args.mas_precision,
        d_hat=args.d_hat,
        resistance=args.resistance,
        friction=args.friction,
    )
    with np.load(args.mesh) as mesh:
        vertices, tets = mesh["vertices"].copy(), mesh["tetrahedra"].copy()
    lower, upper = vertices[:, 2].min() - args.clearance, vertices[:, 2].max() + args.clearance
    xml = args.output / "plates.xml"
    xml.write_text(plate_xml(lower, upper, args.plate_offset_x))
    scene = config.scene(gravity=(0.0, 0.0, 0.0))
    plate = scene.add_mjcf(
        xml, link_type="rigid", position=(args.plate_offset_x, 0.0, float(lower - 0.005)), enable_controller=True
    )
    assert plate.joint_names == ["compression"], plate.joint_names
    if not args.empty:
        geometry = tetmesh(vertices, tets)
        config.material().apply_to(
            geometry, youngs_modulus=config.young, poissons_ratio=config.poisson, mass_density=config.density
        )
        scene.geometries.create("egg", geometry)
    started = perf_counter()
    scene.init()
    torch.cuda.synchronize()
    plate.set_dofs_kp(0.0)
    plate.set_dofs_kv(args.damping)
    plate.set_dofs_force_range(-np.inf, np.inf)
    observer = None
    if not args.empty:
        with torch.cuda.stream(scene.solver.cuda_stream):
            observer = SolidObserver.allocate(scene, config)
    indices = {
        slot.name.rsplit("/", 1)[-1]: slot.geometry.meta["rigid_body_offset"].cpu()[0].item() for slot in plate.links
    }
    owners = scene.sim_systems[GlobalVertexManager].body_id_view()
    assert scene.affine_body.n_bodies == 0
    initial_positions = scene.sim_systems[GlobalVertexManager].positions_view()
    for name, height in (("lower", lower - 0.005), ("upper", upper + 0.005)):
        selected = initial_positions[owners == indices[name]]
        np.testing.assert_allclose(selected[:, 2].min().item(), height - 0.005, rtol=0, atol=1e-12)
        np.testing.assert_allclose(selected[:, 2].max().item(), height + 0.005, rtol=0, atol=1e-12)
    np.savez_compressed(
        args.output / "initial-geometry.npz",
        positions_m=initial_positions.cpu().numpy(),
        global_owners=owners.cpu().numpy(),
    )
    write_json(
        args.output / "start.json",
        {
            "args": {key: str(value) if key in ("mesh", "output") else value for key, value in vars(args).items()},
            "input": file_id(args.mesh),
            "native": file_id(extension.__file__),
            "qipc_import": qipc.__file__,
            "identity": runtime_identity(),
            "config": asdict(config),
            "plate_body_indices": indices,
            "plate_masses_kg": scene.rigid_body.mass.cpu().tolist(),
            "plate_separation_at_rest_m": upper - lower,
            "setup_seconds": perf_counter() - started,
            "scope": "Intrusive short pilot; no reference qualification inferred from completion.",
        },
    )
    np.savez_compressed(args.output / "rest.npz", vertices=vertices, tetrahedra=tets)
    completed, executed, samples = False, 0, []
    newton_caps = 0
    pcg_caps = 0
    predicted_velocity, predicted_q = 0.0, 0.0
    started = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step, (level, within, force, endpoint) in enumerate(force_history(args)):
                if observer is not None:
                    with torch.cuda.stream(scene.solver.cuda_stream):
                        previous_velocity = scene.finite_element.velocities.clone()
                plate.control_dofs_force(force)
                scene.step()
                executed += 1
                stats = solver_statistics(scene)
                newton_caps += stats["newton"] >= config.max_newton
                pcg_caps += stats["pcg_max"] >= config.max_pcg
                if stats["edge_triangle_intersection_count"]:
                    write_json(args.output / "rejected-step.json", {"step": step + 1, **stats})
                    raise RuntimeError("Reference edge-triangle intersection; continuation is rejected")
                predicted_velocity = (0.05 * predicted_velocity + args.dt * force) / (0.05 + args.dt * args.damping)
                predicted_q += args.dt * predicted_velocity
                if (step + 1) % args.record_stride and not endpoint:
                    continue
                torch.cuda.synchronize()
                velocity = -scene.rigid_body.v[indices["upper"], 2].item()
                q = plate.get_dofs_position()[0].item()
                row = {
                    "step": step + 1,
                    "time_s": (step + 1) * args.dt,
                    "level": level,
                    "within_level_step": within + 1,
                    "command_force_n": force,
                    "plate_speed_m_per_s": velocity,
                    "compression_coordinate_m": q,
                    "plate_separation_m": upper - lower - q,
                    "upper_lateral_position_m": scene.rigid_body.t[indices["upper"], :2].cpu().tolist(),
                    "upper_quaternion_wxyz": scene.rigid_body.quat[indices["upper"]].cpu().tolist(),
                    "empty_analytic_position_error_m": q - predicted_q if args.empty else None,
                    "empty_analytic_velocity_error_m_per_s": velocity - predicted_velocity if args.empty else None,
                    **stats,
                }
                if observer is not None:
                    state = observer.read_step(previous_velocity)
                    contact = state.contact
                    active_owners = owners[contact.global_vertex_ids]
                    reactions = {
                        name: contact.global_forces_n[active_owners == index].sum(dim=0)
                        for name, index in indices.items()
                    }
                    row.update(
                        upper_reaction_n=reactions["upper"][2].item(),
                        lower_reaction_n=-reactions["lower"][2].item(),
                        egg_inertia_resultant_norm_n=torch.linalg.norm(state.inertial_reaction_n.sum(dim=0)).item(),
                        egg_kinetic_j=state.kinetic_j.item(),
                        equation_residual_relative=torch.linalg.norm(state.residual_n).item()
                        / max(torch.linalg.norm(contact.fem_forces_n).item(), 1e-12),
                        minimum_det_F=state.field.determinant.min().item(),
                        raw_vm_peak_pa=state.field.von_mises_pa.max().item(),
                        raw_principal_peak_pa=state.field.maximum_principal_pa.max().item(),
                    )
                    if endpoint:
                        row["equilibrium_checks"] = equilibrium(row, force)
                        np.savez_compressed(
                            args.output / "frames" / f"level{level}.npz",
                            positions_m=state.positions_m.cpu().numpy(),
                            velocities_m_per_s=scene.finite_element.velocities.cpu().numpy(),
                            cauchy_pa=state.field.cauchy_pa.cpu().numpy(),
                            contact_forces_n=contact.fem_forces_n.cpu().numpy(),
                            inertial_reaction_n=state.inertial_reaction_n.cpu().numpy(),
                            residual_n=state.residual_n.cpu().numpy(),
                            plate_position_m=scene.rigid_body.t.cpu().numpy(),
                            active_global_ids=contact.global_vertex_ids.cpu().numpy(),
                            active_global_forces_n=contact.global_forces_n.cpu().numpy(),
                            active_global_positions_m=contact.global_positions_m[contact.global_vertex_ids]
                            .cpu()
                            .numpy(),
                            active_global_owners=active_owners.cpu().numpy(),
                        )
                    assert row["minimum_det_F"] > 0 and torch.isfinite(state.field.cauchy_pa).all()
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                samples.append(row)
                if args.empty and endpoint:
                    assert abs(row["empty_analytic_position_error_m"]) < 1e-12
                    assert abs(row["empty_analytic_velocity_error_m_per_s"]) < 1e-10
                if newton_caps:
                    raise RuntimeError("Reference reached its Newton iteration cap")
            if args.cost_steps:
                from .plate_reference_cost import measure

                measure(args, config, scene, plate, observer, owners, indices)
            completed = True
        finally:
            if not completed and observer is not None:
                preserve_contact_failure(scene, args.output)
            write_json(
                args.output / "result.json",
                {
                    "completed": completed,
                    "steps_returned": executed,
                    "newton_cap_steps": newton_caps,
                    "pcg_cap_steps": pcg_caps,
                    "elapsed_seconds_including_observation_IO": perf_counter() - started,
                    "last_observation": samples[-1] if samples else None,
                    "reference_qualified": False,
                },
            )


if __name__ == "__main__":
    main()
