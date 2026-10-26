"""Compare evolving native spatial-atlas fields with their unchanged physical mother."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .check_local_feature_spaces import block_matrix, source_parents
from .check_native_atlas import NativeLifecycle
from .native_adaptation_options import (
    add_native_atlas_arguments,
    native_atlas_options,
    validate_native_atlas_arguments,
    weighted_resultant_error,
)
from .native_global_enrichment import add_global_enrichment_argument, actual_enrichment_space, write_global_basis


@qd.kernel
def kernel_copy_force(source: StressState, target: StressState):
    for i_n, i_b in qd.ndrange(source.force.shape[0], source.force.shape[1]):
        target.force[i_n, i_b] = source.force[i_n, i_b]


def relative_per_environment(actual, expected):
    delta = (actual - expected).reshape((len(actual), -1))
    reference = expected.reshape((len(expected), -1))
    return np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-30)


def weighted_field_error(actual, expected, weights):
    multiplicity = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    difference = np.sum((actual - expected) ** 2 * multiplicity, axis=-1)
    reference = np.sum(expected**2 * multiplicity, axis=-1)
    numerator = np.sum(difference * weights, axis=(1, 2))
    denominator = np.sum(reference * weights, axis=(1, 2))
    return np.sqrt(numerator / np.maximum(denominator, 1e-60))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--regions", type=int, default=12)
    parser.add_argument("--core-scale", type=float, default=1.0)
    parser.add_argument("--halo", type=int, default=0)
    parser.add_argument("--transition", type=float, default=0.003)
    parser.add_argument("--solver", choices=("factor", "inverse", "response", "packed"), default="factor")
    parser.add_argument("--space", choices=("local", "coarse"), default="local")
    parser.add_argument("--shell-bending", action="store_true")
    parser.add_argument("--quadrature", type=int, default=16)
    parser.add_argument("--pressure-quadrature", choices=("edge", "angle"), default="edge")
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--field-steps", nargs="*", type=int, default=(129, 221, 400, 599))
    parser.add_argument("--output", type=Path, required=True)
    add_native_atlas_arguments(parser)
    add_global_enrichment_argument(parser)
    args = parser.parse_args()
    validate_native_atlas_arguments(parser, args, args.regions)
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    result = dict(
        identity=runtime_identity(),
        stress_backend=args.stress_backend,
        level=args.level,
        conditions=file_id(args.conditions),
        complete=False,
        passed=False,
        tensor_budget=0.10,
        peak_budget=0.05,
        load_budget=1e-10,
        field_metric="Area/volume-weighted native sample tensor Frobenius norm; symmetric shears count twice.",
        lifecycle=args.lifecycle,
        global_enrichment_requested=args.global_enrichment,
        rows=[],
    )
    write_json(args.output / "result.json", result)
    assets = Path(gs.__file__).resolve().parent.parent / f"examples/rigid/assets/hollow_egg/level{args.level}"
    basis_path = None
    parents = None
    if args.stress_backend == "shell":
        parents, _xyz, provenance = source_parents(assets, True)
        basis_path = args.output / "basis.npz"
        result["basis_provenance"] = provenance
    if args.stress_backend == "shell" or args.global_enrichment != "none":
        basis_path = args.output / "basis.npz"
        result["global_enrichment"] = write_global_basis(
            np,
            basis_path,
            backend=args.stress_backend,
            mesh=assets / ("shell.npz" if args.stress_backend == "shell" else "elastic.npz"),
            mode=args.global_enrichment,
            parent_nodes=parents,
        )
        result["basis"] = file_id(basis_path)
    else:
        result["global_enrichment"] = dict(mode="none")
    workload = FrankaEgg(
        args.envs,
        level=args.level,
        conditions=args.conditions,
        output_mode="full",
        egg_geometry="mesh64",
        radius_model="hertz",
        pressure_profile="hertz",
        quadrature=args.quadrature,
        pressure_quadrature=args.pressure_quadrature,
        box_edge_policy="planar",
        stress_backend=args.stress_backend,
        atlas_regions=args.regions,
        atlas_basis=basis_path,
        atlas_core_scale=args.core_scale,
        atlas_halo=args.halo,
        atlas_transition=args.transition,
        atlas_solver=args.solver,
        atlas_space=args.space,
        atlas_shell_bending=args.shell_bending,
        inverse_max_bytes=512 * 1024 * 1024,
        **native_atlas_options(args),
    )
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    atlas = entry.feature_atlas
    assert atlas is not None
    result["atlas"] = asdict(atlas.metadata())
    result["global_enrichment_actual"] = actual_enrichment_space(atlas)
    result["genuine_reduction"] = any(
        group.blocks < atlas.metadata().mother_blocks for group in atlas.metadata().groups
    )
    result["covered_contacts"] = 0
    result["active_reduced_groups"] = 0
    result["options"] = entry.options.model_dump(mode="json")
    reference_options = entry.options.model_copy(update={"atlas_regions": 0})
    reference = entry.model.create_state(args.envs, "full")
    matrix = block_matrix(entry.model.info)
    boundary = qd_to_numpy(atlas.regions.info.boundary_nodes)
    if args.stress_backend == "shell":
        weights = np.repeat(qd_to_numpy(entry.model.shell.weights), 2, axis=0) / 2
    else:
        weights = np.broadcast_to(
            qd_to_numpy(entry.model.info.volumes)[:, None] / 4, (entry.model.n_output_elements, 4)
        )
    fields = (
        ("displacement", entry.state.displacement),
        ("stress_tensor", entry.state.stress_tensor),
        ("peak", entry.state.peak),
    )
    lifecycle = NativeLifecycle() if args.lifecycle else None
    all_passed = True
    for step in range(args.steps):
        if lifecycle is not None:
            lifecycle.before_step(workload, step, fields)
        workload.step()
        workload.scene.rigid_solver.check_errno()
        kernel_copy_force(entry.state, reference)
        entry.model.recover(entry.omega, reference, reference_options)
        assert qd_to_numpy(reference.valid).all()
        candidate = qd_to_numpy(entry.state.stress_tensor, transpose=True)
        expected = qd_to_numpy(reference.stress_tensor, transpose=True)
        q = qd_to_numpy(entry.state.displacement, transpose=True)
        q_reference = qd_to_numpy(reference.displacement, transpose=True)
        peak = qd_to_numpy(entry.state.peak, transpose=True)
        peak_reference = qd_to_numpy(reference.peak, transpose=True)
        grouped = qd_to_numpy(atlas.regions.state.force, transpose=True).sum(axis=2)
        raw = qd_to_numpy(entry.state.force, transpose=True)[:, boundary]
        load_error = relative_per_environment(grouped, raw)
        tensor_error = weighted_field_error(candidate, expected, weights)
        peak_error = relative_per_environment(peak, peak_reference)
        rhs = qd_to_numpy(reference.rhs, transpose=True).reshape((args.envs, -1))
        mother_residual = entry.options.young * (matrix @ q.reshape((args.envs, -1)).T).T - rhs
        mother_relative = np.linalg.norm(mother_residual, axis=1) / np.maximum(np.linalg.norm(rhs, axis=1), 1e-30)
        per_errors = dict(
            displacement=relative_per_environment(q, q_reference).tolist(),
            stress_tensor=tensor_error.tolist(),
            peak=peak_error.tolist(),
            grouped_load=load_error.tolist(),
        )
        errors = {name: max(values) for name, values in per_errors.items()}
        if args.stress_backend == "shell":
            for name, actual_field, expected_field in (
                ("membrane_resultant", entry.state.membrane_resultant, reference.membrane_resultant),
                ("bending_resultant", entry.state.bending_resultant, reference.bending_resultant),
                ("transverse_resultant", entry.state.transverse_resultant, reference.transverse_resultant),
            ):
                actual = qd_to_numpy(actual_field, transpose=True)
                wanted = qd_to_numpy(expected_field, transpose=True)
                value = relative_per_environment(actual, wanted)
                per_errors[name] = value.tolist()
                errors[name] = float(value.max())
                weighted = weighted_resultant_error(
                    np, actual, wanted, qd_to_numpy(entry.model.shell.weights), tensor=name != "transverse_resultant"
                )
                per_errors[f"weighted_{name}"] = weighted.tolist()
                errors[f"weighted_{name}"] = float(weighted.max())
        accepted = errors["stress_tensor"] <= 0.10 and errors["peak"] <= 0.05 and errors["grouped_load"] <= 1e-10
        if args.stress_backend == "shell":
            accepted = (
                accepted
                and max(
                    errors[prefix + name]
                    for name in ("membrane_resultant", "bending_resultant", "transverse_resultant")
                    for prefix in ("", "weighted_")
                )
                <= 0.10
            )
        all_passed = all_passed and accepted
        loaded = qd_to_numpy(entry.contacts.valid, transpose=True) & (
            np.linalg.norm(qd_to_numpy(entry.contacts.force, transpose=True), axis=-1) > 0
        )
        fallback = qd_to_numpy(atlas.regions.state.fallback, transpose=True)
        active = qd_to_numpy(atlas.state.active)
        reduced = np.array([group.blocks < atlas.metadata().mother_blocks for group in atlas.metadata().groups[1:]])
        row = dict(
            step=step,
            errors=errors,
            per_environment_errors=per_errors,
            mother_spatial_residual=mother_relative.tolist(),
            feature_numerical_residual_norm=np.sqrt(qd_to_numpy(atlas.state.residual_norm_squared)).tolist(),
            loaded=int(loaded.sum()),
            loaded_environments=loaded.any(axis=1).tolist(),
            coverage_fallbacks=int((loaded & fallback).sum()),
            active_groups=int(active[1:].sum()),
            active_reduced_groups=int(active[1:][reduced].sum()),
            spatial_passed=bool(accepted),
        )
        result["rows"].append(row)
        result["covered_contacts"] += row["loaded"] - row["coverage_fallbacks"]
        result["active_reduced_groups"] += row["active_reduced_groups"]
        if step in args.field_steps:
            np.savez_compressed(
                args.output / f"field-{step:04d}.npz",
                candidate=candidate,
                reference=expected,
                q=q,
                q_reference=q_reference,
                peak=peak,
                peak_reference=peak_reference,
                force=qd_to_numpy(entry.state.force, transpose=True),
                omega=qd_to_numpy(entry.omega, transpose=True),
                grouped_force=qd_to_numpy(atlas.regions.state.force, transpose=True),
                active_groups=qd_to_numpy(atlas.state.active, transpose=True),
            )
        if step % 10 == 0 or step + 1 == args.steps:
            write_json(args.output / "result.json", result)
    all_passed = (
        all_passed
        and result["genuine_reduction"]
        and result["covered_contacts"] > 0
        and result["active_reduced_groups"] > 0
    )
    field_observations = [
        value
        for row in result["rows"]
        for value, loaded in zip(
            row["per_environment_errors"]["stress_tensor"], row["loaded_environments"], strict=True
        )
        if loaded
    ]
    mean_field = float(np.mean(field_observations)) if field_observations else 0.0
    result.update(
        complete=True,
        passed=bool(all_passed and mean_field <= 0.05),
        mean_loaded_field_error=mean_field,
        mean_field_budget=0.05,
        mean_population="loaded environment-frames; releases excluded",
        resultant_metric="Original Euclidean gates plus area-weighted Frobenius N/M and vector Q; both retain the 10% limit.",
        spatial_passed=bool(all(row["spatial_passed"] for row in result["rows"]) and mean_field <= 0.05),
    )
    write_json(args.output / "result.json", result)
    assert result["passed"], (
        "Combined fields exceed the predeclared spatial/load budgets; preserve this rejected configuration."
    )


if __name__ == "__main__":
    main()
