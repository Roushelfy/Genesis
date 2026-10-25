"""Audit actual native atlas loads and complete fields against the same mother model."""

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import quadrants as qd
import scipy.linalg

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .audit_gpu_candidates import per_environment


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


class CpuStressOperator(NamedTuple):
    matrix: np.ndarray
    free: np.ndarray
    modes: np.ndarray
    relief: np.ndarray
    factor: tuple[np.ndarray, bool]


def cpu_operator(info):
    starts = qd_to_numpy(info.row_start)
    columns = qd_to_numpy(info.columns)
    blocks = qd_to_numpy(info.stiffness)
    n = len(starts) - 1
    matrix = np.zeros((n, 3, n, 3))
    rows = np.repeat(np.arange(n), np.diff(starts))
    matrix[rows, :, columns, :] = blocks
    matrix = matrix.reshape((3 * n, 3 * n))
    free = qd_to_numpy(info.is_free).reshape(-1).astype(bool)
    modes = qd_to_numpy(info.modes).reshape((3 * n, 6))
    mass_modes = qd_to_numpy(info.mass_modes).reshape((3 * n, 6))
    # Recompute the six-dimensional relief independently from the native mass/mode inputs.
    relief = scipy.linalg.solve(modes.T @ mass_modes, mass_modes.T, assume_a="sym").T
    factor = scipy.linalg.cho_factor(matrix[np.ix_(free, free)])
    return CpuStressOperator(matrix, free, modes, relief, factor)


def cpu_control(entry):
    info = entry.model.info
    matrix, free, modes, relief, factor = cpu_operator(info)
    n = info.vertices.shape[0]
    force = qd_to_numpy(entry.state.force, transpose=True)[0].reshape(-1)
    spin = qd_to_numpy(entry.omega, transpose=True)[0]
    terms = np.r_[spin**2, spin[0] * spin[1], spin[0] * spin[2], spin[1] * spin[2]]
    force += qd_to_numpy(info.centrifugal).reshape((3 * n, 6)) @ terms
    rhs = force - relief @ (modes.T @ force)
    displacement = np.zeros(3 * n)
    displacement[free] = scipy.linalg.cho_solve(factor, rhs[free]) / entry.options.young
    native = qd_to_numpy(entry.state.displacement, transpose=True)[0].reshape(-1)
    response = entry.model.surface_inverse.info
    nodes = qd_to_numpy(response.nodes)
    inputs = np.zeros((3 * n, 12))
    selected = np.linspace(0, len(nodes) - 1, 4, dtype=int)
    for i, node in enumerate(nodes[selected]):
        inputs[3 * node : 3 * node + 3, 3 * i : 3 * i + 3] = np.eye(3)
    balanced = inputs - relief @ (modes.T @ inputs)
    basis = qd_to_numpy(response.force_blocks)[:, selected].transpose(0, 2, 1, 3).reshape((3 * n, 12))
    cpu_basis = np.zeros_like(basis)
    cpu_basis[free] = scipy.linalg.cho_solve(factor, balanced[free])
    energy = basis.T @ matrix @ basis
    work = basis.T @ balanced
    result = dict(
        displacement_relative=relative(native, displacement),
        response_relative=relative(basis, cpu_basis),
        projected_work_relative=relative(energy, work),
        relieved_rigid_wrench=float(np.linalg.norm(modes.T @ rhs) / max(np.linalg.norm(force), 1e-30)),
        pinned_displacement_max=float(np.max(np.abs(native[~free]))),
        free_dofs=int(np.count_nonzero(free)),
    )
    assert max(result[key] for key in ("displacement_relative", "response_relative", "projected_work_relative")) < 1e-8
    assert result["relieved_rigid_wrench"] < 1e-10
    assert result["pinned_displacement_max"] == 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--pressure-quadrature", choices=("edge", "angle"), default="edge")
    parser.add_argument("--quadrature", type=int, default=16)
    parser.add_argument("--atlas-regions", type=int, default=6)
    parser.add_argument("--atlas-transition", type=float, default=0.003)
    parser.add_argument("--atlas-output", choices=("merged", "independent"), default="merged")
    parser.add_argument("--atlas-partition", choices=("direct", "coarse_local"), default="direct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        identity=runtime_identity(),
        settings=vars(args) | {"conditions": str(args.conditions), "output": str(args.output)},
        rows=[],
        complete=False,
    )
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(
        args.envs,
        conditions=args.conditions,
        output_mode="full",
        radius_model="hertz",
        pressure_profile="hertz",
        quadrature=args.quadrature,
        pressure_quadrature=args.pressure_quadrature,
        box_edge_policy="planar",
        stress_backend=args.stress_backend,
        atlas_regions=args.atlas_regions,
        atlas_transition=args.atlas_transition,
        atlas_output=args.atlas_output,
        atlas_partition=args.atlas_partition,
    )
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    state, atlas = entry.state, entry.atlas.state
    reference_options = entry.options.model_copy(update={"atlas_regions": 0})
    fields = (("displacement", state.displacement), ("stress_tensor", state.stress_tensor), ("peak", state.peak))
    if args.stress_backend == "shell":
        fields += (
            ("membrane_resultant", state.membrane_resultant),
            ("bending_resultant", state.bending_resultant),
            ("transverse_resultant", state.transverse_resultant),
        )
    for step in range(args.steps):
        workload.step()
        # All requested fields are checked at every evolving physical observation.
        candidate = {name: qd_to_numpy(array, transpose=True) for name, array in fields}
        force = qd_to_numpy(state.force, transpose=True)
        grouped = qd_to_numpy(atlas.force).sum(axis=1).transpose(1, 0, 2)
        weights = qd_to_numpy(atlas.weights)
        active = qd_to_numpy(entry.contacts.valid) & (qd_to_numpy(entry.contacts.status) == 0)
        assert qd_to_numpy(state.valid).all()
        assert (np.abs(weights.sum(axis=0)[active] - 1.0) < 1e-14).all()
        row = dict(
            step=step,
            contacts=int(np.count_nonzero(active)),
            grouped_load_relative=float(per_environment(grouped, force).max()),
            maximum_contributors=int(np.count_nonzero(weights, axis=0).max()),
            coverage_fallbacks=int(np.count_nonzero(qd_to_numpy(atlas.fallback) & active)),
            active_groups=int(qd_to_numpy(atlas.count)),
            corrections=qd_to_numpy(state.corrections).tolist(),
            fallbacks=qd_to_numpy(state.fallbacks).tolist(),
        )
        entry.model.recover(entry.omega, state, reference_options, surface_load=True)
        row["errors"] = {
            name: float(per_environment(candidate[name], qd_to_numpy(array, transpose=True)).max())
            for name, array in fields
        }
        result["rows"].append(row)
        write_json(args.output / "result.json", result)
        assert qd_to_numpy(state.valid).all()
        assert row["grouped_load_relative"] < 1e-10
        assert max(row["errors"].values()) < 1e-8
        if step == min(129, args.steps - 1):
            result["cpu_control"] = cpu_control(entry)
            np.savez_compressed(
                args.output / "fields.npz", **candidate, force=force, grouped_force=grouped, weights=weights
            )
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
