"""Audit geometry, virtual work, sampled stress and reduced-space consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .fixtures import make_cases
from .loads import assemble_sources, source_directions
from .model import Mother
from .pipeline import Pipeline, sampled_maximum, tensor_output
from .selection import build_bank


def relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30))


def run(output: Path) -> dict:
    mother = Mother()
    geometry = mother.geometry
    bank = build_bank(mother, count=12)
    checks = {}
    checks["mother_dofs"] = mother.n_dof
    checks["mother_nodes"] = len(geometry.vertices)
    checks["mother_quads"] = len(geometry.quads)
    checks["coarse_dofs"] = mother.P.shape[1]
    checks["common_stress_rows"] = mother.S.shape[0]
    assert (mother.n_dof, len(geometry.vertices), len(geometry.quads)) == (5196, 866, 864)
    assert mother.P.shape == (5196, 336) and mother.S.shape == (20736, 5196)

    edges = np.sort(np.concatenate([geometry.quads[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1)
    unique_edges, multiplicities = np.unique(edges, axis=0, return_counts=True)
    checks["edge_incidence_minmax"] = [int(multiplicities.min()), int(multiplicities.max())]
    checks["euler_characteristic"] = len(geometry.vertices) - len(unique_edges) + len(geometry.quads)
    assert np.all(multiplicities == 2) and checks["euler_characteristic"] == 2
    partition = np.asarray(geometry.P_nodes.sum(axis=1))[:, 0]
    checks["prolongation_partition_error"] = float(np.max(np.abs(partition - 1.0)))
    checks["prolongation_geometry_relative"] = relative(geometry.P_nodes @ geometry.coarse_vertices, geometry.vertices)
    coarse_modes = mother.model.rigid_modes[(6 * geometry.coarse_nodes[:, None] + np.arange(6)).reshape(-1)]
    checks["prolongation_rigid_relative"] = relative(mother.P @ coarse_modes, mother.model.rigid_modes)
    assert checks["prolongation_rigid_relative"] < 1e-12

    modes = mother.model.rigid_modes
    null = mother.K @ modes
    stress_null = mother.S @ modes
    checks["stiffness_rigid_scaled"] = float(np.linalg.norm(null) / (np.linalg.norm(mother.K.data) * np.linalg.norm(modes)))
    checks["stress_rigid_scaled"] = float(np.linalg.norm(stress_null) / (np.linalg.norm(mother.S.data) * np.linalg.norm(modes)))
    assert checks["stiffness_rigid_scaled"] < 1e-13 and checks["stress_rigid_scaled"] < 1e-13

    cases = make_cases()
    sources = tuple(source for case in cases for source in case.sources)
    raw = assemble_sources(mother, sources, order=8, audit=True)
    balanced = mother.balance(raw)
    virtual_work = modes.T @ balanced
    checks["relieved_rigid_virtual_work_scaled"] = float(np.linalg.norm(virtual_work) / max(np.linalg.norm(raw), 1e-30))
    assert checks["relieved_rigid_virtual_work_scaled"] < 1e-12

    source = cases[30].sources[0]
    force = assemble_sources(mother, (source,), order=8)
    q = mother.solve(mother.balance(force))
    local = mother.S @ q
    constitutive = mother.model.stress(q)
    expected_local = np.stack((constitutive["local_top"], constitutive["local_bottom"])).reshape(local.shape)
    expected_tensor = np.stack((constitutive["sigma_top"], constitutive["sigma_bottom"])).reshape((-1, 9, 1))
    actual_tensor = tensor_output(mother, local)
    checks["sparse_stress_vs_constitutive_relative"] = relative(local, expected_local)
    checks["body_tensor_vs_constitutive_relative"] = relative(actual_tensor, expected_tensor)
    tensors = actual_tensor.reshape((-1, 3, 3, 1))
    trace = np.einsum("niib->nb", tensors)
    deviator = tensors - np.eye(3)[None, :, :, None] * trace[:, None, None, :] / 3.0
    tensor_vm = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(1, 2))).max(axis=0)
    checks["plane_vm_vs_body_vm_relative"] = relative(sampled_maximum(local), tensor_vm)
    assert max(checks[key] for key in ("sparse_stress_vs_constitutive_relative",
                                      "body_tensor_vs_constitutive_relative",
                                      "plane_vm_vs_body_vm_relative")) < 1e-12
    residual = mother.K @ q - mother.balance(force)
    checks["mother_free_relative_residual"] = float(np.linalg.norm(residual[mother.free]) /
                                                   max(np.linalg.norm(mother.balance(force)[mother.free]), 1e-30))
    assert checks["mother_free_relative_residual"] < 1e-10

    maximum_feature_residual = 0.0
    maximum_raw_projection_difference = 0.0
    maximum_group_merge_difference = 0.0
    coverage_count = 0
    for source in sources:
        directions, _ = source_directions(source, order=8)
        location = geometry.locate(directions)
        loaded_nodes = np.unique(location.nodes)
        indices, _ = bank.select(source, "continuous")
        for i_space in indices:
            space = bank.spaces[i_space]
            covered = np.zeros(len(geometry.vertices), dtype=bool)
            covered[geometry.coarse_nodes] = True
            covered[space.local_nodes] = True
            assert np.all(covered[loaded_nodes])
            assert np.max(np.abs(space.T[mother.fixed].data), initial=0.0) < 1e-15
            coverage_count += 1
    for i_space, space in enumerate(bank.spaces):
        raw_pair = raw[:, [i_space, (i_space + 1) % raw.shape[1]]]
        pair = mother.balance(raw_pair)
        projected = space.project(pair)
        direct_projected = space.project_raw(raw_pair)
        maximum_raw_projection_difference = max(maximum_raw_projection_difference, relative(direct_projected, projected))
        reduced = space.solve_projected(projected)
        r = space.K @ reduced - projected
        maximum_feature_residual = max(maximum_feature_residual, float(np.linalg.norm(r) / max(np.linalg.norm(projected), 1e-30)))
        merged = space.lift(space.solve(pair.sum(axis=1)))
        independent = space.lift(reduced).sum(axis=1)
        maximum_group_merge_difference = max(maximum_group_merge_difference, relative(merged, independent))
    checks["full_shape_support_coverage_contributions"] = coverage_count
    checks["feature_residual_maximum"] = maximum_feature_residual
    checks["projected_relief_vs_full_relief_maximum"] = maximum_raw_projection_difference
    checks["rhs_group_merge_vs_source_superposition_maximum"] = maximum_group_merge_difference
    assert maximum_feature_residual < 1e-10
    assert maximum_raw_projection_difference < 1e-12 and maximum_group_merge_difference < 1e-11

    pipeline = Pipeline(mother, bank.spaces, bank.select)
    small_cases = tuple(case.sources for case in cases[30:33]) + ((),)
    lifted = pipeline.run(small_cases, strategy="continuous", output="local")
    independent = pipeline.run(small_cases, strategy="continuous", output="local", recovery_mode="independent")
    projected = pipeline.run(small_cases, strategy="continuous", output="local", rhs_mode="projected")
    checks["pipeline_lift_vs_independent_stress"] = relative(lifted.local_stress, independent.local_stress)
    checks["pipeline_projected_vs_balanced_rhs"] = relative(lifted.local_stress, projected.local_stress)
    checks["pipeline_release_exact_zero"] = bool(np.count_nonzero(lifted.local_stress[:, -1]) == 0)
    assert checks["pipeline_lift_vs_independent_stress"] < 1e-11
    assert checks["pipeline_projected_vs_balanced_rhs"] < 1e-11 and checks["pipeline_release_exact_zero"]

    checks["feature_dofs"] = [space.n_dof for space in bank.spaces]
    checks["feature_reduced_to_mother_free_ratio"] = [space.n_dof / len(mother.free) for space in bank.spaces]
    assert all(space.n_dof < len(mother.free) for space in bank.spaces)
    package = Path(__file__).resolve().parent
    source_files = [package / name for name in ("geometry.py", "model.py", "loads.py", "audit.py", "pipeline.py",
                                                "selection.py", "fixtures.py")]
    result = {
        "kind": "reconstructed_common_mother_audit", "passed": True, "checks": checks,
        "geometry_convention": geometry.position_convention,
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
