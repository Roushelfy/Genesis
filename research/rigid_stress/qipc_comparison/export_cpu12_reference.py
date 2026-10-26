"""Freeze the published CPU twelve-feature shell as an independent parity oracle.

Run this scientific driver through Slurm rtx-mid, image 1_26.  Native production
may read inputs/ and cases/; everything under oracles/ is validation-only.
Scientific imports deliberately follow the Slurm guard in main().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

PRIMARY = "a10c5abe3b5bc3c5caa57d3d4d9dadab2b277ade"
SCHEMA = "genesis.cpu12-reference.v1"
STRATEGIES = ("continuous", "nearest", "narrow")
STATIC_INDICES = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 24, 28, 30, 31, 32, 42)
REPOSITORY = Path(__file__).resolve().parents[3]
REFERENCE = REPOSITORY / "research/rigid_stress/atlas_study/cpu_repro"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(git_source: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(git_source), *arguments], text=True).strip()


def _reference_api() -> SimpleNamespace:
    import numpy as np
    import scipy
    from scipy import sparse
    from threadpoolctl import threadpool_limits

    from research.rigid_stress.atlas_study.cpu_repro.boundary_study import boundary_locations
    from research.rigid_stress.atlas_study.cpu_repro.fixtures import Case, cases_manifest, make_cases
    from research.rigid_stress.atlas_study.cpu_repro.geometry import AXES, BETA, THICKNESS
    from research.rigid_stress.atlas_study.cpu_repro.loads import (
        Source,
        assemble_sources,
        cap_rule,
        force_for_source,
        source_directions,
    )
    from research.rigid_stress.atlas_study.cpu_repro.model import Mother
    from research.rigid_stress.atlas_study.cpu_repro.pipeline import aggregate_environments, tensor_output
    from research.rigid_stress.atlas_study.cpu_repro.selection import (
        build_bank,
        field_weights,
        relative_field_error,
        sampled_maximum,
    )

    return SimpleNamespace(**locals())


class Writer:
    def __init__(self, output: Path, api: SimpleNamespace):
        self.output, self.api = output, api
        self.inventory = {}

    def _path(self, relative: str) -> Path:
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _register(self, path: Path, arrays=None):
        entry = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if arrays is not None:
            np = self.api.np
            entry["arrays"] = {
                key: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256_c_order": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                }
                for key, value in arrays.items()
            }
        self.inventory[str(path.relative_to(self.output))] = entry

    def arrays(self, relative: str, **arrays):
        np = self.api.np
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        if any(value.dtype.hasobject for value in arrays.values()):
            raise TypeError("The frozen schema never uses object arrays or pickle.")
        path = self._path(relative)
        np.savez_compressed(path, **arrays)
        self._register(path, arrays)

    def matrix(self, relative: str, matrix):
        matrix = matrix.tocsr()
        path = self._path(relative)
        self.api.sparse.save_npz(path, matrix, compressed=True)
        with self.api.np.load(path, allow_pickle=False) as saved:
            self._register(path, {key: saved[key] for key in saved.files})

    def json(self, relative: str, payload):
        path = self._path(relative)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self._register(path)


def _incidence(np, rows):
    counts = np.array([len(row) for row in rows], dtype=np.int64)
    return np.r_[0, np.cumsum(counts)], np.concatenate(rows).astype(np.int64, copy=False)


def _parent_topology(api, geometry):
    """Replay first macro registration, retaining all four corners, including zero weights."""
    np = api.np
    count, subdivision = len(geometry.vertices), geometry.subdiv
    corners = np.full((count, 4), -1, dtype=np.int64)
    ij = np.full((count, 2), -1, dtype=np.int64)
    for table in geometry.face_tables:
        for j in range(geometry.ncoarse):
            for i in range(geometry.ncoarse):
                parent = (
                    table[j * subdivision, i * subdivision],
                    table[j * subdivision, (i + 1) * subdivision],
                    table[(j + 1) * subdivision, (i + 1) * subdivision],
                    table[(j + 1) * subdivision, i * subdivision],
                )
                for sj in range(subdivision + 1):
                    for si in range(subdivision + 1):
                        node = table[j * subdivision + sj, i * subdivision + si]
                        if corners[node, 0] < 0:
                            corners[node] = parent
                            ij[node] = si, sj
    if np.any(corners < 0) or np.any(ij < 0):
        raise AssertionError("Incomplete first-registration topology.")
    uv = ij / subdivision
    weights = np.column_stack(
        ((1 - uv[:, 0]) * (1 - uv[:, 1]), uv[:, 0] * (1 - uv[:, 1]), uv[:, 0] * uv[:, 1], (1 - uv[:, 0]) * uv[:, 1])
    )
    coarse_index = np.full(count, -1, dtype=np.int64)
    coarse_index[geometry.coarse_nodes] = np.arange(len(geometry.coarse_nodes))
    rebuilt = api.sparse.coo_matrix(
        (weights.ravel(), (np.repeat(np.arange(count), 4), coarse_index[corners].ravel())),
        shape=geometry.P_nodes.shape,
    ).tocsr()
    error = rebuilt - geometry.P_nodes
    if np.max(np.abs(error.data), initial=0.0) > 1e-15:
        raise AssertionError("Integer parent topology does not reproduce original bilinear P.")
    return corners, ij


def _export_model(writer, mother, bank):
    api, np = writer.api, writer.api.np
    geometry, model = mother.geometry, mother.model
    corners, ij = _parent_topology(api, geometry)
    node_indptr, node_indices = _incidence(np, geometry.node_macros)
    neighbor_indptr, neighbor_indices = _incidence(np, geometry.macro_neighbors)
    writer.arrays(
        "inputs/model.npz",
        vertices=geometry.vertices,
        quads=geometry.quads,
        directors=geometry.directors,
        directions=geometry.directions,
        coarse_vertices=geometry.coarse_vertices,
        coarse_directions=geometry.coarse_directions,
        coarse_nodes=geometry.coarse_nodes,
        parent_corners=corners,
        parent_ij=ij,
        ncoarse=geometry.ncoarse,
        subdiv=geometry.subdiv,
        macro_quads=geometry.macro_quads,
        macro_coefficients=geometry.macro_coefficients,
        macro_centers=geometry.macro_centers,
        macro_radii=geometry.macro_radii,
        face_tables=geometry.face_tables,
        node_macro_indptr=node_indptr,
        node_macro_indices=node_indices,
        macro_neighbor_indptr=neighbor_indptr,
        macro_neighbor_indices=neighbor_indices,
        fixed=mother.fixed,
        coarse_fixed=mother.coarse_fixed,
        young=model.E,
        poisson=model.nu,
        thickness=model.h,
        density=model.density,
        drilling_factor=model.drilling_factor,
        egg_axes=api.AXES,
        egg_beta=api.BETA,
    )
    writer.arrays(
        "inputs/selection.npz",
        centers=bank.centers,
        fill_angle=bank.fill_angle,
        support_angle=bank.support_angle,
        core_angle=bank.core_angle,
        narrow_fraction=bank.narrow_fraction,
        maximum_radius=bank.maximum_radius,
    )
    writer.matrix("oracles/operators/P_nodes.npz", geometry.P_nodes)
    writer.matrix("oracles/operators/P.npz", mother.P)
    writer.matrix("oracles/operators/mother/K.npz", mother.K)
    writer.matrix("oracles/operators/mother/K_free.npz", mother.K[mother.free][:, mother.free])
    writer.matrix("oracles/operators/mother/S.npz", mother.S)
    writer.arrays(
        "oracles/operators/mother/numerics.npz",
        rigid_modes=model.rigid_modes,
        mass_modes=mother.mass_modes,
        mass_gram=model.mass_gram,
        relief_coefficients=mother.relief_coefficients,
        nodal_mass=model.nodal_mass,
        rotary_mass=model.rotary_mass,
        mass_diagonal=model.mass_diagonal,
        center_of_mass=model.center_of_mass,
        axes=model.axes,
        gauss_weights=model.gauss_weights,
        gauss_points=model.gauss_points,
        B_membrane=model.B_membrane,
        B_bending=model.B_bending,
        B_shear=model.B_shear,
        element_dofs=model.element_dofs,
        free=mother.free,
        coarse_free=mother.coarse_free,
        field_weights=api.field_weights(mother),
    )
    rows = []
    for index, space in enumerate(bank.spaces):
        prefix = f"oracles/operators/feature-{index:02d}"
        writer.matrix(f"{prefix}/K.npz", space.K)
        writer.matrix(f"{prefix}/T.npz", space.T)
        writer.matrix(f"{prefix}/S.npz", space.S)
        writer.arrays(
            f"{prefix}/space.npz",
            active_macros=space.active_macros,
            local_nodes=space.local_nodes,
            local_dofs=(6 * space.local_nodes[:, None] + np.arange(6)).reshape(-1),
            relief_projection=space.relief_projection,
        )
        # Cardinal coarse rows and local identity bubbles form a block triangular
        # square minor with identity diagonal; no expensive dense rank estimate.
        physical_coarse_rows = (6 * geometry.coarse_nodes[:, None] + np.arange(6)).ravel()[mother.coarse_free]
        coarse_minor = space.T[physical_coarse_rows, : len(mother.coarse_free)]
        cardinal_error = coarse_minor - api.sparse.eye(len(mother.coarse_free), format="csr")
        if np.max(np.abs(cardinal_error.data), initial=0.0) > 1e-15:
            raise AssertionError("Feature coarse cardinal rows are not identity.")
        bubble_rows = (6 * space.local_nodes[:, None] + np.arange(6)).ravel()
        bubble_error = space.T[bubble_rows, len(mother.coarse_free) :] - api.sparse.eye(len(bubble_rows), format="csr")
        if np.max(np.abs(bubble_error.data), initial=0.0) > 1e-15:
            raise AssertionError("Feature bubble rows are not identity.")
        if space.T[physical_coarse_rows, len(mother.coarse_free) :].nnz:
            raise AssertionError("Feature bubbles reach coarse cardinal rows.")
        if np.max(np.abs(space.T[mother.fixed].data), initial=0.0) > 1e-15:
            raise AssertionError("Feature transform violates the common gauge.")
        rows.append(
            {
                "index": index,
                "raw_dofs": space.n_dof_total,
                "free_dofs": space.n_dof,
                "bubble_nodes": len(space.local_nodes),
                "active_macros": int(space.active_macros.sum()),
                "transform_rank": space.n_dof,
                "transform_rank_proof": "cardinal_coarse_and_identity_bubble_minor",
                "K_nnz": space.K.nnz,
                "T_nnz": space.T.nnz,
                "S_nnz": space.S.nnz,
                **_export_factor(writer, prefix, space.factor),
            }
        )
    pin_singular_values = np.linalg.svd(model.rigid_modes[mother.fixed], compute_uv=False)
    if pin_singular_values[-1] <= 1e-12:
        raise AssertionError("Frozen six translation pins do not constrain the rigid modes.")
    return {
        "mother": {
            "nodes": len(geometry.vertices),
            "quads": len(geometry.quads),
            "raw_dofs": mother.n_dof,
            "free_dofs": len(mother.free),
            "K_nnz": mother.K.nnz,
            "S_nnz": mother.S.nnz,
            "P_rank": mother.P.shape[1],
            "coarse_nodes": len(geometry.coarse_nodes),
            "gauge_rank": 6,
            "gauge_singular_values": pin_singular_values.tolist(),
            "assembly_seconds": mother.assembly_seconds,
            **_export_factor(writer, "oracles/operators/mother", mother.factor),
        },
        "features": rows,
        "position_convention": geometry.position_convention,
    }


def _export_factor(writer, prefix, factor):
    writer.matrix(f"{prefix}/factor_L.npz", factor.lu.L)
    writer.matrix(f"{prefix}/factor_U.npz", factor.lu.U)
    writer.arrays(
        f"{prefix}/factor.npz",
        scale=factor.scale,
        perm_r=factor.lu.perm_r,
        perm_c=factor.lu.perm_c,
    )
    return {
        "factor_seconds": factor.seconds,
        "factor_payload_bytes": factor.payload_bytes,
        "factor_L_nnz": factor.lu.L.nnz,
        "factor_U_nnz": factor.lu.U.nnz,
    }


def _export_cases(writer, name, cases, **extra):
    api, np = writer.api, writer.api.np
    sources = tuple(source for case in cases for source in case.sources)
    count = len(sources)

    def vectors(field):
        return np.array(
            [np.zeros(3) if getattr(source, field) is None else getattr(source, field) for source in sources],
            dtype=np.float64,
        ).reshape((count, 3))

    offsets = np.cumsum([0, *(len(case.sources) for case in cases)], dtype=np.int64)
    environment = np.repeat(np.arange(len(cases), dtype=np.int64), np.diff(offsets))
    writer.arrays(
        f"cases/{name}.npz",
        direction=vectors("direction"),
        radius=[source.radius for source in sources],
        force=vectors("force"),
        has_force=[source.force is not None for source in sources],
        normal_force=[source.normal_force for source in sources],
        friction=[source.friction for source in sources],
        tangent=vectors("tangent"),
        has_tangent=[source.tangent is not None for source in sources],
        cap_tangent=vectors("cap_tangent"),
        has_cap_tangent=[source.cap_tangent is not None for source in sources],
        side=np.array([source.side for source in sources], dtype=np.int64),
        source_environment=environment,
        case_offsets=offsets,
        case_names=[case.name for case in cases],
        case_families=[case.family for case in cases],
        source_names=[source.name for source in sources],
        **extra,
    )
    writer.json(f"cases/{name}.json", api.cases_manifest(cases))
    return sources, environment, offsets


def _boundary_cases(api, centers):
    np = api.np
    cases, locations, boundary_indices, epsilons, sides = [], [], [], [], []
    for index, boundary in enumerate(api.boundary_locations(centers)):
        center, tangent = boundary["center"], boundary["tangent"]
        force = -0.1 * center + 0.025 * np.cross(center, tangent)
        locations.append(
            {
                "name": boundary["name"],
                "center": center.tolist(),
                "tangent": tangent.tolist(),
                "force": force.tolist(),
                "expected_ties": list(boundary["expected_ties"]),
            }
        )
        for epsilon in (1e-4, 1e-5, 1e-6):
            for side in (0, 1):
                direction = np.cos(epsilon) * center + (2 * side - 1) * np.sin(epsilon) * tangent
                name = f"{boundary['name']}-eps{epsilon:.0e}-side{side}"
                source = api.Source(direction, 0.002, force=force, cap_tangent=tangent, name=name)
                cases.append(api.Case(name, (source,), boundary["name"]))
                boundary_indices.append(index)
                epsilons.append(epsilon)
                sides.append(side)
    metadata = {
        "boundary_index": boundary_indices,
        "epsilon": epsilons,
        "crossing_side": sides,
        "pair_case_indices": np.arange(len(cases)).reshape((-1, 2)),
    }
    return tuple(cases), locations, metadata


def _relative_columns(np, value, reference):
    return np.linalg.norm(value - reference, axis=0) / np.maximum(np.linalg.norm(reference, axis=0), 1e-30)


def _export_loads(writer, name, mother, sources, environments, count, quadrature):
    api, np = writer.api, writer.api.np
    raw = api.assemble_sources(mother, sources, order=quadrature, audit=True)
    balanced = mother.balance(raw)
    directions, locations, forces = [], [], []
    for source in sources:
        sample_directions, _ = api.source_directions(source, quadrature)
        directions.append(sample_directions)
        locations.append(mother.geometry.locate(sample_directions))
        forces.append(api.force_for_source(mother, source))
    radial, angular, weights = api.cap_rule(quadrature)
    location_arrays = {
        field: np.stack([getattr(location, field) for location in locations])
        for field in ("nodes", "weights", "positions", "normals", "macro_ids", "element_ids", "local")
    }
    physical = location_arrays["positions"] + 0.5 * mother.model.h * np.array([s.side for s in sources])[
        :, None, None
    ] * (np.einsum("sqi,sqij->sqj", location_arrays["weights"], mother.geometry.directors[location_arrays["nodes"]]))
    forces = np.asarray(forces)
    expected_moment = np.einsum("sqj,q->sj", np.cross(physical, forces[:, None]), weights)
    nodal = raw.reshape((-1, 6, len(sources)))
    recovered_force = nodal[:, :3].sum(axis=0).T
    recovered_moment = (
        (np.cross(mother.geometry.vertices[:, :, None], nodal[:, :3], axisa=1, axisb=1, axisc=1) + nodal[:, 3:])
        .sum(axis=0)
        .T
    )
    writer.arrays(
        f"oracles/{name}/loads.npz",
        raw=raw,
        balanced=balanced,
        relief_acceleration=mother.relief_coefficients @ raw,
        source_force=forces,
        radial=radial,
        angular=angular,
        cap_weights=weights,
        cap_directions=np.stack(directions),
        **{f"location_{field}": value for field, value in location_arrays.items()},
        physical_positions=physical,
        expected_moment=expected_moment,
        recovered_force=recovered_force,
        recovered_moment=recovered_moment,
        raw_environment=api.aggregate_environments(raw, environments, count),
        balanced_environment=api.aggregate_environments(balanced, environments, count),
    )
    return raw, balanced


def _output(writer, prefix, mother, q, **extra):
    api, np = writer.api, writer.api.np
    local = mother.S @ q
    peak = api.sampled_maximum(local)
    resultants = mother.model.stress(q)
    writer.arrays(
        f"{prefix}/output.npz",
        q=q,
        stress_local=local,
        stress_full9=api.tensor_output(mother, local),
        peak=peak,
        N=resultants["N"],
        M=resultants["M"],
        Q=resultants["Q"],
        **extra,
    )
    return local, peak


def _export_bank(writer, name, cases, mother, bank, quadrature, case_metadata=None):
    api, np = writer.api, writer.api.np
    sources, environments, offsets = _export_cases(writer, name, cases, **(case_metadata or {}))
    raw, balanced = _export_loads(writer, name, mother, sources, environments, len(cases), quadrature)
    source_q = mother.solve(balanced)
    q_reference = api.aggregate_environments(source_q, environments, len(cases))
    environment_force = api.aggregate_environments(balanced, environments, len(cases))
    q_grouped = mother.solve(environment_force)
    mother_residual = mother.K @ q_grouped - environment_force
    mother_residual = np.linalg.norm(mother_residual[mother.free], axis=0) / np.maximum(
        np.linalg.norm(environment_force[mother.free], axis=0),
        1e-30,
    )
    writer.arrays(f"oracles/{name}/mother/source_response.npz", q=source_q, stress_local=mother.S @ source_q)
    reference, reference_peak = _output(
        writer,
        f"oracles/{name}/mother",
        mother,
        q_reference,
        q_grouped=q_grouped,
        unscaled_residual=mother_residual,
        grouped_vs_source_superposition=_relative_columns(np, q_grouped, q_reference),
    )
    cached = []
    for index, space in enumerate(bank.spaces):
        projected = space.project(balanced)
        z = space.solve_projected(projected)
        projected_raw = space.project_raw(raw)
        residual = np.linalg.norm(space.K @ z - projected, axis=0) / np.maximum(
            np.linalg.norm(projected, axis=0), 1e-30
        )
        writer.arrays(
            f"oracles/{name}/source_features/feature-{index:02d}.npz",
            projected_rhs=projected,
            z=z,
            projected_raw_rhs=projected_raw,
            unscaled_residual=residual,
            projected_relief_relative_difference=_relative_columns(np, projected_raw, projected),
        )
        cached.append(z)
    summaries, boundary_outputs = [], {}
    strategies = (*STRATEGIES, "top2") if name == "boundary18" else STRATEGIES
    for strategy in strategies:
        selection = np.zeros((len(sources), len(bank.spaces)))
        for source_index, source in enumerate(sources):
            indices, coefficients = bank.select(source, strategy)
            selection[source_index, indices] = coefficients
        group_features, group_environments, group_source_indptr, group_sources, group_weights = [], [], [0], [], []
        q, superposed = np.zeros_like(q_reference), np.zeros_like(q_reference)
        groups_per_case = np.zeros(len(cases), dtype=np.int64)
        contributions_per_case = np.zeros(len(cases), dtype=np.int64)
        coverage_failures, max_residual = [], 0.0
        for feature_index, space in enumerate(bank.spaces):
            positive = selection[:, feature_index] > 0
            active_environments = np.unique(environments[positive])
            grouped_force = np.zeros((mother.n_dof, len(active_environments)))
            covered = np.zeros(len(mother.geometry.vertices), dtype=bool)
            covered[mother.geometry.coarse_nodes] = True
            covered[space.local_nodes] = True
            for slot, environment in enumerate(active_environments):
                columns = np.flatnonzero(positive & (environments == environment))
                for column in columns:
                    coefficient = selection[column, feature_index]
                    grouped_force[:, slot] += coefficient * balanced[:, column]
                    superposed[:, environment] += coefficient * space.lift(cached[feature_index][:, column])
                    nodal = raw[:, column].reshape((-1, 6))
                    loaded = np.flatnonzero(np.max(np.abs(nodal), axis=1) > np.max(np.abs(nodal)) * 1e-14)
                    if not np.all(covered[loaded]):
                        coverage_failures.append(
                            {
                                "source": int(column),
                                "feature": feature_index,
                                "nodes": loaded[~covered[loaded]].tolist(),
                            }
                        )
                    group_sources.append(int(column))
                    group_weights.append(float(coefficient))
                group_source_indptr.append(len(group_sources))
                group_features.append(feature_index)
                group_environments.append(int(environment))
                groups_per_case[environment] += 1
                contributions_per_case[environment] += len(columns)
            projected = space.project(grouped_force)
            z = space.solve_projected(projected) if len(active_environments) else projected.copy()
            residual = np.linalg.norm(space.K @ z - projected, axis=0) / np.maximum(
                np.linalg.norm(projected, axis=0),
                1e-30,
            )
            max_residual = max(max_residual, float(residual.max(initial=0.0)))
            q[:, active_environments] += space.lift(z)
            writer.arrays(
                f"oracles/{name}/{strategy}/groups/feature-{feature_index:02d}.npz",
                environments=active_environments,
                grouped_force=grouped_force,
                projected_rhs=projected,
                z=z,
                unscaled_residual=residual,
            )
        local, peak = _output(writer, f"oracles/{name}/{strategy}", mother, q)
        error = api.relative_field_error(local, reference, api.field_weights(mother))
        peak_error = np.abs(peak - reference_peak) / np.maximum(reference_peak, 1e-30)
        superposition = _relative_columns(np, q, superposed)
        writer.arrays(
            f"oracles/{name}/{strategy}/selection.npz",
            weights=selection,
            group_features=np.array(group_features, dtype=np.int64),
            group_environments=np.array(group_environments, dtype=np.int64),
            group_source_indptr=np.array(group_source_indptr, dtype=np.int64),
            group_sources=np.array(group_sources, dtype=np.int64),
            group_weights=group_weights,
            groups_per_case=groups_per_case,
            contributions_per_case=contributions_per_case,
            relative_field_error=error,
            relative_peak_error=peak_error,
            grouped_vs_source_superposition=superposition,
        )
        loaded = np.diff(offsets) > 0
        released = ~loaded
        release_zero = bool(np.count_nonzero(local[:, released]) == 0 and np.count_nonzero(q[:, released]) == 0)
        summary = {
            "strategy": strategy,
            "loaded_cases": int(loaded.sum()),
            "mean_field_error": float(error[loaded].mean()),
            "worst_field_error": float(error[loaded].max()),
            "worst_peak_error": float(peak_error[loaded].max()),
            "mean_groups": float(groups_per_case[loaded].mean()),
            "maximum_groups": int(groups_per_case.max()),
            "mean_contributions_per_source": float((contributions_per_case[loaded] / np.diff(offsets)[loaded]).mean()),
            "worst_unscaled_residual": max_residual,
            "worst_grouped_vs_source_superposition": float(superposition.max()),
            "coverage_failures": coverage_failures,
            "release_exact_zero": release_zero,
        }
        summaries.append(summary)
        if coverage_failures or not release_zero or max_residual > 1e-9 or superposition.max() > 1e-9:
            writer.json(f"oracles/{name}/{strategy}/failure.json", summary)
            raise ArithmeticError(f"CPU oracle self-check failed for {name}/{strategy}.")
        if case_metadata:
            boundary_outputs[strategy] = (local, peak)
    if case_metadata:
        rows = []
        for left, right in case_metadata["pair_case_indices"]:
            for strategy, (local, peak) in boundary_outputs.items():
                peak_scale = max(float((reference_peak[left] + reference_peak[right]) / 2), 1e-30)
                rows.append(
                    {
                        "boundary": cases[left].family,
                        "epsilon_rad": float(case_metadata["epsilon"][left]),
                        "strategy": strategy,
                        "reference_relative_field_change": float(
                            api.relative_field_error(reference[:, right], reference[:, left], api.field_weights(mother))
                        ),
                        "relative_field_jump": float(
                            api.relative_field_error(local[:, right], local[:, left], api.field_weights(mother))
                        ),
                        "relative_sampled_peak_jump": float(abs(peak[right] - peak[left]) / peak_scale),
                    }
                )
        writer.json(f"oracles/{name}/boundary_jumps.json", rows)
    writer.json(f"oracles/{name}/summary.json", summaries)
    return summaries, offsets


def _export_alias(writer, name, published_cases, published_offsets, indices):
    np = writer.api.np
    cases = tuple(published_cases[index] for index in indices)
    _, _, offsets = _export_cases(
        writer,
        name,
        cases,
        reference_bank="published43",
        reference_case_indices=indices,
    )
    source_indices = np.concatenate(
        [np.arange(published_offsets[index], published_offsets[index + 1], dtype=np.int64) for index in indices]
    )
    writer.arrays(
        f"oracles/{name}/alias.npz",
        reference_bank="published43",
        case_indices=indices,
        source_indices=source_indices,
        case_offsets=offsets,
    )
    writer.json(
        f"oracles/{name}/alias.json",
        {
            "reference_bank": "published43",
            "reference_case_indices": [int(index) for index in indices],
            "meaning": "Repeated mathematical columns; no separate CPU timing or independently solved trajectories.",
            "output_case_axis": -1,
            "source_case_axis": -1,
            "groups": "Remap each referenced feature/environment group to the new environment index.",
        },
    )


def _sources(git_source: Path):
    paths = sorted(REFERENCE.glob("*.py")) + [
        REPOSITORY / "research/rigid_stress/shell_cpu/core.py",
        REPOSITORY / "research/rigid_stress/shell_cpu/geometry.py",
    ]
    rows = {}
    for path in paths:
        relative = str(path.relative_to(REPOSITORY))
        published = subprocess.check_output(["git", "-C", str(git_source), "show", f"{PRIMARY}:{relative}"])
        current = _sha256(path)
        primary = hashlib.sha256(published).hexdigest()
        rows[relative] = {"sha256": current, "primary_sha256": primary, "matches_primary": current == primary}
    if not all(row["matches_primary"] for row in rows.values()):
        raise RuntimeError("CPU reference files differ from the pinned primary source.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--git-source",
        type=Path,
        default=Path.cwd(),
        help="Pinned Git checkout for provenance; executed sources remain the captured analysis files.",
    )
    parser.add_argument("--ncoarse", type=int, default=3)
    parser.add_argument("--subdiv", type=int, default=4)
    parser.add_argument("--quadrature", type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Scientific execution requires a Slurm rtx-mid allocation with image 1_26.")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if partition and partition != "rtx-mid":
        parser.error(f"Expected Slurm rtx-mid, got {partition!r}.")
    if min(args.ncoarse, args.subdiv, args.quadrature) < 1:
        parser.error("ncoarse, subdiv and quadrature must be positive.")
    output = args.output.expanduser().resolve()
    if output.is_relative_to(REPOSITORY):
        parser.error("Keep large oracle arrays outside the source checkout; use the project data directory.")
    if output.exists() and any(output.iterdir()):
        parser.error("The output directory must be new or empty; existing evidence is never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    sources_before = _sources(args.git_source)
    api = _reference_api()
    writer = Writer(output, api)
    started = time.perf_counter()
    with api.threadpool_limits(limits=1):
        mother = api.Mother(ncoarse=args.ncoarse, subdiv=args.subdiv)
        mother_preparation_seconds = time.perf_counter() - started
        bank_started = time.perf_counter()
        bank = api.build_bank(mother, count=12)
        bank_preparation_seconds = time.perf_counter() - bank_started
        operators = _export_model(writer, mother, bank)
        published_cases = api.make_cases()
        summaries, offsets = _export_bank(writer, "published43", published_cases, mother, bank, args.quadrature)
        boundary_cases, locations, metadata = _boundary_cases(api, bank.centers)
        writer.json("cases/boundary_locations.json", locations)
        _export_bank(writer, "boundary18", boundary_cases, mother, bank, args.quadrature, metadata)
        _export_alias(writer, "static16", published_cases, offsets, api.np.array(STATIC_INDICES, dtype=api.np.int64))
        _export_alias(writer, "static64", published_cases, offsets, api.np.tile(STATIC_INDICES, 4))
    published_path = REFERENCE / "results/selection/results.json"
    published_bytes = subprocess.check_output(
        ["git", "-C", str(args.git_source), "show", f"{PRIMARY}:{published_path.relative_to(REPOSITORY)}"]
    )
    published = json.loads(published_bytes)
    fixture_hash = hashlib.sha256(
        json.dumps(api.cases_manifest(published_cases), sort_keys=True, indent=2).encode()
    ).hexdigest()
    if fixture_hash != published["fixture_sha256"]:
        raise AssertionError("Published source fixture manifest was not reproduced exactly.")
    matching = (args.ncoarse, args.subdiv, args.quadrature) == (3, 4, 8)
    comparisons = []
    for measured in summaries:
        expected = next(row for row in published["summary"] if row["strategy"] == measured["strategy"])
        errors = {
            key: abs(measured[key] - expected[key])
            for key in (
                "mean_field_error",
                "worst_field_error",
                "worst_peak_error",
                "mean_groups",
                "mean_contributions_per_source",
            )
        }
        comparisons.append(
            {
                "strategy": measured["strategy"],
                "absolute_differences": errors,
                "passed": not matching or max(errors.values()) <= 1e-9,
            }
        )
    writer.json(
        "published.json",
        {
            "source": str(published_path.relative_to(REPOSITORY)),
            "sha256": hashlib.sha256(published_bytes).hexdigest(),
            "matching_default_configuration": matching,
            "reference": published["summary"],
            "measured": summaries,
            "comparisons": comparisons,
            "absolute_tolerance": 1e-9,
            "published_fixture_sha256": published["fixture_sha256"],
            "exported_fixture_sha256": fixture_hash,
        },
    )
    writer.json("oracles/operators/inventory.json", operators)
    writer.json(
        "norms.json",
        {
            "stress_local_shape": "(side=2, element, Gauss=4, component=(xx,yy,xy), environment), flattened leading axes",
            "stress_full9_shape": "(side*element*Gauss, row-major tensor9, environment)",
            "native_comparison": "Native 2*element+side ordering and symmetric6 require explicit permutation/conversion.",
            "field": "sqrt(sum(w_gauss*(dxx^2+dyy^2+2*dxy^2)))/max(reference_weighted_norm,1e-30)",
            "peak": "max_over_both_sides_all_Gauss(sqrt(max(xx^2-xx*yy+yy^2+3*xy^2,0)))",
            "peak_error": "abs(peak-reference_peak)/max(reference_peak,1e-30)",
            "feature_residual": "norm(K_i*z-projected_rhs)/max(norm(projected_rhs),1e-30), unscaled physical operator",
            "mother_residual": "norm((K*q-balanced_rhs)[mother.free])/max(norm(balanced_rhs[mother.free]),1e-30)",
            "coverage": "all raw loaded physical nodes above max(abs(raw))*1e-14 lie in coarse_nodes union local_nodes",
            "release": "All q and recovered stress entries are exactly zero.",
            "boundary_field_jump": "Right-minus-left weighted field norm divided by left weighted field norm.",
            "boundary_peak_jump": "Absolute peak jump divided by mean of both matching mother peaks.",
        },
    )
    sources_after = _sources(args.git_source)
    if sources_before != sources_after:
        raise RuntimeError("CPU reference sources changed during the export.")
    passed = all(row["passed"] for row in comparisons)
    manifest = {
        "schema": SCHEMA,
        "primary_commit": PRIMARY,
        "checkout_commit": _git(args.git_source, "rev-parse", "HEAD"),
        "exporter": {"path": str(Path(__file__).relative_to(REPOSITORY)), "sha256": _sha256(Path(__file__))},
        "source_files": sources_after,
        "source_unchanged": True,
        "passed": passed,
        "configuration": {
            "ncoarse": args.ncoarse,
            "subdiv": args.subdiv,
            "quadrature": args.quadrature,
            "features": 12,
            "floating_dtype": "float64",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": api.np.__version__,
            "scipy": api.scipy.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": partition,
            "container": os.environ.get("APPTAINER_CONTAINER", os.environ.get("SINGULARITY_CONTAINER")),
            "required_container_image": "1_26",
            "host": platform.node(),
        },
        "seconds_including_export_io": time.perf_counter() - started,
        "mother_preparation_seconds": mother_preparation_seconds,
        "bank_preparation_seconds": bank_preparation_seconds,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "production_inputs": ["inputs/model.npz", "inputs/selection.npz", "cases/*.npz"],
        "validation_only": "Everything under oracles/; never feed CPU operators, loads or solutions to native timings.",
        "files": writer.inventory,
    }
    # The manifest inventories all payloads; its own hash is supplied by the job receipt.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output": str(output),
                "passed": passed,
                "files": len(writer.inventory),
                "seconds": manifest["seconds_including_export_io"],
            }
        )
    )
    if not passed:
        raise ArithmeticError("Frozen CPU outputs did not reproduce published default sanity numbers.")


if __name__ == "__main__":
    main()
