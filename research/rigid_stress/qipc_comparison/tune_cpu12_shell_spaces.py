"""Bounded material-cap shell adaptation after exact CPU12 GPU reproduction.

Compare bounded P6 spaces with the matched P3 baseline. The default two
controls retain P3's native local support, isolating coarse quality from local
space shrinkage. All candidates share the mother, cap law, pins and relief.
Run only in Slurm rtx-mid/image 1_26. This is not native DPR/Franka qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .benchmark_cpu12_gpu import Routes, _timing_api
from .check_cpu12_gpu import (
    Checks,
    _hash,
    _integer,
    _load,
    _matrix,
    _outputs,
    _reference_array,
    _scalar_vectors,
    _source_inputs,
    _transform,
    _vector,
    build_native_space,
)

STRATEGIES = ("continuous", "nearest", "narrow")
BUDGETS = {"field": 0.10, "peak": 0.05, "N": 0.10, "M": 0.10, "Q": 0.10, "mean_field": 0.05}
HELD_CONFIGS = ("P6-heldP3-support", "P6-heldP3-support+oneP6halo")
CONFIGURATIONS = (
    ("P3-matched", 3, 1, 1.0),
    ("P6-h0-c1.00", 6, 0, 1.0),
    ("P6-h0-c1.15", 6, 0, 1.15),
    ("P6-h1-c1.00", 6, 1, 1.0),
    ("P6-h1-c1.15", 6, 1, 1.15),
    (HELD_CONFIGS[0], 6, 0, 1.0),
    (HELD_CONFIGS[1], 6, 1, 1.0),
)


@dataclass(frozen=True)
class Topology:
    coarse_nodes: object
    parent_corners: object
    parent_ij: object
    macro_quads: object
    node_macro_indptr: object
    node_macro_indices: object
    macro_neighbor_indptr: object
    macro_neighbor_indices: object


@dataclass(frozen=True)
class FixedSupportLibrary:
    """Hold native P6 maps over explicitly controlled integer active/local masks."""

    maps: tuple
    local_nodes: tuple
    active_macros: tuple
    info: object


def _held_support(api, writer, checks, name, baseline, basis, native_template, topology, halo):
    """Transport actual native P3 support to children, with one optional integer halo."""
    np = api.np
    original = api.qd_to_numpy(baseline.library.info.active_macros)
    assert original.shape == (12, 54)
    child = np.arange(216)
    parent = (child // 36) * 9 + ((child % 36) // 6 // 2) * 3 + (child % 6) // 2
    assert np.array_equal(np.bincount(parent, minlength=54), np.full(54, 4))
    active = original[:, parent].astype(api.gs.np_int, copy=True)
    initial = active.copy()
    for _ in range(halo):
        expanded = active.copy()
        for feature, row in enumerate(active):
            for macro in np.flatnonzero(row):
                first, last = topology.macro_neighbor_indptr[macro : macro + 2]
                expanded[feature, topology.macro_neighbor_indices[first:last]] = 1
        active = expanded
    local = np.zeros((12, 866), dtype=api.gs.np_int)
    for feature in range(12):
        for node in range(866):
            first, last = topology.node_macro_indptr[node : node + 2]
            incident = topology.node_macro_indices[first:last]
            local[feature, node] = basis.coarse_index[node] < 0 and np.all(active[feature, incident] != 0)
        if halo == 0:
            held = baseline.library.local_nodes[feature]
            held = held[basis.coarse_index[held] < 0]
            checks.exact(f"{name}/feature-{feature:02d}/held_P3_local_nodes", np.flatnonzero(local[feature]), held)
    nodes = tuple(np.flatnonzero(row).astype(api.gs.np_int, copy=False) for row in local)
    active_macros = tuple(np.flatnonzero(row).astype(api.gs.np_int, copy=False) for row in active)
    info = replace(
        native_template.info,
        centers=baseline.library.info.centers,
        active_macros=_integer(api, active),
        local_mask=_integer(api, local),
    )
    writer.arrays(
        f"{name}/held_support.npz",
        P3_active_macros=original,
        child_to_P3_macro=parent,
        initial_P6_active_macros=initial,
        active_macros=active,
        local_mask=local,
    )
    writer.json(
        f"{name}/held_support.json",
        {
            "origin": "Actual native P3 active mask; each P3 macro has four P6 child macros.",
            "extra_P6_halo": halo,
            "centers": "The original P3 native QD center tensor is shared unchanged.",
            "local_rule": "Every incident P6 macro active, excluding new P6 coarse nodes.",
            "no_halo_control": "Old P3 bubble nodes filtered by the new P6 coarse set, independently checked.",
            "floating_preparation": "Native BilinearBasis and FeatureLibrary; host work copies integer masks/topology only.",
        },
    )
    return FixedSupportLibrary(tuple(basis.feature(row) for row in nodes), nodes, active_macros, info)


def _p6_topology(np, face_tables, n_nodes):
    """Rebuild integer first-registration Q4 parents on six-by-six macros per face."""
    step, ncoarse = 2, 6
    assert face_tables.shape == (6, 13, 13)
    coarse_nodes = np.unique(face_tables[:, ::step, ::step])
    coarse_index = np.full(n_nodes, -1, dtype=coarse_nodes.dtype)
    coarse_index[coarse_nodes] = np.arange(len(coarse_nodes))
    corners = np.full((n_nodes, 4), -1, dtype=coarse_nodes.dtype)
    ij = np.full((n_nodes, 2), -1, dtype=coarse_nodes.dtype)
    node_macros = [set() for _ in range(n_nodes)]
    macro_quads = []
    for table in face_tables:
        for j in range(ncoarse):
            for i in range(ncoarse):
                parent = np.array(
                    (
                        table[j * step, i * step],
                        table[j * step, (i + 1) * step],
                        table[(j + 1) * step, (i + 1) * step],
                        table[(j + 1) * step, i * step],
                    )
                )
                macro = len(macro_quads)
                macro_quads.append(coarse_index[parent])
                for sj in range(step + 1):
                    for si in range(step + 1):
                        node = table[j * step + sj, i * step + si]
                        node_macros[node].add(macro)
                        if corners[node, 0] < 0:
                            corners[node], ij[node] = parent, (si, sj)
    macro_quads = np.array(macro_quads)
    neighbors = [np.flatnonzero(np.any(np.isin(macro_quads, row), axis=1)) for row in macro_quads]
    incident = [np.array(sorted(row), dtype=coarse_nodes.dtype) for row in node_macros]
    node_indptr = np.r_[0, np.cumsum([len(row) for row in incident])]
    neighbor_indptr = np.r_[0, np.cumsum([len(row) for row in neighbors])]
    assert len(coarse_nodes) == 218 and len(macro_quads) == 216
    assert np.all(corners >= 0) and np.all(ij >= 0) and np.all(coarse_index[corners] >= 0)
    return Topology(
        coarse_nodes,
        corners,
        ij,
        macro_quads,
        node_indptr,
        np.concatenate(incident),
        neighbor_indptr,
        np.concatenate(neighbors),
    )


def _inventory(api, writer, checks, name, space, prepared, parent_maps=None):
    """Audit actual native maps, rank witnesses, work and projected common relief."""
    np = api.np
    model, basis, library = space.model, space.basis, space.library
    modes = api.qd_to_numpy(model.info.modes).reshape((-1, 6))
    relief = api.qd_to_numpy(model.info.relief).reshape((-1, 6))
    vertices = api.qd_to_numpy(model.info.vertices)[::2]
    prolongation = _transform(api, basis.transform, model.info.vertices.shape[0])
    coarse_dofs = (6 * basis.coarse_nodes[:, None] + np.arange(6)).reshape(-1)
    checks.close(f"{name}/P_rigid_modes", prolongation @ modes[coarse_dofs], modes)
    checks.close(f"{name}/P_geometry", prolongation[::6, ::6] @ vertices[basis.coarse_nodes], vertices)
    writer.matrix(f"{name}/operators/P.npz", prolongation)
    rows, matrices, free_indices = [], [], []
    for index, (mapping, operator, feature) in enumerate(
        zip(library.maps, prepared.operators, prepared.features, strict=True)
    ):
        free = np.flatnonzero(mapping.reduced_free.reshape(-1))
        transform = _transform(api, mapping, model.info.vertices.shape[0])[:, free]
        stiffness = _matrix(api, operator.info, young=model.options.young)[free][:, free]
        matrices.append(stiffness)
        free_indices.append(free)
        n_coarse_free = int(mapping.reduced_free[: 2 * len(basis.coarse_nodes)].sum())
        coarse_free = np.flatnonzero(mapping.reduced_free[: 2 * len(basis.coarse_nodes)].reshape(-1))
        coarse_rows = coarse_dofs[coarse_free]
        bubble_rows = (6 * library.local_nodes[index][:, None] + np.arange(6)).reshape(-1)
        identity_coarse = api.sparse.eye(n_coarse_free, format="csr")
        identity_bubble = api.sparse.eye(len(bubble_rows), format="csr")
        checks.close(
            f"{name}/feature-{index:02d}/coarse_rank_witness",
            transform[coarse_rows, :n_coarse_free],
            identity_coarse,
            rtol=0.0,
            atol=0.0,
        )
        checks.close(
            f"{name}/feature-{index:02d}/bubble_rank_witness",
            transform[bubble_rows, n_coarse_free:],
            identity_bubble,
            rtol=0.0,
            atol=0.0,
        )
        checks.condition(
            f"{name}/feature-{index:02d}/coarse_bubble_zero",
            transform[coarse_rows, n_coarse_free:].count_nonzero() == 0,
        )
        checks.condition(
            f"{name}/feature-{index:02d}/common_gauge_zero", transform[space.model_input["fixed"]].count_nonzero() == 0
        )
        containment = None
        if parent_maps is not None:
            parent_map = parent_maps[index]
            parent_free = np.flatnonzero(parent_map.reduced_free.reshape(-1))
            parent_transform = _transform(api, parent_map, model.info.vertices.shape[0])[:, parent_free]
            coarse_coefficients = parent_transform[coarse_rows]
            bubble_coefficients = (
                parent_transform[bubble_rows] - transform[bubble_rows, :n_coarse_free] @ coarse_coefficients
            )
            embedding = api.sparse.vstack((coarse_coefficients, bubble_coefficients), format="csr")
            checks.close(
                f"{name}/feature-{index:02d}/contains_P3_space",
                transform @ embedding,
                parent_transform,
                rtol=1e-12,
                atol=1e-15,
            )
            containment = checks.rows[-1]["passed"]
            writer.matrix(f"{name}/operators/feature-{index:02d}/P3_embedding.npz", embedding)
        projected_relief = api.qd_to_numpy(feature.projected_relief).reshape((-1, 6))[free]
        checks.close(f"{name}/feature-{index:02d}/common_relief", projected_relief, transform.T @ relief)
        checks.exact(
            f"{name}/feature-{index:02d}/free_mask", api.qd_to_numpy(operator.info.is_free), mapping.reduced_free
        )
        writer.matrix(f"{name}/operators/feature-{index:02d}/K.npz", stiffness)
        writer.matrix(f"{name}/operators/feature-{index:02d}/T.npz", transform)
        writer.arrays(
            f"{name}/operators/feature-{index:02d}/space.npz",
            free=free,
            local_nodes=library.local_nodes[index],
            active_macros=library.active_macros[index],
            projected_relief=projected_relief,
        )
        rows.append(
            {
                "index": index,
                "raw_dofs": mapping.n_reduced_blocks * 3,
                "free_dofs": len(free),
                "coarse_free_dofs": n_coarse_free,
                "bubble_nodes": len(library.local_nodes[index]),
                "active_macros": len(library.active_macros[index]),
                "rank": transform.shape[1],
                "rank_proof": "block-triangular cardinal coarse and identity bubble minor",
                "T_numerical_nonzeros": int(transform.count_nonzero()),
                "T_stored_scalar_entries": transform.nnz,
                "K_numerical_nonzeros": int(stiffness.count_nonzero()),
                "K_stored_scalar_entries": stiffness.nnz,
                "K_stored_blocks": operator.info.stiffness.shape[0],
                "factor_lower_blocks": operator.factor.info.lower.shape[0],
                "transform_entries": operator.n_transform_entries,
                "assembly_terms": operator.n_assembly_terms,
                "reduced_to_mother_free_ratio": len(free) / 5190,
                "P3_space_containment_passed": containment,
            }
        )
    writer.json(f"{name}/operators/inventory.json", rows)
    return rows, matrices, free_indices


def _relative_weighted(np, actual, reference, weights):
    actual, reference = actual.reshape((-1, actual.shape[-1])), reference.reshape((-1, reference.shape[-1]))
    scale = weights.reshape((-1, 1))
    absolute = np.linalg.norm(scale * (actual - reference), axis=0)
    denominator = np.linalg.norm(scale * reference, axis=0)
    return absolute / np.maximum(denominator, 1e-30), absolute, denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--pilot-repeats", type=int, choices=(0, 3), default=0)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(row[0] for row in CONFIGURATIONS),
        default=list(HELD_CONFIGS),
        help="Default: only the two new held-P3-support controls.",
    )
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific tuning requires Slurm rtx-mid with GENESIS_IMAGE_VER=1_26.")
    reference, output = args.reference.expanduser().resolve(), args.output.expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[3]) or (output.exists() and any(output.iterdir())):
        parser.error("Use a new/empty project data output directory outside the source checkout.")
    output.mkdir(parents=True, exist_ok=True)
    timing_api = _timing_api()
    api, np = timing_api.api, timing_api.api.np
    writer, checks = api.Writer(output, api), Checks(api)
    manifest = json.loads((reference / "manifest.json").read_text())
    assert manifest["passed"] and manifest["schema"] == "genesis.cpu12-reference.v1"
    for relative, identity in manifest["files"].items():
        assert _hash(reference / relative) == identity["sha256"], relative
    source_files = {
        str(path): _hash(path)
        for path in sorted((Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress").glob("*.py"))
    }
    for source in (
        Path(__file__),
        Path(build_native_space.__code__.co_filename),
        Path(Routes.run.__code__.co_filename),
    ):
        source_files[str(source.resolve())] = _hash(source)
    identity = {
        "kind": "cpu12_material_cap_shell_P6_adaptation_pilot",
        "source_files": source_files,
        "reference_manifest_sha256": _hash(reference / "manifest.json"),
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "precision": "64",
        "bank": "published43",
        "budgets": BUDGETS,
        "pilot_repeats": args.pilot_repeats,
        "configurations": args.configs,
        "scope": "Prescribed material-cap shell adaptation; native DPR, volume and complete Franka remain separate.",
        "baseline": "Matched P3 with twelve centers; P6 changes only coarse interpolation and macro-local supports.",
    }
    writer.json("start.json", identity)
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    setup_started = time.perf_counter()
    baseline = build_native_space(api, reference, output, output_mode="full")
    topology = _p6_topology(np, baseline.model_input["face_tables"], 866)
    assert np.isin(baseline.model_input["fixed"] // 6, topology.coarse_nodes).all()
    writer.arrays("P6_integer_topology.npz", **asdict(topology))
    cases = _load(np, reference / "cases/published43.npz")
    writer.arrays("cases.npz", **cases)
    n_envs, n_sources = len(cases["case_names"]), len(cases["radius"])
    assert n_envs == 43 and (np.diff(cases["case_offsets"]) > 0).sum() == 42
    sources = _source_inputs(api, cases)
    geometry = api.build_material_cap_geometry(
        baseline.model.info.vertices,
        _integer(api, baseline.model_input["face_tables"]),
        baseline.model.shell.directors,
        ncoarse=3,
        subdiv=4,
        vertex_stride=2,
    )
    cap = api.MaterialCap(
        int(manifest["configuration"]["quadrature"]),
        n_sources,
        866,
        ncoarse=3,
        subdiv=4,
        thickness=float(baseline.model_input["thickness"]),
        vertex_stride=2,
        audit=False,
    )
    mother_state = baseline.model.create_state(n_envs, "full")
    mother_valid = api.V(dtype=api.gs.qd_int, shape=(n_envs,))
    omega = _vector(api, np.zeros((n_envs, 3)))
    cap.assemble(geometry, sources)
    timing_api.aggregate(sources, cap.state, mother_state, mother_valid)
    baseline.model.recover(omega, mother_state)
    timing_api.admit(mother_valid, mother_state)
    mother = _outputs(api, baseline.model, mother_state)
    writer.arrays("mother/output.npz", **mother)
    checks.condition("mother/valid", mother["valid"].all() and api.qd_to_numpy(cap.state.valid).all())
    for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
        checks.close(
            f"mother/CPU_oracle/{key}",
            mother[key],
            _reference_array(api, reference, "published43", "mother/output.npz", key),
            rtol=1e-8,
        )
    checks.flush(writer)
    if not all(row["passed"] for row in checks.rows):
        raise ArithmeticError("The unchanged native mother failed outer independent CPU validation.")
    area = api.qd_to_numpy(baseline.model.shell.weights)
    component = np.array([1.0, 1.0, 2.0])
    weights = {
        "field": np.sqrt(np.tile(area.reshape(-1), 2)[:, None] * component).reshape(-1),
        "N": np.sqrt(area[..., None] * component),
        "M": np.sqrt(area[..., None] * component),
        "Q": np.sqrt(np.broadcast_to(area[..., None], (*area.shape, 2))),
    }
    loaded = np.diff(cases["case_offsets"]) > 0
    api.qd.sync()
    writer.json(
        "setup.json",
        {
            "seconds_including_first_mother_JIT": time.perf_counter() - setup_started,
            "physical_geometry_unchanged": True,
            "cap_geometry_ncoarse": 3,
            "cap_subdiv": 4,
            "P6_coarse_nodes": len(topology.coarse_nodes),
            "P6_macros": len(topology.macro_quads),
            "original_pins_are_P6_coarse": True,
        },
    )
    configurations = [row for row in CONFIGURATIONS if row[0] in args.configs]
    decisions = []
    for name, coarse_count, halo, multiplier in configurations:
        candidate_started = time.perf_counter()
        if coarse_count == 3:
            space = baseline
        else:
            basis = api.BilinearBasis(
                baseline.model, topology.coarse_nodes, topology.parent_corners, topology.parent_ij, 2
            )
            library = api.FeatureLibrary(
                basis,
                _vector(api, baseline.model_input["directions"]),
                _vector(api, baseline.model_input["directions"][topology.coarse_nodes]),
                topology.macro_quads,
                topology.node_macro_indptr,
                topology.node_macro_indices,
                topology.macro_neighbor_indptr,
                topology.macro_neighbor_indices,
                core_angle=float(baseline.selection_input["core_angle"]) * multiplier,
                support_angle=float(baseline.selection_input["support_angle"]),
                halo=0 if name in HELD_CONFIGS else halo,
            )
            if name in HELD_CONFIGS:
                library = _held_support(api, writer, checks, name, baseline, basis, library, topology, halo)
            space = replace(baseline, basis=basis, library=library)
        assert len(space.library.maps) == 12
        if any(mapping.reduced_free.sum() >= 5190 for mapping in space.library.maps):
            decisions.append(
                {
                    "configuration": name,
                    "admitted_reduced_space": False,
                    "reason": "A full-mother feature is excluded from this smaller-space pilot.",
                }
            )
            continue
        prepared = api.PreparedAtlas(space.model, space.library.maps, n_envs)
        inventory, matrices, free_indices = _inventory(
            api, writer, checks, name, space, prepared, baseline.library.maps if name in HELD_CONFIGS else None
        )
        selector = api.AtlasSelection(n_sources, n_envs, space.model.info.vertices.shape[0])
        coverage = timing_api.AtlasCoverage(space.basis.coarse_index, space.library.info.local_mask, n_sources, n_envs)
        state = space.model.create_state(n_envs, "full")
        api.qd.sync()
        writer.json(
            f"{name}/setup.json",
            {
                "seconds": time.perf_counter() - candidate_started,
                "atlas_timings": asdict(prepared.build_timings),
                "halo": halo,
                "core_multiplier": multiplier,
                "coarse_nodes": len(space.basis.coarse_nodes),
                "support_angle_unchanged": True,
                "support_construction": "held_P3_native_support" if name in HELD_CONFIGS else "native_P6_angular_core",
            },
        )
        for strategy in STRATEGIES:
            cap.assemble(geometry, sources)
            selected = selector.run(
                sources,
                cap.state,
                space.library.info.centers,
                strategy=strategy,
                support_angle=float(space.selection_input["support_angle"]),
                narrow_fraction=float(space.selection_input["narrow_fraction"]),
                maximum_radius=float(space.selection_input["maximum_radius"]),
            )
            covered = coverage.check(sources, cap.state, selected)
            active = api.qd_to_numpy(selected.active_groups)
            failures = int(api.qd_to_numpy(covered.coverage_failures))
            source_valid = api.qd_to_numpy(selected.source_valid)
            env_valid = api.qd_to_numpy(covered.environment_covered)
            prefix = f"{name}/{strategy}"
            writer.arrays(
                f"{prefix}/selection.npz",
                weights=api.qd_to_numpy(selected.weights).T,
                active_groups=active,
                environment_covered=env_valid,
                source_covered=api.qd_to_numpy(covered.source_covered),
                source_failure_count=api.qd_to_numpy(covered.source_failure_count),
                coverage_failures=failures,
                grouped_force=api.qd_to_numpy(selected.grouped_force),
            )
            decision = {
                "configuration": name,
                "strategy": strategy,
                "admitted_reduced_space": True,
                "coverage_failures": failures,
                "active_groups": int(active.sum()),
                "mean_loaded_groups": float(active[:, loaded].sum(axis=0).mean()),
                "source_feature_contributions": int(api.qd_to_numpy(selected.contribution_count)),
                "casewise_budget_passed": False,
                "mean_field_budget_passed": False,
                "numerical_passed": False,
                "source_coverage_passed": failures == 0,
            }
            if failures or not source_valid.all() or not env_valid.all():
                decision["decision"] = "unqualified_coverage_no_approximate_solve"
                decisions.append(decision)
                writer.json(f"{prefix}/decision.json", decision)
                continue
            check_start = len(checks.rows)
            prepared.recover(selected.grouped_force, state, covered.environment_covered)
            actual = _outputs(api, space.model, state)
            writer.arrays(f"{prefix}/output.npz", **actual)
            checks.condition(f"{prefix}/native_valid", actual["valid"].all())
            checks.close(f"{prefix}/same_current_raw", actual["force"], mother["force"], atol=1e-14)
            checks.close(f"{prefix}/common_balanced_load", actual["rhs"], mother["rhs"], atol=1e-14)
            finite = all(
                np.isfinite(actual[key]).all() for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q")
            )
            checks.condition(f"{prefix}/finite_outputs", finite)
            if not actual["valid"].all() or not finite:
                decision["decision"] = "native_numerical_rejection_no_spatial_qualification"
                decision["valid_environments"] = int(actual["valid"].sum())
                decisions.append(decision)
                writer.json(f"{prefix}/decision.json", decision)
                checks.flush(writer)
                continue
            for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
                checks.condition(f"{prefix}/release/{key}", np.count_nonzero(actual[key][..., ~loaded]) == 0)
            maximum_residual = 0.0
            for index, (feature, free, matrix) in enumerate(zip(prepared.states, free_indices, matrices, strict=True)):
                z, rhs = _scalar_vectors(api, feature.displacement)[free], _scalar_vectors(api, feature.rhs)[free]
                residual = np.linalg.norm(matrix @ z - rhs, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
                maximum_residual = max(maximum_residual, float(residual.max()))
                writer.arrays(
                    f"{prefix}/features/feature-{index:02d}.npz",
                    z=z,
                    projected_rhs=rhs,
                    unscaled_residual=residual,
                    valid=api.qd_to_numpy(feature.valid),
                )
                checks.condition(
                    f"{prefix}/feature-{index:02d}/unscaled_residual",
                    residual.max() <= 1e-8,
                    maximum=float(residual.max()),
                    tolerance=1e-8,
                )
            metrics, absolutes, denominators = {}, {}, {}
            for metric, key in (("field", "stress_local"), ("N", "N"), ("M", "M"), ("Q", "Q")):
                metrics[metric], absolutes[metric], denominators[metric] = _relative_weighted(
                    np, actual[key], mother[key], weights[metric]
                )
            metrics["peak"] = np.abs(actual["peak"] - mother["peak"]) / np.maximum(mother["peak"], 1e-30)
            writer.arrays(
                f"{prefix}/spatial.npz",
                **{f"relative_{key}": value for key, value in metrics.items()},
                **{f"absolute_{key}": value for key, value in absolutes.items()},
                **{f"reference_norm_{key}": value for key, value in denominators.items()},
            )
            worst = {key: float(value[loaded].max()) for key, value in metrics.items()}
            mean_field = float(metrics["field"][loaded].mean())
            numerical = all(row["passed"] for row in checks.rows[check_start:])
            operator_checks = [
                row
                for row in checks.rows
                if row["name"].startswith(f"{name}/feature-") or row["name"].startswith(f"{name}/P_")
            ]
            numerical = numerical and all(row["passed"] for row in operator_checks)
            group_per_feature = active.sum(axis=1)
            decision.update(
                {
                    "numerical_passed": numerical,
                    "maximum_unscaled_feature_residual": maximum_residual,
                    "worst_loaded_errors": worst,
                    "mean_loaded_field_error": mean_field,
                    "casewise_budget_passed": all(worst[key] <= BUDGETS[key] for key in worst),
                    "mean_field_budget_passed": mean_field <= BUDGETS["mean_field"],
                    "active_K_scalar_entries_times_rhs": int(
                        sum(
                            row["K_stored_scalar_entries"] * count
                            for row, count in zip(inventory, group_per_feature, strict=True)
                        )
                    ),
                    "active_factor_lower_blocks_times_rhs": int(
                        sum(
                            row["factor_lower_blocks"] * count
                            for row, count in zip(inventory, group_per_feature, strict=True)
                        )
                    ),
                    "active_lift_entries_times_rhs": int(
                        sum(
                            row["transform_entries"] * count
                            for row, count in zip(inventory, group_per_feature, strict=True)
                        )
                    ),
                }
            )
            decision["qualified_in_published43_only"] = (
                numerical and decision["casewise_budget_passed"] and decision["mean_field_budget_passed"]
            )
            decision["decision"] = (
                "published43_spatial_candidate"
                if decision["qualified_in_published43_only"]
                else "measured_negative_or_unqualified"
            )
            if args.pilot_repeats and numerical:
                route = Routes(
                    timing_api,
                    space,
                    sources,
                    geometry,
                    cap,
                    selector,
                    coverage,
                    prepared,
                    mother_state,
                    state,
                    mother_valid,
                    omega,
                    strategy,
                )
                for method in ("mother", "atlas"):
                    route.run(method)
                    api.qd.sync()
                timings = []
                for pair in range(args.pilot_repeats):
                    for order, method in enumerate(("mother", "atlas") if pair % 2 == 0 else ("atlas", "mother")):
                        api.qd.sync()
                        start = time.perf_counter()
                        route.run(method)
                        api.qd.sync()
                        seconds = time.perf_counter() - start
                        timings.append(
                            {
                                "pair": pair,
                                "order": order,
                                "method": method,
                                "seconds": seconds,
                                **route.snapshot(method),
                            }
                        )
                writer.json(f"{prefix}/pilot_timings.json", timings)
                decision["pilot_admission_passed"] = all(
                    row["valid_environments"] == n_envs
                    and row["invalid_sources"] == 0
                    and row["coverage_failures"] == 0
                    for row in timings
                )
                decision["pilot_scope"] = (
                    "Three paired fresh-cap whole-module probes; source-bound spatial status remains separate."
                )
            writer.json(f"{prefix}/decision.json", decision)
            decisions.append(decision)
            checks.flush(writer)
    checks.flush(writer)
    unchanged = all(_hash(path) == digest for path, digest in source_files.items())
    result = {
        **identity,
        "scientific_execution_completed": True,
        "source_unchanged": unchanged,
        "numerical_checks_passed": all(row["passed"] for row in checks.rows),
        "decisions": decisions,
        "spatial_candidates": [
            f"{row['configuration']}/{row['strategy']}"
            for row in decisions
            if row.get("qualified_in_published43_only", False)
        ],
        "native_DPR_qualified": False,
        "complete_scene_qualified": False,
        "files": writer.inventory,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "source_unchanged": unchanged,
                "numerical_checks_passed": result["numerical_checks_passed"],
                "published43_spatial_candidates": result["spatial_candidates"],
            }
        )
    )
    if not unchanged or not result["numerical_checks_passed"]:
        raise ArithmeticError(
            "Adaptation numerical invariants or source identity failed; spatial negatives are retained separately."
        )


if __name__ == "__main__":
    main()
