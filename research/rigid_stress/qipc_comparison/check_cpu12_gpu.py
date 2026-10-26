"""Compare native Quadrants CPU12 operators, material caps and complete fields.

Scientific execution requires Slurm rtx-mid/image 1_26. CPU arrays are read
only after native construction, for untimed independent comparisons. No CPU
operator, force, selector or solution is uploaded to native execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

STRATEGIES = ("continuous", "nearest", "narrow")
BANKS = ("published43", "boundary18", "static16", "static64")
SOURCE = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class NativeSpace:
    """Share one matched mother and its twelve maps across numerical and timing drivers."""

    model: object
    basis: object
    library: object
    model_input: dict
    selection_input: dict


def _api():
    import numpy as np
    import quadrants as qd
    from scipy import sparse

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.atlas_coverage import AtlasCoverage
    from genesis.engine.solvers.rigid.stress.atlas_selection import AtlasSelection
    from genesis.engine.solvers.rigid.stress.bilinear_basis import BilinearBasis
    from genesis.engine.solvers.rigid.stress.feature_library import FeatureLibrary
    from genesis.engine.solvers.rigid.stress.feature_operator import FeatureOperator
    from genesis.engine.solvers.rigid.stress.material_cap import (
        MaterialCap,
        allocate_material_cap_sources,
        build_material_cap_geometry,
    )
    from genesis.engine.solvers.rigid.stress.prepared_atlas import (
        PreparedAtlas,
        PreparedFeatureInfo,
        kernel_project_common_relief,
        kernel_sum_feature_forces,
    )
    from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
    from genesis.engine.solvers.rigid.stress.solve import kernel_balance
    from genesis.utils.array_class import V_MAT, V_VEC, V
    from genesis.utils.misc import qd_to_numpy

    from .export_cpu12_reference import Writer

    return SimpleNamespace(**locals())


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(np, path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


class Checks:
    """Use explicit whole-array L2/Frobenius tolerances, preserving every failure."""

    def __init__(self, api):
        self.api, self.rows = api, []

    def close(self, name, actual, expected, *, rtol=1e-10, atol=1e-13):
        np, sparse = self.api.np, self.api.sparse
        if sparse.issparse(actual) or sparse.issparse(expected):
            actual, expected = sparse.csr_matrix(actual), sparse.csr_matrix(expected)
            difference = (actual - expected).data
            reference = expected.data
        else:
            actual, expected = np.asarray(actual), np.asarray(expected)
            if actual.shape != expected.shape:
                self.rows.append(
                    {
                        "name": name,
                        "passed": False,
                        "actual_shape": list(actual.shape),
                        "expected_shape": list(expected.shape),
                    }
                )
                return
            difference, reference = actual - expected, expected
        delta, scale = float(np.linalg.norm(difference)), float(np.linalg.norm(reference))
        finite = bool(np.isfinite(delta) and np.isfinite(scale))
        self.rows.append(
            {
                "name": name,
                "passed": finite and delta <= atol + rtol * scale,
                "absolute_l2": delta if finite else None,
                "reference_l2": scale if finite else None,
                "relative_l2": delta / max(scale, 1e-30) if finite else None,
                "maximum_absolute": float(np.max(np.abs(difference), initial=0.0)) if finite else None,
                "rtol": rtol,
                "absolute_l2_floor": atol,
            }
        )

    def exact(self, name, actual, expected):
        np = self.api.np
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.rows.append(
            {
                "name": name,
                "passed": bool(np.array_equal(actual, expected)),
                "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape),
                "comparison": "exact",
            }
        )

    def condition(self, name, passed, **details):
        self.rows.append({"name": name, "passed": bool(passed), **details})

    def flush(self, writer):
        writer.json("checks.json", self.rows)


def _vector(api, values):
    values = api.np.asarray(values)
    tensor = api.V_VEC(values.shape[-1], dtype=api.gs.qd_float, shape=values.shape[:-1])
    tensor.from_numpy(values.astype(api.gs.np_float, copy=False))
    return tensor


def _integer(api, values):
    tensor = api.V(dtype=api.gs.qd_int, shape=values.shape)
    tensor.from_numpy(values.astype(api.gs.np_int, copy=False))
    return tensor


def build_native_space(api, reference, output, *, output_mode="full"):
    """Build only from frozen physical inputs and integer topology; never read CPU operators."""
    model_input = _load(api.np, reference / "inputs/model.npz")
    selection_input = _load(api.np, reference / "inputs/selection.npz")
    asset = output / "native-shell-input.npz"
    api.np.savez_compressed(
        asset,
        vertices=model_input["vertices"],
        quads=model_input["quads"],
        directors=model_input["directors"],
        gauge_dofs=model_input["fixed"],
    )
    options = api.gs.options.RigidStressOptions.dpr(
        mesh=asset,
        backend="shell",
        method="direct",
        output_mode=output_mode,
        factor_scaling=True,
        young=float(model_input["young"]),
        poisson=float(model_input["poisson"]),
        density=float(model_input["density"]),
        shell_thickness=float(model_input["thickness"]),
        shell_drilling_factor=float(model_input["drilling_factor"]),
        shell_gauge="translation",
        tolerance=1e-8,
        absolute_tolerance=1e-11,
        fused_pipeline=False,
        history_size=0,
    )
    model = api.ShellModel(options)
    basis = api.BilinearBasis(
        model,
        model_input["coarse_nodes"],
        model_input["parent_corners"],
        model_input["parent_ij"],
        int(model_input["subdiv"]),
    )
    library = api.FeatureLibrary(
        basis,
        _vector(api, model_input["directions"]),
        _vector(api, model_input["coarse_directions"]),
        model_input["macro_quads"],
        model_input["node_macro_indptr"],
        model_input["node_macro_indices"],
        model_input["macro_neighbor_indptr"],
        model_input["macro_neighbor_indices"],
        core_angle=float(selection_input["core_angle"]),
        support_angle=float(selection_input["support_angle"]),
        halo=1,
    )
    assert len(model_input["vertices"]) == 866 and len(model_input["quads"]) == 864
    assert model.info.vertices.shape[0] * 3 == 5196 and len(model_input["fixed"]) == 6
    assert len(library.maps) == 12, "Matched CPU12 construction requires exactly twelve maps."
    for index, mapping in enumerate(library.maps):
        assert mapping.n_reduced_blocks * 3 == 3522 and mapping.reduced_free.sum() == 3516, index
        assert len(library.local_nodes[index]) == 531 and len(library.active_macros[index]) == 37, index
    return NativeSpace(model, basis, library, model_input, selection_input)


def _scalar_vectors(api, tensor):
    values = api.qd_to_numpy(tensor)
    return values.transpose(0, 2, 1).reshape((-1, values.shape[1]))


def _matrix(api, info, *, young=1.0):
    return api.sparse.bsr_matrix(
        (
            api.qd_to_numpy(info.stiffness) * young,
            api.qd_to_numpy(info.columns),
            api.qd_to_numpy(info.row_start),
        )
    ).tocsr()


def _transform(api, mapping, n_mother):
    return api.sparse.bsr_matrix(
        (api.qd_to_numpy(mapping.values), mapping.columns, mapping.row_start),
        shape=(3 * n_mother, 3 * mapping.n_reduced_blocks),
    ).tocsr()


def _reference_array(api, reference, bank, relative, key, *, source=False):
    alias_path = reference / "oracles" / bank / "alias.npz"
    if alias_path.exists():
        alias = _load(api.np, alias_path)
        value = _load(api.np, reference / "oracles" / str(alias["reference_bank"]) / relative)[key]
        return value[..., alias["source_indices" if source else "case_indices"]]
    return _load(api.np, reference / "oracles" / bank / relative)[key]


def _operators(api, writer, checks, reference, model, basis, library, operators, prepared=None):
    np = api.np
    cpu = _load(np, reference / "oracles/operators/mother/numerics.npz")
    model_input = _load(np, reference / "inputs/model.npz")
    selection_input = _load(np, reference / "inputs/selection.npz")
    native_k = _matrix(api, model.info, young=model.options.young)
    writer.matrix("operators/mother_K.npz", native_k)
    checks.close(
        "operators/mother_K", native_k, api.sparse.load_npz(reference / "oracles/operators/mother/K.npz"), atol=1e-10
    )
    native_p = _transform(api, basis.transform, model.info.vertices.shape[0])
    writer.matrix("operators/P.npz", native_p)
    checks.close("operators/P", native_p, api.sparse.load_npz(reference / "oracles/operators/P.npz"))
    native = {
        "rigid_modes": api.qd_to_numpy(model.info.modes).reshape((-1, 6)),
        "mass_modes": api.qd_to_numpy(model.info.mass_modes).reshape((-1, 6)),
        "mass_gram": api.qd_to_numpy(model.info.gram),
        "gram_inverse": api.qd_to_numpy(model.info.gram_inverse),
        "relief": api.qd_to_numpy(model.info.relief).reshape((-1, 6)),
        "nodal_mass": api.qd_to_numpy(model.shell.nodal_mass),
        "axes": api.qd_to_numpy(model.shell.axes),
        "gauss_weights": api.qd_to_numpy(model.shell.weights),
        "B_membrane": api.qd_to_numpy(model.shell.membrane).transpose(0, 1, 3, 2),
        "B_bending": api.qd_to_numpy(model.shell.bending).transpose(0, 1, 3, 2),
        "B_shear": api.qd_to_numpy(model.shell.shear).transpose(0, 1, 3, 2),
        "directors": api.qd_to_numpy(model.shell.directors),
        "free": np.flatnonzero(api.qd_to_numpy(model.info.is_free).reshape(-1)),
        "pins": api.qd_to_numpy(model.info.pins),
        "centers": api.qd_to_numpy(library.info.centers),
        "macro_centers": api.qd_to_numpy(library.info.macro_centers),
        "macro_radii": api.qd_to_numpy(library.info.macro_radii),
        "active_macros": api.qd_to_numpy(library.info.active_macros),
        "local_mask": api.qd_to_numpy(library.info.local_mask),
    }
    writer.arrays("operators/mother_numerics.npz", **native)
    for name in (
        "rigid_modes",
        "mass_modes",
        "mass_gram",
        "nodal_mass",
        "axes",
        "gauss_weights",
        "B_membrane",
        "B_bending",
        "B_shear",
    ):
        checks.close(f"operators/{name}", native[name], cpu[name], atol=1e-15)
    gram_inverse = np.linalg.inv(cpu["mass_gram"])
    checks.close("operators/gram_inverse", native["gram_inverse"], gram_inverse)
    checks.close("operators/relief", native["relief"], cpu["mass_modes"] @ gram_inverse)
    checks.close(
        "operators/relief_coefficients", native["gram_inverse"] @ native["rigid_modes"].T, cpu["relief_coefficients"]
    )
    checks.exact("operators/free", native["free"], cpu["free"])
    checks.exact("operators/pins", np.sort(native["pins"]), np.sort(model_input["fixed"]))
    for key in ("directors", "macro_centers", "macro_radii"):
        checks.close(f"operators/{key}", native[key], model_input[key])
    checks.close("operators/centers", native["centers"], selection_input["centers"])
    free_indices = []
    for index, (mapping, operator) in enumerate(zip(library.maps, operators, strict=True)):
        prefix = reference / f"oracles/operators/feature-{index:02d}"
        cpu_space = _load(np, prefix / "space.npz")
        free = np.flatnonzero(mapping.reduced_free.reshape(-1))
        free_indices.append(free)
        transform = _transform(api, mapping, model.info.vertices.shape[0])[:, free]
        stiffness = _matrix(api, operator.info, young=model.options.young)[free][:, free]
        writer.matrix(f"operators/feature-{index:02d}/T.npz", transform)
        writer.matrix(f"operators/feature-{index:02d}/K.npz", stiffness)
        checks.close(f"feature-{index:02d}/T", transform, api.sparse.load_npz(prefix / "T.npz"))
        checks.close(f"feature-{index:02d}/K", stiffness, api.sparse.load_npz(prefix / "K.npz"), atol=1e-10)
        checks.exact(f"feature-{index:02d}/local_nodes", library.local_nodes[index], cpu_space["local_nodes"])
        checks.exact(
            f"feature-{index:02d}/active_macros",
            native["active_macros"][index].astype(bool),
            cpu_space["active_macros"],
        )
        checks.exact(f"feature-{index:02d}/free_count", len(free), api.sparse.load_npz(prefix / "T.npz").shape[1])
        checks.exact(f"feature-{index:02d}/operator_free", api.qd_to_numpy(operator.info.is_free), mapping.reduced_free)
        feature = (
            prepared.features[index]
            if prepared is not None
            else api.PreparedFeatureInfo(
                projected_relief=api.V_MAT(3, 6, dtype=api.gs.qd_float, shape=(operator.info.vertices.shape[0],))
            )
        )
        if prepared is None:
            api.kernel_project_common_relief(feature, model.info, operator.transform)
        projected = api.qd_to_numpy(feature.projected_relief).reshape((-1, 6))[free]
        writer.arrays(f"operators/feature-{index:02d}/relief.npz", projected_relief=projected, free=free)
        checks.close(f"feature-{index:02d}/projected_relief", projected, cpu_space["relief_projection"] @ gram_inverse)
    checks.flush(writer)
    return free_indices


def _source_inputs(api, cases):
    sources = api.allocate_material_cap_sources(len(cases["radius"]))
    for field, key, integer in (
        (sources.direction, "direction", False),
        (sources.radius, "radius", False),
        (sources.force, "force", False),
        (sources.normal_force, "normal_force", False),
        (sources.friction, "friction", False),
        (sources.tangent, "tangent", False),
        (sources.cap_tangent, "cap_tangent", False),
        (sources.side, "side", True),
        (sources.environment, "source_environment", True),
        (sources.explicit_force, "has_force", True),
        (sources.has_tangent, "has_tangent", True),
        (sources.has_cap_tangent, "has_cap_tangent", True),
    ):
        field.from_numpy(cases[key].astype(api.gs.np_int if integer else api.gs.np_float))
    return sources


def _loads(api, writer, checks, reference, bank, model, library, model_input, selection_input, quadrature):
    np = api.np
    cases = _load(np, reference / f"cases/{bank}.npz")
    n_sources, n_envs = len(cases["radius"]), len(cases["case_names"])
    sources = _source_inputs(api, cases)
    geometry = api.build_material_cap_geometry(
        model.info.vertices,
        _integer(api, model_input["face_tables"]),
        model.shell.directors,
        ncoarse=int(model_input["ncoarse"]),
        subdiv=int(model_input["subdiv"]),
        vertex_stride=2,
    )
    cap = api.MaterialCap(
        quadrature,
        n_sources,
        len(model_input["vertices"]),
        ncoarse=int(model_input["ncoarse"]),
        subdiv=int(model_input["subdiv"]),
        thickness=float(model_input["thickness"]),
        vertex_stride=2,
        audit=True,
    )
    cap.assemble(geometry, sources)
    native = {
        key: api.qd_to_numpy(field)
        for key, field in (
            ("valid", cap.state.valid),
            ("failed_samples", cap.state.failed_samples),
            ("scanned_samples", cap.state.scanned_samples),
            ("force", cap.state.force),
            ("resultant_force", cap.state.resultant_force),
            ("resultant_moment", cap.state.resultant_moment),
            ("expected_moment", cap.state.expected_moment),
            ("sample_directions", cap.state.sample_directions),
            ("sample_positions", cap.state.sample_positions),
            ("sample_nodes", cap.state.sample_nodes),
            ("sample_shape", cap.state.sample_shape),
            ("sample_macro", cap.state.sample_macro),
            ("sample_element", cap.state.sample_element),
            ("sample_local", cap.state.sample_local),
            ("sample_valid", cap.state.sample_valid),
        )
    }
    native["raw"] = _scalar_vectors(api, cap.state.raw_force)
    native["rule_radial"] = api.qd_to_numpy(cap.rule.radial)
    native["rule_weights"] = api.qd_to_numpy(cap.rule.weights)
    writer.arrays(f"{bank}/loads.npz", **native, macro_coefficients=api.qd_to_numpy(geometry.macro_coefficients))
    checks.condition(f"{bank}/cap_rule_valid", api.qd_to_numpy(cap.rule.valid).all())
    checks.condition(f"{bank}/source_valid", native["valid"].all())
    checks.condition(f"{bank}/sample_valid", native["sample_valid"].all())
    checks.exact(f"{bank}/failed_samples", native["failed_samples"], np.zeros(n_sources, dtype=int))
    checks.close(
        f"{bank}/macro_coefficients", api.qd_to_numpy(geometry.macro_coefficients), model_input["macro_coefficients"]
    )
    # Aliased static banks have source axes both first (samples) and last (raw).
    alias_path = reference / f"oracles/{bank}/alias.npz"
    oracle_bank, source_indices = bank, np.arange(n_sources)
    if alias_path.exists():
        alias = _load(np, alias_path)
        oracle_bank, source_indices = str(alias["reference_bank"]), alias["source_indices"]
    cpu = _load(np, reference / f"oracles/{oracle_bank}/loads.npz")
    checks.close(f"{bank}/raw", native["raw"], cpu["raw"][:, source_indices], atol=1e-14)
    comparisons = {
        "force": "source_force",
        "resultant_force": "recovered_force",
        "resultant_moment": "recovered_moment",
        "expected_moment": "expected_moment",
        "sample_directions": "cap_directions",
        "sample_positions": "location_positions",
        "sample_shape": "location_weights",
        "sample_local": "location_local",
    }
    for native_key, cpu_key in comparisons.items():
        checks.close(f"{bank}/{native_key}", native[native_key], cpu[cpu_key][source_indices], atol=1e-14)
    for native_key, cpu_key in (
        ("sample_nodes", "location_nodes"),
        ("sample_macro", "location_macro_ids"),
        ("sample_element", "location_element_ids"),
    ):
        checks.exact(f"{bank}/{native_key}", native[native_key], cpu[cpu_key][source_indices])
    checks.close(f"{bank}/rule_radial", native["rule_radial"], cpu["radial"][:: 4 * quadrature])
    checks.close(
        f"{bank}/rule_weights", native["rule_weights"], cpu["cap_weights"][:: 4 * quadrature] * (4 * quadrature)
    )
    selection = api.AtlasSelection(n_sources, n_envs, model.info.vertices.shape[0])
    checks.flush(writer)
    if not native["valid"].all() or not native["sample_valid"].all() or native["failed_samples"].any():
        raise ArithmeticError(f"Native material cap rejected source/sample inputs for {bank}; no partial load solve.")
    if not api.qd_to_numpy(cap.rule.valid).all():
        raise ArithmeticError(f"Native material-cap quadrature rule is invalid for {bank}.")
    return cases, sources, cap, selection


def _tensor9(np, components):
    """Input (...,6,B), output (...,3,3,B), symmetric off-diagonals explicit."""
    tensor = np.zeros((*components.shape[:-2], 3, 3, components.shape[-1]))
    for index, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        tensor[..., a, b, :] = components[..., index, :]
        tensor[..., b, a, :] = components[..., index, :]
    return tensor


def _outputs(api, model, state):
    np = api.np
    n_elements, batch = model.n_shell_elements, state.active.shape[0]
    axes = api.qd_to_numpy(model.shell.axes)
    components = api.qd_to_numpy(state.stress_tensor).reshape((n_elements, 2, 4, batch, 6)).transpose(1, 0, 2, 4, 3)
    tensor = _tensor9(np, components)
    first, second = axes[..., 0], axes[..., 1]
    local = np.stack(
        [
            np.einsum("egi,segijb,egj->segb", first, tensor, first),
            np.einsum("egi,segijb,egj->segb", second, tensor, second),
            np.einsum("egi,segijb,egj->segb", first, tensor, second),
        ],
        axis=3,
    ).reshape((-1, batch))
    output = {
        "q": _scalar_vectors(api, state.displacement),
        "rhs": _scalar_vectors(api, state.rhs),
        "force": _scalar_vectors(api, state.force),
        "stress_local": local,
        "stress_full9": tensor.reshape((-1, 9, batch)),
        "peak": api.qd_to_numpy(state.peak),
        "valid": api.qd_to_numpy(state.valid),
    }
    for name, field in (("N", state.membrane_resultant), ("M", state.bending_resultant)):
        body = _tensor9(np, api.qd_to_numpy(field).transpose(0, 1, 3, 2))
        output[name] = np.stack(
            [
                np.einsum("egi,egijb,egj->egb", first, body, first),
                np.einsum("egi,egijb,egj->egb", second, body, second),
                np.einsum("egi,egijb,egj->egb", first, body, second),
            ],
            axis=2,
        )
    transverse = api.qd_to_numpy(state.transverse_resultant).transpose(0, 1, 3, 2)
    output["Q"] = np.einsum("egij,egib->egjb", axes[..., :2], transverse)
    return output


def _selection_reference(api, reference, bank, strategy):
    np = api.np
    alias_path = reference / f"oracles/{bank}/alias.npz"
    oracle_bank, case_indices = bank, None
    if alias_path.exists():
        alias = _load(np, alias_path)
        oracle_bank, case_indices = str(alias["reference_bank"]), alias["case_indices"]
    selected = _load(np, reference / f"oracles/{oracle_bank}/{strategy}/selection.npz")
    if case_indices is not None:
        selected["weights"] = selected["weights"][alias["source_indices"]]
        for key in ("groups_per_case", "contributions_per_case", "relative_field_error", "relative_peak_error"):
            selected[key] = selected[key][case_indices]
    return selected, oracle_bank, case_indices


def _check_output(api, writer, checks, reference, bank, strategy, model, state, cases, output_prefix=None):
    np = api.np
    actual = _outputs(api, model, state)
    if strategy == "mother":
        cpu_k = api.sparse.load_npz(reference / "oracles/operators/mother/K.npz")
        free = _load(np, reference / "oracles/operators/mother/numerics.npz")["free"]
        residual = (cpu_k @ actual["q"] - actual["rhs"])[free]
        actual["unscaled_residual"] = np.linalg.norm(residual, axis=0) / np.maximum(
            np.linalg.norm(actual["rhs"][free], axis=0), 1e-30
        )
        checks.condition(
            f"{bank}/mother/unscaled_residual",
            actual["unscaled_residual"].max() <= 1e-8,
            maximum=float(actual["unscaled_residual"].max()),
            tolerance=1e-8,
        )
    writer.arrays(f"{output_prefix or f'{bank}/{strategy}'}/output.npz", **actual)
    for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
        expected = _reference_array(api, reference, bank, f"{strategy}/output.npz", key)
        checks.close(f"{bank}/{strategy}/{key}", actual[key], expected, rtol=1e-8, atol=1e-13)
    checks.condition(f"{bank}/{strategy}/valid", actual["valid"].all())
    released = np.diff(cases["case_offsets"]) == 0
    for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
        checks.condition(f"{bank}/{strategy}/release/{key}", np.count_nonzero(actual[key][..., released]) == 0)
    return actual


def _bank(
    api, writer, checks, reference, bank, model, library, model_input, selection_input, quadrature, stage, prepared=None
):
    np = api.np
    cases, sources, cap, selector = _loads(
        api, writer, checks, reference, bank, model, library, model_input, selection_input, quadrature
    )
    n_envs = len(cases["case_names"])
    state = model.create_state(n_envs, "full")
    omega = _vector(api, np.zeros((n_envs, 3)))
    if stage == "all" and prepared is None:
        prepared = api.PreparedAtlas(model, library.maps, n_envs)
    coarse_index = api.np.full(len(model_input["vertices"]), -1, dtype=api.gs.np_int)
    coarse_index[model_input["coarse_nodes"]] = api.np.arange(len(model_input["coarse_nodes"]))
    coverage = api.AtlasCoverage(coarse_index, library.info.local_mask, len(cases["radius"]), n_envs)
    axes_weights = _load(np, reference / "oracles/operators/mother/numerics.npz")["field_weights"]
    for strategy in STRATEGIES:
        selected = selector.run(
            sources,
            cap.state,
            library.info.centers,
            strategy=strategy,
            support_angle=float(selection_input["support_angle"]),
            narrow_fraction=float(selection_input["narrow_fraction"]),
            maximum_radius=float(selection_input["maximum_radius"]),
        )
        expected, oracle_bank, case_indices = _selection_reference(api, reference, bank, strategy)
        weights = api.qd_to_numpy(selected.weights).T
        source_valid = api.qd_to_numpy(selected.source_valid)
        environment_valid = api.qd_to_numpy(selected.environment_valid)
        invalid_source_count = api.qd_to_numpy(selected.invalid_source_count)
        active_groups = api.qd_to_numpy(selected.active_groups)
        group_counts = active_groups.sum(axis=0)
        contributions = np.zeros(n_envs, dtype=int)
        np.add.at(contributions, cases["source_environment"], (weights > 0).sum(axis=1))
        writer.arrays(
            f"{bank}/{strategy}/selection.npz",
            weights=weights,
            active_groups=active_groups,
            grouped_force=api.qd_to_numpy(selected.grouped_force),
            groups_per_case=group_counts,
            contributions_per_case=contributions,
            source_valid=source_valid,
            environment_valid=environment_valid,
            invalid_source_count=invalid_source_count,
        )
        checks.close(f"{bank}/{strategy}/weights", weights, expected["weights"], atol=1e-13)
        checks.exact(f"{bank}/{strategy}/groups", group_counts, expected["groups_per_case"])
        checks.exact(f"{bank}/{strategy}/contributions", contributions, expected["contributions_per_case"])
        checks.condition(f"{bank}/{strategy}/sources_valid", source_valid.all())
        checks.condition(f"{bank}/{strategy}/environments_valid", environment_valid.all())
        checks.exact(f"{bank}/{strategy}/invalid_source_count", invalid_source_count, 0)
        checks.flush(writer)
        if not source_valid.all() or not environment_valid.all() or invalid_source_count != 0:
            raise ArithmeticError(f"Native selector rejected {bank}/{strategy}; refusing a partial-load solve.")
        covered = coverage.check(sources, cap.state, selected)
        source_covered = api.qd_to_numpy(covered.source_covered)
        environment_covered = api.qd_to_numpy(covered.environment_covered)
        coverage_failures = api.qd_to_numpy(covered.coverage_failures)
        writer.arrays(
            f"{bank}/{strategy}/coverage.npz",
            source_covered=source_covered,
            environment_covered=environment_covered,
            coverage_failures=coverage_failures,
            source_failure_count=api.qd_to_numpy(covered.source_failure_count),
        )
        checks.condition(f"{bank}/{strategy}/source_coverage", source_covered.all())
        checks.condition(f"{bank}/{strategy}/environment_coverage", environment_covered.all())
        checks.exact(f"{bank}/{strategy}/coverage_failures", coverage_failures, 0)
        checks.flush(writer)
        if not source_covered.all() or not environment_covered.all() or coverage_failures != 0:
            raise ArithmeticError(f"Native coverage rejected {bank}/{strategy}; no approximate solve admitted.")
        api.kernel_sum_feature_forces(selected.grouped_force, state)
        api.kernel_balance(omega, state, model.info, model.options.cooperative_balance)
        native_raw, native_balanced = _scalar_vectors(api, state.force), _scalar_vectors(api, state.rhs)
        for key, value in (("raw_environment", native_raw), ("balanced_environment", native_balanced)):
            checks.close(
                f"{bank}/{strategy}/{key}", value, _reference_array(api, reference, bank, "loads.npz", key), atol=1e-14
            )
        writer.arrays(f"{bank}/{strategy}/common_load.npz", raw=native_raw, balanced=native_balanced)
        if stage != "all":
            checks.flush(writer)
            continue
        prepared.recover(selected.grouped_force, state, covered.environment_covered)
        atlas_output = _check_output(api, writer, checks, reference, bank, strategy, model, state, cases)
        for index, (mapping, feature_state) in enumerate(zip(library.maps, prepared.states, strict=True)):
            cpu = _load(np, reference / f"oracles/{oracle_bank}/{strategy}/groups/feature-{index:02d}.npz")
            free = np.flatnonzero(mapping.reduced_free.reshape(-1))
            z, projected = (
                _scalar_vectors(api, feature_state.displacement)[free],
                _scalar_vectors(api, feature_state.rhs)[free],
            )
            expected_z = np.zeros_like(z)
            expected_rhs = np.zeros_like(projected)
            for target in range(n_envs):
                original = int(case_indices[target]) if case_indices is not None else target
                found = np.flatnonzero(cpu["environments"] == original)
                if found.size:
                    expected_z[:, target] = cpu["z"][:, found[0]]
                    expected_rhs[:, target] = cpu["projected_rhs"][:, found[0]]
            cpu_k = api.sparse.load_npz(reference / f"oracles/operators/feature-{index:02d}/K.npz")
            residual = np.linalg.norm(cpu_k @ z - projected, axis=0) / np.maximum(
                np.linalg.norm(projected, axis=0), 1e-30
            )
            writer.arrays(
                f"{bank}/{strategy}/features/feature-{index:02d}.npz",
                z=z,
                projected_rhs=projected,
                unscaled_cpu_operator_residual=residual,
                valid=api.qd_to_numpy(feature_state.valid),
            )
            checks.close(f"{bank}/{strategy}/feature-{index:02d}/rhs", projected, expected_rhs, atol=1e-14)
            checks.close(f"{bank}/{strategy}/feature-{index:02d}/z", z, expected_z, rtol=1e-8, atol=1e-14)
            checks.condition(
                f"{bank}/{strategy}/feature-{index:02d}/unscaled_residual",
                np.max(residual) <= 1e-8,
                maximum=float(np.max(residual)),
                tolerance=1e-8,
            )
        # Use exactly the native raw force already summed by PreparedAtlas.
        # Its merged approximate displacement is recorded before the mother solve.
        model.recover(omega, state)
        mother_output = _check_output(
            api,
            writer,
            checks,
            reference,
            bank,
            "mother",
            model,
            state,
            cases,
            output_prefix=f"{bank}/{strategy}/mother",
        )
        checks.close(
            f"{bank}/{strategy}/same_raw_mother_atlas",
            mother_output["force"],
            atlas_output["force"],
            rtol=0.0,
            atol=0.0,
        )
        difference = axes_weights[:, None] * (atlas_output["stress_local"] - mother_output["stress_local"])
        denominator = np.maximum(np.linalg.norm(axes_weights[:, None] * mother_output["stress_local"], axis=0), 1e-30)
        spatial_field = np.linalg.norm(difference, axis=0) / denominator
        spatial_peak = np.abs(atlas_output["peak"] - mother_output["peak"]) / np.maximum(mother_output["peak"], 1e-30)
        writer.arrays(
            f"{bank}/{strategy}/spatial.npz", relative_field_error=spatial_field, relative_peak_error=spatial_peak
        )
        checks.close(f"{bank}/{strategy}/spatial_field", spatial_field, expected["relative_field_error"], rtol=1e-8)
        checks.close(f"{bank}/{strategy}/spatial_peak", spatial_peak, expected["relative_peak_error"], rtol=1e-8)
        checks.flush(writer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--bank", choices=(*BANKS, "all"), default="published43")
    parser.add_argument("--stage", choices=("operators", "loads", "all"), default="all")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires Slurm rtx-mid and image 1_26.")
    output, reference = args.output.expanduser().resolve(), args.reference.expanduser().resolve()
    if output.is_relative_to(SOURCE) or (output.exists() and any(output.iterdir())):
        parser.error("Use a new/empty project data output directory outside the source checkout.")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = reference / "manifest.json"
    reference_manifest = json.loads(manifest_path.read_text())
    if not reference_manifest["passed"] or reference_manifest["schema"] != "genesis.cpu12-reference.v1":
        parser.error("The CPU reference must be a successful v1 export.")
    for relative, identity in reference_manifest["files"].items():
        if _hash(reference / relative) != identity["sha256"]:
            raise ArithmeticError(f"CPU frozen artifact changed: {relative}")
    api = _api()
    writer, checks = api.Writer(output, api), Checks(api)
    started = time.perf_counter()
    banks = BANKS if args.bank == "all" else (args.bank,)
    native_source = Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress"
    source_files = {str(path): _hash(path) for path in sorted(native_source.glob("*.py"))}
    source_files[str(Path(__file__).resolve())] = _hash(Path(__file__))
    identity = {
        "reference_manifest_sha256": _hash(manifest_path),
        "stage": args.stage,
        "banks": list(banks),
        "source_files": source_files,
        "native_package": str(api.gs.__file__),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "precision": "64",
        "numerical_oracles_outside_timings": True,
        "performance_qualification": "This driver is an intrusive numerical audit, not a timing benchmark.",
    }
    writer.json("start.json", identity)
    try:
        api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
        space = build_native_space(api, reference, output)
        model, basis, library = space.model, space.basis, space.library
        model_input, selection_input = space.model_input, space.selection_input
        prepared = None
        if args.stage == "all":
            n_envs = len(_load(api.np, reference / f"cases/{banks[0]}.npz")["case_names"])
            prepared = api.PreparedAtlas(model, library.maps, n_envs)
        if args.stage in ("operators", "all"):
            operators = (
                prepared.operators if prepared else tuple(api.FeatureOperator(model.info, *m) for m in library.maps)
            )
            _operators(api, writer, checks, reference, model, basis, library, operators, prepared)
        if args.stage in ("loads", "all"):
            for index, bank in enumerate(banks):
                _bank(
                    api,
                    writer,
                    checks,
                    reference,
                    bank,
                    model,
                    library,
                    model_input,
                    selection_input,
                    int(reference_manifest["configuration"]["quadrature"]),
                    args.stage,
                    prepared if index == 0 else None,
                )
        for path, digest in source_files.items():
            checks.condition(f"source_unchanged/{path}", _hash(path) == digest)
        checks.flush(writer)
        result = {
            **identity,
            "passed": all(row["passed"] for row in checks.rows),
            "checks": len(checks.rows),
            "failed": [row["name"] for row in checks.rows if not row["passed"]],
            "seconds_including_compile_transfers_io": time.perf_counter() - started,
            "files": writer.inventory,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "passed": result["passed"],
                    "checks": result["checks"],
                    "failed": result["failed"],
                }
            )
        )
        if not result["passed"]:
            raise ArithmeticError("Native CPU12 numerical parity checks failed; inspect checks.json.")
    except BaseException:
        checks.flush(writer)
        writer.json("failure.json", {"traceback": traceback.format_exc(), "completed_checks": len(checks.rows)})
        raise


if __name__ == "__main__":
    main()
