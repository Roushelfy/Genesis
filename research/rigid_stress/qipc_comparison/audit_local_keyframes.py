"""Audit a few saved native local-atlas keyframes against same-load controls.

Run only in the captured Slurm rtx-mid / image 1_26 environment. Numerical
assembly, balancing, factorization, projection, solves, lifting and stress
evaluation use Quadrants. SciPy/NumPy independently check those operations
inside that job. Joint operators are cached research controls, with no claim
that their construction belongs in production online recovery.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy import sparse
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.data import StressInfo, create_stress_state
from genesis.engine.solvers.rigid.stress.feature_basis import FeatureBasis
from genesis.engine.solvers.rigid.stress.feature_operator import FeatureOperator
from genesis.engine.solvers.rigid.stress.feature_regions import FeatureRegions
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.solve import kernel_balance, kernel_direct_init, kernel_full_residual
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .check_local_atlas import relative_per_environment, weighted_field_error
from .check_local_feature_spaces import block_matrix, independent_map, independent_stress, vector_columns


@qd.kernel
def kernel_copy_vector(source: qd.Tensor, target: qd.Tensor):
    for i_n, i_b in qd.ndrange(target.shape[0], target.shape[1]):
        target[i_n, i_b] = source[i_n, i_b]


@qd.kernel
def kernel_add_vector(source: qd.Tensor, target: qd.Tensor):
    for i_n, i_b in qd.ndrange(target.shape[0], target.shape[1]):
        target[i_n, i_b] += source[i_n, i_b]


@qd.kernel
def kernel_add_selected(source: qd.Tensor, target: qd.Tensor, selected: qd.types.ndarray()):
    for i_n, i_b in qd.ndrange(target.shape[0], target.shape[1]):
        if selected[i_b]:
            target[i_n, i_b] += source[i_n, i_b]


@qd.kernel
def kernel_difference(first: qd.Tensor, second: qd.Tensor, target: qd.Tensor):
    for i_n, i_b in qd.ndrange(target.shape[0], target.shape[1]):
        target[i_n, i_b] = first[i_n, i_b] - second[i_n, i_b]


@qd.kernel
def kernel_expand_group(grouped: qd.Tensor, boundary: qd.Tensor, group: int, force: qd.Tensor):
    for i_n, i_b in qd.ndrange(force.shape[0], force.shape[1]):
        force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_n, i_b in qd.ndrange(boundary.shape[0], force.shape[1]):
        force[boundary[i_n], i_b] = grouped[i_n, group, i_b]


@qd.kernel
def kernel_expand_sum(grouped: qd.Tensor, boundary: qd.Tensor, force: qd.Tensor):
    for i_n, i_b in qd.ndrange(force.shape[0], force.shape[1]):
        force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_n, i_b in qd.ndrange(boundary.shape[0], force.shape[1]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for i_g in range(grouped.shape[1]):
            value += grouped[i_n, i_g, i_b]
        force[boundary[i_n], i_b] = value


@qd.kernel
def kernel_mother_residual(
    young: float, rhs: qd.Tensor, displacement: qd.Tensor, residual: qd.Tensor, info: StressInfo
):
    for i_n, i_b in qd.ndrange(rhs.shape[0], rhs.shape[1]):
        value = -rhs[i_n, i_b]
        for i_e in range(info.row_start[i_n], info.row_start[i_n + 1]):
            value += young * info.stiffness[i_e] @ displacement[info.columns[i_e], i_b]
        residual[i_n, i_b] = value


def relative(actual, expected):
    norm = sparse.linalg.norm if sparse.issparse(expected) else np.linalg.norm
    return float(norm(actual - expected) / max(float(norm(expected)), 1e-30))


def column_errors(actual, expected):
    return np.linalg.norm(actual - expected, axis=0) / np.maximum(np.linalg.norm(expected, axis=0), 1e-30)


def residual_metrics(residual, rhs, options):
    absolute = np.linalg.norm(residual, axis=0)
    rhs_norm = np.linalg.norm(rhs, axis=0)
    budget = np.maximum(options.absolute_tolerance, options.tolerance * rhs_norm)
    return {
        "absolute": absolute.tolist(),
        "relative": (absolute / np.maximum(rhs_norm, 1e-30)).tolist(),
        "rhs_norm": rhs_norm.tolist(),
        "budget": budget.tolist(),
        "passed": (absolute <= budget).tolist(),
    }


def put_batched(tensor, array):
    tensor.from_numpy(np.ascontiguousarray(array.transpose(1, 0, 2), dtype=gs.np_float))


def read_fields(state):
    values = {
        "q": qd_to_numpy(state.displacement, transpose=True, copy=True),
        "tensor": qd_to_numpy(state.stress_tensor, transpose=True, copy=True),
        "peak": qd_to_numpy(state.peak, copy=True),
    }
    assert all(np.isfinite(value).all() for value in values.values()), "Nonfinite recovered fields."
    return values


def field_errors(actual, expected, weights):
    return {
        "displacement": relative_per_environment(actual["q"], expected["q"]).tolist(),
        "stress_tensor": weighted_field_error(actual["tensor"], expected["tensor"], weights).tolist(),
        "stress_tensor_unweighted": relative_per_environment(actual["tensor"], expected["tensor"]).tolist(),
        "peak": relative_per_environment(actual["peak"], expected["peak"]).tolist(),
    }


def exact_fields_pass(errors):
    return all(max(values) <= 1e-8 for values in errors.values())


def spatial_budget(errors, tensor_budget, peak_budget):
    return {
        "stress_tensor": (np.asarray(errors["stress_tensor"]) <= tensor_budget).tolist(),
        "peak": (np.asarray(errors["peak"]) <= peak_budget).tolist(),
    }


def stress_oracle(model, state, actual):
    tensor, _vm, peak = independent_stress(model, vector_columns(state.displacement))
    return {
        "stress_tensor": relative_per_environment(actual["tensor"], tensor).tolist(),
        "peak": relative_per_environment(actual["peak"], peak).tolist(),
    }


@dataclass
class AuditOperator:
    name: str
    operator: FeatureOperator
    transform: object
    oracle_stiffness: object
    oracle_factor: object
    free: np.ndarray
    state: object
    metadata: dict


def make_operator(name, local_nodes, mother, basis, mother_matrix, n_envs, options, output):
    started = time.perf_counter()
    local_nodes = np.unique(local_nodes[basis.coarse_index[local_nodes] < 0]).astype(gs.np_int, copy=False)
    fmap = basis.transform if not len(local_nodes) else basis.feature(local_nodes)
    operator = FeatureOperator(mother.info, *fmap)
    qd.sync()
    native_preparation_seconds = time.perf_counter() - started
    bpn = basis.blocks_per_node
    xyz = qd_to_numpy(mother.info.vertices)[::bpn]
    mother_free = basis.mother_free.reshape(-1).astype(bool)
    free = fmap.reduced_free.reshape(-1).astype(bool)
    raw = independent_map(basis.parent_nodes, xyz, basis.coarse_nodes, local_nodes, bpn)
    transform = (sparse.diags(mother_free.astype(float)) @ raw @ sparse.diags(free.astype(float))).tocsr()
    native_transform = sparse.bsr_matrix(
        (qd_to_numpy(operator.transform.values), fmap.columns, fmap.row_start), shape=transform.shape
    ).tocsr()
    native_stiffness = block_matrix(operator.info)
    oracle_stiffness = (transform.T @ mother_matrix @ transform).tocsr()
    checks = {
        "constrained_transform": relative(native_transform, transform),
        "galerkin_stiffness": relative(native_stiffness, oracle_stiffness),
        "stiffness_symmetry": relative(native_stiffness, native_stiffness.T),
    }
    coordinate_nodes = np.r_[basis.coarse_nodes, local_nodes]
    coordinate_order = (3 * bpn * coordinate_nodes[:, None] + np.arange(3 * bpn)).reshape(-1)
    cardinal = transform[coordinate_order[free]][:, free]
    rank_factor = splu(cardinal.tocsc())
    pivots = np.abs(rank_factor.U.diagonal())
    assert np.isfinite(pivots).all() and (pivots > 0).all(), "Constrained feature map lost rank."
    np.testing.assert_array_equal(coordinate_order[~free], np.flatnonzero(~mother_free))
    assert transform[~mother_free].nnz == 0 and transform[:, ~free].nnz == 0
    stiffness_nonzeros = native_stiffness.copy()
    stiffness_nonzeros.eliminate_zeros()
    transform_nonzeros = native_transform.copy()
    transform_nonzeros.eliminate_zeros()
    blocks = operator.info.vertices.shape[0]
    lower_blocks = operator.factor.info.lower.shape[0]
    stored_stiffness_blocks = operator.info.columns.shape[0]
    metadata = {
        "name": name,
        "blocks": blocks,
        "scalar_dofs": 3 * blocks,
        "free_scalar_dofs": int(free.sum()),
        "local_physical_nodes": len(local_nodes),
        "map_rank": int(free.sum()),
        "rank_check": "Nonsingular free cardinal submatrix of the independently constructed constrained map.",
        "minimum_cardinal_pivot": float(pivots.min()),
        "stiffness_shape": list(native_stiffness.shape),
        "stiffness_stored_blocks": stored_stiffness_blocks,
        "stiffness_scalar_nonzeros": int(stiffness_nonzeros.nnz),
        "transform_stored_blocks": operator.n_transform_entries,
        "transform_scalar_nonzeros": int(transform_nonzeros.nnz),
        "factor_lower_blocks": lower_blocks,
        "factor_diagonal_blocks": blocks,
        "factor_symmetric_fill_ratio": float((2 * lower_blocks + blocks) / stored_stiffness_blocks),
        "native_preparation_seconds": native_preparation_seconds,
        "structural_checks": checks,
        "structural_passed": bool(max(checks.values()) <= 1e-11),
    }
    path = output / "operators" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        local_nodes=local_nodes,
        reduced_free=free,
        transform_shape=np.asarray(native_transform.shape),
        transform_row_start=native_transform.indptr,
        transform_columns=native_transform.indices,
        transform_values=native_transform.data,
        stiffness_shape=np.asarray(native_stiffness.shape),
        stiffness_row_start=native_stiffness.indptr,
        stiffness_columns=native_stiffness.indices,
        stiffness_values=native_stiffness.data,
        factor_order=qd_to_numpy(operator.factor.info.order),
        factor_row_start=qd_to_numpy(operator.factor.info.row_start),
        factor_columns=qd_to_numpy(operator.factor.info.columns),
    )
    metadata["arrays"] = file_id(path)
    return AuditOperator(
        name,
        operator,
        transform,
        oracle_stiffness,
        splu((options.young * oracle_stiffness[free][:, free]).tocsc()),
        free,
        create_stress_state(blocks, n_envs),
        metadata,
    )


def solve_feature(bundle, balanced, lifted, options):
    operator, state = bundle.operator, bundle.state
    operator.project(balanced.rhs, state.rhs)
    kernel_direct_init(state)
    operator.factor.solve(options.young, state, operator.info, options.cooperative_solve)
    kernel_full_residual(
        options.young, options.tolerance, options.absolute_tolerance, state, operator.info, only_active=False
    )
    operator.residual(options.young, state.rhs, state.displacement, state.residual)
    operator.lift(state.displacement, lifted.displacement)
    rhs = vector_columns(state.rhs)
    z = vector_columns(state.displacement)
    native_residual = vector_columns(state.residual)
    oracle_rhs = bundle.transform.T @ vector_columns(balanced.rhs)
    oracle_residual = options.young * bundle.oracle_stiffness @ z - oracle_rhs
    oracle_z = np.zeros_like(z)
    oracle_z[bundle.free] = bundle.oracle_factor.solve(oracle_rhs[bundle.free])
    checks = {
        "balanced_rhs_projection": column_errors(rhs, oracle_rhs).tolist(),
        "reduced_scipy_solution": column_errors(z, oracle_z).tolist(),
        "lift": column_errors(vector_columns(lifted.displacement), bundle.transform @ z).tolist(),
        "residual_vector_over_rhs": (
            np.linalg.norm(native_residual - oracle_residual, axis=0)
            / np.maximum(np.linalg.norm(oracle_rhs, axis=0), 1e-30)
        ).tolist(),
    }
    numerical = residual_metrics(native_residual, rhs, options)
    independent = residual_metrics(oracle_residual, oracle_rhs, options)
    residual_agreement = residual_metrics(native_residual - oracle_residual, oracle_rhs, options)
    valid = qd_to_numpy(state.valid).astype(bool)
    structural = max(checks["balanced_rhs_projection"] + checks["lift"]) <= 1e-11
    passed = bool(
        valid.all()
        and all(numerical["passed"])
        and all(independent["passed"])
        and structural
        and max(checks["reduced_scipy_solution"]) <= 1e-8
        and all(residual_agreement["passed"])
    )
    details = {
        "operator": bundle.name,
        "checks": checks,
        "native_residual": numerical,
        "independent_residual": independent,
        "residual_vector_agreement": residual_agreement,
        "native_valid": valid.tolist(),
        "passed": passed,
    }
    arrays = {
        "reduced_rhs": rhs,
        "reduced_solution": z,
        "reduced_residual": native_residual,
        "oracle_solution": oracle_z,
    }
    return details, arrays


def projected_residual(bundle, mother_residual, covered_rhs, independent_residual, options):
    bundle.operator.project(mother_residual, bundle.state.residual)
    bundle.operator.project(covered_rhs, bundle.state.rhs)
    residual = vector_columns(bundle.state.residual)
    rhs = vector_columns(bundle.state.rhs)
    independent = bundle.transform.T @ independent_residual
    agreement = np.linalg.norm(residual - independent, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
    return {
        "residual": residual_metrics(residual, rhs, options),
        "independent_vector_agreement_over_rhs": agreement.tolist(),
        "independent_vector_agreement": residual_metrics(residual - independent, rhs, options),
    }, residual


def resolve_asset(path, native):
    path = Path(path)
    candidates = (path, native / path, native / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot locate saved asset {path} from {native}.")


def load_frame(path):
    names = (
        "candidate",
        "reference",
        "q",
        "q_reference",
        "peak",
        "peak_reference",
        "force",
        "omega",
        "grouped_force",
        "active_groups",
    )
    with np.load(path, allow_pickle=False) as saved:
        missing = sorted(set(names) - set(saved.files))
        if missing:
            raise ValueError(f"{path} lacks raw keyframe fields {missing}; regenerate with the current worker.")
        frame = {name: saved[name].copy() for name in names}
    assert all(np.isfinite(array).all() for array in frame.values()), f"Nonfinite saved values in {path}."
    return frame


def audit(args, result):
    native = args.native.resolve()
    native_result = json.loads((native / "result.json").read_text())
    available = sorted(native.glob("field-[0-9][0-9][0-9][0-9].npz"))
    if args.frames is None:
        paths = available[:4]
    else:
        assert args.frames and len(set(args.frames)) == len(args.frames), "Choose distinct saved frame numbers."
        paths = [native / f"field-{step:04d}.npz" for step in args.frames]
    assert paths, f"No saved keyframes in {native}."
    first = load_frame(paths[0])
    n_envs = first["force"].shape[0]
    options_data = dict(native_result["options"])
    options_data["mesh"] = resolve_asset(options_data["mesh"], native)
    basis_path = options_data.get("atlas_basis") or native_result.get("basisPath") or native_result.get("basis_path")
    if basis_path is None and isinstance(native_result.get("basis"), dict):
        basis_path = native_result["basis"]["path"]
    options_data["atlas_basis"] = resolve_asset(basis_path, native) if basis_path else None
    options = RigidStressOptions(**options_data)
    assert options.atlas_regions > 0, "The saved run must declare a local feature library."
    assert options.tolerance == 1e-8 and options.absolute_tolerance == 1e-11
    result.update(
        native_result=file_id(native / "result.json"),
        native_complete=native_result.get("complete"),
        native_passed=native_result.get("passed"),
        native_atlas=native_result.get("atlas"),
        options=options.model_dump(mode="json"),
        frames=[int(path.stem.split("-")[1]) for path in paths],
        n_envs=n_envs,
        stress_backend=options.backend,
        tensor_budget=float(native_result.get("tensor_budget", 0.10)),
        peak_budget=float(native_result.get("peak_budget", 0.05)),
        load_budget=float(native_result.get("load_budget", 1e-10)),
    )
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    # Preserve atlas_regions when constructing the mother: auto then selects
    # the same direct mother, instead of accidentally preparing an inverse.
    mother = ShellModel(options) if options.backend == "shell" else StressModel(options)
    assert mother.selected_method == "direct" and mother.factor is not None
    parent_nodes, fixed_nodes = None, None
    if options.atlas_basis is not None:
        with np.load(options.atlas_basis, allow_pickle=False) as asset:
            parent_nodes = asset["parent_nodes"].copy() if "parent_nodes" in asset else None
            fixed_nodes = asset["fixed_nodes"].copy() if "fixed_nodes" in asset else None
        result["basis"] = file_id(options.atlas_basis)
    basis = FeatureBasis(mother, parent_nodes, fixed_nodes)
    # Selection is deliberately not rerun. Only its deterministic material
    # geometry/library construction is needed to interpret saved group loads.
    contacts = create_contacts(1, n_envs, options.contact_radius)
    regions = FeatureRegions(
        mother,
        basis,
        contacts,
        options.atlas_regions,
        options.atlas_transition,
        options.atlas_core_scale,
        options.atlas_halo,
    )
    boundary = qd_to_numpy(regions.info.boundary_nodes)
    n_blocks = mother.info.vertices.shape[0]
    n_groups = options.atlas_regions + 1
    matrix = block_matrix(mother.info)
    mother_free = basis.mother_free.reshape(-1).astype(bool)
    mother_oracle = splu((options.young * matrix[mother_free][:, mother_free]).tocsc())
    if mother.n_shell_elements:
        weights = np.repeat(qd_to_numpy(mother.shell.weights), 2, axis=0) / 2
    else:
        weights = np.broadcast_to(qd_to_numpy(mother.info.volumes)[:, None] / 4, (mother.n_output_elements, 4))
    assert weights.shape == (mother.n_output_elements, 4) and (weights > 0).all()
    pins = qd_to_numpy(mother.info.pins, copy=True)
    np.testing.assert_array_equal(np.sort(pins), np.flatnonzero(~mother_free))
    local_nodes = [nodes[basis.coarse_index[nodes] < 0] for nodes in regions.local_nodes]
    if options.atlas_space == "coarse":
        local_nodes = [np.empty(0, dtype=gs.np_int)]
    if result["native_atlas"] is not None:
        metadata = result["native_atlas"]
        assert metadata["mother_blocks"] == n_blocks and metadata["boundary_blocks"] == len(boundary)
        assert metadata["coarse_nodes"] == len(basis.coarse_nodes)
    result["mother"] = {
        "blocks": n_blocks,
        "scalar_dofs": 3 * n_blocks,
        "free_scalar_dofs": int(mother_free.sum()),
        "physical_nodes": len(basis.parent_nodes),
        "coarse_physical_nodes": len(basis.coarse_nodes),
        "stiffness_stored_blocks": mother.info.columns.shape[0],
        "factor_lower_blocks": mother.factor.info.lower.shape[0],
        "pins": pins.tolist(),
        "selected_method": mother.selected_method,
    }
    np.savez_compressed(
        args.output / "mother.npz",
        weights=weights,
        boundary_nodes=boundary,
        parent_nodes=basis.parent_nodes,
        coarse_nodes=basis.coarse_nodes,
        is_free=mother_free,
        pins=pins,
        region_seeds=qd_to_numpy(regions.info.seeds),
        region_radii=qd_to_numpy(regions.info.radii),
        stiffness_row_start=matrix.indptr,
        stiffness_columns=matrix.indices,
        stiffness_values=matrix.data,
    )
    coarse = make_operator("coarse", np.empty(0, dtype=gs.np_int), mother, basis, matrix, n_envs, options, args.output)
    result["spaces"] = [coarse.metadata]
    sources = []
    for i_g, nodes in enumerate(local_nodes, start=1):
        source = make_operator(f"feature-{i_g:03d}", nodes, mother, basis, matrix, n_envs, options, args.output)
        sources.append(source)
        result["spaces"].append(source.metadata)
        write_json(args.output / "result.json", result)
    if result["native_atlas"] is not None:
        for saved, actual in zip(result["native_atlas"]["groups"][1:], sources, strict=True):
            for key in ("blocks", "local_nodes", "stiffness_entries", "factor_entries"):
                actual_key = {
                    "local_nodes": "local_physical_nodes",
                    "stiffness_entries": "stiffness_stored_blocks",
                    "factor_entries": "factor_lower_blocks",
                }.get(key, key)
                assert saved[key] == actual.metadata[actual_key], (key, saved, actual.metadata)
    result["joint_cache"] = []
    joint_cache = {(): coarse, **{(i,): source for i, source in enumerate(sources, start=1)}}
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    zero_omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    zero_omega.fill(0)
    grouped = V_VEC(3, dtype=gs.qd_float, shape=(len(boundary), n_groups, n_envs))
    reference = mother.create_state(n_envs, "full")
    fallback_spin = mother.create_state(n_envs)
    covered = create_stress_state(n_blocks, n_envs)
    source_load = create_stress_state(n_blocks, n_envs)
    lifted = create_stress_state(n_blocks, n_envs)
    source_covered = create_stress_state(n_blocks, n_envs)
    joint_covered = create_stress_state(n_blocks, n_envs)
    rhs_sum = create_stress_state(n_blocks, n_envs)
    fields = {name: mother.create_state(n_envs, "full") for name in ("coarse", "local", "joint")}
    write_json(args.output / "result.json", result)
    for path in paths:
        frame = first if path == paths[0] else load_frame(path)
        step = int(path.stem.split("-")[1])
        expected_shapes = {
            "candidate": (n_envs, mother.n_output_elements, 4, 6),
            "reference": (n_envs, mother.n_output_elements, 4, 6),
            "q": (n_envs, n_blocks, 3),
            "q_reference": (n_envs, n_blocks, 3),
            "peak": (n_envs,),
            "peak_reference": (n_envs,),
            "force": (n_envs, n_blocks, 3),
            "omega": (n_envs, 3),
            "grouped_force": (n_envs, len(boundary), n_groups, 3),
            "active_groups": (n_envs, n_groups),
        }
        for name, shape in expected_shapes.items():
            assert frame[name].shape == shape, (path, name, frame[name].shape, shape)
        active = frame["active_groups"].astype(bool)
        actual_active = np.any(frame["grouped_force"] != 0, axis=(1, 3))
        active_matches = np.array_equal(active, actual_active)
        put_batched(reference.force, frame["force"])
        omega.from_numpy(np.ascontiguousarray(frame["omega"], dtype=gs.np_float))
        grouped.from_numpy(np.ascontiguousarray(frame["grouped_force"].transpose(1, 2, 0, 3), dtype=gs.np_float))
        kernel_expand_sum(grouped, regions.info.boundary_nodes, source_load.force)
        grouped_error = relative_per_environment(qd_to_numpy(source_load.force, transpose=True), frame["force"])
        mother.recover(omega, reference)
        kernel_mother_residual(options.young, reference.rhs, reference.displacement, reference.residual, mother.info)
        mother_numerical = residual_metrics(vector_columns(reference.residual), vector_columns(reference.rhs), options)
        fresh = read_fields(reference)
        saved_reference = {"q": frame["q_reference"], "tensor": frame["reference"], "peak": frame["peak_reference"]}
        saved_local = {"q": frame["q"], "tensor": frame["candidate"], "peak": frame["peak"]}
        mother_reconstruction = field_errors(fresh, saved_reference, weights)
        rhs = vector_columns(reference.rhs)
        oracle_q = np.zeros_like(rhs)
        oracle_q[mother_free] = mother_oracle.solve(rhs[mother_free])
        mother_scipy = column_errors(vector_columns(reference.displacement), oracle_q).tolist()
        kernel_expand_group(grouped, regions.info.boundary_nodes, 0, fallback_spin.force)
        mother.recover(omega, fallback_spin)
        kernel_mother_residual(
            options.young, fallback_spin.rhs, fallback_spin.displacement, fallback_spin.residual, mother.info
        )
        fallback_numerical = residual_metrics(
            vector_columns(fallback_spin.residual), vector_columns(fallback_spin.rhs), options
        )
        kernel_copy_vector(fallback_spin.rhs, rhs_sum.rhs)
        kernel_difference(reference.force, fallback_spin.force, covered.force)
        kernel_balance(zero_omega, covered, mother.info, options.cooperative_balance)
        source_covered.displacement.fill(0)
        source_details = []
        detail_dir = args.output / f"frame-{step:04d}"
        detail_dir.mkdir()
        for i_g, bundle in enumerate(sources, start=1):
            kernel_expand_group(grouped, regions.info.boundary_nodes, i_g, source_load.force)
            kernel_balance(zero_omega, source_load, mother.info, options.cooperative_balance)
            kernel_add_vector(source_load.rhs, rhs_sum.rhs)
            details, arrays = solve_feature(bundle, source_load, lifted, options)
            kernel_add_vector(lifted.displacement, source_covered.displacement)
            arrays["balanced_mother_rhs"] = vector_columns(source_load.rhs)
            arrays["lifted_displacement"] = vector_columns(lifted.displacement)
            arrays_path = detail_dir / f"source-{i_g:03d}.npz"
            np.savez_compressed(arrays_path, **arrays)
            details.update(group=i_g, active=active[:, i_g].tolist(), arrays=file_id(arrays_path))
            source_details.append(details)
        kernel_copy_vector(source_covered.displacement, fields["local"].displacement)
        kernel_add_vector(fallback_spin.displacement, fields["local"].displacement)
        coarse_details, coarse_arrays = solve_feature(coarse, covered, fields["coarse"], options)
        kernel_add_vector(fallback_spin.displacement, fields["coarse"].displacement)
        np.savez_compressed(detail_dir / "coarse.npz", **coarse_arrays)
        kernel_copy_vector(fallback_spin.displacement, fields["joint"].displacement)
        joint_covered.displacement.fill(0)
        kernel_mother_residual(
            options.young, covered.rhs, source_covered.displacement, source_covered.residual, mother.info
        )
        independent_source_residual = options.young * matrix @ vector_columns(
            source_covered.displacement
        ) - vector_columns(covered.rhs)
        tuples = [tuple(int(index) for index in np.flatnonzero(row[1:]) + 1) for row in active]
        joint_details = []
        joint_names = [None] * n_envs
        for active_tuple in sorted(set(tuples)):
            if active_tuple not in joint_cache:
                union = np.unique(np.concatenate([local_nodes[i - 1] for i in active_tuple]))
                name = "joint-" + "-".join(f"{i:03d}" for i in active_tuple)
                joint_cache[active_tuple] = make_operator(
                    name, union, mother, basis, matrix, n_envs, options, args.output
                )
                result["spaces"].append(joint_cache[active_tuple].metadata)
            bundle = joint_cache[active_tuple]
            selected = np.asarray([value == active_tuple for value in tuples], dtype=gs.np_int)
            details, arrays = solve_feature(bundle, covered, lifted, options)
            kernel_add_selected(lifted.displacement, fields["joint"].displacement, selected)
            kernel_add_selected(lifted.displacement, joint_covered.displacement, selected)
            summed_residual, raw_summed_residual = projected_residual(
                bundle, source_covered.residual, covered.rhs, independent_source_residual, options
            )
            arrays["source_sum_joint_galerkin_residual"] = raw_summed_residual
            arrays_path = detail_dir / ("joint-" + ("-".join(map(str, active_tuple)) or "coarse") + ".npz")
            np.savez_compressed(arrays_path, **arrays)
            details.update(
                active_groups=list(active_tuple),
                environments=np.flatnonzero(selected).tolist(),
                source_sum_joint_galerkin=summed_residual,
                arrays=file_id(arrays_path),
            )
            joint_details.append(details)
            for i_b in np.flatnonzero(selected):
                joint_names[i_b] = bundle.name
        actual = {}
        spatial = {}
        mother_residuals = {}
        field_oracles = {"mother": stress_oracle(mother, reference, fresh)}
        for name, state in fields.items():
            mother.evaluate_stress(options, state)
            actual[name] = read_fields(state)
            spatial[name] = field_errors(actual[name], saved_reference, weights)
            field_oracles[name] = stress_oracle(mother, state, actual[name])
            kernel_mother_residual(options.young, reference.rhs, state.displacement, state.residual, mother.info)
            mother_residuals[name] = residual_metrics(vector_columns(state.residual), rhs, options)
        source_reconstruction = field_errors(actual["local"], saved_local, weights)
        source_vs_joint = field_errors(actual["local"], actual["joint"], weights)
        load_linearity = column_errors(vector_columns(rhs_sum.rhs), rhs).tolist()
        source_joint_agreement = all(
            all(item["source_sum_joint_galerkin"]["independent_vector_agreement"]["passed"]) for item in joint_details
        )
        numerical_passed = bool(
            active_matches
            and grouped_error.max() <= result["load_budget"]
            and qd_to_numpy(reference.valid).all()
            and qd_to_numpy(fallback_spin.valid).all()
            and all(mother_numerical["passed"])
            and all(fallback_numerical["passed"])
            and exact_fields_pass(mother_reconstruction)
            and exact_fields_pass(source_reconstruction)
            and max(mother_scipy) <= 1e-8
            and max(load_linearity) <= 1e-11
            and all(item["passed"] for item in (*source_details, coarse_details, *joint_details))
            and all(max(values) <= 1e-8 for checks in field_oracles.values() for values in checks.values())
            and source_joint_agreement
        )
        row = {
            "step": step,
            "input": file_id(path),
            "active_groups": active.astype(int).tolist(),
            "active_groups_match_load": active_matches,
            "grouped_load_error": grouped_error.tolist(),
            "load_relief_linearity": load_linearity,
            "mother_saved_reconstruction": mother_reconstruction,
            "mother_scipy_displacement": mother_scipy,
            "mother_numerical_residual": mother_numerical,
            "fallback_spin_numerical_residual": fallback_numerical,
            "source_sum_saved_reconstruction": source_reconstruction,
            "spatial_errors": spatial,
            "spatial_budget_passed": {
                name: spatial_budget(errors, result["tensor_budget"], result["peak_budget"])
                for name, errors in spatial.items()
            },
            "source_sum_vs_joint": source_vs_joint,
            "mother_spatial_residual": mother_residuals,
            "independent_stress_checks": field_oracles,
            "sources": source_details,
            "coarse": coarse_details,
            "joint": joint_details,
            "joint_operator_per_environment": joint_names,
            "numerical_passed": numerical_passed,
        }
        arrays = {
            "q_fallback_spin": qd_to_numpy(fallback_spin.displacement, transpose=True),
            "covered_force": qd_to_numpy(covered.force, transpose=True),
            "covered_balanced_rhs": qd_to_numpy(covered.rhs, transpose=True),
            "q_source_sum_covered": qd_to_numpy(source_covered.displacement, transpose=True),
            "q_joint_covered": qd_to_numpy(joint_covered.displacement, transpose=True),
            "q_mother_fresh": fresh["q"],
            "tensor_mother_fresh": fresh["tensor"],
            "peak_mother_fresh": fresh["peak"],
        }
        for name, values in actual.items():
            arrays.update({f"{key}_{name}": value for key, value in values.items()})
        np.savez_compressed(detail_dir / "fields.npz", **arrays)
        row["arrays"] = file_id(detail_dir / "fields.npz")
        result["rows"].append(row)
        result["joint_cache"] = [
            {"active_groups": list(key), "operator": value.name} for key, value in sorted(joint_cache.items())
        ]
        np.testing.assert_array_equal(qd_to_numpy(mother.info.is_free).reshape(-1), mother_free)
        np.testing.assert_array_equal(qd_to_numpy(mother.info.pins), pins)
        write_json(args.output / "result.json", result)
    result.update(
        complete=True,
        passed=bool(
            all(space["structural_passed"] for space in result["spaces"])
            and all(row["numerical_passed"] for row in result["rows"])
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native", type=Path, required=True, help="Native run containing result.json and field-NNNN.npz."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", help="Saved frame numbers; default: first four available.")
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Run this scientific audit inside a scheduled Slurm job."
    assert os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid", "Use the established rtx-mid partition."
    assert os.environ.get("GENESIS_IMAGE_VER") == "1_26", "Use the captured image 1_26 environment."
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "identity": runtime_identity(),
        "complete": False,
        "passed": False,
        "rows": [],
        "numerical_tolerance": 1e-8,
        "absolute_tolerance": 1e-11,
        "reconstruction_tolerance": 1e-8,
        "structural_tolerance": 1e-11,
        "field_metric": "Area/volume-weighted native sample tensor Frobenius norm; symmetric shears count twice.",
        "approximation_errors_are_diagnostics": True,
        "joint_control": "Union of each environment's active feature-local nodes; solve its total covered mother-relieved load.",
        "fallback_control": "Exact mother recovery of group zero plus the saved spin, added to every approximate control.",
        "independent_source_sum_is_not_automatically_joint": True,
        "preparation_scope": "Native fixed operators cached by active-group tuple for this bounded offline audit; no online-factorization performance claim.",
    }
    write_json(args.output / "result.json", result)
    try:
        audit(args, result)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(args.output / "result.json", result)
        raise
    write_json(args.output / "result.json", result)
    assert result["passed"], (
        "Independent numerical/keyframe reconstruction audit failed; retain result.json and raw details."
    )


if __name__ == "__main__":
    main()
