"""Bounded independent controls for native common-mother local feature spaces.

Run this worker inside the scheduled numerical environment. SciPy is an oracle;
native preparation, projection, factorization, solves and lifting use Quadrants.
The approximate spaces have their own equilibrium contract. Their mother-field
errors are reported separately from numerical solve acceptance.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy import sparse
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import create_stress_state
from genesis.engine.solvers.rigid.stress.feature_basis import FeatureBasis
from genesis.engine.solvers.rigid.stress.feature_operator import FeatureOperator
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.solve import kernel_direct_init, kernel_full_residual
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json


def relative(actual, expected):
    norm = sparse.linalg.norm if sparse.issparse(expected) else np.linalg.norm
    return float(norm(actual - expected) / max(norm(expected), 1e-30))


def per_environment(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    difference = (actual - expected).reshape((len(actual), -1))
    reference = expected.reshape((len(expected), -1))
    return (np.linalg.norm(difference, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-30)).tolist()


def block_matrix(info):
    n = 3 * info.vertices.shape[0]
    return sparse.bsr_matrix(
        (qd_to_numpy(info.stiffness), qd_to_numpy(info.columns), qd_to_numpy(info.row_start)), shape=(n, n)
    ).tocsr()


def vector_columns(tensor):
    """Read native (blocks, environments, xyz) as (scalar coordinates, environments)."""
    array = qd_to_numpy(tensor, copy=True)
    return array.transpose(0, 2, 1).reshape((-1, array.shape[1]))


def put_columns(tensor, values):
    tensor.from_numpy(np.ascontiguousarray(values.reshape((-1, 3, values.shape[1])).transpose(0, 2, 1)))


def source_parents(directory, shell):
    """Recover provenance only for this tracked triangle-to-three-quads author."""
    with np.load(directory / "elastic.npz", allow_pickle=False) as asset:
        volume_vertices, tetrahedra = asset["vertices"], asset["tetrahedra"]
    if not shell:
        edges = np.unique(
            np.sort(tetrahedra[:, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))].reshape((-1, 2)), axis=1),
            axis=0,
        )
        parents = np.full((len(volume_vertices) + len(edges), 3), -1, dtype=np.int64)
        parents[: len(volume_vertices), 0] = np.arange(len(volume_vertices))
        parents[len(volume_vertices) :, :2] = edges
        xyz = np.concatenate((volume_vertices, volume_vertices[edges].mean(axis=1)))
        return parents, xyz, {"author": "native affine P2 tetrahedra", "original_nodes": len(volume_vertices)}

    metadata = json.loads((directory / "shell_metadata.json").read_text())
    config = json.loads((directory / "egg_shell.metadata.json").read_text())["config"]
    for name, key in (
        ("shell.npz", "shell_sha256"),
        ("elastic.npz", "source_elastic_sha256"),
        ("egg_outer.obj", "collision_sha256"),
    ):
        assert file_id(directory / name)["sha256"] == metadata[key], (name, metadata)
    faces = np.array(
        [
            [int(word.split("/")[0]) - 1 for word in line.split()[1:]]
            for line in (directory / "egg_outer.obj").read_text().splitlines()
            if line.startswith("f ")
        ],
        dtype=np.int64,
    )
    n_original = len(volume_vertices) // (config["layers"] + 1)
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert np.array_equal(np.unique(faces), np.arange(n_original))
    edges, inverse = np.unique(
        np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1), axis=0, return_inverse=True
    )
    mids = n_original + inverse.reshape((-1, 3))
    centers = n_original + len(edges) + np.arange(len(faces))
    expected_quads = np.stack(
        [np.column_stack((faces[:, i], mids[:, i], centers, mids[:, (i + 2) % 3])) for i in range(3)], axis=1
    ).reshape((-1, 4))
    with np.load(directory / "shell.npz", allow_pickle=False) as asset:
        xyz, quads, directors = asset["vertices"], asset["quads"], asset["directors"]
    np.testing.assert_array_equal(quads, expected_quads)
    original = np.unique(quads[:, 0])
    np.testing.assert_array_equal(original, np.arange(n_original))
    parent_sets = [set() for _ in xyz]
    for quad in quads:
        for node in quad:
            parent_sets[node].add(int(quad[0]))
    parents = np.full((len(xyz), 3), -1, dtype=np.int64)
    for node, values in enumerate(parent_sets):
        assert 1 <= len(values) <= 3
        parents[node, : len(values)] = sorted(values)
    counts = np.count_nonzero(parents >= 0, axis=1)
    np.testing.assert_array_equal(np.bincount(counts, minlength=4)[1:], [n_original, len(edges), len(faces)])
    np.testing.assert_array_equal(parents[:n_original, 0], original)
    assert (parents[:n_original, 1:] == -1).all()
    np.testing.assert_array_equal(parents[n_original : n_original + len(edges), :2], edges)
    np.testing.assert_array_equal(parents[n_original + len(edges) :], np.sort(faces, axis=1))
    assert len(quads) == 3 * len(faces) == metadata["quads"]
    assert 6 * len(xyz) == metadata["dofs"]
    outer = volume_vertices[:n_original]
    normals = (outer - volume_vertices[config["layers"] * n_original :]) / config["thickness_m"]
    outer = np.concatenate((outer, outer[edges].mean(axis=1), outer[faces].mean(axis=1)))
    normals = np.concatenate((normals, normals[edges].mean(axis=1), normals[faces].mean(axis=1)))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    np.testing.assert_allclose(directors, normals, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(xyz, outer - 0.5 * config["thickness_m"] * normals, rtol=1e-13, atol=1e-15)
    return (
        parents,
        xyz,
        {
            "author": "make_native_shell_assets.py triangle to three quads",
            "validated_source_hashes": True,
            "original_nodes": n_original,
            "edge_nodes": len(edges),
            "triangle_center_nodes": len(faces),
            "parent_cardinalities": np.bincount(counts, minlength=4).tolist(),
            "triangle_count": len(faces),
            "quads": len(quads),
        },
    )


def independent_map(parents, xyz, coarse, local, blocks_per_node, geometric_correction=True):
    """Build scalar CSR directly from physical parents, independently of native CSR entries."""
    coarse_index = {int(node): i for i, node in enumerate(coarse)}
    local_index = {int(node): len(coarse) + i for i, node in enumerate(local)}
    row, column, values = [], [], []

    def append(i, j, matrix):
        for a in range(3):
            for b in range(3):
                if matrix[a, b] != 0:
                    row.append(3 * i + a)
                    column.append(3 * j + b)
                    values.append(matrix[a, b])

    for node, parent_row in enumerate(parents):
        ancestors = np.array([node]) if node in coarse_index else parent_row[parent_row >= 0]
        weight = 1.0 / len(ancestors)
        delta = xyz[node] - xyz[ancestors].mean(axis=0)
        cross = np.array([[0, delta[2], -delta[1]], [-delta[2], 0, delta[0]], [delta[1], -delta[0], 0]])
        for parent in ancestors:
            for block in range(blocks_per_node):
                append(
                    blocks_per_node * node + block,
                    blocks_per_node * coarse_index[int(parent)] + block,
                    weight * np.eye(3),
                )
            if blocks_per_node == 2 and geometric_correction:
                append(2 * node, 2 * coarse_index[int(parent)] + 1, weight * cross)
        if node in local_index:
            for block in range(blocks_per_node):
                append(blocks_per_node * node + block, blocks_per_node * local_index[node] + block, np.eye(3))
    shape = (3 * blocks_per_node * len(xyz), 3 * blocks_per_node * (len(coarse) + len(local)))
    return sparse.coo_matrix((values, (row, column)), shape=shape).tocsr()


def rigid_modes(xyz, blocks_per_node):
    modes = np.zeros((len(xyz), blocks_per_node, 3, 6))
    modes[:, 0, :, :3] = np.eye(3)
    for i, omega in enumerate(np.eye(3)):
        modes[:, 0, :, 3 + i] = np.cross(omega, xyz)
    if blocks_per_node == 2:
        modes[:, 1, :, 3:] = np.eye(3)
    return modes.reshape((-1, 6))


def independent_stress(model, displacement):
    """Evaluate full sampled tensors with NumPy constitutive contractions."""
    options = model.options
    n_envs = displacement.shape[1]
    if model.n_shell_elements:
        quads = qd_to_numpy(model.shell.quads)
        q = displacement.reshape((-1, 6, n_envs))[quads].reshape((len(quads), 24, n_envs))
        strain = np.einsum("egdk,edb->egkb", qd_to_numpy(model.shell.membrane), q)
        curvature = np.einsum("egdk,edb->egkb", qd_to_numpy(model.shell.bending), q)
        nu = options.poisson
        constitutive = options.young / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        membrane = np.einsum("ij,egjb->egib", constitutive, strain)
        bending = np.einsum("ij,egjb->egib", constitutive, curvature)
        local = np.stack(
            (membrane + options.shell_thickness / 2 * bending, membrane - options.shell_thickness / 2 * bending), axis=1
        )
        axes = qd_to_numpy(model.shell.axes)
        e1, e2 = axes[..., 0], axes[..., 1]
        basis = np.stack(
            (
                np.einsum("egi,egj->egij", e1, e1),
                np.einsum("egi,egj->egij", e2, e2),
                np.einsum("egi,egj->egij", e1, e2) + np.einsum("egi,egj->egij", e2, e1),
            ),
            axis=2,
        )
        sigma = np.einsum("esgkb,egkij->besgij", local, basis).reshape((n_envs, -1, 4, 3, 3))
    else:
        elements, edges = qd_to_numpy(model.info.elements), qd_to_numpy(model.info.edges)
        gradient = qd_to_numpy(model.info.gradients)
        gradients = np.empty((len(elements), 4, 10, 3))
        for corner, bary in enumerate(np.eye(4)):
            gradients[:, corner, :4] = (4 * bary - 1)[None, :, None] * gradient
            for edge, (a, b) in enumerate(edges):
                gradients[:, corner, 4 + edge] = 4 * (bary[a] * gradient[:, b] + bary[b] * gradient[:, a])
        q = displacement.reshape((-1, 3, n_envs))[elements]
        derivative = np.einsum("endb,ecnj->becdj", q, gradients)
        mu = options.young / (2 * (1 + options.poisson))
        lam = options.young * options.poisson / ((1 + options.poisson) * (1 - 2 * options.poisson))
        sigma = mu * (derivative + derivative.swapaxes(-1, -2))
        sigma += lam * np.trace(derivative, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
    components = np.stack([sigma[..., a, b] for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1)
    vm = np.sqrt(
        np.maximum(
            0,
            0.5
            * (
                (components[..., 0] - components[..., 1]) ** 2
                + (components[..., 1] - components[..., 2]) ** 2
                + (components[..., 2] - components[..., 0]) ** 2
            )
            + 3 * np.sum(components[..., 3:] ** 2, axis=-1),
        )
    )
    return components, vm, vm.max(axis=(1, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "passed": False,
        "complete": False,
        "stress_backend": args.stress_backend,
        "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
        "spaces": [],
    }
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    directory = Path(gs.__file__).resolve().parent.parent / "examples/rigid/assets/hollow_egg/level1"
    is_shell = args.stress_backend == "shell"
    mesh = directory / ("shell.npz" if is_shell else "elastic.npz")
    parents, xyz, provenance = source_parents(directory, is_shell)
    options = gs.options.RigidStressOptions.dpr(
        mesh=mesh,
        backend=args.stress_backend,
        method="direct",
        surface_inverse=False,
        output_mode="full",
        factor_scaling=True,
    )
    assert options.tolerance == 1e-8 and options.absolute_tolerance == 1e-11
    mother = ShellModel(options) if is_shell else StressModel(options)
    bpn = 2 if is_shell else 1
    np.testing.assert_allclose(qd_to_numpy(mother.info.vertices)[::bpn], xyz, rtol=1e-13, atol=1e-15)
    basis = FeatureBasis(mother, shell_parents=parents if is_shell else None)
    expected_parents = parents.copy()
    expected_parents[basis.coarse_nodes] = -1
    expected_parents[basis.coarse_nodes, 0] = basis.coarse_nodes
    np.testing.assert_array_equal(basis.parent_nodes, expected_parents)
    mother_free = qd_to_numpy(mother.info.is_free, copy=True).reshape(-1).astype(bool)
    pins = qd_to_numpy(mother.info.pins, copy=True)
    np.testing.assert_array_equal(np.sort(pins), np.flatnonzero(~mother_free))
    assert len(pins) == 6
    km = block_matrix(mother.info)
    n_envs = 4
    rng = np.random.default_rng(20261026)
    force = rng.normal(size=(km.shape[0], n_envs)) * 1e-3
    if is_shell:
        force.reshape((-1, 6, n_envs))[:, 3:] *= options.shell_thickness
    force[:, 2] *= 1e-12
    force[:, 3] = 0
    angular = np.array([[1.3, -0.7, 2.1], [-2.4, 0.9, 1.1], [0, 0, 0], [0, 0, 0]], dtype=gs.np_float)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.from_numpy(angular)
    reference = mother.create_state(n_envs, "full")
    put_columns(reference.force, force)
    mother.recover(omega, reference)
    rhs, q_reference = vector_columns(reference.rhs), vector_columns(reference.displacement)
    field_reference = qd_to_numpy(reference.stress_tensor, transpose=True, copy=True)
    peak_reference = qd_to_numpy(reference.peak, copy=True)
    assert qd_to_numpy(reference.valid).all(), "Mother recovery failed its original residual contract."
    cpu_q = np.zeros_like(q_reference)
    cpu_q[mother_free] = splu((options.young * km[mother_free][:, mother_free]).tocsc()).solve(rhs[mother_free])
    assert relative(q_reference, cpu_q) < 1e-8
    cpu_field, cpu_vm, cpu_peak = independent_stress(mother, q_reference)
    mother_checks = {
        "sparse_oracle_displacement": relative(q_reference, cpu_q),
        "numpy_stress": relative(field_reference, cpu_field),
        "numpy_von_mises": relative(qd_to_numpy(reference.von_mises, transpose=True)[..., 0], cpu_vm),
        "numpy_peak": relative(peak_reference, cpu_peak),
    }
    assert max(mother_checks.values()) < 1e-8, mother_checks
    bubbles = np.flatnonzero(basis.coarse_index < 0)
    seed = xyz[np.argmax(xyz[:, 0])]
    count = min(32, max(1, len(bubbles) // 8))
    distance = np.linalg.norm(xyz[bubbles] - seed, axis=1)
    local = np.sort(bubbles[np.argsort(distance, kind="stable")[:count]])
    assert 0 < len(local) < len(bubbles)
    modes = rigid_modes(xyz, bpn)
    coarse_modes = modes.reshape((-1, bpn * 3, 6))[basis.coarse_nodes].reshape((-1, 6))
    result.update(
        {
            "asset": file_id(mesh),
            "provenance": provenance,
            "mother_checks": mother_checks,
            "mother_dofs": km.shape[0],
            "mother_free_dofs": int(mother_free.sum()),
            "mother_pins": pins.tolist(),
            "coarse_physical_nodes": len(basis.coarse_nodes),
            "shared_mother_geometry_mass_gauge_relief": True,
            "load": "arbitrary nodal forces, two nonzero spins, tiny-load and zero controls",
            "local_seed_m": seed.tolist(),
            "local_radius_m": float(np.linalg.norm(xyz[local] - seed, axis=1).max()),
            "tolerance": options.tolerance,
            "absolute_tolerance": options.absolute_tolerance,
        }
    )
    np.savez_compressed(
        args.output / "mother.npz",
        xyz=xyz,
        parents=parents,
        coarse_nodes=basis.coarse_nodes,
        local_nodes=local,
        force=force,
        angular_velocity=angular,
        balanced_rhs=rhs,
        displacement=q_reference,
        stress_tensor=field_reference,
        peak=peak_reference,
        pins=pins,
    )
    write_json(args.output / "result.json", result)
    for name, selected in (("coarse", np.empty(0, dtype=np.int64)), ("local", local), ("all_bubbles", bubbles)):
        started = time.perf_counter()
        fmap = basis.feature(selected)
        raw = independent_map(parents, xyz, basis.coarse_nodes, selected, bpn)
        native_raw = sparse.bsr_matrix(
            (qd_to_numpy(fmap.values), fmap.columns, fmap.row_start), shape=raw.shape
        ).tocsr()
        checks = {"raw_prolongation": relative(native_raw, raw)}
        reduced_modes = np.vstack((coarse_modes, np.zeros((3 * bpn * len(selected), 6))))
        checks["raw_rigid_modes"] = relative(native_raw @ reduced_modes, modes)
        uncorrected = independent_map(parents, xyz, basis.coarse_nodes, selected, bpn, geometric_correction=False)
        transl_rotation_rows = (3 * bpn * np.arange(len(xyz))[:, None] + np.arange(3)).reshape(-1)
        correction_witness = relative(
            (uncorrected @ reduced_modes)[transl_rotation_rows, 3:], modes[transl_rotation_rows, 3:]
        )
        if is_shell:
            assert correction_witness > 1e-8, "Curved shell must exercise the geometric rotation correction."
        free = fmap.reduced_free.reshape(-1).astype(bool)
        physical_order = np.r_[basis.coarse_nodes, selected]
        coordinate_order = (3 * bpn * physical_order[:, None] + np.arange(3 * bpn)).reshape(-1)
        cardinal = raw[coordinate_order[free]][:, free]
        rank_factor = splu(cardinal.tocsc())
        pivots = np.abs(rank_factor.U.diagonal())
        assert np.isfinite(pivots).all() and (pivots > 0).all()
        assert np.count_nonzero(~free) == len(pins)
        np.testing.assert_array_equal(coordinate_order[~free], np.flatnonzero(~mother_free))
        transform = sparse.diags(mother_free.astype(float)) @ raw @ sparse.diags(free.astype(float))
        operator = FeatureOperator(mother.info, *fmap)
        native_transform = sparse.bsr_matrix(
            (qd_to_numpy(operator.transform.values), fmap.columns, fmap.row_start), shape=raw.shape
        ).tocsr()
        checks["constrained_transform"] = relative(native_transform, transform)
        ki = block_matrix(operator.info)
        oracle_ki = (transform.T @ km @ transform).tocsr()
        checks["galerkin_stiffness"] = relative(ki, oracle_ki)
        assert transform[~mother_free].nnz == 0 and transform[:, ~free].nnz == 0
        reduced = create_stress_state(fmap.n_reduced_blocks, n_envs)
        operator.project(reference.rhs, reduced.rhs)
        reduced_rhs = vector_columns(reduced.rhs)
        checks["balanced_rhs_projection"] = relative(reduced_rhs, transform.T @ rhs)
        kernel_direct_init(reduced)
        operator.factor.solve(options.young, reduced, operator.info)
        kernel_full_residual(options.young, options.tolerance, options.absolute_tolerance, reduced, operator.info)
        operator.residual(options.young, reduced.rhs, reduced.displacement, reduced.residual)
        z = vector_columns(reduced.displacement)
        residual = vector_columns(reduced.residual)
        independent_residual = options.young * oracle_ki @ z - reduced_rhs
        checks["residual_vector_absolute"] = float(np.max(np.abs(residual - independent_residual)))
        budgets = np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(reduced_rhs, axis=0))
        residual_norms = np.linalg.norm(residual, axis=0)
        assert (residual_norms <= budgets).all() and qd_to_numpy(reduced.valid).all()
        assert (np.linalg.norm(independent_residual, axis=0) <= budgets).all()
        oracle_z = np.zeros_like(z)
        oracle_z[free] = splu((options.young * oracle_ki[free][:, free]).tocsc()).solve(reduced_rhs[free])
        checks["reduced_scipy_solution"] = relative(z, oracle_z)
        field_state = mother.create_state(n_envs, "full")
        operator.lift(reduced.displacement, field_state.displacement)
        q = vector_columns(field_state.displacement)
        checks["lift"] = relative(q, transform @ z)
        operator.project(reference.force, reduced.force)
        projected_force = vector_columns(reduced.force)
        checks["arbitrary_force_projection"] = relative(projected_force, transform.T @ force)
        work_difference = np.abs(np.sum(q * force, axis=0) - np.sum(z * projected_force, axis=0))
        work_scale = np.maximum(np.linalg.norm(q, axis=0) * np.linalg.norm(force, axis=0), 1e-30)
        checks["virtual_work"] = float(np.max(work_difference / work_scale))
        assert (q[~mother_free] == 0).all()
        mother.evaluate_stress(options, field_state)
        fields = qd_to_numpy(field_state.stress_tensor, transpose=True, copy=True)
        peak = qd_to_numpy(field_state.peak, copy=True)
        expected_fields, expected_vm, expected_peak = independent_stress(mother, transform @ oracle_z)
        checks["independent_spatial_field"] = relative(fields, expected_fields)
        checks["independent_spatial_vm"] = relative(
            qd_to_numpy(field_state.von_mises, transpose=True)[..., 0], expected_vm
        )
        checks["independent_spatial_peak"] = relative(peak, expected_peak)
        spatial = {
            "displacement_relative": per_environment(q.T, q_reference.T),
            "stress_relative": per_environment(fields, field_reference),
            "peak_relative": per_environment(peak[:, None], peak_reference[:, None]),
            "mother_residual_relative": (
                np.linalg.norm(options.young * km @ q - rhs, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
            ).tolist(),
        }
        assert np.isfinite(fields).all() and np.isfinite(peak).all()
        np.testing.assert_array_equal(q[:, 3], 0)
        np.testing.assert_array_equal(fields[3], 0)
        assert max(checks.values()) < 1e-8, (name, checks)
        assert (
            max(
                checks[key]
                for key in (
                    "raw_prolongation",
                    "raw_rigid_modes",
                    "constrained_transform",
                    "galerkin_stiffness",
                    "balanced_rhs_projection",
                    "lift",
                    "arbitrary_force_projection",
                    "virtual_work",
                )
            )
            < 1e-11
        ), checks
        if name == "all_bubbles":
            assert raw.shape[0] == raw.shape[1] and free.sum() == mother_free.sum()
            assert (
                max(max(spatial[key]) for key in ("displacement_relative", "stress_relative", "peak_relative")) < 1e-8
            ), spatial
        else:
            assert raw.shape[1] < raw.shape[0]
        np.testing.assert_array_equal(qd_to_numpy(mother.info.is_free).reshape(-1), mother_free)
        np.testing.assert_array_equal(qd_to_numpy(mother.info.pins), pins)
        arrays = args.output / f"{name}.npz"
        np.savez_compressed(
            arrays,
            local_nodes=selected,
            reduced_rhs=reduced_rhs,
            reduced_solution=z,
            reduced_residual=residual,
            displacement=q,
            stress_tensor=fields,
            peak=peak,
            oracle_reduced_solution=oracle_z,
            oracle_stress_tensor=expected_fields,
            transform_row_start=fmap.row_start,
            transform_columns=fmap.columns,
            transform_values=qd_to_numpy(operator.transform.values),
        )
        row = {
            "name": name,
            "dofs": raw.shape[1],
            "free_dofs": int(free.sum()),
            "actual_numerical_rank": len(pivots),
            "rank_certificate": "sparse LU of free cardinal coordinate rows",
            "rank_minimum_pivot": float(pivots.min()),
            "local_physical_nodes": len(selected),
            "same_six_scalar_pins": True,
            "checks": checks,
            "uncorrected_rotation_translation_error": correction_witness,
            "unscaled_reduced_residual_norm": residual_norms.tolist(),
            "original_residual_budget": budgets.tolist(),
            "spatial_error_separate_from_solve_acceptance": spatial,
            "exact_mother_space": name == "all_bubbles",
            "array_artifact": file_id(arrays),
            "seconds": time.perf_counter() - started,
        }
        result["spaces"].append(row)
        write_json(args.output / "result.json", result)
    assert result["spaces"][0]["dofs"] < result["spaces"][1]["dofs"] < result["spaces"][2]["dofs"]
    result.update(passed=True, complete=True, whole_goal_complete=False)
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
