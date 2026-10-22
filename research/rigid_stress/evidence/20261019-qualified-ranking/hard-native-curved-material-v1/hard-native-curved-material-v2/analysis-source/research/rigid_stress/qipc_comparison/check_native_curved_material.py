"""Check native curved material operators and complete fields against independent elasticity."""

import argparse
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
from scipy import sparse
from scipy.linalg import qr
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skfem import Basis, ElementTetP2, ElementVector, MeshTet1, MeshTet2, asm
from skfem.helpers import sym_grad
from skfem.models.elasticity import lame_parameters, linear_elasticity, linear_stress

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from ..mechanics import shell_mesh
from ..skfem_reference import vector_mass
from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array, matrix


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "operator_relative": 1e-10,
        "same_load_field_relative": 1e-7,
        "uniform_field_relative": 1e-7,
        "quadrature_field_relative": 0.02,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "geometry": file_id(args.geometry),
            "targets": targets,
            "scope": "Native curved P2 shape operators, consistent inertia relief, complete corner/query stresses, max/full and cached/uncached gradients. Compare independent same-quadrature assembly and a higher-rule response; retain affine behavior separately. This checks supplied regular elastic loads, not curved Hertz source admission or contact pressure.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    xyz, tets, faces, _ = shell_mesh(1, 2, 0.0005)
    affine = MeshTet1(xyz.T, tets.T, sort_t=False)
    scalar = Basis(affine, ElementTetP2())
    element, lame = ElementVector(ElementTetP2()), lame_parameters(1e10, 0.3)
    edge_vertices, inverse = np.unique(
        np.sort(tets[:, list(combinations(range(4), 2))].reshape(-1, 2), axis=1), axis=0, return_inverse=True
    )
    bary_queries = np.array([[0.4, 0.2, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4], [0.2, 0.5, 0.1, 0.2]])
    rows = []
    for kind in ["analytic", "estimated"]:
        with np.load(args.geometry) as saved:
            nodes = saved[kind + "_volume_nodes_m"]
        edge_positions = np.empty((len(edge_vertices), 3))
        edge_positions[inverse] = nodes[:, 4:].reshape(-1, 3)
        np.testing.assert_allclose(edge_positions[inverse].reshape(-1, 6, 3), nodes[:, 4:], atol=1e-14)
        asset = args.output / (kind + ".npz")
        np.savez_compressed(
            asset, vertices=xyz, tetrahedra=tets, surface_triangles=faces, edge_positions=edge_positions
        )
        started = perf_counter()
        options = RigidStressOptions(
            mesh=asset, contact_model="hertz", method="direct", young=1e10, poisson=0.3, density=2000
        )
        model = StressModel(options)
        surface = StressSurface(8, model.info)
        qd.sync()
        build_seconds = perf_counter() - started
        coordinates = np.empty((scalar.N, 3))
        coordinates[scalar.element_dofs.T] = nodes[:, [0, 1, 2, 3, 4, 7, 5, 6, 8, 9]]
        mesh = MeshTet2(coordinates.T, tets.T, sort_t=False)
        native_xyz = array(model.info.vertices)
        distance, permutation = cKDTree(coordinates).query(native_xyz)
        assert distance.max() < 1e-12 and len(np.unique(permutation)) == len(coordinates)
        vector_permutation = (3 * permutation[:, None] + np.arange(3)).reshape(-1)
        native_k = (
            sparse.bsr_matrix(
                (array(model.info.stiffness), array(model.info.columns), array(model.info.row_start)),
                shape=(3 * len(coordinates), 3 * len(coordinates)),
            ).tocsr()
            * 1e10
        )
        native_m = sparse.kron(
            sparse.csr_matrix(
                (array(model.info.mass), array(model.info.columns), array(model.info.row_start)),
                shape=(len(coordinates), len(coordinates)),
            ),
            sparse.eye(3),
        ).tocsr()
        reference = []
        for order in [6, 7]:
            knots, weights = np.polynomial.legendre.leggauss(order)
            knots, weights = (knots + 1) / 2, weights / 2
            u, v, w = np.array(np.meshgrid(knots, knots, knots, indexing="ij")).reshape(3, -1)
            wu, wv, ww = np.array(np.meshgrid(weights, weights, weights, indexing="ij")).reshape(3, -1)
            points = np.array([u, (1 - u) * v, (1 - u) * (1 - v) * w])
            basis = Basis(mesh, element, quadrature=(points, wu * wv * ww * (1 - u) ** 2 * (1 - v)))
            stiffness = asm(linear_elasticity(*lame), basis).tocsr()
            mass = asm(vector_mass, basis, density=2000).tocsr()
            reference.append((basis, stiffness, mass))
        basis, stiffness, mass = reference[0]
        operator_error = sparse.linalg.norm(
            native_k - stiffness[vector_permutation][:, vector_permutation]
        ) / sparse.linalg.norm(stiffness)
        mass_error = sparse.linalg.norm(
            native_m - mass[vector_permutation][:, vector_permutation]
        ) / sparse.linalg.norm(mass)
        assert max(operator_error, mass_error) < targets["operator_relative"], (operator_error, mass_error)
        tensor = np.array([[1.2e6, 0.2e6, -0.15e6], [0.2e6, -0.6e6, 0.1e6], [-0.15e6, 0.1e6, 0.3e6]])
        strain = ((1 + 0.3) * tensor - 0.3 * np.trace(tensor) * np.eye(3)) / 1e10
        surface_nodes = array(model.info.surface_nodes).T
        positions = array(surface.info.positions)
        normals, weights, shape = array(surface.info.normals), array(surface.info.weights), array(surface.info.shape)
        load = np.zeros((3, len(coordinates), 3))
        traction = normals @ tensor.T
        nodal = np.einsum(
            "qn,fqi,fq->fni",
            shape,
            traction.reshape(len(surface_nodes), -1, 3),
            weights.reshape(len(surface_nodes), -1),
        )
        np.add.at(load[0], surface_nodes.reshape(-1), nodal.reshape(-1, 3))
        phase = positions / [0.017, 0.019, 0.023]
        traction = np.column_stack([np.sin(phase[:, 1]), np.cos(phase[:, 2]), -np.sin(phase[:, 0])]) * 100
        nodal = np.einsum(
            "qn,fqi,fq->fni",
            shape,
            traction.reshape(len(surface_nodes), -1, 3),
            weights.reshape(len(surface_nodes), -1),
        )
        np.add.at(load[1], surface_nodes.reshape(-1), nodal.reshape(-1, 3))
        exact = (native_xyz @ strain.T).reshape(-1)
        patch_error = np.linalg.norm(native_k @ exact - load[0].reshape(-1)) / np.linalg.norm(load[0])
        assert patch_error < targets["operator_relative"], patch_error
        expected_fields = []
        corner_points = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        corner_basis = Basis(mesh, element, quadrature=(corner_points, np.ones(4)))
        query_basis = Basis(mesh, element, quadrature=(bary_queries[:, 1:].T, np.ones(3)))
        omega_values = np.array([[0, 0, 0], [2, -3, 4], [0, 0, 0]], dtype=gs.np_float)
        for _, k, m in reference:
            dofs = np.stack(basis.split_indices(), axis=1)
            scalar_mass = m[dofs[:, 0]][:, dofs[:, 0]]
            nodal_mass = np.array(scalar_mass.sum(axis=1)).reshape(-1)
            relative = coordinates - (nodal_mass @ coordinates) / nodal_mass.sum()
            modes = np.zeros((basis.N, 6))
            modes[dofs, np.arange(3)[None]] = 1
            for axis in range(3):
                modes[dofs, axis + 3] = np.cross(np.eye(3)[axis], relative)
            mass_modes = m @ modes
            gram = modes.T @ mass_modes
            _, _, pivot = qr((modes / np.linalg.norm(modes, axis=0)).T, mode="economic", pivoting=True)
            free = np.setdiff1d(np.arange(basis.N), pivot[:6])
            factor = splu((k[free][:, free] / 1e10).tocsc())
            corners, queries = [], []
            for env in range(3):
                force = np.empty_like(load[env])
                force[permutation] = load[env]
                acceleration = -np.cross(omega_values[env], np.cross(omega_values[env], relative))
                rhs = force.reshape(-1) + m @ acceleration.reshape(-1)
                rhs -= mass_modes @ np.linalg.solve(gram, modes.T @ rhs)
                displacement = np.zeros(basis.N)
                displacement[free] = factor.solve(rhs[free] / 1e10)
                corners.append(
                    linear_stress(*lame)(sym_grad(corner_basis.interpolate(displacement))).transpose(2, 3, 0, 1)
                )
                queries.append(
                    linear_stress(*lame)(sym_grad(query_basis.interpolate(displacement))).transpose(2, 3, 0, 1)
                )
            expected_fields.append((np.array(corners), np.array(queries)))
        states = [model.create_state(3, mode) for mode in ["full", "max"]]
        omega = V_VEC(3, dtype=gs.qd_float, shape=(3,))
        omega.from_numpy(omega_values)
        hertz = StressHertz(3)
        hertz.state.status.fill(1)
        hertz.state.force.fill(0)
        errno = V(dtype=gs.qd_int, shape=(3,))
        query_positions = query_basis.mapping.F(bary_queries[:, 1:].T).transpose(1, 2, 0).reshape(-1, 3)
        modules, corner_peaks = [], []
        for state, mode, cached in zip(states, ["full", "max"], [True, False]):
            kernel_begin_step(state)
            state.force.from_numpy(load.transpose(1, 0, 2).copy())
            model.recover(omega, state, options.model_copy(update={"cached_peak": cached}))
            assert array(state.valid).all()
            corner_peaks.append(array(state.peak))
            if mode == "full":
                corners = matrix(array(state.stress_tensor))
                errors = [
                    np.linalg.norm(corners[i] - expected_fields[0][0][i])
                    / max(np.linalg.norm(expected_fields[0][0][i]), 1)
                    for i in range(3)
                ]
                assert max(errors) < targets["same_load_field_relative"], errors
                assert (
                    np.linalg.norm(corners[0] - tensor) / np.sqrt(len(tets) * 4) / np.linalg.norm(tensor)
                    < targets["uniform_field_relative"]
                )
            module = StressHertzRecovery(3, 1, len(query_positions), output_mode=mode)
            module.queries.position.from_numpy(query_positions)
            module.queries.element.from_numpy(np.repeat(np.arange(len(tets)), 3).astype(gs.np_int))
            module.queries.barycentric.from_numpy(np.tile(bary_queries, (len(tets), 1)))
            module.state.status.fill(1)
            errno.fill(0)
            module.evaluate(hertz.state, state, model, float(np.finfo(gs.np_float).eps), errno)
            np.testing.assert_array_equal(array(errno), 0)
            modules.append(module)
        queries = matrix(array(modules[0].state.tensor)).reshape(3, len(tets), 3, 3, 3)
        query_errors = [
            np.linalg.norm(queries[i] - expected_fields[0][1][i]) / max(np.linalg.norm(expected_fields[0][1][i]), 1)
            for i in range(3)
        ]
        assert max(query_errors) < targets["same_load_field_relative"], query_errors
        np.testing.assert_allclose(array(states[0].peak), array(states[1].peak), rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(corner_peaks[0], corner_peaks[1], rtol=1e-10, atol=1e-8)
        higher = np.linalg.norm(expected_fields[0][0][1] - expected_fields[1][0][1]) / np.linalg.norm(
            expected_fields[1][0][1]
        )
        assert higher < targets["quadrature_field_relative"], higher
        timings = {}
        for mode, state, module, cached in zip(["full", "max"], states, modules, [True, False]):
            samples = []
            selected = options.model_copy(update={"cached_peak": cached})
            for _ in range(5):
                qd.sync()
                started = perf_counter()
                model.recover(omega, state, selected)
                module.evaluate(hertz.state, state, model, float(np.finfo(gs.np_float).eps), errno)
                qd.sync()
                samples.append(perf_counter() - started)
            timings[mode] = samples
            np.testing.assert_array_equal(array(errno), 0)
        np.savez_compressed(
            args.output / (kind + "-fields.npz"),
            native_corners_pa=corners,
            native_queries_pa=queries,
            reference_corners_pa=expected_fields[0][0],
            reference_queries_pa=expected_fields[0][1],
            higher_corners_pa=expected_fields[1][0],
            load_n=load,
            nodes_m=native_xyz,
        )
        rows.append(
            {
                "geometry": kind,
                "build_seconds": build_seconds,
                "stiffness_relative": float(operator_error),
                "mass_relative": float(mass_error),
                "uniform_load_relative": float(patch_error),
                "corner_relative": [float(v) for v in errors],
                "query_relative": [float(v) for v in query_errors],
                "higher_rule_corner_relative": float(higher),
                "nodes": len(coordinates),
                "tetrahedra": len(tets),
                "regular_core_B3_inclusive_seconds": timings,
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
    invalid = edge_positions.copy()
    invalid[0] += [0, 0, 0.1]
    path = args.output / "inverted.npz"
    np.savez_compressed(path, vertices=xyz, tetrahedra=tets, surface_triangles=faces, edge_positions=invalid)
    rejected = False
    try:
        StressModel(options.model_copy(update={"mesh": path}))
    except gs.GenesisException as exc:
        rejected = "positive sampled geometry Jacobians" in str(exc)
    assert rejected
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "targets": targets,
            "inverted_geometry_rejected": rejected,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
