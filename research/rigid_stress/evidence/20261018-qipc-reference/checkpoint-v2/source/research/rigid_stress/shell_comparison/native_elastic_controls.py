"""Common thin-plate analytic stress fields, independently assembled P2 FEM and native full output."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from ..skfem_reference import SkfemElasticReference
from .common import identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    young, thickness, curvature = 1e7, 0.001, 2.0
    mesh = MeshTet.init_tensor(
        np.linspace(-0.05, 0.05, 5), np.linspace(-0.01, 0.01, 3), np.linspace(-thickness / 2, thickness / 2, 3)
    )
    vertices, tetrahedra = mesh.p.T, mesh.t.T
    boundary = mesh.facets[:, mesh.boundary_facets()].T
    path = args.output / "plate.npz"
    np.savez_compressed(path, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=boundary)
    reference = SkfemElasticReference(vertices, tetrahedra, young, 0.0, 1000.0)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    options = RigidStressOptions(
        mesh=path, young=young, poisson=0.0, density=1000.0, output_mode="full", method="direct"
    )
    model = StressModel(options)
    state = model.create_state(1, "full")
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    omega.fill(0)
    coordinates = qd_to_numpy(model.info.vertices, copy=True)
    permutation = reference.native_dof_permutation(coordinates)
    reports = []
    for kind, value in (("stretch", 1000.0), ("compression", -1000.0), ("bending", curvature)):
        tensor = np.diag([value, 0.0, 0.0])
        force = reference.pure_bending_load(value) if kind == "bending" else reference.uniform_stress_load(tensor)
        independent = reference.solve(force)
        assert independent.residual_budget_fraction <= 1.0
        native_force = force[permutation].reshape(-1, 3)
        state.force.from_numpy(native_force[:, None])
        model.recover(omega, state)
        assert qd_to_numpy(state.valid, transpose=True).all()
        packed = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
        native_vm = qd_to_numpy(state.von_mises, transpose=True, copy=True)[0, ..., 0]
        expected = np.zeros((*packed.shape[:-1], 6))
        expected[..., 0] = young * curvature * vertices[tetrahedra, 2] if kind == "bending" else value
        independent_packed = independent.tensor_pa[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        np.savez_compressed(
            args.output / f"{kind}.npz",
            native_tensor_pa=packed,
            native_vm_pa=native_vm,
            independent_tensor_pa=independent.tensor_pa,
            expected_tensor_pa=expected,
            native_load_n=native_force,
            reference_load_n=force,
            positions_m=vertices[tetrahedra],
        )
        reports.append(
            {
                "case": kind,
                "native_max_tensor_error_pa": float(np.abs(packed - expected).max()),
                "independent_max_tensor_error_pa": float(np.abs(independent_packed - expected).max()),
                "native_vs_independent_max_pa": float(np.abs(packed - independent_packed).max()),
                "native_peak_pa": float(native_vm.max()),
                "independent_peak_pa": float(independent.peak_pa),
                "independent_residual_n": float(independent.residual_n),
                "independent_residual_budget_fraction": float(independent.residual_budget_fraction),
                "net_load_n": native_force.sum(axis=0).tolist(),
                "load_moment_about_origin_nm": np.cross(coordinates, native_force).sum(axis=0).tolist(),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "native-analytic-plate",
            "identity": identity(),
            "young_pa": young,
            "poisson": 0.0,
            "thickness_m": thickness,
            "tetrahedra": len(tetrahedra),
            "dofs": int(reference.basis.N),
            "cases": reports,
        },
    )


if __name__ == "__main__":
    main()
