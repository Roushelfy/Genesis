"""Independent conforming full3D elasticity reference for sticking flat contact controls."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import FacetBasis, MeshTet
from skfem.helpers import sym_grad
from skfem.models.elasticity import linear_stress

from .skfem_reference import SkfemElasticReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--shear", type=float, default=5e-6)
    parser.add_argument("--closing", type=float, default=50e-6)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    axis = np.linspace(-0.005, 0.005, args.resolution)
    vertical = np.linspace(-0.005, 0.005, 2 * args.resolution - 1)
    mesh = MeshTet.init_tensor(axis, axis, vertical)
    reference = SkfemElasticReference(mesh.p.T, mesh.t.T, 1e6, 0, 1000)
    nodes = np.flatnonzero(np.isclose(abs(reference.coordinates[:, 2]), 0.005))
    constrained = reference.component_dofs[nodes].reshape(-1)
    values = np.zeros((len(nodes), 3))
    upper = reference.coordinates[nodes, 2] > 0
    values[upper, 0] = args.shear
    values[upper, 2] = -args.closing
    displacement = np.zeros(reference.basis.N)
    displacement[constrained] = values.reshape(-1)
    free = np.setdiff1d(np.arange(reference.basis.N), constrained)
    displacement[free] = spsolve(
        reference.stiffness[free][:, free],
        -(reference.stiffness[free][:, constrained] @ values.reshape(-1)),
    )
    reactions = reference.stiffness @ displacement
    residual = np.linalg.norm(reactions[free])
    assert residual <= 1e-9
    tensor, vm = reference.corner_stress(displacement)
    centre_facets = np.flatnonzero(np.isclose(reference.mesh.p[2, reference.mesh.facets], 0).all(axis=0))
    interface = FacetBasis(reference.mesh, reference.element, facets=centre_facets, intorder=4)
    stress = linear_stress(*reference.lame)(sym_grad(interface.interpolate(displacement)))
    # Positive z is the lower material's outward interface normal.
    traction = stress[:, 2].transpose(1, 2, 0)
    positions = interface.global_coordinates().transpose(1, 2, 0)
    resultant = np.sum(traction * interface.dx[:, :, None], axis=(0, 1))
    moment = np.sum(np.cross(positions, traction) * interface.dx[:, :, None], axis=(0, 1))
    pressure = -traction[..., 2]
    shear = np.linalg.norm(traction[..., :2], axis=-1)
    cone_margin = args.friction * pressure - shear
    bottom_nodes = np.flatnonzero(np.isclose(reference.coordinates[:, 2], -0.005))
    bottom_reaction = reactions[reference.component_dofs[bottom_nodes]].sum(axis=0)
    interface_force = -bottom_reaction
    interface_moment = -np.cross(
        reference.coordinates[bottom_nodes], reactions[reference.component_dofs[bottom_nodes]]
    ).sum(axis=0)
    np.testing.assert_allclose(reactions[reference.component_dofs].sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(
        np.cross(reference.coordinates, reactions[reference.component_dofs]).sum(axis=0), 0, atol=1e-12
    )
    corner_margins = []
    for side in (0, 1):
        corner_basis = FacetBasis(
            reference.mesh,
            reference.element,
            facets=centre_facets,
            side=side,
            quadrature=(np.array([[0.0, 1, 0], [0.0, 0, 1]]), np.ones(3)),
        )
        corner_stress = linear_stress(*reference.lame)(sym_grad(corner_basis.interpolate(displacement)))
        corner_margins.append(-args.friction * corner_stress[2, 2] - np.linalg.norm(corner_stress[:2, 2], axis=0))
    probes = np.stack(
        np.meshgrid(
            [-0.002, 0, 0.002], [-0.002, 0, 0.002], [-0.004, -0.002, -0.0005, 0.0005, 0.002, 0.004], indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 3)
    probe_tensor, probe_vm = reference.stress_at_points(displacement, probes)
    element, corner = np.unravel_index(np.argmax(vm), vm.shape)
    np.savez_compressed(
        args.output / "fields.npz",
        vertices=mesh.p.T,
        tetrahedra=mesh.t.T,
        displacement=displacement,
        tensor_Pa=tensor,
        vm_Pa=vm,
        interface_positions_m=positions,
        interface_tractions_Pa=traction,
        interface_areas_m2=interface.dx,
        probes_m=probes,
        probe_tensor_Pa=probe_tensor,
        probe_vm_Pa=probe_vm,
    )
    result = {
        "scope": "Conforming zero-gap bonded interface reference; valid for Coulomb sticking only with positive cone margin",
        "resolution": args.resolution,
        "young_Pa": 1e6,
        "poisson": 0,
        "closing_m": args.closing,
        "shear_m": args.shear,
        "friction_for_admissibility": args.friction,
        "complete_free_residual_N": float(residual),
        "interface_force_N": interface_force.tolist(),
        "interface_moment_Nm": interface_moment.tolist(),
        "stress_integral_interface_force_N": resultant.tolist(),
        "stress_integral_interface_moment_Nm": moment.tolist(),
        "stress_integral_force_relative_difference": float(
            np.linalg.norm(resultant - interface_force) / np.linalg.norm(interface_force)
        ),
        "minimum_pressure_Pa": float(pressure.min()),
        "maximum_shear_Pa": float(shear.max()),
        "minimum_coulomb_margin_Pa": float(cone_margin.min()),
        "minimum_both_sides_corner_coulomb_margin_Pa": float(np.min(corner_margins)),
        "raw_global_peak_Pa": float(vm.max()),
        "raw_global_peak_position_m": reference.corner_positions[element, corner].tolist(),
        "strain_energy_J": float(0.5 * displacement @ reactions),
    }
    if args.shear == 0:
        np.testing.assert_allclose(resultant, [0, 0, -0.5], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(vm, 5000, rtol=1e-10, atol=1e-7)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
