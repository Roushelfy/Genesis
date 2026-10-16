"""Refine the same polyhedral egg while retaining actual fingers and held points."""

import numpy as np
from scipy.spatial import cKDTree

from .franka_contact_reference import RigidFingerContactSystem
from .skfem_reference import SkfemElasticReference


def refine_body(system, levels):
    assert levels in (0, 1, 2)
    original = system.body
    held = np.flatnonzero(np.isin(system.parts[0].permutation, original.gauge_dofs))
    fixed_positions = original.coordinates[held // 3]
    fixed_axes = held % 3
    if not levels:
        system.fixed_initializer_points_m = fixed_positions
        system.fixed_initializer_components = fixed_axes
        return system
    mesh = original.mesh.refined(levels)
    body = SkfemElasticReference(mesh.p.T, mesh.t.T, original.young, original.poisson, original.density)
    np.testing.assert_allclose(body.total_mass, original.total_mass, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(body.com, original.com, rtol=0, atol=1e-13)
    distance, nodes = cKDTree(body.coordinates).query(fixed_positions)
    assert distance.max() < 1e-12
    body.gauge_dofs = body.component_dofs[nodes, fixed_axes]
    assert len(np.unique(body.gauge_dofs)) == 6
    body.free_dofs = np.setdiff1d(np.arange(body.basis.N), body.gauge_dofs)
    vertices, faces, centres = [], [], []
    for index, indices in enumerate(system.finger_nodes):
        local = indices[:-1]
        vertices.append(system.rest[local])
        selected = system.faces[system.groups[system.faces[:, 0]] == index + 1]
        assert np.all(np.isin(selected, local))
        faces.append(selected - local[0])
        centres.append(system.rest[indices[-1]])
    result = RigidFingerContactSystem(
        body, vertices, faces, system.axes, system.masses, centres, system.dhat, system.friction
    )
    result.world_coordinate_origin_m = system.world_coordinate_origin_m.copy()
    result.fixed_initializer_points_m = fixed_positions
    result.fixed_initializer_components = fixed_axes
    return result
