"""Refine collision integration with exact P2 traces and unchanged bulk unknowns."""

import ipctk
import numpy as np
from scipy import sparse


def contact_map(system, levels):
    assert levels in (0, 1, 2, 3)
    active = np.unique(system.faces)
    if levels == 0:
        node_map = sparse.coo_matrix(
            (np.ones(len(active)), (np.arange(len(active)), active)), shape=(len(active), len(system.rest))
        ).tocsr()
        faces = np.searchsorted(active, system.faces)
        groups = system.groups[active]
        return node_map, faces, groups
    body = system.body
    mesh = body.mesh
    boundary = mesh.facets[:, mesh.boundary_facets()].T
    edges = np.sort(mesh.edges.T, axis=1)
    edge_keys = edges[:, 0] * mesh.nvertices + edges[:, 1]
    order = np.argsort(edge_keys)
    n = 2 ** (levels + 1)
    rows, columns, values, faces, keys = [], [], [], [], {}

    def add_node(dofs, weights):
        selected = weights != 0
        pairs = sorted(zip(dofs[selected].tolist(), weights[selected].tolist()))
        key = tuple(pairs)
        if key not in keys:
            index = len(keys)
            keys[key] = index
            rows.extend([index] * len(pairs))
            columns.extend(dof for dof, _ in pairs)
            values.extend(weight for _, weight in pairs)
        return keys[key]

    for face in boundary:
        local_edges = np.sort(face[[[0, 1], [0, 2], [1, 2]]], axis=1)
        edge_indices = order[np.searchsorted(edge_keys[order], local_edges[:, 0] * mesh.nvertices + local_edges[:, 1])]
        dofs = np.r_[body.scalar_basis.dofs.nodal_dofs[0, face], body.scalar_basis.dofs.edge_dofs[0, edge_indices]]
        vertices = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                barycentric = np.array([n - i - j, i, j], dtype=float) / n
                weights = np.r_[
                    barycentric * (2 * barycentric - 1), 4 * barycentric[[0, 0, 1]] * barycentric[[1, 2, 2]]
                ]
                vertices[i, j] = add_node(dofs, weights)
        for i in range(n):
            for j in range(n - i):
                faces.append([vertices[i, j], vertices[i + 1, j], vertices[i, j + 1]])
                if i + j < n - 1:
                    faces.append([vertices[i + 1, j], vertices[i + 1, j + 1], vertices[i, j + 1]])
    body_nodes = len(keys)
    finger_active = active[active >= len(body.coordinates)]
    first = body_nodes
    rows.extend(range(first, first + len(finger_active)))
    columns.extend(finger_active.tolist())
    values.extend([1.0] * len(finger_active))
    finger_faces = system.faces[np.all(system.faces >= len(body.coordinates), axis=1)]
    faces.extend((np.searchsorted(finger_active, finger_faces) + first).tolist())
    node_map = sparse.coo_matrix(
        (values, (rows, columns)), shape=(body_nodes + len(finger_active), len(system.rest))
    ).tocsr()
    groups = np.r_[np.zeros(body_nodes, dtype=int), system.groups[finger_active]]
    return node_map, np.array(faces), groups


def refine_contact(system, levels):
    node_map, faces, groups = contact_map(system, levels)
    rest = node_map @ system.rest
    np.testing.assert_allclose(np.asarray(node_map.sum(axis=1)).ravel(), 1, rtol=0, atol=1e-15)
    edges = np.unique(np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
    mesh = ipctk.CollisionMesh(rest, edges, faces, node_map.tocsc())
    mesh.can_collide = ipctk.make_vertex_patches_filter(groups)
    np.testing.assert_allclose(mesh.displace_vertices(np.zeros_like(system.rest)), rest, rtol=0, atol=1e-14)
    system.mesh = mesh
    system.contact_node_map = node_map
    system.contact_surface_faces = faces
    system.contact_surface_groups = groups
    system.contact_surface_rest = rest
    return system


def collision_vertices(system, full_positions):
    """Use original displacements for IPC's non-square interpolation map.

    IPC1.6 vertices() subtracts collision-mesh full_rest_positions, whose count
    need not equal the interpolation's source count. displace_vertices() takes
    source displacements directly and avoids that incompatible subtraction.
    This expression also preserves the original affine arithmetic on velocities.
    """
    mapping = getattr(system, "contact_node_map", None)
    if mapping is not None:
        assert full_positions.shape == system.rest.shape
        return system.mesh.displace_vertices(full_positions - system.rest)
    return system.mesh.vertices(full_positions)
