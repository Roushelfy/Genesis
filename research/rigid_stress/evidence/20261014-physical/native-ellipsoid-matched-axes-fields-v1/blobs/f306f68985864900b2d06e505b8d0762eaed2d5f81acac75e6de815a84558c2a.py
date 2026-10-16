"""Refine rigid finger collision triangles by exact linear interpolation of their existing nodes."""

import ipctk
import numpy as np
from scipy import sparse


def refine_rigid_finger_trace(system, levels):
    assert levels in (0, 1, 2, 3)
    if levels == 0:
        return system
    old_map = system.contact_node_map
    old_faces = system.contact_surface_faces
    old_groups = system.contact_surface_groups
    body = np.flatnonzero(old_groups == 0)
    body_map = old_map[body].tocsr()
    remap = np.full(len(old_groups), -1, dtype=int)
    remap[body] = np.arange(len(body))
    body_faces = old_faces[np.all(old_groups[old_faces] == 0, axis=1)]
    faces = remap[body_faces].tolist()
    keys, rows, columns, values, groups = {}, [], [], [], []
    n = 2**levels
    for face in old_faces[np.all(old_groups[old_faces] != 0, axis=1)]:
        group = int(old_groups[face[0]])
        assert group in (1, 2) and np.all(old_groups[face] == group)
        local = old_map[face].tocsr()
        nodes = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                bary = np.array([n - i - j, i, j], dtype=float) / n
                combination = (sparse.csr_matrix(bary[None]) @ local).tocsr()
                combination.eliminate_zeros()
                combination.sort_indices()
                assert np.all(combination.data >= 0)
                key = (group, tuple(zip(combination.indices.tolist(), combination.data.tolist(), strict=True)))
                if key not in keys:
                    index = len(keys)
                    keys[key] = index
                    rows.extend([index] * len(combination.data))
                    columns.extend(combination.indices.tolist())
                    values.extend(combination.data.tolist())
                    groups.append(group)
                nodes[i, j] = len(body) + keys[key]
        for i in range(n):
            for j in range(n - i):
                faces.append([nodes[i, j], nodes[i + 1, j], nodes[i, j + 1]])
                if i + j < n - 1:
                    faces.append([nodes[i + 1, j], nodes[i + 1, j + 1], nodes[i, j + 1]])
    finger_map = sparse.coo_matrix((values, (rows, columns)), shape=(len(keys), old_map.shape[1])).tocsr()
    mapping = sparse.vstack((body_map, finger_map), format="csr")
    faces = np.asarray(faces, dtype=int)
    groups = np.r_[np.zeros(len(body), dtype=int), groups]
    rest = mapping @ system.rest
    np.testing.assert_allclose(np.asarray(mapping.sum(axis=1)).ravel(), 1, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(rest[: len(body)], system.contact_surface_rest[body])
    edges = np.unique(np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
    mesh = ipctk.CollisionMesh(rest, edges, faces, mapping.tocsc())
    mesh.can_collide = ipctk.make_vertex_patches_filter(groups)
    np.testing.assert_allclose(mesh.displace_vertices(np.zeros_like(system.rest)), rest, rtol=0, atol=1e-14)
    for group in [0, 1, 2]:
        areas = []
        for points, triangles, labels in [
            (system.contact_surface_rest, old_faces, old_groups),
            (rest, faces, groups),
        ]:
            tri = points[triangles[labels[triangles[:, 0]] == group]]
            areas.append(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
        np.testing.assert_allclose(areas[0], areas[1], rtol=1e-12, atol=1e-18)
    system.mesh = mesh
    system.contact_node_map = mapping
    system.contact_surface_faces = faces
    system.contact_surface_groups = groups
    system.contact_surface_rest = rest
    return system
