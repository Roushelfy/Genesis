"""Refine body volume while holding the seed contact triangles and physical nodes fixed."""

import ipctk
import numpy as np
from scipy.spatial import cKDTree


def freeze_body_contact(seed, refined):
    seed_count = len(seed.body.coordinates)
    body_count = len(refined.body.coordinates)
    seed_faces = seed.faces[np.all(seed.faces < seed_count, axis=1)]
    distance, nodes = cKDTree(refined.body.coordinates).query(seed.body.coordinates)
    assert distance.max() <= 1e-12
    assert len(np.unique(nodes)) == seed_count
    body_faces = nodes[seed_faces]
    np.testing.assert_allclose(refined.rest[body_faces], seed.rest[seed_faces], rtol=0, atol=1e-12)
    finger_faces = refined.faces[np.all(refined.faces >= body_count, axis=1)]
    refined.faces = np.vstack((body_faces, finger_faces))
    edges = np.unique(np.sort(refined.faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
    refined.mesh = ipctk.CollisionMesh.build_from_full_mesh(refined.rest, edges, refined.faces)
    refined.mesh.can_collide = ipctk.make_vertex_patches_filter(refined.groups[refined.mesh.to_full_vertex_id()])
    refined.frozen_body_contact_faces = body_faces.copy()
    refined.frozen_seed_body_coordinates = seed.body.coordinates.copy()
    return refined
