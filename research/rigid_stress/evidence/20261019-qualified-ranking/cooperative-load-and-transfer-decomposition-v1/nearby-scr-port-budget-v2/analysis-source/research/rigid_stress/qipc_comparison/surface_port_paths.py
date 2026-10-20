"""Independent CPU surface paths through mesh-identified edge ports."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


@dataclass
class SurfacePortPaths:
    vertices: np.ndarray
    faces: np.ndarray
    positions: np.ndarray
    face_ports: np.ndarray
    graph: sparse.csr_matrix

    @classmethod
    def build(cls, vertices, faces, subdivisions):
        assert subdivisions >= 1
        used, corner_ports = np.unique(faces, return_inverse=True)
        edges, edge_ids = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
                                   axis=0, return_inverse=True)
        fraction = np.arange(1, subdivisions) / subdivisions
        inner = (vertices[edges[:, 0], None] * (1 - fraction)[None, :, None] +
                 vertices[edges[:, 1], None] * fraction[None, :, None])
        positions = np.concatenate((vertices[used], inner.reshape(-1, 3)))
        edge_ports = len(used) + edge_ids[:, None] * len(fraction) + np.arange(len(fraction))
        ports = np.concatenate((corner_ports.reshape(-1, 3), edge_ports.reshape(len(faces), -1)), axis=1)
        first = np.broadcast_to(ports[:, :, None], (len(faces), ports.shape[1], ports.shape[1]))
        second = np.broadcast_to(ports[:, None, :], first.shape)
        pairs = np.unique(np.column_stack((first.reshape(-1), second.reshape(-1))), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        lengths = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
        graph = sparse.csr_matrix((lengths, (pairs[:, 0], pairs[:, 1])), shape=(len(positions), len(positions)))
        return cls(vertices, faces, positions, ports, graph)

    def distances(self, anchor, face, barycentric):
        samples = np.einsum("qc,fci->fqi", barycentric, self.vertices[self.faces])
        initial = np.linalg.norm(self.positions[self.face_ports[face]] - anchor, axis=1)
        # One extra source vertex supplies the exact anchor-to-face-port edges.
        tail = sparse.csr_matrix((initial, (np.zeros(len(initial), dtype=int), self.face_ports[face])),
                                 shape=(1, len(self.positions)))
        augmented = sparse.bmat([[self.graph, tail.T], [tail, None]], format="csr")
        distance = dijkstra(augmented, directed=False, indices=len(self.positions))[:-1]
        upper = np.empty(samples.shape[:2])
        for index, points in enumerate(samples):
            last = np.linalg.norm(points[:, None] - self.positions[self.face_ports[index]][None], axis=-1)
            upper[index] = np.min(last + distance[self.face_ports[index]][None], axis=1)
            shared = np.intersect1d(self.faces[face], self.faces[index])
            if len(shared) == 2:
                first, second = self.vertices[shared]
                edge = second - first
                length = np.linalg.norm(edge)
                direction = edge / length
                start_coordinate = (anchor - first) @ direction
                end_coordinate = (points - first) @ direction
                start_height = np.linalg.norm(anchor - first - start_coordinate * direction)
                end_height = np.linalg.norm(points - first - end_coordinate[:, None] * direction, axis=1)
                height = start_height + end_height
                crossing = np.divide(start_height * end_coordinate + end_height * start_coordinate,
                                     height, out=np.zeros_like(height), where=height > 0)
                crossing_point = first + np.clip(crossing, 0, length)[:, None] * direction
                exact = np.linalg.norm(crossing_point - anchor, axis=1) + np.linalg.norm(points - crossing_point, axis=1)
                upper[index] = np.minimum(upper[index], exact)
        upper[face] = np.linalg.norm(samples[face] - anchor, axis=1)
        assert np.all(upper >= np.linalg.norm(samples - anchor, axis=-1) - 1e-12)
        return upper


def analytic_controls():
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                         [0., 0, 0], [1., 0, 0], [0., 0, 1]])
    faces = np.array([[0, 1, 2], [0, 3, 1], [4, 6, 5]])
    bary = np.array([[0.5, 0.2, 0.3]])
    anchor = np.array([0.2, 0.2, 0.])
    coarse = SurfacePortPaths.build(vertices, faces, 4).distances(anchor, 0, bary)
    fine = SurfacePortPaths.build(vertices, faces, 8).distances(anchor, 0, bary)
    np.testing.assert_allclose(coarse[1, 0], np.sqrt(0.1**2 + 0.4**2), rtol=0, atol=1e-14)
    np.testing.assert_allclose(fine[1, 0], coarse[1, 0], rtol=0, atol=1e-14)
    assert np.isinf(coarse[2, 0]) and np.isinf(fine[2, 0])
    return {"folded_shared_edge_exact_distance": True, "coincident_disconnected_sheet_excluded": True,
            "adjacent_unfolding_independent_of_port_spacing": True}
