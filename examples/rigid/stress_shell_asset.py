"""Convert an orientable midsurface to conforming Q4 elements and a finite-thickness material boundary."""

import numpy as np
import pymeshlab
import trimesh


def check_surface(surface: trimesh.Trimesh, name: str, *, connected: bool) -> None:
    """Reject degenerate, nonmanifold or intersecting triangles without repairing their geometry."""
    points = surface.vertices[surface.faces]
    lengths = np.linalg.norm(np.roll(points, -1, axis=1) - points, axis=2)
    area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    if not len(points) or not np.isfinite(points).all() or (area <= 1e-12 * lengths.max(axis=1) ** 2).any():
        raise ValueError(f"{name} has nonfinite or degenerate triangles; repair the input mesh.")
    if len(np.unique(np.sort(surface.faces, axis=1), axis=0)) != len(surface.faces):
        raise ValueError(f"{name} contains duplicate triangles; remove duplicate faces.")
    _, counts = np.unique(surface.edges_sorted, axis=0, return_counts=True)
    if (counts > 2).any():
        raise ValueError(f"{name} has a nonmanifold edge; each edge must belong to at most two faces.")
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=surface.vertices, face_matrix=surface.faces))
    mesh_set.compute_selection_by_non_manifold_per_vertex()
    if mesh_set.current_mesh().selected_vertex_number():
        raise ValueError(f"{name} has a nonmanifold vertex; separate touching sheets or repair the vertex fan.")
    if connected and len(trimesh.graph.connected_components(surface.face_adjacency, nodes=np.arange(len(points)))) != 1:
        raise ValueError(
            f"{name} has disconnected components; the shell solver requires one connected material domain."
        )
    mesh_set.compute_selection_by_self_intersections_per_face()
    if mesh_set.current_mesh().selected_face_number():
        raise ValueError(f"{name} self-intersects; repair the surface or reduce shell thickness.")


def surface_to_quads(surface: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into three quads, preserving the midsurface, creases, holes and boundary edges.

    Only exactly coincident coordinates are joined, including the repeated vertices of STL triangle soups. There is no
    tolerance welding, projection or smoothing. The output has V + E + F vertices and 3F quadrilaterals.
    """
    vertices, inverse = np.unique(surface.vertices, axis=0, return_inverse=True)
    surface = trimesh.Trimesh(vertices=vertices, faces=inverse[surface.faces], process=False)
    surface.remove_unreferenced_vertices()
    check_surface(surface, "Shell midsurface", connected=True)
    trimesh.repair.fix_winding(surface)
    if not surface.is_winding_consistent:
        raise ValueError("Shell midsurface is not orientable; supply an orientable manifold surface.")
    if surface.is_watertight and surface.volume < 0:
        surface.invert()
    edges, inverse = np.unique(surface.edges_sorted, axis=0, return_inverse=True)
    midpoints = inverse.reshape((-1, 3)) + len(surface.vertices)
    centers = np.arange(len(surface.faces)) + len(surface.vertices) + len(edges)
    vertices = np.concatenate((surface.vertices, surface.vertices[edges].mean(axis=1), surface.triangles.mean(axis=1)))
    quads = np.stack(
        (surface.faces, midpoints, np.broadcast_to(centers[:, None], midpoints.shape), np.roll(midpoints, 1, axis=1)),
        axis=-1,
    ).reshape((-1, 4))
    return vertices, quads


def shell_boundary(
    vertices: np.ndarray, quads: np.ndarray, directors: np.ndarray | None, thickness: float
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build both offset shell sides and open-edge caps, validating orientation and material intersections.

    Side faces are ordered positive then negative, with two triangles per quad; caps follow the sides. Directors are
    normalized or derived from incident quad area vectors, exactly as in the native shell model. Thickness is in metres.
    """
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("Shell thickness must be finite and positive, in metres.")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Shell vertices must be a finite (n, 3) array.")
    if (
        quads.ndim != 2
        or quads.shape[1] != 4
        or not len(quads)
        or not np.issubdtype(quads.dtype, np.integer)
        or quads.min() < 0
        or quads.max() >= len(vertices)
    ):
        raise ValueError("A shell asset needs nonempty in-range integer quads.")
    if len(np.unique(quads)) != len(vertices):
        raise ValueError("Every shell vertex must belong to a quadrilateral.")
    points = vertices[quads]
    if directors is None:
        normal = np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 1])
        directors = np.zeros_like(vertices)
        np.add.at(directors, quads.reshape(-1), np.repeat(normal, 4, axis=0))
    if directors.shape != vertices.shape or not np.isfinite(directors).all():
        raise ValueError("Shell directors must match the finite midsurface coordinates.")
    lengths = np.linalg.norm(directors, axis=1)
    if (lengths == 0).any():
        raise ValueError("Shell directors cancel at a fold; repair the surface orientation or fold geometry.")
    directors = directors / lengths[:, None]
    outer = quads[:, [[0, 1, 2], [0, 2, 3]]].reshape((-1, 3))
    midsurface = trimesh.Trimesh(vertices=vertices, faces=outer, process=False)
    check_surface(midsurface, "Q4 midsurface", connected=True)
    if not midsurface.is_winding_consistent:
        raise ValueError("Q4 shell winding is inconsistent; orient adjacent elements consistently.")

    # Check element corners as well as the native integration points, on both offset sides and through the thickness.
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    for xi in (-1.0, -1 / np.sqrt(3), 0.0, 1 / np.sqrt(3), 1.0):
        for eta in (-1.0, -1 / np.sqrt(3), 0.0, 1 / np.sqrt(3), 1.0):
            shape = (1 + signs[:, 0] * xi) * (1 + signs[:, 1] * eta) / 4
            gradient = (
                np.column_stack((signs[:, 0] * (1 + signs[:, 1] * eta), signs[:, 1] * (1 + signs[:, 0] * xi))) / 4
            )
            tangents = np.einsum("eni,nj->eij", points, gradient)
            normal = np.cross(tangents[:, :, 0], tangents[:, :, 1])
            local_directors = np.einsum("n,eni->ei", shape, directors[quads])
            derivative = np.einsum("eni,nj->eij", directors[quads], gradient)
            quadratic = np.einsum("ei,ei->e", np.cross(derivative[:, :, 0], derivative[:, :, 1]), local_directors)
            linear = np.einsum(
                "ei,ei->e",
                np.cross(tangents[:, :, 0], derivative[:, :, 1]) + np.cross(derivative[:, :, 0], tangents[:, :, 1]),
                local_directors,
            )
            stationary = np.zeros(len(quads))
            np.divide(-linear, 2 * quadratic, out=stationary, where=quadratic > 0)
            stationary = np.clip(stationary, -0.5 * thickness, 0.5 * thickness)
            for offset in (np.full(len(quads), -0.5 * thickness), stationary, np.full(len(quads), 0.5 * thickness)):
                shifted = tangents + offset[:, None, None] * derivative
                product = np.cross(shifted[:, :, 0], shifted[:, :, 1])
                determinant = np.einsum("ei,ei->e", product, local_directors)
                scale = np.linalg.norm(shifted[:, :, 0], axis=1) * np.linalg.norm(shifted[:, :, 1], axis=1)
                if (determinant <= 1e-8 * scale).any() or (np.einsum("ei,ei->e", normal, product) <= 0).any():
                    raise ValueError(
                        "Shell offset has a collapsed or inverted Q4 Jacobian; improve the mesh or reduce thickness."
                    )
    offset = 0.5 * thickness * directors
    boundary_vertices = np.concatenate((vertices + offset, vertices - offset))
    inner = outer[:, ::-1] + len(vertices)
    edges = quads[:, [[0, 1], [1, 2], [2, 3], [3, 0]]].reshape((-1, 2))
    _, first, counts = np.unique(np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True)
    rim = edges[first[counts == 1]]
    a, b = rim[:, 0], rim[:, 1]
    caps = np.concatenate(
        (np.column_stack((a + len(vertices), b + len(vertices), b)), np.column_stack((a + len(vertices), b, a)))
    )
    material = trimesh.Trimesh(vertices=boundary_vertices, faces=np.concatenate((outer, inner, caps)), process=False)
    check_surface(material, "Finite-thickness shell boundary", connected=False)
    if not material.is_volume or (np.linalg.eigvalsh(material.moment_inertia) <= 0).any():
        raise ValueError(
            "The shell boundary must enclose positive material volume and inertia; check thickness and directors."
        )
    return material, directors
