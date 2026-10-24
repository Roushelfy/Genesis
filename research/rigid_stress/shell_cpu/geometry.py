"""Small deterministic shell fixtures and exact smooth egg curvature.

The quad mesh samples a smooth reference surface.  It is not a contact solve.
All dimensions are SI; quads have outward counter-clockwise orientation.
"""
from __future__ import annotations

import numpy as np


def plate_mesh(nx=8, ny=None, Lx=1.0, Ly=None):
    ny = nx if ny is None else ny
    Ly = Lx if Ly is None else Ly
    x = np.linspace(-Lx / 2, Lx / 2, int(nx) + 1)
    y = np.linspace(-Ly / 2, Ly / 2, int(ny) + 1)
    vertices = np.array([[a, b, 0.0] for b in y for a in x])
    quads = []
    for j in range(int(ny)):
        for i in range(int(nx)):
            a = j * (int(nx) + 1) + i
            quads.append([a, a + 1, a + int(nx) + 2, a + int(nx) + 1])
    return vertices, np.asarray(quads, dtype=np.int64)


def _cube_directions(n):
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    index = {}
    vertices, quads = [], []
    # Face coordinate frames: e1 x e2 equals its outward normal.
    frames = [
        ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
        ([-1, 0, 0], [0, -1, 0], [0, 0, 1]),
        ([0, 1, 0], [-1, 0, 0], [0, 0, 1]),
        ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
        ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
        ([0, 0, -1], [-1, 0, 0], [0, 1, 0]),
    ]
    for normal, first, second in frames:
        normal, first, second = map(np.asarray, (normal, first, second))
        grid = np.empty((n + 1, n + 1), dtype=int)
        for j in range(n + 1):
            for i in range(n + 1):
                p = n * normal + (2 * i - n) * first + (2 * j - n) * second
                key = tuple(p)
                if key not in index:
                    index[key] = len(vertices)
                    vertices.append(p / np.linalg.norm(p))
                grid[j, i] = index[key]
        for j in range(n):
            for i in range(n):
                quads.append([grid[j, i], grid[j, i + 1], grid[j + 1, i + 1], grid[j + 1, i]])
    return np.asarray(vertices), np.asarray(quads, dtype=np.int64)


def sphere_mesh(n=4, radius=0.02):
    v, q = _cube_directions(n)
    return v * radius, q


def egg_mesh(n=4, axes=(0.018, 0.018, 0.026), asym=0.15):
    """Smooth asymmetric ellipsoid; positive asym fattens the +z end."""
    v, q = _cube_directions(n)
    a, b, c = map(float, axes)
    scale = 1.0 + asym * v[:, 2]
    return np.column_stack((a * scale * v[:, 0], b * scale * v[:, 1], c * v[:, 2])), q


def egg_geometry(points, axes=(0.018, 0.018, 0.026), asym=0.15):
    """Outward normals and two signed principal curvatures of the smooth egg.

    F=x²/[a²(1+beta*z/c)²]+y²/[b²(1+beta*z/c)²]+z²/c²-1.
    The tangent shape operator is P Hess(F) P / ||grad(F)||.
    Points should lie on this surface; mesh-face interiors are only an
    approximation, so callers should disclose their geometry sampling choice.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    a, b, c = map(float, axes)
    x, y, z = p.T
    k = asym / c
    w = 1.0 + k * z
    T = (x / a) ** 2 + (y / b) ** 2
    grad = np.column_stack((2*x/(a*a*w*w), 2*y/(b*b*w*w), -2*k*T/w**3 + 2*z/(c*c)))
    lengths = np.linalg.norm(grad, axis=1)
    normals = grad / lengths[:, None]
    H = np.zeros((len(p), 3, 3))
    H[:, 0, 0] = 2 / (a*a*w*w)
    H[:, 1, 1] = 2 / (b*b*w*w)
    H[:, 2, 2] = 6*k*k*T/w**4 + 2/(c*c)
    H[:, 0, 2] = H[:, 2, 0] = -4*k*x/(a*a*w**3)
    H[:, 1, 2] = H[:, 2, 1] = -4*k*y/(b*b*w**3)
    curvature = np.empty((len(p), 2))
    for i, normal in enumerate(normals):
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        first = np.cross(normal, axis)
        first /= np.linalg.norm(first)
        tangent = np.column_stack((first, np.cross(normal, first)))
        curvature[i] = np.linalg.eigvalsh(tangent.T @ H[i] @ tangent / lengths[i])
    return normals, curvature


def sphere_geometry(points, radius=0.02):
    p = np.atleast_2d(np.asarray(points, dtype=float))
    return p / np.linalg.norm(p, axis=1)[:, None], np.full((len(p), 2), 1.0 / radius)


def vertex_normals(vertices, quads):
    p = np.asarray(vertices)[np.asarray(quads)]
    f = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0]) + np.cross(p[:, 2] - p[:, 1], p[:, 0] - p[:, 1])
    n = np.zeros_like(vertices, dtype=float)
    for i in range(4):
        np.add.at(n, quads[:, i], f)
    return n / np.linalg.norm(n, axis=1)[:, None]


if __name__ == "__main__":
    for n in (1, 2, 4, 8):
        v, q = sphere_mesh(n)
        pn = np.cross(v[q[:, 1]] - v[q[:, 0]], v[q[:, 3]] - v[q[:, 0]])
        assert np.all(np.sum(pn * v[q].mean(axis=1), axis=1) > 0)
        edges = np.sort(np.concatenate([q[:, [i, (i+1) % 4]] for i in range(4)]), axis=1)
        _, count = np.unique(edges, axis=0, return_counts=True)
        assert np.all(count == 2)
        normal, k = sphere_geometry(v)
        en, ek = egg_geometry(v, axes=(.02, .02, .02), asym=0)
        assert np.max(np.abs(k-ek)) < 1e-12
        assert np.max(np.abs(normal-en)) < 1e-14
        print(n, len(v), len(q), "geometry passed")
