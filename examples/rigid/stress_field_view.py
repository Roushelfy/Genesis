"""Map recovered stress samples to an undeformed, independently colored material boundary."""

from pathlib import Path

import numpy as np
import trimesh

from examples.rigid.stress_mesh_asset import volume_boundary
from examples.rigid.stress_shell_asset import shell_boundary


class StressFieldSurface:
    """Color volume boundary corners or shell-side element maxima without smoothing between material elements.

    Colors span zero to a fixed limit in Pa, with gray for invalid samples. Volume faces retain their unique owning
    tetrahedron's three corner values. Each shell side uses its largest sampled Gauss-point value on both displayed
    triangles. The full-domain maximum is read independently from the native solver and can lie inside the volume.
    """

    def __init__(self, mesh: Path, backend: str, thickness: float = 0.0005):
        self.backend = backend
        with np.load(mesh, allow_pickle=False) as asset:
            vertices = asset["vertices"]
            if backend == "volume":
                triangles, self.owners, self.corners = volume_boundary(asset["tetrahedra"], vertices)
                self.material_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            else:
                quads = asset["quads"]
                self.material_mesh, _directors = shell_boundary(vertices, quads, asset.get("directors"), thickness)
                vertices = self.material_mesh.vertices
                triangles = self.material_mesh.faces[: 4 * len(quads)]
                self.owners = np.tile(np.repeat(np.arange(len(quads)), 2), 2)
                self.sides = np.repeat(np.arange(2), 2 * len(quads))
        if not self.material_mesh.is_volume:
            raise ValueError("The elastic boundary must enclose a consistently oriented material volume.")
        self.vertices = vertices[triangles].reshape((-1, 3))
        self.faces = np.arange(len(self.vertices)).reshape((-1, 3))

    def colored_mesh(self, von_mises: np.ndarray, limit: float) -> trimesh.Trimesh:
        """Return a boundary mesh colored by the supplied element samples and a fixed positive stress limit."""
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("The color scale maximum must be finite and positive, in Pa.")
        if self.backend == "volume":
            values = von_mises[self.owners[:, None], self.corners].reshape(-1)
        else:
            values = np.repeat(von_mises[self.owners, self.sides].max(axis=-1), 3)
        palette = (
            np.array(
                [[20, 10, 70, 255], [45, 80, 170, 255], [25, 180, 160, 255], [250, 210, 40, 255], [240, 40, 20, 255]]
            )
            / 255
        )
        normalized = np.clip(values / limit, 0, 1)
        colors = np.column_stack(
            [np.interp(normalized, np.linspace(0, 1, len(palette)), palette[:, channel]) for channel in range(4)]
        )
        colors[~np.isfinite(values)] = np.array([140, 140, 140, 255]) / 255
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, vertex_colors=colors, process=False)
