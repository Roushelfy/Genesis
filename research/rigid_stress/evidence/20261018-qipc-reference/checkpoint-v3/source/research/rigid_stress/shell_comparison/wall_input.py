"""Explicit research input adapter for geometric per-face wall mass and thickness; PR22 kernels remain unchanged."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from genesis.utils.misc import qd_to_torch

from .common import file_identity


@dataclass
class ShellWallInput:
    path: Path
    thickness: np.ndarray
    mass: np.ndarray
    maximum_midpoint_match_error_m: float

    @classmethod
    def bind(cls, path, entity, origin):
        with np.load(path) as saved:
            midpoint, faces = saved["mid_vertices_m"], saved["mid_faces"]
            reference_centers = midpoint[faces].mean(axis=1)
            actual_centers = entity.init_verts[entity.init_faces].mean(axis=1) - origin
            distance, order = cKDTree(reference_centers).query(actual_centers)
            tolerance = max(1e-14, 16 * np.finfo(entity.init_verts.dtype).eps * np.abs(entity.init_verts).max())
            assert distance.max() < tolerance and len(np.unique(order)) == len(faces)
            mass, thickness = saved["face_mass_kg"][order], saved["face_thickness_m"][order]
        assert (mass > 0).all() and (thickness > 0).all()
        np.testing.assert_allclose(
            mass,
            entity.material.rho * thickness * entity.topology.faces_rest_area,
            rtol=max(2e-12, 32 * np.finfo(entity.topology.faces_rest_area.dtype).eps),
            atol=1e-20,
        )
        entity.topology.faces_mass[:] = mass
        return cls(path, thickness, mass, distance.max().item())

    def apply_thickness(self, solver):
        view = qd_to_torch(solver.shell_state.faces_thickness, transpose=True, copy=False)
        view.copy_(torch.as_tensor(self.thickness, dtype=view.dtype, device=view.device)[None])

    def description(self):
        return {
            "input": file_identity(self.path),
            "mass_kg": self.mass.sum().item(),
            "maximum_midpoint_match_error_m": self.maximum_midpoint_match_error_m,
            "minimum_thickness_m": self.thickness.min().item(),
            "maximum_thickness_m": self.thickness.max().item(),
            "scope": "Separately labeled geometry-only rest-input adapter. Face masses are bound before build and existing thickness state is set before stepping/reset capture. No PR22 solver/contact/constitutive kernel modification.",
        }
