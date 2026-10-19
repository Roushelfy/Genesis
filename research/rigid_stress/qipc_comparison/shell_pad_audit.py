"""Independent end-position pressure-law and work-transfer checks for the physical-pad trajectory."""

from dataclasses import dataclass

import numpy as np

from genesis.utils.misc import qd_to_numpy


@dataclass
class ShellPadAudit:
    points: np.ndarray
    corners: np.ndarray
    jacobians: np.ndarray
    area: np.ndarray
    faces: np.ndarray
    stiffness: float
    lower: float
    upper: float
    previous_normal: np.ndarray | None = None

    @classmethod
    def load(cls, path, faces, stiffness, lower, upper):
        with np.load(path) as saved:
            return cls(
                saved["points_m"],
                saved["corners_m"],
                saved["jacobians"],
                saved["projected_area_m2"],
                faces,
                stiffness,
                lower,
                upper,
            )

    def evaluate(
        self, solver, positions, vertex_force, compression, dt, velocity=None, friction=0.0, env=0, contact_data=None
    ):
        displacement = positions[self.faces] - self.corners
        points = self.points + np.einsum("fkij,fkj->fi", self.jacobians, displacement)
        top = self.points[:, 2] > 0
        sign = np.where(top, -1, 1)
        gap = np.where(top, self.upper - compression - points[:, 2], points[:, 2] - self.lower)
        expected = self.stiffness * self.area * np.maximum(-gap, 0)
        contact = solver._shell_contact
        geom = qd_to_numpy(contact.contacts_geom, transpose=True)[env] if contact_data is None else contact_data[0][env]
        impulse = (
            qd_to_numpy(contact.contacts_impulse, transpose=True, copy=True)[env]
            if contact_data is None
            else contact_data[1][env].copy()
        )
        impulse[geom < 0] = 0
        forces = impulse.reshape(-1, 2, 3).sum(axis=1) / dt
        scale = max(np.linalg.norm(forces), 1e-12)
        pressure_error = np.linalg.norm(sign * forces[:, 2] - expected) / scale
        nodal = np.zeros_like(vertex_force)
        local_force = np.einsum("fkij,fi->fkj", self.jacobians, forces)
        for corner in range(3):
            np.add.at(nodal, self.faces[:, corner], local_force[:, corner])
        transfer_error = np.linalg.norm(nodal - vertex_force) / scale
        row = {
            "normal_law_relative_error": float(pressure_error),
            "nodal_transfer_relative_error": float(transfer_error),
            "normal_forces_n": [float((sign * forces[:, 2])[top == upper].sum()) for upper in (False, True)],
            "compressed_projected_area_m2": [
                float(self.area[(top == upper) & (gap < 0)].sum()) for upper in (False, True)
            ],
            "minimum_body_gap_m": float(gap.min()),
            "normal_law_and_transfer_passed": bool(pressure_error <= 1e-7 and transfer_error <= 1e-10),
        }
        arrays = {
            "pad_points_world_m": points,
            "pad_point_force_n": forces,
            "pad_body_gap_m": gap,
            "pad_projected_area_m2": self.area,
        }
        if velocity is not None:
            point_velocity = np.einsum("fkij,fkj->fi", self.jacobians, velocity[self.faces])
            tangent_velocity = point_velocity[:, :2]
            speed = np.linalg.norm(tangent_velocity, axis=1)
            previous = np.zeros_like(expected) if self.previous_normal is None else self.previous_normal
            bound = friction * previous
            ratio = np.minimum(speed / 1e-4, 1.0)
            magnitude = bound * (2 * ratio - ratio**2)
            expected_tangent = -magnitude[:, None] * tangent_velocity / np.maximum(speed[:, None], 1e-30)
            tangent_error = float(np.linalg.norm(forces[:, :2] - expected_tangent) / scale)
            row["friction"] = {
                "lagged_law_relative_error": tangent_error,
                "instantaneous_cone_excess_n": float(
                    np.maximum(np.linalg.norm(forces[:, :2], axis=1) - friction * expected, 0).max()
                ),
                "tangential_resultant_n": forces[:, :2].sum(axis=0).tolist(),
                "sum_tangential_force_norms_n": float(np.linalg.norm(forces[:, :2], axis=1).sum()),
                "maximum_loaded_slip_speed_m_per_s": float(speed[expected > 0].max(initial=0)),
                "lagged_law_passed": tangent_error <= 1e-7,
            }
            arrays.update(
                pad_tangent_velocity_m_per_s=tangent_velocity,
                pad_previous_normal_force_n=previous,
                pad_expected_tangent_force_n=expected_tangent,
            )
            self.previous_normal = expected.copy()
        return row, arrays
