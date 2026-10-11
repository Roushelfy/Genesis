"""Fit finite surface tractions to a contact wrench with unilateral Coulomb constraints.

A point contact does not uniquely identify a pressure distribution. The declared law minimizes the weighted change
from a compact Gaussian profile while preserving all six wrench components. Infeasible wrenches raise an error.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class WrenchPatch:
    center_m: np.ndarray
    force_n: np.ndarray
    radius_m: float
    friction: float
    moment_nm: np.ndarray
    inward_normal: np.ndarray | None = None
    source_epsilon: float = 0.0


@dataclass(frozen=True)
class WrenchDiagnostics:
    force_error_n: float
    moment_error_nm: float
    minimum_normal_force_n: float
    maximum_cone_excess_n: float
    constrained_evaluations: int
    source_cone_excess_n: float = 0.0
    friction_roundoff_allowance_n: float = 0.0


@dataclass(frozen=True)
class PatchMapping:
    nodal_force_n: np.ndarray
    quadrature_idx: np.ndarray
    quadrature_force_n: np.ndarray
    diagnostics: WrenchDiagnostics
    footprint_center_m: np.ndarray


def project_coulomb(vectors: np.ndarray, normals: np.ndarray, friction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean projection and its Jacobian for local inward circular friction cones.

    Vectors are sample force densities or forces in N/m2 or N. Normals must be unit inward directions. The cone requires
    nonnegative compression and tangential magnitude <= friction times compression at every sample.
    """

    normal = np.einsum("qi,qi->q", vectors, normals)
    tangent = vectors - normal[:, None] * normals
    magnitude = np.linalg.norm(tangent, axis=1)
    direction = tangent / np.maximum(magnitude[:, None], np.finfo(float).tiny)
    compression = (normal + friction * magnitude) / (1 + friction**2)
    cone_direction = normals + friction * direction
    projected = compression[:, None] * cone_direction
    normal_projector = normals[:, :, None] * normals[:, None, :]
    tangent_projector = direction[:, :, None] * direction[:, None, :]
    jacobian = cone_direction[:, :, None] * cone_direction[:, None, :] / (1 + friction**2)
    derivative_scale = np.divide(friction * compression, magnitude, out=np.zeros_like(magnitude), where=magnitude > 0)
    jacobian += derivative_scale[:, None, None] * (np.eye(3) - normal_projector - tangent_projector)
    is_interior = (normal >= 0) & (magnitude <= friction * normal)
    is_polar = normal + friction * magnitude <= 0
    projected[is_interior] = vectors[is_interior]
    jacobian[is_interior] = np.eye(3)
    projected[is_polar] = 0
    jacobian[is_polar] = 0
    return projected, jacobian


class WrenchFit:
    """Solve the six-dimensional dual of a weighted friction-constrained wrench fit.

    For normalized positive sample weights w, minimize sum ||f_q-w_q F||^2/(2 w_q), subject to the required force/moment
    and each sample's inward friction cone. Moments are scaled by patch radius to avoid a dimensionally mixed norm.
    """

    def __init__(self, positions: np.ndarray, weights: np.ndarray, normals: np.ndarray, patch: WrenchPatch):
        self.weights = weights / weights.sum()
        self.normals = normals
        self.patch = patch
        relative = (positions - patch.center_m) / patch.radius_m
        skew = np.zeros((len(positions), 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -relative[:, 2], relative[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = relative[:, 2], -relative[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -relative[:, 1], relative[:, 0]
        self.operator = np.concatenate((np.broadcast_to(np.eye(3), skew.shape), skew), axis=1)
        self.target = np.concatenate((patch.force_n, patch.moment_nm / patch.radius_m))
        self.scale = max(np.linalg.norm(self.target), 1e-12)

    def residual(self, multiplier: np.ndarray) -> np.ndarray:
        vectors = self.patch.force_n - np.einsum("qij,i->qj", self.operator, multiplier)
        projected, _ = project_coulomb(vectors, self.normals, self.patch.friction)
        return (np.einsum("qij,qj,q->i", self.operator, projected, self.weights) - self.target) / self.scale

    def jacobian(self, multiplier: np.ndarray) -> np.ndarray:
        vectors = self.patch.force_n - np.einsum("qij,i->qj", self.operator, multiplier)
        _, derivative = project_coulomb(vectors, self.normals, self.patch.friction)
        return -np.einsum("qij,qjk,qlk,q->il", self.operator, derivative, self.operator, self.weights) / self.scale

    def solve(self) -> tuple[np.ndarray, WrenchDiagnostics]:
        gram = np.einsum("qij,qkj,q->ik", self.operator, self.operator, self.weights)
        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
            raise ValueError("The footprint quadrature does not span a well-conditioned six-component wrench")
        initial = np.einsum("qij,j,q->i", self.operator, self.patch.force_n, self.weights)
        multiplier = np.linalg.solve(gram, initial - self.target)
        residual = self.residual(multiplier)
        evaluations = 0
        if np.linalg.norm(residual) > 1e-10:
            result = least_squares(
                self.residual,
                multiplier,
                jac=self.jacobian,
                ftol=1e-13,
                xtol=1e-13,
                gtol=1e-13,
                max_nfev=300,
            )
            multiplier = result.x
            evaluations = result.nfev
        vectors = self.patch.force_n - np.einsum("qij,i->qj", self.operator, multiplier)
        projected, _ = project_coulomb(vectors, self.normals, self.patch.friction)
        sample_force = self.weights[:, None] * projected
        mismatch = np.einsum("qij,qj->i", self.operator, sample_force) - self.target
        force_error = np.linalg.norm(mismatch[:3])
        moment_error = np.linalg.norm(mismatch[3:]) * self.patch.radius_m
        normal = np.einsum("qi,qi->q", sample_force, self.normals)
        tangent = sample_force - normal[:, None] * self.normals
        excess = np.linalg.norm(tangent, axis=1) - self.patch.friction * normal
        if force_error > 1e-8 * self.scale or moment_error > 1e-8 * self.scale * self.patch.radius_m:
            raise ValueError(
                f"Contact wrench infeasible or unresolved by the finite footprint: force error={force_error:.6g} N, "
                f"moment error={moment_error:.6g} Nm; refine quadrature or supply a justified load model"
            )
        return sample_force, WrenchDiagnostics(force_error, moment_error, normal.min(), excess.max(), evaluations)


class PadPressureFit:
    """Fit nonnegative pressure with a constant resultant direction in the rigid contact frame.

    A compliant pad distributes a constant tangential-to-normal traction ratio on the fixed reference surface. The
    pressure minimizes its weighted squared change from the Gaussian profile. Its centroid lies on the original force
    line, preserving the point wrench exactly. Pressure and friction are defined in the supplied pad frame, which can
    differ from reference face normals. The roundoff allowance is 16 source epsilons times the resultant force norm.
    """

    def __init__(self, positions: np.ndarray, weights: np.ndarray, patch: WrenchPatch):
        self.positions = positions
        self.weights = weights / weights.sum()
        self.patch = patch

    def solve(self) -> tuple[np.ndarray, WrenchDiagnostics]:
        patch = self.patch
        normal = np.asarray(patch.inward_normal, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 1e-12:
            raise ValueError("A finite nonzero inward pad normal is required")
        if np.linalg.norm(patch.moment_nm) > 1e-15:
            raise ValueError(
                "Constant-ratio pad pressure supports point wrenches. Pure moments require vector tractions"
            )
        normal = normal / np.linalg.norm(normal)
        magnitude = np.linalg.norm(patch.force_n)
        if magnitude <= 1e-30:
            return np.zeros_like(self.positions), WrenchDiagnostics(0, 0, 0, 0, 0)
        compression = normal @ patch.force_n
        cone_excess = np.linalg.norm(patch.force_n - compression * normal) - patch.friction * compression
        allowance = 16 * max(patch.source_epsilon, np.finfo(float).eps) * magnitude
        if compression < -allowance or cone_excess > allowance:
            raise ValueError("The complete rigid force violates its supplied pad friction cone beyond source roundoff")
        direction = patch.force_n / magnitude
        tangent = np.cross(direction, np.eye(3)[np.argmin(abs(direction))])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(direction, tangent)
        relative = (self.positions - patch.center_m) / patch.radius_m
        operator = np.column_stack((np.ones(len(relative)), relative @ tangent, relative @ bitangent))
        target = np.array([1.0, 0.0, 0.0])
        gram = operator.T @ (self.weights[:, None] * operator)
        if np.linalg.eigvalsh(gram)[0] <= 1e-12 * np.linalg.eigvalsh(gram)[-1]:
            raise ValueError("Pad quadrature needs independent positions transverse to the complete force")
        multiplier = np.linalg.solve(gram, operator.T @ self.weights - target)
        evaluations = 0
        for evaluations in range(1, 81):
            profile = np.maximum(0, 1 - operator @ multiplier)
            gradient = target - operator.T @ (self.weights * profile)
            if np.linalg.norm(gradient) <= 2e-12:
                break
            active_weights = self.weights * (profile > 0)
            hessian = operator.T @ (active_weights[:, None] * operator)
            step = np.linalg.lstsq(hessian, -gradient, rcond=1e-12)[0]
            objective = 0.5 * (self.weights @ profile**2) + target @ multiplier
            fraction = 1.0
            for _ in range(40):
                candidate = multiplier + fraction * step
                candidate_profile = np.maximum(0, 1 - operator @ candidate)
                candidate_objective = 0.5 * (self.weights @ candidate_profile**2) + target @ candidate
                if candidate_objective <= objective + 1e-4 * fraction * (gradient @ step) + 1e-15:
                    multiplier = candidate
                    break
                fraction *= 0.5
            else:
                break
        pressure = self.weights * np.maximum(0, 1 - operator @ multiplier)
        force = pressure[:, None] * patch.force_n
        force_error = np.linalg.norm(force.sum(axis=0) - patch.force_n)
        moment_error = np.linalg.norm(np.cross(self.positions - patch.center_m, force).sum(axis=0))
        if force_error > 1e-8 * magnitude or moment_error > 1e-8 * magnitude * patch.radius_m:
            raise ValueError(
                f"The pad footprint cannot preserve this point wrench: force error={force_error:.6g} N, "
                f"moment error={moment_error:.6g} Nm"
            )
        return force, WrenchDiagnostics(
            force_error,
            moment_error,
            float(pressure.min() * compression),
            float((pressure * cone_excess).max()),
            evaluations,
            float(cone_excess),
            float(allowance),
        )


class FinitePatchMapper:
    """Map changing exterior footprints while preserving force, moment and local friction admissibility.

    The profile has sigma=0.45 radius and squared compact support. Inward normals follow the affine shell faces. A pure
    moment is relative to center_m. P2 nodal weights remain signed; positivity applies to physical quadrature tractions.
    """

    def __init__(
        self,
        geometry,
        anchor_to_surface: bool = False,
        surface_traction: bool = False,
        adaptive_integration: bool = False,
    ):
        self.geometry = geometry
        self.anchor_to_surface = anchor_to_surface
        self.surface_traction = surface_traction
        self.adaptive_integration = adaptive_integration
        self.positions = geometry.coords.reshape(-1, 3)
        self.weights = geometry.integration_weights.reshape(-1)
        self.tree = cKDTree(self.positions)
        vertices = geometry.f.base_xyz[geometry.f.outer_faces]
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals *= np.where(np.einsum("ij,ij->i", normals, vertices.mean(axis=1) - geometry.f.com) > 0, -1, 1)[:, None]
        self.normals = np.repeat(normals, geometry.shape.shape[0], axis=0)
        self.face_vertices = vertices
        self.face_normals = -normals
        edges = vertices[:, 1:] - vertices[:, :1]
        gram = np.einsum("fai,fbi->fab", edges, edges)
        self.face_dual = np.einsum("fab,fbi->fai", np.linalg.inv(gram), edges)

    def refined_quadrature(self, patch: WrenchPatch):
        """Partition the anchor face around a central triangle whose centroid is the anchor."""
        local = np.einsum("fai,fi->fa", self.face_dual, patch.center_m - self.face_vertices[:, 0])
        plane = np.einsum("fi,fi->f", self.face_normals, patch.center_m - self.face_vertices[:, 0])
        eligible = (abs(plane) < 1e-12) & (local.min(axis=1) >= -2e-12) & (local.sum(axis=1) <= 1 + 2e-12)
        faces = np.flatnonzero(eligible)
        if not len(faces):
            raise ValueError("The anchored footprint needs an exterior containing face")
        i_face = faces[0]
        anchor = np.r_[1 - local[i_face].sum(), local[i_face]]
        inner = (1 - anchor[:, None]) * anchor[None] + np.diag(anchor)
        unit = np.eye(3)
        triangles = np.stack(
            (
                inner,
                np.stack((unit[0], unit[1], inner[1])),
                np.stack((unit[0], inner[1], inner[0])),
                np.stack((unit[1], unit[2], inner[2])),
                np.stack((unit[1], inner[2], inner[1])),
                np.stack((unit[2], unit[0], inner[0])),
                np.stack((unit[2], inner[0], inner[2])),
            )
        )
        bary = np.einsum("qa,tai->tqi", self.geometry.bary, triangles)
        vertices = triangles @ self.face_vertices[i_face]
        jacobian = np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1)
        positions = bary @ self.face_vertices[i_face]
        weights = jacobian[:, None] * self.geometry.weights[None] / 2
        shape = np.concatenate(
            (
                bary * (2 * bary - 1),
                (4 * bary[:, :, 0] * bary[:, :, 1])[:, :, None],
                (4 * bary[:, :, 0] * bary[:, :, 2])[:, :, None],
                (4 * bary[:, :, 1] * bary[:, :, 2])[:, :, None],
            ),
            axis=2,
        )
        count = self.geometry.shape.shape[0]
        ids = np.array(self.tree.query_ball_point(patch.center_m, patch.radius_m, return_sorted=True))
        ids = ids[ids // count != i_face]
        sample_ids = np.r_[ids, len(self.positions) + np.arange(7 * count)]
        positions = np.concatenate((self.positions[ids], positions.reshape(-1, 3)))
        weights = np.r_[self.weights[ids], weights.reshape(-1)]
        shape = np.concatenate((self.geometry.shape[ids % count], shape.reshape(-1, 6)))
        nodes = self.geometry.nodes[np.r_[ids // count, np.full(7 * count, i_face)]]
        distance = np.sum((positions - patch.center_m) ** 2, axis=1) / patch.radius_m**2
        weights *= np.exp(-0.5 * distance / 0.45**2) * np.maximum(0, 1 - distance) ** 2
        active = (distance < 1) & (weights > 0)
        return positions[active], weights[active], shape[active], nodes[active], sample_ids[active]

    def surface_anchor(self, center: np.ndarray, force: np.ndarray) -> np.ndarray:
        """Intersect a force line with the outgoing exterior of the fixed convex reference shell.

        Moving a point along its complete resultant leaves its wrench unchanged. This is an explicit footprint law,
        useful when rigid collision points lie inside the reference surface during penetration. No force or radius is
        changed. An absent intersection is a model failure, never a nearest-point or enlarged-radius fallback.
        """
        direction = -force / np.linalg.norm(force)
        denominator = self.face_normals @ direction
        outgoing = denominator > 1e-12
        parameter = np.divide(
            np.einsum("fi,fi->f", self.face_vertices[:, 0] - center, self.face_normals),
            denominator,
            out=np.full(len(denominator), -np.inf),
            where=outgoing,
        )
        candidates = np.flatnonzero(outgoing)
        relative = center + parameter[candidates, None] * direction - self.face_vertices[candidates, 0]
        barycentric = np.einsum("fai,fi->fa", self.face_dual[candidates], relative)
        inside = (barycentric >= -2e-12).all(axis=1) & (barycentric.sum(axis=1) <= 1 + 2e-12)
        if not inside.any():
            raise ValueError("The complete force line does not intersect the outgoing reference shell")
        return center + parameter[candidates[inside]].max() * direction

    def map(self, patch: WrenchPatch) -> PatchMapping:
        if (
            patch.center_m.shape != (3,)
            or patch.force_n.shape != (3,)
            or patch.moment_nm.shape != (3,)
            or not np.isfinite(np.concatenate((patch.center_m, patch.force_n, patch.moment_nm))).all()
            or not np.isfinite(patch.radius_m)
            or patch.radius_m <= 0
            or not np.isfinite(patch.friction)
            or patch.friction < 0
            or not np.isfinite(patch.source_epsilon)
            or patch.source_epsilon < 0
        ):
            raise ValueError("Finite vectors, positive finite radius and nonnegative finite friction are required")
        if patch.inward_normal is not None and (
            np.asarray(patch.inward_normal).shape != (3,)
            or not np.isfinite(patch.inward_normal).all()
            or np.linalg.norm(patch.inward_normal) <= 1e-12
        ):
            raise ValueError("A finite nonzero inward pad normal is required")
        if not np.any(patch.force_n) and not np.any(patch.moment_nm):
            return PatchMapping(
                np.zeros(self.geometry.f.ndof),
                np.empty(0, dtype=np.int64),
                np.empty((0, 3)),
                WrenchDiagnostics(0, 0, 0, 0, 0),
                patch.center_m.copy(),
            )
        if self.anchor_to_surface:
            if patch.inward_normal is None or np.any(patch.moment_nm):
                raise ValueError("Force-line anchoring requires the point-wrench pad law")
            center = self.surface_anchor(patch.center_m, patch.force_n)
            patch = WrenchPatch(
                center,
                patch.force_n,
                patch.radius_m,
                patch.friction,
                patch.moment_nm,
                patch.inward_normal,
                patch.source_epsilon,
            )
        ids = np.array(self.tree.query_ball_point(patch.center_m, patch.radius_m, return_sorted=True))
        if len(ids) < 3:
            raise ValueError("The footprint needs at least three quadrature samples; refine contact integration")
        distance = np.sum((self.positions[ids] - patch.center_m) ** 2, axis=1) / patch.radius_m**2
        weights = np.exp(-0.5 * distance / 0.45**2) * np.maximum(0, 1 - distance) ** 2 * self.weights[ids]
        is_active = weights > 0
        ids, weights = ids[is_active], weights[is_active]
        if weights.sum() <= 0:
            raise ValueError("The finite footprint has no resolved area")
        samples_per_face = self.geometry.shape.shape[0]
        shape = self.geometry.shape[ids % samples_per_face]
        nodes = self.geometry.nodes[ids // samples_per_face]
        if patch.inward_normal is None or self.surface_traction:
            force, diagnostics = WrenchFit(self.positions[ids], weights, self.normals[ids], patch).solve()
        else:
            try:
                force, diagnostics = PadPressureFit(self.positions[ids], weights, patch).solve()
            except ValueError:
                if not self.adaptive_integration:
                    raise
                positions, weights, shape, nodes, ids = self.refined_quadrature(patch)
                force, diagnostics = PadPressureFit(positions, weights, patch).solve()
        contribution = shape[:, :, None] * force[:, None, :]
        nodal = np.zeros_like(self.geometry.f.xyz)
        np.add.at(nodal, nodes.reshape(-1), contribution.reshape(-1, 3))
        return PatchMapping(nodal.reshape(-1), ids, force, diagnostics, patch.center_m)
