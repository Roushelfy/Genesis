"""Construct independent finite-area boundary loads on full three-dimensional FEM meshes."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from skfem import FacetBasis, LinearForm, asm
from skfem.helpers import dot
from skfem.quadrature import get_quadrature_tri


@LinearForm
def prescribed_vector_traction(v, w):
    return dot(w.traction, v)


@dataclass(frozen=True)
class FinitePatch:
    name: str
    centre_m: np.ndarray
    direction: np.ndarray
    strength_n: float
    radius_m: float
    friction: float
    shear_fraction: float = 0.0
    is_local_normal_profile: bool = False


@dataclass(frozen=True)
class PatchTractionReference:
    specification: FinitePatch
    witness_m: np.ndarray
    point_m: np.ndarray
    force_n: np.ndarray
    free_moment_nm: np.ndarray
    inward_normal: np.ndarray
    positions_m: np.ndarray
    normals: np.ndarray
    area_weights_m2: np.ndarray
    tractions_pa: np.ndarray
    load_vector_n: np.ndarray
    profile_integral_m2: float
    boundary_facets: np.ndarray


def triangle_closest_points(triangles, point):
    """Project one point onto each closed triangle, including all edges."""
    edges = triangles[:, 1:] - triangles[:, :1]
    gram = np.einsum("fai,fbi->fab", edges, edges)
    local = np.linalg.solve(gram, np.einsum("fai,fi->fa", edges, point - triangles[:, 0]))
    projection = triangles[:, 0] + np.einsum("fa,fai->fi", local, edges)
    segments = triangles[:, [[0, 1], [1, 2], [2, 0]]]
    vectors = segments[:, :, 1] - segments[:, :, 0]
    fractions = np.clip(
        np.einsum("fei,fei->fe", point - segments[:, :, 0], vectors) / np.sum(vectors * vectors, axis=2), 0, 1
    )
    candidates = np.concatenate((projection[:, None], segments[:, :, 0] + fractions[:, :, None] * vectors), axis=1)
    distances = np.sum((candidates - point) ** 2, axis=2)
    distances[~((local >= 0).all(axis=1) & (local.sum(axis=1) <= 1)), 0] = np.inf
    selected = np.argmin(distances, axis=1)
    return candidates[np.arange(len(triangles)), selected], distances[np.arange(len(triangles)), selected]


def subdivided_triangle_quadrature(level, order=6):
    """Integrate each triangle using independently subdivided scikit-fem rules."""
    triangles = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    for _ in range(level):
        first, second, third = triangles.transpose(1, 0, 2)
        first_second = (first + second) / 2
        first_third = (first + third) / 2
        second_third = (second + third) / 2
        triangles = np.stack(
            (
                np.stack((first, first_second, first_third), axis=1),
                np.stack((first_second, second, second_third), axis=1),
                np.stack((first_third, second_third, third), axis=1),
                np.stack((first_second, second_third, first_third), axis=1),
            ),
            axis=1,
        ).reshape(-1, 3, 2)
    points, weights = get_quadrature_tri(order)
    edges = triangles[:, 1:] - triangles[:, :1]
    transformed = triangles[:, :1] + np.einsum("nai,aq->nqi", edges, points)
    transformed_weights = abs(np.linalg.det(edges))[:, None] * weights
    return transformed.reshape(-1, 2).T, transformed_weights.reshape(-1)


def assemble_finite_patch(reference, specification, quadrature_level):
    """Integrate a declared compact profile on its witnessed connected boundary sheet."""
    boundary = reference.mesh.boundary_facets()
    faces = reference.mesh.facets[:, boundary].T
    triangles = reference.mesh.p.T[faces]
    closest, distance_squared = triangle_closest_points(triangles, specification.centre_m)
    anchor = np.argmin(distance_squared)
    selected = np.flatnonzero(distance_squared < specification.radius_m**2)
    if anchor not in selected:
        raise ValueError(f"Physical footprint misses the material boundary: {specification.name}")
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    pairs = np.flatnonzero((edges[order][1:] == edges[order][:-1]).all(axis=1))
    neighbours = np.repeat(np.arange(len(faces)), 3)[order]
    adjacency = sparse.csr_matrix(
        (np.ones(len(pairs)), (neighbours[pairs], neighbours[pairs + 1])), shape=(len(faces), len(faces))
    )
    _, components = connected_components(adjacency[selected][:, selected], directed=False)
    selected = selected[components == components[np.flatnonzero(selected == anchor)[0]]]
    boundary_facets = boundary[selected]
    quadrature = subdivided_triangle_quadrature(quadrature_level)
    basis = FacetBasis(reference.mesh, reference.element, facets=boundary_facets, quadrature=quadrature)
    positions = basis.global_coordinates().transpose(1, 2, 0)
    inward = -basis.normals.transpose(1, 2, 0)
    radius_squared = np.sum((positions - specification.centre_m) ** 2, axis=2) / specification.radius_m**2
    profile = np.exp(-radius_squared / (2 * 0.45**2)) * np.maximum(1 - radius_squared, 0) ** 2
    integral = np.sum(profile * basis.dx)
    if integral <= 0:
        raise ValueError(f"Independent quadrature leaves the footprint unresolved: {specification.name}")
    direction = specification.direction / np.linalg.norm(specification.direction)
    traction = profile[:, :, None] * direction
    if specification.is_local_normal_profile:
        shear_direction = np.array([0.0, 0.0, 1.0])
        tangent = shear_direction - np.einsum("fqi,i->fq", inward, shear_direction)[:, :, None] * inward
        tangent /= np.maximum(np.linalg.norm(tangent, axis=2), 1e-30)[:, :, None]
        traction = profile[:, :, None] * (inward + specification.shear_fraction * specification.friction * tangent)
    traction *= specification.strength_n / integral
    load = asm(prescribed_vector_traction, basis, traction=traction.transpose(2, 0, 1))
    weighted = traction * basis.dx[:, :, None]
    force = weighted.sum(axis=(0, 1))
    centroid = np.sum(positions * (profile * basis.dx)[:, :, None], axis=(0, 1)) / integral
    moment = np.cross(positions - centroid, weighted).sum(axis=(0, 1))
    material_centres = reference.mesh.p.T[reference.mesh.t[:, reference.mesh.f2t[0, boundary[anchor]]]].mean(axis=0)
    normal = np.cross(triangles[anchor, 1] - triangles[anchor, 0], triangles[anchor, 2] - triangles[anchor, 0])
    normal *= np.sign(normal @ (material_centres - triangles[anchor].mean(axis=0)))
    normal /= np.linalg.norm(normal)
    active = profile > 0
    return PatchTractionReference(
        specification,
        closest[anchor],
        centroid,
        force,
        moment,
        normal,
        positions[active],
        inward[active],
        basis.dx[active],
        traction[active],
        load,
        integral,
        boundary_facets,
    )


def physical_grip_patches(geometry, case, radius_m=0.003):
    """Declare fixed physical patch centres and finite profiles for each full geometry."""
    if geometry == "ellipsoid":
        centres = np.array([[-0.012, 0, 0], [0.012, 0, 0]])
    elif geometry == "egg":
        centres = np.array([[-0.022, 0, 0], [0.022, 0, 0]])
    elif geometry == "capsule":
        centres = np.array([[-0.008, 0, 0], [0.008, 0, 0]])
    elif geometry == "thin_neck":
        centres = np.array([[-0.022, 0, 0], [0.022, 0, 0]])
    elif geometry in ("u", "narrow_gap"):
        half_gap = 0.005 if geometry == "u" else 0.0005
        centres = np.array([[-half_gap, 0.018, 0], [half_gap, 0.018, 0]])
    elif geometry == "cavity":
        centres = np.array([[-0.008, 0, 0], [0.008, 0, 0]])
    else:
        raise ValueError(f"Unknown full geometry: {geometry}")
    directions = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    if geometry in ("u", "narrow_gap", "cavity"):
        directions *= -1
    strengths = [0.05, 0.05]
    if case == "bending":
        directions[:, 2] = [0.3, -0.3]
    elif case == "asymmetric":
        strengths[1] = 0.08
        centres[1, 2] += 0.0005
    elif case not in ("opposing", "near_slip", "multiple"):
        raise ValueError(f"Unknown finite load case: {case}")
    patches = [
        FinitePatch(
            f"pad{i_p}",
            centre,
            direction,
            strength,
            radius_m,
            0.8 if case == "near_slip" else 2.0,
            0.95 if case == "near_slip" else 0.0,
            case == "near_slip",
        )
        for i_p, (centre, direction, strength) in enumerate(zip(centres, directions, strengths))
    ]
    if case == "multiple":
        patches.append(FinitePatch("pad2", centres[0] + [0, 0, 0.003], directions[0], 0.02, radius_m, 2.0))
    return tuple(patches)
