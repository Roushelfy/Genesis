"""Curved surface regions with exact, common-mother response spaces.

For each region, U_i = K_g^{-1} R E_i: R is the mother's inertia relief,
K_g its pinned stiffness, and E_i the supported boundary generalized loads.
Thus U_i f_i is also the Galerkin solution in range(U_i), including its global
extension through the complete wall. Dependent response columns need no new
factorization because f_i supplies their exact source coordinates. No local
rigid gauge, mass approximation or independent physical surface is introduced.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V
from genesis.utils.misc import qd_to_numpy

from .contact import StressContactState
from .atlas_coarse import StressAtlasCoarse
from .data import StressAtlasScatter, StressInfo, StressState
from .quadrilateral import func_quad_inverse_prepared
from .surface import StressSurfaceInfo
from .surface_inverse import StressSurfaceInverseInfo

if TYPE_CHECKING:
    from .atlas_output import StressAtlasOutput
    from .model import StressModel


@dataclass(frozen=True)
class StressAtlasInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    starts: qd.Tensor
    neighbors: qd.Tensor
    distances: qd.Tensor
    seeds: qd.Tensor
    radii: qd.Tensor
    coverage: qd.Tensor
    force_rows: qd.Tensor


@dataclass(frozen=True)
class StressAtlasState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    weights: qd.Tensor
    fallback: qd.Tensor
    force: qd.Tensor
    displacement: qd.Tensor
    columns: qd.Tensor
    counts: qd.Tensor
    pairs: qd.Tensor
    count: qd.Tensor


class StressAtlas:
    def __init__(
        self,
        model: "StressModel",
        contacts: StressContactState,
        n_features: int,
        width: float,
        partition: str = "direct",
    ):
        if model.surface_inverse is None:
            gs.raise_exception("A response atlas requires the FP64 native surface inverse.")
        n_nodes = model.info.vertices.shape[0]
        n_physical = model.load_info.vertices.shape[0]
        n_boundary = model.surface_inverse.info.nodes.shape[0]
        n_contacts, n_envs = contacts.valid.shape
        n_groups = n_features + 1  # Group zero retains the complete mother fallback.
        n_face_nodes = 4 if model.n_shell_elements else 6
        faces = qd_to_numpy(model.load_info.surface_nodes)[:, :n_face_nodes]
        # This is only surface topology. Distances and feature preparation run in Quadrants.
        pairs = np.stack(np.broadcast_arrays(faces[:, :, None], faces[:, None, :]), axis=-1)
        pairs = np.unique(pairs.reshape((-1, 2)), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        starts = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=n_physical))]
        self.info = StressAtlasInfo(
            starts=V(dtype=gs.qd_int, shape=(n_physical + 1,)),
            neighbors=V(dtype=gs.qd_int, shape=(len(pairs),)),
            distances=V(dtype=gs.qd_float, shape=(n_features, n_physical)),
            seeds=V(dtype=gs.qd_int, shape=(n_features,)),
            radii=V(dtype=gs.qd_float, shape=(n_features,)),
            coverage=V(dtype=gs.qd_bool, shape=(n_features, n_physical)),
            force_rows=V(dtype=gs.qd_int, shape=(n_nodes,)),
        )
        self.info.starts.from_numpy(starts.astype(gs.np_int, copy=False))
        self.info.neighbors.from_numpy(pairs[:, 1].astype(gs.np_int, copy=False))
        visited = V(dtype=gs.qd_bool, shape=(n_physical,))
        kernel_distances(model.load_info, self.info, visited)
        kernel_coverage(width, model.load_info, self.info)
        kernel_force_rows(model.surface_inverse.info, self.info)
        self.state = StressAtlasState(
            weights=V(dtype=gs.qd_float, shape=(n_groups, n_contacts, n_envs)),
            fallback=V(dtype=gs.qd_bool, shape=(n_contacts, n_envs)),
            force=V_VEC(3, dtype=gs.qd_float, shape=(n_boundary, n_groups, n_envs)),
            displacement=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes, n_groups, n_envs)),
            columns=V(dtype=gs.qd_int, shape=(n_boundary, n_groups, n_envs)),
            counts=V(dtype=gs.qd_int, shape=(n_groups, n_envs)),
            pairs=V_VEC(2, dtype=gs.qd_int, shape=(n_groups * n_envs,)),
            count=V(dtype=gs.qd_int, shape=()),
        )
        self.response = model.surface_inverse.info
        self.coarse = StressAtlasCoarse(model, n_envs) if partition == "coarse_local" else None
        if self.coarse is not None:
            self.response = self.coarse.response
        self.state.weights.fill(0)
        self.scatter = StressAtlasScatter(self.state.weights, self.state.force, self.info.force_rows)
        self.output: StressAtlasOutput | None = None

    def prepare(self, contacts: StressContactState, load_info: StressInfo, surface: StressSurfaceInfo, width: float):
        kernel_select(width, contacts, load_info, surface, self.info, self.state)

    def apply(self, young: float, omega: qd.Tensor, state: StressState, only_failed: bool = False):
        if self.coarse is not None:
            self.coarse.apply(state)
        kernel_pack(state, self.response, self.state, only_failed)
        kernel_apply(young, self.response, self.state)
        kernel_merge(young, omega, state, self.response, self.state, self.coarse, only_failed)


@qd.kernel
def kernel_force_rows(response: StressSurfaceInverseInfo, atlas: StressAtlasInfo):
    for i_n in range(atlas.force_rows.shape[0]):
        atlas.force_rows[i_n] = -1
    for j in range(response.nodes.shape[0]):
        i_n = response.nodes[j]
        atlas.force_rows[i_n] = j


@qd.kernel
def kernel_distances(load_info: StressInfo, atlas: StressAtlasInfo, visited: qd.Tensor):
    for _ in range(1):
        for i_feature in range(atlas.seeds.shape[0]):
            seed, farthest = -1, gs.qd_float(-1.0)
            for i_n in range(load_info.vertices.shape[0]):
                nearest = gs.qd_float(1e30)
                for previous in range(i_feature):
                    nearest = qd.min(nearest, atlas.distances[previous, i_n])
                if atlas.starts[i_n + 1] > atlas.starts[i_n] and nearest > farthest:
                    seed, farthest = i_n, nearest
            atlas.seeds[i_feature] = seed
            for i_n in range(load_info.vertices.shape[0]):
                atlas.distances[i_feature, i_n] = qd.select(i_n == seed, 0.0, 1e30)
                visited[i_n] = False
            for _visit in range(load_info.vertices.shape[0]):
                closest, distance = -1, gs.qd_float(1e30)
                for i_n in range(load_info.vertices.shape[0]):
                    if not visited[i_n] and atlas.distances[i_feature, i_n] < distance:
                        closest, distance = i_n, atlas.distances[i_feature, i_n]
                if closest >= 0:
                    visited[closest] = True
                    for entry in range(atlas.starts[closest], atlas.starts[closest + 1]):
                        other = atlas.neighbors[entry]
                        length = (load_info.vertices[other] - load_info.vertices[closest]).norm()
                        atlas.distances[i_feature, other] = qd.min(atlas.distances[i_feature, other], distance + length)


@qd.kernel
def kernel_coverage(width: float, load_info: StressInfo, atlas: StressAtlasInfo):
    for i_feature in range(atlas.seeds.shape[0]):
        radius, longest = gs.qd_float(0.0), gs.qd_float(0.0)
        for i_n in range(load_info.vertices.shape[0]):
            distance = atlas.distances[i_feature, i_n]
            nearest = distance
            for other in range(atlas.seeds.shape[0]):
                nearest = qd.min(nearest, atlas.distances[other, i_n])
            if distance <= nearest and distance < 1e29:
                radius = qd.max(radius, distance)
            for entry in range(atlas.starts[i_n], atlas.starts[i_n + 1]):
                longest = qd.max(longest, (load_info.vertices[i_n] - load_info.vertices[atlas.neighbors[entry]]).norm())
        atlas.radii[i_feature] = radius + 2.0 * width + longest
        for i_n in range(load_info.vertices.shape[0]):
            atlas.coverage[i_feature, i_n] = atlas.distances[i_feature, i_n] <= atlas.radii[i_feature]


@qd.kernel(graph=True)
def kernel_select(
    width: float,
    contacts: StressContactState,
    load_info: StressInfo,
    surface: StressSurfaceInfo,
    atlas: StressAtlasInfo,
    state: StressAtlasState,
):
    for i_n, i_g, i_b in qd.ndrange(state.force.shape[0], state.force.shape[1], state.force.shape[2]):
        state.force[i_n, i_g, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        state.fallback[i_c, i_b] = False
        if contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] == 0:
            for i_g in range(state.weights.shape[0]):
                state.weights[i_g, i_c, i_b] = 0.0
            i_f = contacts.anchor_face[i_c, i_b]
            shape = qd.Vector.zero(gs.qd_float, 4)
            regular = i_f >= 0
            if i_f >= 0:
                if surface.quad_vertices.shape[0] > 0:
                    basis = surface.face_dual[i_f]
                    origin = surface.face_origin[i_f]
                    shape, _position, _normal, _ratio, regular = func_quad_inverse_prepared(
                        surface.quad_coefficients[i_f],
                        origin,
                        basis,
                        basis @ (contacts.center[i_c, i_b] - origin),
                        surface.quad_projection[i_f],
                        surface.quad_scale[i_f],
                    )
                else:
                    local = surface.face_dual[i_f] @ (contacts.center[i_c, i_b] - surface.face_origin[i_f])
                    shape = qd.Vector([1.0 - local[0] - local[1], local[0], local[1], 0.0])
                    regular = shape.min() >= -2e-12
            nearest = gs.qd_float(1e30)
            for i_feature in range(atlas.seeds.shape[0]):
                distance = gs.qd_float(0.0)
                if regular:
                    for a in range(qd.select(surface.quad_vertices.shape[0] > 0, 4, 3)):
                        distance += qd.max(0.0, shape[a]) * atlas.distances[i_feature, load_info.surface_nodes[i_f, a]]
                else:
                    distance = 1e30
                state.weights[i_feature + 1, i_c, i_b] = distance
                nearest = qd.min(nearest, distance)
            total = gs.qd_float(0.0)
            for i_feature in range(atlas.seeds.shape[0]):
                distance = state.weights[i_feature + 1, i_c, i_b]
                taper = qd.max(0.0, 1.0 - (distance - nearest) / width)
                weight = qd.select(distance < 1e29, taper * taper * (3.0 - 2.0 * taper), 0.0)
                state.weights[i_feature + 1, i_c, i_b] = weight
                total += weight
            covered = total > 0.0
            # AABB overlap is conservative for the entire finite support ball. This may
            # fall back unnecessarily, but can never omit an intersecting physical face.
            for j_f in range(surface.face_origin.shape[0]):
                center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
                delta = qd.max(surface.face_bounds_low[j_f] - center, 0.0)
                delta += qd.max(center - surface.face_bounds_high[j_f], 0.0)
                normal, origin = surface.face_normal[j_f], surface.face_origin[j_f]
                warp = gs.qd_float(0.0)
                if surface.quad_vertices.shape[0] > 0:
                    for a in qd.static(range(4)):
                        warp = qd.max(warp, qd.abs((surface.quad_vertices[j_f][a, :] - origin).dot(normal)))
                # A bilinear Q4 lies in its vertex convex hull. This slab test
                # safely excludes nearby opposite sheets whose AABBs overlap.
                intersects = delta.norm_sqr() < radius * radius
                epsilon = 1.1920928955078125e-7 if qd.static(gs.qd_float == qd.f32) else 2.220446049250313e-16
                margin = 16.0 * epsilon * (center.norm() + origin.norm() + radius + warp)
                intersects = intersects and qd.abs((center - origin).dot(normal)) <= radius + warp + margin
                if intersects:
                    for i_feature in range(atlas.seeds.shape[0]):
                        if state.weights[i_feature + 1, i_c, i_b] > 0.0:
                            for a in range(qd.select(surface.quad_vertices.shape[0] > 0, 4, 6)):
                                covered = covered and atlas.coverage[i_feature, load_info.surface_nodes[j_f, a]]
            for i_feature in range(atlas.seeds.shape[0]):
                state.weights[i_feature + 1, i_c, i_b] = qd.select(
                    covered, state.weights[i_feature + 1, i_c, i_b] / qd.max(total, 1e-30), 0.0
                )
            state.weights[0, i_c, i_b] = qd.select(covered, 0.0, 1.0)
            state.fallback[i_c, i_b] = not covered


@qd.kernel(graph=True)
def kernel_pack(
    state: StressState, response: StressSurfaceInverseInfo, atlas: StressAtlasState, only_failed: qd.template()
):
    for _ in range(1):
        atlas.count[None] = 0
    for i_g, i_b in qd.ndrange(atlas.counts.shape[0], atlas.counts.shape[1]):
        count = 0
        enabled = True
        if qd.static(only_failed):
            enabled = not state.valid[i_b]
        if enabled:
            for j in range(response.nodes.shape[0]):
                force = atlas.force[j, i_g, i_b]
                if force[0] != 0.0 or force[1] != 0.0 or force[2] != 0.0:
                    atlas.columns[count, i_g, i_b] = j
                    count += 1
        atlas.counts[i_g, i_b] = count
        if count:
            slot = qd.atomic_add(atlas.count[None], 1)
            atlas.pairs[slot] = qd.Vector([i_g, i_b])


@qd.kernel(graph=True)
def kernel_apply(young: float, response: StressSurfaceInverseInfo, atlas: StressAtlasState):
    for i_n, i_pair in qd.ndrange(atlas.displacement.shape[0], atlas.count[None]):
        i_g, i_b = atlas.pairs[i_pair][0], atlas.pairs[i_pair][1]
        value = qd.Vector.zero(gs.qd_float, 3)
        for slot in range(atlas.counts[i_g, i_b]):
            j = atlas.columns[slot, i_g, i_b]
            value += response.force_blocks[i_n, j] @ atlas.force[j, i_g, i_b]
        atlas.displacement[i_n, i_g, i_b] = value / young


@qd.kernel(graph=True)
def kernel_merge(
    young: float,
    omega: qd.Tensor,
    state: StressState,
    response: StressSurfaceInverseInfo,
    atlas: StressAtlasState,
    coarse: qd.template(),
    only_failed: qd.template(),
):
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.active.shape[0]):
        enabled = True
        if qd.static(only_failed):
            enabled = not state.valid[i_b]
        if enabled:
            w = omega[i_b]
            terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
            value = response.centrifugal[i_n] @ terms / young
            if qd.static(isinstance(coarse, StressAtlasCoarse)):
                for i_c in range(coarse.info.basis.shape[1]):
                    value += coarse.info.basis[i_n, i_c] * coarse.state.coefficients[i_c, i_b] / young
            for i_g in range(atlas.counts.shape[0]):
                if atlas.counts[i_g, i_b]:
                    value += atlas.displacement[i_n, i_g, i_b]
            state.displacement[i_n, i_b] = value
    for i_b in range(state.active.shape[0]):
        state.active[i_b] = 1
        if qd.static(only_failed):
            state.active[i_b] = gs.qd_int(not state.valid[i_b])
        state.valid[i_b] = True
