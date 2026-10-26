"""Material-graph feature selection and complete finite-footprint coverage."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V
from genesis.utils.misc import qd_to_numpy

from .contact import StressContactState
from .data import StressAtlasScatter, StressInfo
from .quadrilateral import func_quad_inverse_prepared
from .surface import StressSurfaceInfo


@dataclass(frozen=True)
class FeatureRegionInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    starts: qd.Tensor
    neighbors: qd.Tensor
    positions: qd.Tensor
    elements: qd.Tensor
    distances: qd.Tensor
    seeds: qd.Tensor
    radii: qd.Tensor
    coverage: qd.Tensor
    face_coverage: qd.Tensor
    force_rows: qd.Tensor
    boundary_nodes: qd.Tensor


@dataclass(frozen=True)
class FeatureRegionState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    weights: qd.Tensor
    fallback: qd.Tensor
    force: qd.Tensor


class FeatureRegions:
    def __init__(
        self, model, basis, contacts: StressContactState, n_features: int, width: float, core_scale: float, halo: int
    ):
        n_physical = len(basis.parent_nodes)
        elements = qd_to_numpy(model.shell.quads if model.n_shell_elements else model.info.elements)
        pairs = np.stack(np.broadcast_arrays(elements[:, :, None], elements[:, None, :]), axis=-1)
        pairs = np.unique(pairs.reshape((-1, 2)), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        starts = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=n_physical))]
        surface_nodes = qd_to_numpy(model.load_info.surface_nodes)
        if model.n_shell_elements:
            load_nodes = qd_to_numpy(model.load_info.load_nodes)
            nodes = np.unique(load_nodes[surface_nodes])
            boundary = (2 * nodes[:, None] + np.arange(2)).reshape((-1,))
        else:
            boundary = np.unique(surface_nodes)
        self.info = FeatureRegionInfo(
            starts=V(dtype=gs.qd_int, shape=(n_physical + 1,)),
            neighbors=V(dtype=gs.qd_int, shape=(len(pairs),)),
            positions=V_VEC(3, dtype=gs.qd_float, shape=(n_physical,)),
            elements=V(dtype=gs.qd_int, shape=elements.shape),
            distances=V(dtype=gs.qd_float, shape=(n_features, n_physical)),
            seeds=V(dtype=gs.qd_int, shape=(n_features,)),
            radii=V(dtype=gs.qd_float, shape=(n_features,)),
            coverage=V(dtype=gs.qd_bool, shape=(n_features, n_physical)),
            face_coverage=V(dtype=gs.qd_bool, shape=(n_features, model.load_info.surface_nodes.shape[0])),
            force_rows=V(dtype=gs.qd_int, shape=(model.info.vertices.shape[0],)),
            boundary_nodes=V(dtype=gs.qd_int, shape=(len(boundary),)),
        )
        self.info.starts.from_numpy(starts.astype(gs.np_int, copy=False))
        self.info.neighbors.from_numpy(pairs[:, 1].astype(gs.np_int, copy=False))
        self.info.elements.from_numpy(elements.astype(gs.np_int, copy=False))
        self.info.boundary_nodes.from_numpy(boundary.astype(gs.np_int, copy=False))
        visited = V(dtype=gs.qd_int, shape=(n_physical,))
        active = V(dtype=gs.qd_int, shape=(n_features, len(elements)))
        support = V(dtype=gs.qd_int, shape=(n_features, n_physical))
        coarse = V(dtype=gs.qd_int, shape=(n_physical,))
        coarse.from_numpy((basis.coarse_index >= 0).astype(gs.np_int))
        kernel_region_geometry(model.info, self.info, basis.blocks_per_node)
        kernel_region_distances(self.info, visited)
        kernel_region_elements(width, core_scale, self.info, active)
        for _ in range(halo):
            kernel_region_halo_nodes(self.info, active, support)
            kernel_region_halo_elements(self.info, active, support)
        kernel_region_local_nodes(self.info, active, support)
        local = qd_to_numpy(support)
        self.local_nodes = [np.flatnonzero(row).astype(gs.np_int) for row in local]
        kernel_region_coverage(self.info, support, coarse)
        kernel_region_face_coverage(model.load_info, self.info)
        n_contacts, n_envs = contacts.valid.shape
        self.state = FeatureRegionState(
            weights=V(dtype=gs.qd_float, shape=(n_features + 1, n_contacts, n_envs)),
            fallback=V(dtype=gs.qd_bool, shape=(n_contacts, n_envs)),
            force=V_VEC(3, dtype=gs.qd_float, shape=(len(boundary), n_features + 1, n_envs)),
        )
        self.state.weights.fill(0)
        self.state.fallback.fill(False)
        self.scatter = StressAtlasScatter(self.state.weights, self.state.force, self.info.force_rows)

    def prepare(self, contacts: StressContactState, load_info: StressInfo, surface: StressSurfaceInfo, width: float):
        kernel_select(width, contacts, load_info, surface, self.info, self.state, self.info.seeds.shape[0])


@qd.kernel
def kernel_region_geometry(mother: StressInfo, regions: FeatureRegionInfo, blocks_per_node: int):
    for i_n in range(regions.positions.shape[0]):
        regions.positions[i_n] = mother.vertices[blocks_per_node * i_n]
    for i_n in range(regions.force_rows.shape[0]):
        regions.force_rows[i_n] = -1
    for j_n in range(regions.boundary_nodes.shape[0]):
        i_n = regions.boundary_nodes[j_n]
        regions.force_rows[i_n] = j_n


@qd.kernel
def kernel_region_distances(regions: FeatureRegionInfo, visited: qd.Tensor):
    for _ in range(1):
        for i_f in range(regions.seeds.shape[0]):
            seed, farthest = -1, gs.qd_float(-1.0)
            for i_n in range(regions.positions.shape[0]):
                nearest = gs.qd_float(1e30)
                for previous in range(i_f):
                    nearest = qd.min(nearest, regions.distances[previous, i_n])
                if regions.starts[i_n + 1] > regions.starts[i_n] and nearest > farthest:
                    seed, farthest = i_n, nearest
            regions.seeds[i_f] = seed
            for i_n in range(regions.positions.shape[0]):
                regions.distances[i_f, i_n] = qd.select(i_n == seed, 0.0, 1e30)
                visited[i_n] = 0
            for _visit in range(regions.positions.shape[0]):
                closest, distance = -1, gs.qd_float(1e30)
                for i_n in range(regions.positions.shape[0]):
                    if not visited[i_n] and regions.distances[i_f, i_n] < distance:
                        closest, distance = i_n, regions.distances[i_f, i_n]
                if closest >= 0:
                    visited[closest] = 1
                    for i_e in range(regions.starts[closest], regions.starts[closest + 1]):
                        other = regions.neighbors[i_e]
                        length = (regions.positions[other] - regions.positions[closest]).norm()
                        regions.distances[i_f, other] = qd.min(regions.distances[i_f, other], distance + length)


@qd.kernel
def kernel_region_elements(width: float, scale: float, regions: FeatureRegionInfo, active: qd.Tensor):
    for i_f in range(regions.seeds.shape[0]):
        radius, longest = gs.qd_float(0.0), gs.qd_float(0.0)
        for i_n in range(regions.positions.shape[0]):
            distance = regions.distances[i_f, i_n]
            nearest = distance
            for other in range(regions.seeds.shape[0]):
                nearest = qd.min(nearest, regions.distances[other, i_n])
            if distance <= nearest and distance < 1e29:
                radius = qd.max(radius, distance)
            for i_e in range(regions.starts[i_n], regions.starts[i_n + 1]):
                other = regions.neighbors[i_e]
                longest = qd.max(longest, (regions.positions[other] - regions.positions[i_n]).norm())
        regions.radii[i_f] = scale * radius + 2.0 * width + longest
    for i_f, i_e in qd.ndrange(active.shape[0], active.shape[1]):
        inside = True
        for i_a in range(regions.elements.shape[1]):
            i_n = regions.elements[i_e, i_a]
            inside = inside and regions.distances[i_f, i_n] <= regions.radii[i_f]
        active[i_f, i_e] = qd.cast(inside, gs.qd_int)


@qd.kernel
def kernel_region_halo_nodes(regions: FeatureRegionInfo, active: qd.Tensor, support: qd.Tensor):
    for i_f, i_n in qd.ndrange(support.shape[0], support.shape[1]):
        support[i_f, i_n] = 0
    for i_f, i_e in qd.ndrange(active.shape[0], active.shape[1]):
        if active[i_f, i_e]:
            for i_a in range(regions.elements.shape[1]):
                i_n = regions.elements[i_e, i_a]
                support[i_f, i_n] = 1


@qd.kernel
def kernel_region_halo_elements(regions: FeatureRegionInfo, active: qd.Tensor, support: qd.Tensor):
    for i_f, i_e in qd.ndrange(active.shape[0], active.shape[1]):
        inside = active[i_f, i_e] != 0
        for i_a in range(regions.elements.shape[1]):
            i_n = regions.elements[i_e, i_a]
            inside = inside or support[i_f, i_n] != 0
        active[i_f, i_e] = qd.cast(inside, gs.qd_int)


@qd.kernel
def kernel_region_local_nodes(regions: FeatureRegionInfo, active: qd.Tensor, support: qd.Tensor):
    for i_f, i_n in qd.ndrange(support.shape[0], support.shape[1]):
        support[i_f, i_n] = 1
    for i_f, i_e in qd.ndrange(active.shape[0], active.shape[1]):
        if not active[i_f, i_e]:
            for i_a in range(regions.elements.shape[1]):
                i_n = regions.elements[i_e, i_a]
                qd.atomic_min(support[i_f, i_n], 0)


@qd.kernel
def kernel_region_coverage(regions: FeatureRegionInfo, support: qd.Tensor, coarse: qd.Tensor):
    for i_f, i_n in qd.ndrange(support.shape[0], support.shape[1]):
        regions.coverage[i_f, i_n] = support[i_f, i_n] != 0 or coarse[i_n] != 0


@qd.kernel
def kernel_region_face_coverage(load_info: StressInfo, regions: FeatureRegionInfo):
    for i_f, i_face in qd.ndrange(regions.coverage.shape[0], load_info.surface_nodes.shape[0]):
        covered = True
        for i_a in range(qd.select(load_info.load_nodes.shape[0] > 0, 4, 6)):
            i_n = load_info.surface_nodes[i_face, i_a]
            if load_info.load_nodes.shape[0] > 0:
                i_load = i_n
                i_n = load_info.load_nodes[i_load][0]
            covered = covered and regions.coverage[i_f, i_n]
        regions.face_coverage[i_f, i_face] = covered


@qd.func
def func_region_distance(i_feature: int, i_face: int, i_local: int, load_info: StressInfo, regions: FeatureRegionInfo):
    i_n = load_info.surface_nodes[i_face, i_local]
    if load_info.load_nodes.shape[0] > 0:
        i_load = i_n
        i_n = load_info.load_nodes[i_load][0]
    return regions.distances[i_feature, i_n]


@qd.func
def func_region_covered(i_feature: int, i_face: int, i_local: int, load_info: StressInfo, regions: FeatureRegionInfo):
    i_n = load_info.surface_nodes[i_face, i_local]
    if load_info.load_nodes.shape[0] > 0:
        i_load = i_n
        i_n = load_info.load_nodes[i_load][0]
    return regions.coverage[i_feature, i_n]


@qd.kernel(graph=True)
def kernel_select(
    width: float,
    contacts: StressContactState,
    load_info: StressInfo,
    surface: StressSurfaceInfo,
    atlas: FeatureRegionInfo,
    state: FeatureRegionState,
    n_features: qd.template(),
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
                        distance += qd.max(0.0, shape[a]) * func_region_distance(i_feature, i_f, a, load_info, atlas)
                else:
                    distance = 1e30
                state.weights[i_feature + 1, i_c, i_b] = distance
                nearest = qd.min(nearest, distance)
            total = gs.qd_float(0.0)
            for i_feature in range(atlas.seeds.shape[0]):
                distance = state.weights[i_feature + 1, i_c, i_b]
                taper = qd.select(distance <= nearest, 1.0, 0.0)
                if width > 0.0:
                    taper = qd.max(0.0, 1.0 - (distance - nearest) / width)
                weight = qd.select(distance < 1e29, taper * taper * (3.0 - 2.0 * taper), 0.0)
                state.weights[i_feature + 1, i_c, i_b] = weight
                total += weight
            clearance = qd.Vector.zero(gs.qd_float, n_features)
            for i_feature in qd.static(range(n_features)):
                clearance[i_feature] = 1e30
            # Lower bounds on distance to every unsupported face certify the whole
            # finite footprint. Fade to the mother before unsupported support is reached.
            for j_f in range(surface.face_origin.shape[0]):
                center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
                delta = qd.max(surface.face_bounds_low[j_f] - center, 0.0)
                delta += qd.max(center - surface.face_bounds_high[j_f], 0.0)
                normal, origin = surface.face_normal[j_f], surface.face_origin[j_f]
                warp = gs.qd_float(0.0)
                if surface.quad_vertices.shape[0] > 0:
                    for a in qd.static(range(4)):
                        warp = qd.max(warp, qd.abs((surface.quad_vertices[j_f][a, :] - origin).dot(normal)))
                epsilon = 1.1920928955078125e-7 if qd.static(gs.qd_float == qd.f32) else 2.220446049250313e-16
                margin = 16.0 * epsilon * (center.norm() + origin.norm() + radius + warp)
                slab = qd.max(0.0, qd.abs((center - origin).dot(normal)) - warp - margin)
                lower = qd.max(delta.norm(), slab) - radius - margin
                for i_feature in qd.static(range(n_features)):
                    if state.weights[i_feature + 1, i_c, i_b] > 0.0 and not atlas.face_coverage[i_feature, j_f]:
                        clearance[i_feature] = qd.min(clearance[i_feature], lower)
            weight_sum = gs.qd_float(0.0)
            fully_covered = total > 0.0
            for i_feature in qd.static(range(n_features)):
                guard = qd.max(width, 1e-6 * atlas.radii[i_feature])
                taper = qd.min(1.0, qd.max(0.0, clearance[i_feature] / guard))
                availability = taper * taper * (3.0 - 2.0 * taper)
                nominal = state.weights[i_feature + 1, i_c, i_b]
                if nominal > 0.0 and availability < 1.0:
                    fully_covered = False
                weight = nominal * availability / qd.max(total, 1e-30)
                state.weights[i_feature + 1, i_c, i_b] = weight
                weight_sum += weight
            mother_weight = qd.select(fully_covered, 0.0, qd.max(0.0, 1.0 - weight_sum))
            state.weights[0, i_c, i_b] = mother_weight
            state.fallback[i_c, i_b] = mother_weight > 0.0
