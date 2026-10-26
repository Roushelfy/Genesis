"""Native material-graph adaptation with complete finite-DPR coverage.

The fixed twelve directions seed this native graph after a QD bbox-centered,
axis-radius-scaled mapping. Graph distances, native coarse/local spaces, DPR
loads and coverage differ from the CPU material-cap construction. The legacy
farthest-point/cubic selector remains an explicit historical control.
"""

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
    directions: qd.Tensor
    centers: qd.Tensor
    mapping_center: qd.Tensor
    mapping_radius: qd.Tensor
    boundary_mask: qd.Tensor
    elements: qd.Tensor
    distances: qd.Tensor
    seeds: qd.Tensor
    radii: qd.Tensor
    support_radii: qd.Tensor
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
        self,
        model,
        basis,
        contacts: StressContactState,
        n_features: int,
        width: float,
        core_scale: float,
        halo: int,
        *,
        center_mode: str = "farthest_point",
        selection: str = "legacy",
        narrow_fraction: float = 0.12,
    ):
        if center_mode not in ("farthest_point", "icosahedral"):
            gs.raise_exception("Native atlas center_mode must be farthest_point or icosahedral.")
        if selection not in ("legacy", "continuous", "nearest", "narrow") or not 0 < narrow_fraction <= 1:
            gs.raise_exception("Native atlas selection and narrow_fraction are invalid.")
        if center_mode == "icosahedral" and (n_features != 12 or selection == "legacy"):
            gs.raise_exception(
                "The fixed native adaptation requires twelve centers and continuous/nearest/narrow selection."
            )
        self.center_mode = center_mode
        self.selection = selection
        self.selection_code = ("legacy", "continuous", "nearest", "narrow").index(selection)
        self.narrow_fraction = narrow_fraction
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
            directions=V_VEC(3, dtype=gs.qd_float, shape=(n_physical,)),
            centers=V_VEC(3, dtype=gs.qd_float, shape=(n_features,)),
            mapping_center=V_VEC(3, dtype=gs.qd_float, shape=()),
            mapping_radius=V_VEC(3, dtype=gs.qd_float, shape=()),
            boundary_mask=V(dtype=gs.qd_int, shape=(n_physical,)),
            elements=V(dtype=gs.qd_int, shape=elements.shape),
            distances=V(dtype=gs.qd_float, shape=(n_features, n_physical)),
            seeds=V(dtype=gs.qd_int, shape=(n_features,)),
            radii=V(dtype=gs.qd_float, shape=(n_features,)),
            support_radii=V(dtype=gs.qd_float, shape=(n_features,)),
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
        kernel_region_mapping(self.info)
        if center_mode == "icosahedral":
            kernel_region_icosahedron(self.info)
        kernel_region_distances(self.info, visited, center_mode == "icosahedral")
        seeds = qd_to_numpy(self.info.seeds)
        if (seeds < 0).any() or len(np.unique(seeds)) != n_features:
            gs.raise_exception("Native feature centers did not map to distinct connected graph seeds.")
        kernel_region_elements(width, core_scale, self.info, active, selection != "legacy")
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

    def prepare(
        self,
        contacts: StressContactState,
        load_info: StressInfo,
        surface: StressSurfaceInfo,
        width: float,
        cooperative: bool = True,
    ):
        """Select complete footprints with cooperative CUDA or scalar comparison scheduling."""
        kernel_select(
            width,
            contacts,
            load_info,
            surface,
            self.info,
            self.state,
            self.info.seeds.shape[0],
            cooperative,
            self.selection_code,
            self.narrow_fraction,
        )


@qd.kernel
def kernel_region_geometry(mother: StressInfo, regions: FeatureRegionInfo, blocks_per_node: int):
    for i_n in range(regions.positions.shape[0]):
        regions.positions[i_n] = mother.vertices[blocks_per_node * i_n]
        regions.boundary_mask[i_n] = 0
    for i_n in range(regions.force_rows.shape[0]):
        regions.force_rows[i_n] = -1
    for j_n in range(regions.boundary_nodes.shape[0]):
        i_n = regions.boundary_nodes[j_n]
        regions.force_rows[i_n] = j_n
        qd.atomic_max(regions.boundary_mask[i_n // blocks_per_node], 1)


@qd.kernel
def kernel_region_mapping(regions: FeatureRegionInfo):
    """Map actual load-boundary bbox positions to directions, entirely in QD."""
    for _ in range(1):
        low, high = qd.Vector([1e30, 1e30, 1e30]), qd.Vector([-1e30, -1e30, -1e30])
        for node in range(regions.positions.shape[0]):
            if regions.boundary_mask[node]:
                low = qd.min(low, regions.positions[node])
                high = qd.max(high, regions.positions[node])
        regions.mapping_center[None] = 0.5 * (low + high)
        regions.mapping_radius[None] = qd.max(0.5 * (high - low), 1e-12)
    for node in range(regions.positions.shape[0]):
        vector = (regions.positions[node] - regions.mapping_center[None]) / regions.mapping_radius[None]
        regions.directions[node] = vector / qd.max(vector.norm(), 1e-30)


@qd.kernel
def kernel_region_icosahedron(regions: FeatureRegionInfo):
    """CPU-order twelve direction identities; their native spaces remain distinct."""
    for index in range(regions.centers.shape[0]):
        golden = 0.5 * (1.0 + qd.sqrt(5.0))
        sign = qd.cast(2 * (index % 2) - 1, gs.qd_float)
        center = qd.Vector.zero(gs.qd_float, 3)
        if index < 2:
            center = qd.Vector([-golden, 0.0, sign])
        elif index < 4:
            center = qd.Vector([-1.0, sign * golden, 0.0])
        elif index < 6:
            center = qd.Vector([0.0, -1.0, sign * golden])
        elif index < 8:
            center = qd.Vector([0.0, 1.0, sign * golden])
        elif index < 10:
            center = qd.Vector([1.0, sign * golden, 0.0])
        else:
            center = qd.Vector([golden, 0.0, sign])
        regions.centers[index] = center / qd.sqrt(1.0 + golden * golden)


@qd.kernel
def kernel_region_distances(regions: FeatureRegionInfo, visited: qd.Tensor, fixed_centers: qd.template() = False):
    for _ in range(1):
        for i_f in range(regions.seeds.shape[0]):
            seed, farthest = -1, gs.qd_float(-1.0)
            if qd.static(fixed_centers):
                # Prefer physical load-boundary nodes and retain graph topology:
                # nearby points on another nonconvex sheet do not create edges.
                farthest = -2.0
                for i_n in range(regions.positions.shape[0]):
                    used = False
                    for previous in range(i_f):
                        used = used or regions.seeds[previous] == i_n
                    score = regions.centers[i_f].dot(regions.directions[i_n])
                    if (
                        regions.boundary_mask[i_n]
                        and regions.starts[i_n + 1] > regions.starts[i_n]
                        and not used
                        and score > farthest
                    ):
                        seed, farthest = i_n, score
            else:
                for i_n in range(regions.positions.shape[0]):
                    nearest = gs.qd_float(1e30)
                    for previous in range(i_f):
                        nearest = qd.min(nearest, regions.distances[previous, i_n])
                    if regions.starts[i_n + 1] > regions.starts[i_n] and nearest > farthest:
                        seed, farthest = i_n, nearest
                if seed >= 0:
                    regions.centers[i_f] = regions.directions[seed]
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
def kernel_region_elements(
    width: float,
    scale: float,
    regions: FeatureRegionInfo,
    active: qd.Tensor,
    compact_support: qd.template() = False,
):
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
        regions.support_radii[i_f] = 1.2 * qd.max(radius, 1e-12)
        core_radius = radius
        if qd.static(compact_support):
            core_radius = regions.support_radii[i_f]
        regions.radii[i_f] = scale * core_radius + 2.0 * width + longest
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
    cooperative: qd.template(),
    selection: int = 0,
    narrow_fraction: float = 0.12,
):
    for i_n, i_g, i_b in qd.ndrange(state.force.shape[0], state.force.shape[1], state.force.shape[2]):
        state.force[i_n, i_g, i_b] = qd.Vector.zero(gs.qd_float, 3)
    lanes = qd.static(32 if cooperative and gs.backend == gs.cuda else 1)
    qd.loop_config(block_dim=128)
    for i_thread in range(contacts.valid.shape[0] * contacts.valid.shape[1] * lanes):
        lane = i_thread % lanes
        i_pair = i_thread // lanes
        i_c, i_b = i_pair // contacts.valid.shape[1], i_pair % contacts.valid.shape[1]
        if lane == 0:
            state.fallback[i_c, i_b] = False
            for i_g in range(n_features + 1):
                state.weights[i_g, i_c, i_b] = 0.0
        if contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] == 0:
            weights = qd.Vector.zero(gs.qd_float, n_features)
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
            nearest, nearest_index = gs.qd_float(1e30), 0
            for i_feature in range(atlas.seeds.shape[0]):
                distance = gs.qd_float(0.0)
                if regular:
                    for a in range(qd.select(surface.quad_vertices.shape[0] > 0, 4, 3)):
                        distance += qd.max(0.0, shape[a]) * func_region_distance(i_feature, i_f, a, load_info, atlas)
                else:
                    distance = 1e30
                weights[i_feature] = distance
                if distance < nearest:
                    nearest, nearest_index = distance, i_feature
            total = gs.qd_float(0.0)
            for i_feature in range(atlas.seeds.shape[0]):
                distance = weights[i_feature]
                weight = gs.qd_float(0.0)
                if selection == 0:
                    taper = qd.select(distance <= nearest, 1.0, 0.0)
                    if width > 0.0:
                        taper = qd.max(0.0, 1.0 - (distance - nearest) / width)
                    weight = qd.select(distance < 1e29, taper * taper * (3.0 - 2.0 * taper), 0.0)
                elif selection == 2:
                    weight = qd.cast(i_feature == nearest_index and distance < 1e29, gs.qd_float)
                else:
                    weight = func_region_wendland(distance / atlas.support_radii[i_feature])
                    if selection == 3:
                        weight *= func_region_wendland(
                            (distance - nearest) / (narrow_fraction * atlas.support_radii[i_feature])
                        )
                weights[i_feature] = weight
                total += weight
            clearance = qd.Vector.zero(gs.qd_float, n_features)
            for i_feature in qd.static(range(n_features)):
                clearance[i_feature] = 1e30
            # Lower bounds on distance to every unsupported face certify the whole
            # finite footprint. Fade to the mother before unsupported support is reached.
            for i_chunk in range((surface.face_origin.shape[0] + lanes - 1) // lanes):
                j_f = i_chunk * lanes + lane
                if j_f < surface.face_origin.shape[0]:
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
                        if weights[i_feature] > 0.0 and not atlas.face_coverage[i_feature, j_f]:
                            clearance[i_feature] = qd.min(clearance[i_feature], lower)
            # All lanes share the contact gate and reach each reduction after the face loop reconverges.
            for i_feature in qd.static(range(n_features)):
                if qd.static(lanes > 1):
                    clearance[i_feature] = qd.simt.subgroup.reduce_all_min(clearance[i_feature])
            if lane == 0:
                weight_sum = gs.qd_float(0.0)
                fully_covered = total > 0.0
                for i_feature in qd.static(range(n_features)):
                    guard = qd.max(width, 1e-6 * atlas.radii[i_feature])
                    taper = qd.min(1.0, qd.max(0.0, clearance[i_feature] / guard))
                    availability = taper * taper * (3.0 - 2.0 * taper)
                    nominal = weights[i_feature]
                    if nominal > 0.0 and availability < 1.0:
                        fully_covered = False
                    weight = nominal * availability / qd.max(total, 1e-30)
                    state.weights[i_feature + 1, i_c, i_b] = weight
                    weight_sum += weight
                mother_weight = qd.select(fully_covered, 0.0, qd.max(0.0, 1.0 - weight_sum))
                state.weights[0, i_c, i_b] = mother_weight
                state.fallback[i_c, i_b] = mother_weight > 0.0


@qd.func
def func_region_wendland(distance: float):
    remainder = qd.max(1.0 - distance, 0.0)
    return remainder**4 * (1.0 + 4.0 * distance)
