"""Conservative continuous-surface force and wrench incompatibility certificates."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V

from .boundary_certificate import func_boundary_wrench_penalty
from .contact import StressContactState
from .provenance import StressSourceState, func_closest_triangle
from .surface import StressSurfaceInfo
from .traction import (
    StressTractionInfo,
    StressTractionState,
    StressTractionStatus,
    func_coulomb_projection,
    func_traction_direction,
    func_traction_geometry,
    func_traction_source_friction,
    func_traction_support_range,
)


@dataclass(frozen=True)
class StressWrenchSearchState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    triangles: qd.Tensor
    depths: qd.Tensor
    bound: qd.Tensor
    visited: qd.Tensor
    evaluations: qd.Tensor


def create_wrench_search(n_contacts: int, n_envs: int, maximum_depth: int) -> StressWrenchSearchState:
    # A depth-first quarter subdivision needs exactly 1 + 3 * depth pending triangles.
    shape = (1 + 3 * maximum_depth, n_contacts, n_envs)
    return StressWrenchSearchState(
        triangles=V_VEC(6, dtype=gs.qd_float, shape=shape),
        depths=V(dtype=gs.qd_int, shape=shape),
        bound=V(dtype=gs.qd_float, shape=(n_contacts, n_envs)),
        visited=V(dtype=gs.qd_int, shape=(n_contacts, n_envs)),
        evaluations=V(dtype=gs.qd_int, shape=(n_contacts, n_envs)),
    )


@qd.func
def func_wrench_polar_penalty(
    i_c: int,
    i_b: int,
    ray: qd.types.vector(6),
    target: qd.types.vector(6),
    penalty: float,
    contact_state: StressContactState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
    n_samples: int,
):
    delta = ray - target
    energy, gradient = 0.5 * delta.dot(delta), delta
    hessian = qd.Matrix.identity(gs.qd_float, 6)
    first, last = func_traction_support_range(i_c, i_b, traction_state)
    for i_s in range(first, last):
        friction = func_traction_source_friction(i_s, i_b, contact_state.friction[i_c, i_b], traction_state)
        for i_q in range(traction_state.sample_count[i_s, i_b]):
            if traction_state.profile[i_q, i_s, i_b] > 0.0:
                position, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                relative = (position - contact_state.center[i_c, i_b]) / contact_state.radius[i_c, i_b]
                value = ray[0:3] + ray[3:6].cross(relative)
                projected, jacobian = func_coulomb_projection(value, normal, friction)
                operator = qd.Matrix.zero(gs.qd_float, 6, 3)
                for a in qd.static(range(3)):
                    operator[a, a] = 1.0
                operator[3, 1], operator[3, 2] = -relative[2], relative[1]
                operator[4, 0], operator[4, 2] = relative[2], -relative[0]
                operator[5, 0], operator[5, 1] = -relative[1], relative[0]
                energy += 0.5 * penalty * projected.dot(projected) / n_samples
                gradient += penalty * (operator @ projected) / n_samples
                hessian += penalty * (operator @ jacobian @ operator.transpose()) / n_samples
    return energy, gradient, hessian


def kernel_search_wrench_certificate(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    search_state: StressWrenchSearchState,
    traction_info: StressTractionInfo,
    epsilon: float,
    wrench_tolerance: float,
    boundary_witnesses: bool = False,
):
    kernel_search_wrench_certificate_impl(
        contact_state,
        source_state,
        traction_state,
        search_state,
        traction_info,
        epsilon,
        wrench_tolerance,
        boundary_witnesses,
    )


@qd.kernel
def kernel_search_wrench_certificate_impl(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    search_state: StressWrenchSearchState,
    traction_info: StressTractionInfo,
    epsilon: float,
    wrench_tolerance: float,
    boundary_witnesses: qd.template(),
):
    """Search a bounded separating ray only for unresolved, already validated traction inputs."""
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if qd.static(not boundary_witnesses):
            search_state.evaluations[i_c, i_b] = 0
        status = traction_state.status[i_c, i_b]
        unresolved = status == StressTractionStatus.NUMERICAL_FAILURE
        unresolved = unresolved or status == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
        if contact_state.valid[i_c, i_b] and unresolved:
            radius = contact_state.radius[i_c, i_b]
            scale = qd.max(
                contact_state.force[i_c, i_b].norm(),
                source_state.anchor_moment[i_c, i_b].norm() / radius,
            )
            if scale > 0.0:
                target = qd.Vector.zero(gs.qd_float, 6)
                target[0:3] = contact_state.force[i_c, i_b] / scale
                target[3:6] = source_state.anchor_moment[i_c, i_b] / (radius * scale)
                budget_scale = contact_state.force[i_c, i_b].norm()
                if budget_scale == 0.0:
                    budget_scale = scale
                offset = (contact_state.position[i_c, i_b] - contact_state.center[i_c, i_b]) / radius
                first, last = func_traction_support_range(i_c, i_b, traction_state)
                n_samples = 0
                normal_sum = qd.Vector.zero(gs.qd_float, 3)
                for i_s in range(first, last):
                    for i_q in range(traction_state.sample_count[i_s, i_b]):
                        if traction_state.profile[i_q, i_s, i_b] > 0.0:
                            _, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                            normal_sum += normal
                            n_samples += 1
                if n_samples > 0 and normal_sum.norm() > 0.0:
                    reference = normal_sum.normalized()
                    depth = gs.qd_float(float("inf"))
                    for i_s in range(first, last):
                        friction = func_traction_source_friction(
                            i_s, i_b, contact_state.friction[i_c, i_b], traction_state
                        )
                        for i_q in range(traction_state.sample_count[i_s, i_b]):
                            if traction_state.profile[i_q, i_s, i_b] > 0.0:
                                _, normal = func_traction_geometry(i_q, i_s, i_b, traction_state, traction_info)
                                compression = reference.dot(normal)
                                depth = qd.min(
                                    depth,
                                    compression - friction * (reference - compression * normal).norm(),
                                )
                    ray = qd.Vector.zero(gs.qd_float, 6)
                    penalty = gs.qd_float(100.0)
                    found = False
                    for _phase in range(5):
                        if not found:
                            proposal, direction = ray, ray
                            energy, slope, alpha = 0.0, 0.0, 1.0
                            gradient_squared = gs.qd_float(float("inf"))
                            trial, active = False, True
                            for _ in range(128):
                                if active:
                                    trial_energy = gs.qd_float(0.0)
                                    gradient = qd.Vector.zero(gs.qd_float, 6)
                                    hessian = qd.Matrix.zero(gs.qd_float, 6, 6)
                                    if qd.static(boundary_witnesses):
                                        trial_energy, gradient, hessian = func_boundary_wrench_penalty(
                                            i_c,
                                            i_b,
                                            proposal,
                                            target,
                                            penalty,
                                            contact_state,
                                            traction_state,
                                            traction_info,
                                            epsilon,
                                        )
                                    else:
                                        trial_energy, gradient, hessian = func_wrench_polar_penalty(
                                            i_c,
                                            i_b,
                                            proposal,
                                            target,
                                            penalty,
                                            contact_state,
                                            traction_state,
                                            traction_info,
                                            n_samples,
                                        )
                                    search_state.evaluations[i_c, i_b] += 1
                                    squared = gradient.dot(gradient)
                                    allowance = 64.0 * epsilon * (1.0 + qd.abs(energy) + qd.abs(trial_energy))
                                    accepted = not trial or trial_energy <= energy - 1e-4 * alpha * slope
                                    accepted = accepted or (
                                        trial_energy <= energy + allowance and squared < gradient_squared
                                    )
                                    if accepted:
                                        ray, energy, gradient_squared = (
                                            proposal,
                                            trial_energy,
                                            squared,
                                        )
                                        if gradient.norm() <= 1e-9:
                                            active = False
                                        else:
                                            direction, positive = func_traction_direction(hessian, gradient, 0.0)
                                            slope = gradient.dot(direction)
                                            if not positive or slope <= 0.0 or not slope < qd.math.inf:
                                                direction = gradient / qd.max(hessian.trace(), 1.0)
                                                slope = gradient.dot(direction)
                                            fraction = qd.min(
                                                1.0,
                                                2.0 * qd.max(1.0, ray.norm()) / qd.max(direction.norm(), 1e-30),
                                            )
                                            direction *= fraction
                                            slope *= fraction
                                            alpha, proposal, trial = (
                                                1.0,
                                                ray - direction,
                                                True,
                                            )
                                    else:
                                        alpha *= 0.5
                                        proposal = ray - alpha * direction
                            adjusted = ray
                            # Reuse one local-cone evaluation for the raw ray and the
                            # strict polar correction, which changes no input traction.
                            for i_check in range(2):
                                maximum = gs.qd_float(-float("inf"))
                                for i_s in range(first, last):
                                    friction = func_traction_source_friction(
                                        i_s,
                                        i_b,
                                        contact_state.friction[i_c, i_b],
                                        traction_state,
                                    )
                                    for i_q in range(traction_state.sample_count[i_s, i_b]):
                                        if traction_state.profile[i_q, i_s, i_b] > 0.0:
                                            position, normal = func_traction_geometry(
                                                i_q,
                                                i_s,
                                                i_b,
                                                traction_state,
                                                traction_info,
                                            )
                                            relative = (position - contact_state.center[i_c, i_b]) / radius
                                            value = adjusted[0:3] + adjusted[3:6].cross(relative)
                                            compression = value.dot(normal)
                                            operand_scale = (
                                                position.norm() + contact_state.center[i_c, i_b].norm()
                                            ) / radius
                                            allowance = 128.0 * epsilon * (1.0 + friction)
                                            allowance *= 1.0 + value.norm() + adjusted[3:6].norm() * operand_scale
                                            maximum = qd.max(
                                                maximum,
                                                compression
                                                + friction * (value - compression * normal).norm()
                                                + allowance,
                                            )
                                if i_check == 0:
                                    if depth > 128.0 * epsilon:
                                        adjusted[0:3] -= qd.max(maximum + 1e-12, 0.0) / depth * reference
                                    if adjusted.norm() > 0.0:
                                        adjusted /= adjusted.norm()
                                elif adjusted.norm() > 0.0:
                                    # Final budgets concern moment about the original
                                    # point, so propagate their polar bound to the anchor.
                                    budget = wrench_tolerance * budget_scale / scale
                                    budget *= (adjusted[0:3] + adjusted[3:6].cross(offset)).norm() + adjusted[
                                        3:6
                                    ].norm()
                                    operands = (
                                        contact_state.position[i_c, i_b].norm() + contact_state.center[i_c, i_b].norm()
                                    )
                                    operands *= contact_state.force[i_c, i_b].norm() / (radius * scale)
                                    roundoff = (
                                        128.0 * epsilon * (1.0 + qd.abs(target).sum() + operands * adjusted[3:6].norm())
                                    )
                                    margin = target.dot(adjusted)
                                    if maximum < -roundoff and margin > budget + roundoff:
                                        traction_state.certificate[i_c, i_b] = adjusted
                                        traction_state.certificate_margin[i_c, i_b] = margin - roundoff
                                        traction_state.status[i_c, i_b] = StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
                                        found = True
                            penalty *= 100.0


def kernel_certify_continuous_wrench(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    search_state: StressWrenchSearchState,
    surface_info: StressSurfaceInfo,
    epsilon: float,
    maximum_depth: int,
    maximum_nodes: int,
    wrench_tolerance: float,
    accumulate: bool = False,
):
    kernel_certify_continuous_wrench_impl(
        contact_state,
        source_state,
        traction_state,
        search_state,
        surface_info,
        epsilon,
        maximum_depth,
        maximum_nodes,
        wrench_tolerance,
        accumulate,
    )


@qd.kernel
def kernel_certify_continuous_wrench_impl(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    search_state: StressWrenchSearchState,
    surface_info: StressSurfaceInfo,
    epsilon: float,
    maximum_depth: int,
    maximum_nodes: int,
    wrench_tolerance: float,
    accumulate: qd.template(),
):
    """Certify strict separation on a conservative superset of the unchanged intrinsic disks.

    The polar support is convex on each affine material face. Subtriangle vertex maxima bound every interior position;
    Euclidean disk exclusion uses a conservative AABB distance. Depth or work exhaustion leaves sampled infeasibility
    unresolved in the continuum. No force, moment, friction, provenance or radius is modified.
    """
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        previous_nodes = 0
        if qd.static(accumulate):
            previous_nodes = search_state.visited[i_c, i_b]
        else:
            search_state.bound[i_c, i_b] = gs.qd_float(float("inf"))
            search_state.visited[i_c, i_b] = 0
        if (
            contact_state.valid[i_c, i_b]
            and traction_state.status[i_c, i_b] == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
        ):
            search_state.bound[i_c, i_b] = gs.qd_float(float("inf"))
            ray = traction_state.certificate[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            scale = qd.max(
                contact_state.force[i_c, i_b].norm(),
                source_state.anchor_moment[i_c, i_b].norm() / radius,
            )
            if ray.norm() > 0.0 and scale > 0.0:
                ray /= ray.norm()
                target = qd.Vector.zero(gs.qd_float, 6)
                target[0:3] = contact_state.force[i_c, i_b] / scale
                target[3:6] = source_state.anchor_moment[i_c, i_b] / (radius * scale)
                offset = (contact_state.position[i_c, i_b] - contact_state.center[i_c, i_b]) / radius
                budget_scale = contact_state.force[i_c, i_b].norm()
                if budget_scale == 0.0:
                    budget_scale = scale
                budget = wrench_tolerance * budget_scale / scale
                budget *= (ray[0:3] + ray[3:6].cross(offset)).norm() + ray[3:6].norm()
                operands = contact_state.position[i_c, i_b].norm() + contact_state.center[i_c, i_b].norm()
                operands *= contact_state.force[i_c, i_b].norm() / (radius * scale)
                roundoff = 128.0 * epsilon * (1.0 + qd.abs(target).sum() + operands * ray[3:6].norm())
                margin = target.dot(ray)
                resolved, n_faces, n_nodes = True, 0, 0
                resolved = margin > budget + roundoff
                maximum = gs.qd_float(-float("inf"))
                first, last = func_traction_support_range(i_c, i_b, traction_state)
                for i_s in range(first, last):
                    center, support_radius = contact_state.center[i_c, i_b], radius
                    if qd.static(len(traction_state.support.center.shape) == 2):
                        center = traction_state.support.center[i_s, i_b]
                        support_radius = traction_state.support.radius[i_s, i_b]
                    friction = func_traction_source_friction(i_s, i_b, contact_state.friction[i_c, i_b], traction_state)
                    geometric_roundoff = 128.0 * epsilon * (center.norm() + support_radius)
                    for i_f in range(surface_info.face_origin.shape[0]):
                        if resolved:
                            closest, _ = func_closest_triangle(i_f, center, surface_info)
                            if (closest - center).norm() <= support_radius + geometric_roundoff:
                                n_faces += 1
                                search_state.triangles[0, i_c, i_b] = qd.Vector([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
                                search_state.depths[0, i_c, i_b] = 0
                                size = 1
                                for _node in range(maximum_nodes):
                                    if size > 0 and resolved:
                                        size -= 1
                                        barycentric = search_state.triangles[size, i_c, i_b]
                                        depth = search_state.depths[size, i_c, i_b]
                                        n_nodes += 1
                                        if n_nodes > maximum_nodes:
                                            resolved = False
                                        else:
                                            origin = surface_info.face_origin[i_f]
                                            edges = surface_info.face_edges[i_f]
                                            points = qd.Matrix.zero(gs.qd_float, 3, 3)
                                            for a in qd.static(range(3)):
                                                points[a, :] = (
                                                    origin
                                                    + barycentric[2 * a] * edges[0, :]
                                                    + barycentric[2 * a + 1] * edges[1, :]
                                                )
                                            nearest = qd.Vector.zero(gs.qd_float, 3)
                                            for a in qd.static(range(3)):
                                                low = qd.min(
                                                    points[0, a],
                                                    points[1, a],
                                                    points[2, a],
                                                )
                                                high = qd.max(
                                                    points[0, a],
                                                    points[1, a],
                                                    points[2, a],
                                                )
                                                nearest[a] = qd.max(
                                                    low - center[a],
                                                    center[a] - high,
                                                    0.0,
                                                )
                                            if nearest.norm() <= support_radius + geometric_roundoff:
                                                normal = -surface_info.face_normal[i_f]
                                                upper = gs.qd_float(-float("inf"))
                                                for a in qd.static(range(3)):
                                                    relative = (points[a, :] - contact_state.center[i_c, i_b]) / radius
                                                    value = ray[0:3] + ray[3:6].cross(relative)
                                                    compression = value.dot(normal)
                                                    polar = (
                                                        compression + friction * (value - compression * normal).norm()
                                                    )
                                                    allowance = (
                                                        128.0
                                                        * epsilon
                                                        * (1.0 + friction)
                                                        * (1.0 + value.norm() + relative.norm())
                                                    )
                                                    operand_scale = (
                                                        origin.norm() + edges[0, :].norm() + edges[1, :].norm()
                                                    )
                                                    operand_scale += contact_state.center[i_c, i_b].norm()
                                                    allowance += (
                                                        128.0
                                                        * epsilon
                                                        * (1.0 + friction)
                                                        * ray[3:6].norm()
                                                        * operand_scale
                                                        / radius
                                                    )
                                                    upper = qd.max(upper, polar + allowance)
                                                if upper < 0.0:
                                                    maximum = qd.max(maximum, upper)
                                                elif (
                                                    depth >= maximum_depth or size + 4 > search_state.triangles.shape[0]
                                                ):
                                                    resolved = False
                                                else:
                                                    a, b, c = (
                                                        barycentric[0:2],
                                                        barycentric[2:4],
                                                        barycentric[4:6],
                                                    )
                                                    ab, bc, ca = (
                                                        0.5 * (a + b),
                                                        0.5 * (b + c),
                                                        0.5 * (c + a),
                                                    )
                                                    search_state.triangles[size, i_c, i_b] = qd.Vector(
                                                        [
                                                            a[0],
                                                            a[1],
                                                            ab[0],
                                                            ab[1],
                                                            ca[0],
                                                            ca[1],
                                                        ]
                                                    )
                                                    search_state.triangles[size + 1, i_c, i_b] = qd.Vector(
                                                        [
                                                            ab[0],
                                                            ab[1],
                                                            b[0],
                                                            b[1],
                                                            bc[0],
                                                            bc[1],
                                                        ]
                                                    )
                                                    search_state.triangles[size + 2, i_c, i_b] = qd.Vector(
                                                        [
                                                            ca[0],
                                                            ca[1],
                                                            bc[0],
                                                            bc[1],
                                                            c[0],
                                                            c[1],
                                                        ]
                                                    )
                                                    search_state.triangles[size + 3, i_c, i_b] = qd.Vector(
                                                        [
                                                            ab[0],
                                                            ab[1],
                                                            bc[0],
                                                            bc[1],
                                                            ca[0],
                                                            ca[1],
                                                        ]
                                                    )
                                                    for child in qd.static(range(4)):
                                                        search_state.depths[size + child, i_c, i_b] = depth + 1
                                                    size += 4
                                if size > 0:
                                    resolved = False
                if resolved and n_faces > 0:
                    search_state.bound[i_c, i_b] = maximum
                search_state.visited[i_c, i_b] = previous_nodes + n_nodes
                if resolved and n_faces > 0 and maximum < 0.0:
                    traction_state.status[i_c, i_b] = StressTractionStatus.CONTINUOUS_WRENCH_INFEASIBLE
                    traction_state.certificate[i_c, i_b] = ray
                    traction_state.certificate_margin[i_c, i_b] = margin - roundoff


def kernel_diagnose_wrenches(
    contact_state: StressContactState,
    source_state: StressSourceState,
    traction_state: StressTractionState,
    search_state: StressWrenchSearchState,
    traction_info: StressTractionInfo,
    surface_info: StressSurfaceInfo,
    epsilon: float,
    wrench_tolerance: float,
) -> None:
    """Retain existing strict proofs, then retry only unresolved contacts with boundary witnesses.

    Every phase stays device-masked. Retry counts include the first phase, and an established continuous proof and
    its bound survive the retry kernels. Finite witnesses only guide search; unchanged continuous checks prove it.
    """
    kernel_search_wrench_certificate(
        contact_state,
        source_state,
        traction_state,
        search_state,
        traction_info,
        epsilon,
        wrench_tolerance,
    )
    kernel_certify_continuous_wrench(
        contact_state,
        source_state,
        traction_state,
        search_state,
        surface_info,
        epsilon,
        8,
        16384,
        wrench_tolerance,
    )
    kernel_search_wrench_certificate(
        contact_state,
        source_state,
        traction_state,
        search_state,
        traction_info,
        epsilon,
        wrench_tolerance,
        boundary_witnesses=True,
    )
    kernel_certify_continuous_wrench(
        contact_state,
        source_state,
        traction_state,
        search_state,
        surface_info,
        epsilon,
        12,
        65536,
        wrench_tolerance,
        accumulate=True,
    )


@qd.kernel
def kernel_certify_continuous_force(
    contact_state: StressContactState,
    traction_state: StressTractionState,
    surface_info: StressSurfaceInfo,
    epsilon: float,
    force_tolerance: float,
):
    """Certify only forces outside every possible local continuous cone sum.

    The Euclidean radius ball contains the declared intrinsic footprint. All
    intersecting triangles therefore form a conservative support superset,
    including branches the intrinsic law cannot reach. A strict common polar
    ray bounds every finite traction distribution on that superset; its force
    margin is a lower bound on the normalized resultant error. This test does
    not establish general six-dimensional feasibility or integration accuracy.
    """
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        status = traction_state.status[i_c, i_b]
        unresolved = status == StressTractionStatus.NUMERICAL_FAILURE
        unresolved = unresolved or status == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
        if contact_state.valid[i_c, i_b] and unresolved:
            i_anchor = contact_state.anchor_face[i_c, i_b]
            force = contact_state.force[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            friction = contact_state.friction[i_c, i_b]
            anchor = contact_state.center[i_c, i_b]
            if 0 <= i_anchor < surface_info.face_normal.shape[0] and force.norm() > 0.0:
                normal = -surface_info.face_normal[i_anchor]
                unit_force = force / force.norm()
                compression = unit_force.dot(normal)
                tangent = unit_force - compression * normal
                if tangent.norm() > friction * compression and tangent.norm() > 0.0:
                    # Construct the polar analytically: subtracting a tiny
                    # force projection would lose relative accuracy near slip.
                    ray = (tangent / tangent.norm() - friction * normal) / qd.sqrt(1.0 + friction**2)
                    ray = (ray - 1024.0 * epsilon * (1.0 + friction) * normal).normalized()
                    roundoff = 64.0 * epsilon * (1.0 + friction)
                    maximum_support = gs.qd_float(-float("inf"))
                    start, end = func_traction_support_range(i_c, i_b, traction_state)
                    for i_s in range(start, end):
                        source_anchor, source_radius = anchor, radius
                        if qd.static(len(traction_state.support.center.shape) == 2):
                            source_anchor = traction_state.support.center[i_s, i_b]
                            source_radius = traction_state.support.radius[i_s, i_b]
                        source_friction = func_traction_source_friction(i_s, i_b, friction, traction_state)
                        geometric_roundoff = 64.0 * epsilon * (source_anchor.norm() + source_radius)
                        for i_f in range(surface_info.face_normal.shape[0]):
                            closest, _ = func_closest_triangle(i_f, source_anchor, surface_info)
                            if (closest - source_anchor).norm() <= source_radius + geometric_roundoff:
                                inward = -surface_info.face_normal[i_f]
                                component = ray.dot(inward)
                                support = component + source_friction * (ray - component * inward).norm()
                                maximum_support = qd.max(maximum_support, support)
                    margin = unit_force.dot(ray)
                    if maximum_support < -roundoff and margin > force_tolerance + roundoff:
                        traction_state.status[i_c, i_b] = StressTractionStatus.CONTINUOUS_FORCE_INFEASIBLE
                        traction_state.certificate[i_c, i_b] = qd.Vector([ray[0], ray[1], ray[2], 0.0, 0.0, 0.0])
                        traction_state.certificate_margin[i_c, i_b] = margin - roundoff
