"""Selective angular pressure integration with independent source-load admission."""

import quadrants as qd

import genesis as gs

from .contact import (
    _PRESSURE_FIT_TOLERANCE,
    StressContactState,
    func_contact_ray_at,
    func_contact_sample_at,
    func_contact_section,
    func_force_frame,
    func_hertz_disk_moments,
    func_pack_contacts,
    func_pressure_correct_warp,
    func_pressure_initial,
    func_pressure_step,
    func_pressure_warp,
)
from .data import StressInfo, StressState
from .scatter import StressScatterWorkspace, func_pressure_moments
from .surface import StressSurfaceInfo, func_surface_face_admitted

_ADMISSION_TOLERANCE = 1e-3
_FIT_FAILURE = 1
_CHECK_GRADIENT = 2
_CHECK_LOAD = 4
_SOURCE_WRENCH = 8
_NONFINITE = 16
_MISSING_SUPPORT = 32
_INVALID_SOURCE = 64
_NORMALIZE_CHECK_MASS = False


@qd.func
def func_adaptive_face(
    i_f: int,
    i_c: int,
    i_b: int,
    i_rule: int,
    i_lane: int,
    coefficient: qd.types.vector(3),
    reference: qd.types.vector(3),
    weight_sum: float,
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    initial: bool,
    cooperative: qd.template(),
):
    """Evaluate one original triangle under a selected shared angular rule."""
    gram = qd.Matrix.zero(gs.qd_float, 3, 3)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    nodal = qd.Vector.zero(gs.qd_float, 6)
    objective = gs.qd_float(0.0)
    candidates, invalid = 0, 0
    first, second = func_force_frame(contacts.force[i_c, i_b])
    center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
    closest, remaining, contained = func_contact_section(i_f, center, radius, surface)
    if remaining > 0.0 and func_surface_face_admitted(i_f, i_c, i_b, surface):
        margin = gs.qd_float(1e-12)
        if qd.static(gs.qd_float == qd.f32):
            margin = 1e-5
        positive = coefficient[0] - qd.sqrt(coefficient[1] ** 2 + coefficient[2] ** 2)
        reference_positive = reference[0] - qd.sqrt(reference[1] ** 2 + reference[2] ** 2)
        analytical = contained and contacts.hertz_profile[None]
        analytical = analytical and (
            initial or (positive > margin * coefficient.norm() and reference_positive > margin * reference.norm())
        )
        if analytical:
            if i_lane == 0:
                exact_gram, load = func_hertz_disk_moments(
                    i_f, closest, remaining, center, radius, first, second, surface
                )
                gram = exact_gram / weight_sum
                if not initial:
                    gradient = gram @ coefficient
                    nodal = load @ coefficient / weight_sum
                    objective = -0.5 * (coefficient - reference).dot(gram @ (coefficient + reference))
                candidates = 1
        else:
            n = surface.pressure_rules.orders[i_rule][0]
            i_start = surface.pressure_rules.offsets[i_rule][0]
            stride = 1
            if qd.static(cooperative):
                stride = 32
            n_rays = qd.select(contained, n * n, 3 * n)
            for i_chunk in range((n_rays + stride - 1) // stride):
                i_ray = i_chunk * stride + i_lane
                if i_ray < n_rays:
                    i_triangle = qd.select(contained, 0, i_ray // n)
                    i_angular = i_ray % n
                    v = surface.pressure_rules.nodes[i_start + i_angular][0]
                    weight_v = surface.pressure_rules.weights[i_start + i_angular][0]
                    ray = qd.Vector.zero(gs.qd_float, 3)
                    area, limit = gs.qd_float(0.0), gs.qd_float(0.0)
                    n_radial = 1
                    if not contained:
                        ray, area, limit = func_contact_ray_at(
                            i_f, i_triangle, v, closest, closest - center, remaining, surface
                        )
                        n_radial = qd.select(area > 0.0 and limit > 0.0, n, 0)
                    for i_radial_ in range(n_radial):
                        i_radial = qd.select(contained, i_ray // n, i_radial_)
                        u = surface.pressure_rules.nodes[i_start + i_radial][0]
                        weight_u = surface.pressure_rules.weights[i_start + i_radial][0]
                        coordinates, weight, _position, shape, _face = func_contact_sample_at(
                            i_f,
                            i_triangle,
                            i_c,
                            i_b,
                            u,
                            v,
                            weight_u,
                            weight_v,
                            qd.Vector.zero(gs.qd_float, 3),
                            0.0,
                            qd.Vector.zero(gs.qd_float, 6),
                            first,
                            second,
                            closest,
                            remaining,
                            contained,
                            ray,
                            area,
                            limit,
                            contacts,
                            surface,
                            True,
                        )
                        invalid += gs.qd_int(qd.math.isnan(weight) or qd.math.isinf(weight) or weight < 0.0)
                        if weight > 0.0:
                            candidates += 1
                            weight /= weight_sum
                            profile = qd.max(0.0, coordinates.dot(coefficient))
                            reference_profile = qd.max(0.0, coordinates.dot(reference))
                            if initial or profile > 0.0:
                                gram += weight * coordinates.outer_product(coordinates)
                            if not initial:
                                gradient += weight * profile * coordinates
                                nodal += weight * profile * shape
                                difference = profile - reference_profile
                                if profile > 0.0 and reference_profile > 0.0:
                                    difference = coordinates.dot(coefficient - reference)
                                objective -= 0.5 * weight * difference * (profile + reference_profile)
    if qd.static(cooperative):
        for a, b in qd.static(qd.ndrange(3, 3)):
            gram[a, b] = qd.simt.subgroup.reduce_all_add(gram[a, b])
        for a in qd.static(range(3)):
            gradient[a] = qd.simt.subgroup.reduce_all_add(gradient[a])
        for a in qd.static(range(6)):
            nodal[a] = qd.simt.subgroup.reduce_all_add(nodal[a])
        objective = qd.simt.subgroup.reduce_all_add(objective)
        candidates = qd.simt.subgroup.reduce_all_add(candidates)
        invalid = qd.simt.subgroup.reduce_all_add(invalid)
    total = gradient[0] * contacts.force[i_c, i_b]
    moment = qd.Vector.zero(gs.qd_float, 3)
    for i_local in qd.static(range(6)):
        i_n = info.surface_nodes[i_f, i_local]
        moment += (info.vertices[i_n] - contacts.position[i_c, i_b]).cross(nodal[i_local] * contacts.force[i_c, i_b])
    return gram, gradient, objective, nodal, total, moment, candidates, invalid


@qd.func
def func_adaptive_integrals(
    i_c: int,
    i_b: int,
    i_rule: int,
    i_lane: int,
    coefficient: qd.types.vector(3),
    reference: qd.types.vector(3),
    weight_sum: float,
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    initial: bool,
    cooperative: qd.template(),
):
    gram = qd.Matrix.zero(gs.qd_float, 3, 3)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    objective = gs.qd_float(0.0)
    candidates, invalid = 0, 0
    for i_f in range(surface.face_origin.shape[0]):
        face_gram, face_gradient, face_objective, _nodal, _total, _moment, face_candidates, face_invalid = (
            func_adaptive_face(
                i_f,
                i_c,
                i_b,
                i_rule,
                i_lane,
                coefficient,
                reference,
                weight_sum,
                contacts,
                info,
                surface,
                initial,
                cooperative,
            )
        )
        gram += face_gram
        gradient += face_gradient
        objective += face_objective
        candidates += face_candidates
        invalid += face_invalid
    if not initial:
        gradient[0] -= 1.0
        objective += coefficient[0] - reference[0]
    return gram, gradient, objective, candidates, invalid


@qd.func
def func_adaptive_fit(
    i_c: int,
    i_b: int,
    i_rule: int,
    i_lane: int,
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    cooperative: qd.template(),
):
    coefficient = qd.Vector.zero(gs.qd_float, 3)
    weight_sum = gs.qd_float(1.0)
    status, candidates, invalid, iterations = 3, 0, 0, 0
    accepted, line = False, False
    fraction, slope = gs.qd_float(1.0), gs.qd_float(0.0)
    step = qd.Vector.zero(gs.qd_float, 3)
    line_evaluations = 0
    # One evaluator serves initialization, Newton evaluation and Armijo backtracking.
    for i_evaluation in range(1 + 80 * 41):
        if accepted or invalid > 0 or (i_evaluation > 0 and not line and iterations >= 80):
            break
        candidate = coefficient
        if line:
            candidate = coefficient - fraction * step
        gram, gradient, objective, sample_count, sample_invalid = func_adaptive_integrals(
            i_c,
            i_b,
            i_rule,
            i_lane,
            candidate,
            coefficient,
            weight_sum,
            contacts,
            info,
            surface,
            i_evaluation == 0,
            cooperative,
        )
        invalid += sample_invalid
        if i_evaluation == 0:
            coefficient, weight_sum, status = func_pressure_initial(gram)
            candidates = sample_count
            accepted = status == 0
            coefficient *= weight_sum
            if status == 3:
                invalid += gs.qd_int(qd.math.isnan(gram.norm()) or qd.math.isinf(gram.norm()))
                if invalid == 0:
                    break
        elif not line:
            iterations += 1
            accepted = gradient.norm() <= _PRESSURE_FIT_TOLERANCE
            proposed_step, nonsingular = func_pressure_step(gram, gradient)
            if not accepted and nonsingular:
                step = proposed_step
                slope = gradient.dot(step)
                fraction, line = gs.qd_float(1.0), True
                line_evaluations = 0
            elif not accepted:
                break
        else:
            line_evaluations += 1
            if objective >= 1e-4 * fraction * slope - 1e-15:
                coefficient, line = candidate, False
            else:
                fraction *= 0.5
                if line_evaluations >= 40:
                    line = False
    if weight_sum > 0.0:
        coefficient /= weight_sum
    status = 0 if accepted and invalid == 0 else 3
    return coefficient, weight_sum, status, candidates, invalid, iterations


@qd.func
def func_adaptive_check(
    i_c: int,
    i_b: int,
    i_rule: int,
    i_check: int,
    i_lane: int,
    coefficient: qd.types.vector(3),
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    cooperative: qd.template(),
):
    """Compare fixed-coefficient original face loads before shared-node cancellation."""
    check_gradient = qd.Vector.zero(gs.qd_float, 3)
    total = qd.Vector.zero(gs.qd_float, 3)
    moment = qd.Vector.zero(gs.qd_float, 3)
    raw_error, normalized_error = gs.qd_float(0.0), gs.qd_float(0.0)
    check_mass = gs.qd_float(0.0)
    invalid, missing = 0, 0
    for a in qd.static(range(3)):
        invalid += gs.qd_int(qd.math.isnan(coefficient[a]) or qd.math.isinf(coefficient[a]))
    # The first pass finds one mass over the complete source. The second compares local weights.
    for i_phase in range(2):
        for i_f in range(surface.face_origin.shape[0]):
            candidate_weights = qd.Vector.zero(gs.qd_float, 6)
            candidate_count = 0
            for i_pass in range(qd.select(i_phase == 0, 1, 2)):
                i_current = qd.select(i_phase == 0 or i_pass == 1, i_check, i_rule)
                _gram, face_gradient, _objective, nodal, face_total, face_moment, candidates, face_invalid = (
                    func_adaptive_face(
                        i_f,
                        i_c,
                        i_b,
                        i_current,
                        i_lane,
                        coefficient,
                        coefficient,
                        1.0,
                        contacts,
                        info,
                        surface,
                        False,
                        cooperative,
                    )
                )
                invalid += face_invalid
                if i_phase == 0:
                    check_gradient += face_gradient
                elif i_pass == 0:
                    candidate_weights = nodal
                    candidate_count = candidates
                    total += face_total
                    moment += face_moment
                else:
                    missing += gs.qd_int(candidate_count > 0 and candidates == 0)
                    for a in qd.static(range(6)):
                        raw_error += qd.abs(candidate_weights[a] - nodal[a])
                        normalized_error += qd.abs(
                            candidate_weights[a] - nodal[a] / qd.select(check_mass > 0.0, check_mass, 1.0)
                        )
        if i_phase == 0:
            check_mass = check_gradient[0]
    target = qd.Vector([1.0, 0.0, 0.0])
    raw_gradient = (check_gradient - target).norm()
    normalized_gradient = (check_gradient / qd.select(check_mass > 0.0, check_mass, 1.0) - target).norm()
    errors = qd.Vector([raw_gradient, raw_error, normalized_gradient, normalized_error])
    for a in qd.static(range(4)):
        invalid += gs.qd_int(qd.math.isnan(errors[a]) or qd.math.isinf(errors[a]))
    invalid += gs.qd_int(qd.math.isnan(check_mass) or qd.math.isinf(check_mass))
    missing += gs.qd_int(check_mass <= 0.0)
    force = contacts.force[i_c, i_b]
    force_error = (total - force).norm()
    moment_error = moment.norm()
    reason = 0
    i_error = 0
    if qd.static(_NORMALIZE_CHECK_MASS):
        i_error = 2
    if errors[i_error] > _ADMISSION_TOLERANCE:
        reason |= _CHECK_GRADIENT
    if errors[i_error + 1] > _ADMISSION_TOLERANCE:
        reason |= _CHECK_LOAD
    if force_error > 1e-8 * force.norm() or moment_error > 1e-8 * force.norm() * contacts.radius[i_c, i_b]:
        reason |= _SOURCE_WRENCH
    if missing > 0:
        reason |= _MISSING_SUPPORT
    if invalid > 0:
        reason |= _NONFINITE
    return reason, errors, check_mass, force_error, moment_error


@qd.func
def func_adaptive_contact(
    i_pair: int,
    i_c: int,
    i_b: int,
    i_lane: int,
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    prepared: qd.template(),
    reuse_moments: qd.template(),
    cooperative: qd.template(),
):
    coefficient = qd.Vector.zero(gs.qd_float, 3)
    weight_sum = gs.qd_float(1.0)
    status = 3
    if qd.static(prepared):
        coefficient = contacts.coefficient[i_c, i_b]
        weight_sum = contacts.weight_sum[i_c, i_b]
        status = contacts.status[i_c, i_b]
    reason = _FIT_FAILURE
    errors = qd.Vector.zero(gs.qd_float, 4)
    check_mass = gs.qd_float(0.0)
    force_error, moment_error = gs.qd_float(0.0), gs.qd_float(0.0)
    attempts, selected_rule = 0, 1
    if i_lane == 0:
        qd.atomic_add(contacts.admission.counters[i_b][0], 1)
    # Each source owns its stage transitions. Accepted neighbors retain their fitted coefficient.
    for i_stage in range(3):
        if reason != 0 and (reason & _NONFINITE) == 0:
            selected_rule = i_stage + 1
            attempts += 1
            invalid = 0
            candidates, iterations = 0, 0
            fit_current = True
            if qd.static(prepared):
                fit_current = i_stage > 0
                if not fit_current:
                    iterations = contacts.evaluations[i_c, i_b]
            if fit_current:
                coefficient, weight_sum, status, candidates, invalid, iterations = func_adaptive_fit(
                    i_c, i_b, selected_rule, i_lane, contacts, info, surface, cooperative
                )
            if i_lane == 0:
                qd.atomic_add(contacts.admission.counters[i_b][2 + i_stage], 1)
                contacts.candidate_count[i_c, i_b] += candidates
                if fit_current:
                    contacts.evaluations[i_c, i_b] += iterations
                if i_stage == 0 and status == 0 and iterations == 0:
                    qd.atomic_add(contacts.admission.counters[i_b][8], 1)
            reason, errors, check_mass, force_error, moment_error = func_adaptive_check(
                i_c, i_b, selected_rule, i_stage, i_lane, coefficient, contacts, info, surface, cooperative
            )
            if status != 0:
                reason |= _FIT_FAILURE
            if invalid > 0:
                reason |= _NONFINITE
            if i_lane == 0:
                qd.atomic_add(contacts.admission.counters[i_b][1], 1)
                for a in qd.static(range(4)):
                    contacts.admission.stage_errors[i_c, i_b][i_stage, a] = errors[a]
                contacts.admission.stage_mass[i_c, i_b][i_stage] = check_mass
                contacts.admission.stage_status[i_c, i_b][i_stage] = status
                contacts.admission.stage_reason[i_c, i_b][i_stage] = reason
                contacts.admission.stage_evaluations[i_c, i_b][i_stage] = iterations
    if i_lane == 0:
        contacts.coefficient[i_c, i_b] = coefficient
        contacts.weight_sum[i_c, i_b] = weight_sum
        contacts.admission.rule[i_c, i_b] = qd.Vector([selected_rule])
        contacts.admission.reason[i_c, i_b] = qd.Vector([reason])
        contacts.admission.errors[i_c, i_b] = errors
        contacts.admission.check_mass[i_c, i_b] = qd.Vector([check_mass])
        contacts.admission.attempts[i_c, i_b] = qd.Vector([attempts])
        contacts.force_error[i_c, i_b] = force_error
        contacts.moment_error[i_c, i_b] = moment_error
        failure_status = 3
        if (reason & _SOURCE_WRENCH) != 0 and (reason & _FIT_FAILURE) == 0:
            failure_status = 4
        contacts.status[i_c, i_b] = 0 if reason == 0 else failure_status
        if attempts > 1:
            qd.atomic_add(contacts.admission.counters[i_b][5], 1)
            if qd.static(reuse_moments):  # noqa: SIM102 - specialize before testing the cache index
                if i_pair < workspace.affine_pressure.shape[0]:
                    workspace.affine_pressure[i_pair] = False
        if reason != 0:
            qd.atomic_add(contacts.admission.counters[i_b][6], 1)


@qd.func
def func_adaptive_initialize(contacts: StressContactState):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        contacts.admission.rule[i_c, i_b] = qd.Vector([1])
        contacts.admission.reason[i_c, i_b] = qd.Vector([0])
        contacts.admission.errors[i_c, i_b] = qd.Vector.zero(gs.qd_float, 4)
        contacts.admission.check_mass[i_c, i_b] = qd.Vector([0.0])
        contacts.admission.attempts[i_c, i_b] = qd.Vector([0])
        contacts.admission.stage_errors[i_c, i_b] = qd.Matrix.zero(gs.qd_float, 3, 4)
        contacts.admission.stage_mass[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        contacts.admission.stage_status[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        contacts.admission.stage_reason[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        contacts.admission.stage_evaluations[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        if contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] != 0:
            contacts.admission.reason[i_c, i_b] = qd.Vector([_INVALID_SOURCE])
            qd.atomic_add(contacts.admission.counters[i_b][0], 1)
            qd.atomic_add(contacts.admission.counters[i_b][6], 1)
    for i_b in range(contacts.admission.overflow.shape[0]):
        contacts.admission.overflow[i_b] = qd.Vector([0])


@qd.func
def func_adaptive_pressure(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
    reuse_moments: qd.template(),
):
    func_adaptive_initialize(contacts)
    func_pack_contacts(contacts)
    if qd.static(reuse_moments):
        func_pressure_moments(contacts, state, info, surface, workspace, cached_bounds)
        func_pressure_warp(contacts, surface, True, workspace.refined_pressure, True)
    else:
        func_pressure_warp(contacts, surface, True, contacts.valid, False)
    func_pressure_correct_warp(contacts, surface)
    qd.loop_config(block_dim=128)
    for i_thread in range(contacts.active_count[None] * 32):
        i_pair, i_lane = i_thread // 32, i_thread % 32
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        func_adaptive_contact(i_pair, i_c, i_b, i_lane, contacts, info, surface, workspace, True, reuse_moments, True)


@qd.kernel(graph=True)
def kernel_adaptive_pressure_warp(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
    reuse_moments: qd.template(),
):
    func_adaptive_pressure(contacts, state, info, surface, workspace, cached_bounds, reuse_moments)


@qd.kernel(graph=True)
def kernel_adaptive_pressure_serial(
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
):
    func_adaptive_initialize(contacts)
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] == 0 and contacts.force[i_c, i_b].norm() > 1e-30:
            func_adaptive_contact(0, i_c, i_b, 0, contacts, info, surface, workspace, False, False, False)


def kernel_adaptive_pressure(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cooperative: bool,
    cached_bounds: bool,
    reuse_moments: bool,
):
    if cooperative and gs.backend == gs.cuda:
        kernel_adaptive_pressure_warp(contacts, state, info, surface, workspace, cached_bounds, reuse_moments)
    else:
        kernel_adaptive_pressure_serial(contacts, info, surface, workspace)


@qd.kernel
def kernel_reset_adaptive(envs_idx: qd.types.ndarray(), contacts: StressContactState):
    for i_c, i_selected in qd.ndrange(contacts.valid.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        contacts.admission.rule[i_c, i_b] = qd.Vector([1])
        contacts.admission.reason[i_c, i_b] = qd.Vector([0])
        contacts.admission.errors[i_c, i_b] = qd.Vector.zero(gs.qd_float, 4)
        contacts.admission.check_mass[i_c, i_b] = qd.Vector([0.0])
        contacts.admission.attempts[i_c, i_b] = qd.Vector([0])
        contacts.admission.stage_errors[i_c, i_b] = qd.Matrix.zero(gs.qd_float, 3, 4)
        contacts.admission.stage_mass[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        contacts.admission.stage_status[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        contacts.admission.stage_reason[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        contacts.admission.stage_evaluations[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        contacts.admission.counters[i_b] = qd.Vector.zero(qd.i64, 9)
        contacts.admission.overflow[i_b] = qd.Vector([0])
