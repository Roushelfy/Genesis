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
from .scatter import StressScatterWorkspace, func_prepare_scatter, func_pressure_moments
from .surface import StressSurfaceInfo, func_surface_face_admitted

_ADMISSION_TOLERANCE = 1e-3
_FIT_FAILURE = 1
_CHECK_GRADIENT = 2
_CHECK_LOAD = 4
_SOURCE_WRENCH = 8
_NONFINITE = 16
_MISSING_SUPPORT = 32
_INVALID_SOURCE = 64
_ADMISSION_VERSION = 4


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
    i_pair: int,
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
    workspace: StressScatterWorkspace,
    cached_faces: bool,
    initial: bool,
    cooperative: qd.template(),
):
    gram = qd.Matrix.zero(gs.qd_float, 3, 3)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    objective = gs.qd_float(0.0)
    candidates, invalid = 0, 0
    n_faces = surface.face_origin.shape[0]
    i_task = -1
    if cached_faces:
        n_faces = workspace.source_heads[i_pair][1]
        i_task = workspace.source_heads[i_pair][0]
    for i_face in range(n_faces):
        i_f = i_face
        if cached_faces:
            i_f = workspace.tasks[i_task][1]
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
        if cached_faces:
            i_task = workspace.next_task[i_task][0]
    if not initial:
        gradient[0] -= 1.0
        objective += coefficient[0] - reference[0]
    return gram, gradient, objective, candidates, invalid


@qd.func
def func_adaptive_fit(
    i_pair: int,
    i_c: int,
    i_b: int,
    i_rule: int,
    i_lane: int,
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_faces: qd.template(),
    cooperative: qd.template(),
):
    # The pool is an order-independent geometric superset for every rule and
    # pressure coefficient. Validate ownership/completeness once per fit; the
    # immutable chain is then reused by initialization and every Armijo trial.
    cached = False
    if qd.static(cached_faces):
        cached = workspace.count[None] <= workspace.tasks.shape[0] and i_pair < workspace.source_heads.shape[0]
        n_tasks, i_task = 0, -1
        if cached:
            n_tasks, i_task = workspace.source_heads[i_pair][1], workspace.source_heads[i_pair][0]
            cached = n_tasks >= 0 and n_tasks <= surface.face_origin.shape[0]
        for _entry in range(qd.select(cached, n_tasks, 0)):
            if cached:
                cached = i_task >= 0 and i_task < workspace.count[None]
                if cached:
                    task = workspace.tasks[i_task]
                    cached = task[0] == i_pair and task[1] >= 0 and task[1] < surface.face_origin.shape[0]
                    i_task = workspace.next_task[i_task][0]
        cached = cached and i_task == -1
    coefficient = qd.Vector.zero(gs.qd_float, 3)
    weight_sum = gs.qd_float(1.0)
    status, candidates, invalid, iterations = 3, 0, 0, 0
    accepted, line = False, False
    fraction, slope = gs.qd_float(1.0), gs.qd_float(0.0)
    step = qd.Vector.zero(gs.qd_float, 3)
    line_evaluations = 0
    stop = False
    # One evaluator serves initialization, Newton evaluation and Armijo backtracking.
    for i_evaluation in range(1 + 80 * 41):
        if stop or accepted or invalid > 0 or (i_evaluation > 0 and not line and iterations >= 80):
            break
        candidate = coefficient
        if line:
            candidate = coefficient - fraction * step
        gram, gradient, objective, sample_count, sample_invalid = func_adaptive_integrals(
            i_pair,
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
            workspace,
            cached,
            i_evaluation == 0,
            cooperative,
        )
        if i_lane == 0:
            invalid += sample_invalid
            if i_evaluation == 0:
                coefficient, weight_sum, status = func_pressure_initial(gram)
                candidates = sample_count
                accepted = status == 0
                coefficient *= weight_sum
                if status == 3:
                    invalid += gs.qd_int(qd.math.isnan(gram.norm()) or qd.math.isinf(gram.norm()))
                    if invalid == 0:
                        stop = True
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
                    stop = True
            else:
                line_evaluations += 1
                if objective >= 1e-4 * fraction * slope - 1e-15:
                    coefficient, line = candidate, False
                else:
                    fraction *= 0.5
                    if line_evaluations >= 40:
                        line = False
        if qd.static(cooperative):
            # One lane owns iteration decisions; all lanes enter subsequent collectives together
            for a in qd.static(range(3)):
                coefficient[a] = qd.simt.subgroup.broadcast(coefficient[a], 0)
                step[a] = qd.simt.subgroup.broadcast(step[a], 0)
            weight_sum = qd.simt.subgroup.broadcast(weight_sum, 0)
            fraction = qd.simt.subgroup.broadcast(fraction, 0)
            slope = qd.simt.subgroup.broadcast(slope, 0)
            status = qd.simt.subgroup.broadcast(status, 0)
            candidates = qd.simt.subgroup.broadcast(candidates, 0)
            invalid = qd.simt.subgroup.broadcast(invalid, 0)
            iterations = qd.simt.subgroup.broadcast(iterations, 0)
            line_evaluations = qd.simt.subgroup.broadcast(line_evaluations, 0)
            accepted = qd.simt.subgroup.broadcast(gs.qd_int(accepted), 0) != 0
            line = qd.simt.subgroup.broadcast(gs.qd_int(line), 0) != 0
            stop = qd.simt.subgroup.broadcast(gs.qd_int(stop), 0) != 0
    if weight_sum > 0.0:
        coefficient /= weight_sum
    status = 0 if accepted and invalid == 0 else 3
    return coefficient, weight_sum, status, candidates, invalid, iterations


@qd.func
def func_adaptive_node_error(i_pair: int, n_tasks: int, surface_nodes: qd.Tensor, workspace: StressScatterWorkspace):
    """Assemble one source's signed original P2-node differences before taking L1."""
    node_error = gs.qd_float(0.0)
    i_task = workspace.source_heads[i_pair][0]
    for _entry in range(n_tasks):
        if i_task >= 0:
            i_f = workspace.tasks[i_task][1]
            for a in qd.static(range(6)):
                node = surface_nodes[i_f, a]
                first = True
                value = gs.qd_float(0.0)
                i_other = workspace.source_heads[i_pair][0]
                before_current = True
                for _other_entry in range(n_tasks):
                    if i_other >= 0:
                        other_face = workspace.tasks[i_other][1]
                        for b in qd.static(range(6)):
                            if surface_nodes[other_face, b] == node:
                                value += workspace.load_difference[i_other][b]
                                if (before_current and i_other != i_task) or (i_other == i_task and b < a):
                                    first = False
                        if i_other == i_task:
                            before_current = False
                        i_other = workspace.next_task[i_other][0]
                if first:
                    node_error += qd.abs(value)
            i_task = workspace.next_task[i_task][0]
    return node_error


@qd.func
def func_adaptive_check(
    i_pair: int,
    i_c: int,
    i_b: int,
    i_rule: int,
    i_check: int,
    i_lane: int,
    coefficient: qd.types.vector(3),
    check_coefficient: qd.types.vector(3),
    contacts: StressContactState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    prepared: qd.template(),
    cooperative: qd.template(),
):
    """Compare independently fitted source loads after assembly on original P2 nodes."""
    candidate_gradient = qd.Vector.zero(gs.qd_float, 3)
    check_gradient = qd.Vector.zero(gs.qd_float, 3)
    total = qd.Vector.zero(gs.qd_float, 3)
    moment = qd.Vector.zero(gs.qd_float, 3)
    check_total = qd.Vector.zero(gs.qd_float, 3)
    check_moment = qd.Vector.zero(gs.qd_float, 3)
    face_error, node_error = gs.qd_float(0.0), gs.qd_float(0.0)
    invalid, missing = 0, 0
    cached = False
    if qd.static(prepared):
        cached = workspace.count[None] <= workspace.tasks.shape[0] and i_pair < workspace.source_heads.shape[0]
    n_tasks = surface.face_origin.shape[0]
    if cached:
        n_tasks = workspace.source_heads[i_pair][1]
        cached = n_tasks >= 0 and n_tasks <= surface.face_origin.shape[0]
    if not cached:
        n_tasks = surface.face_origin.shape[0]
    for a in qd.static(range(3)):
        invalid += gs.qd_int(qd.math.isnan(coefficient[a]) or qd.math.isinf(coefficient[a]))
        invalid += gs.qd_int(qd.math.isnan(check_coefficient[a]) or qd.math.isinf(check_coefficient[a]))
    i_task = -1
    if cached:
        i_task = workspace.source_heads[i_pair][0]
    for i_face in range(n_tasks):
        i_f = i_face
        if cached:
            if i_task < 0:
                break
            i_f = workspace.tasks[i_task][1]
        candidate_weights = qd.Vector.zero(gs.qd_float, 6)
        candidate_count = 0
        difference = qd.Vector.zero(gs.qd_float, 6)
        for i_pass in range(2):
            i_current = qd.select(i_pass == 0, i_rule, i_check)
            beta = coefficient
            if i_pass == 1:
                beta = check_coefficient
            _gram, face_gradient, _objective, nodal, face_total, face_moment, candidates, face_invalid = (
                func_adaptive_face(
                    i_f, i_c, i_b, i_current, i_lane, beta, beta, 1.0, contacts, info, surface, False, cooperative
                )
            )
            invalid += face_invalid
            if i_pass == 0:
                candidate_gradient += face_gradient
                candidate_weights = nodal
                candidate_count = candidates
                total += face_total
                moment += face_moment
            else:
                check_gradient += face_gradient
                check_total += face_total
                check_moment += face_moment
                missing += gs.qd_int(candidate_count > 0 and candidates == 0)
                difference = candidate_weights - nodal
                for a in qd.static(range(6)):
                    face_error += qd.abs(difference[a])
        if cached:
            if i_lane == 0:
                workspace.load_difference[i_task] = difference
            i_task = workspace.next_task[i_task][0]
    if cached:
        if qd.static(cooperative):
            qd.simt.subgroup.sync()
        if i_lane == 0:
            node_error = func_adaptive_node_error(i_pair, n_tasks, info.surface_nodes, workspace)
        if qd.static(cooperative):
            node_error = qd.simt.subgroup.reduce_all_add(node_error)
    else:
        # Complete portable/overflow route: evaluate only faces incident to each
        # original node. No truncated pool and no dense contact-by-node storage.
        for node in range(info.vertices.shape[0]):
            value = gs.qd_float(0.0)
            for i_f in range(surface.face_origin.shape[0]):
                local = -1
                for a in qd.static(range(6)):
                    if info.surface_nodes[i_f, a] == node:
                        local = a
                if local >= 0:
                    for i_pass in range(2):
                        i_current = qd.select(i_pass == 0, i_rule, i_check)
                        beta = coefficient
                        if i_pass == 1:
                            beta = check_coefficient
                        _gram, _gradient, _objective, nodal, _total, _moment, _count, _invalid = func_adaptive_face(
                            i_f,
                            i_c,
                            i_b,
                            i_current,
                            i_lane,
                            beta,
                            beta,
                            1.0,
                            contacts,
                            info,
                            surface,
                            False,
                            cooperative,
                        )
                        value += qd.select(i_pass == 0, 1.0, -1.0) * nodal[local]
            node_error += qd.abs(value)
    check_mass = check_gradient[0]
    target = qd.Vector([1.0, 0.0, 0.0])
    errors = qd.Vector([(check_gradient - target).norm(), face_error, (candidate_gradient - target).norm(), node_error])
    for a in qd.static(range(4)):
        invalid += gs.qd_int(qd.math.isnan(errors[a]) or qd.math.isinf(errors[a]))
    invalid += gs.qd_int(qd.math.isnan(check_mass) or qd.math.isinf(check_mass))
    missing += gs.qd_int(check_mass <= 0.0)
    force = contacts.force[i_c, i_b]
    force_error = (total - force).norm()
    moment_error = moment.norm()
    checker_force_error = (check_total - force).norm()
    checker_moment_error = check_moment.norm()
    reason = 0
    if errors[0] > _PRESSURE_FIT_TOLERANCE or errors[2] > _PRESSURE_FIT_TOLERANCE:
        reason |= _CHECK_GRADIENT
    if node_error > _ADMISSION_TOLERANCE:
        reason |= _CHECK_LOAD
    if force_error > 1e-8 * force.norm() or moment_error > 1e-8 * force.norm() * contacts.radius[i_c, i_b]:
        reason |= _SOURCE_WRENCH
    if (
        checker_force_error > 1e-8 * force.norm()
        or checker_moment_error > 1e-8 * force.norm() * contacts.radius[i_c, i_b]
    ):
        reason |= _SOURCE_WRENCH
    if missing > 0:
        reason |= _MISSING_SUPPORT
    if invalid > 0:
        reason |= _NONFINITE
    return reason, errors, check_mass, force_error, moment_error, checker_force_error, checker_moment_error


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
    checker_coefficient = qd.Vector.zero(gs.qd_float, 3)
    attempts, selected_rule = 0, 1
    if i_lane == 0:
        qd.atomic_add(contacts.admission.counters[i_b][0], 1)
    # Each source owns its stage transitions. Accepted neighbors retain their fitted coefficient.
    for i_stage in range(3):
        if reason != 0:
            selected_rule = i_stage + 1
            attempts += 1
            invalid = 0
            candidates, iterations = 0, 0
            check_status, check_iterations = 3, 0
            checker_coefficient = qd.Vector.zero(gs.qd_float, 3)
            errors = qd.Vector.zero(gs.qd_float, 4)
            check_mass = gs.qd_float(0.0)
            force_error, moment_error = gs.qd_float(0.0), gs.qd_float(0.0)
            checker_force_error, checker_moment_error = gs.qd_float(0.0), gs.qd_float(0.0)
            fit_current = True
            if qd.static(prepared):
                fit_current = i_stage > 0
                if not fit_current:
                    iterations = contacts.evaluations[i_c, i_b]
            if fit_current:
                coefficient, weight_sum, status, candidates, invalid, iterations = func_adaptive_fit(
                    i_pair,
                    i_c,
                    i_b,
                    selected_rule,
                    i_lane,
                    contacts,
                    info,
                    surface,
                    workspace,
                    qd.static(prepared and reuse_moments),
                    cooperative,
                )
            if i_lane == 0:
                qd.atomic_add(contacts.admission.counters[i_b][2 + i_stage], 1)
                contacts.candidate_count[i_c, i_b] += candidates
                if fit_current:
                    contacts.evaluations[i_c, i_b] += iterations
                if i_stage == 0 and status == 0 and iterations == 0:
                    qd.atomic_add(contacts.admission.counters[i_b][8], 1)
            reason = _FIT_FAILURE
            if status == 0 and invalid == 0:
                checker_coefficient, _check_weight, check_status, _check_count, check_invalid, check_iterations = (
                    func_adaptive_fit(
                        i_pair,
                        i_c,
                        i_b,
                        i_stage,
                        i_lane,
                        contacts,
                        info,
                        surface,
                        workspace,
                        qd.static(prepared and reuse_moments),
                        cooperative,
                    )
                )
                invalid += check_invalid
                if check_status == 0 and check_invalid == 0:
                    reason, errors, check_mass, force_error, moment_error, checker_force_error, checker_moment_error = (
                        func_adaptive_check(
                            i_pair,
                            i_c,
                            i_b,
                            selected_rule,
                            i_stage,
                            i_lane,
                            coefficient,
                            checker_coefficient,
                            contacts,
                            info,
                            surface,
                            workspace,
                            prepared,
                            cooperative,
                        )
                    )
                if i_lane == 0:
                    qd.atomic_add(contacts.admission.counters[i_b][1], 1)
            if invalid > 0:
                reason |= _NONFINITE
            if i_lane == 0:
                for a in qd.static(range(4)):
                    contacts.admission.stage_errors[i_c, i_b][i_stage, a] = errors[a]
                force_norm = contacts.force[i_c, i_b].norm()
                radius = contacts.radius[i_c, i_b]
                contacts.admission.stage_source_errors[i_c, i_b][i_stage, 0] = force_error / force_norm
                contacts.admission.stage_source_errors[i_c, i_b][i_stage, 1] = moment_error / (force_norm * radius)
                contacts.admission.stage_source_errors[i_c, i_b][i_stage, 2] = checker_force_error / force_norm
                contacts.admission.stage_source_errors[i_c, i_b][i_stage, 3] = checker_moment_error / (
                    force_norm * radius
                )
                contacts.admission.stage_mass[i_c, i_b][i_stage] = check_mass
                contacts.admission.stage_status[i_c, i_b][i_stage] = status
                contacts.admission.stage_checker_status[i_c, i_b][i_stage] = check_status
                contacts.admission.stage_reason[i_c, i_b][i_stage] = reason
                contacts.admission.stage_evaluations[i_c, i_b][i_stage] = iterations
                contacts.admission.stage_checker_evaluations[i_c, i_b][i_stage] = check_iterations
    if i_lane == 0:
        contacts.coefficient[i_c, i_b] = coefficient
        contacts.weight_sum[i_c, i_b] = weight_sum
        contacts.admission.rule[i_c, i_b] = qd.Vector([selected_rule])
        contacts.admission.reason[i_c, i_b] = qd.Vector([reason])
        contacts.admission.errors[i_c, i_b] = errors
        contacts.admission.check_mass[i_c, i_b] = qd.Vector([check_mass])
        contacts.admission.checker_coefficient[i_c, i_b] = checker_coefficient
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
        contacts.admission.checker_coefficient[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        contacts.admission.stage_checker_status[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        contacts.admission.stage_checker_evaluations[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        contacts.admission.stage_source_errors[i_c, i_b] = qd.Matrix.zero(gs.qd_float, 3, 4)
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
        func_prepare_scatter(contacts, state, info, surface, workspace, cached_bounds)
        func_pressure_warp(contacts, surface, True, contacts.valid, False)
    func_pressure_correct_warp(contacts, surface)
    # All initial sources retain their pair identity even when their first fit
    # fails. The complete task pool precedes stage transitions and final scatter.
    cached = (
        workspace.count[None] <= workspace.tasks.shape[0]
        and contacts.active_count[None] <= workspace.source_heads.shape[0]
    )
    for i_pair in range(qd.select(cached, contacts.active_count[None], 0)):
        workspace.source_heads[i_pair] = qd.Vector([-1, 0])
    for i_task in range(qd.select(cached, workspace.count[None], 0)):
        i_pair = workspace.tasks[i_task][0]
        previous = qd.atomic_exchange(workspace.source_heads[i_pair][0], i_task)
        qd.atomic_add(workspace.source_heads[i_pair][1], 1)
        workspace.next_task[i_task] = qd.Vector([previous])
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
        contacts.admission.checker_coefficient[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        contacts.admission.stage_checker_status[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        contacts.admission.stage_checker_evaluations[i_c, i_b] = qd.Vector.zero(gs.qd_int, 3)
        contacts.admission.stage_source_errors[i_c, i_b] = qd.Matrix.zero(gs.qd_float, 3, 4)
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        contacts.admission.counters[i_b] = qd.Vector.zero(qd.i64, 9)
        contacts.admission.overflow[i_b] = qd.Vector([0])
