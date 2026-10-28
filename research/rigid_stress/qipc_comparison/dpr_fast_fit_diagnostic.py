"""Replay archived volume pressure sources and trace the unchanged native fit.

Scientific execution requires Slurm rtx-mid/image 1_26. The isolated replay
preserves the archived application point, force, center, radius and anchor;
it never advances rigid dynamics or changes production tolerances. A research
copy adds telemetry and is compared with the unmodified native pressure path.
"""

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace


def _api():
    import numpy as np
    import quadrants as qd
    import scipy.optimize
    import scipy.spatial

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.contact import (
        _PRESSURE_FIT_TOLERANCE,
        StressContactState,
        create_contacts,
        func_contact_cell,
        func_contact_index,
        func_contact_ray,
        func_contact_sample,
        func_contact_sample_prepared,
        func_contact_section,
        func_force_frame,
        func_pack_contacts,
        func_patch_grid,
        func_pressure_step,
        func_pressure_warp,
        func_refine_contacts,
        func_surface_face_admitted,
    )
    from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
    from genesis.engine.solvers.rigid.stress.model import StressModel
    from genesis.engine.solvers.rigid.stress.scatter import (
        StressScatterWorkspace,
        create_scatter_workspace,
        func_pressure_moments,
        kernel_pressure_moments,
    )
    from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
    from genesis.utils.array_class import V_MAT, V_VEC, V
    from genesis.utils.misc import qd_to_numpy

    @dataclasses.dataclass(frozen=True)
    class FitTrace:
        events: qd.Tensor
        scalars: qd.Tensor
        coefficient: qd.Tensor
        candidate: qd.Tensor
        gradient: qd.Tensor
        hessian: qd.Tensor
        step: qd.Tensor
        nonsingular: qd.Tensor
        next_state: qd.Tensor
        count: qd.Tensor
        entry: qd.Tensor
        entry_values: qd.Tensor

    @qd.func
    def _fit_trace(
        contact_state: StressContactState,
        surface_info: StressSurfaceInfo,
        trace: FitTrace,
        phase: int,
        target_env: int,
        target_slot: int,
    ):
        n_warps = qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 1)
        qd.loop_config(block_dim=128)
        for i_thread in range(contact_state.active_count[None] * 32 * n_warps):
            lane = i_thread % 32
            i_warp = (i_thread // 32) % n_warps
            slot = i_thread // (32 * n_warps)
            sh_fit = qd.simt.block.SharedArray((4, 13), gs.qd_float)
            sh_faces = qd.simt.block.SharedArray((128,), gs.qd_int)
            sh_count = qd.simt.block.SharedArray((1,), gs.qd_int)
            pair = contact_state.active_pairs[slot]
            i_c, i_b = pair[0], pair[1]
            # DPR_TRACE_BEGIN
            selected = i_c == target_slot and i_b == target_env
            if selected and lane == 0 and i_warp == 0:
                trace.entry[phase] = qd.Vector(
                    [
                        contact_state.status[i_c, i_b],
                        gs.qd_int(contact_state.is_refined[i_c, i_b]),
                        contact_state.candidate_count[i_c, i_b],
                        contact_state.evaluations[i_c, i_b],
                    ]
                )
                trace.entry_values[phase] = qd.Vector(
                    [
                        contact_state.weight_sum[i_c, i_b],
                        contact_state.coefficient[i_c, i_b][0],
                        contact_state.coefficient[i_c, i_b][1],
                        contact_state.coefficient[i_c, i_b][2],
                    ]
                )
            # DPR_TRACE_END
            if contact_state.status[i_c, i_b] == 5:
                first, second = func_force_frame(contact_state.force[i_c, i_b])
                center, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
                low, dimensions = func_patch_grid(center, radius, surface_info.grid)
                is_cached = n_warps > 1 and contact_state.is_refined[i_c, i_b]
                n_faces = surface_info.face_origin.shape[0]
                if is_cached:
                    if lane == 0 and i_warp == 0:
                        sh_count[0] = 0
                    qd.simt.block.sync()
                    for i_chunk in range((n_faces + 127) // 128):
                        i_f = i_chunk * 128 + i_warp * 32 + lane
                        if i_f < n_faces:
                            distance = qd.max(surface_info.face_bounds_low[i_f] - center, 0.0)
                            distance += qd.max(center - surface_info.face_bounds_high[i_f], 0.0)
                            if distance.norm_sqr() < radius * radius and func_surface_face_admitted(
                                i_f, i_c, i_b, surface_info
                            ):
                                i_cached = qd.atomic_add(sh_count[0], 1)
                                if i_cached < 128:
                                    sh_faces[i_cached] = i_f
                    qd.simt.block.sync()
                    # Oversized footprints retain the exhaustive face walk and all original samples.
                    is_cached = sh_count[0] <= 128
                weight_sum = contact_state.weight_sum[i_c, i_b]
                coefficient = contact_state.coefficient[i_c, i_b] * weight_sum
                accepted, line = False, False
                iterations, line_evaluations = 0, 0
                fraction, reference_objective = gs.qd_float(1.0), gs.qd_float(0.0)
                step = qd.Vector.zero(gs.qd_float, 3)
                slope = gs.qd_float(0.0)
                # At most 80 Newton evaluations, each followed by at most 40 line evaluations.
                # One evaluator body avoids duplicating the integration helper in the kernel.
                for _ in range(80 * 41):
                    if accepted or (not line and iterations >= 80):
                        break
                    candidate = coefficient
                    if line:
                        candidate = coefficient - fraction * step
                    objective = gs.qd_float(0.0)
                    # DPR_TRACE_BEGIN
                    active_samples, active_weight = 0, gs.qd_float(0.0)
                    # DPR_TRACE_END
                    gradient = qd.Vector.zero(gs.qd_float, 3)
                    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
                    n_cells = dimensions[0] * dimensions[1] * dimensions[2] + 1
                    if contact_state.is_refined[i_c, i_b]:
                        n_cells = n_faces
                        if is_cached:
                            n_cells = sh_count[0]
                    for i_group in range((n_cells + n_warps - 1) // n_warps):
                        i_cell = i_group * n_warps + i_warp
                        start, end = 0, 0
                        if i_cell < n_cells:
                            if is_cached:
                                i_cell = sh_faces[i_cell]
                            start, end = func_contact_cell(
                                i_cell, i_c, i_b, low, dimensions, contact_state, surface_info
                            )
                        closest = qd.Vector.zero(gs.qd_float, 3)
                        remaining, contained = gs.qd_float(0.0), False
                        if contact_state.is_refined[i_c, i_b] and end > start:
                            closest, remaining, contained = func_contact_section(i_cell, center, radius, surface_info)
                        n = surface_info.gauss.shape[0]
                        n_quadrature = surface_info.shape.shape[0]
                        is_clipped = contact_state.is_refined[i_c, i_b] and not contained and end > start
                        n_rays = qd.select(
                            is_clipped, qd.select(surface_info.quad_vertices.shape[0] > 0, 4, 3) * n, end - start
                        )
                        for i_chunk in range((n_rays + 31) // 32):
                            i_ray = i_chunk * 32 + lane
                            if i_ray < n_rays:
                                ray = qd.Vector.zero(gs.qd_float, 3)
                                area, limit = gs.qd_float(0.0), gs.qd_float(0.0)
                                n_radial = 1
                                if is_clipped:
                                    ray, area, limit = func_contact_ray(
                                        i_cell,
                                        i_ray // n,
                                        i_ray % n,
                                        closest,
                                        closest - center,
                                        remaining,
                                        surface_info,
                                    )
                                    n_radial = qd.select(area > 0.0 and limit > 0.0, n, 0)
                                for i_radial in range(n_radial):
                                    i_sample = qd.select(
                                        is_clipped, (i_ray // n) * n_quadrature + i_radial * n + i_ray % n, i_ray
                                    )
                                    i_q = func_contact_index(start + i_sample, surface_info)
                                    coordinates, weight, _position, _shape, _face = func_contact_sample_prepared(
                                        i_q,
                                        i_c,
                                        i_b,
                                        first,
                                        second,
                                        contact_state,
                                        surface_info,
                                        closest,
                                        remaining,
                                        contained,
                                        ray,
                                        area,
                                        limit,
                                    )
                                    weight /= weight_sum
                                    profile = qd.max(0.0, coordinates.dot(candidate))
                                    # DPR_TRACE_BEGIN
                                    active_samples += gs.qd_int(profile > 0.0 and weight > 0.0)
                                    active_weight += qd.select(profile > 0.0, weight, 0.0)
                                    # DPR_TRACE_END
                                    objective -= 0.5 * weight * profile * profile
                                    if not line:
                                        gradient += weight * profile * coordinates
                                        if profile > 0.0:
                                            hessian += weight * coordinates.outer_product(coordinates)
                    objective = qd.simt.subgroup.reduce_all_add(objective)
                    # DPR_TRACE_BEGIN
                    active_samples = qd.simt.subgroup.reduce_all_add(active_samples)
                    active_weight = qd.simt.subgroup.reduce_all_add(active_weight)
                    # DPR_TRACE_END
                    if not line:
                        for a in qd.static(range(3)):
                            gradient[a] = qd.simt.subgroup.reduce_all_add(gradient[a])
                        for a, b in qd.static(qd.ndrange(3, 3)):
                            hessian[a, b] = qd.simt.subgroup.reduce_all_add(hessian[a, b])
                    if n_warps > 1:
                        if lane == 0:
                            sh_fit[i_warp, 0] = objective
                            for a in qd.static(range(3)):
                                sh_fit[i_warp, 1 + a] = gradient[a]
                            for a, b in qd.static(qd.ndrange(3, 3)):
                                sh_fit[i_warp, 4 + 3 * a + b] = hessian[a, b]
                        qd.simt.block.sync()
                        objective = 0.0
                        gradient = qd.Vector.zero(gs.qd_float, 3)
                        hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
                        for j_warp in qd.static(range(4)):
                            objective += sh_fit[j_warp, 0]
                            for a in qd.static(range(3)):
                                gradient[a] += sh_fit[j_warp, 1 + a]
                            for a, b in qd.static(qd.ndrange(3, 3)):
                                hessian[a, b] += sh_fit[j_warp, 4 + 3 * a + b]
                        qd.simt.block.sync()
                    objective += candidate[0]
                    # DPR_TRACE_BEGIN
                    if selected and lane == 0 and i_warp == 0:
                        trace.count[phase] = _ + 1
                        trace.events[_, phase] = qd.Vector(
                            [gs.qd_int(line), iterations, line_evaluations, active_samples]
                        )
                        trace.scalars[_, phase] = qd.Vector(
                            [
                                objective,
                                reference_objective,
                                fraction,
                                slope,
                                weight_sum,
                                active_weight,
                                reference_objective + 1e-4 * fraction * slope - 1e-15,
                                objective - reference_objective,
                            ]
                        )
                        trace.coefficient[_, phase] = coefficient
                        trace.candidate[_, phase] = candidate
                        trace.gradient[_, phase] = gradient - qd.Vector([1.0, 0.0, 0.0])
                        trace.hessian[_, phase] = hessian
                    # DPR_TRACE_END
                    if not line:
                        gradient[0] -= 1.0
                        iterations += 1
                        accepted = gradient.norm() <= _PRESSURE_FIT_TOLERANCE
                        proposed_step, nonsingular = func_pressure_step(hessian, gradient)
                        # DPR_TRACE_BEGIN
                        if selected and lane == 0 and i_warp == 0:
                            trace.step[_, phase] = proposed_step
                            trace.nonsingular[_, phase] = nonsingular
                        # DPR_TRACE_END
                        if not accepted and nonsingular:
                            step = proposed_step
                            slope = gradient.dot(step)
                            reference_objective = objective
                            fraction, line = gs.qd_float(1.0), True
                            line_evaluations = 0
                    else:
                        line_evaluations += 1
                        if objective >= reference_objective + 1e-4 * fraction * slope - 1e-15:
                            coefficient, line = candidate, False
                        else:
                            fraction *= 0.5
                            if line_evaluations == 40:
                                line = False
                    # DPR_TRACE_BEGIN
                    if selected and lane == 0 and i_warp == 0:
                        trace.next_state[_, phase] = qd.Vector(
                            [gs.qd_int(accepted), gs.qd_int(line), iterations, line_evaluations]
                        )
                    # DPR_TRACE_END
                if lane == 0 and i_warp == 0:
                    contact_state.coefficient[i_c, i_b] = coefficient / weight_sum
                    contact_state.status[i_c, i_b] = 0 if accepted else 3
                    contact_state.evaluations[i_c, i_b] += iterations

    @qd.kernel
    def _trace_pipeline(
        contacts: StressContactState,
        state: StressState,
        info: StressInfo,
        surface: StressSurfaceInfo,
        workspace: StressScatterWorkspace,
        trace: FitTrace,
        target_env: int,
        target_slot: int,
    ):
        func_pack_contacts(contacts)
        func_pressure_moments(contacts, state, info, surface, workspace, True)
        _fit_trace(contacts, surface, trace, 0, target_env, target_slot)
        func_refine_contacts(contacts)
        func_pressure_warp(contacts, surface, True, workspace.refined_pressure, True)
        _fit_trace(contacts, surface, trace, 1, target_env, target_slot)

    @qd.kernel
    def _samples(
        contacts: StressContactState,
        surface: StressSurfaceInfo,
        target_env: int,
        target_slot: int,
        coordinates: qd.Tensor,
        weights: qd.Tensor,
        positions: qd.Tensor,
        shapes: qd.Tensor,
        faces: qd.Tensor,
    ):
        for i in range(weights.shape[0]):
            count, offset = surface.positions.shape[0], 0
            if contacts.is_refined[target_slot, target_env]:
                count = surface.face_origin.shape[0] * 3 * surface.shape.shape[0]
                offset = surface.positions.shape[0]
            weights[i] = 0.0
            faces[i] = -1
            if i < count:
                first, second = func_force_frame(contacts.force[target_slot, target_env])
                value, weight, point, shape, face = func_contact_sample(
                    offset + i, target_slot, target_env, first, second, contacts, surface
                )
                coordinates[i], weights[i], positions[i], shapes[i], faces[i] = value, weight, point, shape, face

    return SimpleNamespace(**locals())


def _identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return {"nonfinite": repr(value), "raw_value_in_npz": True}
    return value


def _write(path, value):
    path.write_text(json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _algorithm_proof(native_path):
    source = Path(__file__).read_text().splitlines()
    node = next(
        item
        for item in ast.walk(ast.parse("\n".join(source)))
        if isinstance(item, ast.FunctionDef) and item.name == "_fit_trace"
    )
    lines, inside = [], False
    for line in source[node.lineno - 1 : node.end_lineno]:
        if "# DPR_TRACE_BEGIN" in line:
            inside = True
        elif "# DPR_TRACE_END" in line:
            inside = False
        elif not inside:
            lines.append(line.removeprefix("    "))
    traced = ast.parse("\n".join(lines)).body[0]
    native = next(
        item
        for item in ast.parse(native_path.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "func_pressure_correct_warp"
    )
    traced.name, traced.args = native.name, native.args
    traced.decorator_list, native.decorator_list = [], []
    matched = ast.dump(traced, include_attributes=False) == ast.dump(native, include_attributes=False)
    if not matched:
        raise RuntimeError(
            "Instrumented corrector no longer matches the imported native algorithm after telemetry removal."
        )
    return {
        "telemetry_removed_AST_equal": matched,
        "native_source": _identity(native_path),
        "native_function_AST_sha256": hashlib.sha256(ast.dump(native).encode()).hexdigest(),
    }


def _arrays(api, contacts):
    return {
        field.name: api.qd_to_numpy(vars(contacts)[field.name], transpose=True, copy=True)
        for field in dataclasses.fields(contacts)
    }


def _load_contacts(api, saved):
    prefix = "contact_" if "contact_valid" in saved else ""
    valid = saved[prefix + "valid"]
    if valid.ndim != 2:
        raise ValueError("Expected archived environment-first contact arrays.")
    contacts = api.create_contacts(valid.shape[1], valid.shape[0], 0.001, hertz_profile=True)
    for field in dataclasses.fields(contacts):
        vars(contacts)[field.name].fill(0)
    contacts.hertz_profile.fill(True)
    aliases = {"position": "point" if prefix else "position"}
    for name in (
        "position",
        "force",
        "normal",
        "friction",
        "radius",
        "radius_status",
        "valid",
        "center",
        "anchor_face",
    ):
        value = saved[prefix + aliases.get(name, name)]
        vars(contacts)[name].from_numpy(api.np.ascontiguousarray(api.np.swapaxes(value, 0, 1)))
    for name in ("frame_position", "frame_quaternion"):
        if prefix + name in saved:
            vars(contacts)[name].from_numpy(api.np.ascontiguousarray(saved[prefix + name]))
    return contacts


def _trace_summary(api, arrays, phase):
    count = int(arrays["count"][phase])
    events, scalars, gradient, hessian = (
        arrays[name][phase, :count] for name in ("events", "scalars", "gradient", "hessian")
    )
    rows = []
    for index in range(count):
        line = bool(events[index, 0])
        row = {
            "event": index,
            "line_search": line,
            "iterations_before": int(events[index, 1]),
            "line_evaluations_before": int(events[index, 2]),
            "positive_active_samples": int(events[index, 3]),
            "objective": float(scalars[index, 0]),
            "reference_objective": float(scalars[index, 1]),
            "fraction": float(scalars[index, 2]),
            "slope": float(scalars[index, 3]),
            "weight_sum": float(scalars[index, 4]),
            "active_normalized_weight": float(scalars[index, 5]),
            "original_Armijo_absolute_threshold": float(scalars[index, 6]),
            "rounded_objective_difference": float(scalars[index, 7]),
            "coefficient_normalized": arrays["coefficient"][phase, index].tolist(),
            "candidate_normalized": arrays["candidate"][phase, index].tolist(),
            "actual_rounded_candidate_delta": (
                arrays["candidate"][phase, index] - arrays["coefficient"][phase, index]
            ).tolist(),
            "next_accepted_line_iterations_lineevals": arrays["next_state"][phase, index].tolist(),
        }
        if not line:
            matrix = hessian[index]
            row.update(
                gradient=gradient[index].tolist(),
                gradient_norm=float(api.np.linalg.norm(gradient[index])),
                hessian=matrix.tolist(),
                step=arrays["step"][phase, index].tolist(),
                nonsingular=bool(arrays["nonsingular"][phase, index]),
            )
            if api.np.isfinite(matrix).all():
                row["hessian_eigenvalues"] = api.np.linalg.eigvalsh(matrix).tolist()
                if (api.np.diag(matrix) > 0).all():
                    scale = 1 / api.np.sqrt(api.np.diag(matrix))
                    balanced = scale[:, None] * matrix * scale[None, :]
                    row["balanced_hessian_eigenvalues"] = api.np.linalg.eigvalsh(balanced).tolist()
                    row["balanced_hessian_condition_2"] = float(api.np.linalg.cond(balanced))
        rows.append(row)
    entry_status = int(arrays["entry"][phase, 0])
    reason = {
        0: "initial_affine_fit_accepted",
        3: "initial_Gram_or_support_unresolved",
        5: "constrained_corrector_not_entered",
    }.get(entry_status, "native_status_not_admitted")
    if count:
        final = arrays["next_state"][phase, count - 1]
        reason = "fit_tolerance_met" if final[0] else "Newton_evaluation_limit_or_total_evaluator_limit"
    return {
        "phase": phase,
        "entry_status_refined_candidates_evaluations": arrays["entry"][phase].tolist(),
        "entry_weight_sum_coefficient": arrays["entry_values"][phase].tolist(),
        "events": count,
        "stopping_reason": reason,
        "Newton_evaluations": sum(not row["line_search"] for row in rows),
        "Armijo_evaluations": sum(row["line_search"] for row in rows),
        "singular_Newton_evaluations": sum(row.get("nonsingular") is False for row in rows),
        "max_line_searches": sum(
            row["line_search"] and row["next_accepted_line_iterations_lineevals"][3] == 40 for row in rows
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--env", type=int, required=True)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--quadrature", type=int, default=16)
    parser.add_argument("--pressure-quadrature", choices=("edge", "angle"), default="edge")
    parser.add_argument("--tasks-per-env", type=int, default=32)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    if min(args.env, args.slot) < 0 or min(args.quadrature, args.tasks_per_env) < 1:
        parser.error("Use nonnegative source indices and positive integration/workspace sizes.")
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh data-side output.")
    args.output.mkdir(parents=True, exist_ok=False)
    from .artifacts import runtime_identity

    api = _api()
    api.gs.init(backend=api.gs.cuda, precision="64", seed=510000, logging_level="warning")
    native_path = Path(api.gs.__file__).parent / "engine/solvers/rigid/stress/contact.py"
    proof = _algorithm_proof(native_path)
    identities = {
        "archive": _identity(args.archive),
        "mesh": _identity(args.mesh),
        "worker": _identity(Path(__file__)),
        "native_contact": _identity(native_path),
    }
    with api.np.load(args.archive, allow_pickle=False) as saved:
        archived = {key: saved[key] for key in saved.files}
    prefix = "contact_" if "contact_valid" in archived else ""
    replay_inputs = {}
    for name in (
        "valid",
        "position",
        "force",
        "normal",
        "friction",
        "radius",
        "radius_status",
        "center",
        "anchor_face",
    ):
        alias = "point" if prefix and name == "position" else name
        replay_inputs[name] = archived[prefix + alias]
    for name in ("frame_position", "frame_quaternion"):
        if prefix + name in archived:
            replay_inputs[name] = archived[prefix + name]
    if "body_omega" in archived:
        replay_inputs["body_omega"] = archived["body_omega"]
    api.np.savez_compressed(args.output / "replay-inputs.npz", **replay_inputs)
    native_contacts, traced_contacts = _load_contacts(api, archived), _load_contacts(api, archived)
    if args.env >= native_contacts.valid.shape[1] or args.slot >= native_contacts.valid.shape[0]:
        raise ValueError("Selected source is outside the archive.")
    input_before = _arrays(api, native_contacts)
    if not bool(input_before["valid"][args.env, args.slot]):
        raise ValueError("Selected archived source is not valid.")
    result = {
        "schema": "genesis.dpr-fast-native-fit-diagnostic.v1",
        "complete": False,
        "goal_complete": False,
        "primary_timing": False,
        "spatial_qualification": False,
        "identity": runtime_identity(),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": identities,
        "trace_algorithm_control": proof,
        "native_fit_tolerance": api._PRESSURE_FIT_TOLERANCE,
        "unchanged_controls": {
            "Newton_evaluations": 80,
            "max_Armijo_trials": 40,
            "Armijo_slope": 1e-4,
            "Armijo_absolute_slack": 1e-15,
            "production_source_wrench_relative": 1e-8,
        },
        "scope": "Exact archived pressure inputs and original anchor, no reanchor/rigid advancement, no production tolerance changes. Native-versus-instrumented fit and independent sampled-problem diagnostics; neither status3 nor SciPy termination alone proves physical infeasibility or application stress qualification.",
    }
    result["replay_input_contract"] = {
        "file": "replay-inputs.npz",
        "orientation": "environment, contact slot, optional component; frame/omega are environment, component",
        "fields": sorted(replay_inputs),
        "missing_context": [
            name for name in ("body_omega", "frame_position", "frame_quaternion") if name not in replay_inputs
        ],
        "limit": "Missing historical pose/omega are not manufactured. Pressure replay does not need them; full original-frame elastic-field qualification would require complete mechanical context or an explicitly conditional control.",
    }
    _write(args.output / "result.json", result)
    options = api.gs.options.RigidStressOptions.dpr(
        mesh=args.mesh,
        method="auto",
        quadrature=args.quadrature,
        pressure_profile="hertz",
        radius_model="hertz",
        box_edge_policy="planar",
        pressure_quadrature=args.pressure_quadrature,
        inverse_precision="64",
        history_size=0,
    )
    model = api.StressModel(options)
    surface = api.StressSurface(args.quadrature, model.load_info, angular=args.pressure_quadrature == "angle")
    if surface.info.quad_vertices.shape[0]:
        raise ValueError("This diagnostic is volume-only; no shell trace is inferred.")
    api.np.savez_compressed(
        args.output / "replay-model.npz",
        physical_vertices=api.qd_to_numpy(model.load_info.vertices, transpose=True, copy=True),
        physical_surface_nodes=api.qd_to_numpy(model.load_info.surface_nodes, transpose=True, copy=True).T.copy(),
        elements=api.qd_to_numpy(model.info.elements, transpose=True, copy=True).T.copy(),
    )
    result["stress_options"] = options.model_dump(mode="json")
    result["model_array_orientation"] = "physical_vertices[node,xyz], physical_surface_nodes[face,6], elements[tet,10]"
    n_envs = native_contacts.valid.shape[1]
    native_state, traced_state = model.create_state(n_envs, "max"), model.create_state(n_envs, "max")
    native_workspace = api.create_scatter_workspace(n_envs, args.tasks_per_env, reuse_moments=True)
    traced_workspace = api.create_scatter_workspace(n_envs, args.tasks_per_env, reuse_moments=True)
    trace = api.FitTrace(
        events=api.V_VEC(4, dtype=api.gs.qd_int, shape=(80 * 41, 2)),
        scalars=api.V_VEC(8, dtype=api.gs.qd_float, shape=(80 * 41, 2)),
        coefficient=api.V_VEC(3, dtype=api.gs.qd_float, shape=(80 * 41, 2)),
        candidate=api.V_VEC(3, dtype=api.gs.qd_float, shape=(80 * 41, 2)),
        gradient=api.V_VEC(3, dtype=api.gs.qd_float, shape=(80 * 41, 2)),
        hessian=api.V_MAT(3, 3, dtype=api.gs.qd_float, shape=(80 * 41, 2)),
        step=api.V_VEC(3, dtype=api.gs.qd_float, shape=(80 * 41, 2)),
        nonsingular=api.V(dtype=api.gs.qd_bool, shape=(80 * 41, 2)),
        next_state=api.V_VEC(4, dtype=api.gs.qd_int, shape=(80 * 41, 2)),
        count=api.V(dtype=api.gs.qd_int, shape=(2,)),
        entry=api.V_VEC(4, dtype=api.gs.qd_int, shape=(2,)),
        entry_values=api.V_VEC(4, dtype=api.gs.qd_float, shape=(2,)),
    )
    for field in dataclasses.fields(trace):
        vars(trace)[field.name].fill(0)
    api.kernel_pressure_moments(native_contacts, native_state, model.load_info, surface.info, native_workspace, True)
    native = _arrays(api, native_contacts)
    api.np.savez_compressed(args.output / "native-contacts.npz", **native)
    api._trace_pipeline(
        traced_contacts, traced_state, model.load_info, surface.info, traced_workspace, trace, args.env, args.slot
    )
    traced = _arrays(api, traced_contacts)
    trace_arrays = {
        field.name: api.qd_to_numpy(vars(trace)[field.name], transpose=True, copy=True)
        for field in dataclasses.fields(trace)
    }
    api.np.savez_compressed(args.output / "traced-contacts.npz", **traced)
    api.np.savez_compressed(args.output / "trace.npz", **trace_arrays)
    physical_names = (
        "position",
        "force",
        "normal",
        "friction",
        "radius",
        "radius_status",
        "valid",
        "center",
        "anchor_face",
    )
    physical_unchanged = all(
        input_before[name].tobytes() == native[name].tobytes() == traced[name].tobytes() for name in physical_names
    )
    control_names = ("status", "is_refined", "candidate_count", "evaluations", "weight_sum", "coefficient")
    result["native_vs_traced"] = {
        name: {
            "bitwise_equal": native[name].tobytes() == traced[name].tobytes(),
            "native_selected": native[name][args.env, args.slot].tolist(),
            "traced_selected": traced[name][args.env, args.slot].tolist(),
        }
        for name in control_names
    }
    result["physical_inputs_unchanged"] = physical_unchanged
    result["trace_passes"] = [_trace_summary(api, trace_arrays, phase) for phase in range(2)]
    count = surface.info.face_origin.shape[0] * 3 * surface.info.shape.shape[0]
    coordinates = api.V_VEC(3, dtype=api.gs.qd_float, shape=(count,))
    weights = api.V(dtype=api.gs.qd_float, shape=(count,))
    positions = api.V_VEC(3, dtype=api.gs.qd_float, shape=(count,))
    shapes = api.V_VEC(6, dtype=api.gs.qd_float, shape=(count,))
    faces = api.V(dtype=api.gs.qd_int, shape=(count,))
    api._samples(native_contacts, surface.info, args.env, args.slot, coordinates, weights, positions, shapes, faces)
    samples = {
        name: api.qd_to_numpy(value, transpose=True, copy=True)
        for name, value in (
            ("coordinates", coordinates),
            ("weights", weights),
            ("positions", positions),
            ("shapes", shapes),
            ("faces", faces),
        )
    }
    if not api.np.isfinite(samples["weights"]).all() or (samples["weights"] < 0).any():
        api.np.savez_compressed(args.output / "invalid-native-sampled-problem.npz", **samples)
        result["sample_failure"] = (
            "Native weights contain nonfinite or negative entries; complete raw sample arrays preserved."
        )
        _write(args.output / "result.json", result)
        raise RuntimeError(result["sample_failure"])
    positive = samples["weights"] > 0
    samples = {key: value[positive] for key, value in samples.items()}
    x, w, p = samples["coordinates"], samples["weights"], samples["positions"]
    weight_sum = float(w.sum())
    if not len(w) or not api.np.isfinite(w).all() or not weight_sum > 0:
        raise RuntimeError("Selected native sampled footprint is empty or nonfinite.")
    coefficient = native["coefficient"][args.env, args.slot]
    force, point, radius = (native[name][args.env, args.slot] for name in ("force", "position", "radius"))
    mass = w * api.np.maximum(x @ coefficient, 0)
    normalized = w / weight_sum
    a = coefficient * weight_sum
    profiles = x @ a
    active = profiles > 0
    gradient = x.T @ (normalized * api.np.maximum(profiles, 0)) - api.np.array([1.0, 0.0, 0.0])
    gram = x.T @ (normalized[:, None] * x)
    hessian = x[active].T @ (normalized[active, None] * x[active])
    error_force = abs(float(mass.sum()) - 1) * float(api.np.linalg.norm(force))
    moment = api.np.cross(p - point, mass[:, None] * force).sum(axis=0)
    feasible = api.scipy.optimize.linprog(
        api.np.zeros(len(w)), A_eq=x.T, b_eq=[1.0, 0.0, 0.0], bounds=(0, None), method="highs"
    )
    support = {
        "linprog_success": bool(feasible.success),
        "linprog_status": int(feasible.status),
        "linprog_message": feasible.message,
    }
    if api.np.linalg.matrix_rank(x[:, 1:] - x[:, 1:].mean(axis=0)) == 2:
        hull = api.scipy.spatial.ConvexHull(x[:, 1:])
        witness = hull.equations[int(api.np.argmax(hull.equations[:, -1]))]
        support.update(separating_halfspace=witness.tolist(), target_distance_lower_bound=float(max(0, witness[-1])))
        samples["convex_hull_halfspaces"] = hull.equations
    samples.update(
        coefficient=coefficient,
        normalized_coefficient=a,
        normalized_weights=normalized,
        pressure_masses=mass,
        force=force,
        original_point=point,
        radius=radius,
    )
    api.np.savez_compressed(args.output / "sampled-problem.npz", **samples)
    result["selected_sampled_problem"] = {
        "positive_weight_samples": len(w),
        "positive_active_samples": int(active.sum()),
        "raw_weight_sum": weight_sum,
        "native_weight_sum": float(native["weight_sum"][args.env, args.slot]),
        "weight_effective_samples": float(weight_sum**2 / (w @ w)),
        "pressure_effective_samples": float(mass.sum() ** 2 / (mass @ mass)) if mass @ mass > 0 else None,
        "gradient": gradient.tolist(),
        "gradient_norm": float(api.np.linalg.norm(gradient)),
        "gram_eigenvalues": api.np.linalg.eigvalsh(gram).tolist(),
        "active_hessian_eigenvalues": api.np.linalg.eigvalsh(hessian).tolist(),
        "support": support,
        "diagnostic_force_error_N": error_force,
        "diagnostic_force_error_relative": error_force / float(api.np.linalg.norm(force)),
        "diagnostic_original_point_moment_Nm": moment.tolist(),
        "diagnostic_original_point_moment_error_Nm": float(api.np.linalg.norm(moment)),
        "diagnostic_original_point_moment_error_relative": float(
            api.np.linalg.norm(moment) / (api.np.linalg.norm(force) * radius)
        ),
        "meaning": "Independent reintegration of the native coefficient even when rejected. This is not a production accepted load; it never substitutes stale recorded error fields or certifies the full stress field.",
    }
    result["complete"] = True
    result["passed"] = physical_unchanged and all(
        result["native_vs_traced"][name]["bitwise_equal"] for name in control_names
    )
    result["trace_interpretation"] = (
        "Trace is exact only when native_vs_traced controls all match; status/iteration/coefficient drift remains explicit and cannot certify the original stopping sequence. Diagnostic CPU LP/eigens are computed only within this Slurm allocation."
    )
    result["input_files_unchanged"] = identities["archive"] == _identity(args.archive) and identities[
        "mesh"
    ] == _identity(args.mesh)
    _write(args.output / "result.json", result)
    if not result["passed"] or not result["input_files_unchanged"]:
        raise RuntimeError(
            "Replay/telemetry control failed; preserve this diagnostic without inferring an exact original trace."
        )


if __name__ == "__main__":
    main()
