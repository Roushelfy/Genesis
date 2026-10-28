"""Audit complete native volume-DPR histories using exact-input native references.

The sole Franka scene supplies scripted or FP32-policy dynamics and stress
feedback. A separate full-output replay of its selected production route and
an explicit fixed angular/q128 reference share its original elastic model.
Neither replay advances dynamics, associates sources again, or supplies policy
feedback. Scientific execution requires Slurm rtx-mid/image1_26.
"""

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from .artifacts import file_id, runtime_identity
from .check_planar_native_trajectory import _gamma, _policy_state, _policy_step, _wrench_roundoff
from .dpr_fast_fit_diagnostic import _write

PHYSICAL = (
    "position",
    "force",
    "normal",
    "friction",
    "radius",
    "radius_status",
    "valid",
    "center",
    "anchor_face",
    "frame_position",
    "frame_quaternion",
    "omega",
    "hertz_profile",
)
REGIONS = ("global", "loaded", "remote")
REGION_METRICS = (
    "tensor_error_RMS_Pa",
    "reference_tensor_RMS_Pa",
    "tensor_relative_L2",
    "reference_peak_Pa",
    "candidate_peak_Pa",
    "peak_signed_error_Pa",
    "peak_absolute_error_Pa",
    "peak_bidirectional_relative",
    "peak_underestimate_relative",
    "normalization_floor_Pa",
)
RUNTIME_FLAGS = (
    "valid",
    "elastic_valid",
    "source_valid",
    "finite_valid",
    "physical_valid",
    "errno_valid",
    "source_point_valid",
    "assembled_origin_accounting_valid",
    "strict_origin_diagnostic_passed",
)
COUNTERS = (
    "attempted_sources",
    "checker_calls",
    "stage0_calls",
    "stage1_calls",
    "stage2_calls",
    "refined_sources",
    "rejected_sources",
    "reserved_index7",
    "stage0_initial_affine_sources",
)


def _api():
    import numpy as np
    import quadrants as qd
    import torch

    import genesis as gs
    from examples.rigid.franka_egg_stress import FrankaEgg
    from examples.speed_benchmark.rigid_stress import create_inference_policy
    from genesis.engine.solvers.rigid.stress.contact import _PRESSURE_FIT_TOLERANCE, StressContactState, create_contacts
    from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
    from genesis.engine.solvers.rigid.stress.scatter import (
        create_scatter_workspace,
        kernel_pressure_moments,
        kernel_scatter_faces,
    )
    from genesis.engine.solvers.rigid.stress.surface import StressSurface
    from genesis.utils.array_class import V_VEC, V
    from genesis.utils.misc import qd_to_numpy, tensor_to_array

    from . import check_planar_native_trajectory, dpr_fast_rule_qualification as rules
    from .check_native_atlas import cpu_operator

    copy_api = check_planar_native_trajectory._api()

    @qd.kernel
    def kernel_copy_context(
        source: StressContactState, target: StressContactState, source_omega: qd.Tensor, target_omega: qd.Tensor
    ):
        for i_c, i_b in qd.ndrange(source.valid.shape[0], source.valid.shape[1]):
            target.position[i_c, i_b] = source.position[i_c, i_b]
            target.force[i_c, i_b] = source.force[i_c, i_b]
            target.normal[i_c, i_b] = source.normal[i_c, i_b]
            target.friction[i_c, i_b] = source.friction[i_c, i_b]
            target.radius[i_c, i_b] = source.radius[i_c, i_b]
            target.radius_status[i_c, i_b] = source.radius_status[i_c, i_b]
            target.valid[i_c, i_b] = source.valid[i_c, i_b]
            target.center[i_c, i_b] = source.center[i_c, i_b]
            target.anchor_face[i_c, i_b] = source.anchor_face[i_c, i_b]
        for i_b in range(source_omega.shape[0]):
            target.frame_position[i_b] = source.frame_position[i_b]
            target.frame_quaternion[i_b] = source.frame_quaternion[i_b]
            target_omega[i_b] = source_omega[i_b]
        for _ in range(1):
            target.hertz_profile[None] = source.hertz_profile[None]

    return SimpleNamespace(**locals())


def _same(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _snapshot(api, state, prefix=""):
    result = {}
    for field in dataclasses.fields(state):
        value = vars(state)[field.name]
        if dataclasses.is_dataclass(value):
            result.update(_snapshot(api, value, prefix + field.name + "__"))
        else:
            result[prefix + field.name] = api.qd_to_numpy(value, transpose=True, copy=True)
    return result


def _copy_fields(api, source, target):
    for field in dataclasses.fields(source):
        original, copied = vars(source)[field.name], vars(target)[field.name]
        if dataclasses.is_dataclass(original):
            _copy_fields(api, original, copied)
        else:
            if original.shape != copied.shape:
                raise RuntimeError("A fitted native snapshot must preserve all tensor capacities.")
            rank = len(original.shape)
            if rank == 0:
                api.copy_api.kernel_copy_scalar(original, copied)
            elif rank == 1:
                api.copy_api.kernel_copy_vector(original, copied)
            elif rank == 2:
                api.copy_api.kernel_copy_matrix(original, copied)
            elif rank == 3:
                api.copy_api.kernel_copy_cube(original, copied)
            else:
                raise RuntimeError("Unverified native tensor rank in exact-snapshot copy.")


def _physical(api, contacts, omega):
    result = {
        name: api.qd_to_numpy(vars(contacts)[name], transpose=True, copy=True) for name in PHYSICAL if name != "omega"
    }
    result["omega"] = api.qd_to_numpy(omega, transpose=True, copy=True)
    return result


def _outputs(api, state, full):
    names = (
        "force",
        "rhs",
        "displacement",
        "peak",
        "step_peak",
        "valid",
        "step_valid",
        "residual_norm_squared",
        "rhs_norm_squared",
        "invalid_steps",
        "iterations",
        "corrections",
        "fallbacks",
    )
    result = {name: api.qd_to_numpy(vars(state)[name], transpose=True, copy=True) for name in names}
    if full:
        result["stress_tensor"] = api.qd_to_numpy(state.stress_tensor, transpose=True, copy=True)
        result["von_mises"] = api.qd_to_numpy(state.von_mises, transpose=True, copy=True)[..., 0]
        result["tensor_full9"] = api.rules.tensor_matrix(result["stress_tensor"])
    return result


@dataclasses.dataclass
class ReplayRoute:
    api: object
    entry: object
    epsilon: float
    reference: bool
    capacity: int
    options: object = dataclasses.field(init=False)
    surface: object = dataclasses.field(init=False)
    contacts: object = dataclasses.field(init=False)
    state: object = dataclasses.field(init=False)
    scatter: object = dataclasses.field(init=False)
    omega: object = dataclasses.field(init=False)
    errno: object = dataclasses.field(init=False)

    def __post_init__(self):
        api, entry = self.api, self.entry
        update = {"output_mode": "full", "history_size": 0}
        if self.reference:
            update.update(pressure_quadrature="angle", quadrature=128)
        self.options = entry.options.model_copy(update=update)
        adaptive = self.options.pressure_quadrature == "adaptive"
        self.surface = api.StressSurface(128, entry.model.load_info, angular=True) if self.reference else entry.surface
        n_contacts, n_envs = entry.contacts.valid.shape
        self.contacts = api.create_contacts(
            n_contacts, n_envs, entry.options.contact_radius, True, True, True, adaptive
        )
        self.state = entry.model.create_state(n_envs, "full")
        self.scatter = api.create_scatter_workspace(n_envs, self.capacity, reuse_moments=True, adaptive=adaptive)
        self.omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(n_envs,))
        self.errno = api.V(dtype=api.gs.qd_int, shape=(n_envs,))
        self.errno.fill(0)

    def recover(self):
        api, entry = self.api, self.entry
        api.kernel_copy_context(entry.contacts, self.contacts, entry.omega, self.omega)
        api.kernel_begin_step(self.state)
        self.errno.fill(0)
        if self.reference:
            api.rules.kernel_prepare_frozen(self.epsilon, self.contacts, self.surface.info)
            api.kernel_pressure_moments(
                self.contacts,
                self.state,
                entry.model.load_info,
                self.surface.info,
                self.scatter,
                self.options.cached_face_bounds,
            )
        else:
            # The full mirror consumes the actual accepted fit and capacity route.
            # Re-fitting could select a different adaptive stage after roundoff.
            _copy_fields(api, entry.contacts, self.contacts)
            _copy_fields(api, entry.scatter, self.scatter)
            self.state.force.fill(0.0)
            self.scatter.wrench.fill(0.0)
        api.kernel_scatter_faces(
            self.contacts,
            self.state,
            entry.model.load_info,
            self.surface.info,
            self.scatter,
            self.options.cached_face_bounds,
            True,
        )
        entry.model.recover(self.omega, self.state, self.options, history=None, surface_load=True)
        api.kernel_accept(self.state, self.contacts, self.errno)
        return _outputs(api, self.state, True)


@dataclasses.dataclass
class RawWitness:
    api: object
    route: object
    contacts: object = dataclasses.field(init=False)
    scatter: object = dataclasses.field(init=False)
    state: object = dataclasses.field(init=False)

    def __post_init__(self):
        api, route = self.api, self.route
        n_contacts, n_envs = route.contacts.valid.shape
        self.contacts = api.create_contacts(
            n_contacts,
            n_envs,
            route.options.contact_radius,
            True,
            True,
            True,
            route.options.pressure_quadrature == "adaptive",
        )
        self.scatter = api.create_scatter_workspace(
            n_envs,
            route.scatter.tasks.shape[0] // n_envs,
            reuse_moments=True,
            adaptive=route.options.pressure_quadrature == "adaptive",
        )
        self.state = route.entry.model.create_state(n_envs, "max")

    def run(self, physical, actual, environments, vertices, output):
        api, route, np = self.api, self.route, self.api.np
        rows, raw_sources = [], []
        reconstructed = np.zeros_like(actual["force"])
        counts = np.zeros(len(reconstructed), dtype=api.gs.np_int)
        mass = np.zeros(len(reconstructed))
        passed = True
        for env in environments:
            loaded = physical["valid"][env] & (np.linalg.norm(physical["force"][env], axis=1) > 1e-30)
            for slot in np.flatnonzero(loaded):
                _copy_fields(api, route.contacts, self.contacts)
                _copy_fields(api, route.scatter, self.scatter)
                self.state.force.fill(0.0)
                self.scatter.wrench.fill(0.0)
                api.copy_api.kernel_mask_source(self.contacts, int(env), int(slot))
                api.kernel_scatter_faces(
                    self.contacts,
                    self.state,
                    route.entry.model.load_info,
                    route.surface.info,
                    self.scatter,
                    route.options.cached_face_bounds,
                    True,
                )
                raw = api.qd_to_numpy(self.state.force, transpose=True, copy=True)[env]
                force, point, radius = (physical[name][env, slot] for name in ("force", "position", "radius"))
                force_error = np.linalg.norm(raw.sum(axis=0) - force)
                moment_error = np.linalg.norm(np.cross(vertices - point, raw).sum(axis=0))
                magnitude = np.linalg.norm(force)
                valid = bool(
                    np.isfinite(raw).all()
                    and force_error <= 1e-8 * magnitude
                    and moment_error <= 1e-8 * magnitude * radius
                )
                passed &= valid
                reconstructed[env] += raw
                counts[env] += 1
                mass[env] += np.abs(force).sum()
                raw_sources.append(raw)
                rows.append(
                    {
                        "environment": int(env),
                        "slot": int(slot),
                        "passed": valid,
                        "force_error_N": float(force_error),
                        "force_error_relative": float(force_error / magnitude),
                        "original_point_moment_error_Nm": float(moment_error),
                        "original_point_moment_error_relative": float(moment_error / (magnitude * radius)),
                    }
                )
        # Same per-face nodal contributions and preserved task/capacity route;
        # only face/source accumulation order changes, not quadrature or fitting.
        corners = max(1, int(counts.max(initial=0))) * route.surface.info.face_origin.shape[0] * 6
        unit = np.finfo(api.gs.np_float).eps / 2
        positive_mass = 2 * (1 + np.sqrt(3) * 1e-8) * mass
        bound = (_gamma(unit, 32 * corners + 32) + _gamma(unit, int(counts.max(initial=0)) + 4)) * positive_mass
        difference = np.abs(reconstructed - actual["force"]).sum(axis=(1, 2))
        reconstruction_passed = bool((difference[environments] <= bound[environments]).all())
        passed &= reconstruction_passed
        np.savez_compressed(
            output.with_suffix(".npz"),
            raw_source=np.stack(raw_sources) if raw_sources else np.empty((0, len(vertices), 3)),
            source_environment=np.array([row["environment"] for row in rows], dtype=api.gs.np_int),
            source_slot=np.array([row["slot"] for row in rows], dtype=api.gs.np_int),
            reconstructed=reconstructed,
            original=actual["force"],
            difference=difference,
            reconstruction_roundoff_bound=bound,
            physical_vertices=vertices,
        )
        result = {
            "passed": passed,
            "sources": rows,
            "raw_reconstruction_passed": reconstruction_passed,
            "scope": "Exact native nested admission/contact/scatter snapshot; masked scatter only, no fit, anchor, radius change or rigid step. Source point checks retain1e-8 with no added floor. Reconstruction gamma bound covers accumulation of the same fitted nodal contributions, not fitting/geometry error.",
        }
        _write(output.with_suffix(".json"), result)
        return result


def _runtime(api, outputs, contacts, errno, physical, vertices, operator, options, n_faces, order):
    np = api.np
    n_envs = len(outputs["valid"])
    q = outputs["displacement"].reshape((n_envs, -1)).T
    rhs = outputs["rhs"].reshape((n_envs, -1)).T
    residual = options.young * (operator.matrix @ q) - rhs
    residual_norm = np.linalg.norm(residual[operator.free], axis=0)
    rhs_norm = np.linalg.norm(rhs[operator.free], axis=0)
    elastic = (
        outputs["valid"]
        & outputs["step_valid"]
        & (residual_norm <= np.maximum(options.absolute_tolerance, options.tolerance * rhs_norm))
    )
    physical_source_finite = np.isfinite(physical["friction"]) & np.isfinite(physical["radius"])
    for name in ("position", "force", "normal", "center"):
        physical_source_finite &= np.isfinite(physical[name]).all(axis=-1)
    loaded = physical["valid"] & (np.linalg.norm(physical["force"], axis=-1) > 1e-30)
    physical_source_finite &= (~loaded) | (physical["radius"] > 0)
    physical_finite = ((~physical["valid"]) | physical_source_finite).all(axis=1)
    for name in ("omega", "frame_position", "frame_quaternion"):
        physical_finite &= np.isfinite(physical[name]).all(axis=1)
    physical_finite &= bool(physical["hertz_profile"])
    state_finite = np.ones(n_envs, dtype=bool)
    for name, value in outputs.items():
        state_finite &= np.isfinite(value).reshape((n_envs, -1)).all(axis=1)
    status = api.qd_to_numpy(contacts.status, transpose=True, copy=True)
    error_codes = api.qd_to_numpy(errno, transpose=True, copy=True)
    force_error = api.qd_to_numpy(contacts.force_error, transpose=True, copy=True)
    moment_error = api.qd_to_numpy(contacts.moment_error, transpose=True, copy=True)
    coefficient = api.qd_to_numpy(contacts.coefficient, transpose=True, copy=True)
    magnitude = np.linalg.norm(physical["force"], axis=-1)
    source_point = (~loaded) | (
        np.isfinite(force_error)
        & np.isfinite(moment_error)
        & (force_error >= 0)
        & (moment_error >= 0)
        & (force_error <= 1e-8 * magnitude)
        & (moment_error <= 1e-8 * magnitude * physical["radius"])
    )
    source = ((status == 0) | ~physical["valid"]).all(axis=1)
    source &= (np.isfinite(coefficient).all(axis=-1) | ~loaded).all(axis=1)
    forces, points = (
        np.where(loaded[..., None], physical["force"], 0.0),
        np.where(loaded[..., None], physical["position"], 0.0),
    )
    total_force, expected_force = outputs["force"].sum(axis=1), forces.sum(axis=1)
    total_moment = np.cross(vertices[None], outputs["force"]).sum(axis=1)
    expected_moment = np.cross(points, forces).sum(axis=1)
    absolute_force_error = np.linalg.norm(total_force - expected_force, axis=1)
    absolute_moment_error = np.linalg.norm(total_moment - expected_moment, axis=1)
    scale = np.linalg.norm(forces, axis=-1).sum(axis=1)
    moment_scale = (np.linalg.norm(forces, axis=-1) * np.where(loaded, physical["radius"], 0.0)).sum(axis=1)
    strict_origin = (~loaded.any(axis=1)) | (
        (absolute_force_error <= 1e-8 * scale) & (absolute_moment_error <= 1e-8 * moment_scale)
    )
    reported_force, reported_moment = np.where(loaded, force_error, 0), np.where(loaded, moment_error, 0)
    nodal = np.concatenate((outputs["force"], np.zeros_like(outputs["force"])), axis=2).transpose((1, 2, 0))
    # P2 barycentric shapes have sum(abs(N))<=5/3 on the simplex; the old
    # post-fit mapping bound2 leaves room for the admitted coordinate margin.
    # This is an accumulation bound after fitted weights, not a whole-pipeline bound.
    contributions = max(1, int(loaded.sum(axis=1).max(initial=0))) * n_faces * 18 * order * order
    round_force, round_moment = _wrench_roundoff(
        api, nodal, forces, points, vertices, reported_force, (contributions, 0.0)
    )
    force_bound = reported_force.sum(axis=1) + round_force
    moment_bound = (np.linalg.norm(points, axis=-1) * reported_force + reported_moment).sum(axis=1) + round_moment
    accounting = (absolute_force_error <= force_bound) & (absolute_moment_error <= moment_bound)
    unloaded_exact = np.count_nonzero(outputs["force"][~loaded.any(axis=1)]) == 0
    valid = (
        elastic & source & state_finite & physical_finite & (error_codes == 0) & source_point.all(axis=1) & accounting
    )
    return {
        "valid": valid,
        "elastic_valid": elastic,
        "source_valid": source,
        "finite_valid": state_finite,
        "physical_valid": physical_finite,
        "errno_valid": error_codes == 0,
        "source_point_valid": source_point.all(axis=1),
        "assembled_origin_accounting_valid": accounting,
        "strict_origin_diagnostic_passed": strict_origin,
        "elastic_residual": residual_norm,
        "original_elastic_residual_budget": np.maximum(options.absolute_tolerance, options.tolerance * rhs_norm),
        "source_force_error_N": force_error,
        "source_original_point_moment_error_Nm": moment_error,
        "source_relative_force_error": reported_force / np.maximum(magnitude, 1e-30),
        "source_relative_point_moment_error": reported_moment / np.maximum(magnitude * physical["radius"], 1e-30),
        "assembled_force_error_N": absolute_force_error,
        "assembled_origin_moment_error_Nm": absolute_moment_error,
        "assembled_force_bound_N": force_bound,
        "assembled_origin_moment_bound_Nm": moment_bound,
        "strict_origin_force_relative": absolute_force_error / np.maximum(scale, 1e-30),
        "strict_origin_moment_relative": absolute_moment_error / np.maximum(moment_scale, 1e-30),
        "unloaded_raw_force_exact_zero": bool(unloaded_exact),
        "status": status,
        "errno": error_codes,
    }


def _masks(api, physical, sample_positions):
    result = []
    for env in range(len(physical["valid"])):
        ids = api.np.flatnonzero(physical["valid"][env] & (api.np.linalg.norm(physical["force"][env], axis=-1) > 1e-30))
        loaded = api.np.zeros(sample_positions.shape[:2], dtype=bool)
        for slot in ids:
            loaded |= (
                api.np.linalg.norm(sample_positions - physical["center"][env, slot], axis=-1)
                <= 2 * physical["radius"][env, slot]
            )
        result.append({"global": api.np.ones_like(loaded), "loaded": loaded, "remote": ~loaded})
    return result


def _legacy(api, actual, expected):
    return {
        name: api.rules.legacy_error(actual[key], expected[key])
        for name, key in (
            ("generalized_load", "force"),
            ("rhs", "rhs"),
            ("displacement", "displacement"),
            ("stored6_tensor", "stress_tensor"),
            ("peak", "peak"),
        )
    }


def _rigid_snapshot(api, workload):
    links = workload.scene.rigid_solver.dyn_state.links
    result = {
        name: api.qd_to_numpy(value, transpose=True, copy=True)
        for name, value in (("pos", links.pos), ("quat", links.quat), ("cd_ang", links.cd_ang))
    }
    result["tick"] = workload.tick
    return result


def _saved_frame(api, directory, physical, outputs, runtime, contacts, admission):
    directory.mkdir()
    api.np.savez_compressed(directory / "physical-inputs.npz", **physical)
    for label, values in outputs.items():
        api.np.savez_compressed(directory / (label + "-fields.npz"), **values)
        api.np.savez_compressed(directory / (label + "-runtime.npz"), **runtime[label])
    api.np.savez_compressed(directory / "production-contacts.npz", **contacts)
    if admission:
        api.np.savez_compressed(directory / "production-admission.npz", **admission)


def _lifecycle(api, workload, entry, policy):
    checkpoint = workload.scene.get_state()
    before = _snapshot(api, entry.state)
    admission_before = _snapshot(api, entry.contacts.admission)
    selected = 1
    others = api.np.arange(workload.n_envs) != selected
    rows = []
    for name in ("partial_reset", "partial_checkpoint_restore"):
        if name == "partial_reset":
            workload.reset(api.np.array([selected], dtype=api.gs.np_int))
        else:
            workload.scene.reset(state=checkpoint, envs_idx=[selected])
        after = _snapshot(api, entry.state)
        unchanged = all(
            _same(before[key][others], value[others])
            for key, value in after.items()
            if value.ndim
            and value.shape[0] == workload.n_envs
            and key
            in (
                "force",
                "rhs",
                "displacement",
                "peak",
                "step_peak",
                "valid",
                "step_valid",
                "stress_tensor",
                "von_mises",
            )
        )
        clear = bool(api.np.count_nonzero(after["displacement"][selected]) == 0 and after["peak"][selected] == 0)
        derived = True
        if entry.output_mode == "full":
            derived = bool(
                api.np.isnan(after["stress_tensor"][selected]).all()
                and api.np.isnan(after["von_mises"][selected]).all()
            )
        admission_after = _snapshot(api, entry.contacts.admission)
        admission_clear = True
        if entry.options.pressure_quadrature == "adaptive":
            admission_clear = bool(
                api.np.count_nonzero(admission_after["counters"][selected]) == 0
                and api.np.count_nonzero(admission_after["reason"][selected]) == 0
                and api.np.count_nonzero(admission_after["attempts"][selected]) == 0
            )
            unchanged &= all(
                _same(value[others], admission_after[key][others]) for key, value in admission_before.items()
            )
        rows.append(
            {
                "name": name,
                "passed": bool(unchanged and clear and derived and admission_clear),
                "other_environments_exact": bool(unchanged),
                "selected_displacement_peak_clear": clear,
                "selected_derived_output_invalidated": derived,
                "selected_admission_clear": admission_clear,
            }
        )
    if policy is None:
        workload.step()
    else:
        _policy_step(api, workload, policy)
    workload.scene.rigid_solver.check_errno()
    output = _outputs(api, entry.state, entry.output_mode == "full")
    fresh = bool(output["step_valid"].all() and output["valid"].all() and api.np.isfinite(output["step_peak"]).all())
    if entry.output_mode == "full":
        fresh &= bool(api.np.isfinite(output["stress_tensor"]).all())
    rows.append({"name": "fresh_observation_after_restore", "passed": fresh})
    return {
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "scope": "Additional scene reset/checkpoint invalidation and fresh observation after the audited history; not counted in history steps or rates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--envs", type=int, choices=(4, 64), required=True)
    parser.add_argument("--steps", type=int, default=730)
    parser.add_argument("--output-mode", choices=("max", "full"), required=True)
    parser.add_argument("--pressure-quadrature", choices=("adaptive", "edge", "angle"), default="adaptive")
    parser.add_argument("--quadrature", type=int, choices=(16, 128), default=16)
    parser.add_argument("--egg-geometry", choices=("mesh64", "urdf"), default="mesh64")
    parser.add_argument("--policy-inference", action="store_true")
    parser.add_argument("--tasks-per-env", type=int, default=32)
    parser.add_argument("--reference-tasks-per-env", type=int, default=32)
    parser.add_argument("--keyframe-every", type=int, default=100)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific execution requires Slurm rtx-mid/image1_26.")
    if min(args.steps, args.tasks_per_env, args.reference_tasks_per_env, args.keyframe_every) < 1:
        parser.error("Use positive steps, capacities and keyframe interval.")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output.")
    budget = json.loads(args.budgets.read_text())
    gates = {
        "tensor_relative_L2_max": 0.05,
        "sampled_peak_bidirectional_relative_max": 0.05,
        "sampled_peak_underestimate_relative_max": 0.02,
    }
    if (
        budget["schema"] != "genesis.automatic-dpr-application-budgets.v1"
        or any(budget["application_gates"][name] != gates for name in REGIONS)
        or budget["metrics"]["floor_values"]
        != {"absolute_stress_floor_Pa": 1.0, "regional_floor_fraction_of_reference_global_peak": 0.01}
    ):
        parser.error("Use the unchanged predeclared application budgets/floors.")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "genesis.dpr-fast-native-trajectory.v1",
        "complete": False,
        "passed": False,
        "full_trajectory_qualified": False,
        "bounded_history_qualified": False,
        "goal_complete": False,
        "primary_timing": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": file_id(args.conditions),
        "budgets": file_id(args.budgets),
        "worker": file_id(Path(__file__)),
        "pid": os.getpid(),
        "command": sys.argv,
        "rows": [],
        "witnesses": [],
        "keyframes": [],
        "warmup_steps": 0,
        "scope": "One production scene drives every history step and policy observation. Same-route full mirror and fixed angular128 native reference use exact copied sources/anchor/center/omega/pose and the same K/material/gauge, never association/rigid advance or policy feedback. Intrusive complete-history accuracy, not primary throughput or continuum contact qualification.",
    }
    _write(args.output / "result.json", result)
    api = _api()
    api.gs.init(backend=api.gs.cuda, precision="64", seed=510000, logging_level="warning")
    if Path(api.gs.__file__).resolve().parent.parent != args.git_source.resolve():
        raise RuntimeError("Imported Genesis is not the declared frozen engine.")
    source_files = [
        args.git_source / "genesis/engine/solvers/rigid/stress" / name
        for name in (
            "contact.py",
            "scatter.py",
            "adaptive_pressure.py",
            "surface.py",
            "model.py",
            "solve.py",
            "recovery.py",
        )
    ]
    source_files.extend(
        [
            Path(api.rules.__file__),
            Path(__file__),
            Path(_wrench_roundoff.__code__.co_filename),
            Path(api.cpu_operator.__code__.co_filename),
            args.git_source / "examples/rigid/franka_egg_stress.py",
            args.git_source / "examples/rigid/egg_stress_controller.py",
            args.git_source / "examples/speed_benchmark/rigid_stress.py",
        ]
    )
    source_before = [file_id(path) for path in source_files]
    workload = api.FrankaEgg(
        args.envs,
        conditions=args.conditions,
        output_mode=args.output_mode,
        egg_geometry=args.egg_geometry,
        stress_backend="volume",
        box_edge_policy="planar",
        quadrature=args.quadrature,
        pressure_quadrature=args.pressure_quadrature,
        scatter_tasks_per_env=args.tasks_per_env,
        inverse_precision="64",
        history_size=0,
        timestep=0.01,
        substeps=1,
    )
    solver = workload.scene.rigid_solver
    recovery, entry = solver.stress_recovery, solver.stress_recovery.links[0]
    if (
        entry.options.backend != "volume"
        or entry.planar is not None
        or entry.options.contact_model != "pressure"
        or entry.options.load_transfer != "patch"
        or entry.options.radius_model != "hertz"
        or entry.options.pressure_profile != "hertz"
        or not entry.reuse_contact_moments
        or not entry.options.cooperative_pressure
        or not entry.options.cooperative_scatter
        or entry.options.tolerance != 1e-8
        or entry.options.absolute_tolerance != 1e-11
        or entry.automatic_radius is None
        or api._PRESSURE_FIT_TOLERANCE != 1e-9
    ):
        raise RuntimeError(
            "Require original volume Hertz DPR, moment reuse, native cooperative route and unchanged K gates."
        )
    mirror = ReplayRoute(api, entry, recovery.source_epsilon, False, args.tasks_per_env)
    reference = ReplayRoute(api, entry, recovery.source_epsilon, True, args.reference_tasks_per_env)
    operator = api.cpu_operator(entry.model.info)
    vertices = api.qd_to_numpy(entry.model.info.vertices, transpose=True, copy=True)
    elements = api.qd_to_numpy(entry.model.info.elements, transpose=False, copy=True)
    volumes = api.qd_to_numpy(entry.model.info.volumes, transpose=True, copy=True)
    sample_positions = vertices[elements[:, :4]]
    weights = api.np.broadcast_to(volumes[:, None] / 4, sample_positions.shape[:2]).copy()
    if not api.np.isfinite(sample_positions).all() or not api.np.isfinite(weights).all() or not (weights > 0).all():
        raise ArithmeticError("Reference output samples and physical volume weights must be finite and positive.")
    api.np.savez_compressed(
        args.output / "native-model.npz",
        **_snapshot(api, entry.model.info),
        sample_positions=sample_positions,
        sample_weights=weights,
    )
    result.update(
        identity=runtime_identity(),
        source_revision=subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_files=source_before,
        options=entry.options.model_dump(mode="json"),
        selected_method=entry.model.selected_method,
        actual_fused_pipeline=entry.fused_pipeline,
        actual_contact_moment_reuse=entry.reuse_contact_moments,
        actual_automatic_radius=entry.automatic_radius is not None,
        actual_output_mode=entry.output_mode,
        pressure_fit_normalized_gradient_stop=api._PRESSURE_FIT_TOLERANCE,
        model=file_id(args.output / "native-model.npz"),
        reference_options=reference.options.model_dump(mode="json"),
        reference_fixed_angular=bool(api.qd_to_numpy(reference.surface.info.angular_quadrature, transpose=True)),
        reference_quadrature=reference.surface.info.gauss.shape[0],
        route_counter_columns=COUNTERS,
        full_mirror_mode="Exact actual fitted contact/admission/task/moment snapshot; same native scatter and K/full recovery, no re-fit.",
        raw_source_witness_scope="First loaded and first recontact observation of each environment plus every disputed strict-origin frame; every history step still retains the original native per-source1e-8 gate and independent K residual.",
        reference_stability_scope="Fixed angular128 same-model reference each frame; bounded q128/q256 stability belongs to the separately frozen rule corpus, not a per-frame continuum certificate.",
    )
    result["cuda_graph_environment"] = {
        "QD_GRAPH": os.environ.get("QD_GRAPH"),
        "scope": "Process environment receipt plus captured helper pre-import export; not a direct imported Quadrants constant readback.",
    }
    policy = None
    policy_reference = None
    actions = []
    observations = []
    if args.policy_inference:
        policy = api.create_inference_policy()
        policy_reference = copy.deepcopy(policy)
        policy = policy.to(dtype=api.torch.float32)
        policy_values, policy_hash = _policy_state(api, policy)
        api.np.savez_compressed(args.output / "actual-policy-weights.npz", **policy_values)
        result["policy"] = {
            "seed": 99173,
            "layers": [26, 128, 128, 7],
            "construction_dtype": str(api.gs.tc_float),
            "inference_dtype": "float32",
            "weights_sha256": policy_hash,
            "weights_file": file_id(args.output / "actual-policy-weights.npz"),
            "residual_scale_rad": 1e-4,
            "FP64_control_budget_rad": 1e-9,
            "FP64_control_maximum_error_rad": 0.0,
            "feedback": "Only actual production public stress getter/1e6; no reference feedback.",
        }
    history = []
    loaded_records = []
    worst = {}
    raw_witnesses = {}
    step_error = None
    previously_loaded = api.np.zeros(args.envs, dtype=bool)
    load_seen = previously_loaded.copy()
    release_seen = previously_loaded.copy()
    recontact_seen = previously_loaded.copy()
    reset_counts = api.np.zeros(args.envs, dtype=api.gs.np_int)
    try:
        for step in range(args.steps):
            tick = workload.tick
            result["attempted_steps"] = step + 1
            reset = (tick > workload.delays) & ((tick - workload.delays) % 600 == 0)
            before_counts = api.qd_to_numpy(entry.contacts.admission.counters, transpose=True, copy=True)
            if policy is None:
                workload.step()
            else:
                with api.torch.inference_mode():
                    observation = workload.observation()
                    if not api.torch.isfinite(observation).all():
                        raise ArithmeticError("Invalid production observation must not enter policy inference.")
                    action = policy(observation.to(dtype=api.torch.float32))
                    if not api.torch.isfinite(action).all():
                        raise ArithmeticError("Invalid policy action must not advance dynamics.")
                    if step % 100 == 0:
                        expected = policy_reference(observation)
                        control = 1e-4 * (action.to(dtype=api.torch.float64) - expected).abs().max().item()
                        if not control < 1e-9:
                            raise ArithmeticError("FP32 policy exceeds the original1e-9rad action-control budget.")
                        result["policy"]["FP64_control_maximum_error_rad"] = max(
                            result["policy"]["FP64_control_maximum_error_rad"], control
                        )
                    observations.append(api.tensor_to_array(observation).copy())
                    actions.append(api.tensor_to_array(action).copy())
                    workload.step(action)
            solver.check_errno()
            physical = _physical(api, entry.contacts, entry.omega)
            production = _outputs(api, entry.state, args.output_mode == "full")
            production_state = _snapshot(api, entry.state)
            production_contacts = _snapshot(api, entry.contacts)
            rigid_before = _rigid_snapshot(api, workload)
            candidate, high = mirror.recover(), reference.recover()
            rigid_after = _rigid_snapshot(api, workload)
            state_after = _snapshot(api, entry.state)
            contacts_after = _snapshot(api, entry.contacts)
            physical_after = _physical(api, entry.contacts, entry.omega)
            copied_physical = [_physical(api, route.contacts, route.omega) for route in (mirror, reference)]
            preserved = all(_same(value, state_after[name]) for name, value in production_state.items())
            preserved &= all(_same(value, contacts_after[name]) for name, value in production_contacts.items())
            preserved &= all(_same(value, physical_after[name]) for name, value in physical.items())
            preserved &= all(
                _same(value, copied[name]) for copied in copied_physical for name, value in physical.items()
            )
            preserved &= rigid_before["tick"] == rigid_after["tick"] and all(
                _same(rigid_before[name], rigid_after[name]) for name in ("pos", "quat", "cd_ang")
            )
            routes = (
                ("production", entry.state, entry.contacts, solver._errno, production, entry.surface, entry.options),
                ("mirror", mirror.state, mirror.contacts, mirror.errno, candidate, mirror.surface, mirror.options),
                (
                    "reference128",
                    reference.state,
                    reference.contacts,
                    reference.errno,
                    high,
                    reference.surface,
                    reference.options,
                ),
            )
            runtime = {}
            for label, state, contacts, errno, output, surface, options in routes:
                order = 128 if options.pressure_quadrature == "adaptive" else options.quadrature
                runtime[label] = _runtime(
                    api,
                    output,
                    contacts,
                    errno,
                    physical,
                    vertices,
                    operator,
                    options,
                    surface.info.face_origin.shape[0],
                    order,
                )
            numeric = all(check["valid"].all() and check["unloaded_raw_force_exact_zero"] for check in runtime.values())
            if not numeric:
                directory = args.output / f"failed-native-frame-{step:06d}"
                _saved_frame(
                    api,
                    directory,
                    physical,
                    {"production": production, "mirror": candidate, "reference128": high},
                    runtime,
                    production_contacts,
                    {},
                )
                result["rows"].append(
                    {
                        "step": step,
                        "attempted_tick": tick,
                        "passed": False,
                        "numerical_passed": False,
                        "application_regions_evaluated": False,
                        "application_regions_passed": False,
                        "legacy_parity_passed": False,
                        "legacy_spatial_per_frame_5mean_10worst_5peak_diagnostic": False,
                        "invalid_environments": {
                            label: api.np.flatnonzero(~check["valid"]).tolist() for label, check in runtime.items()
                        },
                        "raw_failure": str(directory),
                    }
                )
                raise ArithmeticError(
                    "Native source/elastic/finite/errno failure: application fields are not compared or qualified."
                )
            loaded_contacts = physical["valid"] & (api.np.linalg.norm(physical["force"], axis=-1) > 1e-30)
            loaded = loaded_contacts.any(axis=1)
            released = previously_loaded & ~loaded
            recontact = release_seen & loaded & ~previously_loaded
            newly_loaded = loaded & ~load_seen
            first_recontact = recontact & ~recontact_seen
            first_loaded = bool(newly_loaded.any())
            load_seen |= loaded
            release_seen |= released
            recontact_seen |= recontact
            reset_counts += reset
            masks = _masks(api, physical, sample_positions)
            region_rows = api.rules.region_errors(candidate, high, weights, masks)
            spatial = all(region["passed"] is not False for row in region_rows for region in row.values())
            legacy = _legacy(api, candidate, high)
            legacy_parity = (
                (legacy["generalized_load"] < 1e-6)
                & (legacy["displacement"] < 1e-4)
                & (legacy["stored6_tensor"] < 1e-4)
                & (legacy["peak"] < 1e-4)
            )
            legacy_spatial = bool(
                (legacy["stored6_tensor"][loaded] <= 0.1).all()
                and (legacy["peak"][loaded] <= 0.05).all()
                and (not loaded.any() or legacy["stored6_tensor"][loaded].mean() <= 0.05)
            )
            consistency = {
                name: api.rules.legacy_error(production[name], candidate[name])
                for name in ("force", "rhs", "displacement", "peak", "step_peak")
            }
            if args.output_mode == "full":
                consistency["stored6_tensor"] = api.rules.legacy_error(
                    production["stress_tensor"], candidate["stress_tensor"]
                )
            getter = api.tensor_to_array(workload.link.get_max_stress()).copy()
            consistency["public_getter_peak"] = api.np.abs(getter - candidate["step_peak"]) / api.np.maximum(
                api.np.abs(candidate["step_peak"]), 1e-30
            )
            consistent = all((value <= 1e-8).all() for value in consistency.values())
            admission = _snapshot(api, entry.contacts.admission)
            counters = admission["counters"]
            delta = (
                counters - before_counts + api.np.where(reset[:, None], before_counts, 0) if counters.size else counters
            )
            accounting = True
            if counters.size:
                accounting = bool(
                    (delta >= 0).all()
                    and api.np.array_equal(delta[:, 0], loaded_contacts.sum(axis=1))
                    and api.np.array_equal(delta[:, 1], delta[:, 2:5].sum(axis=1))
                )
            frame_passed = bool(numeric and spatial and consistent and preserved and accounting)
            region_values = api.np.full((args.envs, 3, len(REGION_METRICS)), api.np.nan)
            region_samples = api.np.zeros((args.envs, 3), dtype=api.gs.np_int)
            for env, row in enumerate(region_rows):
                for index, name in enumerate(REGIONS):
                    region_samples[env, index] = row[name]["samples"]
                    if row[name]["applicability"] != "empty":
                        region_values[env, index] = [row[name][metric] for metric in REGION_METRICS]
                        for metric in (
                            "tensor_relative_L2",
                            "peak_bidirectional_relative",
                            "peak_underestimate_relative",
                        ):
                            key = name + "-" + metric
                            if key not in worst or row[name][metric] > worst[key]["value"]:
                                worst[key] = {
                                    "value": row[name][metric],
                                    "step": step,
                                    "environment": env,
                                    "region": name,
                                    "metric": metric,
                                    "payload": (
                                        physical,
                                        {"production": production, "mirror": candidate, "reference128": high},
                                        runtime,
                                        production_contacts,
                                        admission,
                                    ),
                                }
            history.append(
                {
                    "regions": region_values,
                    "region_samples": region_samples,
                    "loaded": loaded,
                    "body_omega": physical["omega"],
                    "material_frame_position": physical["frame_position"],
                    "material_frame_quaternion": physical["frame_quaternion"],
                    "runtime": api.np.stack([[checks[name] for name in RUNTIME_FLAGS] for checks in runtime.values()]),
                    "legacy": api.np.stack(list(legacy.values())),
                    "route_counters": delta,
                    "original_source_max_force_moment_relative": api.np.stack(
                        [
                            [
                                checks[name].max(axis=1, initial=0)
                                for name in ("source_relative_force_error", "source_relative_point_moment_error")
                            ]
                            for checks in runtime.values()
                        ]
                    ),
                    "original_elastic_residual": api.np.stack(
                        [checks["elastic_residual"] for checks in runtime.values()]
                    ),
                    "old_strict_origin_force_moment_relative": api.np.stack(
                        [
                            [checks[name] for name in ("strict_origin_force_relative", "strict_origin_moment_relative")]
                            for checks in runtime.values()
                        ]
                    ),
                    "reset": reset,
                    "release": released,
                    "recontact": recontact,
                }
            )
            env_ids, slot_ids = api.np.nonzero(physical["valid"])
            records = {
                "step": api.np.full(len(env_ids), step, dtype=api.gs.np_int),
                "environment": env_ids,
                "slot": slot_ids,
            }
            records.update(
                {
                    name: value[env_ids, slot_ids]
                    for name, value in production_contacts.items()
                    if value.ndim >= 2 and value.shape[:2] == physical["valid"].shape
                }
            )
            loaded_records.append(records)
            row = {
                "step": step,
                "attempted_tick": tick,
                "loaded_environments": int(loaded.sum()),
                "loaded_sources": int(loaded_contacts.sum()),
                "passed": frame_passed,
                "numerical_passed": bool(numeric),
                "application_regions_passed": bool(spatial),
                "application_regions_evaluated": True,
                "same_mechanics_max_full_passed": bool(consistent),
                "inputs_target_state_rigid_unchanged": bool(preserved),
                "route_counters_accounted": bool(accounting),
                "legacy_parity_passed": bool(legacy_parity.all()),
                "legacy_spatial_per_frame_5mean_10worst_5peak_diagnostic": legacy_spatial,
                "legacy_strict_origin_diagnostic_passed": all(
                    check["strict_origin_diagnostic_passed"].all() for check in runtime.values()
                ),
                "reset_environments": api.np.flatnonzero(reset).tolist(),
                "release_environments": api.np.flatnonzero(released).tolist(),
                "recontact_environments": api.np.flatnonzero(recontact).tolist(),
                "maximum_application_tensor_error": max(
                    region["tensor_relative_L2"]
                    for envrow in region_rows
                    for region in envrow.values()
                    if region["applicability"] != "empty"
                ),
                "invalid_environments": {
                    label: api.np.flatnonzero(~checks["valid"]).tolist() for label, checks in runtime.items()
                },
            }
            result["rows"].append(row)
            save = (
                step % args.keyframe_every == 0
                or step == args.steps - 1
                or first_loaded
                or reset.any()
                or released.any()
                or recontact.any()
                or not frame_passed
            )
            if save:
                directory = args.output / f"frame-{step:06d}"
                _saved_frame(
                    api,
                    directory,
                    physical,
                    {"production": production, "mirror": candidate, "reference128": high},
                    runtime,
                    production_contacts,
                    admission,
                )
                _write(directory / "regional-errors.json", region_rows)
                _write(directory / "legacy-errors.json", {key: value.tolist() for key, value in legacy.items()})
                result["keyframes"].append({"step": step, "path": str(directory)})
            for label, route, output in (
                ("production", entry, production),
                ("mirror", mirror, candidate),
                ("reference128", reference, high),
            ):
                disputed = api.np.flatnonzero(
                    ~runtime[label]["strict_origin_diagnostic_passed"] | newly_loaded | first_recontact
                )
                if len(disputed) and runtime[label]["source_valid"][disputed].all():
                    if label not in raw_witnesses:
                        actual_route = (
                            route
                            if label != "production"
                            else SimpleNamespace(
                                entry=entry,
                                contacts=entry.contacts,
                                scatter=entry.scatter,
                                surface=entry.surface,
                                options=entry.options,
                            )
                        )
                        raw_witnesses[label] = RawWitness(api, actual_route)
                    witness = raw_witnesses[label].run(
                        physical, output, disputed, vertices, args.output / f"witness-{step:06d}-{label}"
                    )
                    result["witnesses"].append({"step": step, "route": label, **witness})
                    frame_passed &= witness["passed"]
                    row["passed"] = frame_passed
            if any(item["step"] == step for item in result["witnesses"]):
                after_witness_state = _snapshot(api, entry.state)
                after_witness_contacts = _snapshot(api, entry.contacts)
                after_witness_rigid = _rigid_snapshot(api, workload)
                preserved &= all(_same(value, after_witness_state[name]) for name, value in production_state.items())
                preserved &= all(
                    _same(value, after_witness_contacts[name]) for name, value in production_contacts.items()
                )
                preserved &= after_witness_rigid["tick"] == rigid_before["tick"] and all(
                    _same(after_witness_rigid[name], rigid_before[name]) for name in ("pos", "quat", "cd_ang")
                )
                row["inputs_target_state_rigid_unchanged"] = bool(preserved)
                row["passed"] &= bool(preserved)
            previously_loaded = loaded
            if step % 20 == 0 or not frame_passed:
                _write(args.output / "result.json", result)
                print(json.dumps({"step": step, "passed": frame_passed, "loaded": int(loaded.sum())}), flush=True)
            witnesses_passed = all(item["passed"] for item in result["witnesses"] if item["step"] == step)
            if not (numeric and consistent and preserved and accounting and witnesses_passed):
                raise ArithmeticError(
                    "A complete-history numerical, max/full, input, raw-witness or route contract failed."
                )
        result["history_terminal_tick"] = workload.tick
        result["lifecycle"] = _lifecycle(api, workload, entry, policy)
    except Exception as error:  # noqa: BLE001 - worker preserves and forwards the original failure below
        # This top-level worker forwards the original error after preserving the
        # complete partial journal/raw failed state; it never continues policy.
        step_error = error
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "tick": workload.tick,
        }
        api.np.savez_compressed(args.output / "failure-native-state.npz", **_snapshot(api, entry.state))
        api.np.savez_compressed(
            args.output / "failure-native-contacts.npz",
            **_snapshot(api, entry.contacts),
            omega=api.qd_to_numpy(entry.omega, transpose=True, copy=True),
        )
    finally:
        if history:
            api.np.savez_compressed(
                args.output / "all-steps.npz",
                **{key: api.np.stack([row[key] for row in history]) for key in history[0]},
            )
        if loaded_records:
            api.np.savez_compressed(
                args.output / "all-valid-source-rows.npz",
                **{key: api.np.concatenate([row[key] for row in loaded_records]) for key in loaded_records[0]},
            )
        if actions:
            api.np.savez_compressed(
                args.output / "policy-actions.npz",
                actions=api.np.stack(actions),
                observations_before_step=api.np.stack(observations),
            )
        worst_rows = []
        saved_worst = set()
        for key, value in worst.items():
            if value["step"] not in saved_worst:
                _saved_frame(api, args.output / f"worst-frame-{value['step']:06d}", *value["payload"])
                saved_worst.add(value["step"])
            worst_rows.append({k: v for k, v in value.items() if k != "payload"})
        result["worst_application_errors"] = worst_rows
        source_after = [file_id(path) for path in source_files]
        input_unchanged = (
            file_id(args.conditions) == result["conditions"] and file_id(args.budgets) == result["budgets"]
        )
        result.update(
            complete=step_error is None and len(result["rows"]) == args.steps,
            completed_observations=len(result["rows"]),
            terminal_tick=workload.tick,
            source_files_after=source_after,
            source_files_unchanged=source_before == source_after,
            input_files_unchanged=input_unchanged,
            loaded_seen=load_seen.tolist(),
            release_seen=release_seen.tolist(),
            recontact_seen=recontact_seen.tolist(),
            reset_counts=reset_counts.tolist(),
            all_environments_complete_600_step_cycle=bool((args.steps - workload.delays >= 600).all()),
            array_contract={
                "regions": REGIONS,
                "region_metrics": REGION_METRICS,
                "runtime_flags": RUNTIME_FLAGS,
                "runtime_routes": ["production", "mirror", "reference128"],
                "legacy_metrics": ["generalized_load", "rhs", "displacement", "stored6_tensor", "peak"],
            },
        )
        result["history_environment_counts"] = {
            "attempted": result.get("attempted_steps", 0) * args.envs,
            "fully_audited": len(history) * args.envs,
            "valid_fully_audited_by_route": {
                label: int(sum(row["runtime"][index, 0].sum() for row in history))
                for index, label in enumerate(("production", "mirror", "reference128"))
            },
            "scope": "Audited history only; lifecycle fresh observation is additional. A failed native attempt is retained in its raw failure receipt and is never counted as a fully audited valid frame.",
        }
        result["passed"] = bool(
            result["complete"]
            and all(row["passed"] for row in result["rows"])
            and result["lifecycle"]["passed"]
            and source_before == source_after
            and input_unchanged
        )
        result["history_coverage"] = {
            "at_least_one_loaded_environment": bool(load_seen.any()),
            "release_observed": bool(release_seen.any()),
            "recontact_observed": bool(recontact_seen.any()),
            "automatic_partial_reset_observed": bool(reset_counts.sum() > 0),
        }
        result["bounded_history_qualified"] = bool(
            result["passed"] and args.steps >= 730 and all(result["history_coverage"].values())
        )
        result["full_trajectory_qualified"] = bool(
            result["bounded_history_qualified"] and result["all_environments_complete_600_step_cycle"]
        )
        result["legacy_parity_passed"] = (
            all(row["legacy_parity_passed"] for row in result["rows"]) if result["rows"] else False
        )
        loaded_legacy_field = (
            api.np.concatenate([row["legacy"][3, row["loaded"]] for row in history]) if history else api.np.empty(0)
        )
        loaded_legacy_peak = (
            api.np.concatenate([row["legacy"][4, row["loaded"]] for row in history]) if history else api.np.empty(0)
        )
        legacy_spatial_applicable = bool(len(loaded_legacy_field))
        legacy_spatial_mean = float(loaded_legacy_field.mean()) if legacy_spatial_applicable else None
        legacy_spatial_worst = float(loaded_legacy_field.max()) if legacy_spatial_applicable else None
        legacy_spatial_peak = float(loaded_legacy_peak.max()) if legacy_spatial_applicable else None
        result["legacy_spatial_5mean_10worst_5peak"] = {
            "applicability": "loaded_environment_frames" if legacy_spatial_applicable else "empty",
            "loaded_environment_frames": len(loaded_legacy_field),
            "stored6_field_mean": legacy_spatial_mean,
            "stored6_field_worst": legacy_spatial_worst,
            "peak_worst": legacy_spatial_peak,
            "scope": "Unweighted stored6 relative field error averaged over all loaded environment/frame pairs; original5%mean/10%worst/5%peak. The stricter per-frame cross-environment mean remains a separate diagnostic. This is distinct from1e-6/1e-4 volume parity and new weighted full9 regional application budgets.",
        }
        result["legacy_spatial_5mean_10worst_5peak_passed"] = bool(
            result["complete"]
            and legacy_spatial_applicable
            and legacy_spatial_mean <= 0.05
            and legacy_spatial_worst <= 0.1
            and legacy_spatial_peak <= 0.05
        )
        if policy is not None:
            _values, final_hash = _policy_state(api, policy)
            result["policy"]["weights_sha256_after"] = final_hash
            result["policy"]["weights_unchanged"] = final_hash == result["policy"]["weights_sha256"]
            result["policy"]["actions_before_attempted_steps"] = len(actions)
            result["policy"]["observations_before_attempted_steps"] = len(observations)
            result["policy"]["every_audited_step_has_original_action"] = len(actions) == args.steps == len(observations)
            result["passed"] &= result["policy"]["weights_unchanged"]
            result["passed"] &= result["policy"]["every_audited_step_has_original_action"]
            result["full_trajectory_qualified"] &= result["policy"]["weights_unchanged"]
            result["full_trajectory_qualified"] &= result["policy"]["every_audited_step_has_original_action"]
            result["bounded_history_qualified"] &= result["policy"]["weights_unchanged"]
            result["bounded_history_qualified"] &= result["policy"]["every_audited_step_has_original_action"]
        _write(args.output / "result.json", result)
    if step_error is not None:
        raise step_error
    if not result["passed"]:
        raise ArithmeticError("Complete-history qualification failed; preserve all negative evidence.")


if __name__ == "__main__":
    main()
