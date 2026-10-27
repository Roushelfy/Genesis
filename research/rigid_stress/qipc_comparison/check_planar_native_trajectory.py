"""Audit evolving production planar DPR fields against original Q4 loading at two integration orders.

Run in Slurm rtx-mid/image 1_26. Each reference consumes saved body-frame source
inputs and angular velocity from the completed production recovery. The scene
advances once per observation; all reference work is an intrusive accuracy audit.
"""

import argparse
import json
import math
import os
import subprocess
import time
import traceback
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace

COMPARISONS = ("planar_vs_resolved_Q4", "production_Q4_vs_resolved_Q4", "planar_vs_production_Q4")
METRICS = ("field", "peak", "N", "M", "Q")
ROUTE_COUNTERS = (
    "attempted_loaded_contacts",
    "proxy_contacts",
    "fallback_contacts",
    "analytic_sections",
    "overflow_steps",
)


def _json_number(value):
    return float(value) if math.isfinite(value) else None


def _api():
    import numpy as np
    import quadrants as qd
    from scipy import sparse

    import genesis as gs
    from examples.rigid.franka_egg_stress import FrankaEgg
    from genesis.engine.solvers.rigid.stress.contact import (
        StressContactState,
        create_contacts,
        kernel_anchor,
        kernel_pressure,
        kernel_scatter,
    )
    from genesis.engine.solvers.rigid.stress.planar_loading import PlanarLoading, kernel_planar_scatter
    from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
    from genesis.engine.solvers.rigid.stress.scatter import (
        create_scatter_workspace,
        kernel_pressure_moments,
        kernel_scatter_faces,
    )
    from genesis.engine.solvers.rigid.stress.surface import StressSurface
    from genesis.utils.array_class import V_VEC, V
    from genesis.utils.misc import qd_to_numpy

    from .artifacts import file_id, runtime_identity, write_json
    from .check_planar_loading_gpu import native_outputs, unbatched_array

    @qd.kernel
    def kernel_copy_physical(
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
        for i_b in range(source_omega.shape[0]):
            target.frame_position[i_b] = source.frame_position[i_b]
            target.frame_quaternion[i_b] = source.frame_quaternion[i_b]
            target_omega[i_b] = source_omega[i_b]
        for _ in range(1):
            target.hertz_profile[None] = source.hertz_profile[None]

    @qd.kernel
    def kernel_copy_scalar(source: qd.Tensor, target: qd.Tensor):
        for _ in range(1):
            target[None] = source[None]

    @qd.kernel
    def kernel_copy_vector(source: qd.Tensor, target: qd.Tensor):
        for i in range(source.shape[0]):
            target[i] = source[i]

    @qd.kernel
    def kernel_copy_matrix(source: qd.Tensor, target: qd.Tensor):
        for i, j in qd.ndrange(source.shape[0], source.shape[1]):
            target[i, j] = source[i, j]

    @qd.kernel
    def kernel_copy_cube(source: qd.Tensor, target: qd.Tensor):
        for i, j, k in qd.ndrange(source.shape[0], source.shape[1], source.shape[2]):
            target[i, j, k] = source[i, j, k]

    @qd.kernel
    def kernel_mask_source(contacts: StressContactState, environment: int, slot: int):
        for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
            if i_c != slot or i_b != environment:
                contacts.valid[i_c, i_b] = False
                contacts.status[i_c, i_b] = 4

    @qd.kernel
    def kernel_mask_proxy(eligible: qd.Tensor, fallback: qd.Tensor, environment: int, slot: int):
        for i_c, i_b in qd.ndrange(eligible.shape[0], eligible.shape[1]):
            if i_c != slot or i_b != environment:
                eligible[i_c, i_b], fallback[i_c, i_b] = False, False

    return SimpleNamespace(**locals())


def _copy_native_fields(api, source, target):
    """Copy an existing fitted snapshot in QD; no CPU numerical payload enters the replay."""
    kernels = (api.kernel_copy_scalar, api.kernel_copy_vector, api.kernel_copy_matrix, api.kernel_copy_cube)
    source_values, target_values = vars(source), vars(target)
    for item in fields(source):
        original, copied = source_values[item.name], target_values[item.name]
        if original.shape != copied.shape:
            raise ValueError(f"Independent scatter snapshot shape differs: {item.name}.")
        if all(original.shape):
            kernels[len(original.shape)](original, copied)


def _source_identity(api):
    root = Path(api.gs.__file__).resolve().parent
    paths = [
        *sorted((root / "engine/solvers/rigid/stress").glob("*.py")),
        root / "options/rigid_stress.py",
        root.parent / "examples/rigid/franka_egg_stress.py",
        root.parent / "examples/rigid/egg_stress_controller.py",
        Path(__file__).resolve(),
    ]
    return {str(path): api.file_id(path) for path in paths}


def _physical_inputs(api, contacts, omega):
    return {
        name: api.qd_to_numpy(field, transpose=True)
        for name, field in (
            ("point", contacts.position),
            ("force", contacts.force),
            ("normal", contacts.normal),
            ("friction", contacts.friction),
            ("radius", contacts.radius),
            ("radius_status", contacts.radius_status),
            ("valid", contacts.valid),
            ("frame_position", contacts.frame_position),
            ("frame_quaternion", contacts.frame_quaternion),
            ("omega", omega),
        )
    }


@dataclass
class OriginalRoute:
    api: object
    entry: object
    quadrature: int
    source_epsilon: float
    reference_tasks: int
    options: object = field(init=False)
    surface: object = field(init=False)
    contacts: object = field(init=False)
    state: object = field(init=False)
    scatter: object = field(init=False)
    omega: object = field(init=False)
    errno: object = field(init=False)

    def __post_init__(self):
        api, entry = self.api, self.entry
        n_contacts, n_envs = entry.contacts.valid.shape
        self.options = entry.options.model_copy(
            update={"shell_loading": "q4", "quadrature": self.quadrature, "history_size": 0}
        )
        self.surface = api.StressSurface(
            self.quadrature, entry.model.load_info, angular=self.options.pressure_quadrature == "angle"
        )
        self.contacts = api.create_contacts(
            n_contacts,
            n_envs,
            entry.options.contact_radius,
            entry.options.cooperative_pressure,
            entry.options.cooperative_scatter,
            True,
        )
        self.state = entry.model.create_state(n_envs, "full")
        self.scatter = api.create_scatter_workspace(n_envs, self.reference_tasks, entry.reuse_contact_moments)
        self.omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(n_envs,))
        self.errno = api.V(dtype=api.gs.qd_int, shape=(n_envs,))
        self.errno.fill(0)
        self.omega.fill(0.0)

    def recover(self):
        api, entry = self.api, self.entry
        api.kernel_copy_physical(entry.contacts, self.contacts, entry.omega, self.omega)
        self.errno.fill(0)
        api.kernel_begin_step(self.state)
        api.kernel_anchor(self.source_epsilon, self.contacts, self.surface.info)
        if entry.reuse_contact_moments:
            api.kernel_pressure_moments(
                self.contacts,
                self.state,
                entry.model.load_info,
                self.surface.info,
                self.scatter,
                self.options.cached_face_bounds,
            )
        else:
            api.kernel_pressure(self.contacts, self.surface.info, self.options.cooperative_pressure)
        if self.scatter.tasks.shape[0] and self.options.cooperative_scatter:
            api.kernel_scatter_faces(
                self.contacts,
                self.state,
                entry.model.load_info,
                self.surface.info,
                self.scatter,
                self.options.cached_face_bounds,
                entry.reuse_contact_moments,
            )
        else:
            api.kernel_scatter(
                self.contacts,
                self.state,
                entry.model.load_info,
                self.surface.info,
                self.options.cooperative_scatter,
                self.options.cached_face_bounds,
            )
        entry.model.recover(self.omega, self.state, self.options, history=None, surface_load=True)
        api.kernel_accept(self.state, self.contacts, self.errno)
        return api.native_outputs(api, entry.model, self.state)


class SourceScatterWitness:
    """Mask a saved fitted source and replay only its native scatter, preserving fit and capacity route."""

    def __init__(self, api, entry, surface, capacity, planar=None):
        self.api, self.entry, self.surface, self.planar_source = api, entry, surface, planar
        n_contacts, n_envs = entry.contacts.valid.shape
        self.contacts = api.create_contacts(
            n_contacts,
            n_envs,
            entry.options.contact_radius,
            entry.options.cooperative_pressure,
            entry.options.cooperative_scatter,
            True,
        )
        self.state = entry.model.create_state(n_envs, "max")
        self.scatter = api.create_scatter_workspace(n_envs, capacity, entry.reuse_contact_moments)
        self.planar = (
            api.PlanarLoading(
                surface.info,
                entry.model.load_info,
                self.contacts,
                entry.options.loading_subdivisions,
                geometry=planar.info,
                tasks_per_env=planar.state.tasks.shape[0],
                exact_radial=planar.exact_radial,
            )
            if planar is not None
            else None
        )

    def raw(self, contacts, scatter, environment, slot):
        api, entry = self.api, self.entry
        _copy_native_fields(api, contacts, self.contacts)
        _copy_native_fields(api, scatter, self.scatter)
        self.scatter.wrench.fill(0.0)
        self.state.force.fill(0.0)
        api.kernel_mask_source(self.contacts, environment, slot)
        current_contacts, surface = self.contacts, self.surface.info
        if self.planar is not None:
            _copy_native_fields(api, self.planar_source.state, self.planar.state)
            _copy_native_fields(api, self.planar_source.material_state, self.planar.material_state)
            api.kernel_mask_proxy(self.planar.state.eligible, self.planar.state.fallback, environment, slot)
            current_contacts, surface = self.planar.fallback_contacts, self.planar.material_surface
        if self.scatter.tasks.shape[0] and entry.options.cooperative_scatter:
            api.kernel_scatter_faces(
                current_contacts,
                self.state,
                entry.model.load_info,
                surface,
                self.scatter,
                entry.options.cached_face_bounds,
                entry.reuse_contact_moments,
            )
        else:
            api.kernel_scatter(
                current_contacts,
                self.state,
                entry.model.load_info,
                surface,
                entry.options.cooperative_scatter,
                entry.options.cached_face_bounds,
            )
        if self.planar is not None:
            api.kernel_planar_scatter(self.contacts, self.planar.state, entry.model.load_info, self.state)
        force = api.qd_to_numpy(self.state.force, transpose=True)
        return force[environment].reshape((-1, 6))


def _gamma(unit_roundoff, operations):
    product = unit_roundoff * operations
    if product >= 1.0:
        raise ArithmeticError("The explicit floating-point operation bound is too large.")
    return product / (1.0 - product)


def _wrench_roundoff(api, nodal, forces, points, vertices, reported_force, contribution_bound):
    """Absolute-sum forward error bound for scatter mapping and host wrench reductions.

    u=eps/2 and gamma(n)=nu/(1-nu). Each fitted per-node load
    contribution needs at most32 scalar arithmetic/atomic operations for
    load weighting, offset cross product and its two nodal blocks. This bounds
    load mapping/accumulation after the fitted weights and physical basis;
    it does not certify all geometry inversion/trigonometry/pressure fitting.
    The declared contribution bound covers every face/sample/corner, including
    the serial overflow route. Host cross products and reductions are bounded
    separately; target moment cancellation never reduces the error scale.
    """
    np, unit = api.np, api.np.finfo(api.np.float64).eps / 2.0
    count, n_contacts = len(vertices), forces.shape[1]
    # ||deltaF||_1 <= sqrt(3)||deltaF||_2. Q4 coordinates admitted
    # within the engine's tiny boundary tolerance and P1 convex maps have
    # sum(abs(shape))<2; use that geometric bound, including signed roundoff.
    force_mass = 2.0 * (np.abs(forces).sum(axis=(1, 2)) + np.sqrt(3.0) * reported_force.sum(axis=1))
    force_terms = np.abs(nodal[:, :3]).sum(axis=(0, 1)) + np.abs(forces).sum(axis=(1, 2))
    position_extent = np.abs(vertices).sum(axis=1).max(initial=0.0)
    # For the shell load map, each director offset has norm h/2; the caller
    # includes its actual maximum L1 magnitude in contribution_bound[1].
    offset_extent = contribution_bound[1]
    native = _gamma(unit, 32 * contribution_bound[0] + 32)
    force_roundoff = native * force_mass + _gamma(unit, count + n_contacts + 4) * force_terms
    nodal_moment_terms = (
        np.einsum("n,nkb->b", np.abs(vertices).sum(axis=1), np.abs(nodal[:, :3]))
        + np.abs(nodal[:, 3:]).sum(axis=(0, 1))
        + (np.abs(points).sum(axis=2) * np.abs(forces).sum(axis=2)).sum(axis=1)
    )
    moment_roundoff = native * (position_extent + offset_extent) * force_mass
    moment_roundoff += _gamma(unit, 8 * count + 8 * n_contacts + 32) * nodal_moment_terms
    return force_roundoff, moment_roundoff


def _source_witness(api, replay, contacts, scatter, physical, actual, environments, vertices, output):
    """Independently evaluate raw nodal source-point wrenches from the exact fitted/routed snapshot."""
    np, arrays, rows = api.np, [], []
    passed = np.ones(len(physical["valid"]), dtype=bool)
    reconstructed = np.zeros_like(actual["force"])
    force_mass = np.zeros(len(passed))
    source_count = np.zeros(len(passed), dtype=np.int64)
    reported_force = api.qd_to_numpy(contacts.force_error, transpose=True)
    reported_moment = api.qd_to_numpy(contacts.moment_error, transpose=True)
    for environment in environments:
        for slot in np.flatnonzero(
            physical["valid"][environment] & (np.linalg.norm(physical["force"][environment], axis=1) > 1e-30)
        ):
            raw = replay.raw(contacts, scatter, int(environment), int(slot))
            force, point = physical["force"][environment, slot], physical["point"][environment, slot]
            magnitude, radius = np.linalg.norm(force), physical["radius"][environment, slot]
            force_error = np.linalg.norm(raw[:, :3].sum(axis=0) - force)
            point_moment = np.cross(vertices - point, raw[:, :3]).sum(axis=0) + raw[:, 3:].sum(axis=0)
            moment_error = np.linalg.norm(point_moment)
            source_passed = bool(
                np.isfinite(raw).all() and force_error <= 1e-8 * magnitude and moment_error <= 1e-8 * magnitude * radius
            )
            passed[environment] &= source_passed
            reconstructed[:, environment] += raw.reshape(-1)
            force_mass[environment] += np.abs(force).sum()
            source_count[environment] += 1
            arrays.append(raw)
            rows.append(
                {
                    "environment": int(environment),
                    "slot": int(slot),
                    "passed": source_passed,
                    "absolute_force_error": _json_number(force_error),
                    "absolute_point_moment_error": _json_number(moment_error),
                    "relative_force_error": _json_number(force_error / magnitude),
                    "relative_point_moment_error": _json_number(moment_error / (magnitude * radius)),
                    "reported_absolute_force_error": _json_number(reported_force[environment, slot]),
                    "reported_absolute_point_moment_error": _json_number(reported_moment[environment, slot]),
                }
            )
    # The fitted coefficients, refinement flags, task buffers, material mask,
    # capacity overflow route and proxy weights are copied verbatim in QD.
    # Thus only scalar accumulation order differs between these raw scatters.
    difference = (reconstructed - actual["force"]).reshape((len(vertices), 6, len(passed)))
    original = actual["force"].reshape((len(vertices), 6, len(passed)))
    unit = np.finfo(np.float64).eps / 2.0
    faces = replay.surface.info.face_origin.shape[0]
    contributions = max(1, int(source_count.max(initial=0))) * faces * 4
    if replay.planar is not None:
        contributions += replay.planar.state.tasks.shape[0] * 4
    # The same sum(abs(shape))<2 geometric bound also covers per-source
    # accumulation with opposite inner/outer rotational contributions.
    force_mass *= 2.0 * (1.0 + np.sqrt(3.0) * 1e-8)
    roundoff = _gamma(unit, 32 * contributions + 32) * force_mass
    offset = api.unbatched_array(api, replay.entry.model.load_info.load_offsets)
    offset_extent = np.abs(offset).sum(axis=1).max(initial=0.0)
    translation_difference = np.abs(difference[:, :3]).sum(axis=(0, 1))
    rotation_difference = np.abs(difference[:, 3:]).sum(axis=(0, 1))
    # Summing independently rounded source arrays adds at most gamma(C)
    # times their absolute sum, bounded by positive pressure force mass.
    host = _gamma(unit, int(source_count.max(initial=0)) + 4) * force_mass
    translation_bound = roundoff + host
    rotation_bound = offset_extent * (roundoff + host)
    reconstruction = (translation_difference <= translation_bound) & (rotation_difference <= rotation_bound)
    # With moment reuse disabled the face API rebuilds tasks, so its snapshot
    # capacity route has not been proved identical. Keep this control negative
    # rather than certifying a disputed frame from a different execution path.
    route_preserved = replay.entry.reuse_contact_moments or not replay.entry.options.cooperative_scatter
    passed[environments] &= reconstruction[environments] & route_preserved
    source_environment = np.array([row["environment"] for row in rows], dtype=np.int64)
    source_slot = np.array([row["slot"] for row in rows], dtype=np.int64)
    api.np.savez_compressed(
        output.with_suffix(".npz"),
        raw_source=np.stack(arrays) if arrays else np.empty((0, len(vertices), 6)),
        source_environment=source_environment,
        source_slot=source_slot,
        source_point=physical["point"][source_environment, source_slot],
        source_force=physical["force"][source_environment, source_slot],
        source_radius=physical["radius"][source_environment, source_slot],
        reported_source_force_error=reported_force[source_environment, source_slot],
        reported_source_point_moment_error=reported_moment[source_environment, source_slot],
        physical_vertices=vertices,
        reconstructed_raw=reconstructed,
        original_raw=actual["force"],
        translation_difference=translation_difference,
        rotation_difference=rotation_difference,
        translation_roundoff_bound=translation_bound,
        rotation_roundoff_bound=rotation_bound,
    )
    summary = {
        "passed": bool(passed[environments].all()),
        "environments": list(map(int, environments)),
        "sources": rows,
        "source_point_tolerance": 1e-8,
        "environment_passed": passed.tolist(),
        "raw_reconstruction_passed": reconstruction[environments].tolist(),
        "same_fitted_capacity_route_preserved": bool(route_preserved),
        "contract": "QD copy of the exact fitted and routed snapshot; masked single-source scatter only, no pressure refit, association, rigid step, elastic solve or CPU oracle. Independent raw node translations/rotations determine application-point force/moment; source sums reconstruct the primary raw load within explicit accumulation roundoff.",
    }
    api.write_json(output.with_suffix(".json"), summary)
    return passed, summary


def _errors(api, actual, expected, weights):
    np, values = api.np, []
    multiplicity = np.array([1.0, 1.0, 2.0])
    for name in METRICS:
        if name == "peak":
            error = np.abs(actual[name] - expected[name]) / np.maximum(np.abs(expected[name]), 1e-30)
        elif name == "field":
            delta = actual["stress_full9"] - expected["stress_full9"]
            squared = np.einsum("segcb,segcb,eg->b", delta, delta, weights)
            norm = np.einsum("segcb,segcb,eg->b", expected["stress_full9"], expected["stress_full9"], weights)
            error = np.sqrt(squared) / np.maximum(np.sqrt(norm), 1e-30)
        else:
            delta = actual[name] - expected[name]
            if name in ("N", "M"):
                squared = np.einsum("egcb,egcb,eg,c->b", delta, delta, weights, multiplicity)
                norm = np.einsum("egcb,egcb,eg,c->b", expected[name], expected[name], weights, multiplicity)
            else:
                squared = np.einsum("egcb,egcb,eg->b", delta, delta, weights)
                norm = np.einsum("egcb,egcb,eg->b", expected[name], expected[name], weights)
            error = np.sqrt(squared) / np.maximum(np.sqrt(norm), 1e-30)
        values.append(error)
    return np.stack(values)


def _runtime_checks(api, state, contacts, errno, output, matrix, free, options, physical, vertices, rounding):
    np = api.np
    residual = (matrix @ output["q"] - output["rhs"])[free]
    residual_norm = np.linalg.norm(residual, axis=0)
    rhs_norm = np.linalg.norm(output["rhs"][free], axis=0)
    numerical = residual_norm <= np.maximum(options.absolute_tolerance, options.tolerance * rhs_norm)
    status = api.qd_to_numpy(contacts.status, transpose=True)
    source_valid = physical["valid"]
    contact_valid = ((status == 0) | ~source_valid).all(axis=1)
    finite = np.ones(len(output["valid"]), dtype=bool)
    for key in ("force", "rhs", "q", "stress_full9", "peak", "N", "M", "Q"):
        finite &= np.isfinite(output[key]).reshape((-1, len(finite))).all(axis=0)
    step_valid = api.qd_to_numpy(state.step_valid, transpose=True)
    error_codes = api.qd_to_numpy(errno, transpose=True)
    loaded = source_valid & (np.linalg.norm(physical["force"], axis=-1) > 1e-30)
    physical_contact_finite = np.isfinite(physical["radius"]) & np.isfinite(physical["friction"])
    for key in ("point", "force", "normal"):
        physical_contact_finite &= np.isfinite(physical[key]).all(axis=2)
    physical_contact_finite &= (~loaded) | (physical["radius"] > 0.0)
    physical_valid = ((~source_valid) | physical_contact_finite).all(axis=1)
    for key in ("omega", "frame_position", "frame_quaternion"):
        physical_valid &= np.isfinite(physical[key]).all(axis=1)
    forces = np.where(loaded[..., None], physical["force"], 0.0)
    points = np.where(loaded[..., None], physical["point"], 0.0)
    scale = np.linalg.norm(forces, axis=-1).sum(axis=1)
    radius = np.where(loaded, physical["radius"], 0.0)
    moment_scale = (np.linalg.norm(forces, axis=-1) * radius).sum(axis=1)
    nodal = output["force"].reshape((len(vertices), 6, len(scale)))
    total_force = nodal[:, :3].sum(axis=0).T
    total_moment = np.cross(vertices[:, :, None], nodal[:, :3], axisa=1, axisb=1, axisc=1).sum(axis=0).T
    total_moment += nodal[:, 3:].sum(axis=0).T
    force_error = np.linalg.norm(total_force - forces.sum(axis=1), axis=1) / np.maximum(scale, 1e-30)
    moment_error = np.linalg.norm(total_moment - np.cross(points, forces).sum(axis=1), axis=1) / np.maximum(
        moment_scale, 1e-30
    )
    loaded_env = loaded.any(axis=1)
    strict_origin = (~loaded_env) | ((force_error <= 1e-8) & (moment_error <= 1e-8))
    reported_force = api.qd_to_numpy(contacts.force_error, transpose=True)
    reported_moment = api.qd_to_numpy(contacts.moment_error, transpose=True)
    magnitude = np.linalg.norm(forces, axis=2)
    source_finite = np.isfinite(reported_force) & np.isfinite(reported_moment)
    source_point = (~loaded) | (
        source_finite
        & (reported_force >= 0.0)
        & (reported_moment >= 0.0)
        & (reported_force <= 1e-8 * magnitude)
        & (reported_moment <= 1e-8 * magnitude * radius)
    )
    absolute_reported_force = np.where(loaded, reported_force, 0.0)
    absolute_reported_moment = np.where(loaded, reported_moment, 0.0)
    force_roundoff, moment_roundoff = _wrench_roundoff(
        api, nodal, forces, points, vertices, absolute_reported_force, rounding
    )
    assembled_force_bound = absolute_reported_force.sum(axis=1) + force_roundoff
    assembled_moment_bound = (np.linalg.norm(points, axis=2) * absolute_reported_force + absolute_reported_moment).sum(
        axis=1
    ) + moment_roundoff
    absolute_force_error = np.linalg.norm(total_force - forces.sum(axis=1), axis=1)
    absolute_moment_error = np.linalg.norm(total_moment - np.cross(points, forces).sum(axis=1), axis=1)
    accounting = (absolute_force_error <= assembled_force_bound) & (absolute_moment_error <= assembled_moment_bound)
    unloaded_zero = np.count_nonzero(nodal[:, :, ~loaded_env]) == 0
    base_valid = output["valid"] & step_valid & contact_valid & numerical & finite & physical_valid & (error_codes == 0)
    valid = base_valid & source_point.all(axis=1) & accounting
    return {
        "valid": valid,
        "base_valid": base_valid,
        "physical_input_valid": physical_valid,
        "elastic_residual_valid": numerical,
        "native_state_valid": output["valid"] & step_valid,
        "contact_status_valid": contact_valid,
        "finite_output_valid": finite,
        "source_point_valid": source_point.all(axis=1),
        "source_force_error": reported_force,
        "source_point_moment_error": reported_moment,
        "source_relative_force_error": absolute_reported_force / np.maximum(magnitude, 1e-30),
        "source_relative_point_moment_error": absolute_reported_moment / np.maximum(magnitude * radius, 1e-30),
        "assembled_origin_accounting_valid": accounting,
        "assembled_origin_absolute_force_error": absolute_force_error,
        "assembled_origin_absolute_moment_error": absolute_moment_error,
        "assembled_origin_force_bound": assembled_force_bound,
        "assembled_origin_moment_bound": assembled_moment_bound,
        "assembled_origin_force_roundoff": force_roundoff,
        "assembled_origin_moment_roundoff": moment_roundoff,
        "strict_origin_diagnostic_passed": strict_origin,
        "legacy_valid_with_strict_origin": base_valid & strict_origin,
        "absolute_residual": residual_norm,
        "relative_residual": residual_norm / np.maximum(rhs_norm, 1e-30),
        "contact_status": status,
        "errno": error_codes,
        "force_error": force_error,
        "moment_error": moment_error,
        "unloaded_raw_force_exact_zero": bool(unloaded_zero),
    }


def _route_counts(api, state):
    return api.np.stack(
        (
            api.qd_to_numpy(state.attempted_loaded_contacts, transpose=True),
            api.qd_to_numpy(state.proxy_contacts, transpose=True),
            api.qd_to_numpy(state.fallback_contacts, transpose=True),
            api.qd_to_numpy(state.analytic_sections, transpose=True),
            api.qd_to_numpy(state.overflow_steps, transpose=True),
        )
    )


def _keyframe(step, physical, candidate, low, high, runtime, routing):
    arrays = {f"input_{name}": value for name, value in physical.items()}
    arrays.update(step=step)
    for label, output in (("planar", candidate), ("Q4_production", low), ("Q4_resolved", high)):
        arrays.update({f"{label}_{key}": value for key, value in output.items()})
    for label, checks in runtime.items():
        arrays.update({f"{label}_{key}": value for key, value in checks.items()})
    arrays.update({f"routing_{name}": value for name, value in routing.items()})
    return arrays


def _lifecycle(api, workload, entry):
    """Check selected-environment scene reset and checkpoint invalidation after the primary rollout."""
    np = api.np
    checkpoint = workload.scene.get_state()
    fields = {
        "displacement": entry.state.displacement,
        "stress_tensor": entry.state.stress_tensor,
        "peak": entry.state.peak,
        "N": entry.state.membrane_resultant,
        "M": entry.state.bending_resultant,
        "Q": entry.state.transverse_resultant,
        "attempted": entry.planar.state.attempted_loaded_contacts,
        "proxy": entry.planar.state.proxy_contacts,
        "fallback": entry.planar.state.fallback_contacts,
        "analytic": entry.planar.state.analytic_sections,
        "overflow": entry.planar.state.overflow_steps,
    }
    before = {name: api.qd_to_numpy(field, transpose=True) for name, field in fields.items()}
    others = np.arange(workload.n_envs) != 1
    rows = []
    for name in ("partial_reset", "partial_checkpoint_restore"):
        if name == "partial_reset":
            workload.reset(np.array([1]))
        else:
            workload.scene.reset(state=checkpoint, envs_idx=[1])
        after = {key: api.qd_to_numpy(field, transpose=True) for key, field in fields.items()}
        unchanged = all(
            np.array_equal(after[key][others], value[others], equal_nan=True) for key, value in before.items()
        )
        selected_clear = bool(np.count_nonzero(after["displacement"][1]) == 0 and after["peak"][1] == 0)
        selected_clear &= all(after[key][1] == 0 for key in ("attempted", "proxy", "fallback", "analytic", "overflow"))
        selected_clear &= not api.qd_to_numpy(entry.planar.state.eligible, transpose=True)[1].any()
        selected_clear &= not api.qd_to_numpy(entry.planar.state.fallback, transpose=True)[1].any()
        output_invalidated = all(np.isnan(after[key][1]).all() for key in ("stress_tensor", "N", "M", "Q"))
        rows.append(
            {
                "name": name,
                "passed": bool(unchanged and selected_clear and output_invalidated),
                "other_environments_exact": bool(unchanged),
                "selected_state_clear": bool(selected_clear),
                "selected_derived_output_invalidated": bool(output_invalidated),
            }
        )
    workload.step()
    workload.scene.rigid_solver.check_errno()
    fresh = api.native_outputs(api, entry.model, entry.state)
    admitted = api.qd_to_numpy(entry.state.step_valid, transpose=True).all() and fresh["valid"].all()
    finite = all(np.isfinite(fresh[key]).all() for key in ("stress_full9", "N", "M", "Q", "peak"))
    rows.append({"name": "fresh_observation_after_restore", "passed": bool(admitted and finite)})
    return {
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "scope": "Immediate scene-mutation isolation and fresh observation; primary 600-step error statistics end before these controls.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--subdivisions", type=int, choices=(4, 8, 16, 32), default=8)
    parser.add_argument("--planar-integration", choices=("radial", "gauss"), default="radial")
    parser.add_argument("--cooperative-anchor", action="store_true")
    parser.add_argument("--quadrature", type=int, choices=(10, 16, 24, 48), default=16)
    parser.add_argument("--production-reference-quadrature", type=int, default=None)
    parser.add_argument("--reference-quadrature", type=int, default=48)
    parser.add_argument("--pressure-quadrature", choices=("edge", "angle"), default="edge")
    parser.add_argument("--tasks-per-env", type=int, default=32)
    parser.add_argument("--reference-tasks-per-env", type=int, default=32)
    parser.add_argument("--method", choices=("auto", "direct", "inverse"), default="auto")
    parser.add_argument("--field-steps", type=int, nargs="*", default=[129, 221, 400, 599])
    parser.add_argument("--lifecycle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    production_reference_quadrature = (
        args.quadrature if args.production_reference_quadrature is None else args.production_reference_quadrature
    )
    if (
        min(args.envs, args.steps, args.tasks_per_env, args.reference_tasks_per_env) < 1
        or production_reference_quadrature < 1
        or args.reference_quadrature < max(args.quadrature, production_reference_quadrature)
    ):
        parser.error(
            "Use positive dimensions/capacities and a reference order at least as large as the production order."
        )
    if args.lifecycle and args.envs < 2:
        parser.error("Partial-environment lifecycle controls require at least two environments.")
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh output under the project data directory /mnt/data/zhaofeng.")
    args.output.mkdir(parents=True, exist_ok=False)
    api, started = _api(), time.perf_counter()
    np = api.np
    budgets = json.loads(args.budgets.read_text())
    if not budgets["frozen_before_native_tuning"]:
        raise ValueError("Use the error budgets frozen before native tuning.")
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    source_before = _source_identity(api)
    identity = api.runtime_identity()
    checkout = subprocess.check_output(["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "schema": "genesis.planar-native-trajectory.v2",
        "identity": identity,
        "checkout_commit": checkout,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions": api.file_id(args.conditions),
        "budget_input": api.file_id(args.budgets),
        "budgets": budgets,
        "complete": False,
        "passed": False,
        "primary_timing": False,
        "rows": [],
        "scope": "Scripted Franka controller, rigid dynamics and production both-side planar DPR; independent saved-input original-Q4 references each observation.",
        "frame_contract": "Copy solved body-frame physical contacts and angular velocity after the target step. Reference calls advance neither the rigid scene nor contact association.",
        "quadrature_scope": "Target and selected production-reference orders are recorded separately. Original-Q4 low versus resolved order measures baseline integration sensitivity. The default low order follows the target, including genuine low-order rejections.",
        "requested_horizon_is_full_cycle": args.steps >= 600,
        "simulation": {"dt_seconds": 0.01, "substeps": 1, "precision": "64", "controller": "native scripted"},
        "counter_total_scope": "Completed audited observations; terminal counters preserve an interrupted attempt.",
        "production_reference_quadrature": production_reference_quadrature,
        "wrench_contract": {
            "source_point_gate": "For each loaded original source, absolute eF<=1e-8|F| and application-point eM<=1e-8|F|r; native status, errno, finite fields and original K residual remain separate gates.",
            "assembled_body_origin_gate": "|deltaF|<=sum(eF)+roundoffF; |deltaM0|<=sum(|p|eF+eM)+roundoffM, since deltaM0=sum(m_c+p_c cross deltaF_c).",
            "legacy_strict_origin_diagnostic": "The previous body-origin residual divided by sum(|F|r)<=1e-8 remains an explicitly reported diagnostic, including every failure. It is not the per-source application-point gate.",
            "roundoff": "u=eps/2, gamma(n)=nu/(1-nu); explicit scatter face/sample/corner operation upper bound plus host absolute-product cross/reduction bounds, independently checked by exact-snapshot single-source raw scatter at disputed frames.",
            "independent_witness": "QD copy preserves original fitted coefficients, refinement, capacity route, task buffers, material masks and proxy weights. Only scatter is masked by source; independently computed raw node application-point wrenches and sourcewise reconstruction must pass.",
            "floating_point_scope": "FP64, finite normal-range physical data, nonnegative pressure and positive Q4/P1 physical basis; no overflow/underflow. The declared finite operation bound must have nu<1.",
        },
        "strict_origin_failed_observations": [],
        "numerical_failed_observations": [],
        "wrench_witnesses": [],
    }
    api.write_json(args.output / "result.json", result)
    workload = api.FrankaEgg(
        args.envs,
        level=args.level,
        conditions=args.conditions,
        output_mode="full",
        egg_geometry="mesh64",
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        stress_backend="shell",
        shell_loading="planar",
        loading_subdivisions=args.subdivisions,
        quadrature=args.quadrature,
        pressure_quadrature=args.pressure_quadrature,
        scatter_tasks_per_env=args.tasks_per_env,
        method=args.method,
        inverse_precision="64",
        history_size=0,
        timestep=0.01,
        substeps=1,
    )
    solver = workload.scene.rigid_solver
    recovery, entry = solver.stress_recovery, solver.stress_recovery.links[0]
    if entry.planar is None or api.qd_to_numpy(entry.planar.state.side_filter, transpose=True).any():
        raise ArithmeticError(
            "Production trajectory qualification requires the enabled planar route with its default both-side scope."
        )
    # Read the declared production attribute before assigning, so an older
    # engine fails instead of silently acquiring a new research-only property.
    if not isinstance(entry.planar.exact_radial, bool):
        raise TypeError("PlanarLoading.exact_radial must be declared as a production boolean.")
    entry.planar.exact_radial = args.planar_integration == "radial"
    if not isinstance(entry.cooperative_anchor, bool):
        raise TypeError("StressLink.cooperative_anchor must be declared as a production boolean.")
    entry.cooperative_anchor = args.cooperative_anchor
    anchor_mode = (
        "unused"
        if entry.intrinsic is not None or entry.resolved is not None
        else "fused_volume_serial"
        if entry.fused_pipeline
        else "separate_warp"
        if entry.cooperative_anchor and api.gs.backend == api.gs.cuda and entry.contacts.active_pairs.shape[0] > 0
        else "separate_serial"
    )
    low = OriginalRoute(
        api, entry, production_reference_quadrature, recovery.source_epsilon, args.reference_tasks_per_env
    )
    high = OriginalRoute(api, entry, args.reference_quadrature, recovery.source_epsilon, args.reference_tasks_per_env)
    matrix = api.sparse.bsr_matrix(
        (
            api.unbatched_array(api, entry.model.info.stiffness) * entry.options.young,
            api.unbatched_array(api, entry.model.info.columns),
            api.unbatched_array(api, entry.model.info.row_start),
        )
    ).tocsr()
    free = api.unbatched_array(api, entry.model.info.is_free).reshape(-1).astype(bool)
    weights = api.unbatched_array(api, entry.model.shell.weights)
    with np.load(entry.options.mesh, allow_pickle=False) as mesh:
        vertices = mesh["vertices"]
    offset_extent = np.abs(api.unbatched_array(api, entry.model.load_info.load_offsets)).sum(axis=1).max(initial=0.0)
    result.update(
        options=entry.options.model_dump(mode="json"),
        mesh=api.file_id(Path(entry.options.mesh)),
        selected_method=entry.model.selected_method,
        surface_inverse=entry.model.surface_inverse is not None,
        reuse_contact_moments=entry.reuse_contact_moments,
        condition_count=workload.condition_count,
        side_filter="production default zero: both physical shell surfaces",
        source_files=source_before,
        planar_integration_requested=args.planar_integration,
        planar_integration="radial" if entry.planar.exact_radial else "gauss",
        sample_count_unit=entry.planar.sample_count_unit,
        cooperative_anchor_requested=args.cooperative_anchor,
        anchor_mode=anchor_mode,
        original_reference_anchor_mode="separate_serial",
    )
    error_history, loaded_history, valid_history, route_history = [], [], [], []
    residual_history, wrench_history, input_history, errno_history = [], [], [], []
    source_point_history, origin_accounting_history, legacy_valid_history = [], [], []
    scatter_witnesses = {}
    witnesses, first_loaded_saved = {}, False
    counters_total = np.zeros((len(ROUTE_COUNTERS), args.envs), dtype=np.int64)
    invalid_total = np.zeros((3, args.envs), dtype=np.int64)
    all_numeric, input_unchanged, route_accounted, rigid_unchanged, step_error = True, True, True, True, None
    original_limits = budgets["original_budgets"]
    limits = np.array(
        [
            original_limits["worst_weighted_tensor_field"],
            original_limits["worst_sampled_peak"],
            original_limits["N"],
            original_limits["M"],
            original_limits["Q"],
        ]
    )
    target_applicable = args.subdivisions in (8, 16, 32)
    target_reference_key = "k16" if args.subdivisions == 32 else f"k{args.subdivisions}"
    target = budgets["CPU_demonstrated_fidelity_targets"][target_reference_key] if target_applicable else None
    fidelity_limits = (
        np.array([target["worst_field"], target["worst_peak"], target["N"], target["M"], target["Q"]])
        if target_applicable
        else None
    )
    result["CPU_fidelity_target_status"] = "applicable" if target_applicable else "not_applicable"
    result["CPU_fidelity_target_reference_key"] = target_reference_key if target_applicable else None
    result["CPU_fidelity_target_inherited"] = args.subdivisions == 32
    result["CPU_fidelity_target_limits"] = target
    try:
        for step in range(args.steps):
            counts_before = _route_counts(api, entry.planar.state)
            invalid_before = np.stack(
                [api.qd_to_numpy(state.invalid_steps, transpose=True) for state in (entry.state, low.state, high.state)]
            )
            tick = workload.tick
            reset = (tick > workload.delays) & ((tick - workload.delays) % 600 == 0)
            workload.step()
            solver.check_errno()
            physical = _physical_inputs(api, entry.contacts, entry.omega)
            candidate = api.native_outputs(api, entry.model, entry.state)
            reference_tick = workload.tick
            poses_before = [
                api.qd_to_numpy(value, transpose=True)
                for value in (solver.dyn_state.links.pos, solver.dyn_state.links.quat)
            ]
            original_low, original_high = low.recover(), high.recover()
            poses_after = [
                api.qd_to_numpy(value, transpose=True)
                for value in (solver.dyn_state.links.pos, solver.dyn_state.links.quat)
            ]
            same_rigid = workload.tick == reference_tick and all(
                np.array_equal(first, second, equal_nan=True) for first, second in zip(poses_before, poses_after)
            )
            rigid_unchanged &= same_rigid
            copied = [_physical_inputs(api, route.contacts, route.omega) for route in (low, high)]
            same_inputs = all(
                np.array_equal(values[name], value, equal_nan=True)
                for values in copied
                for name, value in physical.items()
            )
            input_unchanged &= same_inputs
            loaded_contacts = physical["valid"] & (np.linalg.norm(physical["force"], axis=-1) > 1e-30)
            loaded = loaded_contacts.any(axis=1)
            runtime, disputed_frame = {}, False
            for label, state, contacts, errno, output, surface, scatter, order, capacity, planar in (
                (
                    "planar",
                    entry.state,
                    entry.contacts,
                    solver._errno,
                    candidate,
                    entry.surface,
                    entry.scatter,
                    args.quadrature,
                    args.tasks_per_env,
                    entry.planar,
                ),
                (
                    "Q4_production",
                    low.state,
                    low.contacts,
                    low.errno,
                    original_low,
                    low.surface,
                    low.scatter,
                    production_reference_quadrature,
                    args.reference_tasks_per_env,
                    None,
                ),
                (
                    "Q4_resolved",
                    high.state,
                    high.contacts,
                    high.errno,
                    original_high,
                    high.surface,
                    high.scatter,
                    args.reference_quadrature,
                    args.reference_tasks_per_env,
                    None,
                ),
            ):
                max_sources = int(loaded_contacts.sum(axis=1).max(initial=0))
                # Original Q4: four sectors*q^2 samples*four corners. Proxy
                # clipping stores11 angle breakpoints, hence at most10
                # intervals*q^2 samples*four original parent corners.
                contributions = max_sources * surface.info.face_origin.shape[0] * 16 * order * order
                if planar is not None:
                    contributions += capacity * 40 * order * order
                checks = _runtime_checks(
                    api,
                    state,
                    contacts,
                    errno,
                    output,
                    matrix,
                    free,
                    entry.options,
                    physical,
                    vertices,
                    (contributions, float(offset_extent)),
                )
                strict_fail = np.flatnonzero(~checks["strict_origin_diagnostic_passed"])
                if len(strict_fail):
                    disputed_frame = True
                    result["strict_origin_failed_observations"].append(
                        {
                            "step": step,
                            "route": label,
                            "environments": strict_fail.tolist(),
                            "relative_force_errors": [
                                _json_number(value) for value in checks["force_error"][strict_fail]
                            ],
                            "relative_body_origin_moment_errors": [
                                _json_number(value) for value in checks["moment_error"][strict_fail]
                            ],
                        }
                    )
                disputed = np.flatnonzero(~checks["valid"] | ~checks["strict_origin_diagnostic_passed"])
                if len(disputed):
                    disputed_frame = True
                    if label not in scatter_witnesses:
                        scatter_witnesses[label] = SourceScatterWitness(api, entry, surface, capacity, planar)
                    witness_valid, summary = _source_witness(
                        api,
                        scatter_witnesses[label],
                        contacts,
                        scatter,
                        physical,
                        output,
                        disputed,
                        vertices,
                        args.output / f"wrench-{step:04d}-{label}",
                    )
                    checks["independent_scatter_witness_valid"] = witness_valid
                    checks["valid"] &= witness_valid
                    result["wrench_witnesses"].append({"step": step, "route": label, **summary})
                else:
                    checks["independent_scatter_witness_valid"] = np.ones(args.envs, dtype=bool)
                if not checks["valid"].all():
                    result["numerical_failed_observations"].append(
                        {
                            "step": step,
                            "route": label,
                            "environments": np.flatnonzero(~checks["valid"]).tolist(),
                            "base_valid": checks["base_valid"].tolist(),
                            "physical_input_valid": checks["physical_input_valid"].tolist(),
                            "finite_output_valid": checks["finite_output_valid"].tolist(),
                            "contact_status_valid": checks["contact_status_valid"].tolist(),
                            "elastic_residual_valid": checks["elastic_residual_valid"].tolist(),
                            "source_point_valid": checks["source_point_valid"].tolist(),
                            "assembled_origin_accounting_valid": checks["assembled_origin_accounting_valid"].tolist(),
                            "independent_scatter_witness_valid": checks["independent_scatter_witness_valid"].tolist(),
                            "errno": checks["errno"].tolist(),
                        }
                    )
                runtime[label] = checks
            if disputed_frame:
                poses_after_witness = [
                    api.qd_to_numpy(value, transpose=True)
                    for value in (solver.dyn_state.links.pos, solver.dyn_state.links.quat)
                ]
                same_rigid &= workload.tick == reference_tick and all(
                    np.array_equal(first, second, equal_nan=True)
                    for first, second in zip(poses_before, poses_after_witness)
                )
                rigid_unchanged &= same_rigid
                physical_after_witness = _physical_inputs(api, entry.contacts, entry.omega)
                same_inputs &= all(
                    np.array_equal(physical_after_witness[name], value, equal_nan=True)
                    for name, value in physical.items()
                )
                input_unchanged &= same_inputs
            valid = np.stack([runtime[label]["valid"] for label in ("planar", "Q4_production", "Q4_resolved")])
            all_numeric &= bool(
                valid.all() and all(value["unloaded_raw_force_exact_zero"] for value in runtime.values())
            )
            errors = np.stack(
                (
                    _errors(api, candidate, original_high, weights),
                    _errors(api, original_low, original_high, weights),
                    _errors(api, candidate, original_low, weights),
                )
            )
            after = _route_counts(api, entry.planar.state)
            delta = np.where(reset[None, :], after, after - counts_before)
            accounting = bool(
                (delta >= 0).all()
                and np.array_equal(delta[0], loaded_contacts.sum(axis=1))
                and np.array_equal(delta[0], delta[1] + delta[2])
            )
            route_accounted &= accounting
            counters_total += delta
            invalid_after = np.stack(
                [api.qd_to_numpy(state.invalid_steps, transpose=True) for state in (entry.state, low.state, high.state)]
            )
            invalid_delta = invalid_after - invalid_before
            invalid_total += invalid_delta
            error_history.append(errors)
            loaded_history.append(loaded)
            valid_history.append(valid)
            route_history.append(delta)
            residual_history.append(
                np.stack([runtime[label]["relative_residual"] for label in ("planar", "Q4_production", "Q4_resolved")])
            )
            wrench_history.append(
                np.stack(
                    [
                        [runtime[label][key] for key in ("force_error", "moment_error")]
                        for label in ("planar", "Q4_production", "Q4_resolved")
                    ]
                )
            )
            input_history.append(same_inputs)
            errno_history.append(
                np.stack([runtime[label]["errno"] for label in ("planar", "Q4_production", "Q4_resolved")])
            )
            source_point_history.append(
                np.stack(
                    [
                        [
                            runtime[label][key].max(axis=1, initial=0.0)
                            for key in ("source_relative_force_error", "source_relative_point_moment_error")
                        ]
                        for label in ("planar", "Q4_production", "Q4_resolved")
                    ]
                )
            )
            origin_accounting_history.append(
                np.stack(
                    [
                        [
                            runtime[label][key]
                            for key in (
                                "assembled_origin_absolute_force_error",
                                "assembled_origin_force_bound",
                                "assembled_origin_absolute_moment_error",
                                "assembled_origin_moment_bound",
                            )
                        ]
                        for label in ("planar", "Q4_production", "Q4_resolved")
                    ]
                )
            )
            legacy_valid_history.append(
                np.stack(
                    [
                        runtime[label]["legacy_valid_with_strict_origin"]
                        for label in ("planar", "Q4_production", "Q4_resolved")
                    ]
                )
            )
            routing = {
                "fallback": api.qd_to_numpy(entry.planar.state.fallback, transpose=True),
                "reason": api.qd_to_numpy(entry.planar.state.reason, transpose=True),
                "eligible": api.qd_to_numpy(entry.planar.state.eligible, transpose=True),
                "task_count": api.qd_to_numpy(entry.planar.state.task_count, transpose=True),
                "counter_delta": delta,
            }
            summaries = {}
            for index, name in enumerate(COMPARISONS):
                maximum = errors[index][:, loaded].max(axis=1) if loaded.any() else np.zeros(len(METRICS))
                summaries[name] = {
                    metric: float(value) if np.isfinite(value) else None for metric, value in zip(METRICS, maximum)
                }
            current_errors = errors[0][:, loaded]
            spatial_passed = bool(np.isfinite(current_errors).all() and (current_errors <= limits[:, None]).all())
            fidelity_passed = (
                bool(np.isfinite(current_errors).all() and (current_errors <= fidelity_limits[:, None]).all())
                if target_applicable
                else None
            )
            row = {
                "step": step,
                "tick": tick,
                "loaded_contacts": int(loaded_contacts.sum()),
                "loaded_environments": int(loaded.sum()),
                "numerical_passed": bool(valid.all()),
                "spatial_passed": spatial_passed,
                "CPU_fidelity_target_passed": fidelity_passed,
                "CPU_fidelity_target_status": ("passed" if fidelity_passed else "failed")
                if target_applicable
                else "not_applicable",
                "strict_origin_diagnostic_passed": all(
                    value["strict_origin_diagnostic_passed"].all() for value in runtime.values()
                ),
                "source_point_gates_passed": all(value["source_point_valid"].all() for value in runtime.values()),
                "physical_inputs_finite": all(value["physical_input_valid"].all() for value in runtime.values()),
                "assembled_origin_accounting_passed": all(
                    value["assembled_origin_accounting_valid"].all() for value in runtime.values()
                ),
                "physical_inputs_exact": bool(same_inputs),
                "route_counters_accounted": accounting,
                "counter_delta_totals": dict(zip(ROUTE_COUNTERS, delta.sum(axis=1).tolist())),
                "rigid_scene_exact_during_references": bool(same_rigid),
                "invalid_environment_step_deltas": invalid_delta.sum(axis=1).tolist(),
                "invalid_environments": {
                    label: np.flatnonzero(~value["valid"]).tolist() for label, value in runtime.items()
                },
                "errors": summaries,
                "controller_reset_environments": np.flatnonzero(reset).tolist(),
            }
            result["rows"].append(row)
            if step in args.field_steps or disputed_frame or (loaded.any() and not first_loaded_saved):
                api.np.savez_compressed(
                    args.output / f"field-{step:04d}.npz",
                    **_keyframe(step, physical, candidate, original_low, original_high, runtime, routing),
                )
                first_loaded_saved |= bool(loaded.any())
            if loaded.any():
                for index, label in ((0, "field"), (1, "peak")):
                    value = float(np.max(errors[0, index, loaded]))
                    if np.isfinite(value) and (label not in witnesses or value > witnesses[label]["value"]):
                        witnesses[label] = {
                            "step": step,
                            "value": value,
                            "arrays": _keyframe(
                                step, physical, candidate, original_low, original_high, runtime, routing
                            ),
                        }
            if step % 25 == 0 or step + 1 == args.steps:
                api.write_json(args.output / "result.json", result)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loaded": int(loaded.sum()),
                            "numeric": bool(valid.all()),
                            "worst_field": summaries[COMPARISONS[0]]["field"],
                            "fallback_contacts": int(delta[2].sum()),
                        }
                    ),
                    flush=True,
                )
    except Exception as error:  # noqa: BLE001 - Preserve failed observations before the deferred re-raise
        step_error = error
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "completed_observations": len(result["rows"]),
        }
        (args.output / "failure-traceback.txt").write_text(traceback.format_exc())
        result["terminal_invalid_counters"] = {
            label: api.qd_to_numpy(state.invalid_steps, transpose=True).tolist()
            for label, state in (("planar", entry.state), ("Q4_production", low.state), ("Q4_resolved", high.state))
        }
        result["terminal_errno"] = {
            label: api.qd_to_numpy(errno, transpose=True).tolist()
            for label, errno in (("planar", solver._errno), ("Q4_production", low.errno), ("Q4_resolved", high.errno))
        }
        result["terminal_planar_route_counters"] = _route_counts(api, entry.planar.state).tolist()
    if error_history:
        error_array, loaded_array = np.stack(error_history), np.stack(loaded_history)
        api.np.savez_compressed(
            args.output / "environment-metrics.npz",
            errors=error_array,
            loaded=loaded_array,
            valid=np.stack(valid_history),
            route_counter_deltas=np.stack(route_history),
            relative_unscaled_residual=np.stack(residual_history),
            original_wrench_errors=np.stack(wrench_history),
            body_origin_wrench_errors=np.stack(wrench_history),
            maximum_source_point_relative_errors=np.stack(source_point_history),
            assembled_origin_errors_and_bounds=np.stack(origin_accounting_history),
            legacy_valid_with_strict_origin=np.stack(legacy_valid_history),
            physical_inputs_exact=np.array(input_history),
            errno=np.stack(errno_history),
        )
        selected = error_array[:, 0, 0, :][loaded_array]
        mean_field = float(selected.mean()) if len(selected) and np.isfinite(selected).all() else None
        result["mean_loaded_field_error"] = mean_field
        result["loaded_environment_observations"] = int(loaded_array.sum())
        comparison_summaries = {}
        for index, name in enumerate(COMPARISONS):
            maximum, nonfinite = {}, {}
            for metric, key in enumerate(METRICS):
                values = error_array[:, index, metric, :][loaded_array]
                finite = np.isfinite(values)
                nonfinite[key] = int((~finite).sum())
                maximum[key] = float(values.max()) if len(values) and finite.all() else None
            fields = error_array[:, index, 0, :][loaded_array]
            comparison_summaries[name] = {
                "maximum_loaded_errors": maximum,
                "nonfinite_loaded_errors": nonfinite,
                "mean_loaded_field_error": float(fields.mean()) if len(fields) and np.isfinite(fields).all() else None,
            }
        result["comparison_summaries"] = comparison_summaries
        result["metrics_layout"] = {
            "errors": "(step,comparison,metric,environment)",
            "comparisons": COMPARISONS,
            "metrics": METRICS,
            "valid": "(step,route=(planar,Q4-production,Q4-resolved),environment)",
            "route_counters": ROUTE_COUNTERS,
            "original_wrench_errors": "Legacy name: body-origin force/moment residual divided by sum(|F|) and sum(|F|r). Retained unchanged as diagnostic; not a source-point moment.",
            "maximum_source_point_relative_errors": "(step,route,metric=(reported_force/|F|,reported_application_point_moment/(|F|r)),environment); maximum over loaded contacts",
            "assembled_origin_errors_and_bounds": "(step,route,metric=(absolute_force_error,force_bound,absolute_body_origin_moment_error,moment_bound),environment)",
        }
        result["per_casewise_spatial_budgets_passed"] = all(row["spatial_passed"] for row in result["rows"])
        result["per_casewise_CPU_fidelity_targets_passed"] = (
            all(row["CPU_fidelity_target_passed"] for row in result["rows"]) if target_applicable else None
        )
        result["CPU_fidelity_target_status"] = (
            ("passed" if result["per_casewise_CPU_fidelity_targets_passed"] else "failed")
            if target_applicable
            else "not_applicable"
        )
        result["legacy_numerical_with_strict_origin_passed"] = bool(np.stack(legacy_valid_history).all())
        result["strict_origin_diagnostic_passed"] = all(
            row["strict_origin_diagnostic_passed"] for row in result["rows"]
        )
        result["source_point_gates_passed"] = all(row["source_point_gates_passed"] for row in result["rows"])
        result["assembled_origin_accounting_passed"] = all(
            row["assembled_origin_accounting_passed"] for row in result["rows"]
        )
        result["mean_field_budget_passed"] = (
            mean_field is not None and mean_field <= original_limits["mean_weighted_tensor_field"]
        )
    for label, witness in witnesses.items():
        api.np.savez_compressed(args.output / f"worst-{label}-field.npz", **witness["arrays"])
    result["worst_witnesses"] = {
        label: {"step": witness["step"], "relative_error": witness["value"]} for label, witness in witnesses.items()
    }
    result["route_counter_totals"] = dict(zip(ROUTE_COUNTERS, counters_total.sum(axis=1).tolist()))
    result["invalid_environment_steps_by_route"] = dict(
        zip(("planar", "Q4_production", "Q4_resolved"), invalid_total.sum(axis=1).tolist())
    )
    if args.lifecycle and step_error is None:
        try:
            result["lifecycle"] = _lifecycle(api, workload, entry)
        except Exception as error:  # noqa: BLE001 - Preserve failed lifecycle evidence before the deferred re-raise
            step_error = error
            result["lifecycle"] = {
                "performed": True,
                "passed": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
    else:
        result["lifecycle"] = {
            "performed": False,
            "passed": False,
            "reason": "disabled" if not args.lifecycle else "primary trajectory raised",
        }
    source_after = _source_identity(api)
    inputs_unchanged = (
        api.file_id(args.conditions) == result["conditions"]
        and api.file_id(args.budgets) == result["budget_input"]
        and api.file_id(Path(entry.options.mesh)) == result["mesh"]
    )
    result.update(
        source_files_after=source_after,
        source_unchanged=source_before == source_after,
        complete=step_error is None and len(result["rows"]) == args.steps,
        numerical_passed=bool(all_numeric and len(result["rows"]) == args.steps),
        physical_inputs_exact=bool(input_unchanged),
        route_counters_accounted=bool(route_accounted),
        rigid_scene_exact_during_references=bool(rigid_unchanged),
        input_files_unchanged=bool(inputs_unchanged),
        seconds_including_intrusive_audit=time.perf_counter() - started,
    )
    result["passed"] = bool(
        result["complete"]
        and result["numerical_passed"]
        and result["physical_inputs_exact"]
        and result["route_counters_accounted"]
        and result["source_unchanged"]
        and result["rigid_scene_exact_during_references"]
        and result["input_files_unchanged"]
        and result.get("mean_field_budget_passed", False)
        and result.get("per_casewise_spatial_budgets_passed", False)
        and (not target_applicable or result.get("per_casewise_CPU_fidelity_targets_passed", False))
        and (not args.lifecycle or result["lifecycle"]["passed"])
    )
    result["full_trajectory_qualified"] = result["passed"] and args.steps >= 600
    api.write_json(args.output / "result.json", result)
    if step_error is not None:
        raise step_error
    if not result["passed"]:
        raise ArithmeticError(
            "Production planar DPR trajectory exceeded its original numerical/spatial or frozen CPU-fidelity contracts."
        )


if __name__ == "__main__":
    main()
