"""Measure frozen volume-DPR rule errors and empirical admission scores.

Run on Slurm rtx-mid/image1_26. Physical contacts and original anchors are
replayed without rigid advancement. Rejected rules remain recorded negatives.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from itertools import permutations
from pathlib import Path
from typing import Any

from .artifacts import file_id, runtime_identity, write_json

IS_SLURM = (
    bool(os.environ.get("SLURM_JOB_ID"))
    and os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid"
    and os.environ.get("GENESIS_IMAGE_VER") == "1_26"
)
RULES = ("edge16", "angle8", "angle16", "angle32", "angle64", "angle128")

# Scientific imports and kernel registration occur only inside the allocated job.
if IS_SLURM:
    import numpy as np
    import quadrants as qd
    import scipy.linalg

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.contact import (
        StressContactState,
        create_contacts,
        func_anchor_source,
        func_contact_sample,
        func_contact_section,
        func_force_frame,
        func_hertz_disk_moments,
        kernel_anchor,
    )
    from genesis.engine.solvers.rigid.stress.model import StressModel
    from genesis.engine.solvers.rigid.stress.scatter import (
        StressScatterWorkspace,
        create_scatter_workspace,
        kernel_pressure_moments,
        kernel_scatter_faces,
    )
    from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
    from genesis.utils.array_class import V_MAT, V_VEC, V
    from genesis.utils.misc import qd_to_numpy
    from research.rigid_stress.field_cpu import stress_field

    from .check_native_atlas import cpu_operator

    @qd.kernel
    def kernel_prepare_frozen(source_epsilon: float, contacts: StressContactState, surface: StressSurfaceInfo):
        for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
            is_loaded = func_anchor_source(i_c, i_b, source_epsilon, contacts)
            if is_loaded:
                i_f = contacts.anchor_face[i_c, i_b]
                if i_f < 0 or i_f >= surface.face_origin.shape[0]:
                    contacts.status[i_c, i_b] = 2
                else:
                    edges = surface.face_edges[i_f]
                    diameter = qd.max(edges[0, :].norm(), edges[1, :].norm(), (edges[1, :] - edges[0, :]).norm())
                    contacts.is_refined[i_c, i_b] = (
                        surface.angular_quadrature[None]
                        or contacts.radius[i_c, i_b] * surface.gauss.shape[0] < diameter
                    )

    @qd.kernel
    def kernel_affine_flags(affine: qd.Tensor, contacts: StressContactState, workspace: StressScatterWorkspace):
        for i_c, i_b in qd.ndrange(affine.shape[0], affine.shape[1]):
            affine[i_c, i_b] = False
        for i_pair in range(qd.min(contacts.active_count[None], workspace.affine_pressure.shape[0])):
            pair = contacts.active_pairs[i_pair]
            affine[pair[0], pair[1]] = workspace.affine_pressure[i_pair]

    @qd.kernel
    def kernel_shape_moments(
        loads: qd.Tensor,
        gradients: qd.Tensor,
        grams: qd.Tensor,
        hessians: qd.Tensor,
        statistics: qd.Tensor,
        affine: qd.Tensor,
        contacts: StressContactState,
        surface: StressSurfaceInfo,
        force_fan: bool,
    ):
        for i_f, i_c, i_b in qd.ndrange(loads.shape[0], loads.shape[1], loads.shape[2]):
            load = qd.Vector.zero(gs.qd_float, 6)
            gradient = qd.Vector.zero(gs.qd_float, 3)
            gram = qd.Matrix.zero(gs.qd_float, 3, 3)
            hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
            stats = qd.Vector.zero(gs.qd_float, 8)
            if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 1e-30:
                center, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
                separation = qd.max(surface.face_bounds_low[i_f] - center, 0.0)
                separation += qd.max(center - surface.face_bounds_high[i_f], 0.0)
                if separation.norm_sqr() < radius * radius:
                    first, second = func_force_frame(contacts.force[i_c, i_b])
                    is_fan = force_fan or contacts.is_refined[i_c, i_b]
                    closest = qd.Vector.zero(gs.qd_float, 3)
                    remaining, is_contained = gs.qd_float(0.0), False
                    n_q = surface.shape.shape[0]
                    start, count = i_f * n_q, n_q
                    if is_fan:
                        closest, remaining, is_contained = func_contact_section(i_f, center, radius, surface)
                        start = surface.positions.shape[0] + i_f * 3 * n_q
                        count = qd.select(remaining > 0.0, qd.select(is_contained, n_q, 3 * n_q), 0)
                    is_analytic = is_fan and is_contained and affine[i_c, i_b]
                    if is_analytic:
                        gram, moments = func_hertz_disk_moments(
                            i_f, closest, remaining, center, radius, first, second, surface
                        )
                        load = moments @ contacts.coefficient[i_c, i_b]
                        gradient = gram @ contacts.coefficient[i_c, i_b]
                        hessian = gram
                        stats[0], stats[4], stats[7] = gram[0, 0], gradient[0], 1.0
                    else:
                        for i_q in range(count):
                            coordinates, weight, _position, shape, _face = func_contact_sample(
                                start + i_q, i_c, i_b, first, second, contacts, surface
                            )
                            raw_profile = coordinates.dot(contacts.coefficient[i_c, i_b])
                            is_finite = weight >= 0.0 and weight < float("inf")
                            is_finite = is_finite and raw_profile > -float("inf") and raw_profile < float("inf")
                            for a in qd.static(range(3)):
                                is_finite = is_finite and coordinates[a] > -float("inf")
                                is_finite = is_finite and coordinates[a] < float("inf")
                            stats[6] += gs.qd_float(not is_finite)
                            if is_finite and weight > 0.0:
                                profile = qd.max(0.0, raw_profile)
                                fraction = weight * profile
                                load += fraction * shape
                                gradient += fraction * coordinates
                                gram += weight * coordinates.outer_product(coordinates)
                                if profile > 0.0:
                                    hessian += weight * coordinates.outer_product(coordinates)
                                stats[0] += weight
                                stats[1] += weight * weight
                                stats[2] += 1.0
                                stats[3] += gs.qd_float(profile > 0.0)
                                stats[4] += fraction
                                stats[5] += fraction * fraction
            loads[i_f, i_c, i_b] = load
            gradients[i_f, i_c, i_b] = gradient
            grams[i_f, i_c, i_b] = gram
            hessians[i_f, i_c, i_b] = hessian
            statistics[i_f, i_c, i_b] = stats


@dataclasses.dataclass(frozen=True)
class FrozenFrame:
    path: Path
    name: str
    environment_indices: Any
    valid: Any
    position: Any
    force: Any
    normal: Any
    friction: Any
    radius: Any
    radius_status: Any
    center: Any
    anchor_face: Any
    omega: Any
    frame_position: Any
    frame_quaternion: Any
    is_complete_mechanical_input: bool


def voxel_mesh(shape, cells, scale, path):
    """Write an affine tetrahedral volume from the established six-tetrahedra voxel construction."""
    steps = np.eye(3, dtype=gs.np_int)[np.array(list(permutations(range(3))))]
    offsets = np.concatenate((np.zeros((6, 1, 3), dtype=gs.np_int), steps.cumsum(axis=1)), axis=1)
    coordinates = np.array(cells)[:, None, None, :] + offsets
    grid_shape = tuple(size + 1 for size in shape)
    tetrahedra = np.ravel_multi_index(np.moveaxis(coordinates, -1, 0), grid_shape).reshape((-1, 4))
    used, inverse = np.unique(tetrahedra, return_inverse=True)
    vertices = np.indices(grid_shape).reshape((3, -1)).T[used] * scale
    tetrahedra = inverse.reshape((-1, 4))
    local_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    unique, counts = np.unique(np.sort(tetrahedra[:, local_faces].reshape((-1, 3)), axis=1), axis=0, return_counts=True)
    surface = unique[counts == 1]
    surface[::2] = surface[::2, ::-1]
    np.savez_compressed(path, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)


def author_contacts(recipe, base, model, output):
    """Freeze one new manual physical source and its original native anchor before comparing integration rules."""
    contacts = create_contacts(1, 1, base["fixture_scale_m"] * recipe["radius_over_cell"], hertz_profile=True)
    contacts.position.from_numpy((np.array(recipe["point_cell_units"]) * base["fixture_scale_m"])[None, None])
    contacts.force.from_numpy(np.array(recipe["force_N"])[None, None])
    contacts.normal.from_numpy(np.array(recipe["inward_normal"], dtype=gs.np_float)[None, None])
    contacts.friction.fill(recipe["friction"])
    contacts.valid.fill(True)
    contacts.frame_position.from_numpy(np.array(base["pose_position_m"], dtype=gs.np_float)[None])
    contacts.frame_quaternion.from_numpy(np.array(base["pose_quaternion_wxyz"], dtype=gs.np_float)[None])
    surface = StressSurface(16, model.load_info)
    kernel_anchor(float(np.finfo(gs.np_float).eps), contacts, surface.info, cooperative=True)
    if np.any(qd_to_numpy(contacts.status, transpose=True)):
        raise ValueError("An authored source violates the original source/anchor controls.")
    physical = {
        key: qd_to_numpy(vars(contacts)[key], transpose=True)
        for key in (
            "valid",
            "position",
            "force",
            "normal",
            "friction",
            "radius",
            "radius_status",
            "center",
            "anchor_face",
            "frame_position",
            "frame_quaternion",
        )
    }
    physical["body_omega"] = np.array(base["omega_rad_per_s"], dtype=gs.np_float)[None]
    path = output / "authored-input.npz"
    np.savez_compressed(path, **physical)
    write_json(
        path.with_suffix(".json"),
        {
            "schema": "genesis.dpr-fast-authored-physical-input.v1",
            "id": recipe["id"],
            "recipe": recipe,
            "base": base,
            "physical_input": file_id(path),
            "stress_options": model.options.model_dump(mode="json"),
            "scope": "New manual finite-support physical fixture with known zero omega; not an inflation repair or altered historical input.",
        },
    )
    return path


def finite_json(value):
    """Encode nonfinite diagnostics explicitly without creating accepted zero values."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return {"nonfinite": str(value)}
    return value


def load_frame(path, name, allow_conditional_zero_omega, environments):
    with np.load(path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    prefix = "contact_" if "contact_valid" in saved else ""
    valid = saved[prefix + "valid"]
    position = saved[prefix + ("point" if prefix else "position")]
    if valid.ndim != 2 or valid.dtype != bool:
        raise ValueError("Frozen contacts require environment-first Boolean valid[B,C].")
    shape = valid.shape
    scalars = [saved[prefix + key] for key in ("friction", "radius", "radius_status", "anchor_face")]
    vectors = [position, *(saved[prefix + key] for key in ("force", "normal", "center"))]
    if any(value.shape != shape for value in scalars) or any(value.shape != (*shape, 3) for value in vectors):
        raise ValueError("Frozen source arrays do not share the declared environment/contact dimensions.")
    if not all(np.isfinite(value[valid]).all() for value in [*scalars, *vectors]):
        raise ValueError("A valid physical source contains nonfinite input.")
    if not np.issubdtype(saved[prefix + "anchor_face"].dtype, np.integer):
        raise ValueError("Original anchor IDs must be integer topology inputs.")
    omega_key = next((key for key in ("body_omega", "angular_velocity", "omega") if key in saved), None)
    is_complete = omega_key is not None
    if not is_complete and not allow_conditional_zero_omega:
        raise ValueError(
            "Missing original omega; use --allow-conditional-zero-omega only for a labelled conditional audit."
        )
    omega = saved[omega_key] if is_complete else np.zeros((shape[0], 3))
    if omega.shape != (shape[0], 3) or not np.isfinite(omega).all():
        raise ValueError("Original body-frame omega must be finite[B,3].")
    indices = list(range(shape[0])) if environments is None else environments
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= shape[0]:
        raise ValueError("Select unique existing environments and retain all their contact slots.")
    frame_position = saved.get(prefix + "frame_position")
    frame_quaternion = saved.get(prefix + "frame_quaternion")
    return FrozenFrame(
        path,
        name,
        np.array(indices),
        valid[indices],
        position[indices],
        saved[prefix + "force"][indices],
        saved[prefix + "normal"][indices],
        saved[prefix + "friction"][indices],
        saved[prefix + "radius"][indices],
        saved[prefix + "radius_status"][indices],
        saved[prefix + "center"][indices],
        saved[prefix + "anchor_face"][indices],
        omega[indices],
        frame_position[indices] if frame_position is not None else None,
        frame_quaternion[indices] if frame_quaternion is not None else None,
        is_complete,
    )


def prepare_contacts(frame):
    contacts = create_contacts(frame.valid.shape[1], frame.valid.shape[0], 0.001, hertz_profile=True)
    for field in dataclasses.fields(contacts):
        vars(contacts)[field.name].fill(0)
    contacts.hertz_profile.fill(True)
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
        vars(contacts)[name].from_numpy(np.ascontiguousarray(np.swapaxes(vars(frame)[name], 0, 1)))
    for name in ("frame_position", "frame_quaternion"):
        value = vars(frame)[name]
        if value is not None:
            vars(contacts)[name].from_numpy(np.ascontiguousarray(value))
    return contacts


def contact_arrays(contacts):
    return {
        field.name: qd_to_numpy(vars(contacts)[field.name], transpose=True, copy=True)
        for field in dataclasses.fields(contacts)
    }


def shape_witness(contacts, surface, workspace, force_fan):
    shape = (surface.face_origin.shape[0], contacts.valid.shape[0], contacts.valid.shape[1])
    loads = V_VEC(6, dtype=gs.qd_float, shape=shape)
    gradients = V_VEC(3, dtype=gs.qd_float, shape=shape)
    grams = V_MAT(3, 3, dtype=gs.qd_float, shape=shape)
    hessians = V_MAT(3, 3, dtype=gs.qd_float, shape=shape)
    statistics = V_VEC(8, dtype=gs.qd_float, shape=shape)
    affine = V(dtype=gs.qd_bool, shape=contacts.valid.shape)
    kernel_affine_flags(affine, contacts, workspace)
    kernel_shape_moments(loads, gradients, grams, hessians, statistics, affine, contacts, surface, force_fan)
    return {
        name: np.swapaxes(qd_to_numpy(tensor, transpose=True), 1, 2)
        for name, tensor in (
            ("face_load_fractions", loads),
            ("face_gradient", gradients),
            ("face_gram", grams),
            ("face_active_hessian", hessians),
            ("face_statistics", statistics),
        )
    }


def tensor_matrix(tensor):
    matrix = np.zeros((*tensor.shape[:-1], 3, 3))
    matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2] = np.moveaxis(tensor[..., :3], -1, 0)
    matrix[..., 0, 1] = matrix[..., 1, 0] = tensor[..., 3]
    matrix[..., 0, 2] = matrix[..., 2, 0] = tensor[..., 4]
    matrix[..., 1, 2] = matrix[..., 2, 1] = tensor[..., 5]
    return matrix


def legacy_error(actual, expected):
    axes = tuple(range(1, actual.ndim))
    difference = np.sqrt(np.sum((actual - expected) ** 2, axis=axes))
    scale = np.sqrt(np.sum(expected**2, axis=axes))
    return difference / np.maximum(scale, 1e-30)


def normalized_admission(native, checker):
    """Compare per-face pressure measures after one checker mass normalization per source."""
    first = checker["face_gradient"].sum(axis=2)
    mass = first[..., 0]
    is_valid = np.isfinite(mass) & (mass > 0)
    is_valid &= checker["face_statistics"][..., 6].sum(axis=2) == 0
    is_valid &= np.isfinite(checker["face_load_fractions"]).all(axis=(-2, -1))
    is_valid &= np.isfinite(native["face_load_fractions"]).all(axis=(-2, -1))
    denominator = np.where(is_valid, mass, 1.0)
    normalized = checker["face_load_fractions"] / denominator[..., None, None]
    gradient = first / denominator[..., None] - np.array([1.0, 0.0, 0.0])
    difference = normalized - native["face_load_fractions"]
    return {
        "checker_total_pressure_mass": mass,
        "checker_supported_finite": is_valid,
        "normalized_face_load_fractions": normalized,
        "normalized_gradient": gradient,
        "normalized_gradient_norm": np.linalg.norm(gradient, axis=-1),
        "normalized_P2_L1_discrepancy": np.abs(difference).sum(axis=(-2, -1)),
    }


def raw_admission(native, checker):
    """Evaluate the two unnormalized per-source scores without changing array layout."""
    gradient = checker["face_gradient"].sum(axis=2) - np.array([1.0, 0.0, 0.0])
    difference = checker["face_load_fractions"] - native["face_load_fractions"]
    return np.linalg.norm(gradient, axis=-1), np.abs(difference).sum(axis=(-2, -1))


def restore_face_contact_batch_layout(value):
    """Restore the native face/contact/batch/value readback strides without changing logical bytes."""
    axes = (2, 1, 0, *range(3, value.ndim))
    return np.transpose(np.ascontiguousarray(np.transpose(value, axes)), axes)


def diagnose_export_layout(args, source_revision):
    """Locate exact-score discrepancies for contiguous and reconstructed readback layouts."""
    inputs = [file_id(path / "result.json") for path in args.recompute_admission]
    rows = []
    for directory in args.recompute_admission:
        original = json.loads((directory / "result.json").read_text())
        for case in original["rows"]:
            for record in case["rules"]:
                identities = [record[key] for key in ("moment_witness", "admission", "contacts")]
                if identities != [file_id(item["path"]) for item in identities]:
                    raise ValueError("An exported moment/contact artifact changed before layout diagnosis.")
                with np.load(record["moment_witness"]["path"], allow_pickle=False) as saved:
                    native = {key: saved[key] for key in saved.files}
                with np.load(record["admission"]["path"], allow_pickle=False) as saved:
                    checker = {key[6:]: saved[key] for key in saved.files if key.startswith("check_")}
                expected = (
                    np.array(record["independent_gradient_norm"]),
                    np.array(record["local_P2_load_L1_discrepancy"]),
                )
                layout_rows = []
                for layout in ("saved_contiguous", "native_face_contact_batch", "fortran"):
                    if layout == "native_face_contact_batch":
                        current_native = {
                            key: restore_face_contact_batch_layout(value) for key, value in native.items()
                        }
                        current_checker = {
                            key: restore_face_contact_batch_layout(value) for key, value in checker.items()
                        }
                    elif layout == "fortran":
                        current_native = {key: np.asfortranarray(value) for key, value in native.items()}
                        current_checker = {key: np.asfortranarray(value) for key, value in checker.items()}
                    else:
                        current_native, current_checker = native, checker
                    actual = raw_admission(current_native, current_checker)
                    metrics = []
                    for name, observed, reference in zip(("gradient_norm", "P2_L1"), actual, expected):
                        is_equal = observed == reference
                        differences = np.abs(observed - reference)
                        bits_observed, bits_reference = observed.view(np.uint64), reference.view(np.uint64)
                        ulps = np.maximum(bits_observed, bits_reference) - np.minimum(bits_observed, bits_reference)
                        mismatches = []
                        for env, slot in np.argwhere(~is_equal):
                            mismatches.append(
                                {
                                    "environment": int(env),
                                    "original_environment": case["original_environment_indices"][env],
                                    "slot": int(slot),
                                    "recorded": float(reference[env, slot]),
                                    "recomputed": float(observed[env, slot]),
                                    "recorded_hex": float(reference[env, slot]).hex(),
                                    "recomputed_hex": float(observed[env, slot]).hex(),
                                    "absolute_difference": float(differences[env, slot]),
                                    "ulps": int(ulps[env, slot]),
                                }
                            )
                        metrics.append(
                            {
                                "score": name,
                                "exact": bool(is_equal.all()),
                                "max_absolute_difference": float(differences.max(initial=0)),
                                "max_ulps": int(ulps.max(initial=0)),
                                "mismatches": mismatches,
                            }
                        )
                    fields = []
                    for name in ("face_gradient", "face_load_fractions"):
                        value = current_checker[name]
                        logical = value.tobytes(order="C")
                        fields.append(
                            {
                                "name": name,
                                "shape": list(value.shape),
                                "strides": list(value.strides),
                                "C_contiguous": bool(value.flags.c_contiguous),
                                "F_contiguous": bool(value.flags.f_contiguous),
                                "logical_C_bytes_sha256": hashlib.sha256(logical).hexdigest(),
                                "logical_bytes_equal_saved": logical == checker[name].tobytes(order="C"),
                            }
                        )
                    layout_rows.append({"layout": layout, "scores": metrics, "fields": fields})
                rows.append(
                    {
                        "case": case["name"],
                        "rule": record["rule"],
                        "source_result": file_id(directory / "result.json"),
                        "source_artifacts": identities,
                        "layouts": layout_rows,
                    }
                )
    is_unchanged = inputs == [file_id(path / "result.json") for path in args.recompute_admission]
    result = {
        "schema": "genesis.dpr-fast-export-layout-diagnostic.v1",
        "complete": True,
        "passed": is_unchanged,
        "source_revision": source_revision,
        "worker": file_id(Path(__file__)),
        "original_results": inputs,
        "original_results_unchanged": is_unchanged,
        "rows": rows,
        "scope": "Slurm-only diagnostic of exact raw score reproduction. No tolerance relaxation, new v2 acceptance, mechanical replay or original artifact write.",
    }
    write_json(args.output / "result.json", finite_json(result))


def recompute_admission(args, source_revision):
    """Evaluate both empirical methods from immutable exported moments without replaying mechanics."""
    if args.admission_method is None:
        raise ValueError("A predeclared estimator-method identity is required for the v2 postprocess.")
    method = json.loads(args.admission_method.read_text())
    if method["schema"] != "genesis.dpr-fast-empirical-admission-method.v2" or method["scores"]["thresholds"] != {
        "P2_shape_L1_max": 0.001,
        "centroid_gradient_norm_max": 0.001,
    }:
        raise ValueError("The postprocess requires the predeclared v2 formula and unchanged1e-3 thresholds.")
    inputs = [file_id(path / "result.json") for path in args.recompute_admission]
    result = {
        "schema": "genesis.dpr-fast-admission-postprocess.v2",
        "complete": False,
        "passed": False,
        "application_detector_validated": False,
        "primary_timing": False,
        "goal_complete": False,
        "source_revision": source_revision,
        "worker": file_id(Path(__file__)),
        "method": file_id(args.admission_method),
        "budgets": file_id(args.budgets),
        "original_results": inputs,
        "scope": "Slurm-only algebra on completed per-face moments. No physics replay, coefficient re-fit, source change or original result rewrite.",
        "rows": [],
    }
    for directory in args.recompute_admission:
        original = json.loads((directory / "result.json").read_text())
        if original["complete"] is not True or original["passed"] is not True:
            raise ValueError("Only completed trustworthy frozen-rule controls may be postprocessed.")
        for case in original["rows"]:
            case_rows = []
            for record in case["rules"]:
                identities = [record[key] for key in ("moment_witness", "admission", "contacts")]
                if identities != [file_id(item["path"]) for item in identities]:
                    raise ValueError("An exported moment/contact artifact differs from its terminal receipt.")
                with np.load(record["moment_witness"]["path"], allow_pickle=False) as saved:
                    native = {key: saved[key] for key in saved.files}
                with np.load(record["admission"]["path"], allow_pickle=False) as saved:
                    checker = {key[6:]: saved[key] for key in saved.files if key.startswith("check_")}
                with np.load(record["contacts"]["path"], allow_pickle=False) as saved:
                    loaded = saved["valid"] & (np.linalg.norm(saved["force"], axis=-1) > 1e-30)
                    status = saved["status"]
                # NPZ preserves values but serializes these views contiguously; restore their reduction order.
                native = {key: restore_face_contact_batch_layout(value) for key, value in native.items()}
                checker = {key: restore_face_contact_batch_layout(value) for key, value in checker.items()}
                scores = normalized_admission(native, checker)
                is_admitted = scores["checker_supported_finite"] & (status == 0)
                is_admitted &= scores["normalized_gradient_norm"] <= 1e-3
                is_admitted &= scores["normalized_P2_L1_discrepancy"] <= 1e-3
                environment_admitted = np.all(is_admitted | ~loaded, axis=1)
                environment_admitted &= record["source_and_elastic_eligible_per_environment"]
                admitted_list = environment_admitted.tolist()
                raw_gradient, raw_L1 = raw_admission(native, checker)
                raw_match = np.array_equal(raw_gradient, record["independent_gradient_norm"])
                raw_match &= np.array_equal(raw_L1, record["local_P2_load_L1_discrepancy"])
                if not raw_match:
                    raise RuntimeError("The exported raw-v1 scores do not reproduce their original receipt.")
                truth = record["application_passed_per_environment"]
                output = args.output / f"{directory.name}-{case['name']}-{record['rule']}.npz"
                np.savez_compressed(
                    output,
                    **scores,
                    loaded=loaded,
                    empirical_contact_admitted=is_admitted,
                    empirical_environment_admitted=environment_admitted,
                    original_raw_gradient_norm=raw_gradient,
                    original_raw_P2_L1=raw_L1,
                )
                case_rows.append(
                    {
                        "rule": record["rule"],
                        "source_artifacts": identities,
                        "derived_scores": file_id(output),
                        "loaded_contacts": int(loaded.sum()),
                        "checker_mass_min_loaded": float(scores["checker_total_pressure_mass"][loaded].min())
                        if loaded.any()
                        else None,
                        "checker_mass_max_loaded": float(scores["checker_total_pressure_mass"][loaded].max())
                        if loaded.any()
                        else None,
                        "raw_v1_reproduced_exact": bool(raw_match),
                    "raw_reproduction_layout": "native_face_contact_batch_value",
                        "raw_v1_environment_admitted": record["empirical_environment_admitted"],
                        "normalized_gradient_max_loaded": float(
                            scores["normalized_gradient_norm"][loaded].max(initial=0)
                        ),
                        "normalized_P2_L1_max_loaded": float(
                            scores["normalized_P2_L1_discrepancy"][loaded].max(initial=0)
                        ),
                        "v2_environment_admitted": admitted_list,
                        "application_passed_per_environment": truth,
                        "legacy_passed_per_environment": record["legacy_passed_per_environment"],
                        "false_accept_count": sum(
                            admit and actual is False for admit, actual in zip(admitted_list, truth)
                        ),
                        "false_reject_count": sum(
                            not admit and actual is True for admit, actual in zip(admitted_list, truth)
                        ),
                        "converged_but_application_inaccurate_count": record[
                            "converged_but_application_inaccurate_count"
                        ],
                    }
                )
            result["rows"].append(
                {
                    "original_case": case["name"],
                    "original_result": file_id(directory / "result.json"),
                    "whole_frame_qualification": case["whole_frame_qualification"],
                    "rules": case_rows,
                }
            )
    result["original_results_unchanged"] = inputs == [
        file_id(path / "result.json") for path in args.recompute_admission
    ]
    result["converged_bad_rule_detection_exercised"] = any(
        row["converged_but_application_inaccurate_count"] for case in result["rows"] for row in case["rules"]
    )
    result["complete"] = True
    result["passed"] = result["original_results_unchanged"]
    write_json(args.output / "result.json", finite_json(result))
    if not result["passed"]:
        raise RuntimeError("An immutable original result changed during the postprocess.")


def region_errors(actual, expected, weights, masks):
    rows = []
    actual_full, expected_full = actual["tensor_full9"], expected["tensor_full9"]
    reference_peak = expected["von_mises"].reshape((len(masks), -1)).max(axis=1)
    for env, regions in enumerate(masks):
        env_rows = {}
        for name, mask in regions.items():
            count = int(mask.sum())
            if not count:
                env_rows[name] = {"applicability": "empty", "samples": 0, "passed": None}
                continue
            w = weights[mask]
            difference = actual_full[env][mask] - expected_full[env][mask]
            error_rms = np.sqrt(np.sum(w * np.sum(difference**2, axis=(-2, -1))) / w.sum())
            reference_rms = np.sqrt(np.sum(w * np.sum(expected_full[env][mask] ** 2, axis=(-2, -1))) / w.sum())
            reference_region_peak = float(expected["von_mises"][env][mask].max())
            candidate_region_peak = float(actual["von_mises"][env][mask].max())
            floor = max(1.0, 0.01 * reference_peak[env]) if name != "global" else 1.0
            tensor_error = float(error_rms / max(reference_rms, floor))
            peak_delta = candidate_region_peak - reference_region_peak
            peak_error = abs(peak_delta) / max(reference_region_peak, floor)
            underestimate = max(-peak_delta, 0.0) / max(reference_region_peak, floor)
            indices = np.flatnonzero(mask)
            reference_index = int(indices[np.argmax(expected["von_mises"][env][mask])])
            candidate_index = int(indices[np.argmax(actual["von_mises"][env][mask])])
            env_rows[name] = {
                "applicability": "sampled_region",
                "samples": count,
                "tensor_error_RMS_Pa": float(error_rms),
                "reference_tensor_RMS_Pa": float(reference_rms),
                "tensor_relative_L2": tensor_error,
                "reference_peak_Pa": reference_region_peak,
                "candidate_peak_Pa": candidate_region_peak,
                "peak_signed_error_Pa": peak_delta,
                "peak_absolute_error_Pa": abs(peak_delta),
                "peak_bidirectional_relative": peak_error,
                "peak_underestimate_relative": underestimate,
                "normalization_floor_Pa": float(floor),
                "reference_peak_element_corner": list(divmod(reference_index, mask.shape[-1])),
                "candidate_peak_element_corner": list(divmod(candidate_index, mask.shape[-1])),
                "candidate_VM_at_reference_peak_Pa": float(actual["von_mises"][env].reshape(-1)[reference_index]),
                "reference_VM_at_candidate_peak_Pa": float(expected["von_mises"][env].reshape(-1)[candidate_index]),
                "passed": bool(tensor_error <= 0.05 and peak_error <= 0.05 and underestimate <= 0.02),
            }
        rows.append(env_rows)
    return rows


def rule_fields(model, options, frame, rule, directory, tasks_per_env):
    directory.mkdir()
    is_angle = rule.startswith("angle")
    degree = int(rule[5:] if is_angle else rule[4:])
    surface = StressSurface(degree, model.load_info, angular=is_angle)
    contacts = prepare_contacts(frame)
    physical = contact_arrays(contacts)
    state = model.create_state(frame.valid.shape[0], "full")
    workspace = create_scatter_workspace(frame.valid.shape[0], tasks_per_env, reuse_moments=True)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(frame.valid.shape[0],))
    omega.from_numpy(np.ascontiguousarray(frame.omega))
    kernel_prepare_frozen(float(np.finfo(gs.np_float).eps), contacts, surface.info)
    started = time.perf_counter()
    kernel_pressure_moments(contacts, state, model.load_info, surface.info, workspace, cached_bounds=True)
    kernel_scatter_faces(
        contacts, state, model.load_info, surface.info, workspace, cached_bounds=True, reuse_moments=True
    )
    model.recover(omega, state, options, surface_load=True)
    qd.sync()
    elapsed = time.perf_counter() - started
    native = contact_arrays(contacts)
    arrays = {
        name: qd_to_numpy(tensor, transpose=True)
        for name, tensor in (
            ("nodal_force", state.force),
            ("rhs", state.rhs),
            ("displacement", state.displacement),
            ("stress_tensor", state.stress_tensor),
            ("von_mises", state.von_mises),
            ("peak", state.peak),
            ("elastic_valid", state.valid),
            ("residual_squared", state.residual_norm_squared),
            ("rhs_squared", state.rhs_norm_squared),
        )
    }
    arrays["von_mises"] = arrays["von_mises"][..., 0]
    arrays["tensor_full9"] = tensor_matrix(arrays["stress_tensor"])
    loaded = frame.valid & (np.linalg.norm(frame.force, axis=-1) > 1e-30)
    force_norm = np.linalg.norm(frame.force, axis=-1)
    force_error = native["force_error"] / np.maximum(force_norm, 1e-30)
    moment_error = native["moment_error"] / np.maximum(force_norm * frame.radius, 1e-30)
    unchanged = {
        key: physical[key].tobytes() == native[key].tobytes()
        for key in (
            "valid",
            "position",
            "force",
            "normal",
            "friction",
            "radius",
            "radius_status",
            "center",
            "anchor_face",
        )
    }
    source_passed = np.all((native["status"] == 0) | ~frame.valid, axis=1)
    source_passed &= np.all((force_error <= 1e-8) | ~loaded, axis=1)
    source_passed &= np.all((moment_error <= 1e-8) | ~loaded, axis=1)
    source_passed &= np.all(np.isfinite(native["coefficient"]).all(axis=-1) | ~loaded, axis=1)
    residual = np.sqrt(arrays["residual_squared"])
    residual_budget = np.maximum(options.absolute_tolerance, options.tolerance * np.sqrt(arrays["rhs_squared"]))
    finite = np.ones(frame.valid.shape[0], dtype=bool)
    for value in arrays.values():
        finite &= np.isfinite(value).reshape((len(finite), -1)).all(axis=1)
    eligible = source_passed & arrays["elastic_valid"] & (residual <= residual_budget) & finite
    if not all(unchanged.values()):
        raise RuntimeError("A frozen physical source or original anchor changed during the rule replay.")
    np.savez_compressed(directory / "fields.npz", **arrays, omega=frame.omega)
    np.savez_compressed(directory / "contacts.npz", **native)
    write_json(directory / "result.json", {"rule": rule, "complete": False, "fields_saved": True})
    witness = shape_witness(contacts, surface.info, workspace, force_fan=False)
    check_surface = StressSurface(8 if degree != 8 else 16, model.load_info, angular=True)
    independent = shape_witness(contacts, check_surface.info, workspace, force_fan=True)
    delta = independent["face_load_fractions"] - witness["face_load_fractions"]
    local_l1 = np.abs(delta).sum(axis=(-2, -1))
    gradient = independent["face_gradient"].sum(axis=2) - np.array([1.0, 0.0, 0.0])
    gradient_norm = np.linalg.norm(gradient, axis=-1)
    gradient_native = witness["face_gradient"].sum(axis=2) - np.array([1.0, 0.0, 0.0])
    nodal_witness = np.zeros_like(arrays["nodal_force"])
    face_nodes = qd_to_numpy(model.load_info.surface_nodes, transpose=False)
    for local in range(6):
        contribution = (witness["face_load_fractions"][..., local, None] * frame.force[:, :, None, :]).sum(axis=1)
        for env in range(len(nodal_witness)):
            np.add.at(nodal_witness[env], face_nodes[:, local], contribution[env])
    reconstruction_error = legacy_error(nodal_witness, arrays["nodal_force"])
    # Failed fits export their attempted moments without treating discarded loads as accepted.
    witness_gate = (reconstruction_error <= 1e-8) | ~eligible
    np.savez_compressed(directory / "moments.npz", **witness)
    if not witness_gate.all():
        write_json(
            directory / "witness-failure.json", {"moment_reconstruction_relative": reconstruction_error.tolist()}
        )
        raise RuntimeError("The independent per-face P2 moment witness does not reconstruct the accepted native load.")
    empirical_contact = (local_l1 <= 1e-3) & (gradient_norm <= 1e-3) & (native["status"] == 0)
    empirical_contact &= witness["face_statistics"][..., 6].sum(axis=2) == 0
    empirical_contact &= independent["face_statistics"][..., 6].sum(axis=2) == 0
    empirical_environment = np.all(empirical_contact | ~loaded, axis=1) & eligible
    normalized = normalized_admission(witness, independent)
    normalized_contact = normalized["checker_supported_finite"] & (native["status"] == 0)
    normalized_contact &= normalized["normalized_gradient_norm"] <= 1e-3
    normalized_contact &= normalized["normalized_P2_L1_discrepancy"] <= 1e-3
    normalized_environment = np.all(normalized_contact | ~loaded, axis=1) & eligible
    np.savez_compressed(
        directory / "admission.npz",
        **{"check_" + key: value for key, value in independent.items()},
        **{"v2_" + key: value for key, value in normalized.items()},
        local_face_P2_L1_discrepancy=local_l1,
        independent_gradient_norm=gradient_norm,
        native_gradient=gradient_native,
        empirical_contact_admitted=empirical_contact,
        empirical_environment_admitted=empirical_environment,
    )
    report = {
        "rule": rule,
        "quadrature": degree,
        "explicit_fixed_angular_fan": is_angle,
        "fields": file_id(directory / "fields.npz"),
        "contacts": file_id(directory / "contacts.npz"),
        "moment_witness": file_id(directory / "moments.npz"),
        "admission": file_id(directory / "admission.npz"),
        "intrusive_seconds_including_runtime_JIT": elapsed,
        "physical_inputs_bitwise_unchanged": unchanged,
        "source_and_elastic_eligible_per_environment": eligible.tolist(),
        "status": native["status"].tolist(),
        "evaluations": native["evaluations"].tolist(),
        "candidate_count": native["candidate_count"].tolist(),
        "is_refined": native["is_refined"].tolist(),
        "force_error_relative": force_error.tolist(),
        "original_point_moment_error_relative": moment_error.tolist(),
        "K_residual": residual.tolist(),
        "K_residual_budget": residual_budget.tolist(),
        "workspace_actual_count": int(qd_to_numpy(workspace.count, transpose=True)),
        "workspace_capacity": workspace.tasks.shape[0],
        "workspace_overflow_calls": int(qd_to_numpy(workspace.overflow_calls, transpose=True)),
        "native_gradient_norm": np.linalg.norm(gradient_native, axis=-1).tolist(),
        "independent_rule": "angle8" if degree != 8 else "angle16",
        "independent_gradient_norm": gradient_norm.tolist(),
        "local_P2_load_L1_discrepancy": local_l1.tolist(),
        "moment_statistics_per_contact": witness["face_statistics"].sum(axis=2).tolist(),
        "moment_statistics_columns": [
            "raw_weight_sum",
            "raw_weight_squared_sum",
            "positive_weight_samples",
            "positive_active_samples",
            "pressure_mass_sum",
            "pressure_mass_squared_sum",
            "nonfinite_samples",
            "analytic_sections",
        ],
        "moment_reconstruction_relative": reconstruction_error.tolist(),
        "empirical_environment_admitted": empirical_environment.tolist(),
        "mass_normalized_v2_environment_admitted": normalized_environment.tolist(),
        "mass_normalized_v2_gradient_norm": normalized["normalized_gradient_norm"].tolist(),
        "mass_normalized_v2_P2_L1_discrepancy": normalized["normalized_P2_L1_discrepancy"].tolist(),
        "independent_checker_total_mass": normalized["checker_total_pressure_mass"].tolist(),
        "estimator_scope": "Same-coefficient independent-rule3-gradient and original six P2 shape moments; empirical1e-3 screens only. No high-order true-error bound or complete geometric admission certificate is claimed.",
    }
    report["complete"] = True
    write_json(directory / "result.json", finite_json(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--archives", nargs="+", type=Path)
    source_group.add_argument("--recompute-admission", nargs="+", type=Path)
    source_group.add_argument("--manual-control", choices=("convex-edge", "convex-tip", "u-inner-seam", "u-gap"))
    parser.add_argument("--manual-plan", type=Path)
    parser.add_argument("--admission-method", type=Path)
    parser.add_argument("--diagnose-export-layout", action="store_true")
    parser.add_argument("--envs", nargs="+", type=int)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--rules", nargs="+", choices=RULES, default=RULES)
    parser.add_argument("--reference256", action="store_true")
    parser.add_argument("--coarse-controls", action="store_true")
    parser.add_argument("--allow-conditional-zero-omega", action="store_true")
    parser.add_argument("--corpus-role", choices=("diagnostic", "calibration", "holdout"), default="diagnostic")
    parser.add_argument("--tasks-per-env", type=int, default=32)
    parser.add_argument("--young", type=float, default=1e10)
    parser.add_argument("--poisson", type=float, default=0.3)
    parser.add_argument("--density", type=float, default=2000.0)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not IS_SLURM:
        parser.error("Scientific execution requires Slurm rtx-mid/image1_26.")
    if args.archives is not None and (len(args.archives) > 12 or (args.reference256 and len(args.archives) > 6)):
        parser.error("Use at most12 frozen inputs and at most6 q256 reference checks.")
    if "angle128" not in args.rules or len(set(args.rules)) != len(args.rules) or args.tasks_per_env < 1:
        parser.error("Include the angle128 reference, unique rules and a positive workspace capacity.")
    if not args.output.resolve().is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output.")
    budget = json.loads(args.budgets.read_text())
    if budget["schema"] != "genesis.automatic-dpr-application-budgets.v1":
        raise ValueError("Use the predeclared application budget schema.")
    expected_gates = {
        "tensor_relative_L2_max": 0.05,
        "sampled_peak_bidirectional_relative_max": 0.05,
        "sampled_peak_underestimate_relative_max": 0.02,
    }
    if any(budget["application_gates"][key] != expected_gates for key in ("global", "loaded", "remote")):
        raise ValueError("This checker implements the frozen v1 thresholds exactly.")
    if budget["metrics"]["floor_values"] != {
        "absolute_stress_floor_Pa": 1.0,
        "regional_floor_fraction_of_reference_global_peak": 0.01,
    }:
        raise ValueError("This checker implements the frozen v1 denominator floors exactly.")
    if budget["reference"]["stability_limits"] != {
        "tensor_relative_L2_max": 0.01,
        "sampled_peak_bidirectional_relative_max": 0.005,
    }:
        raise ValueError("This checker implements the frozen reference-stability limits exactly.")
    args.output.mkdir(parents=True, exist_ok=False)
    if Path(gs.__file__).resolve().parent.parent != args.git_source.resolve():
        raise RuntimeError("Imported Genesis is not the explicitly declared native source.")
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.recompute_admission is not None:
        if args.diagnose_export_layout:
            diagnose_export_layout(args, source_revision)
        else:
            recompute_admission(args, source_revision)
        return
    if args.manual_control is not None and (args.manual_plan is None or args.coarse_controls):
        parser.error("Authored controls require their frozen --manual-plan and cannot add coarse-rule probes.")
    if args.mesh is None and args.manual_control is None:
        parser.error("Native frozen-rule replay requires --mesh.")
    gs.init(backend=gs.cuda, precision="64", seed=510000, logging_level="warning")
    manual = None
    recipe = None
    if args.manual_control is not None:
        manual = json.loads(args.manual_plan.read_text())
        if manual["schema"] != "genesis.dpr-fast-authored-controls.v1":
            raise ValueError("Use the frozen authored-control plan.")
        recipe = next(item for item in manual["controls"] if item["id"] == args.manual_control)
        if manual["base"]["material"] != {"young": args.young, "poisson": args.poisson, "density": args.density}:
            raise ValueError("The authored control material must match its frozen plan.")
        args.mesh = args.output / "authored-mesh.npz"
        voxel_mesh(recipe["grid_shape"], recipe["cells"], manual["base"]["fixture_scale_m"], args.mesh)
    producer_files = [
        args.git_source / "genesis/engine/solvers/rigid/stress" / name
        for name in ("contact.py", "scatter.py", "surface.py", "model.py", "solve.py", "balance.py")
    ]
    producer_identities = [file_id(path) for path in producer_files]
    options = gs.options.RigidStressOptions.dpr(
        mesh=args.mesh,
        output_mode="full",
        young=args.young,
        poisson=args.poisson,
        density=args.density,
        quadrature=16,
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        pressure_quadrature="edge",
        method="auto",
        inverse_precision="64",
        history_size=0,
    )
    model = StressModel(options)
    if manual is not None:
        args.archives = [author_contacts(recipe, manual["base"], model, args.output)]
    identities = [file_id(path) for path in [args.mesh, args.budgets, *args.archives]]
    if model.load_info.load_nodes.shape[0] != 0:
        raise ValueError("This checker qualifies volume FEM only.")
    vertices = qd_to_numpy(model.info.vertices, transpose=True)
    elements = qd_to_numpy(model.info.elements, transpose=False)
    volumes = qd_to_numpy(model.info.volumes, transpose=True)
    sample_positions = vertices[elements[:, :4]]
    weights = np.broadcast_to(volumes[:, None] / 4, sample_positions.shape[:-1])
    operator = cpu_operator(model.info)
    model_gradients = qd_to_numpy(model.info.gradients, transpose=True)
    centrifugal = qd_to_numpy(model.info.centrifugal, transpose=True).reshape((-1, 6))
    model_arrays = {
        field.name: qd_to_numpy(vars(model.info)[field.name], transpose=False)
        for field in dataclasses.fields(model.info)
    }
    np.savez_compressed(
        args.output / "model.npz", **model_arrays, sample_positions=sample_positions, sample_weights=weights
    )
    result = {
        "schema": "genesis.dpr-fast-frozen-rule-qualification.v1",
        "complete": False,
        "passed": False,
        "application_qualified_for_arbitrary_contacts": False,
        "goal_complete": False,
        "primary_timing": False,
        "identity": runtime_identity(),
        "source_revision": source_revision,
        "source": str(args.git_source),
        "worker": file_id(Path(__file__)),
        "inputs": identities,
        "native_producers": producer_identities,
        "authored_manual_control": args.manual_control,
        "authored_manual_plan": file_id(args.manual_plan) if manual is not None else None,
        "admission_method": file_id(args.admission_method) if args.admission_method is not None else None,
        "options": options.model_dump(mode="json"),
        "corpus_role": args.corpus_role,
        "model": file_id(args.output / "model.npz"),
        "rows": [],
        "scope": "Frozen original point/F/anchor/R/friction and omega; native production moment/scatter route, no rigid advance, no CPU-oracle injection. Wrapper passed indicates trustworthy completed controls, not that every tested candidate passed.",
    }
    write_json(args.output / "result.json", finite_json(result))
    reference_passed = True
    coarse_count = 0
    for index, archive in enumerate(args.archives):
        frame = load_frame(archive, f"input{index:02d}-{archive.stem}", args.allow_conditional_zero_omega, args.envs)
        directory = args.output / frame.name
        directory.mkdir()
        physical = {
            field.name: vars(frame)[field.name]
            for field in dataclasses.fields(frame)
            if field.name not in ("path", "name", "is_complete_mechanical_input")
            and vars(frame)[field.name] is not None
        }
        np.savez_compressed(directory / "physical-inputs.npz", **physical)
        material_binding = None
        archived_metadata = archive.with_suffix(".json")
        if not archived_metadata.is_file():
            archived_metadata = archive.parent / "result.json"
        if archived_metadata.is_file():
            metadata = json.loads(archived_metadata.read_text())
            recorded = metadata.get("stress_options")
            if isinstance(recorded, dict):
                for key in ("young", "poisson", "density"):
                    if recorded[key] != vars(options)[key]:
                        raise ValueError(f"Frozen archive material differs from the declared model: {key}.")
                if recorded["pressure_profile"] != "hertz" or recorded["radius_model"] != "hertz":
                    raise ValueError("This audit requires the archived automatic-radius Hertz pressure law.")
                material_binding = file_id(archived_metadata)
        topology_binding = None
        archived_model = archive.parent / "replay-model.npz"
        if archived_model.is_file():
            with np.load(archived_model, allow_pickle=False) as recorded:
                for key, expected in (
                    ("physical_vertices", vertices),
                    ("physical_surface_nodes", qd_to_numpy(model.info.surface_nodes, transpose=False)),
                    ("elements", elements),
                ):
                    if not np.array_equal(recorded[key], expected):
                        raise ValueError(f"Frozen archive/model node or face mapping differs: {key}.")
            topology_binding = file_id(archived_model)
        rules = list(args.rules)
        if args.reference256:
            rules.append("angle256")
        if args.coarse_controls:
            rules.extend(("edge4", "angle2")[: max(0, 4 - coarse_count)])
            coarse_count += len(rules) - len(args.rules) - int(args.reference256)
        records = [rule_fields(model, options, frame, rule, directory / rule, args.tasks_per_env) for rule in rules]
        record_map = {record["rule"]: record for record in records}
        with np.load(record_map["angle128"]["fields"]["path"], allow_pickle=False) as saved:
            reference = {key: saved[key] for key in saved.files}
        loaded = frame.valid & (np.linalg.norm(frame.force, axis=-1) > 1e-30)
        masks = []
        for env in range(len(frame.valid)):
            delta = sample_positions[:, :, None, :] - frame.center[env, loaded[env]]
            region = (np.linalg.norm(delta, axis=-1) <= 2 * frame.radius[env, loaded[env]]).any(axis=-1)
            masks.append({"global": np.ones_like(region), "loaded": region, "remote": ~region})
        cpu_rows = []
        _matrix, free, modes, relief, factor = operator
        for env in range(len(frame.valid)):
            spin = frame.omega[env]
            terms = np.r_[spin**2, spin[0] * spin[1], spin[0] * spin[2], spin[1] * spin[2]]
            force = reference["nodal_force"][env].reshape(-1) + centrifugal @ terms
            rhs = force - relief @ (modes.T @ force)
            q = np.zeros_like(rhs)
            q[free] = scipy.linalg.cho_solve(factor, rhs[free]) / options.young
            sigma, vm = stress_field(model_gradients, elements, q, options.young, options.poisson)
            q_error = float(
                np.linalg.norm(q - reference["displacement"][env].reshape(-1)) / max(np.linalg.norm(q), 1e-30)
            )
            tensor_error = float(
                np.linalg.norm(sigma - reference["stress_tensor"][env]) / max(np.linalg.norm(sigma), 1e-30)
            )
            cpu_rows.append(
                {
                    "environment": env,
                    "q_relative": q_error,
                    "tensor_relative": tensor_error,
                    "passed": bool(q_error <= 1e-8 and tensor_error <= 1e-8),
                }
            )
            np.savez_compressed(
                directory / f"cpu-linear-control-env{env}.npz",
                rhs=rhs,
                displacement=q,
                stress_tensor=sigma,
                von_mises=vm,
            )
        reference_valid = [
            native and cpu["passed"]
            for native, cpu in zip(record_map["angle128"]["source_and_elastic_eligible_per_environment"], cpu_rows)
        ]
        stability = None
        if args.reference256:
            with np.load(record_map["angle256"]["fields"]["path"], allow_pickle=False) as saved:
                higher = {key: saved[key] for key in saved.files}
            stability = region_errors(reference, higher, weights, masks)
            for env, env_regions in enumerate(stability):
                is_stable = record_map["angle256"]["source_and_elastic_eligible_per_environment"][env]
                for values in env_regions.values():
                    if values["samples"]:
                        values["reference_stability_passed"] = (
                            values["tensor_relative_L2"] <= 0.01 and values["peak_bidirectional_relative"] <= 0.005
                        )
                        is_stable &= values["reference_stability_passed"]
                reference_valid[env] &= is_stable
        reference_passed &= all(reference_valid)
        for record in records:
            with np.load(record["fields"]["path"], allow_pickle=False) as saved:
                fields = {key: saved[key] for key in saved.files}
            metrics = region_errors(fields, reference, weights, masks)
            application = []
            for env, values in enumerate(metrics):
                if not reference_valid[env]:
                    application.append(None)
                    continue
                passed = record["source_and_elastic_eligible_per_environment"][env]
                passed &= all(row["passed"] is not False for row in values.values())
                is_zero_load = not loaded[env].any() and not np.any(frame.omega[env])
                if is_zero_load:
                    max_stress = float(np.linalg.norm(fields["tensor_full9"][env], axis=(-2, -1)).max())
                    passed &= max_stress <= 1.0
                    values["zero_load"] = {"maximum_tensor_Frobenius_Pa": max_stress, "passed": max_stress <= 1.0}
                application.append(bool(passed))
            old_errors = {
                key: legacy_error(fields[key], reference[key]).tolist()
                for key in ("nodal_force", "displacement", "stress_tensor", "peak")
            }
            old_passed = [
                record["source_and_elastic_eligible_per_environment"][env]
                and all(values[env] < (1e-6 if key == "nodal_force" else 1e-4) for key, values in old_errors.items())
                for env in range(len(frame.valid))
            ]
            record.update(
                application_regions=metrics,
                application_passed_per_environment=application,
                truth_evaluable_per_environment=reference_valid,
                legacy_errors=old_errors,
                legacy_passed_per_environment=old_passed,
                false_accept_count=sum(
                    admit and truth is False
                    for admit, truth in zip(record["empirical_environment_admitted"], application)
                ),
                false_reject_count=sum(
                    not admit and truth is True
                    for admit, truth in zip(record["empirical_environment_admitted"], application)
                ),
                mass_normalized_v2_false_accept_count=sum(
                    admit and truth is False
                    for admit, truth in zip(record["mass_normalized_v2_environment_admitted"], application)
                ),
                mass_normalized_v2_false_reject_count=sum(
                    not admit and truth is True
                    for admit, truth in zip(record["mass_normalized_v2_environment_admitted"], application)
                ),
                converged_but_application_inaccurate_count=sum(
                    eligible and truth is False
                    for eligible, truth in zip(record["source_and_elastic_eligible_per_environment"], application)
                ),
            )
        result["rows"].append(
            {
                "input": file_id(archive),
                "physical_inputs": file_id(directory / "physical-inputs.npz"),
                "archived_material_binding": material_binding,
                "archived_topology_binding": topology_binding,
                "name": frame.name,
                "original_environment_indices": frame.environment_indices.tolist(),
                "whole_frame_qualification": frame.is_complete_mechanical_input,
                "mechanical_scope": "Original archived body-frame omega retained."
                if frame.is_complete_mechanical_input
                else "CONDITIONAL CONTACT-LOAD FIELD ONLY: omega=0 explicitly supplied; no historical whole-frame qualification.",
                "CPU_same_operator_linear_control": cpu_rows,
                "reference256_stability": stability,
                "reference_evaluable_per_environment": reference_valid,
                "reference_stability_status": "checked_q128_vs_q256" if args.reference256 else "not_requested_here",
                "rules": records,
            }
        )
        write_json(args.output / "result.json", finite_json(result))
    result["input_files_unchanged"] = identities == [
        file_id(path) for path in [args.mesh, args.budgets, *args.archives]
    ]
    result["producer_files_unchanged"] = producer_identities == [file_id(path) for path in producer_files]
    result["complete"] = True
    result["passed"] = bool(reference_passed and result["input_files_unchanged"] and result["producer_files_unchanged"])
    result["all_rows_whole_frame_inputs"] = all(row["whole_frame_qualification"] for row in result["rows"])
    result["converged_bad_rule_detection_exercised"] = any(
        record["converged_but_application_inaccurate_count"] for row in result["rows"] for record in row["rules"]
    )
    write_json(args.output / "result.json", finite_json(result))
    if not result["passed"]:
        raise RuntimeError("Reference/numerical/input control failed; preserve all completed negative rule outputs.")


if __name__ == "__main__":
    main()
