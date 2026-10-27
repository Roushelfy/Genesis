"""Check warped original-surface fallback and ambiguous material-anchor rejection.

The warped fixture comes from the pinned CPU reference. An additional source
on the same physical surface supplies a nonzero elastic-field control. The
overlapping-patch primitive exercises material rejection and load-error
propagation independently of elasticity. Scientific execution requires Slurm
rtx-mid and image 1_26.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

PRIMARY = "797203553f246ca155bbef909c1bb33e7cd7b90c"
TOLERANCE = 1e-3


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load(np, path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _cpu_worker(args):
    sys.path.insert(0, str(args.reference / "reference-source"))
    import numpy as np
    from scipy import sparse
    from threadpoolctl import threadpool_limits

    from research.rigid_stress.shell_cpu.contacts import Contact, _shape
    from research.rigid_stress.shell_cpu.core import ShellModel
    from research.rigid_stress.triangular_load_cpu.reference import Q4LoadTransfer

    names = (
        "research.rigid_stress.shell_cpu.contacts",
        "research.rigid_stress.shell_cpu.core",
        "research.rigid_stress.triangular_load_cpu.reference",
        "research.rigid_stress.wrench",
    )
    imports = [_identity(sys.modules[name].__file__) for name in names]
    if not all(Path(row["path"]).is_relative_to(args.reference / "reference-source") for row in imports):
        raise RuntimeError("The guard oracle requires the frozen primary CPU modules.")
    control = args.reference / "inputs/controls/warped-miss"
    inputs = _load(np, control / "model.npz")
    frozen_cases = json.loads((control / "cases.json").read_text())
    frozen_raw = _load(np, args.reference / "oracles/controls/warped-miss/Q4-q48.npz")["raw"]
    inputs.update(density=1000.0, drilling_factor=1e-3)
    physical = inputs["vertices"] + 0.5 * inputs["thickness"] * inputs["directors"]
    coordinate = np.array([0.43, -0.27])
    shape, dxi, deta = _shape(*coordinate)
    normal = np.cross(dxi @ physical, deta @ physical)
    normal /= np.linalg.norm(normal)
    tangent = dxi @ physical
    tangent /= np.linalg.norm(tangent)
    asymmetric = {
        "point": (shape @ physical).tolist(),
        "force": (-0.05 * normal + 0.008 * tangent).tolist(),
        "normal": normal.tolist(),
        "radius": 0.0002,
        "friction": 0.5,
        "curvatures": [50.0, 50.0],
        "side": 1,
        "face_hint": 0,
    }
    original = frozen_cases[0]["contacts"][0]
    cases = [
        frozen_cases[0],
        {"name": "asymmetric-warp", "category": "asymmetric", "contacts": [asymmetric]},
        {"name": "warped-pair", "category": "multiple", "contacts": [original, asymmetric]},
        {"name": "release", "category": "release", "contacts": []},
    ]
    destination = args.output / "inputs/guard"
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "model.npz", **inputs)
    _json(destination / "cases.json", cases)
    model = ShellModel(
        inputs["vertices"],
        inputs["quads"],
        E=float(inputs["young"]),
        nu=float(inputs["poisson"]),
        h=float(inputs["thickness"]),
        density=inputs["density"],
        directors=inputs["directors"],
        drilling_factor=inputs["drilling_factor"],
        gauge="translation",
    )
    stiffness_before = hashlib.sha256(
        model.K.data.tobytes() + model.K.indices.tobytes() + model.K.indptr.tobytes()
    ).hexdigest()
    oracle = args.output / "oracles"
    oracle.mkdir()
    sparse.save_npz(oracle / "K.npz", model.K)
    np.savez_compressed(
        oracle / "numerics.npz",
        mass_modes=model.mass_modes,
        rigid_modes=model.rigid_modes,
        mass_gram=model.mass_gram,
        gauss_weights=model.gauss_weights,
    )
    transfer = Q4LoadTransfer(model)
    diagnostics = []
    with threadpool_limits(limits=1):
        for order in (24, 48):
            raw = np.zeros((model.ndof, len(cases)))
            source_raw = np.zeros((model.ndof, 2, len(cases)))
            rows = []
            for i_b, case in enumerate(cases):
                for i_c, source in enumerate(case["contacts"]):
                    contact = Contact(
                        np.array(source["point"]),
                        np.array(source["force"]),
                        np.array(source["normal"]),
                        radius=source["radius"],
                        friction=source["friction"],
                        side=source["side"],
                        face_hint=source["face_hint"],
                    )
                    load, record = transfer.transfer(contact, order=order)
                    source_raw[:, i_c, i_b] = load
                    raw[:, i_b] += load
                    rows.append({"environment": i_b, "slot": i_c, "name": case["name"], **record})
            balanced, acceleration = model.balance_load(raw)
            q = model.solve(raw)
            fields = model.stress(q)
            local = np.stack((fields["local_top"], fields["local_bottom"]))
            xx, yy, xy = local[:, :, :, 0], local[:, :, :, 1], local[:, :, :, 2]
            peak = np.sqrt(np.maximum(0, xx * xx + yy * yy - xx * yy + 3 * xy * xy)).max(axis=(0, 1, 2))
            np.savez_compressed(
                oracle / f"q{order}.npz",
                raw=raw,
                source_raw=source_raw,
                balanced=balanced,
                relief_acceleration=acceleration,
                q=q,
                stress_local=local,
                stress_full9=np.stack((fields["sigma_top"], fields["sigma_bottom"])).reshape(
                    (2, len(model.quads), 4, 9, len(cases))
                ),
                peak=peak,
                N=fields["N"],
                M=fields["M"],
                Q=fields["Q"],
            )
            diagnostics.append({"quadrature": order, "sources": rows})
    q48 = _load(np, oracle / "q48.npz")
    frozen_delta = np.linalg.norm(q48["raw"][:, 0] - frozen_raw) / np.linalg.norm(frozen_raw)
    if frozen_delta > 1e-12 or not np.linalg.norm(frozen_raw) > 0:
        raise ArithmeticError("The independent CPU replay changed the frozen nonzero warped Q4 load.")
    stiffness_after = hashlib.sha256(
        model.K.data.tobytes() + model.K.indices.tobytes() + model.K.indptr.tobytes()
    ).hexdigest()
    if stiffness_after != stiffness_before or [_identity(row["path"]) for row in imports] != imports:
        raise RuntimeError("The CPU oracle stiffness or frozen source changed during the check.")
    _json(
        args.output / "cpu-source.json",
        {
            "primary": PRIMARY,
            "imports": imports,
            "frozen_control": [_identity(control / name) for name in ("model.npz", "cases.json")],
            "frozen_raw": _identity(args.reference / "oracles/controls/warped-miss/Q4-q48.npz"),
            "replay_relative_load_error": float(frozen_delta),
            "stiffness_sha256": stiffness_before,
            "added_elastic_parameters": {"density": inputs["density"], "drilling_factor": inputs["drilling_factor"]},
            "scope": "The frozen fixture provides load-only evidence. This worker constructs the original physical Q4 elasticity independently and adds a source on that same surface.",
            "asymmetric_coordinate": coordinate.tolist(),
            "methods": diagnostics,
        },
    )


def _native_api():
    from .check_planar_loading_gpu import (
        _api,
        build_native_mother,
        native_outputs,
        prepare_physical_contacts,
        unbatched_array,
    )
    from .check_planar_material import _errors, _snapshot_contacts, _upload_contacts

    api = _api()
    from genesis.engine.solvers.rigid.stress.data import allocate_stress_info, create_stress_state
    from genesis.engine.solvers.rigid.stress.planar_geometry import build_planar_geometry
    from genesis.engine.solvers.rigid.stress.planar_loading import (
        _PLANAR_NORMAL_ERROR_FRACTION,
        PlanarFallback,
        PlanarLoading,
    )
    from genesis.engine.solvers.rigid.stress.planar_material import (
        PlanarMaterialStatus,
        create_planar_material,
        kernel_planar_material,
    )
    from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
    from genesis.utils.array_class import ErrorCode

    return SimpleNamespace(
        **(vars(api) | {name: value for name, value in locals().items() if name not in ("api", "_api")})
    )


def _error_maxima(errors, loaded):
    return {key: float(value[loaded].max(initial=0)) for key, value in errors.items()}


def _check_fields(api, checks, label, actual, expected, numerics, field_applicable, loaded):
    np = api.np
    errors = api._errors(np, actual, expected, numerics["gauss_weights"])
    errors["load"] = np.linalg.norm(actual["force"] - expected["raw"], axis=0) / np.maximum(
        np.linalg.norm(expected["raw"], axis=0), 1e-30
    )
    load_max = float(errors["load"][loaded].max(initial=0))
    field_maxima = _error_maxima(
        {key: value for key, value in errors.items() if key != "load"}, loaded & field_applicable
    )
    checks.condition(
        f"{label}/nonzero_full_nodal_load_reference", load_max <= TOLERANCE, maximum=load_max, tolerance=TOLERANCE
    )
    checks.condition(
        f"{label}/nondegenerate_elastic_fields",
        all(value <= TOLERANCE for value in field_maxima.values()),
        maxima=field_maxima,
        applicable=(loaded & field_applicable).tolist(),
        tolerance=TOLERANCE,
    )
    return errors, {"maximum_load_error": load_max, "maximum_applicable_field_errors": field_maxima}


def _warped(api, args, writer, checks):
    np = api.np
    cpu = args.output / "cpu"
    native = api.build_native_mother(api, cpu, args.output, "guard", quadrature=args.quadrature)
    model = native.model
    checks.exact("warped/native_model_quadrature", model.options.quadrature, args.quadrature)
    expected24, expected48 = _load(np, cpu / "oracles/q24.npz"), _load(np, cpu / "oracles/q48.npz")
    numerics = _load(np, cpu / "oracles/numerics.npz")
    oracle_k = api.sparse.load_npz(cpu / "oracles/K.npz")
    actual_k = api.sparse.bsr_matrix(
        (
            api.unbatched_array(api, model.info.stiffness) * model.options.young,
            api.unbatched_array(api, model.info.columns),
            api.unbatched_array(api, model.info.row_start),
        ),
        shape=oracle_k.shape,
    ).tocsr()
    checks.close("warped/original_Q4_stiffness", actual_k, oracle_k, rtol=1e-10, atol=1e-8)
    checks.close(
        "warped/original_mass_modes",
        api.unbatched_array(api, model.info.mass_modes).reshape((-1, 6)),
        numerics["mass_modes"],
        rtol=1e-10,
        atol=1e-14,
    )
    loaded = np.linalg.norm(expected48["raw"], axis=0) > 0
    relief_fraction = np.linalg.norm(expected48["balanced"], axis=0) / np.maximum(
        np.linalg.norm(expected48["raw"], axis=0), 1e-30
    )
    field_applicable = loaded & (relief_fraction > 1e-10)
    checks.condition("warped/asymmetric_elastic_witness", field_applicable[1] and expected48["peak"][1] > 0)
    uncertainty, cpu_record = _check_fields(
        api,
        checks,
        "CPU/q24_q48",
        expected24 | {"force": expected24["raw"]},
        expected48,
        numerics,
        field_applicable,
        loaded,
    )
    writer.arrays("warped/CPU-integration-uncertainty.npz", **uncertainty)
    writer.json(
        "warped/field-scope.json",
        {
            "balanced_to_raw_load_ratio": relief_fraction.tolist(),
            "field_comparison_applicable": field_applicable.tolist(),
            "rule": "A load with balanced/raw norm <= 1e-10 supplies load equivalence and equilibrium evidence. Nondegenerate cases additionally gate the complete elastic tensor, sampled peak and N/M/Q at 0.1%.",
            "CPU_integration": cpu_record,
        },
    )
    if not all(row["passed"] for row in checks.rows):
        raise ArithmeticError(
            "Resolve the original Q4 operators and independent integration before native guard loads."
        )
    free = api.unbatched_array(api, model.info.is_free).reshape(-1).astype(bool)
    records = []
    for subdivision in args.subdivisions:
        for cooperative in (True, False):
            mode = "warp" if cooperative else "serial"
            contacts, cases, valid, _sides = api.prepare_physical_contacts(
                api, cpu, "guard", model.options.young, model.options.poisson
            )
            source = api._snapshot_contacts(api, contacts)
            surface = api.StressSurface(args.quadrature, model.load_info, angular=True)
            checks.exact(
                f"warped/k{subdivision}/{mode}/native_surface_quadrature", surface.info.gauss.shape[0], args.quadrature
            )
            loading = api.PlanarLoading(surface.info, model.load_info, contacts, subdivision, tasks_per_env=64)
            state = model.create_state(len(cases), "full")
            omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(len(cases),))
            omega.fill(0.0)
            workspace = api.create_scatter_workspace(len(cases), 128, False)
            options = model.options.model_copy(
                update={"cooperative_pressure": cooperative, "cooperative_scatter": cooperative}
            )
            normal_error = api.unbatched_array(api, loading.info.normal_error)
            checks.condition(
                f"warped/k{subdivision}/{mode}/normal_error_witness",
                (normal_error > api._PLANAR_NORMAL_ERROR_FRACTION * source["radius"][valid].max()).all(),
                fraction=api._PLANAR_NORMAL_ERROR_FRACTION,
            )
            baseline = None
            for phase in ("initial", "slot_reverse", "release_all", "reload"):
                reverse, release = phase == "slot_reverse", phase == "release_all"
                api._upload_contacts(api, contacts, source, reverse=reverse, release=release)
                current_valid = valid[::-1].copy() if reverse else valid.copy()
                if release:
                    current_valid.fill(False)
                api.kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
                original_centers = api.unbatched_array(api, contacts.center).copy()
                loading.scatter(state, model.load_info, surface.info, options, workspace, False)
                model.recover(omega, state)
                actual = api.native_outputs(api, model, state)
                label = f"warped/k{subdivision}/{mode}/{phase}"
                expected = {key: value.copy() for key, value in expected48.items()}
                if release:
                    for value in expected.values():
                        value.fill(0)
                errors, record = _check_fields(
                    api, checks, label, actual, expected, numerics, field_applicable, loaded & (not release)
                )
                balanced = actual["force"] - numerics["mass_modes"] @ np.linalg.solve(
                    numerics["mass_gram"], numerics["rigid_modes"].T @ actual["force"]
                )
                checks.close(f"{label}/original_global_relief", actual["rhs"], balanced, rtol=1e-10, atol=1e-13)
                residual = (oracle_k @ actual["q"] - actual["rhs"])[free]
                residual_norm, rhs_norm = np.linalg.norm(residual, axis=0), np.linalg.norm(actual["rhs"][free], axis=0)
                checks.condition(
                    f"{label}/original_unscaled_K_residual",
                    (residual_norm <= np.maximum(1e-11, 1e-8 * rhs_norm)).all(),
                    absolute_norm=residual_norm.tolist(),
                    relative_norm=(residual_norm / np.maximum(rhs_norm, 1e-30)).tolist(),
                )
                checks.condition(f"{label}/native_admission", actual["valid"].all())
                checks.condition(
                    f"{label}/both_physical_sides_enabled",
                    not api.unbatched_array(api, loading.state.side_filter).any(),
                )
                if release:
                    checks.exact(f"{label}/zero_raw_load", actual["force"], np.zeros_like(actual["force"]))
                else:
                    nodal = actual["force"].reshape((len(native.inputs["vertices"]), 6, len(cases)))
                    for i_b, case in enumerate(cases):
                        if len(case["contacts"]) == 1:
                            physical_source = case["contacts"][0]
                            source_force = np.array(physical_source["force"])
                            source_point = np.array(physical_source["point"])
                            source_magnitude = np.linalg.norm(source_force)
                            resultant = nodal[:, :3, i_b].sum(axis=0)
                            moment = np.cross(native.inputs["vertices"] - source_point, nodal[:, :3, i_b]).sum(
                                axis=0
                            ) + nodal[:, 3:, i_b].sum(axis=0)
                            checks.condition(
                                f"{label}/raw_original_point_wrench/{case['name']}",
                                np.linalg.norm(resultant - source_force) <= 1e-8 * source_magnitude
                                and np.linalg.norm(moment) <= 1e-8 * source_magnitude * physical_source["radius"],
                            )
                fallback = api.unbatched_array(api, loading.state.fallback)
                reason = api.unbatched_array(api, loading.state.reason)
                checks.exact(f"{label}/original_surface_fallback", fallback, current_valid)
                checks.condition(
                    f"{label}/normal_error_geometry_guard", (reason[current_valid] == api.PlanarFallback.GEOMETRY).all()
                )
                checks.condition(
                    f"{label}/material_component_resolved",
                    not api.unbatched_array(api, loading.material_state.status)[current_valid].any(),
                )
                for name in ("position", "force", "radius", "normal", "friction"):
                    target = source[name][::-1] if reverse else source[name]
                    checks.exact(
                        f"{label}/unchanged_source/{name}", api.unbatched_array(api, vars(contacts)[name]), target
                    )
                checks.exact(
                    f"{label}/unchanged_anchored_center", api.unbatched_array(api, contacts.center), original_centers
                )
                target_position = source["position"][::-1] if reverse else source["position"]
                checks.close(
                    f"{label}/original_surface_center",
                    original_centers[current_valid],
                    target_position[current_valid],
                    rtol=1e-12,
                    atol=1e-14,
                )
                magnitude = np.linalg.norm(source["force"][::-1] if reverse else source["force"], axis=-1)
                radius = source["radius"][::-1] if reverse else source["radius"]
                force_error = api.unbatched_array(api, contacts.force_error)
                moment_error = api.unbatched_array(api, contacts.moment_error)
                checks.condition(
                    f"{label}/original_source_wrench",
                    (force_error[current_valid] <= 1e-8 * magnitude[current_valid]).all()
                    and (moment_error[current_valid] <= 1e-8 * magnitude[current_valid] * radius[current_valid]).all(),
                )
                if baseline is None:
                    baseline = actual
                else:
                    for key in ("force", "rhs", "q", "stress_full9", "peak", "N", "M", "Q"):
                        target = np.zeros_like(baseline[key]) if release else baseline[key]
                        checks.close(
                            f"{label}/same_current_load/{key}",
                            actual[key],
                            target,
                            rtol=1e-9,
                            atol=1e-10 if key.startswith("stress") or key == "peak" else 1e-13,
                        )
                writer.arrays(
                    f"{label}/output.npz", **actual, **{f"error_{key}": value for key, value in errors.items()}
                )
                writer.arrays(
                    f"{label}/routing.npz",
                    fallback=fallback,
                    reason=reason,
                    normal_error=normal_error,
                    material_status=api.unbatched_array(api, loading.material_state.status),
                    material_queried=api.unbatched_array(api, loading.material_state.queried),
                    source_status=api.unbatched_array(api, contacts.status),
                    force_error=force_error,
                    moment_error=moment_error,
                    center=api.unbatched_array(api, contacts.center),
                )
                records.append({"label": label, "native_quadrature": surface.info.gauss.shape[0], **record})
    checks.close(
        "warped/K_preserved_after_loading",
        api.sparse.bsr_matrix(
            (
                api.unbatched_array(api, model.info.stiffness) * model.options.young,
                api.unbatched_array(api, model.info.columns),
                api.unbatched_array(api, model.info.row_start),
            ),
            shape=oracle_k.shape,
        ).tocsr(),
        oracle_k,
        rtol=1e-10,
        atol=1e-8,
    )
    return records


def _tie(api, args, writer, checks):
    np = api.np
    thickness, radius = 0.0005, 0.0008
    patch = np.array([[-0.002, -0.002, 0.0], [0.002, -0.002, 0.0], [0.002, 0.002, 0.0], [-0.002, 0.002, 0.0]])
    midsurface = np.vstack((patch, patch))
    quads = np.arange(8).reshape((2, 4))
    directors = np.broadcast_to([0.0, 0.0, 1.0], midsurface.shape).copy()
    vertices = np.vstack((midsurface + [0.0, 0.0, thickness / 2], midsurface - [0.0, 0.0, thickness / 2]))
    faces = np.zeros((4, 6), dtype=api.gs.np_int)
    faces[:2, :4], faces[2:, :4] = quads, (quads + 8)[:, (0, 3, 2, 1)]
    load_info = api.allocate_stress_info(16, 0, 4, 0, 0, 16)
    load_info.vertices.from_numpy(vertices.astype(api.gs.np_float, copy=False))
    load_info.surface_nodes.from_numpy(faces)
    load_info.load_nodes.fill(0)
    load_info.load_offsets.fill(0.0)
    surface = api.StressSurface(args.quadrature, load_info, angular=True)
    checks.exact("tie/native_surface_quadrature", surface.info.gauss.shape[0], args.quadrature)
    geometry = api.build_planar_geometry(surface.info, load_info, 4)
    material = api.create_planar_material(2, 2, 1)
    contacts = api.create_contacts(2, 1, 1e-4, hertz_profile=True)
    source = {
        "position": np.broadcast_to([0.0, 0.0, thickness / 2], (2, 1, 3)).copy(),
        "force": np.broadcast_to([0.0, 0.0, -0.05], (2, 1, 3)).copy(),
        "normal": np.broadcast_to([0.0, 0.0, -1.0], (2, 1, 3)).copy(),
        "radius": np.full((2, 1), radius),
        "friction": np.full((2, 1), 0.5),
        "valid": np.array([[True], [False]]),
    }
    fallback = api.V(dtype=api.gs.qd_bool, shape=(2, 1))
    errno = api.V(dtype=api.gs.qd_int, shape=(1,))
    state = api.create_stress_state(16, 1)
    state.valid.fill(True)
    state.invalid_steps.fill(0)
    writer.arrays("tie/physical-inputs.npz", vertices=midsurface, quads=quads, directors=directors, **source)
    writer.json(
        "tie/independent-geometry-answer.json",
        {
            "parents": 2,
            "native_quadrature": surface.info.gauss.shape[0],
            "authored_components": [[0], [1]],
            "equal_distance_m": [0.0, 0.0],
            "expected_first_outer_anchor_face": 0,
            "expected_positive_parent_bits": 3,
            "expected_selected_parent_bits": 1,
            "expected_selected_parents": 1,
            "expected_excluded_parents": 1,
            "expected_material_status": "AMBIGUOUS_ANCHOR",
            "expected_contact_status": 9,
            "scope": "Duplicated coincident Q4 patches use disjoint node IDs, matching physical orientation and both physical sides. This primitive tests rejection and lifecycle error propagation. It assembles no elastic matrix and performs no factorization or solve.",
        },
    )
    for phase in ("initial", "slot_reverse", "release_all", "reload"):
        reverse, release = phase == "slot_reverse", phase == "release_all"
        api._upload_contacts(api, contacts, source, reverse=reverse, release=release)
        valid = source["valid"][::-1].copy() if reverse else source["valid"].copy()
        if release:
            valid.fill(False)
        fallback.from_numpy(valid)
        errno.fill(0)
        api.kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
        label = f"tie/{phase}"
        checks.condition(f"{label}/initial_anchor_admitted", not api.unbatched_array(api, contacts.status)[valid].any())
        checks.exact(
            f"{label}/deterministic_first_anchor",
            api.unbatched_array(api, contacts.anchor_face)[valid],
            np.zeros(valid.sum(), dtype=api.gs.np_int),
        )
        api.kernel_planar_material(
            float(np.finfo(np.float64).eps), contacts, fallback, surface.info, geometry, material
        )
        api.kernel_begin_step(state)
        api.kernel_accept(state, contacts, errno)
        queried = api.unbatched_array(api, material.queried)
        material_status = api.unbatched_array(api, material.status)
        contact_status = api.unbatched_array(api, contacts.status)
        hits = api.unbatched_array(api, material.hits)[..., 0].astype(np.uint32)
        selected = api.unbatched_array(api, material.mask)[..., 0].astype(np.uint32)
        selected_count = api.unbatched_array(api, material.selected_count)
        excluded_count = api.unbatched_array(api, material.excluded_count)
        error = api.qd_to_numpy(errno, transpose=True)
        checks.exact(f"{label}/queried_current_source", queried, valid)
        checks.condition(
            f"{label}/material_ambiguity",
            (material_status[valid] == api.PlanarMaterialStatus.AMBIGUOUS_ANCHOR).all(),
        )
        checks.condition(f"{label}/source_rejected", (contact_status[valid] == 9).all())
        checks.condition(f"{label}/independent_positive_components", (hits[0][valid] == 3).all())
        checks.condition(f"{label}/independent_anchor_component", (selected[0][valid] == 1).all())
        checks.condition(f"{label}/selected_count", (selected_count[valid] == 1).all())
        checks.condition(f"{label}/excluded_count", (excluded_count[valid] == 1).all())
        checks.condition(
            f"{label}/INVALID_STRESS_LOAD",
            ((error & api.ErrorCode.INVALID_STRESS_LOAD) != 0).all() if valid.any() else not error.any(),
        )
        checks.condition(f"{label}/elastic_solve_error_absent", not (error & api.ErrorCode.INVALID_STRESS_SOLVE).any())
        checks.condition(f"{label}/step_validity", api.qd_to_numpy(state.step_valid, transpose=True).all() == release)
        for name in ("position", "force", "normal", "radius", "friction"):
            target = source[name][::-1] if reverse else source[name]
            checks.exact(f"{label}/unchanged_source/{name}", api.unbatched_array(api, vars(contacts)[name]), target)
        checks.close(
            f"{label}/unchanged_physical_center",
            api.unbatched_array(api, contacts.center)[valid],
            (source["position"][::-1] if reverse else source["position"])[valid],
            rtol=1e-12,
            atol=1e-14,
        )
        writer.arrays(
            f"{label}/native.npz",
            queried=queried,
            material_status=material_status,
            contact_status=contact_status,
            hits=hits,
            selected=selected,
            selected_count=selected_count,
            excluded_count=excluded_count,
            errno=error,
            step_valid=api.qd_to_numpy(state.step_valid, transpose=True),
            anchor=api.unbatched_array(api, contacts.anchor_face),
            center=api.unbatched_array(api, contacts.center),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=("all", "warped", "tie"), default="all")
    parser.add_argument("--quadrature", type=int, choices=(10, 16, 24, 48), default=48)
    parser.add_argument("--subdivisions", nargs="+", type=int, choices=(4, 8, 16, 32), default=(4, 8, 16))
    parser.add_argument("--_cpu-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific execution requires Slurm rtx-mid and image 1_26.")
    args.reference, args.output = args.reference.resolve(), args.output.resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output.")
    if args._cpu_worker:
        _cpu_worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "genesis.planar-guards.v1",
        "complete": False,
        "passed": False,
        "worker": _identity(__file__),
        "reference": _identity(args.reference / "manifest.json"),
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "CPU_worker_returncode": None,
        "stage": args.stage,
        "native_quadrature": args.quadrature,
        "raw_and_applicable_field_tolerance": TOLERANCE,
        "source_wrench_tolerance": 1e-8,
        "original_K_residual": {"relative": 1e-8, "absolute": 1e-11},
        "input_policy": "Native receives original physical vertices/directors/Q4 topology/material and contact inputs. Frozen CPU loads and fields remain independent comparison data.",
    }
    _json(args.output / "result.json", result)
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if manifest["primary_commit"] != PRIMARY or not manifest["complete"] or not manifest["passed"]:
        raise RuntimeError("Use the successful pinned primary CPU reference.")
    snapshot = json.loads((args.reference / "source-snapshot.json").read_text())
    for row in snapshot["files"]:
        if _identity(args.reference / row["snapshot_path"])["sha256"] != row["sha256"]:
            raise RuntimeError(f"Frozen CPU source changed: {row['snapshot_path']}")
    for relative in (
        "inputs/controls/warped-miss/model.npz",
        "inputs/controls/warped-miss/cases.json",
        "oracles/controls/warped-miss/Q4-q48.npz",
    ):
        if _identity(args.reference / relative)["sha256"] != manifest["files"][relative]["sha256"]:
            raise RuntimeError(f"Frozen physical control changed: {relative}")
    if args.stage != "tie":
        cpu = args.output / "cpu"
        cpu.mkdir()
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH="",
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        with (args.output / "cpu.log").open("w") as stream:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_cpu-worker",
                    "--reference",
                    str(args.reference),
                    "--output",
                    str(cpu),
                ],
                cwd=args.reference / "reference-source",
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result["CPU_worker_returncode"] = completed.returncode
        _json(args.output / "result.json", result)
        completed.check_returncode()
    api = _native_api()
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    writer, checks = api.Writer(args.output, api), api.Checks(api)
    engine = Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress"
    source_before = [
        _identity(engine / name)
        for name in (
            "planar_loading.py",
            "planar_geometry.py",
            "planar_material.py",
            "planar_moments.py",
            "contact.py",
            "surface.py",
            "lifecycle.py",
            "recovery.py",
            "shell_model.py",
            "shell_operators.py",
            "scatter.py",
            "quadrilateral.py",
            "data.py",
        )
    ]
    research_before = [
        _identity(Path(__file__).with_name(name))
        for name in ("check_planar_loading_gpu.py", "check_planar_material.py")
    ]
    result.update(identity=api.runtime_identity(), native_sources=source_before, research_helpers=research_before)
    result["normal_error_fraction"] = api._PLANAR_NORMAL_ERROR_FRACTION
    writer.json("result.json", result)
    started = time.perf_counter()
    try:
        if args.stage != "tie":
            result["warped"] = _warped(api, args, writer, checks)
        if args.stage != "warped":
            _tie(api, args, writer, checks)
        checks.condition(
            "provenance/native_sources_unchanged", [_identity(row["path"]) for row in source_before] == source_before
        )
        checks.condition(
            "provenance/research_helpers_unchanged",
            [_identity(row["path"]) for row in research_before] == research_before,
        )
        checks.condition("provenance/worker_unchanged", _identity(__file__) == result["worker"])
        checks.condition(
            "provenance/reference_manifest_unchanged",
            _identity(args.reference / "manifest.json") == result["reference"],
        )
        result.update(
            complete=True,
            passed=all(row["passed"] for row in checks.rows),
            checks=len(checks.rows),
            seconds=time.perf_counter() - started,
        )
    finally:
        checks.flush(writer)
        if sys.exc_info()[0] is not None:
            (args.output / "failure.txt").write_text(traceback.format_exc())
        writer.json("result.json", result)
    if not result["passed"]:
        raise ArithmeticError("Planar guard qualification failed. Preserve the full diagnostic evidence.")
    print(json.dumps({"passed": True, "checks": result["checks"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
