"""Check material-component fallback on a connected nonconvex shell.

An independent CPU quadrilateral oracle selects the positive-footprint parent
component from authored topology. It fits both physical sides together for
each source. Native validation includes every original fallback schedule,
whole-object fields and current-source lifecycle. Scientific execution requires
Slurm rtx-mid and image 1_26.
"""

import argparse
import ast
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
MODES = ("faces", "moments", "serial", "face-overflow")
BUDGETS = {"field": 0.1, "full_field": 0.1, "peak": 0.05, "N": 0.1, "M": 0.1, "Q": 0.1, "mean_field": 0.05}
REFERENCE_TOLERANCE = 1e-3


def _identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load(np, path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _cpu_api(reference):
    sys.path.insert(0, str(reference / "reference-source"))
    import numpy as np
    from scipy import sparse
    from threadpoolctl import threadpool_limits

    from research.rigid_stress.shell_cpu.contacts import ShellDPRTransfer, _basis
    from research.rigid_stress.shell_cpu.core import ShellModel
    from research.rigid_stress.triangular_load_cpu.reference import inverse_bilinear
    from research.rigid_stress.wrench import PadPressureFit, WrenchPatch

    names = (
        "research.rigid_stress.shell_cpu.contacts",
        "research.rigid_stress.shell_cpu.core",
        "research.rigid_stress.triangular_load_cpu.reference",
        "research.rigid_stress.wrench",
    )
    imports = [_identity(Path(sys.modules[name].__file__).resolve()) for name in names]
    if not all(Path(row["path"]).is_relative_to(reference / "reference-source") for row in imports):
        raise RuntimeError("The CPU material oracle must use the frozen primary modules.")
    return SimpleNamespace(**locals())


def _author_u(np, output):
    source = Path(__file__).with_name("check_local_geometry.py")
    text = source.read_text()
    tree = ast.parse(text)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "shell_neck")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102 - evaluate only the captured mesh author
    asset = output / "authored-u.npz"
    _parents, _vertices, _fixed, provenance = namespace["shell_neck"](asset)
    return _load(np, asset), {
        "source": _identity(source),
        "function_sha256": hashlib.sha256(ast.get_source_segment(text, function).encode()).hexdigest(),
        **provenance,
    }


def _cases(thickness):
    cases = []
    for arm, x in (("left", 0.0007), ("right", 0.0023)):
        for side in (-1, 1):
            for friction in (False, True):
                for radius in (0.0016, 0.0018):
                    force = [0.0, 0.0, -0.05 * side]
                    if friction:
                        force[:2] = [0.01 if arm == "left" else -0.01, 0.005]
                    source = {
                        "point": [x, 0.0025, side * thickness / 2],
                        "force": force,
                        "normal": [0.0, 0.0, float(side)],
                        "radius": radius,
                        "curvatures": [50.0, 70.0],
                        "friction": 0.5,
                        "side": side,
                        "face_hint": None,
                    }
                    cases.append(
                        {
                            "name": f"{arm}-s{side}-{'friction' if friction else 'normal'}-a{radius:g}",
                            "category": arm,
                            "contacts": [source],
                        }
                    )
    left = next(row["contacts"][0] for row in cases if row["name"] == "left-s1-friction-a0.0018")
    right = next(row["contacts"][0] for row in cases if row["name"] == "right-s-1-friction-a0.0018")
    cases.append({"name": "opposed-arm-friction", "category": "multiple", "contacts": [left, right]})
    left = next(row["contacts"][0] for row in cases if row["name"] == "left-s1-normal-a0.0018")
    right = next(row["contacts"][0] for row in cases if row["name"] == "right-s-1-normal-a0.0018")
    cases.append({"name": "opposed-arm-normal", "category": "multiple", "contacts": [left, right]})
    cases.append({"name": "release", "category": "release", "contacts": []})
    return cases


def _q4_samples(api, model, source, order):
    np = api.np
    point, radius = np.array(source["point"]), source["radius"]
    samples, hits, anchors = [], set(), []
    for side in (-1, 1):
        physical = model.vertices + side * 0.5 * model.h * model.directors
        for parent, nodes in enumerate(model.quads):
            vertices = physical[nodes]
            normal = np.cross(vertices[1] - vertices[0], vertices[3] - vertices[0])
            normal /= np.linalg.norm(normal)
            height = np.dot(point - vertices[0], normal)
            remaining = radius * radius - height * height
            if remaining <= 0:
                continue
            center = point - height * normal
            plane = api._basis(normal)
            chart = (vertices - center) @ plane
            if side == source["side"]:
                anchor_shape, _coordinate, valid = api.inverse_bilinear(chart, [[0.0, 0.0]])
                if valid[0] and np.linalg.norm(anchor_shape[0] @ vertices - point) <= 2e-11:
                    anchors.append(parent)
            section_radius = np.sqrt(remaining)
            points, weights = api.ShellDPRTransfer._face_polar_samples(chart, section_radius, order)
            if not len(weights):
                continue
            shape, _coordinate, valid = api.inverse_bilinear(chart, points)
            if not valid.all():
                raise ArithmeticError("The independent quadrilateral material inverse failed.")
            positions = center + points @ plane.T
            weights = weights * (2 * np.pi * section_radius**3 / (3 * radius))
            positive = weights > 0
            if positive.any():
                hits.add(parent)
                samples.append((parent, side, positions[positive], weights[positive], shape[positive]))
    if not anchors or not samples:
        raise ArithmeticError("The authored material source has no resolved original-surface footprint.")
    return samples, hits, min(anchors)


def _component(quads, hits, seed):
    edges = {}
    for parent, nodes in enumerate(quads):
        for i in range(4):
            edge = tuple(sorted((int(nodes[i]), int(nodes[(i + 1) % 4]))))
            edges.setdefault(edge, []).append(parent)
    adjacent = [set() for _ in quads]
    for parents in edges.values():
        for parent in parents:
            adjacent[parent].update(other for other in parents if other != parent)
    selected, todo = {seed}, [seed]
    while todo:
        parent = todo.pop()
        for other in adjacent[parent] & hits - selected:
            selected.add(other)
            todo.append(other)
    if not selected <= hits:
        raise ArithmeticError("The independent material seed has no positive footprint.")
    return selected


def _source_load(api, model, source, order, restrict):
    np = api.np
    samples, hits, seed = _q4_samples(api, model, source, order)
    selected = _component(model.quads, hits, seed) if restrict else hits
    active = [row for row in samples if row[0] in selected]
    positions = np.vstack([row[2] for row in active])
    weights = np.concatenate([row[3] for row in active])
    patch = api.WrenchPatch(
        np.array(source["point"]),
        np.array(source["force"]),
        source["radius"],
        source["friction"],
        np.zeros(3),
        -np.array(source["normal"]),
    )
    forces, diagnostics = api.PadPressureFit(positions, weights, patch).solve()
    raw = np.zeros((len(model.vertices), 6))
    offset = 0
    for parent, side, _positions, local_weights, shape in active:
        n = len(local_weights)
        nodal = shape.T @ forces[offset : offset + n]
        nodes = model.quads[parent]
        np.add.at(raw[:, :3], nodes, nodal)
        np.add.at(raw[:, 3:], nodes, side * 0.5 * model.h * np.cross(model.directors[nodes], nodal))
        offset += n
    resultant = raw[:, :3].sum(axis=0)
    moment = np.cross(model.vertices - np.array(source["point"]), raw[:, :3]).sum(axis=0) + raw[:, 3:].sum(axis=0)
    magnitude = np.linalg.norm(source["force"])
    force_error = np.linalg.norm(resultant - source["force"]) / magnitude
    moment_error = np.linalg.norm(moment) / (magnitude * source["radius"])
    if max(force_error, moment_error) > 1e-8:
        raise ArithmeticError("The independent component pressure fit violates its original wrench.")
    record = {
        "seed": seed,
        "hit_parents": sorted(hits),
        "selected_parents": sorted(selected),
        "excluded_parents": sorted(hits - selected),
        "both_sides": sorted({row[1] for row in active}),
        "samples": len(weights),
        "force_error": float(force_error),
        "moment_error": float(moment_error),
        "fit_evaluations": diagnostics.constrained_evaluations,
    }
    return raw.reshape(-1), record


def _cpu_fields(np, model, raw):
    balanced, acceleration = model.balance_load(raw)
    q = model.solve(raw)
    fields = model.stress(q)
    local = np.stack((fields["local_top"], fields["local_bottom"]))
    xx, yy, xy = local[:, :, :, 0], local[:, :, :, 1], local[:, :, :, 2]
    peak = np.sqrt(np.maximum(0, xx * xx + yy * yy - xx * yy + 3 * xy * xy)).max(axis=(0, 1, 2))
    return {
        "raw": raw,
        "balanced": balanced,
        "relief_acceleration": acceleration,
        "q": q,
        "stress_local": local,
        "stress_full9": np.stack((fields["sigma_top"], fields["sigma_bottom"])).reshape(
            (2, len(model.quads), 4, 9, raw.shape[-1])
        ),
        "peak": peak,
        "N": fields["N"],
        "M": fields["M"],
        "Q": fields["Q"],
    }


def _cpu_worker(args):
    api = _cpu_api(args.reference)
    np = api.np
    geometry, author = _author_u(np, args.output)
    thickness, young, poisson, density, drilling = 8e-5, 1e10, 0.3, 2000.0, 1e-3
    model = api.ShellModel(
        geometry["vertices"],
        geometry["quads"],
        E=young,
        nu=poisson,
        h=thickness,
        density=density,
        directors=geometry["directors"],
        gauge="translation",
        drilling_factor=drilling,
    )
    stiffness_before = hashlib.sha256(
        model.K.data.tobytes() + model.K.indices.tobytes() + model.K.indptr.tobytes()
    ).hexdigest()
    inputs = args.output / "inputs/material"
    inputs.mkdir(parents=True)
    np.savez_compressed(
        inputs / "model.npz",
        **geometry,
        thickness=thickness,
        young=young,
        poisson=poisson,
        density=density,
        drilling_factor=drilling,
    )
    cases = _cases(thickness)
    _json(inputs / "cases.json", cases)
    oracle = args.output / "oracles"
    oracle.mkdir()
    api.sparse.save_npz(oracle / "K.npz", model.K)
    api.sparse.save_npz(oracle / "S.npz", model.stress_matrix())
    np.savez_compressed(
        oracle / "numerics.npz",
        mass_modes=model.mass_modes,
        rigid_modes=model.rigid_modes,
        mass_gram=model.mass_gram,
        gauss_weights=model.gauss_weights,
    )
    diagnostics = []
    with api.threadpool_limits(limits=1):
        for order, restrict, label in ((24, True, "q24"), (48, True, "q48"), (48, False, "unmasked-q48")):
            raw = np.zeros((model.ndof, len(cases)))
            source_raw = np.zeros((model.ndof, 2, len(cases)))
            hits = np.zeros((len(model.quads), 2, len(cases)), dtype=bool)
            selected = np.zeros_like(hits)
            rows = []
            for i_b, case in enumerate(cases):
                for i_c, source in enumerate(case["contacts"]):
                    load, record = _source_load(api, model, source, order, restrict)
                    raw[:, i_b] += load
                    source_raw[:, i_c, i_b] = load
                    hits[record["hit_parents"], i_c, i_b] = True
                    selected[record["selected_parents"], i_c, i_b] = True
                    rows.append({"environment": i_b, "slot": i_c, "case": case["name"], **record})
            fields = _cpu_fields(np, model, raw)
            np.savez_compressed(oracle / f"{label}.npz", **fields, source_raw=source_raw, hits=hits, selected=selected)
            _json(oracle / f"{label}.json", rows)
            diagnostics.append({"method": label, "sources": len(rows), "rows": rows})
    stiffness_after = hashlib.sha256(
        model.K.data.tobytes() + model.K.indices.tobytes() + model.K.indptr.tobytes()
    ).hexdigest()
    if stiffness_after != stiffness_before or [_identity(row["path"]) for row in api.imports] != api.imports:
        raise RuntimeError("The independent global stiffness or frozen CPU sources changed during material validation.")
    _json(
        args.output / "cpu-source.json",
        {
            "primary": PRIMARY,
            "imports": api.imports,
            "author": author,
            "cases": len(cases),
            "elastic_DOF": model.ndof,
            "stiffness_sha256": stiffness_before,
            "source_unchanged": True,
            "semantics": "Original global U stiffness/mass/relief. Per-source both-side samples receive one pressure fit. Component selection uses independent positive Q4 integration and authored edge adjacency.",
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

    api = _api()
    import torch

    from genesis.engine.solvers.rigid.stress.contact import kernel_pressure
    from genesis.engine.solvers.rigid.stress.planar_loading import PlanarLoading
    from genesis.engine.solvers.rigid.stress.scatter import kernel_scatter_faces
    from genesis.utils.array_class import CHECKPOINT_KINDS
    from genesis.utils.misc import qd_to_torch

    return SimpleNamespace(
        **(vars(api) | {name: value for name, value in locals().items() if name not in ("api", "_api")})
    )


def _errors(np, actual, expected, weights):
    errors = {}
    multiplicity = np.array([1.0, 1.0, 2.0])
    for key in ("stress_local", "stress_full9", "N", "M", "Q"):
        value, delta = expected[key], actual[key] - expected[key]
        if key == "stress_local":
            numerator = np.einsum("segcb,segcb,eg,c->b", delta, delta, weights, multiplicity)
            denominator = np.einsum("segcb,segcb,eg,c->b", value, value, weights, multiplicity)
            output_key = "field"
        elif key == "stress_full9":
            numerator = np.einsum("segcb,segcb,eg->b", delta, delta, weights)
            denominator = np.einsum("segcb,segcb,eg->b", value, value, weights)
            output_key = "full_field"
        else:
            scale = np.ones(2) if key == "Q" else multiplicity
            numerator = np.einsum("egcb,egcb,eg,c->b", delta, delta, weights, scale)
            denominator = np.einsum("egcb,egcb,eg,c->b", value, value, weights, scale)
            output_key = key
        errors[output_key] = np.sqrt(numerator) / np.maximum(np.sqrt(denominator), 1e-30)
    errors["peak"] = np.abs(actual["peak"] - expected["peak"]) / np.maximum(expected["peak"], 1e-30)
    return errors


def _decode(api, field, n_parents):
    np = api.np
    packed = api.unbatched_array(api, field)[..., 0].astype(np.uint32)
    parents = np.arange(n_parents)
    return ((packed[parents // 32] >> (parents % 32)[:, None, None]) & 1).astype(bool)


def _upload_contacts(api, contacts, values, envs=None, reverse=False, release=False):
    np = api.np
    for name, original in values.items():
        value = original.copy()
        if envs is not None:
            value = value[:, envs]
        if reverse:
            value = value[::-1].copy()
        if release and name == "valid":
            value.fill(False)
        vars(contacts)[name].from_numpy(value.astype(np.bool_ if name == "valid" else api.gs.np_float, copy=False))


def _snapshot_contacts(api, contacts):
    return {
        name: api.unbatched_array(api, vars(contacts)[name]).copy()
        for name in ("position", "force", "normal", "friction", "radius", "valid")
    }


def _data_snapshot(api, loading):
    saved = []
    for item in loading.data("material_check"):
        if item.kind in api.CHECKPOINT_KINDS and all(item.value.shape):
            saved.append((item.name, item.value, api.qd_to_torch(item.value, transpose=True, copy=False).clone()))
    return saved


def _check_run(api, writer, checks, label, actual, expected, numerics, free, oracle_k, loaded, quadrature):
    np = api.np
    balanced = actual["force"] - numerics["mass_modes"] @ np.linalg.solve(
        numerics["mass_gram"], numerics["rigid_modes"].T @ actual["force"]
    )
    checks.close(f"{label}/common_global_relief", actual["rhs"], balanced, rtol=1e-10, atol=1e-13)
    residual = (oracle_k @ actual["q"] - actual["rhs"])[free]
    norm, rhs_norm = np.linalg.norm(residual, axis=0), np.linalg.norm(actual["rhs"][free], axis=0)
    checks.condition(
        f"{label}/original_global_K_residual",
        (norm <= np.maximum(1e-11, 1e-8 * rhs_norm)).all(),
        maximum_relative=float(np.max(norm / np.maximum(rhs_norm, 1e-30))),
    )
    checks.condition(f"{label}/native_admission", actual["valid"].all())
    errors = _errors(np, actual, expected, numerics["gauss_weights"])
    maxima = {key: float(value[loaded].max(initial=0)) for key, value in errors.items()}
    load_error = np.linalg.norm(actual["force"] - expected["raw"], axis=0) / np.maximum(
        np.linalg.norm(expected["raw"], axis=0), 1e-30
    )
    reference_errors = {"load": float(load_error[loaded].max(initial=0)), **maxima}
    if quadrature == 48:
        checks.condition(
            f"{label}/original_Q4_reference_consistency",
            all(value <= REFERENCE_TOLERANCE for value in reference_errors.values()),
            maxima=reference_errors,
            tolerance=REFERENCE_TOLERANCE,
        )
    mean = float(errors["field"][loaded].mean()) if loaded.any() else 0.0
    checks.condition(
        f"{label}/spatial_budgets",
        mean <= BUDGETS["mean_field"] and all(maxima[key] <= BUDGETS[key] for key in maxima),
        mean_field=mean,
        maxima=maxima,
        budgets=BUDGETS,
    )
    writer.arrays(
        f"{label}/outputs.npz",
        **actual,
        error_load=load_error,
        **{f"error_{key}": value for key, value in errors.items()},
    )
    return {
        "label": label,
        "mean_field": mean,
        "maximum_errors": maxima,
        "reference_errors": reference_errors,
        "strict_reference_gate_applicable": quadrature == 48,
    }


def _mode(api, args, writer, checks, native, expected24, expected48, numerics, oracle_k, mode):
    np, model = api.np, native.model
    contacts, cases, valid, _sides = api.prepare_physical_contacts(
        api, args.output / "cpu", "material", model.options.young, model.options.poisson
    )
    values = _snapshot_contacts(api, contacts)
    n_envs, n_parents = len(cases), len(native.inputs["quads"])
    surface = api.StressSurface(args.quadrature, model.load_info, angular=True)
    loading = api.PlanarLoading(surface.info, model.load_info, contacts, args.refinement, tasks_per_env=1)
    state = model.create_state(n_envs, "full")
    omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(n_envs,))
    omega.fill(0.0)
    options = model.options.model_copy(
        update={"cooperative_pressure": mode != "serial", "cooperative_scatter": mode != "serial"}
    )
    capacity = 0 if mode == "serial" else 1 if mode == "face-overflow" else 128
    workspace = api.create_scatter_workspace(n_envs, capacity, reuse_moments=mode == "moments")
    free = api.unbatched_array(api, model.info.is_free).reshape(-1).astype(bool)
    phases = (
        "initial",
        "slot_reverse",
        "environment_reverse",
        "release_all",
        "reload",
        "partial_reset",
        "checkpoint_arrays",
    )
    records, baseline = [], None
    saved = None
    for phase in phases:
        envs, reverse, release = None, False, False
        if phase == "slot_reverse":
            reverse = True
        elif phase == "environment_reverse":
            envs = np.arange(n_envs - 1, -1, -1)
        elif phase == "release_all":
            release = True
        _upload_contacts(api, contacts, values, envs, reverse, release)
        if phase == "partial_reset":
            indices = api.torch.tensor([0], dtype=api.torch.int64, device=api.gs.device)
            loading.reset(indices)
            active = values["valid"].copy()
            active[:, 0] = False
            contacts.valid.from_numpy(active)
        elif phase == "checkpoint_arrays":
            if saved is None:
                raise RuntimeError("The loading data snapshot is missing.")
            loading.reset(api.torch.arange(n_envs, dtype=api.torch.int64, device=api.gs.device))
            for name, field, value in saved:
                view = api.qd_to_torch(field, transpose=True, copy=False)
                view.copy_(value)
                checks.condition(f"{mode}/{phase}/restore/{name}", api.torch.equal(view, value))
            api.qd.sync()
        api.kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
        loading.scatter(state, model.load_info, surface.info, options, workspace, mode == "moments")
        model.recover(omega, state)
        actual = api.native_outputs(api, model, state)
        label = f"{mode}/{phase}"
        order = np.arange(n_envs) if envs is None else envs
        expected = {
            key: value[..., order].copy()
            for key, value in expected48.items()
            if key in ("raw", "balanced", "q", "stress_local", "stress_full9", "peak", "N", "M", "Q")
        }
        loaded = valid.any(axis=0)[order].copy()
        if release:
            for value in expected.values():
                value.fill(0)
            loaded.fill(False)
        elif phase == "partial_reset":
            for value in expected.values():
                value[..., 0] = 0
            loaded[0] = False
        records.append(
            _check_run(api, writer, checks, label, actual, expected, numerics, free, oracle_k, loaded, args.quadrature)
        )
        current_valid = valid[:, order].copy()
        if reverse:
            current_valid = current_valid[::-1]
        if release:
            current_valid.fill(False)
        elif phase == "partial_reset":
            current_valid[:, 0] = False
        queried = api.unbatched_array(api, loading.material_state.queried)
        status = api.unbatched_array(api, loading.material_state.status)
        checks.exact(f"{label}/material_query_independent_of_prefix", queried, current_valid)
        checks.condition(f"{label}/material_success", not status[current_valid].any())
        hits = _decode(api, loading.material_state.hits, n_parents)
        selected = _decode(api, loading.material_state.mask, n_parents)
        expected_hits, expected_selected = expected24["hits"][:, :, order], expected24["selected"][:, :, order]
        if reverse:
            expected_hits, expected_selected = expected_hits[:, ::-1], expected_selected[:, ::-1]
        checks.exact(f"{label}/positive_parents", hits[:, current_valid], expected_hits[:, current_valid])
        checks.exact(f"{label}/selected_component", selected[:, current_valid], expected_selected[:, current_valid])
        selected_count = api.unbatched_array(api, loading.material_state.selected_count)
        excluded_count = api.unbatched_array(api, loading.material_state.excluded_count)
        checks.exact(
            f"{label}/selected_count", selected_count[current_valid], expected_selected.sum(axis=0)[current_valid]
        )
        checks.exact(
            f"{label}/excluded_count",
            excluded_count[current_valid],
            (expected_hits & ~expected_selected).sum(axis=0)[current_valid],
        )
        checks.condition(f"{label}/other_arm_witness", (excluded_count[current_valid] > 0).all())
        checks.condition(f"{label}/both_sides_enabled", not api.unbatched_array(api, loading.state.side_filter).any())
        fallback = api.unbatched_array(api, loading.state.fallback)
        checks.exact(f"{label}/bounded_proxy_capacity_fallback", fallback, current_valid)
        force_error = api.unbatched_array(api, contacts.force_error)
        moment_error = api.unbatched_array(api, contacts.moment_error)
        force = values["force"][:, order]
        radius = values["radius"][:, order]
        if reverse:
            force, radius = force[::-1], radius[::-1]
        magnitude = np.linalg.norm(force, axis=-1)
        checks.condition(
            f"{label}/original_source_wrench",
            (force_error[current_valid] <= 1e-8 * magnitude[current_valid]).all()
            and (moment_error[current_valid] <= 1e-8 * magnitude[current_valid] * radius[current_valid]).all(),
        )
        for i_b, case_index in enumerate(order):
            case = cases[case_index]
            if len(case["contacts"]) == 1 and loaded[i_b]:
                other = (
                    native.inputs["vertices"][:, 0] > 0.0021
                    if case["category"] == "left"
                    else native.inputs["vertices"][:, 0] < 0.0009
                )
                far = np.repeat(other, 6)
                checks.condition(
                    f"{label}/far_component_raw_zero/{case['name']}", not np.count_nonzero(actual["force"][far, i_b])
                )
        writer.arrays(
            f"{label}/routing.npz",
            queried=queried,
            material_status=status,
            hits=hits,
            selected=selected,
            selected_count=selected_count,
            excluded_count=excluded_count,
            fallback=fallback,
            reason=api.unbatched_array(api, loading.state.reason),
            force_error=force_error,
            moment_error=moment_error,
        )
        if baseline is None:
            baseline = actual
            saved = _data_snapshot(api, loading)
        else:
            for key in ("force", "rhs", "q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
                target = baseline[key][..., order].copy()
                if release:
                    target.fill(0)
                elif phase == "partial_reset":
                    target[..., 0] = 0
                checks.close(
                    f"{label}/same_current_load/{key}",
                    actual[key],
                    target,
                    rtol=1e-8,
                    atol=1e-10 if key.startswith("stress") or key == "peak" else 1e-13,
                )
    if mode == "face-overflow":
        checks.condition(
            "face-overflow/original_overflow_witness",
            api.qd_to_numpy(workspace.overflow_calls, transpose=True).item() > 0,
        )
    return {
        "mode": mode,
        "phases": records,
        "scope": "The isolated loading data checkpoint contract and source replay are exercised. Full scene checkpoint APIs require the focused integration tests.",
    }


def _unmasked_native(api, args, writer, native, expected, numerics):
    model = native.model
    contacts, cases, _valid, _sides = api.prepare_physical_contacts(
        api, args.output / "cpu", "material", model.options.young, model.options.poisson
    )
    surface = api.StressSurface(args.quadrature, model.load_info, angular=True)
    state = model.create_state(len(cases), "full")
    workspace = api.create_scatter_workspace(len(cases), 128, False)
    omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(len(cases),))
    omega.fill(0.0)
    api.kernel_anchor(float(api.np.finfo(api.np.float64).eps), contacts, surface.info)
    api.kernel_pressure(contacts, surface.info, True)
    api.kernel_scatter_faces(contacts, state, model.load_info, surface.info, workspace, True, False)
    model.recover(omega, state)
    actual = api.native_outputs(api, model, state)
    writer.arrays("unmasked-native-output.npz", **actual)
    errors = _errors(api.np, actual, expected, numerics["gauss_weights"])
    loaded = expected["peak"] > 0
    return {
        "native_admitted": actual["valid"].tolist(),
        "maximum_errors_against_own_unmasked_CPU": {
            key: float(value[loaded].max(initial=0)) for key, value in errors.items()
        },
        "surface_mask_disabled": surface.info.face_mask.shape[0] == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--refinement", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--quadrature", type=int, choices=(16, 24, 48), default=48)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
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
        "schema": "genesis.planar-material-check.v1",
        "complete": False,
        "passed": False,
        "worker": _identity(__file__),
        "reference": _identity(args.reference / "manifest.json"),
        "CPU_worker_returncode": None,
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "native_quadrature": args.quadrature,
        "strict_reference_gate_applicable": args.quadrature == 48,
        "strict_reference_tolerance": REFERENCE_TOLERANCE,
        "native_input_policy": "Authored physical U vertices, directors, topology, material and original sources. CPU component masks, loads and fields are oracle-only.",
        "scope": "Untimed connected-material original-route fallback and isolated loading lifecycle, using unchanged global U elasticity.",
    }
    _json(args.output / "result.json", result)
    _json(
        args.output / "protocol.json",
        {
            "primary": PRIMARY,
            "budgets": BUDGETS,
            "integration_uncertainty": 1e-3,
            "point_x_m": [0.0007, 0.0023],
            "point_y_m": 0.0025,
            "radius_m": [0.0016, 0.0018],
            "neck_y_max_m": 0.0003,
            "proxy_tasks_per_env": 1,
            "modes": list(args.modes),
            "semantics": "Both sides per selected positive parent component. One nonnegative pressure fit per source preserves its original point wrench.",
        },
    )
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if manifest["primary_commit"] != PRIMARY or not manifest["passed"] or not manifest["complete"]:
        raise RuntimeError("Use the successful pinned planar-loading reference.")
    snapshot = json.loads((args.reference / "source-snapshot.json").read_text())
    for row in snapshot["files"]:
        if _identity(args.reference / row["snapshot_path"])["sha256"] != row["sha256"]:
            raise RuntimeError(f"The pinned CPU source changed: {row['snapshot_path']}")
    cpu = args.output / "cpu"
    cpu.mkdir()
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_cpu-worker",
        "--reference",
        str(args.reference),
        "--output",
        str(cpu),
    ]
    with (args.output / "cpu.log").open("w") as stream:
        completed = subprocess.run(
            command,
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
            "planar_material.py",
            "planar_geometry.py",
            "contact.py",
            "scatter.py",
            "surface.py",
        )
    ]
    result.update(identity=api.runtime_identity(), native_sources=source_before)
    writer.json("result.json", result)
    started = time.perf_counter()
    try:
        native = api.build_native_mother(api, cpu, args.output, "material", method="direct", quadrature=args.quadrature)
        expected24, expected48 = _load(api.np, cpu / "oracles/q24.npz"), _load(api.np, cpu / "oracles/q48.npz")
        numerics = _load(api.np, cpu / "oracles/numerics.npz")
        oracle_k = api.sparse.load_npz(cpu / "oracles/K.npz")
        actual_k = api.sparse.bsr_matrix(
            (
                api.unbatched_array(api, native.model.info.stiffness) * native.model.options.young,
                api.unbatched_array(api, native.model.info.columns),
                api.unbatched_array(api, native.model.info.row_start),
            ),
            shape=oracle_k.shape,
        ).tocsr()
        checks.close("operators/unchanged_global_K", actual_k, oracle_k, rtol=1e-10, atol=1e-8)
        checks.close(
            "operators/global_mass_modes",
            api.unbatched_array(api, native.model.info.mass_modes).reshape((-1, 6)),
            numerics["mass_modes"],
            rtol=1e-10,
            atol=1e-14,
        )
        checks.exact("CPU/q24_q48_parent_hits", expected24["hits"], expected48["hits"])
        checks.exact("CPU/q24_q48_selected_component", expected24["selected"], expected48["selected"])
        loaded = expected48["peak"] > 0
        uncertainty = _errors(api.np, expected24, expected48, numerics["gauss_weights"])
        uncertainty["load"] = api.np.linalg.norm(expected24["raw"] - expected48["raw"], axis=0) / api.np.maximum(
            api.np.linalg.norm(expected48["raw"], axis=0), 1e-30
        )
        uncertainty_maxima = {key: float(value[loaded].max(initial=0)) for key, value in uncertainty.items()}
        resolved = all(value <= REFERENCE_TOLERANCE for value in uncertainty_maxima.values())
        checks.condition(
            "CPU/independently_resolved_quadrature",
            resolved,
            maxima=uncertainty_maxima,
            tolerance=REFERENCE_TOLERANCE,
        )
        result["CPU_integration_uncertainty"] = uncertainty_maxima
        if not resolved:
            raise ArithmeticError("Resolve the independent CPU q24/q48 integration before evaluating native U loads.")
        records = json.loads((cpu / "oracles/q48.json").read_text())
        checks.condition("CPU/disconnected_positive_patch_witness", all(row["excluded_parents"] for row in records))
        checks.condition("CPU/both_sides_fit_together", all(row["both_sides"] == [-1, 1] for row in records))
        result["modes"] = [
            _mode(api, args, writer, checks, native, expected24, expected48, numerics, oracle_k, mode)
            for mode in args.modes
        ]
        unmasked = _load(api.np, cpu / "oracles/unmasked-q48.npz")
        diagnostic = _errors(api.np, unmasked, expected48, numerics["gauss_weights"])
        native_diagnostic = _unmasked_native(api, args, writer, native, unmasked, numerics)
        writer.json(
            "unmasked-diagnostic.json",
            {
                "scope": "Independent unmasked original Q4 integration and the actual mask-disabled native route are diagnostics. They are not the selected-component oracle or a required spatial failure.",
                "maximum_errors": {key: float(value[loaded].max(initial=0)) for key, value in diagnostic.items()},
                **native_diagnostic,
            },
        )
        checks.close(
            "operators/K_after_loading",
            api.sparse.bsr_matrix(
                (
                    api.unbatched_array(api, native.model.info.stiffness) * native.model.options.young,
                    api.unbatched_array(api, native.model.info.columns),
                    api.unbatched_array(api, native.model.info.row_start),
                ),
                shape=oracle_k.shape,
            ).tocsr(),
            oracle_k,
            rtol=1e-10,
            atol=1e-8,
        )
        checks.condition(
            "provenance/native_source_unchanged", [_identity(row["path"]) for row in source_before] == source_before
        )
        checks.condition("provenance/worker_unchanged", _identity(__file__) == result["worker"])
        checks.condition(
            "provenance/reference_unchanged", _identity(args.reference / "manifest.json") == result["reference"]
        )
        result.update(
            complete=True,
            passed=all(row["passed"] for row in checks.rows),
            checks=len(checks.rows),
            seconds=time.perf_counter() - started,
            CPU_integration_uncertainty=uncertainty_maxima,
        )
    finally:
        checks.flush(writer)
        if sys.exc_info()[0] is not None:
            (args.output / "failure.txt").write_text(traceback.format_exc())
        writer.json("result.json", result)
    if not result["passed"]:
        raise ArithmeticError("Material-component fallback qualification failed. Preserve all rejected evidence.")
    print(json.dumps({"passed": True, "checks": result["checks"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
