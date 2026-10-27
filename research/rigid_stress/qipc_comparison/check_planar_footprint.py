"""Check physical shared-edge components and reject disconnected single-Q4 footprints.

The two-panel book has one authored hinge outside each tip footprint. The CPU
oracle tests the physical edge segments independently and uses the frozen Q4
elasticity and pressure fit. A strong saddle supplies an analytical rejection
primitive, without an elastic qualification claim. Run science in Slurm rtx-mid
with image 1_26.
"""

import argparse
import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

PRIMARY = "797203553f246ca155bbef909c1bb33e7cd7b90c"
MODES = ("faces", "moments", "serial", "face-overflow")
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


def _finite_json(value):
    """Retain nonfinite diagnostics as explicit nulls and count them without changing acceptance."""
    count = 0
    if isinstance(value, float) and not math.isfinite(value):
        value, count = None, 1
    elif isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[key], nonfinite = _finite_json(item)
            count += nonfinite
        value = result
    elif isinstance(value, (tuple, list)):
        result = []
        for item in value:
            clean, nonfinite = _finite_json(item)
            result.append(clean)
            count += nonfinite
        value = result
    return value, count


def _book(np, fixture):
    tip_x, tip_z = (20.0, 0.4) if fixture == "closed" else (10.0 * math.sqrt(3.0), 10.0)
    vertices = (
        np.array([[0, -5, 0], [0, 5, 0], [20, -5, 0], [20, 5, 0], [tip_x, -5, tip_z], [tip_x, 5, tip_z]], dtype=float)
        * 1e-3
    )
    quads = np.array([[0, 2, 3, 1], [0, 1, 5, 4]])
    normal_a = np.array([0.0, 0.0, 1.0])
    normal_b = np.cross(vertices[1] - vertices[0], vertices[4] - vertices[0])
    normal_b /= np.linalg.norm(normal_b)
    hinge = normal_a + normal_b
    hinge /= np.linalg.norm(hinge)
    directors = np.stack((hinge, hinge, normal_a, normal_a, normal_b, normal_b))
    return vertices, quads, directors


def _source(np, vertices, quads, directors, thickness, parent, side, x, friction, radius):
    xi, eta = (2 * x / 0.02 - 1, 0.0) if parent == 0 else (0.0, 2 * x / 0.02 - 1)
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    shape = 0.25 * (1 + xi * signs[:, 0]) * (1 + eta * signs[:, 1])
    dxi = 0.25 * signs[:, 0] * (1 + eta * signs[:, 1])
    deta = 0.25 * signs[:, 1] * (1 + xi * signs[:, 0])
    physical = (vertices + side * thickness * 0.5 * directors)[quads[parent]]
    first, second = dxi @ physical, deta @ physical
    normal = side * np.cross(first, second)
    normal /= np.linalg.norm(normal)
    tangent = first / np.linalg.norm(first)
    return {
        "point": (shape @ physical).tolist(),
        "force": (-0.05 * normal + (0.008 * tangent if friction else 0.0)).tolist(),
        "normal": normal.tolist(),
        "radius": radius,
        "friction": 0.5,
        "curvatures": [50.0, 70.0],
        "side": side,
        "face_hint": parent,
    }


def _cases(np, vertices, quads, directors, thickness, fixture):
    radius = 0.001 if fixture == "closed" else 0.009
    cases = []
    for parent in (0, 1):
        for side in (-1, 1):
            for friction in (False, True):
                cases.append(
                    {
                        "name": f"tip-p{parent}-s{side}-{'friction' if friction else 'normal'}",
                        "category": "tip",
                        "expected_parents": [parent],
                        "contacts": [
                            _source(np, vertices, quads, directors, thickness, parent, side, 0.016, friction, radius)
                        ],
                    }
                )
            cases.append(
                {
                    "name": f"hinge-p{parent}-s{side}",
                    "category": "hinge",
                    "expected_parents": [0, 1],
                    "contacts": [
                        _source(np, vertices, quads, directors, thickness, parent, side, 0.75 * radius, True, radius)
                    ],
                }
            )
    cases.append(
        {
            "name": "opposed-tip-sources",
            "category": "multiple",
            "contacts": [cases[1]["contacts"][0], cases[10]["contacts"][0]],
        }
    )
    cases.append({"name": "release", "category": "release", "contacts": []})
    return cases


def _edge_component(np, model, source, hits, seed):
    """Resolve original-parent edges using physical segment distances, independently of native helpers."""
    edges = {}
    for parent, nodes in enumerate(model.quads):
        for i in range(4):
            edge = tuple(sorted((int(nodes[i]), int(nodes[(i + 1) % 4]))))
            edges.setdefault(edge, []).append(parent)
    center, radius = np.array(source["point"]), source["radius"]
    adjacent, witnesses = [set() for _ in model.quads], []
    for edge, parents in edges.items():
        if len(parents) < 2:
            continue
        distances = []
        for side in (-1, 1):
            physical = model.vertices + side * model.h * 0.5 * model.directors
            first, second = physical[list(edge)]
            delta = second - first
            fraction = np.clip(np.dot(center - first, delta) / np.dot(delta, delta), 0, 1)
            distances.append(float(np.linalg.norm(first + fraction * delta - center)))
        connected = min(distances) < radius
        witnesses.append({"edge": list(edge), "parents": parents, "distance_m": distances, "connected": connected})
        if connected:
            for parent in parents:
                adjacent[parent].update(other for other in parents if other != parent)
    selected, todo = {seed}, [seed]
    while todo:
        parent = todo.pop()
        for other in adjacent[parent] & hits - selected:
            selected.add(other)
            todo.append(other)
    if seed not in hits:
        raise ArithmeticError("The book source has no positive footprint on its authored anchor.")
    return selected, witnesses


def _source_load(api, model, source, order, restrict):
    # The sample integrator and pressure fitter are frozen; only component geometry is authored here.
    np = api.np
    samples, hits, seed = api.q4_samples(api, model, source, order)
    component, witnesses = _edge_component(np, model, source, hits, seed)
    selected = component if restrict else hits
    active = [row for row in samples if row[0] in selected]
    positions, weights = np.vstack([row[2] for row in active]), np.concatenate([row[3] for row in active])
    patch = api.WrenchPatch(
        np.array(source["point"]),
        np.array(source["force"]),
        source["radius"],
        source["friction"],
        np.zeros(3),
        -np.array(source["normal"]),
    )
    forces, diagnostics = api.PadPressureFit(positions, weights, patch).solve()
    raw, offset = np.zeros((len(model.vertices), 6)), 0
    for parent, side, _positions, local_weights, shape in active:
        count = len(local_weights)
        nodal = shape.T @ forces[offset : offset + count]
        nodes = model.quads[parent]
        np.add.at(raw[:, :3], nodes, nodal)
        np.add.at(raw[:, 3:], nodes, side * model.h * 0.5 * np.cross(model.directors[nodes], nodal))
        offset += count
    magnitude = np.linalg.norm(source["force"])
    force_error = np.linalg.norm(raw[:, :3].sum(axis=0) - source["force"]) / magnitude
    moment = np.cross(model.vertices - source["point"], raw[:, :3]).sum(axis=0) + raw[:, 3:].sum(axis=0)
    moment_error = np.linalg.norm(moment) / (magnitude * source["radius"])
    if max(force_error, moment_error) > 1e-8:
        raise ArithmeticError("The independent book pressure fit violates the original source wrench.")
    return raw.reshape(-1), {
        "seed": seed,
        "hit_parents": sorted(hits),
        "selected_parents": sorted(selected),
        "excluded_parents": sorted(hits - selected),
        "both_sides": sorted({row[1] for row in active}),
        "edge_witnesses": witnesses,
        "force_error": float(force_error),
        "moment_error": float(moment_error),
        "fit_evaluations": diagnostics.constrained_evaluations,
    }


def _proxy_source_load(api, model, source, proxy, order, selected):
    """Integrate both frozen P1 proxy sides on independently selected authored components."""
    np = api.np
    point, radius = np.array(source["point"]), source["radius"]
    samples = []
    for side, surface in ((1, proxy.outer), (-1, proxy.inner)):
        for triangle, vertices in enumerate(surface.vertices):
            parent = int(surface.parent_faces[triangle])
            separation = np.maximum(vertices.min(axis=0) - point, 0) + np.maximum(point - vertices.max(axis=0), 0)
            if parent not in selected or separation @ separation >= radius**2:
                continue
            normal = surface.normals[triangle]
            height = (point - vertices[0]) @ normal
            remaining = radius**2 - height**2
            if remaining <= 0:
                continue
            center, plane = point - height * normal, api._basis(normal)
            polygon = (vertices - center) @ plane
            points, weights = api.ShellDPRTransfer._face_polar_samples(polygon, np.sqrt(remaining), order)
            if not len(weights):
                continue
            positions = center + points @ plane.T
            local = (positions - vertices[0]) @ surface.dual[triangle].T
            barycentric = np.column_stack((1 - local.sum(axis=1), local))
            if barycentric.min() < -2e-8:
                raise ArithmeticError("Independent P1 proxy quadrature leaves its material triangle.")
            shapes = barycentric @ surface.parent_shapes[triangle]
            weights = weights * (2 * np.pi * remaining**1.5 / (3 * radius))
            samples.append((parent, side, positions, weights, shapes))
    positions = np.vstack([row[2] for row in samples])
    weights = np.concatenate([row[3] for row in samples])
    patch = api.WrenchPatch(
        point, np.array(source["force"]), radius, source["friction"], np.zeros(3), -np.array(source["normal"])
    )
    forces, diagnostics = api.PadPressureFit(positions, weights, patch).solve()
    raw, offset = np.zeros((len(model.vertices), 6)), 0
    for parent, side, _positions, local_weights, shapes in samples:
        count = len(local_weights)
        nodal = shapes.T @ forces[offset : offset + count]
        nodes = model.quads[parent]
        np.add.at(raw[:, :3], nodes, nodal)
        np.add.at(raw[:, 3:], nodes, side * model.h * 0.5 * np.cross(model.directors[nodes], nodal))
        offset += count
    magnitude = np.linalg.norm(source["force"])
    force_error = np.linalg.norm(raw[:, :3].sum(axis=0) - source["force"]) / magnitude
    moment = np.cross(model.vertices - point, raw[:, :3]).sum(axis=0) + raw[:, 3:].sum(axis=0)
    moment_error = np.linalg.norm(moment) / (magnitude * radius)
    if max(force_error, moment_error) > 1e-8:
        raise ArithmeticError("The independent P1 pressure fit violates the original source wrench.")
    return raw.reshape(-1), {
        "proxy_samples": len(weights),
        "proxy_force_error": float(force_error),
        "proxy_moment_error": float(moment_error),
        "proxy_fit_evaluations": diagnostics.constrained_evaluations,
    }


def _cpu_fields(np, model, raw, refinements):
    """Refine the independent original-LU solve against the unchanged unscaled matrix."""
    balanced, acceleration = model.balance_load(raw)
    q = model.solve(raw)
    factor = model.factorize()
    free = factor.free_dofs
    residual = model.K @ q - balanced
    best_norm = np.linalg.norm(residual[free], axis=0)
    best_q, selected = q.copy(), np.zeros(raw.shape[-1], dtype=int)
    history = [best_norm.copy()]
    for correction in range(refinements):
        delta = factor.lu.solve(np.asfortranarray(-residual[free]))
        q[free] += delta
        residual = model.K @ q - balanced
        norm = np.linalg.norm(residual[free], axis=0)
        history.append(norm)
        improved = norm < best_norm
        best_q[:, improved], best_norm[improved], selected[improved] = q[:, improved], norm[improved], correction + 1
    fields = model.stress(best_q)
    local = np.stack((fields["local_top"], fields["local_bottom"]))
    xx, yy, xy = local[:, :, :, 0], local[:, :, :, 1], local[:, :, :, 2]
    peak = np.sqrt(np.maximum(0, xx * xx + yy * yy - xx * yy + 3 * xy * xy)).max(axis=(0, 1, 2))
    return (
        {
            "raw": raw,
            "balanced": balanced,
            "relief_acceleration": acceleration,
            "q": best_q,
            "stress_local": local,
            "stress_full9": np.stack((fields["sigma_top"], fields["sigma_bottom"])).reshape(
                (2, len(model.quads), 4, 9, raw.shape[-1])
            ),
            "peak": peak,
            "N": fields["N"],
            "M": fields["M"],
            "Q": fields["Q"],
        },
        np.stack(history),
        selected,
    )


def _cpu_worker(args):
    # Loading this helper as a standalone file leaves the research package unbound until the frozen import.
    support_path = Path(__file__).with_name("check_planar_material.py")
    support_identity = _identity(support_path)
    support = runpy.run_path(str(support_path))
    base = support["_cpu_api"](args.reference)
    from research.rigid_stress.triangular_load_cpu.proxy import TriangularLoadTransfer

    api = SimpleNamespace(
        **(vars(base) | {"q4_samples": support["_q4_samples"], "TriangularLoadTransfer": TriangularLoadTransfer})
    )
    api.imports.append(_identity(sys.modules[TriangularLoadTransfer.__module__].__file__))
    np = api.np
    vertices, quads, directors = _book(np, args.fixture)
    thickness, young, poisson, density, drilling = 5e-5, 1e10, 0.3, 2000.0, 1e-3
    model = api.ShellModel(
        vertices,
        quads,
        E=young,
        nu=poisson,
        h=thickness,
        density=density,
        directors=directors,
        gauge="translation",
        drilling_factor=drilling,
    )
    before = hashlib.sha256(model.K.data.tobytes() + model.K.indices.tobytes() + model.K.indptr.tobytes()).hexdigest()
    destination, oracle = args.output / "inputs/book", args.output / "oracles"
    destination.mkdir(parents=True)
    oracle.mkdir()
    np.savez_compressed(
        destination / "model.npz",
        vertices=vertices,
        quads=quads,
        directors=directors,
        thickness=thickness,
        young=young,
        poisson=poisson,
        density=density,
        drilling_factor=drilling,
    )
    cases = _cases(np, vertices, quads, directors, thickness, args.fixture)
    _json(destination / "cases.json", cases)
    api.sparse.save_npz(oracle / "K.npz", model.K)
    np.savez_compressed(
        oracle / "numerics.npz",
        mass_modes=model.mass_modes,
        rigid_modes=model.rigid_modes,
        mass_gram=model.mass_gram,
        gauss_weights=model.gauss_weights,
        gauge_dofs=model.gauge_dofs,
    )
    methods = []
    proxy = api.TriangularLoadTransfer(model, args.refinement)
    with api.threadpool_limits(limits=1):
        for order, restrict, label in (
            (24, True, "q24"),
            (48, True, "q48"),
            (48, False, "unmasked-q48"),
            (24, True, "proxy-q24"),
            (48, True, "proxy-q48"),
        ):
            raw = np.zeros((model.ndof, len(cases)))
            source_raw = np.zeros((model.ndof, 2, len(cases)))
            hits = np.zeros((2, 2, len(cases)), dtype=bool)
            selected = np.zeros_like(hits)
            rows = []
            for i_b, case in enumerate(cases):
                for i_c, source in enumerate(case["contacts"]):
                    load, record = _source_load(api, model, source, order, restrict)
                    if label.startswith("proxy-") and case["category"] == "hinge":
                        load, proxy_record = _proxy_source_load(
                            api, model, source, proxy, order, set(record["selected_parents"])
                        )
                        record.update(proxy_record)
                    if (
                        restrict
                        and len(case["contacts"]) == 1
                        and record["selected_parents"] != case["expected_parents"]
                    ):
                        raise ArithmeticError(
                            "Independent physical-edge components disagree with authored book labels."
                        )
                    if record["hit_parents"] != [0, 1]:
                        raise ArithmeticError("The book must hit both panels despite its remote hinge.")
                    raw[:, i_b] += load
                    source_raw[:, i_c, i_b] = load
                    hits[record["hit_parents"], i_c, i_b] = True
                    selected[record["selected_parents"], i_c, i_b] = True
                    rows.append({"environment": i_b, "slot": i_c, "case": case["name"], **record})
            fields, refinement_history, selected_correction = _cpu_fields(np, model, raw, args.cpu_refinements)
            free = model.factorize().free_dofs
            residual_norm = np.linalg.norm((model.K @ fields["q"] - fields["balanced"])[free], axis=0)
            rhs_norm = np.linalg.norm(fields["balanced"][free], axis=0)
            np.savez_compressed(
                oracle / f"{label}.npz",
                **fields,
                source_raw=source_raw,
                hits=hits,
                selected=selected,
                original_K_residual_norm=residual_norm,
                original_rhs_norm=rhs_norm,
                original_K_refinement_residual_norm=refinement_history,
                selected_correction=selected_correction,
            )
            _json(oracle / f"{label}.json", rows)
            methods.append(
                {
                    "method": label,
                    "sources": len(rows),
                    "original_K_residual_absolute": residual_norm.tolist(),
                    "original_K_residual_relative": (residual_norm / np.maximum(rhs_norm, 1e-30)).tolist(),
                    "original_K_gate_passed": (residual_norm <= np.maximum(1e-11, 1e-8 * rhs_norm)).tolist(),
                    "original_K_refinement_residual_absolute": refinement_history.tolist(),
                    "original_K_refinement_residual_relative": (
                        refinement_history / np.maximum(rhs_norm[None], 1e-30)
                    ).tolist(),
                    "selected_correction": selected_correction.tolist(),
                }
            )
    after = hashlib.sha256(model.K.data.tobytes() + model.K.indices.tobytes() + model.K.indptr.tobytes()).hexdigest()
    if (
        before != after
        or [_identity(row["path"]) for row in api.imports] != api.imports
        or _identity(support_path) != support_identity
    ):
        raise RuntimeError("The frozen CPU source or original book stiffness changed.")
    free = model.factorize().free_dofs
    free_matrix = model.K[free][:, free].toarray()
    diagonal = np.sqrt(np.diag(free_matrix))
    condition = np.linalg.cond(free_matrix)
    scaled_condition = np.linalg.cond(free_matrix / diagonal[:, None] / diagonal[None, :])
    _json(
        args.output / "cpu-source.json",
        {
            "primary": PRIMARY,
            "fixture": args.fixture,
            "cpu_refinements": args.cpu_refinements,
            "imports": api.imports,
            "research_support": support_identity,
            "stiffness_sha256": before,
            "methods": methods,
            "gauge_dofs": model.gauge_dofs.tolist(),
            "free_K_condition_2": float(condition),
            "diagonally_scaled_free_K_condition_2": float(scaled_condition),
            "hinge_director_dot_panel_normal": [float(directors[0] @ model.axes[parent, 0, :, 2]) for parent in (0, 1)],
            "scope": "Original global book Q4 elasticity. Independent physical edge segment-ball component selection; authored tip and hinge labels are checked separately. Both sides share one source pressure fit.",
        },
    )


def _native_api():
    from .check_planar_guards import _native_api as guard_api
    from .check_planar_material import BUDGETS, _data_snapshot, _decode, _errors

    base = guard_api()
    import torch

    from genesis.engine.solvers.rigid.stress.planar_geometry import func_quad_graph_curvature, func_segment_ball
    from genesis.engine.solvers.rigid.stress.planar_loading import kernel_planar_query, kernel_planar_query_warp
    from genesis.utils.array_class import CHECKPOINT_KINDS
    from genesis.utils.misc import qd_to_torch

    return SimpleNamespace(
        **(vars(base) | {name: value for name, value in locals().items() if name not in ("base", "guard_api")})
    )


def _check_book_run(
    api, writer, checks, label, actual, expected, numerical, numerics, free, oracle_k, loaded, quadrature
):
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
    numeric_errors = api._errors(np, actual, numerical, numerics["gauss_weights"])
    numeric_errors["load"] = np.linalg.norm(actual["force"] - numerical["raw"], axis=0) / np.maximum(
        np.linalg.norm(numerical["raw"], axis=0), 1e-30
    )
    numeric_maxima = {key: float(value[loaded].max(initial=0)) for key, value in numeric_errors.items()}
    if quadrature == 48:
        checks.condition(
            f"{label}/same_loading_space_numerical_consistency",
            all(value <= TOLERANCE for value in numeric_maxima.values()),
            maxima=numeric_maxima,
            tolerance=TOLERANCE,
            reference="Both-side independent original Q4 for fallback, both-side independent frozen P1 proxy for proxy contacts.",
        )
    spatial_errors = api._errors(np, actual, expected, numerics["gauss_weights"])
    spatial_maxima = {key: float(value[loaded].max(initial=0)) for key, value in spatial_errors.items()}
    mean = float(spatial_errors["field"][loaded].mean()) if loaded.any() else 0.0
    checks.condition(
        f"{label}/original_Q4_loading_approximation_budgets",
        mean <= api.BUDGETS["mean_field"] and all(value <= api.BUDGETS[key] for key, value in spatial_maxima.items()),
        mean_field=mean,
        maxima=spatial_maxima,
        budgets=api.BUDGETS,
    )
    writer.arrays(
        f"{label}/outputs.npz",
        **actual,
        **{f"numeric_error_{key}": value for key, value in numeric_errors.items()},
        **{f"spatial_error_{key}": value for key, value in spatial_errors.items()},
    )
    return {
        "label": label,
        "numerical_maxima": numeric_maxima,
        "spatial_maxima": spatial_maxima,
        "mean_field": mean,
        "strict_numerical_gate_applicable": quadrature == 48,
    }


def _book_mode(api, args, writer, checks, native, expected24, expected48, proxy48, numerics, oracle_k, mode, capacity):
    np, model = api.np, native.model
    contacts, cases, valid, _sides = api.prepare_physical_contacts(
        api, args.output / "cpu", "book", model.options.young, model.options.poisson
    )
    source = api._snapshot_contacts(api, contacts)
    n_envs = len(cases)
    surface = api.StressSurface(args.quadrature, model.load_info, angular=True)
    loading = api.PlanarLoading(surface.info, model.load_info, contacts, args.refinement, tasks_per_env=capacity)
    state = model.create_state(n_envs, "full")
    omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(n_envs,))
    omega.fill(0.0)
    options = model.options.model_copy(
        update={"cooperative_pressure": mode != "serial", "cooperative_scatter": mode != "serial"}
    )
    scatter_capacity = 0 if mode == "serial" else 1 if mode == "face-overflow" else 128
    workspace = api.create_scatter_workspace(n_envs, scatter_capacity, reuse_moments=mode == "moments")
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
    baseline, saved, records = None, None, []
    for phase in phases:
        order = np.arange(n_envs - 1, -1, -1) if phase == "environment_reverse" else np.arange(n_envs)
        reverse, release = phase == "slot_reverse", phase == "release_all"
        api._upload_contacts(api, contacts, source, envs=order, reverse=reverse, release=release)
        current_valid = valid[:, order].copy()
        if reverse:
            current_valid = current_valid[::-1].copy()
        if release:
            current_valid.fill(False)
        if phase == "partial_reset":
            loading.reset(api.torch.tensor([0], dtype=api.torch.int64, device=api.gs.device))
            current_valid[:, 0] = False
            contacts.valid.from_numpy(current_valid)
        elif phase == "checkpoint_arrays":
            if saved is None:
                raise RuntimeError("Missing isolated planar checkpoint arrays.")
            loading.reset(api.torch.arange(n_envs, dtype=api.torch.int64, device=api.gs.device))
            for name, field, value in saved:
                view = api.qd_to_torch(field, transpose=True, copy=False)
                view.copy_(value)
                checks.condition(f"{mode}/cap{capacity}/{phase}/restore/{name}", api.torch.equal(view, value))
            api.qd.sync()
        api.kernel_anchor(float(np.finfo(api.gs.np_float).eps), contacts, surface.info)
        centers = api.unbatched_array(api, contacts.center).copy()
        loading.scatter(state, model.load_info, surface.info, options, workspace, mode == "moments")
        model.recover(omega, state)
        actual = api.native_outputs(api, model, state)
        label = f"book/{mode}/cap{capacity}/{phase}"
        expected = {key: value[..., order].copy() for key, value in expected48.items()}
        numerical = {key: value[..., order].copy() for key, value in (expected48 if capacity == 1 else proxy48).items()}
        loaded = current_valid.any(axis=0)
        if release:
            for value in expected.values():
                value.fill(0)
            for value in numerical.values():
                value.fill(0)
        elif phase == "partial_reset":
            for value in expected.values():
                value[..., 0] = 0
            for value in numerical.values():
                value[..., 0] = 0
        records.append(
            _check_book_run(
                api,
                writer,
                checks,
                label,
                actual,
                expected,
                numerical,
                numerics,
                free,
                oracle_k,
                loaded,
                args.quadrature,
            )
        )
        queried = api.unbatched_array(api, loading.material_state.queried)
        material_status = api.unbatched_array(api, loading.material_state.status)
        fallback = api.unbatched_array(api, loading.state.fallback)
        reason = api.unbatched_array(api, loading.state.reason)
        expected_hits, expected_selected = expected24["hits"][:, :, order], expected24["selected"][:, :, order]
        if reverse:
            expected_hits, expected_selected = expected_hits[:, ::-1], expected_selected[:, ::-1]
        hits, selected = (
            api._decode(api, loading.material_state.hits, 2),
            api._decode(api, loading.material_state.mask, 2),
        )
        checks.exact(f"{label}/material_query_current_fallback", queried, fallback & current_valid)
        checks.condition(f"{label}/material_success", not material_status[queried].any())
        checks.exact(f"{label}/fallback_positive_parents", hits[:, queried], expected_hits[:, queried])
        checks.exact(f"{label}/fallback_selected_parents", selected[:, queried], expected_selected[:, queried])
        tips = np.array([case["category"] in ("tip", "multiple") for case in cases])[order]
        tip_valid = current_valid & tips[None, :]
        checks.condition(f"{label}/disconnected_tip_fallback", fallback[tip_valid].all())
        if capacity > 1:
            checks.condition(
                f"{label}/actual_material_rejection", (reason[tip_valid] == api.PlanarFallback.MATERIAL).all()
            )
            hinge_valid = current_valid & np.array([case["category"] == "hinge" for case in cases])[order][None, :]
            checks.condition(f"{label}/physical_hinge_connection_retains_proxy", not fallback[hinge_valid].any())
        else:
            checks.exact(f"{label}/capacity_fallback", fallback, current_valid)
        checks.condition(f"{label}/both_sides_quotient", not api.unbatched_array(api, loading.state.side_filter).any())
        magnitude = np.linalg.norm(source["force"][:, order][::-1] if reverse else source["force"][:, order], axis=-1)
        radius = source["radius"][:, order][::-1] if reverse else source["radius"][:, order]
        force_error = api.unbatched_array(api, contacts.force_error)
        moment_error = api.unbatched_array(api, contacts.moment_error)
        checks.condition(
            f"{label}/original_source_wrench",
            (force_error[current_valid] <= 1e-8 * magnitude[current_valid]).all()
            and (moment_error[current_valid] <= 1e-8 * magnitude[current_valid] * radius[current_valid]).all(),
        )
        for name in ("position", "force", "normal", "radius", "friction"):
            target = source[name][:, order][::-1] if reverse else source[name][:, order]
            checks.exact(f"{label}/unchanged_source/{name}", api.unbatched_array(api, vars(contacts)[name]), target)
        checks.exact(f"{label}/unchanged_anchor", api.unbatched_array(api, contacts.center), centers)
        nodal = actual["force"].reshape((6, 6, n_envs))
        for i_b, case_index in enumerate(order):
            case = cases[case_index]
            if loaded[i_b] and len(case["contacts"]) == 1:
                physical_source = case["contacts"][0]
                force, point = np.array(physical_source["force"]), np.array(physical_source["point"])
                resultant = nodal[:, :3, i_b].sum(axis=0)
                moment = np.cross(native.inputs["vertices"] - point, nodal[:, :3, i_b]).sum(axis=0) + nodal[
                    :, 3:, i_b
                ].sum(axis=0)
                checks.condition(
                    f"{label}/raw_source_wrench/{case['name']}",
                    np.linalg.norm(resultant - force) <= 1e-8 * np.linalg.norm(force)
                    and np.linalg.norm(moment) <= 1e-8 * np.linalg.norm(force) * physical_source["radius"],
                )
                if case["category"] == "tip":
                    far = [4, 5] if case["expected_parents"] == [0] else [2, 3]
                    checks.condition(
                        f"{label}/far_exclusive_nodes_zero/{case['name']}", not np.count_nonzero(nodal[far, :, i_b])
                    )
        if baseline is None:
            baseline, saved = actual, api._data_snapshot(api, loading)
        else:
            for key in ("force", "rhs", "q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
                target = baseline[key][..., order].copy()
                if release:
                    target.fill(0)
                elif phase == "partial_reset":
                    target[..., 0] = 0
                checks.close(
                    f"{label}/current_load_lifecycle/{key}",
                    actual[key],
                    target,
                    rtol=1e-8,
                    atol=1e-10 if key.startswith("stress") or key == "peak" else 1e-13,
                )
        writer.arrays(
            f"{label}/routing.npz",
            queried=queried,
            material_status=material_status,
            hits=hits,
            selected=selected,
            fallback=fallback,
            reason=reason,
            force_error=force_error,
            moment_error=moment_error,
            center=centers,
            contact_status=api.unbatched_array(api, contacts.status),
            radius_status=api.unbatched_array(api, contacts.radius_status),
            state_valid=api.qd_to_numpy(state.valid, transpose=True),
            step_valid=api.qd_to_numpy(state.step_valid, transpose=True),
            invalid_steps=api.qd_to_numpy(state.invalid_steps, transpose=True),
            rhs_norm_squared=api.qd_to_numpy(state.rhs_norm_squared, transpose=True),
            residual_norm_squared=api.qd_to_numpy(state.residual_norm_squared, transpose=True),
        )
    if mode == "face-overflow":
        checks.condition(
            f"book/{mode}/cap{capacity}/original_scatter_overflow",
            api.qd_to_numpy(workspace.overflow_calls, transpose=True).item() > 0,
        )
    return {"mode": mode, "proxy_capacity": capacity, "phases": records}


def _geometry_primitives(api, writer, checks):
    """Check scale invariance and explicit uncertainty without assembling elasticity."""
    np, qd = api.np, api.qd
    chart_rows, radii, expected, labels = [], [], [], []
    for scale in (1e-80, 1.0, 1e80):
        for label, lengths, radius, connected in (
            ("flat", (0.01, 0.01, 0.0), 0.001, True),
            ("weak", (0.01, 0.01, 0.002), 0.0002, True),
            ("strong", (0.001, 0.001, 0.006), 0.003, False),
        ):
            chart_rows.append(scale * np.diag(lengths))
            radii.append(scale * radius)
            expected.append(connected)
            labels.append(f"{label}/{scale:g}")
    chart_rows.extend(
        (
            [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
            [[1e-160, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1e-200]],
        )
    )
    radii.extend((0.1, 0.1, 0.1))
    expected.extend((False, False, False))
    labels.extend(("rank-deficient", "subnormal-aspect-ratio", "nonzero-mixed-underflow"))
    coefficients = api.V_VEC(3, dtype=api.gs.qd_float, shape=(len(chart_rows), 3))
    coefficients.from_numpy(np.array(chart_rows, dtype=api.gs.np_float))
    bounds = api.V(dtype=api.gs.qd_float, shape=(len(chart_rows),))
    graph_bound = api.func_quad_graph_curvature

    @qd.kernel
    def kernel_bounds(values: qd.Tensor, output: qd.Tensor, epsilon: float, minimum_normal: float):
        for i in range(output.shape[0]):
            output[i] = graph_bound(values[i, 0], values[i, 1], values[i, 2], epsilon, minimum_normal)

    epsilon, minimum_normal = float(np.finfo(api.gs.np_float).eps), float(np.finfo(api.gs.np_float).tiny)
    kernel_bounds(coefficients, bounds, epsilon, minimum_normal)
    actual_bounds = api.qd_to_numpy(bounds, transpose=True)
    certified = np.array(radii) * actual_bounds < 1 - 128 * epsilon
    checks.exact("geometry/scale_invariant_connected_chart_certificates", certified, np.array(expected))
    checks.condition("geometry/extreme_aspect_and_mixed_underflow_fail_closed", np.isinf(actual_bounds[-3:]).all())
    segment_rows, segment_radii, segment_expected = [], [], []
    for scale in (1e-80, 1.0, 1e80):
        for height, answer in ((0.0, (True, True)), (2.0, (False, True)), (1.0, (False, False))):
            segment_rows.append(scale * np.array([[-2, height, 0], [2, height, 0], [0, 0, 0]]))
            segment_radii.append(scale)
            segment_expected.append(answer)
    segment_rows.extend((np.zeros((3, 3)), np.array([[float("nan"), 0, 0], [1, 0, 0], [0, 0, 0]])))
    segment_radii.extend((1.0, 1.0))
    segment_expected.extend(((False, False), (False, False)))
    segments = api.V_VEC(3, dtype=api.gs.qd_float, shape=(len(segment_rows), 3))
    segments.from_numpy(np.array(segment_rows, dtype=api.gs.np_float))
    radius_field = api.V(dtype=api.gs.qd_float, shape=(len(segment_rows),))
    radius_field.from_numpy(np.array(segment_radii, dtype=api.gs.np_float))
    answers = api.V_VEC(2, dtype=api.gs.qd_int, shape=(len(segment_rows),))
    edge_witness = api.func_segment_ball

    @qd.kernel
    def kernel_segments(values: qd.Tensor, radius: qd.Tensor, output: qd.Tensor, epsilon: float, minimum_normal: float):
        for i in range(output.shape[0]):
            inside, certain = edge_witness(values[i, 0], values[i, 1], values[i, 2], radius[i], epsilon, minimum_normal)
            output[i] = qd.Vector([inside, certain])

    kernel_segments(segments, radius_field, answers, epsilon, minimum_normal)
    segment_actual = api.qd_to_numpy(answers, transpose=True).astype(bool)
    checks.exact("geometry/strict_open_ball_edge_witnesses_and_uncertainty", segment_actual, np.array(segment_expected))
    writer.arrays(
        "geometry/primitive-readback.npz",
        coefficients=np.array(chart_rows),
        radius=np.array(radii),
        curvature_bound=actual_bounds,
        certified=certified,
        segments=np.array(segment_rows),
        segment_radius=np.array(segment_radii),
        segment_answer=segment_actual,
    )
    writer.json(
        "geometry/primitive-labels.json",
        {
            "charts": labels,
            "segment_order": ["inside", "outside", "tangent"] * 3 + ["degenerate", "nonfinite"],
            "scope": "Analytical chart and segment guards; no pressure fit, scatter or elastic solve.",
        },
    )


def _strong_saddle(api, writer, checks):
    np = api.np
    thickness, radius = 5e-5, 0.003
    patch = np.array([[-1, -1, 6], [1, -1, -6], [1, 1, 6], [-1, 1, -6]]) * 1e-3
    vertices = np.vstack((patch + [0, 0, thickness / 2], patch - [0, 0, thickness / 2]))
    faces = np.zeros((2, 6), dtype=api.gs.np_int)
    faces[0, :4], faces[1, :4] = [0, 1, 2, 3], [4, 7, 6, 5]
    load_info = api.allocate_stress_info(8, 0, 2, 0, 0, 8)
    load_info.vertices.from_numpy(vertices.astype(api.gs.np_float, copy=False))
    load_info.surface_nodes.from_numpy(faces)
    load_info.load_nodes.fill(0)
    load_info.load_offsets.fill(0)
    surface = api.StressSurface(48, load_info, angular=True)
    point = np.array([0.8, 0.8, 3.84]) * 1e-3 + [0, 0, thickness / 2]
    normal = np.array([-4.8, -4.8, 1.0])
    normal /= np.linalg.norm(normal)
    source = {
        "position": np.broadcast_to(point, (2, 1, 3)).copy(),
        "force": np.broadcast_to(-0.05 * normal, (2, 1, 3)).copy(),
        "normal": np.broadcast_to(-normal, (2, 1, 3)).copy(),
        "radius": np.full((2, 1), radius),
        "friction": np.full((2, 1), 0.5),
        "valid": np.array([[True], [False]]),
    }
    opposite = np.array([-0.8, -0.8, 3.84]) * 1e-3 + [0, 0, thickness / 2]
    checks.condition(
        "saddle/independent_second_patch_inside_ball", np.dot(opposite - point, opposite - point) < radius**2
    )
    separating_distance_squared = 0.0008**2 + 0.00384**2
    checks.condition("saddle/axes_outside_ball_disconnect_parameter_patches", separating_distance_squared > radius**2)
    writer.json(
        "saddle/analytical-proof.json",
        {
            "map_m": "X(u,v)=(.001*u,.001*v,.006*u*v)+outer_offset",
            "center_m": point.tolist(),
            "other_patch_m": opposite.tolist(),
            "radius_m": radius,
            "other_patch_distance_squared_m2": float(np.dot(opposite - point, opposite - point)),
            "separating_axes_distance_squared_lower_m2": separating_distance_squared,
            "exact_global_RL": 18,
            "scope": "An explicitly unsupported disconnected-footprint guard. No elastic solve or stress-field qualification.",
        },
    )
    for cooperative in (True, False):
        contacts = api.create_contacts(2, 1, radius, hertz_profile=True)
        loading = api.PlanarLoading(surface.info, load_info, contacts, 32, tasks_per_env=4096)
        state = api.create_stress_state(8, 1)
        state.valid.fill(True)
        state.invalid_steps.fill(0)
        errno = api.V(dtype=api.gs.qd_int, shape=(1,))
        for phase in ("initial", "slot_reverse", "release_all", "reload"):
            reverse, release = phase == "slot_reverse", phase == "release_all"
            api._upload_contacts(api, contacts, source, reverse=reverse, release=release)
            valid = source["valid"][::-1].copy() if reverse else source["valid"].copy()
            if release:
                valid.fill(False)
            errno.fill(0)
            api.kernel_anchor(float(np.finfo(api.gs.np_float).eps), contacts, surface.info)
            label = f"saddle/{'warp' if cooperative else 'serial'}/{phase}"
            checks.condition(
                f"{label}/original_anchor_valid", not api.unbatched_array(api, contacts.status)[valid].any()
            )
            if cooperative:
                api.kernel_planar_query_warp(contacts, loading.info, loading.state)
            else:
                api.kernel_planar_query(contacts, loading.info, loading.state)
            api.kernel_planar_material(
                float(np.finfo(api.gs.np_float).eps),
                contacts,
                loading.state.fallback,
                surface.info,
                loading.info,
                loading.material_state,
            )
            api.kernel_begin_step(state)
            api.kernel_accept(state, contacts, errno)
            reason = api.unbatched_array(api, loading.state.reason)
            material_status = api.unbatched_array(api, loading.material_state.status)
            contact_status = api.unbatched_array(api, contacts.status)
            error = api.qd_to_numpy(errno, transpose=True)
            normal_error = api.unbatched_array(api, loading.info.normal_error)
            bound = api.unbatched_array(api, loading.info.graph_curvature_bound)
            checks.condition(f"{label}/refinement_guard_admitted", (normal_error <= 0.005 * radius).all())
            checks.condition(f"{label}/single_parent_certificate_rejects", (radius * bound >= 1).all())
            checks.condition(f"{label}/geometry_fallback", (reason[valid] == api.PlanarFallback.GEOMETRY).all())
            checks.condition(
                f"{label}/unresolved_material",
                (material_status[valid] == api.PlanarMaterialStatus.UNRESOLVED_FOOTPRINT).all(),
            )
            checks.condition(f"{label}/contact10", (contact_status[valid] == 10).all())
            checks.condition(
                f"{label}/INVALID_STRESS_LOAD",
                ((error & api.ErrorCode.INVALID_STRESS_LOAD) != 0).all() if valid.any() else not error.any(),
            )
            checks.condition(f"{label}/no_elastic_solve_error", not (error & api.ErrorCode.INVALID_STRESS_SOLVE).any())
            checks.condition(
                f"{label}/step_validity", api.qd_to_numpy(state.step_valid, transpose=True).all() == release
            )
            for name in ("position", "force", "normal", "radius", "friction"):
                checks.exact(
                    f"{label}/unchanged_source/{name}",
                    api.unbatched_array(api, vars(contacts)[name]),
                    source[name][::-1] if reverse else source[name],
                )
            checks.close(
                f"{label}/physical_center_unchanged",
                api.unbatched_array(api, contacts.center)[valid],
                (source["position"][::-1] if reverse else source["position"])[valid],
                rtol=1e-12,
                atol=1e-14,
            )
            writer.arrays(
                f"{label}/native.npz",
                reason=reason,
                material_status=material_status,
                contact_status=contact_status,
                errno=error,
                normal_error=normal_error,
                graph_curvature_bound=bound,
                center=api.unbatched_array(api, contacts.center),
                anchor=api.unbatched_array(api, contacts.anchor_face),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=("all", "book", "saddle"), default="all")
    parser.add_argument("--fixture", choices=("closed", "open30"), default="closed")
    parser.add_argument("--cpu-refinements", choices=(0, 3), type=int, default=0)
    parser.add_argument("--refinement", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--quadrature", type=int, choices=(24, 48), default=48)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--capacities", nargs="+", type=int, choices=(1, 4096), default=(1, 4096))
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
        "schema": "genesis.planar-footprint.v1",
        "complete": False,
        "passed": False,
        "worker": _identity(__file__),
        "reference": _identity(args.reference / "manifest.json"),
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "stage": args.stage,
        "fixture": args.fixture,
        "cpu_refinements": args.cpu_refinements,
        "original_wrench_tolerance": 1e-8,
        "original_K_residual": {"relative": 1e-8, "absolute": 1e-11},
        "strict_Q4_reference_tolerance": TOLERANCE,
        "CPU_worker_returncode": None,
    }
    _json(args.output / "result.json", result)
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if manifest["primary_commit"] != PRIMARY or not manifest["complete"] or not manifest["passed"]:
        raise RuntimeError("Use the successful pinned CPU primary.")
    for row in json.loads((args.reference / "source-snapshot.json").read_text())["files"]:
        if _identity(args.reference / row["snapshot_path"])["sha256"] != row["sha256"]:
            raise RuntimeError(f"Frozen CPU source changed: {row['snapshot_path']}")
    if args.stage != "saddle":
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
                    "--fixture",
                    args.fixture,
                    "--refinement",
                    str(args.refinement),
                    "--cpu-refinements",
                    str(args.cpu_refinements),
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
    native_sources = [
        _identity(engine / name)
        for name in (
            "planar_loading.py",
            "planar_geometry.py",
            "planar_material.py",
            "contact.py",
            "scatter.py",
            "surface.py",
        )
    ]
    result.update(identity=api.runtime_identity(), native_sources=native_sources)
    started = time.perf_counter()
    try:
        if args.stage != "saddle":
            cpu = args.output / "cpu"
            native = api.build_native_mother(api, cpu, args.output, "book", quadrature=args.quadrature)
            expected24, expected48 = _load(api.np, cpu / "oracles/q24.npz"), _load(api.np, cpu / "oracles/q48.npz")
            proxy24, proxy48 = (
                _load(api.np, cpu / "oracles/proxy-q24.npz"),
                _load(api.np, cpu / "oracles/proxy-q48.npz"),
            )
            numerics = _load(api.np, cpu / "oracles/numerics.npz")
            oracle_k = api.sparse.load_npz(cpu / "oracles/K.npz")
            model = native.model
            actual_k = api.sparse.bsr_matrix(
                (
                    api.unbatched_array(api, model.info.stiffness) * model.options.young,
                    api.unbatched_array(api, model.info.columns),
                    api.unbatched_array(api, model.info.row_start),
                ),
                shape=oracle_k.shape,
            ).tocsr()
            checks.close("book/original_global_K", actual_k, oracle_k, rtol=1e-10, atol=1e-8)
            api.sparse.save_npz(writer.path("book/native-K.npz"), actual_k)
            native_free = api.unbatched_array(api, model.info.is_free).reshape(-1).astype(bool)
            native_free_k = oracle_k[native_free][:, native_free].toarray()
            native_diagonal = api.np.sqrt(api.np.diag(native_free_k))
            writer.json(
                "book/gauge-comparison.json",
                {
                    "CPU_fixed_dofs": numerics["gauge_dofs"].tolist(),
                    "native_fixed_dofs": api.np.flatnonzero(~native_free).tolist(),
                    "both_gauges": "translation",
                    "young": model.options.young,
                    "thickness": model.options.shell_thickness,
                    "drilling_factor": model.options.shell_drilling_factor,
                    "tolerance": model.options.tolerance,
                    "absolute_tolerance": model.options.absolute_tolerance,
                    "native_gauge_free_K_condition_2": float(api.np.linalg.cond(native_free_k)),
                    "native_gauge_scaled_free_K_condition_2": float(
                        api.np.linalg.cond(native_free_k / native_diagonal[:, None] / native_diagonal[None, :])
                    ),
                },
            )
            checks.close(
                "book/original_global_mass_modes",
                api.unbatched_array(api, model.info.mass_modes).reshape((-1, 6)),
                numerics["mass_modes"],
                rtol=1e-10,
                atol=1e-14,
            )
            checks.exact("book/CPU_independent_hits", expected24["hits"], expected48["hits"])
            checks.exact("book/CPU_independent_components", expected24["selected"], expected48["selected"])
            cpu_identity = json.loads((cpu / "cpu-source.json").read_text())
            for method in cpu_identity["methods"]:
                if method["method"] != "unmasked-q48":
                    checks.condition(
                        f"book/CPU/{method['method']}/original_K_gate",
                        all(method["original_K_gate_passed"]),
                        relative=method["original_K_residual_relative"],
                        absolute=method["original_K_residual_absolute"],
                    )
            loaded = expected48["peak"] > 0
            uncertainty = api._errors(api.np, expected24, expected48, numerics["gauss_weights"])
            uncertainty["load"] = api.np.linalg.norm(expected24["raw"] - expected48["raw"], axis=0) / api.np.maximum(
                api.np.linalg.norm(expected48["raw"], axis=0), 1e-30
            )
            maxima = {key: float(value[loaded].max(initial=0)) for key, value in uncertainty.items()}
            resolved = all(value <= TOLERANCE for value in maxima.values())
            checks.condition("book/CPU_independently_resolved_q24_q48", resolved, maxima=maxima, tolerance=TOLERANCE)
            proxy_uncertainty = api._errors(api.np, proxy24, proxy48, numerics["gauss_weights"])
            proxy_uncertainty["load"] = api.np.linalg.norm(proxy24["raw"] - proxy48["raw"], axis=0) / api.np.maximum(
                api.np.linalg.norm(proxy48["raw"], axis=0), 1e-30
            )
            proxy_maxima = {key: float(value[loaded].max(initial=0)) for key, value in proxy_uncertainty.items()}
            checks.condition(
                "book/CPU_independently_resolved_P1_proxy_q24_q48",
                all(value <= TOLERANCE for value in proxy_maxima.values()),
                maxima=proxy_maxima,
                tolerance=TOLERANCE,
            )
            if not resolved or not all(row["passed"] for row in checks.rows):
                raise ArithmeticError(
                    "Resolve the independent book integration and operators before evaluating native loads."
                )
            result["book_modes"] = [
                _book_mode(
                    api,
                    args,
                    writer,
                    checks,
                    native,
                    expected24,
                    expected48,
                    proxy48,
                    numerics,
                    oracle_k,
                    mode,
                    capacity,
                )
                for mode in args.modes
                for capacity in args.capacities
            ]
            unmasked = _load(api.np, cpu / "oracles/unmasked-q48.npz")
            diagnostics = api._errors(api.np, unmasked, expected48, numerics["gauss_weights"])
            cases = json.loads((cpu / "inputs/book/cases.json").read_text())
            masked_nodes = expected48["raw"].reshape((6, 6, len(cases)))
            unmasked_nodes = unmasked["raw"].reshape((6, 6, len(cases)))
            for i_b, case in enumerate(cases):
                if case["category"] == "tip":
                    far = [4, 5] if case["expected_parents"] == [0] else [2, 3]
                    checks.condition(
                        f"book/CPU/old_coarse_graph_wrong_far_load/{case['name']}",
                        not api.np.count_nonzero(masked_nodes[far, :, i_b])
                        and api.np.linalg.norm(unmasked_nodes[far, :, i_b]) > 0,
                    )
            writer.arrays(
                "book/unmasked-diagnostic.npz",
                **{f"error_{key}": value for key, value in diagnostics.items()},
                raw=unmasked["raw"],
                selected_raw=expected48["raw"],
            )
            checks.close(
                "book/original_K_after_loading",
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
            result["CPU_integration_uncertainty"] = maxima
            result["CPU_proxy_integration_uncertainty"] = proxy_maxima
        if args.stage != "book":
            _geometry_primitives(api, writer, checks)
            _strong_saddle(api, writer, checks)
        checks.condition(
            "provenance/native_source_unchanged", [_identity(row["path"]) for row in native_sources] == native_sources
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
        )
    finally:
        if sys.exc_info()[0] is not None:
            (args.output / "failure.txt").write_text(traceback.format_exc())
        safe_checks = []
        for row in checks.rows:
            clean, nonfinite = _finite_json(row)
            if nonfinite:
                clean.update(passed=False, nonfinite_diagnostic_count=nonfinite)
            safe_checks.append(clean)
        writer.json("checks.json", safe_checks)
        safe_result, nonfinite = _finite_json(result)
        if nonfinite:
            safe_result.update(passed=False, nonfinite_diagnostic_count=nonfinite)
        writer.json("result.json", safe_result)
    if not result["passed"]:
        raise ArithmeticError("Physical-footprint controls failed. Preserve every rejection and uncertainty.")
    print(json.dumps({"passed": True, "checks": result["checks"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
