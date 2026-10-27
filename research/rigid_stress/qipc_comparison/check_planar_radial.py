"""Check isolated exact-radial Hertz primitives against independent CPU integration.

Scientific execution requires Slurm rtx-mid/image 1_26. The production planar
runtime retains its existing integration. High precision endpoint checks,
independent phi quadrature and the old q24 numerical gate remain separate.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

PRIMARY = "797203553f246ca155bbef909c1bb33e7cd7b90c"
ORDERS = (24, 48, 96)
PROFILES = ((1.0, 0.0, 0.0), (1.0, 0.3, -0.2), (0.15, 1.2, -0.8), (-0.2, -0.7, 0.9), (-2.0, 0.0, 0.0))


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load(np, path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _angles(np, polygon, radius):
    values = [0.0, 2 * np.pi]
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        if np.linalg.norm(first) > 1e-18:
            values.append(np.arctan2(first[1], first[0]) % (2 * np.pi))
        edge = second - first
        aa, bb, cc = edge @ edge, 2 * (first @ edge), first @ first - radius * radius
        discriminant = bb * bb - 4 * aa * cc
        if aa > 0 and discriminant > 0:
            for fraction in ((-bb - np.sqrt(discriminant)) / (2 * aa), (-bb + np.sqrt(discriminant)) / (2 * aa)):
                if 0 < fraction < 1:
                    point = first + fraction * edge
                    values.append(np.arctan2(point[1], point[0]) % (2 * np.pi))
    return np.unique(np.round(values, 14))


def _cpu_geometry(api, physical, expected, index):
    np = api.np
    model = api.SurfaceGeometry(
        physical["parent_vertices"][index],
        np.array([[0, 1, 2, 3]]),
        physical["parent_directors"][index],
        h=float(physical["thickness"]),
        E=float(physical["young"]),
        nu=float(physical["poisson"]),
    )
    surface = api.proxy._triangle_surface(model, int(physical["refinement"][index]), int(physical["side"][index]))
    triangle = physical["triangle"][index]
    vertices, dual = surface.vertices[triangle], surface.dual[triangle]
    normal, shapes = surface.normals[triangle], surface.parent_shapes[triangle]
    point, radius = physical["point"][index], expected["radius"][index]
    height = (point - vertices[0]) @ normal
    remaining = radius * radius - height * height
    center, plane = point - height * normal, api.proxy._pressure_frame(normal)
    polygon = (vertices - center) @ plane
    section_radius = np.sqrt(remaining)
    contained = api.ShellDPRTransfer._full_disk(polygon, section_radius)
    return SimpleNamespace(
        vertices=vertices,
        dual=dual,
        normal=normal,
        shapes=shapes,
        point=point,
        radius=radius,
        center=center,
        plane=plane,
        polygon=polygon,
        section_radius=section_radius,
        contained=contained,
        angles=_angles(np, polygon, section_radius),
        frame=api.proxy._pressure_frame(physical["force"][index]),
        force=physical["force"][index],
    )


def _phi_reference(api, geometry, order):
    """Integrate each positive radial subinterval with independent high-order phi Gauss."""
    np = api.np
    angular, angular_weight = np.polynomial.legendre.leggauss(order)
    radial, radial_weight = np.polynomial.legendre.leggauss(192)
    if geometry.contained:
        theta = 2 * np.pi * (np.arange(max(16, 2 * order)) + 0.5) / max(16, 2 * order)
        weight = np.full(len(theta), 2 * np.pi / len(theta))
        low, high = np.zeros(len(theta)), np.full(len(theta), geometry.section_radius)
    else:
        left, right = geometry.angles[:-1, None], geometry.angles[1:, None]
        node = 0.5 * (angular + 1)
        theta = (left + (right - left) * np.sin(0.5 * np.pi * node) ** 2).reshape(-1)
        weight = ((right - left) * np.pi / 4 * np.sin(np.pi * node) * angular_weight).reshape(-1)
        low, high = api.ShellDPRTransfer._radial_interval(geometry.polygon, theta, geometry.section_radius)
    live = high > low
    theta, weight, low, high = theta[live], weight[live], low[live], high[live]
    low, high = low / geometry.section_radius, high / geometry.section_radius
    direction = np.column_stack((np.cos(theta), np.sin(theta))) @ geometry.plane.T
    mean = np.r_[1.0, (geometry.center - geometry.point) @ geometry.frame / geometry.radius]
    slope = np.column_stack(
        (np.zeros(len(theta)), direction @ geometry.frame * (geometry.section_radius / geometry.radius))
    )
    gram, load, wrench = [], [], []
    for coefficient in np.array(PROFILES):
        lower, upper = low.copy(), high.copy()
        intercept, derivative = mean @ coefficient, slope @ coefficient
        first, second = intercept + low * derivative, intercept + high * derivative
        empty = (first <= 0) & (second <= 0)
        crossing = (first < 0) & (second > 0)
        lower[crossing] = low[crossing] + (high[crossing] - low[crossing]) * (-first[crossing]) / (
            second[crossing] - first[crossing]
        )
        crossing = (first > 0) & (second < 0)
        upper[crossing] = low[crossing] + (high[crossing] - low[crossing]) * first[crossing] / (
            first[crossing] - second[crossing]
        )
        upper[empty] = lower[empty]
        lower, upper = np.clip(lower, low, high), np.clip(upper, low, high)
        phi_low, phi_high = np.arcsin(lower), np.arcsin(upper)
        phi = 0.5 * (phi_low[:, None] + phi_high[:, None]) + 0.5 * (phi_high - phi_low)[:, None] * radial
        rho = np.sin(phi)
        positions = geometry.center + geometry.section_radius * rho[:, :, None] * direction[:, None, :]
        local = (positions - geometry.vertices[0]) @ geometry.dual.T
        barycentric = np.concatenate((1 - local.sum(axis=-1, keepdims=True), local), axis=-1)
        parent = barycentric @ geometry.shapes
        coordinates = np.concatenate(
            (np.ones((*rho.shape, 1)), (positions - geometry.point) @ geometry.frame / geometry.radius), axis=-1
        )
        integration = geometry.section_radius**3 / geometry.radius * np.sin(phi) * np.cos(phi) ** 2
        integration *= 0.5 * (phi_high - phi_low)[:, None] * radial_weight * weight[:, None]
        gram.append(np.einsum("rni,rn,rnj->ij", coordinates, integration, coordinates))
        load.append(np.einsum("rni,rn,rnj->ij", parent, integration, coordinates))
        pressure = coordinates @ coefficient
        force = integration[:, :, None] * pressure[:, :, None] * geometry.force
        moment = np.cross(positions - geometry.point, force)
        wrench.append(np.r_[force.sum(axis=(0, 1)), moment.sum(axis=(0, 1))])
    gram, load, wrench = np.array(gram), np.array(load), np.array(wrench)
    mass = gram[0, 0, 0]
    coefficient = np.array(PROFILES)
    return {
        "gram": gram,
        "load": load,
        "objective": coefficient[:, 0] - 0.5 * np.einsum("pi,pij,pj->p", coefficient, gram, coefficient) / mass,
        "residual": np.einsum("pij,pj->pi", gram, coefficient) / mass - np.array([1.0, 0.0, 0.0]),
        "jacobian": gram / mass,
        "weights": np.einsum("pij,pj->pi", load, coefficient) / mass,
        "wrench": wrench / mass,
    }


def _cpu_worker(args):
    fixture = args.output / "physical"
    fixture.mkdir()
    extraction = Path(__file__).with_name("check_planar_primitives.py")
    with (args.output / "primary-extraction.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(extraction),
                "--_cpu-worker",
                "--reference",
                str(args.reference),
                "--output",
                str(fixture),
            ],
            cwd=args.reference / "reference-source",
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    sys.path.insert(0, str(args.reference / "reference-source"))
    import mpmath as mp
    import numpy as np
    from threadpoolctl import threadpool_limits

    from research.rigid_stress.shell_cpu.contacts import ShellDPRTransfer
    from research.rigid_stress.triangular_load_cpu import proxy
    from research.rigid_stress.triangular_load_cpu.checks import SurfaceGeometry

    api = SimpleNamespace(**locals())
    intervals = [(0.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.2, 0.9), (0.25, np.nextafter(0.25, 1.0))]
    for power in (1, 8, 20, 40, 52, 100, 200):
        delta = 2.0**-power
        intervals.extend(((0.0, delta), (delta, 2 * delta)))
        if power <= 52:
            intervals.append((1.0 - delta, 1.0))
    windows, moments = [], []
    window_coefficient = ((1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (-0.3, 1.0), (0.7, -1.0), (-1e-15, 1.0))
    with mp.workdps(400):
        for lower, upper in intervals:
            lo, hi = mp.mpf(float(lower)), mp.mpf(float(upper))
            row = []
            for power in (1, 2, 3):
                row.append(float(mp.betainc(mp.mpf(power + 1) / 2, mp.mpf(3) / 2, lo * lo, hi * hi) / 2))
            moments.append(row)
            roots = (lower + (upper - lower) * 2.0**-20, upper - (upper - lower) * 2.0**-20)
            local_coefficients = [
                *window_coefficient,
                *[(-root, 1.0) for root in roots],
                *[(root, -1.0) for root in roots],
            ]
            for intercept, derivative in local_coefficients:
                a, b = mp.mpf(float(intercept)), mp.mpf(float(derivative))
                first, second = a + lo * b, a + hi * b
                active_lo, active_hi = lo, hi
                if hi <= lo or (first <= 0 and second <= 0):
                    active_hi = active_lo
                elif first < 0:
                    active_lo = -a / b
                elif second < 0:
                    active_hi = -a / b
                windows.append((lower, upper, intercept, derivative, float(active_lo), float(active_hi)))
    np.savez_compressed(
        args.output / "one-dimensional.npz",
        intervals=np.array(intervals),
        moments=np.array(moments),
        windows=np.array(windows),
    )
    records = []
    with threadpool_limits(limits=1):
        for directory in sorted(fixture.iterdir()):
            if not directory.is_dir():
                continue
            physical, old = _load(np, directory / "physical.npz"), _load(np, directory / "oracles.npz")
            count = len(physical["radius"])
            geometries = [_cpu_geometry(api, physical, old, i) for i in range(count)]
            values = {}
            for order in ORDERS:
                rows = [_phi_reference(api, geometry, order) for geometry in geometries]
                values.update({f"{key}{order}": np.array([row[key] for row in rows]) for key in rows[0]})
            values.update(
                vertices=np.array([geometry.vertices for geometry in geometries]),
                dual=np.array([geometry.dual for geometry in geometries]),
                shapes=np.array([geometry.shapes for geometry in geometries]),
                normal=np.array([geometry.normal for geometry in geometries]),
            )
            np.savez_compressed(args.output / f"{directory.name}-reference.npz", **values)
            records.append({"fixture": directory.name, "triangles": count, "orders": list(ORDERS), "radial_order": 192})
    imports = [
        _identity(sys.modules[name].__file__)
        for name in (
            "research.rigid_stress.shell_cpu.contacts",
            "research.rigid_stress.triangular_load_cpu.proxy",
            "research.rigid_stress.triangular_load_cpu.checks",
        )
    ]
    if not all(Path(row["path"]).is_relative_to(args.reference / "reference-source") for row in imports):
        raise RuntimeError("CPU geometry must come from the frozen primary.")
    _json(
        args.output / "source.json",
        {
            "primary": PRIMARY,
            "imports": imports,
            "extractor": _identity(extraction),
            "mpmath": mp.__version__,
            "mpmath_source": _identity(mp.__file__),
            "primitive_precision_decimal_digits": 400,
            "radial_reference": "Independent 192-node phi Gauss after a high-precision-tested pressure-window split. No native radial antiderivative is used for the geometric controls.",
            "records": records,
        },
    )


def _native_api():
    from .check_planar_primitives import _field, _native_api as primitive_api, _surface

    api = primitive_api()
    qd = api.qd
    gs = api.gs
    func_planar_clip_geometry = api.func_planar_clip_geometry
    func_force_frame = api.func_force_frame
    primitive_state, primitive_info = api.PrimitiveState, api.PrimitiveInfo
    from genesis.engine.solvers.rigid.stress.planar_radial import (
        func_hertz_radial_moments,
        func_planar_active_radial_interval,
        func_planar_hertz_ray,
        func_planar_ray_count,
        func_planar_ray_moments,
    )

    @dataclass(frozen=True)
    class RadialState:
        gram: qd.Tensor
        load: qd.Tensor
        objective: qd.Tensor
        residual: qd.Tensor
        jacobian: qd.Tensor
        weights: qd.Tensor
        wrench: qd.Tensor
        ray_count: qd.Tensor

    @qd.kernel
    def kernel_intervals(intervals: qd.Tensor, windows: qd.Tensor, moments: qd.Tensor, active: qd.Tensor):
        for i in range(intervals.shape[0]):
            moments[i] = func_hertz_radial_moments(intervals[i][0], intervals[i][1])
        for i in range(windows.shape[0]):
            mean, slope = qd.Vector([1.0, 0.0, 0.0]), qd.Vector([0.0, 1.0, 0.0])
            coefficient = qd.Vector([windows[i][2], windows[i][3], 0.0])
            low, high = func_planar_active_radial_interval(windows[i][0], windows[i][1], mean, slope, coefficient)
            active[i] = qd.Vector([low, high])

    @qd.kernel
    def kernel_integrals(
        gauss: qd.Tensor,
        gauss_weights: qd.Tensor,
        coefficient: qd.Tensor,
        state: primitive_state,
        info: primitive_info,
        current: RadialState,
    ):
        for i, i_p in qd.ndrange(state.radius.shape[0], coefficient.shape[0]):
            vertices = state.vertices[i]
            origin = vertices[0, :]
            edges = qd.Matrix.rows([vertices[1, :] - origin, vertices[2, :] - origin])
            section, plane, polygon, remaining, angles, count, contained = func_planar_clip_geometry(
                origin, edges, state.normal[i], info.point[i], state.radius[i]
            )
            first, second = func_force_frame(info.force[i])
            gram, load = qd.Matrix.zero(gs.qd_float, 3, 3), qd.Matrix.zero(gs.qd_float, 4, 3)
            n_rays = func_planar_ray_count(count, gauss.shape[0], contained)
            for i_ray in range(n_rays):
                low, high, scale, mean, slope, parent_mean, parent_slope = func_planar_hertz_ray(
                    i_ray,
                    origin,
                    state.dual[i],
                    state.shapes[i],
                    section,
                    plane,
                    polygon,
                    remaining,
                    angles,
                    count,
                    info.point[i],
                    state.radius[i],
                    first,
                    second,
                    gauss,
                    gauss_weights,
                    contained,
                )
                lower, upper = func_planar_active_radial_interval(low, high, mean, slope, coefficient[i_p])
                ray_gram, ray_load = func_planar_ray_moments(
                    lower, upper, scale, mean, slope, parent_mean, parent_slope
                )
                gram += ray_gram
                load += ray_load
            row = i * coefficient.shape[0] + i_p
            current.gram[row], current.load[row], current.ray_count[row] = gram, load, n_rays

    @qd.kernel
    def kernel_outputs(
        thickness: float, coefficient: qd.Tensor, state: primitive_state, info: primitive_info, current: RadialState
    ):
        for i, i_p in qd.ndrange(state.radius.shape[0], coefficient.shape[0]):
            row = i * coefficient.shape[0] + i_p
            normalizer = current.gram[i * coefficient.shape[0]][0, 0]
            gram, load, value = current.gram[row], current.load[row], coefficient[i_p]
            current.objective[row] = value[0] - 0.5 * value.dot(gram @ value) / normalizer
            current.residual[row] = (gram @ value) / normalizer - qd.Vector([1.0, 0.0, 0.0])
            current.jacobian[row] = gram / normalizer
            weights = (load @ value) / normalizer
            current.weights[row] = weights
            total, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
            for a in qd.static(range(4)):
                point = info.vertices[i][a, :] + info.side[i] * 0.5 * thickness * info.directors[i][a, :]
                force = weights[a] * info.force[i]
                total += force
                moment += (point - info.point[i]).cross(force)
            current.wrench[row] = qd.Vector([total[0], total[1], total[2], moment[0], moment[1], moment[2]])

    return SimpleNamespace(
        **(vars(api) | {name: value for name, value in locals().items() if name not in ("api", "primitive_api")})
    )


def _physical_state(api, physical):
    fields = (
        ("vertices", "parent_vertices", 2, False),
        ("directors", "parent_directors", 2, False),
        ("point", "point", 1, False),
        ("force", "force", 1, False),
        ("normal", "normal", 1, False),
        ("curvatures", "curvatures", 1, False),
        ("radius", "radius", 0, False),
        ("radius_given", "radius_given", 0, True),
        ("side", "side", 0, True),
        ("triangle", "triangle", 0, True),
        ("refinement", "refinement", 0, True),
    )
    info = api.PrimitiveInfo(
        **{name: api._field(api, physical[key], rank, integer) for name, key, rank, integer in fields}
    )
    count = len(physical["radius"])
    shapes = {
        "vertices": (3, 3),
        "dual": (2, 3),
        "normal": (3,),
        "shapes": (3, 4),
        "gram": (3, 3),
        "load": (4, 3),
        "numerical_gram": (3, 3),
        "numerical_load": (4, 3),
    }
    values = {}
    for name, shape in shapes.items():
        factory = api.V_VEC if len(shape) == 1 else api.V_MAT
        values[name] = factory(*shape, dtype=api.gs.qd_float, shape=(count,))
    for name in ("radius", "mass", "minimum", "negative_minimum", "work_error", "rigid_error", "minimum_barycentric"):
        values[name] = api.V(dtype=api.gs.qd_float, shape=(count,))
    for name in ("contained", "sample_count"):
        values[name] = api.V(dtype=api.gs.qd_int, shape=(count,))
    state = api.PrimitiveState(**values)
    api.kernel_prepare(float(physical["young"]), float(physical["poisson"]), float(physical["thickness"]), state, info)
    return state, info


def _checks(api, args, writer, checks):
    np = api.np
    one = _load(np, args.output / "cpu/one-dimensional.npz")
    intervals, windows = api._field(api, one["intervals"], 1), api._field(api, one["windows"][:, :4], 1)
    moments = api.V_VEC(3, dtype=api.gs.qd_float, shape=(len(one["intervals"]),))
    active = api.V_VEC(2, dtype=api.gs.qd_float, shape=(len(one["windows"]),))
    api.kernel_intervals(intervals, windows, moments, active)
    actual = api.qd_to_numpy(moments, transpose=True)
    relative = np.abs(actual - one["moments"]) / np.maximum(np.abs(one["moments"]), 1e-300)
    checks.condition(
        "primitive/high_precision_radial_moments", (relative <= 1e-11).all(), maximum_relative=float(relative.max())
    )
    checks.condition("primitive/nonnegative_radial_moments", (actual >= 0).all())
    active_values = api.qd_to_numpy(active, transpose=True)
    checks.close(
        "primitive/high_precision_active_windows",
        active_values,
        one["windows"][:, 4:],
        rtol=1e-12,
        atol=1e-15,
    )
    interval_width = one["windows"][:, 1] - one["windows"][:, 0]
    window_error = np.max(np.abs(active_values - one["windows"][:, 4:]), axis=1) / np.maximum(interval_width, 1e-300)
    checks.condition(
        "primitive/active_windows_relative_to_interval_width",
        (window_error <= 1e-12).all(),
        maximum=float(window_error.max()),
    )
    writer.arrays(
        "primitive.npz",
        actual=actual,
        expected=one["moments"],
        relative_error=relative,
        active=active_values,
        expected_active=one["windows"][:, 4:],
        window_error_relative_to_interval_width=window_error,
    )
    source = _load(np, args.reference / "inputs/egg75/model.npz")
    nodes = source["quads"][0]
    rule_source = {
        "vertices": source["vertices"][nodes],
        "directors": source["directors"][nodes],
        "quads": np.array([[0, 1, 2, 3]]),
        "thickness": source["thickness"],
    }
    rules = {order: api._surface(api, rule_source, order)[1] for order in ORDERS}
    coefficients = api._field(api, np.array(PROFILES), 1)
    records = []
    for directory in sorted((args.output / "cpu/physical").iterdir()):
        if not directory.is_dir():
            continue
        physical = _load(np, directory / "physical.npz")
        expected = _load(np, args.output / f"cpu/{directory.name}-reference.npz")
        old = _load(np, directory / "oracles.npz")
        state, info = _physical_state(api, physical)
        for name in ("vertices", "dual", "normal", "shapes"):
            checks.close(
                f"{directory.name}/geometry/{name}",
                api.qd_to_numpy(vars(state)[name], transpose=True),
                expected[name],
                rtol=1e-10,
                atol=1e-13,
            )
        count = len(physical["radius"])
        size = count * len(PROFILES)
        current = api.RadialState(
            gram=api.V_MAT(3, 3, dtype=api.gs.qd_float, shape=(size,)),
            load=api.V_MAT(4, 3, dtype=api.gs.qd_float, shape=(size,)),
            objective=api.V(dtype=api.gs.qd_float, shape=(size,)),
            residual=api.V_VEC(3, dtype=api.gs.qd_float, shape=(size,)),
            jacobian=api.V_MAT(3, 3, dtype=api.gs.qd_float, shape=(size,)),
            weights=api.V_VEC(4, dtype=api.gs.qd_float, shape=(size,)),
            wrench=api.V_VEC(6, dtype=api.gs.qd_float, shape=(size,)),
            ray_count=api.V(dtype=api.gs.qd_int, shape=(size,)),
        )
        values = {}
        for order in ORDERS:
            rule = rules[order].info
            api.kernel_integrals(rule.gauss, rule.gauss_weights, coefficients, state, info, current)
            api.kernel_outputs(float(physical["thickness"]), coefficients, state, info, current)
            for name in ("gram", "load", "objective", "residual", "jacobian", "weights", "wrench"):
                measured = api.qd_to_numpy(vars(current)[name], transpose=True)
                measured = measured.reshape((count, len(PROFILES), *measured.shape[1:]))
                values[f"{name}{order}"] = measured
                checks.close(
                    f"{directory.name}/qa{order}/independent_phi/{name}",
                    measured,
                    expected[f"{name}{order}"],
                    rtol=1e-9,
                    atol=1e-20,
                )
                delta = (measured - expected[f"{name}{order}"]).reshape((count, len(PROFILES), -1))
                reference = expected[f"{name}{order}"].reshape((count, len(PROFILES), -1))
                norm, scale = np.linalg.norm(delta, axis=2), np.linalg.norm(reference, axis=2)
                values[f"relative_{name}{order}"] = norm / np.maximum(scale, 1e-300)
                checks.condition(
                    f"{directory.name}/qa{order}/each_triangle_profile/{name}",
                    (norm <= 1e-20 + 1e-9 * scale).all(),
                    maximum_relative=float(values[f"relative_{name}{order}"].max()),
                )
            if order == 24:
                for name in ("gram", "load"):
                    checks.close(
                        f"{directory.name}/original_q24_port/{name}",
                        values[f"{name}{order}"][:, 0],
                        old[f"{name}24"],
                        rtol=1e-8,
                        atol=1e-20,
                    )
            checks.condition(
                f"{directory.name}/qa{order}/nonnegative_parent_weights",
                (values[f"weights{order}"] >= -2e-12).all(),
                minimum=float(values[f"weights{order}"].min()),
            )
            values[f"rays{order}"] = api.qd_to_numpy(current.ray_count, transpose=True).reshape((count, len(PROFILES)))
        uncertainty = {}
        for name in ("gram", "load", "objective", "residual", "jacobian", "weights", "wrench"):
            lower, higher = expected[f"{name}48"], expected[f"{name}96"]
            difference, scale = (
                np.linalg.norm((lower - higher).reshape((count, len(PROFILES), -1)), axis=2),
                np.linalg.norm(higher.reshape((count, len(PROFILES), -1)), axis=2),
            )
            relative_error = difference / np.maximum(scale, 1e-30)
            uncertainty[name] = float(relative_error.max())
            checks.condition(
                f"{directory.name}/qa48_qa96_uncertainty/{name}",
                (relative_error <= 1e-3).all(),
                maximum=float(relative_error.max()),
                tolerance=1e-3,
            )
        writer.arrays(f"{directory.name}/native.npz", **values)
        records.append({"fixture": directory.name, "triangles": count, "CPU_angular_uncertainty": uncertainty})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
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
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if manifest["primary_commit"] != PRIMARY or not manifest["complete"] or not manifest["passed"]:
        raise RuntimeError("Use the successful pinned CPU primary reference.")
    snapshot = json.loads((args.reference / "source-snapshot.json").read_text())
    for row in snapshot["files"]:
        if _identity(args.reference / row["snapshot_path"])["sha256"] != row["sha256"]:
            raise RuntimeError(f"Frozen source changed: {row['snapshot_path']}")
    result = {
        "schema": "genesis.planar-radial-primitives.v1",
        "complete": False,
        "passed": False,
        "worker": _identity(__file__),
        "reference": _identity(args.reference / "manifest.json"),
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "CPU_worker_returncode": None,
        "production_runtime_integrated": False,
        "profiles": PROFILES,
        "orders": ORDERS,
        "radial_reference_order": 192,
        "scope": "Isolated mathematical integration primitives. No original K/load acceptance gate is modified and no complete stress, performance or production integration claim follows.",
        "native_input_policy": "Physical parent vertices/directors/material, original sources, integer triangle fixtures and authored pressure coefficients only. CPU-derived geometry/moments/wrenches remain readback comparisons.",
        "primitive_harness": _identity(Path(__file__).with_name("check_planar_primitives.py")),
        "old_q24_policy": "The original q24 numerical comparison retains rtol1e-8. A changed finite quadrature may fail this gate even when an independently converged continuous integral passes. Preserve and label that difference as quadrature approximation rather than floating-point equivalence.",
        "derivative_convention": "residual = -gradient(concave objective), jacobian = -Hessian(concave objective), matching the existing native Newton convention.",
    }
    _json(args.output / "result.json", result)
    cpu = args.output / "cpu"
    cpu.mkdir()
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    with (args.output / "cpu.log").open("w") as log:
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
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result["CPU_worker_returncode"] = completed.returncode
    _json(args.output / "result.json", result)
    completed.check_returncode()
    api = _native_api()
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    from .artifacts import runtime_identity
    from .check_cpu12_gpu import Checks
    from .export_planar_loading_reference import Writer

    writer, checks = Writer(args.output, api), Checks(api)
    engine = Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress"
    before = [
        _identity(engine / name) for name in ("planar_radial.py", "planar_moments.py", "quadrilateral.py", "contact.py")
    ]
    result["native_sources"] = before
    result["identity"] = runtime_identity()
    started = time.perf_counter()
    try:
        result["fixtures"] = _checks(api, args, writer, checks)
        checks.condition("sources/unchanged", [_identity(row["path"]) for row in before] == before)
        checks.condition("worker/unchanged", _identity(__file__) == result["worker"])
        checks.condition(
            "primitive_harness/unchanged",
            _identity(Path(__file__).with_name("check_planar_primitives.py")) == result["primitive_harness"],
        )
        port_checks = [row for row in checks.rows if "/original_q24_port/" in row["name"]]
        math_checks = [row for row in checks.rows if "/original_q24_port/" not in row["name"]]
        result.update(
            complete=True,
            passed=all(row["passed"] for row in checks.rows),
            independent_math_passed=all(row["passed"] for row in math_checks),
            original_q24_port_passed=all(row["passed"] for row in port_checks),
            checks=len(checks.rows),
            seconds=time.perf_counter() - started,
        )
    finally:
        checks.flush(writer)
        if sys.exc_info()[0] is not None:
            (args.output / "failure.txt").write_text(traceback.format_exc())
        writer.json("result.json", result)
    if not result["passed"]:
        raise ArithmeticError(
            "Exact-radial candidate checks failed. Retain independent integration and old-q24 diagnostics."
        )
    print(json.dumps({"passed": True, "checks": result["checks"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
