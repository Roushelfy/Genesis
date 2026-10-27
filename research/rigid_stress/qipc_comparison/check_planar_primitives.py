"""Check native planar geometry and Hertz moments against the frozen CPU primary.

Scientific execution requires Slurm rtx-mid and image 1_26. A separate CPU
interpreter extracts the primary's twelve clipped controls. Native kernels
receive physical inputs and integer fixtures. CPU geometry and moments remain
independent readback oracles. This driver does not measure throughput.
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


def _identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load(np, path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _cpu_worker(args):
    frozen = args.reference / "reference-source"
    sys.path.insert(0, str(frozen))
    import numpy as np
    from threadpoolctl import threadpool_limits

    from research.rigid_stress.shell_cpu.contacts import ShellDPRTransfer, hertz_radius
    from research.rigid_stress.shell_cpu.core import ShellModel
    from research.rigid_stress.triangular_load_cpu import checks, proxy
    from research.rigid_stress.triangular_load_cpu.checks import SurfaceGeometry
    from research.rigid_stress.triangular_load_cpu.proxy import TriangularLoadTransfer

    imports = [
        Path(sys.modules[name].__file__).resolve()
        for name in (
            "research.rigid_stress.shell_cpu.contacts",
            "research.rigid_stress.shell_cpu.core",
            "research.rigid_stress.triangular_load_cpu.checks",
            "research.rigid_stress.triangular_load_cpu.proxy",
        )
    ]
    if not all(path.is_relative_to(frozen) for path in imports):
        raise RuntimeError("The CPU primitive oracle must import only the frozen primary.")
    before = [_identity(path) for path in imports]
    physical = _load(np, args.reference / "inputs/egg75/model.npz")
    recorded = []

    class RecordingProxy(TriangularLoadTransfer):
        def transfer(self, contact, order=24, analytic=True, anchor_mode="fixed"):
            if order == 24:
                recorded.append(contact)
            return super().transfer(contact, order, analytic, anchor_mode)

    with threadpool_limits(limits=1):
        model = ShellModel(
            physical["vertices"],
            physical["quads"],
            E=float(physical["young"]),
            nu=float(physical["poisson"]),
            h=float(physical["thickness"]),
            density=float(physical["density"]),
            directors=physical["directors"],
            gauge="translation",
        )
        original_control = checks._proxy_quadrature_controls(model, RecordingProxy)
        if len(recorded) != 12 or not original_control["passed"]:
            raise ArithmeticError("The exact primary twelve-contact integration control failed.")
        names = [row["name"] for row in original_control["rows"]]
        _cpu_fixture(args.output / "clipped12", model, recorded, names, 8, np, proxy, ShellDPRTransfer, hertz_radius)
        _json(args.output / "clipped12/primary-check.json", original_control)
        for name in ("affine-side--1", "affine-side-1", "tilted-affine"):
            directory = args.reference / "inputs/controls" / name
            raw = _load(np, directory / "model.npz")
            control_model = SurfaceGeometry(raw["vertices"], raw["quads"], raw["directors"])
            control_model.h = float(raw["thickness"])
            control_model.E = float(raw["young"])
            control_model.nu = float(raw["poisson"])
            cases = json.loads((directory / "cases.json").read_text())
            source = cases[0]["contacts"][0]
            contact = checks.Contact(
                np.array(source["point"]),
                np.array(source["force"]),
                np.array(source["normal"]),
                tuple(source["curvatures"]),
                source["side"],
                source["radius"],
                source["friction"],
                source["face_hint"],
            )
            _cpu_fixture(
                args.output / name,
                control_model,
                [contact],
                [name],
                1,
                np,
                proxy,
                ShellDPRTransfer,
                hertz_radius,
            )
    after = [_identity(path) for path in imports]
    if before != after:
        raise RuntimeError("The frozen primary changed during primitive fixture extraction.")
    _json(args.output / "cpu-source.json", {"primary": PRIMARY, "imports": before, "source_unchanged": True})


def _cpu_fixture(directory, model, contacts, names, refinement, np, proxy, shell_transfer, hertz_radius):
    directory.mkdir(parents=True, exist_ok=False)
    transfer = proxy.TriangularLoadTransfer(model, refinement)
    entries, moments, source_rows = [], [], []
    for i_c, (contact, name) in enumerate(zip(contacts, names)):
        radius = contact.radius
        if radius is None:
            radius = hertz_radius(-contact.force @ contact.normal, model.E, model.nu, contact.curvatures)
        surface = transfer.outer if contact.side > 0 else transfer.inner
        sections, _candidates = transfer._sections(surface, contact.point, radius, contact.face_hint)
        source_rows.append({"name": name, "source": vars(contact), "radius": float(radius), "sections": len(sections)})
        frame = proxy._pressure_frame(contact.force)
        for triangle, section_center, plane, polygon, section_radius in sections:
            parent = int(surface.parent_faces[triangle])
            entries.append((i_c, parent, triangle % (2 * refinement * refinement)))
            values = {}
            for order in (24, 48):
                points, weights = shell_transfer._face_polar_samples(polygon, section_radius, order)
                weights *= 2 * np.pi * section_radius**3 / (3 * radius)
                positions = section_center + points @ plane.T
                local = (positions - surface.vertices[triangle, 0]) @ surface.dual[triangle].T
                barycentric = np.column_stack((1 - local.sum(axis=1), local))
                parent_shape = barycentric @ surface.parent_shapes[triangle]
                coordinates = np.column_stack((np.ones(len(points)), (positions - contact.point) @ frame / radius))
                values[f"gram{order}"] = coordinates.T @ (weights[:, None] * coordinates)
                values[f"load{order}"] = parent_shape.T @ (weights[:, None] * coordinates)
            values["contained"] = shell_transfer._full_disk(polygon, section_radius)
            values["radius"] = radius
            moments.append(values)
    indices = np.array(entries, dtype=np.int64)
    sources = [contacts[i_c] for i_c in indices[:, 0]]
    parents = model.quads[indices[:, 1]]
    np.savez_compressed(
        directory / "physical.npz",
        parent_vertices=model.vertices[parents],
        parent_directors=model.directors[parents],
        point=np.array([source.point for source in sources]),
        force=np.array([source.force for source in sources]),
        normal=np.array([source.normal for source in sources]),
        curvatures=np.array([source.curvatures for source in sources]),
        radius=np.array([0.0 if source.radius is None else source.radius for source in sources]),
        radius_given=np.array([source.radius is not None for source in sources]),
        side=np.array([source.side for source in sources]),
        triangle=indices[:, 2],
        refinement=np.full(len(entries), refinement, dtype=np.int64),
        thickness=model.h,
        young=model.E,
        poisson=model.nu,
    )
    np.savez_compressed(
        directory / "oracles.npz",
        **{key: np.array([row[key] for row in moments]) for key in moments[0]},
    )
    serial = []
    for row in source_rows:
        source = row["source"]
        serial.append(
            {
                "name": row["name"],
                "sections": row["sections"],
                "radius": row["radius"],
                "point": source["point"].tolist(),
                "force": source["force"].tolist(),
                "normal": source["normal"].tolist(),
                "side": source["side"],
            }
        )
    _json(
        directory / "fixtures.json",
        {
            "sources": serial,
            "entries": indices.tolist(),
            "refinement": refinement,
            "scope": "Integer triangle fixtures select audited primitive calls. They do not qualify production queries.",
        },
    )


def _native_api():
    import numpy as np
    import quadrants as qd

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.contact import func_force_frame
    from genesis.engine.solvers.rigid.stress.data import StressInfo, allocate_stress_info
    from genesis.engine.solvers.rigid.stress.planar_geometry import (
        PlanarLoadingInfo,
        build_planar_geometry,
        func_triangle_geometry,
    )
    from genesis.engine.solvers.rigid.stress.planar_moments import (
        func_planar_affine_minimum,
        func_planar_clip_geometry,
        func_planar_hertz_moments,
        func_planar_hertz_sample,
        func_planar_sample_count,
    )
    from genesis.engine.solvers.rigid.stress.quadrilateral import func_quad_shape
    from genesis.engine.solvers.rigid.stress.surface import StressSurface
    from genesis.utils.array_class import V_MAT, V_VEC, V
    from genesis.utils.misc import qd_to_numpy

    @dataclass(frozen=True)
    class PrimitiveInfo:
        vertices: qd.Tensor
        directors: qd.Tensor
        point: qd.Tensor
        force: qd.Tensor
        normal: qd.Tensor
        curvatures: qd.Tensor
        radius: qd.Tensor
        radius_given: qd.Tensor
        side: qd.Tensor
        triangle: qd.Tensor
        refinement: qd.Tensor

    @dataclass(frozen=True)
    class PrimitiveState:
        vertices: qd.Tensor
        dual: qd.Tensor
        normal: qd.Tensor
        shapes: qd.Tensor
        radius: qd.Tensor
        gram: qd.Tensor
        load: qd.Tensor
        numerical_gram: qd.Tensor
        numerical_load: qd.Tensor
        mass: qd.Tensor
        contained: qd.Tensor
        minimum: qd.Tensor
        negative_minimum: qd.Tensor
        work_error: qd.Tensor
        rigid_error: qd.Tensor
        sample_count: qd.Tensor
        minimum_barycentric: qd.Tensor

    @qd.kernel
    def kernel_physical_sides(thickness: float, vertices: qd.Tensor, directors: qd.Tensor, info: StressInfo):
        for i in range(vertices.shape[0]):
            for i_side in range(2):
                sign = qd.select(i_side == 0, 1.0, -1.0)
                i_surface = i_side * vertices.shape[0] + i
                offset = sign * 0.5 * thickness * directors[i]
                info.vertices[i_surface] = vertices[i] + offset
                info.load_nodes[i_surface] = qd.Vector([i])
                info.load_offsets[i_surface] = offset

    @qd.kernel
    def kernel_geometry(
        vertices: qd.Tensor,
        dual: qd.Tensor,
        normals: qd.Tensor,
        shapes: qd.Tensor,
        info: PlanarLoadingInfo,
    ):
        for i in range(vertices.shape[0]):
            n_triangles = info.triangle_nodes.shape[1]
            i_face, i_triangle = i // n_triangles, i % n_triangles
            value, derivative, normal, parent_shape = func_triangle_geometry(i_face, i_triangle, info)
            canonical_shape = qd.Matrix.zero(gs.qd_float, 3, 4)
            permutation = qd.Vector([0, 3, 2, 1])
            for a, b in qd.static(qd.ndrange(3, 4)):
                column = b
                if info.side_index[i_face] == 1:
                    column = permutation[b]
                canonical_shape[a, b] = parent_shape[a, column]
            vertices[i], dual[i], normals[i], shapes[i] = value, derivative, normal, canonical_shape

    @qd.kernel
    def kernel_prepare(young: float, poisson: float, thickness: float, state: PrimitiveState, info: PrimitiveInfo):
        for i in range(info.radius.shape[0]):
            refinement = info.refinement[i]
            triangle = info.triangle[i]
            cell = triangle // 2
            xi, eta = 2.0 * (cell % refinement) / refinement - 1.0, 2.0 * (cell // refinement) / refinement - 1.0
            step = 2.0 / refinement
            coordinates = qd.Matrix([[xi, eta], [xi + step, eta], [xi + step, eta + step]])
            if triangle % 2 == 1:
                coordinates[1, :] = qd.Vector([xi + step, eta + step])
                coordinates[2, :] = qd.Vector([xi, eta + step])
            vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
            shapes = qd.Matrix.zero(gs.qd_float, 3, 4)
            for a in qd.static(range(3)):
                i_a = a
                if info.side[i] < 0 and a > 0:
                    i_a = 3 - a
                shape, _dxi, _deta = func_quad_shape(coordinates[i_a, 0], coordinates[i_a, 1])
                shapes[a, :] = shape
                for b in qd.static(range(4)):
                    offset = info.side[i] * 0.5 * thickness * info.directors[i][b, :]
                    vertices[a, :] += shape[b] * (info.vertices[i][b, :] + offset)
            edges = qd.Matrix.rows([vertices[1, :] - vertices[0, :], vertices[2, :] - vertices[0, :]])
            gram = edges @ edges.transpose()
            state.vertices[i], state.shapes[i] = vertices, shapes
            state.dual[i] = gram.inverse() @ edges
            state.normal[i] = edges[0, :].cross(edges[1, :]).normalized()
            radius = info.radius[i]
            if not info.radius_given[i]:
                compression = -info.force[i].dot(info.normal[i].normalized())
                modulus = young / (1.0 - poisson * poisson)
                curvature = 0.5 * (info.curvatures[i][0] + info.curvatures[i][1])
                radius = (3.0 * compression / (4.0 * modulus * curvature)) ** (1.0 / 3.0)
            state.radius[i] = radius
            nodal_force = qd.Matrix.zero(gs.qd_float, 4, 3)
            vertex_work, parent_work, rigid_error = gs.qd_float(0.0), gs.qd_float(0.0), gs.qd_float(0.0)
            translation = qd.Vector([0.17, -0.23, 0.31])
            spin = qd.Vector([-0.19, 0.13, 0.11])
            for a in qd.static(range(3)):
                displacement, rigid = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
                probe_force = (a + 1.0) * qd.Vector([0.07, -0.03, 0.05])
                for b in qd.static(range(4)):
                    offset = info.side[i] * 0.5 * thickness * info.directors[i][b, :]
                    motion = qd.Vector([0.13 * b, 0.17 - 0.07 * b, 0.11 + 0.03 * b])
                    rotation = qd.Vector([0.19 - 0.03 * b, 0.11 * b, -0.05 + 0.02 * b])
                    displacement += shapes[a, b] * (motion + rotation.cross(offset))
                    rigid += shapes[a, b] * (translation + spin.cross(info.vertices[i][b, :]) + spin.cross(offset))
                    nodal_force[b, :] += shapes[a, b] * probe_force
                vertex_work += probe_force.dot(displacement)
                rigid_error = qd.max(rigid_error, (rigid - translation - spin.cross(vertices[a, :])).norm())
            for b in qd.static(range(4)):
                offset = info.side[i] * 0.5 * thickness * info.directors[i][b, :]
                motion = qd.Vector([0.13 * b, 0.17 - 0.07 * b, 0.11 + 0.03 * b])
                rotation = qd.Vector([0.19 - 0.03 * b, 0.11 * b, -0.05 + 0.02 * b])
                parent_work += nodal_force[b, :].dot(motion) + offset.cross(nodal_force[b, :]).dot(rotation)
            state.work_error[i], state.rigid_error[i] = qd.abs(vertex_work - parent_work), rigid_error

    @qd.kernel
    def kernel_exact(state: PrimitiveState, info: PrimitiveInfo):
        for i in range(state.radius.shape[0]):
            vertices, normal, radius = state.vertices[i], state.normal[i], state.radius[i]
            origin = vertices[0, :]
            edges = qd.Matrix.rows([vertices[1, :] - origin, vertices[2, :] - origin])
            first, second = func_force_frame(info.force[i])
            gram, load, mass, is_contained = func_planar_hertz_moments(
                origin,
                edges,
                state.dual[i],
                normal,
                state.shapes[i],
                info.point[i],
                info.point[i],
                radius,
                first,
                second,
            )
            state.gram[i], state.load[i], state.mass[i], state.contained[i] = gram, load, mass, is_contained
            minimum, negative = gs.qd_float(0.0), gs.qd_float(0.0)
            if is_contained:
                coefficient = (gram / mass).inverse() @ qd.Vector([1.0, 0.0, 0.0])
                section_center = info.point[i] - normal.dot(info.point[i] - origin) * normal
                remaining = radius * radius - (section_center - info.point[i]).norm_sqr()
                minimum = func_planar_affine_minimum(
                    section_center, normal, info.point[i], remaining, radius, first, second, coefficient
                )
                probe = qd.Vector([0.0, 1.0, 0.0])
                negative = func_planar_affine_minimum(
                    section_center, normal, info.point[i], remaining, radius, first, second, probe
                )
            state.minimum[i], state.negative_minimum[i] = minimum, negative

    @qd.kernel
    def kernel_numerical(
        gauss_angular: qd.Tensor,
        weights_angular: qd.Tensor,
        gauss_radial: qd.Tensor,
        weights_radial: qd.Tensor,
        state: PrimitiveState,
        info: PrimitiveInfo,
    ):
        for i in range(state.radius.shape[0]):
            vertices, radius = state.vertices[i], state.radius[i]
            origin = vertices[0, :]
            edges = qd.Matrix.rows([vertices[1, :] - origin, vertices[2, :] - origin])
            section, plane, polygon, remaining, angles, n_angles, is_contained = func_planar_clip_geometry(
                origin, edges, state.normal[i], info.point[i], radius
            )
            first, second = func_force_frame(info.force[i])
            count = func_planar_sample_count(n_angles, gauss_angular.shape[0], gauss_radial.shape[0], is_contained)
            gram, load = qd.Matrix.zero(gs.qd_float, 3, 3), qd.Matrix.zero(gs.qd_float, 4, 3)
            minimum_barycentric = gs.qd_float(1.0)
            for i_q in range(count):
                position, shape, weight, coordinates = func_planar_hertz_sample(
                    i_q,
                    origin,
                    state.dual[i],
                    state.shapes[i],
                    section,
                    plane,
                    polygon,
                    remaining,
                    angles,
                    n_angles,
                    info.point[i],
                    radius,
                    first,
                    second,
                    gauss_angular,
                    weights_angular,
                    gauss_radial,
                    weights_radial,
                    is_contained,
                )
                gram += weight * coordinates.outer_product(coordinates)
                load += shape.outer_product(weight * coordinates)
                if weight > 0.0:
                    local = state.dual[i] @ (position - origin)
                    barycentric = qd.Vector([1.0 - local.sum(), local[0], local[1]])
                    minimum_barycentric = qd.min(minimum_barycentric, barycentric.min())
            state.numerical_gram[i], state.numerical_load[i] = gram, load
            state.sample_count[i], state.minimum_barycentric[i] = count, minimum_barycentric

    return SimpleNamespace(**locals())


def _field(api, values, rank=0, integer=False):
    dtype = api.gs.qd_int if integer else api.gs.qd_float
    factory = api.V if rank == 0 else api.V_VEC if rank == 1 else api.V_MAT
    element = () if rank == 0 else (values.shape[-1],) if rank == 1 else values.shape[-2:]
    shape = values.shape if rank == 0 else values.shape[:-rank]
    result = factory(*element, dtype=dtype, shape=shape)
    result.from_numpy(values.astype(api.gs.np_int if integer else api.gs.np_float, copy=False))
    return result


def _surface(api, physical, order):
    np = api.np
    n_nodes, n_faces = len(physical["vertices"]), len(physical["quads"])
    info = api.allocate_stress_info(2 * n_nodes, 0, 2 * n_faces, 0, 0, n_load_nodes=2 * n_nodes)
    sides = np.concatenate((physical["quads"], n_nodes + physical["quads"][:, (0, 3, 2, 1)]))
    nodes = np.column_stack((sides, sides[:, (0, 0)]))
    info.surface_nodes.from_numpy(nodes.astype(api.gs.np_int, copy=False))
    vertices, directors = _field(api, physical["vertices"], 1), _field(api, physical["directors"], 1)
    api.kernel_physical_sides(float(physical["thickness"]), vertices, directors, info)
    return info, api.StressSurface(order, info, angular=True)


def _geometry_checks(api, args, checks, surface, load_info):
    np = api.np
    results = []
    for refinement in (8, 16):
        info = api.build_planar_geometry(surface.info, load_info, refinement)
        count = info.grid_positions.shape[0] * info.triangle_nodes.shape[1]
        vertices = api.V_MAT(3, 3, dtype=api.gs.qd_float, shape=(count,))
        dual = api.V_MAT(2, 3, dtype=api.gs.qd_float, shape=(count,))
        normals = api.V_VEC(3, dtype=api.gs.qd_float, shape=(count,))
        shapes = api.V_MAT(3, 4, dtype=api.gs.qd_float, shape=(count,))
        api.kernel_geometry(vertices, dual, normals, shapes, info)
        actual = {
            name: api.qd_to_numpy(value, transpose=True)
            for name, value in (("vertices", vertices), ("dual", dual), ("normals", normals), ("parent_shapes", shapes))
        }
        oracles = [
            _load(np, args.reference / f"oracles/egg75/proxy-k{refinement}/{side}.npz") for side in ("outer", "inner")
        ]
        expected = {name: np.concatenate([oracle[name] for oracle in oracles]) for name in actual}
        for name, value in actual.items():
            checks.close(f"geometry/k{refinement}/{name}", value, expected[name], rtol=1e-10, atol=1e-12)
        checks.condition(f"geometry/k{refinement}/valid", api.qd_to_numpy(info.face_valid, transpose=True).all())
        destination = args.output / f"geometry-k{refinement}.npz"
        np.savez_compressed(destination, **actual)
        results.append({"refinement": refinement, "triangles": count, "readback": _identity(destination)})
    return results


def _primitive_checks(api, directory, output, checks, rules):
    np = api.np
    physical, expected = _load(np, directory / "physical.npz"), _load(np, directory / "oracles.npz")
    values = (
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
    info = api.PrimitiveInfo(**{name: _field(api, physical[key], rank, integer) for name, key, rank, integer in values})
    n = len(physical["radius"])
    state = api.PrimitiveState(
        vertices=api.V_MAT(3, 3, dtype=api.gs.qd_float, shape=(n,)),
        dual=api.V_MAT(2, 3, dtype=api.gs.qd_float, shape=(n,)),
        normal=api.V_VEC(3, dtype=api.gs.qd_float, shape=(n,)),
        shapes=api.V_MAT(3, 4, dtype=api.gs.qd_float, shape=(n,)),
        radius=api.V(dtype=api.gs.qd_float, shape=(n,)),
        gram=api.V_MAT(3, 3, dtype=api.gs.qd_float, shape=(n,)),
        load=api.V_MAT(4, 3, dtype=api.gs.qd_float, shape=(n,)),
        numerical_gram=api.V_MAT(3, 3, dtype=api.gs.qd_float, shape=(n,)),
        numerical_load=api.V_MAT(4, 3, dtype=api.gs.qd_float, shape=(n,)),
        mass=api.V(dtype=api.gs.qd_float, shape=(n,)),
        contained=api.V(dtype=api.gs.qd_int, shape=(n,)),
        minimum=api.V(dtype=api.gs.qd_float, shape=(n,)),
        negative_minimum=api.V(dtype=api.gs.qd_float, shape=(n,)),
        work_error=api.V(dtype=api.gs.qd_float, shape=(n,)),
        rigid_error=api.V(dtype=api.gs.qd_float, shape=(n,)),
        sample_count=api.V(dtype=api.gs.qd_int, shape=(n,)),
        minimum_barycentric=api.V(dtype=api.gs.qd_float, shape=(n,)),
    )
    api.kernel_prepare(float(physical["young"]), float(physical["poisson"]), float(physical["thickness"]), state, info)
    api.kernel_exact(state, info)
    initialized = (
        "vertices",
        "dual",
        "normal",
        "shapes",
        "radius",
        "gram",
        "load",
        "mass",
        "contained",
        "minimum",
        "negative_minimum",
        "work_error",
        "rigid_error",
    )
    actual = {name: api.qd_to_numpy(vars(state)[name], transpose=True) for name in initialized}
    prefix = directory.name
    checks.close(f"{prefix}/radius", actual["radius"], expected["radius"], rtol=1e-12, atol=1e-15)
    checks.exact(f"{prefix}/contained", actual["contained"].astype(bool), expected["contained"])
    checks.condition(f"{prefix}/virtual_work", np.max(actual["work_error"]) <= 1e-13)
    checks.condition(f"{prefix}/rigid_motion", np.max(actual["rigid_error"]) <= 1e-13)
    mask = expected["contained"].astype(bool)
    if mask.any():
        checks.close(
            f"{prefix}/exact_gram_vs_CPUq48", actual["gram"][mask], expected["gram48"][mask], rtol=1e-10, atol=1e-20
        )
        checks.close(
            f"{prefix}/exact_load_vs_CPUq48", actual["load"][mask], expected["load48"][mask], rtol=1e-10, atol=1e-20
        )
        checks.condition(f"{prefix}/positive_affine", (actual["minimum"][mask] >= -2e-12).all())
        checks.condition(f"{prefix}/negative_affine_probe", (actual["negative_minimum"][mask] < 0.0).all())
    if prefix != "clipped12":
        checks.condition(f"{prefix}/contained_witness", mask.any())
    for order in (24, 48):
        angular, radial = rules[order], rules[order // 2]
        api.kernel_numerical(
            angular.info.gauss, angular.info.gauss_weights, radial.info.gauss, radial.info.gauss_weights, state, info
        )
        for name, key in (("numerical_gram", "gram"), ("numerical_load", "load")):
            measured = api.qd_to_numpy(vars(state)[name], transpose=True)
            checks.close(f"{prefix}/q{order}/{key}", measured, expected[f"{key}{order}"], rtol=1e-9, atol=1e-20)
            actual[f"{name}{order}"] = measured
        minimum = api.qd_to_numpy(state.minimum_barycentric, transpose=True)
        checks.condition(f"{prefix}/q{order}/inside_triangle", (minimum >= -2e-8).all(), minimum=float(minimum.min()))
        actual[f"sample_count{order}"] = api.qd_to_numpy(state.sample_count, transpose=True)
    output.mkdir()
    np.savez_compressed(output / "native.npz", **actual)
    return {"fixture": prefix, "triangles": n, "contained": int(mask.sum()), "output": _identity(output / "native.npz")}


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
        parser.error("Choose a fresh output under /mnt/data/zhaofeng.")
    if args._cpu_worker:
        _cpu_worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    from .artifacts import runtime_identity
    from .check_material_cap_gpu import Checks

    manifest = json.loads((args.reference / "manifest.json").read_text())
    if (
        manifest["schema"] != "genesis.planar-loading-reference.v1"
        or manifest["primary_commit"] != PRIMARY
        or not manifest["complete"]
        or not manifest["passed"]
    ):
        raise RuntimeError("Use a successful frozen planar-loading CPU reference.")
    snapshot = json.loads((args.reference / "source-snapshot.json").read_text())
    for row in snapshot["files"]:
        if _identity(args.reference / row["snapshot_path"])["sha256"] != row["sha256"]:
            raise RuntimeError(f"The frozen primary source changed: {row['snapshot_path']}")
    verified_inputs = []
    for relative, record in manifest["files"].items():
        if relative.startswith(("inputs/", "oracles/egg75/proxy-k8/", "oracles/egg75/proxy-k16/")):
            identity = _identity(args.reference / relative)
            if identity["sha256"] != record["sha256"] or identity["bytes"] != record["bytes"]:
                raise RuntimeError(f"The frozen primitive input or oracle changed: {relative}")
            verified_inputs.append(identity)
    result = {
        "schema": "genesis.planar-primitives-check.v1",
        "complete": False,
        "passed": False,
        "identity": runtime_identity(),
        "worker": _identity(__file__),
        "reference": _identity(args.reference / "manifest.json"),
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "primary": PRIMARY,
        "verified_reference_files": verified_inputs,
        "scope": "Untimed geometry, interpolation and pressure-integration primitives. No complete stress or throughput qualification.",
        "native_input_policy": "Physical vertices/directors/material/source vectors and integer triangle fixtures only. CPU geometry, maps and moments remain readback oracles.",
    }
    _json(args.output / "result.json", result)
    started, checks = time.perf_counter(), None
    try:
        cpu = args.output / "cpu-fixtures"
        cpu.mkdir()
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH="",
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
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
        with (args.output / "cpu-fixtures.log").open("w") as log:
            subprocess.run(
                command,
                cwd=args.reference / "reference-source",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        api = _native_api()
        checks = Checks(api)
        engine = Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress"
        source_before = [
            _identity(engine / name)
            for name in ("planar_geometry.py", "planar_moments.py", "contact.py", "surface.py", "quadrilateral.py")
        ]
        api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
        physical = _load(api.np, args.reference / "inputs/egg75/model.npz")
        rules, load_info = {}, None
        for order in (12, 24, 48):
            load_info, rules[order] = _surface(api, physical, order)
            nodes, weights = api.np.polynomial.legendre.leggauss(order)
            checks.close(
                f"gauss/q{order}/nodes",
                api.qd_to_numpy(rules[order].info.gauss, transpose=True),
                0.5 * (nodes + 1.0),
                rtol=1e-12,
                atol=1e-14,
            )
            checks.close(
                f"gauss/q{order}/weights",
                api.qd_to_numpy(rules[order].info.gauss_weights, transpose=True),
                0.5 * weights,
                rtol=1e-12,
                atol=1e-14,
            )
        result["geometry"] = _geometry_checks(api, args, checks, rules[48], load_info)
        fixtures = []
        for directory in sorted(path for path in cpu.iterdir() if path.is_dir()):
            fixtures.append(_primitive_checks(api, directory, args.output / directory.name, checks, rules))
        result["fixtures"] = fixtures
        clipped = json.loads((cpu / "clipped12/fixtures.json").read_text())
        checks.condition("clipped12/contact_count", len(clipped["sources"]) == 12)
        checks.condition("clipped12/multiface_witnesses", all(row["sections"] > 1 for row in clipped["sources"]))
        checks.condition(
            "provenance/source_unchanged", [_identity(Path(row["path"])) for row in source_before] == source_before
        )
        checks.condition(
            "provenance/reference_unchanged", _identity(args.reference / "manifest.json") == result["reference"]
        )
        checks.condition(
            "provenance/inputs_unchanged", [_identity(row["path"]) for row in verified_inputs] == verified_inputs
        )
        checks.condition("provenance/worker_unchanged", _identity(__file__) == result["worker"])
        result.update(
            complete=True,
            passed=checks.passed(),
            checks=len(checks.rows),
            source_files=source_before,
            seconds=time.perf_counter() - started,
        )
    finally:
        if checks is not None:
            _json(args.output / "checks.json", checks.rows)
        if sys.exc_info()[0] is not None:
            (args.output / "failure.txt").write_text(traceback.format_exc())
        _json(args.output / "result.json", result)
    if not result["passed"]:
        raise ArithmeticError("Planar primitive qualification failed. Preserve all numerical evidence.")
    print(json.dumps({"passed": True, "checks": result["checks"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
