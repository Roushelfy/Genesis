"""Curved native atlas controls for junctions, footprint coverage and material sides."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.atlas import StressAtlas
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.scatter import create_scatter_workspace, kernel_scatter_faces
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.shell_cpu import geometry

from .artifacts import runtime_identity, write_json
from .audit_gpu_candidates import per_environment
from .voxel_mesh import voxel_mesh
from .check_native_atlas import cpu_control
from .check_native_candidate_kernels import Control


def surface_point(vertices, face, local):
    if len(face) == 4:
        x, y = local
        weights = 0.25 * np.array([(1 - x) * (1 - y), (1 + x) * (1 - y), (1 + x) * (1 + y), (1 - x) * (1 + y)])
        dx = 0.25 * np.array([y - 1, 1 - y, 1 + y, -1 - y])
        dy = 0.25 * np.array([x - 1, -1 - x, 1 + x, 1 - x])
        tangent = dx @ vertices[face]
        normal = np.cross(tangent, dy @ vertices[face])
    else:
        weights = np.array([1 - local.sum(), *local])
        tangent = vertices[face[1]] - vertices[face[0]]
        normal = np.cross(tangent, vertices[face[2]] - vertices[face[0]])
    return weights @ vertices[face], normal / np.linalg.norm(normal), tangent / np.linalg.norm(tangent), weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--geometry", choices=("egg", "nonconvex"), default="egg")
    parser.add_argument("--complete-boundary", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    result = dict(identity=runtime_identity(), quadrature=32, pressure_quadrature="edge", complete=False, rows=[])
    write_json(args.output / "result.json", result)
    if args.geometry == "egg":
        asset = Path(gs.__file__).parent.parent / "examples/rigid/assets/hollow_egg/level1"
        asset = asset / ("shell.npz" if args.stress_backend == "shell" else "elastic.npz")
        if args.complete_boundary and args.stress_backend == "volume":
            with np.load(asset) as original:
                vertices, tetrahedra = original["vertices"], original["tetrahedra"]
            faces, counts = np.unique(
                np.sort(tetrahedra[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape((-1, 3)), axis=1),
                axis=0,
                return_counts=True,
            )
            asset = args.output / "complete-boundary.npz"
            np.savez(asset, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces[counts == 1])
        width, radius = 0.003, 8e-5
    else:
        asset = args.output / "nonconvex.npz"
        width, radius = 0.0003, 3e-5
        if args.stress_backend == "volume":
            vertices, tetrahedra, faces = voxel_mesh("u")
            np.savez(asset, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
        else:
            vertices, quads = geometry.plate_mesh(3, Lx=0.004)
            vertices[:, 2] = 0.15 * np.abs(vertices[:, 0])
            np.savez(asset, vertices=vertices, quads=quads)
    options = gs.options.RigidStressOptions.dpr(
        mesh=asset, backend=args.stress_backend, output_mode="full", quadrature=32, atlas_regions=6
    )
    model = ShellModel(options) if args.stress_backend == "shell" else StressModel(options)
    state = model.create_state(4, "full")
    contacts = create_contacts(4, 4, radius, hertz_profile=True)
    surface = StressSurface(32, model.load_info)
    atlas = StressAtlas(model, contacts, 6, width)
    workspace = create_scatter_workspace(4, 32)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(4,))
    omega.fill(0)
    vertices = qd_to_numpy(model.load_info.vertices)
    faces = qd_to_numpy(model.load_info.surface_nodes)[:, : (4 if args.stress_backend == "shell" else 3)]
    normals = qd_to_numpy(surface.info.face_normal)
    inner = np.einsum("ij,ij->i", vertices[faces].mean(axis=1) - vertices.mean(axis=0), normals) < 0
    result["authored_inner_faces"] = int(np.count_nonzero(inner))
    result["complete_boundary"] = args.complete_boundary
    distances = qd_to_numpy(atlas.info.distances)
    boundary_nodes = qd_to_numpy(model.surface_inverse.info.nodes)
    axis = np.linspace(-0.95, 0.95, 17) if args.stress_backend == "shell" else np.linspace(0.02, 0.96, 17)
    coordinates = np.array([(x, y) for x in axis for y in axis if args.stress_backend == "shell" or x + y < 0.99])
    best = (np.inf, 0, coordinates[0])
    for i_f, face in enumerate(faces):
        for local in coordinates:
            *_, weights = surface_point(vertices, face, local)
            values = np.sort(distances[:, face] @ weights)
            gap = values[2] - values[0] if values[2] < 1e29 else np.inf
            if gap < best[0]:
                best = (gap, i_f, local)
    gap, junction, local = best
    result["junction"] = dict(gap=float(gap), face=int(junction), local=local.tolist(), transition=width)
    reference = options.model_copy(update={"atlas_regions": 0})
    fields = (("displacement", state.displacement), ("stress_tensor", state.stress_tensor), ("peak", state.peak))
    if args.stress_backend == "shell":
        fields += (
            ("membrane", state.membrane_resultant),
            ("bending", state.bending_resultant),
            ("shear", state.transverse_resultant),
        )
    previous = None
    for step in range(8):
        positions, forces, inward = (np.zeros((4, 4, 3)) for _ in range(3))
        valid = np.ones((4, 4), dtype=bool)
        for i_c in range(4):
            i_f = junction if i_c == 0 else (junction + i_c * len(faces) // 4) % len(faces)
            if args.complete_boundary and i_c in (1, 2):
                available = np.flatnonzero(inner if i_c == 1 else ~inner)
                i_f = available[len(available) // 2]
            for i_b in range(4):
                location = local.copy() if i_c == 0 else np.array([0.17, 0.21])
                location[0] += (1 if step % 2 else -1) * 1e-7
                point, outward, tangent, _ = surface_point(vertices, faces[i_f], location)
                if outward @ normals[i_f] < 0:
                    outward *= -1
                friction = (1 if step < 4 else -1) * (0.05 + 0.03 * i_b)
                positions[i_c, i_b] = point
                forces[i_c, i_b] = 0.01 * (1 + 0.2 * i_b) * (-outward + friction * tangent)
                inward[i_c, i_b] = -outward
        if step == 4:
            valid[:, 1] = False
        elif step == 5:
            valid[:] = False
        if step >= 6:
            positions, forces, inward, valid = (value[::-1].copy() for value in (positions, forces, inward, valid))
        contacts.position.from_numpy(positions)
        contacts.force.from_numpy(forces)
        contacts.normal.from_numpy(inward)
        contacts.valid.from_numpy(valid)
        contacts.friction.fill(0.5)
        kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
        kernel_pressure(contacts, surface.info)
        kernel_scatter(contacts, state, model.load_info, surface.info)
        assert (qd_to_numpy(contacts.status) == 0).all()
        reference_load = qd_to_numpy(state.force, transpose=True)
        model.recover(omega, state, reference, surface_load=True)
        expected = {name: qd_to_numpy(array, transpose=True) for name, array in fields}
        atlas.prepare(contacts, model.load_info, surface.info, width)
        kernel_scatter_faces(contacts, state, model.load_info, surface.info, workspace, True, atlas=atlas.scatter)
        actual_load = qd_to_numpy(state.force, transpose=True)
        grouped = np.zeros_like(reference_load)
        grouped[:, boundary_nodes] = qd_to_numpy(atlas.state.force).sum(axis=1).transpose(1, 0, 2)
        model.recover(omega, state, options, surface_load=True, atlas=atlas)
        actual = {name: qd_to_numpy(array, transpose=True) for name, array in fields}
        weights = qd_to_numpy(atlas.state.weights)
        weights *= valid[None]
        row = dict(
            step=step,
            load_error=float(per_environment(actual_load, reference_load).max()),
            grouped_error=float(per_environment(grouped, reference_load).max()),
            errors={name: float(per_environment(actual[name], expected[name]).max()) for name, _ in fields},
            contributors=int(np.count_nonzero(weights, axis=0).max()),
            coverage_fallbacks=int(np.count_nonzero(qd_to_numpy(atlas.state.fallback))),
            force_error=float(qd_to_numpy(contacts.force_error).max()),
            moment_error=float(qd_to_numpy(contacts.moment_error).max()),
        )
        if step == 1:
            row["continuous_weights_change"] = float(np.max(np.abs(weights - previous)))
            assert row["continuous_weights_change"] < 1e-4
        previous = weights
        result["rows"].append(row)
        write_json(args.output / "result.json", result)
        assert qd_to_numpy(state.valid).all()
        assert max(row["load_error"], row["grouped_error"]) < 1e-10
        assert max(row["errors"].values()) < 1e-8
        if step == 0:
            result["cpu_control"] = cpu_control(Control(model, state, omega, options))
            np.savez_compressed(args.output / "fields.npz", **actual, load=actual_load, weights=weights)
        if step == 5:
            assert np.count_nonzero(actual_load) == 0 and np.count_nonzero(actual["stress_tensor"]) == 0
    assert max(row["contributors"] for row in result["rows"]) >= 3
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
