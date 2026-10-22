"""Check native shape-only curvature against independent one-ring fits."""

import argparse
import time
from pathlib import Path

import numpy as np
import quadrants as qd
import trimesh

import genesis as gs
from genesis.engine.solvers.rigid.stress.curvature import StressCurvature
from genesis.utils.misc import qd_to_numpy

from ..mechanics import shell_mesh
from .artifacts import runtime_identity, write_json
from .study_quadratic_rest_surface import exact_surface
from .study_vertex_curvature import vertex_fits


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--fit", choices=["baseline", "quadric"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "fit": args.fit,
            "targets": {"native_relative": 1e-9, "normal_reversal_relative": 1e-10},
            "scope": "Same-shape numerical conformance on three closed shapes at two resolutions plus a rank-deficient flat mesh; sign reversal and disconnected identity. Warm constructor measurements include allocations and synchronization, not scene throughput.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    controls = []
    for level in [1, 2]:
        xyz, _, faces, metadata = shell_mesh(level, 2, 0.0005)
        original = xyz[: metadata.outer_surface_vertices]
        unit = np.column_stack(
            [original[:, :2] / (0.022 * (1 - 0.18 * original[:, 2:3] / 0.03)), original[:, 2] / 0.03]
        )
        for shape in ["sphere", "ellipsoid", "egg"]:
            vertices = (
                unit * 0.02
                if shape == "sphere"
                else unit * np.array([0.012, 0.015, 0.009])
                if shape == "ellipsoid"
                else original
            )
            path = args.output / f"{shape}-{level}.obj"
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
            entity = scene.add_entity(morph=gs.morphs.Mesh(file=path, convexify=False, decimate=False, fixed=True))
            controls.append((shape, level, entity))
    path = args.output / "flat.obj"
    trimesh.Trimesh(
        vertices=[[-0.01, -0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0], [-0.01, 0.01, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    ).export(path)
    flat = scene.add_entity(morph=gs.morphs.Mesh(file=path, convexify=False, decimate=False, fixed=True))
    scene.build(n_envs=1)
    info = scene.rigid_solver.dyn_info
    epsilon = float(np.finfo(gs.np_float).eps)
    cache = StressCurvature(info, epsilon)
    native = qd_to_numpy(cache.info.vertices_tensor, transpose=True, copy=True)
    original_normals = qd_to_numpy(info.verts.init_normal, transpose=True, copy=True)
    fitted = np.ones(len(native), dtype=bool)
    if args.fit == "quadric":
        fitted = qd_to_numpy(cache.info.vertices_quadric_valid, transpose=True, copy=True)
    rows = []
    for shape, level, entity in controls:
        assert len(entity.base_link.geoms) == 1
        geometry = entity.base_link.geoms[0]
        vertices, faces, normals = geometry.init_verts, geometry.init_faces, geometry.init_normals
        expected, _, _ = vertex_fits(vertices, faces, normals)
        selection = slice(geometry.vert_start, geometry.vert_end)
        target = expected["implicit_quadric" if args.fit == "quadric" else "face_average"]
        error = np.linalg.norm(native[selection] - target) / np.linalg.norm(target)
        assert error < 1e-9 and fitted[selection].all(), (shape, level, error)
        _, _, analytic = exact_surface(vertices, shape)
        physical = np.linalg.norm(native[selection] - analytic) / np.linalg.norm(analytic)
        rows.append(
            {
                "shape": shape,
                "level": level,
                "vertices": len(vertices),
                "faces": len(faces),
                "native_relative": float(error),
                "analytic_curvature_rms_relative": float(physical),
                "quadric_vertices": int(fitted[selection].sum()) if args.fit == "quadric" else 0,
            }
        )
    geometry = flat.base_link.geoms[0]
    selection = slice(geometry.vert_start, geometry.vert_end)
    # Exercise exact planar input after asset import, which may thicken open collision meshes.
    positions = qd_to_numpy(info.verts.init_pos, transpose=True, copy=True)
    planar_positions, planar_normals = positions.copy(), original_normals.copy()
    planar_positions[selection, 2] = 0
    planar_normals[selection] = [0, 0, 1]
    info.verts.init_pos.from_numpy(planar_positions)
    info.verts.init_normal.from_numpy(planar_normals)
    planar_cache = StressCurvature(info, epsilon)
    planar = qd_to_numpy(planar_cache.info.vertices_tensor, transpose=True, copy=True)
    np.testing.assert_allclose(planar[selection], 0, atol=1e-12, rtol=0)
    if args.fit == "quadric":
        planar_fitted = qd_to_numpy(planar_cache.info.vertices_quadric_valid, transpose=True, copy=True)
        assert not planar_fitted[selection].any()
    info.verts.init_pos.from_numpy(positions)
    info.verts.init_normal.from_numpy(-original_normals)
    reversed_cache = StressCurvature(info, epsilon)
    reversed_tensor = qd_to_numpy(reversed_cache.info.vertices_tensor, transpose=True, copy=True)
    reversal_error = np.linalg.norm(native + reversed_tensor) / np.linalg.norm(native)
    assert reversal_error < 1e-10, reversal_error
    info.verts.init_normal.from_numpy(original_normals)
    timings = []
    for _ in range(5):
        qd.sync()
        started = time.perf_counter()
        cache = StressCurvature(info, epsilon)
        qd.sync()
        timings.append(time.perf_counter() - started)
    np.savez_compressed(
        args.output / "fields.npz",
        tensor_per_m=native,
        reversed_tensor_per_m=reversed_tensor,
        quadric_valid=fitted,
        normals=original_normals,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "fit": args.fit,
            "rows": rows,
            "normal_reversal_relative": float(reversal_error),
            "warm_constructor_seconds": timings,
            "persistent_added_bytes": len(native) if args.fit == "quadric" else 0,
            "temporary_fit_bytes": len(native) * (8 + 4 + 36 * 8 + 6 * 8) if args.fit == "quadric" else 0,
            "flat_vertices": geometry.n_verts,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
