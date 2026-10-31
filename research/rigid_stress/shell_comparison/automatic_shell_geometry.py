"""Reproducible low-resolution inputs and bounded geometry/mechanical controls for automatic shell preparation."""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh

from examples.rigid import stress_mesh_asset
from examples.rigid.stress_mesh_asset import prepare_stress_mesh
from examples.rigid.stress_shell_asset import surface_to_quads
from research.rigid_stress.qipc_comparison.artifacts import file_id, runtime_identity, write_json
from research.rigid_stress.shell_cpu import core


def surface_recipe(name: str, level: int = 0) -> trimesh.Trimesh:
    """Return a closed egg, a closed narrow neck, an open U, or a triangulated flat strip in metres."""
    if name == "grasp":
        # The established grasp fits this exterior. Here it is explicitly reinterpreted as a midsurface.
        return stress_mesh_asset.load_surface(
            Path(stress_mesh_asset.__file__).parent / "assets/hollow_egg/level1/egg_outer.obj"
        )
    if name in ("egg", "neck"):
        surface = trimesh.creation.icosphere(subdivisions=level, radius=1.0)
        if name == "neck":
            surface.vertices[:, :2] *= (0.35 + 0.65 * surface.vertices[:, 2] ** 2)[:, None]
        surface.vertices *= [0.021, 0.016, 0.03]
        return surface
    if name == "u":
        path = np.array([[-0.02, 0.02], [-0.02, -0.02], [0, -0.03], [0.02, -0.02], [0.02, 0.02]])
        vertices = np.zeros((len(path), 2, 3))
        vertices[:, :, 0] = path[:, :1]
        vertices[:, :, 1] = [-0.015, 0.015]
        vertices[:, :, 2] = path[:, 1:]
        first = 2 * np.arange(len(path) - 1)
        faces = np.stack(
            (np.column_stack((first, first + 2, first + 3)), np.column_stack((first, first + 3, first + 1))), axis=1
        ).reshape((-1, 3))
        return trimesh.Trimesh(vertices=vertices.reshape((-1, 3)), faces=faces, process=False)
    if name != "plate":
        raise ValueError(f"Unknown shell recipe: {name}")
    nx, ny = 2 ** (level + 1), 2**level
    x, y = np.meshgrid(np.linspace(0, 0.04, nx + 1), np.linspace(-0.005, 0.005, ny + 1))
    vertices = np.column_stack((x.reshape(-1), y.reshape(-1), np.zeros(x.size)))
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    first = (j * (nx + 1) + i).reshape(-1)
    faces = np.stack(
        (np.column_stack((first, first + 1, first + nx + 2)), np.column_stack((first, first + nx + 2, first + nx + 1))),
        axis=1,
    ).reshape((-1, 3))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def mechanical_controls(output: Path) -> list:
    """Check membrane/bending patch energies, mechanisms and a bounded cantilever refinement against beam theory."""
    results = []
    for level in range(3):
        vertices, quads = surface_to_quads(surface_recipe("plate", level))
        model = core.ShellModel(vertices, quads, nu=0.0, gauge="translation")
        x, y = vertices[:, 0], vertices[:, 1]
        membrane, bending = np.zeros((2, len(vertices), 6))
        membrane[:, 0] = 1e-4 * x
        bending[:, 2] = 0.5 * x**2
        bending[:, 4] = -x
        membrane_energy = membrane.reshape(-1) @ (model.K @ membrane.reshape(-1)) / 2
        bending_energy = bending.reshape(-1) @ (model.K @ bending.reshape(-1)) / 2
        area, length, width = 0.04 * 0.01, 0.04, 0.01
        expected_membrane = model.E * model.h * 1e-8 * area / 2
        expected_bending = model.E * model.h**3 * area / 24
        force = np.zeros((len(vertices), 6))
        tip = np.flatnonzero(np.isclose(x, length))
        tip = tip[np.argsort(y[tip])]
        intervals = np.diff(y[tip])
        force[tip[:-1], 2] += intervals / (2 * width)
        force[tip[1:], 2] += intervals / (2 * width)
        fixed = (6 * np.flatnonzero(np.isclose(x, 0))[:, None] + np.arange(6)).reshape(-1)
        displacement = model.solve(force.reshape(-1), fixed_dofs=fixed).reshape((-1, 6))
        compliance = np.sum(force * displacement)
        expected_compliance = 4 * length**3 / (model.E * width * model.h**3)
        expected_compliance += length / ((5 / 6) * model.G * width * model.h)
        row = {
            "level": level,
            "nodes": len(vertices),
            "quads": len(quads),
            "membrane_energy_relative_error": abs(membrane_energy / expected_membrane - 1),
            "bending_energy_relative_error": abs(bending_energy / expected_bending - 1),
            "cantilever_compliance": compliance,
            "beam_compliance": expected_compliance,
            "cantilever_relative_error": abs(compliance / expected_compliance - 1),
        }
        if level == 0:
            diagonal = np.sqrt(model.K.diagonal())
            eigenvalues = np.linalg.eigvalsh(model.K.toarray() / diagonal[:, None] / diagonal[None, :])
            row["near_zero_modes"] = int(np.count_nonzero(abs(eigenvalues) < 1e-10))
            row["smallest_elastic_eigenvalue"] = eigenvalues[6]
            np.save(output / "patch-eigenvalues.npy", eigenvalues)
            assert row["near_zero_modes"] == 6, row
        results.append(row)
        write_json(output / "mechanics.json", results)
        assert row["membrane_energy_relative_error"] < 1e-9, row
        assert row["bending_energy_relative_error"] < 1e-8, row
    assert results[-1]["cantilever_relative_error"] < results[0]["cantilever_relative_error"], results
    assert results[-1]["cantilever_relative_error"] < 0.05, results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanics", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "runtime.json", runtime_identity(require_gpu_inventory=False))
    results = []
    for name, level in (("egg", 0), ("neck", 1), ("u", 0), ("plate", 0)):
        surface = surface_recipe(name, level)
        source = args.output / f"{name}.obj"
        surface.export(source)
        started = time.perf_counter()
        prepared = prepare_stress_mesh(source, None, "shell", 1.0, None, args.output / "cache")
        cold = time.perf_counter() - started
        started = time.perf_counter()
        cached = prepare_stress_mesh(source, None, "shell", 1.0, None, args.output / "cache")
        warm = time.perf_counter() - started
        assert cached.elastic == prepared.elastic and cached.collision == prepared.collision
        assert abs(cached.minimum_z - prepared.minimum_z) < 1e-15
        with np.load(prepared.elastic) as asset:
            nodes, quads = len(asset["vertices"]), len(asset["quads"])
        collision = trimesh.load_scene(prepared.collision).to_mesh()
        results.append(
            {
                "name": name,
                "level": level,
                "triangles": len(surface.faces),
                "nodes": nodes,
                "quads": quads,
                "dofs": 6 * nodes,
                "cold_seconds": cold,
                "warm_seconds": warm,
                "material_volume": collision.volume,
                "is_convex": collision.is_convex,
                "prepared": {key: str(value) for key, value in asdict(prepared).items()},
                "source": file_id(source),
                "asset": file_id(prepared.elastic),
            }
        )
        write_json(args.output / "geometry.json", results)
    if args.mechanics:
        mechanical_controls(args.output)


if __name__ == "__main__":
    main()
