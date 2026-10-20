"""Compare conservative outer P2 transfer with saved finite-radius DPR loads on CPU.

Both paths use the same measured centroid forces, mesh, material and body load.
This isolates transfer/elastic response; it does not predict unknown pressure or
admit the pending fine physical reference. Original native fields stay immutable.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json
from .p2_elastic import FreeElasticP2, quadratic_triangle_weights


def field_norm(stress, volumes):
    # Exact physical-volume integral of squared affine P2 elastic stress.
    mean_square = ((stress**2).sum(axis=(1, 2, 3)) + (stress.sum(axis=1)**2).sum(axis=(1, 2))) / 20
    return np.sqrt(volumes @ mean_square)


def peaks(stress):
    deviator = stress - np.trace(stress, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return {"raw_vm_pa": np.sqrt(1.5 * (deviator**2).sum(axis=(-2, -1))).max().item(),
            "raw_tensile_pa": np.linalg.eigvalsh(stress)[..., -1].max().item()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--levels", type=int, nargs="+", default=[0, 1])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    start = json.loads((args.run / "start.json").read_text())
    terminal = json.loads((args.run / "result.json").read_text())
    assert terminal["completed"] and terminal["mechanical_history_passed"]
    assert start["args"]["method"] == "dpr" and start["args"]["source_reduction"] == "none"
    mesh = Path(start["mesh"]["path"])
    assert file_id(mesh) == start["mesh"]
    targets = {"force_and_moment": 1e-12, "virtual_work": 1e-12,
               "elastic_equation": 1e-7, "same_load_cpu_native_stress": 1e-6}
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "run": str(args.run), "original_start": file_id(args.run / "start.json"),
        "original_result": file_id(args.run / "result.json"), "mesh": file_id(mesh),
        "targets": targets, "levels": args.levels})
    with np.load(mesh) as saved:
        vertices, tets, outer, faces = (saved[k] for k in ("vertices", "tetrahedra", "outer_vertices", "mid_faces"))
    started = perf_counter()
    model = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000, column_faces=faces,
                          through_wall_ordering=True, cvxopt_factor=True)
    setup_seconds = perf_counter() - started
    _, _, face_nodes = model.surface(faces)
    shape = quadratic_triangle_weights(np.full((1, 3), 1 / 3))[0]
    points = outer[faces].mean(axis=1)
    np.testing.assert_allclose(np.einsum("c,fci->fi", shape, model.vertices[face_nodes]), points, rtol=0, atol=1e-14)
    volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
    assert volumes.min() > 0
    rng = np.random.default_rng(20261020)
    virtual_displacement = rng.normal(size=model.vertices.shape)
    rows = []
    for level in args.levels:
        original = args.run / f"frames/stress-level{level}.npz"
        with np.load(original) as saved:
            coordinates, native_tets, measured, original_points, native_force, native_rhs, native_stress = (
                saved[k] for k in ("vertices_m", "tetrahedra", "original_contact_forces_n",
                    "original_contact_points_m", "forces_n", "rhs_n", "stress_pa"))
        np.testing.assert_array_equal(native_tets, tets)
        np.testing.assert_allclose(original_points, points, rtol=0, atol=1e-14)
        distance, permutation = cKDTree(model.vertices).query(coordinates)
        assert distance.max() < 1e-14 and len(np.unique(permutation)) == len(model.vertices)
        baseline, body = np.empty_like(native_force), np.empty_like(native_force)
        baseline[permutation], body[permutation] = native_force, native_rhs - native_force
        direct = np.zeros_like(model.vertices)
        np.add.at(direct, face_nodes.reshape(-1), (shape[None, :, None] * measured[:, None]).reshape(-1, 3))
        scale = np.linalg.norm(measured, axis=1).sum()
        force_error = np.linalg.norm(direct.sum(axis=0) - measured.sum(axis=0)) / scale
        moment_error = np.linalg.norm(np.cross(model.vertices, direct).sum(axis=0) - np.cross(points, measured).sum(axis=0)) / (scale * 0.03)
        point_delta = np.einsum("c,fci->fi", shape, virtual_displacement[face_nodes])
        expected_work = np.sum(measured * point_delta)
        work_error = abs(np.sum(direct * virtual_displacement) - expected_work) / np.sum(np.abs(measured * point_delta))
        assert max(force_error, moment_error) <= targets["force_and_moment"]
        assert work_error <= targets["virtual_work"]
        fields, audits = {}, {}
        for variant, contact in (("fixed_radius", baseline), ("distributed_outer", direct)):
            stress, displacement, correction, residual = model.solve(contact + body)
            rhs = contact + body - correction
            equation_error = np.linalg.norm(residual) / np.linalg.norm(rhs)
            assert equation_error <= targets["elastic_equation"]
            path = args.output / "frames" / f"{variant}-level{level}.npz"
            np.savez_compressed(path, vertices_m=vertices, tetrahedra=tets, stress_pa=stress,
                displacement_p2_m=displacement, p2_coordinates_m=model.vertices, forces_n=contact,
                rhs_n=rhs, equation_residual_n=residual, body_forces_n=body,
                source_points_m=points, source_forces_n=measured)
            fields[variant] = stress
            audits[variant] = {"field": file_id(path), "elastic_equation_relative_residual": equation_error.item(),
                "raw_peaks": peaks(stress), "mass_projected_force_norm_n": np.linalg.norm(correction).item()}
        conformance = field_norm(fields["fixed_radius"] - native_stress, volumes) / field_norm(native_stress, volumes)
        assert conformance <= targets["same_load_cpu_native_stress"]
        rows.append({"level": level, "original": file_id(original), "force_relative_error": force_error.item(),
            "moment_relative_error": moment_error.item(), "virtual_work_relative_error": work_error.item(),
            "cpu_native_same_load_stress_relative_error": conformance.item(),
            "transfer_induced_stress_relative_change": (field_norm(fields["distributed_outer"] - fields["fixed_radius"], volumes) / field_norm(fields["distributed_outer"], volumes)).item(),
            "variants": audits})
        write_json(args.output / "progress.json", {"cases": rows})
    write_json(args.output / "result.json", {"passed": True, "cases": rows, "targets": targets,
        "setup_seconds": setup_seconds, "elapsed_seconds": perf_counter() - started,
        "whole_goal_complete": False, "native_variant_admitted": False,
        "scope": "Independent CPU linear P2 response on the unchanged native geometry. Exact outer-face centroid transfer preserves force, moment and virtual work, without a second finite patch. Original fixed-radius loads and identical recorded body loads supply the paired control. Same-load CPU/native complete-stress conformance is required before interpreting the load-transfer contrast. Numerical gauge elimination is followed by free-link mass projection and full-equation checks, not physical shell pins. This neither predicts pressure from collider points nor establishes a fine-reference physical stress error, continuous history or production throughput."})


if __name__ == "__main__":
    main()
