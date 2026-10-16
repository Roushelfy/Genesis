"""Separate frozen-polyhedron volume changes from CAD changes and reference quadrature error."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from skfem import MeshTet


def reentrant_seed_edges(mesh):
    boundary = mesh.boundary_facets()
    faces = mesh.facets[:, boundary].T
    triangles = mesh.p.T[faces]
    centres = mesh.p.T[mesh.t[:, mesh.f2t[0, boundary]].T].mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals[np.einsum("fi,fi->f", normals, centres - triangles.mean(axis=1)) > 0] *= -1
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    local = np.array([[0, 1], [0, 2], [1, 2]])
    edges = np.sort(faces[:, local].reshape(-1, 2), axis=1)
    owners = np.repeat(np.arange(len(faces)), 3)
    opposite = faces[:, [2, 1, 0]].reshape(-1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    starts = np.flatnonzero((edges[order][1:] == edges[order][:-1]).all(axis=1))
    first, second = order[starts], order[starts + 1]
    angles = np.arccos(np.clip(np.einsum("ei,ei->e", normals[owners[first]], normals[owners[second]]), -1, 1))
    outside = np.einsum("ei,ei->e", normals[owners[first]], mesh.p.T[opposite[second]] - mesh.p.T[edges[first, 0]])
    selected = (outside > 1e-12) & (angles > np.deg2rad(2))
    return mesh.p.T[edges[first[selected]]], angles[selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, quadrature_rows = [], []
    for geometry in ("ellipsoid", "u"):
        mesh_dir = args.runtime_root / f"fixed-poly-{geometry}-v1/meshes"
        with np.load(mesh_dir / "level0.npz") as mesh_file:
            seed = MeshTet(mesh_file["vertices"].T, mesh_file["tetrahedra"].T, sort_t=False)
        creases, angles = reentrant_seed_edges(seed)
        for case in ("opposing", "bending", "asymmetric", "near_slip", "multiple"):
            previous_tensor, previous_peak, previous_energy = None, None, None
            for level in range(3):
                run = args.runtime_root / f"fixed-poly-ref-{geometry}-l{level}-v1"
                assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
                snapshot = json.loads((run / f"results/{case}/snapshot.json").read_text())
                probes = Path(str(run) + "-probes") / f"results/{case}/reference_field.npz"
                with np.load(probes) as saved:
                    assert saved["valid"].all()
                    tensor = saved["tensor_Pa"]
                peak, energy = snapshot["reference_peak_Pa"], snapshot["reference_energy_J"]
                row = {
                    "geometry": geometry,
                    "case": case,
                    "level": level,
                    "tetrahedra": json.loads((mesh_dir / f"level{level}.json").read_text())["tetrahedra"],
                    "raw_global_peak_Pa": peak,
                    "raw_global_peak_position_m": snapshot["reference_peak_position_m"],
                    "energy_J": energy,
                    "complete_budget_fraction": snapshot["reference_complete_budget_fraction"],
                }
                if len(creases):
                    position = np.array(snapshot["reference_peak_position_m"])
                    vectors = creases[:, 1] - creases[:, 0]
                    fractions = np.clip(
                        np.einsum("ei,ei->e", position - creases[:, 0], vectors) / np.sum(vectors * vectors, axis=1),
                        0,
                        1,
                    )
                    distances = np.linalg.norm(creases[:, 0] + fractions[:, None] * vectors - position, axis=1)
                    selected = np.argmin(distances)
                    row["distance_to_seed_reentrant_crease_m"] = float(distances[selected])
                    row["nearest_seed_crease_angle_deg"] = float(np.rad2deg(angles[selected]))
                if previous_tensor is not None:
                    row["adjacent_probe_tensor_change"] = float(
                        np.linalg.norm(previous_tensor - tensor) / np.linalg.norm(tensor)
                    )
                    row["adjacent_raw_peak_change"] = abs(previous_peak / peak - 1)
                    row["adjacent_energy_change"] = abs(previous_energy / energy - 1)
                rows.append(row)
                previous_tensor, previous_peak, previous_energy = tensor, peak, energy
    for geometry in ("u", "capsule"):
        for case in ("opposing", "bending", "asymmetric", "near_slip", "multiple"):
            previous_tensor, previous_force = None, None
            for level in (2, 3, 4):
                run = args.runtime_root / f"ref-quad-{geometry}-l{level}-v1"
                snapshot = json.loads((run / f"results/{case}/snapshot.json").read_text())
                with np.load(run / f"results/{case}/reference_field.npz") as saved:
                    tensor, force = saved["tensor_Pa"], saved["force_N"]
                row = {
                    "geometry": geometry,
                    "case": case,
                    "subdivision_level": level,
                    "raw_global_peak_Pa": snapshot["reference_peak_Pa"],
                    "energy_J": snapshot["reference_energy_J"],
                }
                if previous_tensor is not None:
                    row["adjacent_full_tensor_change"] = float(
                        np.linalg.norm(previous_tensor - tensor) / np.linalg.norm(tensor)
                    )
                    row["adjacent_nodal_force_change"] = float(
                        np.linalg.norm(previous_force - force) / np.linalg.norm(force)
                    )
                quadrature_rows.append(row)
                previous_tensor, previous_force = tensor, force
    for filename, dataset in (("fixed_geometry.csv", rows), ("reference_quadrature.csv", quadrature_rows)):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in dataset for key in row}))
            writer.writeheader()
            writer.writerows(dataset)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Frozen polyhedron volume refinement and separate prescribed-traction quadrature studies",
                "volume_rows": rows,
                "quadrature_rows": quadrature_rows,
                "limitations": "Growing raw peaks at seed reentrant edges describe the polygonal problem; smooth-CAD convergence requires independent finer geometry",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"volume_rows": len(rows), "quadrature_rows": len(quadrature_rows)}))


if __name__ == "__main__":
    main()
