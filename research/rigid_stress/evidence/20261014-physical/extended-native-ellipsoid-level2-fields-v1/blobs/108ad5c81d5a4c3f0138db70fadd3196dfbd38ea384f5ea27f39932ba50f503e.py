"""Measure U mesh geometry and mass approximation against independent analytic and CAD measures."""

import argparse
import hashlib
import json
from pathlib import Path

import gmsh
import numpy as np
from skfem import MeshTet

from .summarize_fine_references import closed_source
from .validation_meshes import build_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, nargs="+", required=True)
    parser.add_argument("--field-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.field_audit)
    assert int((args.field_audit / "returncode.txt").read_text()) == 0
    fields = json.loads((args.field_audit / "results/summary.json").read_text())
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("u_geometry_measures")
        build_geometry("u", 1)
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        assert len(volumes) == 1
        cad_volume = sum(gmsh.model.occ.getMass(d, t) for d, t in volumes)
        boundary = gmsh.model.getBoundary(volumes, combined=True, oriented=False)
        cad_area = sum(gmsh.model.occ.getMass(d, t) for d, t in boundary)
    finally:
        gmsh.finalize()
    # Semicircular tube plus two straight cylinders and their two outer hemispheres.
    bend, tube, arm, density = 0.008, 0.003, 0.024, 2000
    length = np.pi * bend + 2 * arm
    analytic_volume = np.pi * tube**2 * length + 4 * np.pi * tube**3 / 3
    analytic_area = 2 * np.pi * tube * length + 4 * np.pi * tube**2
    np.testing.assert_allclose(cad_volume, analytic_volume, rtol=1e-10)
    np.testing.assert_allclose(cad_area, analytic_area, rtol=1e-10)
    rows = []
    for path in args.mesh:
        producer = path.parent.parent
        closed_source(producer)
        assert int((producer / "returncode.txt").read_text()) == 0
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            vertices, tets = saved["vertices"], saved["tetrahedra"]
        mesh = MeshTet(vertices.T, tets.T, sort_t=False)
        volume = abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])).sum() / 6
        triangles = vertices[mesh.facets[:, mesh.boundary_facets()].T]
        area = (
            np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum()
            / 2
        )
        matched = [row for row in fields["rows"] if row["body_mesh_sha256"] == sha]
        if matched:
            assert len(matched) == 1
            np.testing.assert_allclose(matched[0]["parts"][0]["discrete_mass_kg"], density * volume, rtol=1e-12)
        rows.append(
            {
                "mesh": str(path),
                "mesh_sha256": sha,
                "tetrahedra": len(tets),
                "mesh_volume_m3": float(volume),
                "mesh_surface_area_m2": float(area),
                "discrete_mass_kg": float(density * volume),
                "volume_relative_error_from_CAD": float(volume / cad_volume - 1),
                "area_relative_error_from_CAD": float(area / cad_area - 1),
                "gravity_force_magnitude_N": float(density * volume * 9.81),
                "independent_consistent_mass_match": bool(matched),
            }
        )
    result = {
        "scope": __doc__,
        "cad_volume_m3": cad_volume,
        "analytic_volume_m3": float(analytic_volume),
        "cad_surface_area_m2": cad_area,
        "analytic_surface_area_m2": float(analytic_area),
        "continuum_mass_kg_at_declared_density": cad_volume * density,
        "declared_density_kg_m3": density,
        "rows": rows,
        "goal_completion": False,
        "limitations": "Analytic and OpenCASCADE measures independently check the declared full U geometry; affine mesh geometry and consistent FEM mass then quantify approximation errors without density, geometry or force retuning. These geometric errors accompany coupled mesh/contact changes but do not by themselves decompose the full stress error. A fixed-body pad study fixes the body geometry/mass exactly; separate geometry/volume/contact convergence and all original physical acceptance requirements remain open.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
