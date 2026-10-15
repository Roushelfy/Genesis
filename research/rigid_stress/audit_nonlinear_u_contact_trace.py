"""Audit final nonlinear U contact traces against independently balanced saved fields."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np

from .contact_reference import IndependentContactSystem
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def density(force, area):
    magnitude = np.linalg.norm(force, axis=1)
    active = magnitude > 1e-8
    return {
        "active_samples_above_1em8_N": int(active.sum()),
        "active_lumped_area_m2": float(area[active].sum()),
        "unsigned_sample_force_N": float(magnitude.sum()),
        "maximum_sample_force_over_lumped_area_Pa": float((magnitude / area).max()),
        "sample_force_density_L2_N_per_m": float(np.sqrt(np.sum(magnitude**2 / area))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-audit", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, producers, comparisons = {}, {}, []

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    for audit in args.field_audit:
        closed_source(audit)
        assert int((audit / "returncode.txt").read_text()) == 0
        retain(audit / "results/summary.json")
        report = json.loads((audit / "results/summary.json").read_text())
        comparisons.append({"audit": str(audit), "axis": report["declared_spatial_axis"]})
        for path, digest in report["input_sha256"].items():
            assert retain(Path(path)) == digest
        for row in report["rows"]:
            assert (
                max(
                    row["independent_body_residual_N"],
                    row["independent_original_mapped_residual_N"],
                    row["maximum_recorded_original_residual_N"],
                )
                <= 1e-9
            )
            run = Path(row["run"])
            if run in producers:
                assert producers[run] == row
            producers[run] = row

    rows = []
    for index, (run, qualified) in enumerate(producers.items()):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        directory = run / "solution"
        config = json.loads((directory / "configuration.json").read_text())
        assert config["geometry"] == "u" and config["updated_friction"]
        assert config["body_material"] == config["pad_material"] == "neo_hookean"
        with np.load(directory / "fields.npz") as saved:
            rest, faces, displacement, saved_force = [
                saved[key].copy() for key in ["rest_positions", "surface_triangles", "displacement", "contact_force_N"]
            ]
        references = []
        for name, path, young, poisson, rho in [
            ("body", Path(config["mesh"]), 1e10, 0.3, 2000),
            ("left_shoe", directory / "left_shoe_mesh.npz", config["pad_young"], 0, 1000),
            ("right_shoe", directory / "right_shoe_mesh.npz", config["pad_young"], 0, 1000),
        ]:
            retain(path)
            with np.load(path) as saved:
                references.append(
                    (name, SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, rho))
                )
        system = IndependentContactSystem(references, config["dhat"], 1e6, config["friction"], 1e-8)
        np.testing.assert_array_equal(system.rest, rest)
        np.testing.assert_array_equal(system.faces, faces)
        normal, current = system.normal_collisions(displacement)
        assert not ipctk.has_intersections(system.mesh, current)
        ids = system.mesh.to_full_vertex_id()
        groups = system.groups[ids]
        total = system.mesh.vertices(saved_force.reshape(-1, 3))
        normal_force = np.zeros_like(current)
        by_source = {source: np.zeros_like(current) for source in [1, 2]}
        support = {source: np.zeros(len(current), dtype=bool) for source in [1, 2]}
        kinds = {source: {} for source in [1, 2]}
        for stencil_index in range(len(normal)):
            collision = normal[stencil_index]
            nodes = np.asarray(collision.vertex_ids(system.mesh.edges, system.mesh.faces))
            nodes = nodes[nodes >= 0]
            stencil_groups = set(groups[nodes])
            assert stencil_groups in ({0, 1}, {0, 2})
            source = max(stencil_groups)
            force = -system.barrier.gradient(collision, current[nodes].reshape(-1)).reshape(-1, 3)
            np.add.at(normal_force, nodes, force)
            np.add.at(by_source[source], nodes, force)
            support[source][nodes] = True
            kind = type(collision).__name__
            kinds[source][kind] = kinds[source].get(kind, 0) + 1
            del collision
        full_normal = -system.mesh.to_full_dof(system.barrier.gradient(normal, system.mesh, current))
        reconstructed = system.mesh.to_full_dof(normal_force.reshape(-1))
        np.testing.assert_allclose(reconstructed, full_normal, rtol=1e-12, atol=1e-12)
        shear = total - normal_force
        body = groups == 0
        assert not np.any(body & support[1] & support[2]), "Ambiguous body-node attribution between shoes"
        assigned = support[1] | support[2]
        assert np.max(np.linalg.norm(shear[~assigned], axis=1), initial=0) <= 1e-12
        area = np.zeros(len(current))
        triangles = current[system.mesh.faces]
        triangle_area = (
            np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
        )
        for corner in range(3):
            np.add.at(area, system.mesh.faces[:, corner], triangle_area / 3)
        assert np.all(area > 0)
        body_reference = references[0][1]
        # Use the declared reference COM as the same moment origin on both sides.
        centre = body_reference.com
        source_rows = []
        for source in [1, 2]:
            selected = body & support[source]
            counterpart = groups == source
            n, s = by_source[source], np.where(support[source][:, None], shear, 0)
            source_rows.append(
                {
                    "source": source,
                    "name": references[source][0],
                    "normal_stencil_types": kinds[source],
                    "normal_force_on_body_N": n[body].sum(axis=0).tolist(),
                    "saved_total_minus_normal_shear_on_body_N": s[body].sum(axis=0).tolist(),
                    "total_force_on_body_N": (n[body] + s[body]).sum(axis=0).tolist(),
                    "normal_moment_on_body_about_reference_com_Nm": np.cross(current[body] - centre, n[body])
                    .sum(axis=0)
                    .tolist(),
                    "total_moment_on_body_about_reference_com_Nm": np.cross(current[body] - centre, n[body] + s[body])
                    .sum(axis=0)
                    .tolist(),
                    "normal_action_reaction_force_error_N": float(np.linalg.norm(n.sum(axis=0))),
                    "normal_action_reaction_moment_error_Nm": float(
                        np.linalg.norm(np.cross(current - centre, n).sum(axis=0))
                    ),
                    "saved_total_action_reaction_force_error_N": float(np.linalg.norm((n + s).sum(axis=0))),
                    "saved_total_action_reaction_moment_Nm": np.cross(current - centre, n + s).sum(axis=0).tolist(),
                    "body_normal_density": density(n[body], area[body]),
                    "body_shear_density": density(s[body], area[body]),
                    "shoe_normal_density": density(n[counterpart], area[counterpart]),
                    "body_support_samples": int(selected.sum()),
                }
            )
        rows.append(
            {
                "run": str(run),
                "body_tetrahedra": qualified["body_tetrahedra"],
                "pad_resolution": qualified["pad_resolution"],
                "dhat_m": config["dhat"],
                "friction_mollifier_m": 1e-8,
                "normal_stencils": len(normal),
                "normal_gradient_reconstruction_max_difference_N": float(np.max(abs(reconstructed - full_normal))),
                "moment_origin_reference_com_m": centre.tolist(),
                "body_total_force_N": total[body].sum(axis=0).tolist(),
                "sources": source_rows,
                "raw_global_peak_lower_Pa": qualified["raw_global_peak_lower_Pa"],
                "raw_global_peak_upper_Pa": qualified["raw_global_peak_upper_Pa"],
            }
        )
        np.savez_compressed(
            args.output / f"case{index:02d}.npz",
            current_surface_positions_m=current,
            full_vertex_ids=ids,
            groups=groups,
            surface_triangles=system.mesh.faces,
            current_vertex_lumped_area_m2=area,
            normal_surface_force_N=normal_force,
            saved_total_minus_normal_shear_surface_force_N=shear,
            saved_total_surface_force_N=total,
            source1_normal_surface_force_N=by_source[1],
            source2_normal_surface_force_N=by_source[2],
            source1_stencil_support=support[1],
            source2_stencil_support=support[2],
        )
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "comparisons": comparisons,
        "rows": rows,
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "Final normal IPC stencils and forces are independently rebuilt; shear is the unchanged saved total contact force minus that normal force, not an independent friction-history audit. The original final/history force budgets are inherited only from the specified closed field audits. Shoe groups are physical source identities, not independently identified continuum islands. Disjoint body stencil supports permit exact source attribution; ambiguous nodes are rejected. Sample force divided by current incident-triangle lumped area is a discrete density, not continuum pressure/contact area. The1e-8N threshold affects reported active counts/areas only, never forces, norms, raw maxima or acceptance. No pressure convergence or goal completion follows from these diagnostics.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "normal_stencils": [r["normal_stencils"] for r in rows]}), flush=True)


if __name__ == "__main__":
    main()
