"""Reconstruct all covered-contact source forces and discrete surface components against complete independent audits."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .summarize_fine_references import closed_source


def wrench(positions, force, selected):
    return {
        "force_N": np.asarray(force[selected].sum(axis=0), dtype=float).tolist(),
        "moment_about_reference_origin_Nm": np.asarray(
            np.cross(positions[selected], force[selected]).sum(axis=0), dtype=float
        ).tolist(),
    }


def density(force, area, selected):
    magnitude = np.linalg.norm(force[selected], axis=1)
    weights = area[selected]
    active = magnitude > 1e-8
    return {
        "active_samples_above_1em8_N": int(active.sum()),
        "active_lumped_area_m2": float(weights[active].sum()),
        "unsigned_sample_force_N": float(magnitude.sum()),
        "maximum_sample_force_over_lumped_area_Pa": float((magnitude / weights).max(initial=0)),
        "sample_force_density_L2_N_per_m": float(np.sqrt(np.sum(magnitude**2 / weights))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--qualification-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) == len(args.audit)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, rows = {}, []

    def retain(path):
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == expected
        inputs[path] = expected
        return expected

    qualification = None
    if args.qualification_run:
        closed_source(args.qualification_run)
        assert int((args.qualification_run / "returncode.txt").read_text()) == 0
        qualified_path = args.qualification_run / "results/summary.json"
        qualification = json.loads(qualified_path.read_text())
        assert qualification["selector_mesh_reconstruction_qualified"] and len(qualification["rows"]) == 5
        assert all(row["body_tetrahedra"] <= 2000 for row in qualification["rows"])
        qualified_source = args.qualification_run / "source/research/rigid_stress/audit_covered_contact_sources.py"
        assert (
            hashlib.sha256(qualified_source.read_bytes()).digest()
            == hashlib.sha256(Path(__file__).read_bytes()).digest()
        )
        retain(qualified_path)
    for i_case, (run, audit_run) in enumerate(zip(args.run, args.audit, strict=True)):
        for closed in [run, audit_run]:
            closed_source(closed)
            assert int((closed / "returncode.txt").read_text()) == 0
        config_path = run / "results/configuration.json"
        audit_path = audit_run / "results/summary.json"
        config, audit = [json.loads(path.read_text()) for path in [config_path, audit_path]]
        assert Path(audit["run"]).resolve() == run.resolve()
        assert config["mode"] == "cad" and config["contact_refinement"] == 0
        assert config["steps"] == audit["states_checked"] == len(audit["rows"]) == 5
        if max(row["body_tetrahedra"] for row in audit["rows"]) > 2000:
            assert qualification is not None
        assert all(
            max(row["complete_body_dynamic_residual_N"], row["complete_mapped_dynamic_residual_N"]) <= 1e-9
            for row in audit["rows"]
        )
        closure_path = Path(config["closure_run"]) / "results/configuration.json"
        closure = json.loads(closure_path.read_text())
        capture_path = Path(closure["capture"]) / "results/summary.json"
        closed_source(Path(closure["capture"]))
        capture = json.loads(capture_path.read_text())
        frame = [row for row in capture["captures"] if row["mode"] == "stationary_lift_hold"][-1]
        friction = {contact["friction"] for contact in frame["contacts"]}
        assert len(friction) == 1
        friction = friction.pop()
        geometry_path = run / "results/contact_geometry.npz"
        map_path = run / "results/contact_node_map.npz"
        with np.load(geometry_path) as saved:
            rest, faces, groups = [saved[key].copy() for key in ["rest_positions", "faces", "groups"]]
        mapping = sparse.load_npz(map_path).tocsr()
        assert np.all(np.diff(mapping.indptr) == 1) and np.all(mapping.data == 1)
        source_nodes = mapping.indices.copy()
        assert len(np.unique(source_nodes)) == len(source_nodes)
        edges = np.unique(np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
        mesh = ipctk.CollisionMesh(rest, edges, faces, mapping.tocsc())
        mesh.can_collide = ipctk.make_vertex_patches_filter(groups)
        barrier = ipctk.BarrierPotential(closure["dhat"], 1e6, use_physical_barrier=True)
        friction_potential = ipctk.FrictionPotential(config["dt"] * 4e-6)
        previous_path = run / "results/closure_state.npz"
        with np.load(previous_path) as saved:
            previous = np.asarray(saved["displacement"], dtype=np.longdouble)
        for path in [config_path, audit_path, closure_path, capture_path, geometry_path, map_path, previous_path]:
            retain(path)
        for qualified in audit["rows"]:
            step = qualified["step"]
            state_path = run / f"results/step{step:04d}.npz"
            field_path = audit_run / f"results/step{step:04d}.npz"
            with np.load(state_path) as saved:
                displacement = np.asarray(saved["displacement"], dtype=np.longdouble)
                saved_contact = saved["contact_force_N"].copy()
            # Match the original add/subtract and float64 collision-query arithmetic exactly.
            increment = displacement.reshape(-1, 3).copy()
            increment[source_nodes] = (np.asarray(rest, dtype=np.longdouble) + increment[source_nodes]) - rest
            current = mesh.displace_vertices(np.asarray(increment, dtype=float))
            positions = np.asarray(rest, dtype=np.longdouble) + displacement.reshape(-1, 3)[source_nodes]
            normal = ipctk.NormalCollisions()
            normal.use_area_weighting = True
            normal.collision_set_type = ipctk.NormalCollisions.IMPROVED_MAX_APPROX
            normal.build(mesh, current, closure["dhat"])
            assert len(normal) == qualified["normal_stencils"]
            normal_by_source = np.zeros((2, len(rest), 3))
            shear_by_source = np.zeros_like(normal_by_source)
            magnitudes, stencils = {}, []
            for i_stencil in range(len(normal)):
                collision = normal[i_stencil]
                nodes = np.asarray(collision.vertex_ids(mesh.edges, mesh.faces))
                nodes = nodes[nodes >= 0]
                group_set = set(groups[nodes])
                assert group_set in ({0, 1}, {0, 2})
                source = int(max(group_set))
                centered = np.asarray(positions[nodes] - positions[nodes[0]], dtype=float).reshape(-1)
                force = -barrier.gradient(collision, centered).reshape(-1, 3)
                np.add.at(normal_by_source[source - 1], nodes, force)
                magnitudes[tuple(nodes)] = barrier.force_magnitude(collision.compute_distance(centered), collision.dmin)
                stencils.append(
                    {
                        "source": source,
                        "type": type(collision).__name__,
                        "surface_nodes": nodes.tolist(),
                        "stored_weight": float(collision.weight),
                        "normal_stencil_force_N": force.tolist(),
                    }
                )
                del collision
            tangential = ipctk.TangentialCollisions()
            tangential.build(mesh, current, normal, barrier, friction)
            tangent_records = []
            delta = np.asarray(mapping @ (displacement - previous).reshape(-1, 3), dtype=float)
            for i_stencil in range(len(tangential)):
                collision = tangential[i_stencil]
                nodes = np.asarray(collision.vertex_ids(mesh.edges, mesh.faces))
                nodes = nodes[nodes >= 0]
                source = int(max(groups[nodes]))
                collision.normal_force_magnitude = magnitudes[tuple(nodes)]
                force = -friction_potential.gradient(collision, collision.dof(delta, mesh.edges, mesh.faces)).reshape(
                    -1, 3
                )
                np.add.at(shear_by_source[source - 1], nodes, force)
                tangent_records.append(
                    {
                        "source": source,
                        "type": type(collision).__name__,
                        "surface_nodes": nodes.tolist(),
                        "shear_stencil_force_N": force.tolist(),
                        "stored_weight": float(collision.weight),
                        "unmollified_normal_force_magnitude": float(collision.normal_force_magnitude),
                    }
                )
                del collision
            normal_force, shear_force = normal_by_source.sum(axis=0), shear_by_source.sum(axis=0)
            with np.load(field_path) as saved:
                np.testing.assert_array_equal(positions, saved["surface_positions_m"])
                expected_normal, expected_shear, area = [
                    saved[key].copy()
                    for key in ["normal_surface_force_N", "shear_surface_force_N", "surface_vertex_lumped_area_m2"]
                ]
            assert len(area) == len(rest) and np.all(np.isfinite(area)) and np.all(area > 0)
            np.testing.assert_allclose(normal_force, expected_normal, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(shear_force, expected_shear, rtol=1e-12, atol=1e-12)
            pullback = mesh.to_full_dof((normal_force + shear_force).reshape(-1))
            np.testing.assert_allclose(pullback, saved_contact, rtol=1e-12, atol=1e-12)
            source_rows, component_labels = [], []
            for source in [1, 2]:
                n, s = normal_by_source[source - 1], shear_by_source[source - 1]
                total = n + s
                body, counterpart = groups == 0, groups == source
                active = body & (np.linalg.norm(n, axis=1) > 1e-8)
                selected = np.flatnonzero(active)
                active_edges = edges[active[edges].all(axis=1)]
                graph = sparse.coo_matrix(
                    (
                        np.ones(len(active_edges) * 2),
                        (np.r_[active_edges[:, 0], active_edges[:, 1]], np.r_[active_edges[:, 1], active_edges[:, 0]]),
                    ),
                    shape=(len(rest),) * 2,
                ).tocsr()[selected][:, selected]
                count, labels = connected_components(graph, directed=False)
                full_labels = np.full(len(rest), -1, dtype=np.int32)
                full_labels[selected] = labels
                components = []
                for i_component in range(count):
                    component = selected[labels == i_component]
                    components.append(
                        {
                            "component": i_component,
                            "nodes": len(component),
                            "lumped_area_m2": float(area[component].sum()),
                            "wrench": wrench(positions, total, component),
                        }
                    )
                source_rows.append(
                    {
                        "source": source,
                        "captured_finger_link": frame["fingers"][source - 1]["link"],
                        "normal_body_wrench": wrench(positions, n, body),
                        "shear_body_wrench": wrench(positions, s, body),
                        "total_body_wrench": wrench(positions, total, body),
                        "counterbody_wrench": wrench(positions, total, counterpart),
                        "normal_body_density": density(n, area, body),
                        "shear_body_density": density(s, area, body),
                        "normal_finger_density": density(n, area, counterpart),
                        "shear_finger_density": density(s, area, counterpart),
                        "discrete_active_surface_components": components,
                        "below_threshold_body_tail_wrench": wrench(positions, total, body & ~active),
                        "pair_force_balance_N": float(np.linalg.norm(total.sum(axis=0))),
                        "pair_normal_moment_about_reference_origin_Nm": np.asarray(
                            np.cross(positions, n).sum(axis=0), dtype=float
                        ).tolist(),
                        "pair_total_moment_about_reference_origin_Nm": np.asarray(
                            np.cross(positions, total).sum(axis=0), dtype=float
                        ).tolist(),
                    }
                )
                component_labels.append(full_labels)
                component_force = (
                    np.asarray([component["wrench"]["force_N"] for component in components]).reshape(-1, 3).sum(axis=0)
                )
                component_moment = (
                    np.asarray([component["wrench"]["moment_about_reference_origin_Nm"] for component in components])
                    .reshape(-1, 3)
                    .sum(axis=0)
                )
                tail = source_rows[-1]["below_threshold_body_tail_wrench"]
                np.testing.assert_allclose(
                    component_force + tail["force_N"],
                    source_rows[-1]["total_body_wrench"]["force_N"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    component_moment + tail["moment_about_reference_origin_Nm"],
                    source_rows[-1]["total_body_wrench"]["moment_about_reference_origin_Nm"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                assert source_rows[-1]["pair_force_balance_N"] <= 1e-12
            row = {
                "run": str(run),
                "step": step,
                "time_s": qualified["time_s"],
                "body_tetrahedra": qualified["body_tetrahedra"],
                "normal_stencils": len(normal),
                "normal_reconstruction_max_error_N": float(abs(normal_force - expected_normal).max()),
                "shear_reconstruction_max_error_N": float(abs(shear_force - expected_shear).max()),
                "full_force_pullback_max_error_N": float(abs(pullback - saved_contact).max()),
                "sources": source_rows,
                "stencils": stencils,
                "tangential_stencils": tangent_records,
                "raw_global_peak_lower_Pa": qualified["raw_global_peak_lower_Pa"],
                "raw_global_peak_upper_Pa": qualified["raw_global_peak_upper_Pa"],
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"case{i_case:02d}-step{step:04d}.npz",
                rest_surface_positions_m=rest,
                current_surface_positions_m=positions,
                surface_triangles=faces,
                groups=groups,
                current_vertex_lumped_area_m2=area,
                normal_force_by_source_N=normal_by_source,
                shear_force_by_source_N=shear_by_source,
                discrete_component_labels_by_source=np.asarray(component_labels),
            )
            retain(state_path)
            retain(field_path)
            previous = displacement
            print(
                json.dumps(
                    {
                        "run": str(run),
                        "step": step,
                        "normal_stencils": len(normal),
                        "source_component_counts": [
                            len(source["discrete_active_surface_components"]) for source in source_rows
                        ],
                    }
                ),
                flush=True,
            )
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    result = {
        "scope": __doc__,
        "rows": rows,
        "selector_mesh_reconstruction_qualified": True,
        "input_sha256": {str(path): expected for path, expected in inputs.items()},
        "new_physical_states_accepted": 0,
        "goal_completion": False,
        "limitations": "Original closed five-state force audits qualify each producer. Every current normal/friction stencil and full mapped force must match the independently audited arrays before source attribution. Selector maps permit exact standalone collision reconstruction without reassembling bulk elasticity. Every source force is retained; active surface components use actual triangle adjacency above1e-8N, with the complete below-threshold wrench explicit. These are discrete graph components and lumped force densities, not continuum contact islands, pressures or areas. Finite-gap friction moments remain measured outputs. No spatial/contact/geometry/regularization/model/feedback or warmed-cost acceptance follows.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
