"""Archive closed physical-reference fields, qualifications and diagnostics without accepting physical convergence."""

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path

from research.rigid_stress.scripts.publish_nonconvex_checkpoint import publish_bounded
from research.rigid_stress.scripts.verify_nonconvex_publication import digest, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert (
        plan["kind"]
        in [
            "fields",
            "sources",
            "native_followup",
            "fixed_distance",
            "bulk_inputs",
            "bulk_fields",
            "bulk_audit",
            "axis_setup",
            "matrix_setup",
            "matrix_fields",
            "matrix_summary",
            "peak_boundary",
            "path_model_regression",
            "native_finite_fields",
            "fixed_polyhedron_setup",
            "matrix_figures",
            "fixed_polyhedron_fields",
            "fixed_polyhedron_peak_cells",
            "native_matched_axis_fields",
            "native_matched_axes_summary",
            "local_finite_setup",
            "u_shared_anchor_diagnosis",
        ]
        and plan["goal_completion"] is False
    )
    p = args.runtime_root / "20261014-physical-validation"
    label = plan["label"]
    meta = p / f"{label}-publication"
    meta.mkdir(parents=True, exist_ok=False)
    runs = [p / name for name in plan["closed_jobs"]]
    extras, records, identities, input_records = set(), [], [], {}
    cache, parents, parent_sources = {}, [], {}
    for parent in plan["parent_manifests"]:
        manifest = args.destination_root / parent["label"] / "manifest.json"
        assert digest(manifest)[0] == parent["sha256"]
        proof = verify_manifest(manifest, True, cache)
        value = json.loads(manifest.read_text())
        stage = p / "publication-staging" / parent["label"]
        for row in value["files"]:
            original = Path(row["original"])
            if original.is_relative_to(stage):
                original = args.runtime_root / original.relative_to(stage)
            if original.exists() and digest(original) == (row["sha256"], row["bytes"]):
                parent_sources[str(original)] = row["sha256"]
        for row in value["summary"].get("sharded_original_files", []):
            assert digest(Path(row["original"])) == (row["sha256"], row["bytes"])
            parent_sources[row["original"]] = row["sha256"]
        parents.append({**parent, "manifest": str(manifest), "complete_archive_reverification": proof})

    def retain(path, expected=None):
        actual = digest(path)[0]
        assert expected is None or actual == expected, path
        assert str(path) not in input_records or input_records[str(path)]["sha256"] == actual
        covered = parent_sources.get(str(path)) == actual
        input_records[str(path)] = {"sha256": actual, "covered_by_reverified_parent": covered}
        if not covered and not any(path.is_relative_to(run) for run in runs):
            # All large fields belong to complete archived runs, where the
            # bounded publisher shards exact bytes. Extras may never bypass it.
            assert path.stat().st_size < 50_000_000, path
            if not path.is_relative_to(args.runtime_root):
                copy = meta / "external-inputs" / (actual + path.suffix)
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copy)
                assert digest(copy)[0] == actual
                input_records[str(path)]["archived_external_copy"] = str(copy)
                path = copy
            extras.add(path)

    query = [
        "sacct",
        "-n",
        "-P",
        "-X",
        "-j",
        ",".join(str(row["job"]).split("_")[0] for row in plan["closed_jobs"].values()),
    ]
    account = subprocess.check_output(
        [*query, "--format=JobID%40,State%40,ExitCode,Elapsed,Start,End,NodeList"], text=True
    )
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(
        subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    )
    states = {line.split("|")[0].strip(): line.split("|")[1:3] for line in account.splitlines() if line}
    summaries = {}
    for name, expected in plan["closed_jobs"].items():
        run = p / name
        code = expected["returncode"]
        state = "COMPLETED" if code == 0 else "FAILED"
        actual_state, exit_code = states[expected["job"]]
        assert actual_state.strip() == state and exit_code.strip() == f"{code}:0"
        assert int((run / "returncode.txt").read_text()) == code
        for kind in ["source", "input-source"]:
            before = run / f"{kind}.sha256"
            if not before.exists():
                continue
            assert before.read_bytes() == (run / f"{kind}-after.sha256").read_bytes()
            for line in before.read_text().splitlines():
                sha, relative = line.split("  ", 1)
                path = Path(relative) if Path(relative).is_absolute() else run / kind / relative
                assert digest(path)[0] == sha
                identities.append({"run": name, "kind": kind, "path": str(path), "sha256": sha})
        records.append({"run": name, "job": expected["job"], "scheduler_state": state, "actual_wrapper_code": code})
        extras.update(p.glob(f"*{expected['job']}.log"))
        summary_path = run / "results/summary.json"
        if summary_path.exists():
            value = json.loads(summary_path.read_text())
            summaries[name] = value
            for key in ["input_sha256", "inputs"]:
                for original, sha in value.get(key, {}).items():
                    retain(Path(original), sha)

    if plan["kind"] == "fields":
        fields = summaries["egg-covered-cad-three-level-fields-v1"]
        linear = summaries["egg-covered-cad-three-level-linearization-v1"]
        assert [row["tetrahedra"] for row in fields["rows"]] == [31668, 64716, 171228]
        assert fields["common_physical_probes"] == 158340 and len(fields["comparisons"]) == 2
        assert [row["tetrahedra"] for row in linear["rows"]] == [31668, 64716, 171228]
        assert all(row["linear_complete_residual_N"] <= 1e-9 for row in linear["rows"])
        audit = summaries["egg-covered-cad-level3-schur-audit-v1"]
        coverage = summaries["egg-covered-cad-level3-schur-coverage-v1"]
        assert audit["states_checked"] == len(audit["rows"]) == len(coverage["rows"]) == 5
        assert all(
            max(row["complete_body_dynamic_residual_N"], row["complete_mapped_dynamic_residual_N"]) <= 1e-9
            for row in audit["rows"]
        )
        assert all(row["active_samples_outside_marking_caps"] == 0 for row in coverage["rows"])
        producer = p / "egg-covered-cad-level3-schur-contact-v1"
        provenance = json.loads((producer / "results/precision_provenance.json").read_text())
        original = Path(provenance["source_run"])
        retain(original / "results/configuration.json", provenance["original_configuration_sha256"])
        for item in provenance["initial_state_and_meshes"]:
            retain(Path(item["source"]), item["sha256"])
            assert digest(producer / "results" / Path(item["source"]).name)[0] == item["sha256"]
        for item in provenance["qualifications"]:
            retain(Path(item["run"]) / "results/summary.json", item["summary_sha256"])
        assert len(summaries[producer.name]["steps"]) == 5
        scientific_checks = {
            "full_original_fine_force_and_coverage_states": 5,
            "all_original_common_probes": 158340,
            "three_covered_mesh_fields_and_independent_linearizations": True,
        }
    elif plan["kind"] == "sources":
        source = summaries["covered-cad-three-level-sources-v1"]
        assert source["selector_mesh_reconstruction_qualified"] and len(source["rows"]) == 15
        assert all(
            max(
                row[key]
                for key in [
                    "normal_reconstruction_max_error_N",
                    "shear_reconstruction_max_error_N",
                    "full_force_pullback_max_error_N",
                ]
            )
            <= 1e-12
            for row in source["rows"]
        )
        local = summaries["covered-cad-tensor-change-localization-v1"]
        assert local["complete_probe_count"] == 158340 and local["all_original_probes_retained"]
        assert local["original_complete_weighted_tensor_changes_reproduced"]
        assert local["hydrostatic_deviatoric_and_spatial_partitions_recompose_complete_squared_change"]
        assert len(local["rows"]) == 2
        native = [
            value for name, value in summaries.items() if name.startswith("egg-covered-cad-native-linearization-level")
        ]
        assert len(native) >= 2 and all(
            value["all_existing_gates_passed"] and len(value["rows"]) == 2 for value in native
        )
        for value in native:
            assert all(
                row["all_existing_gates_passed"] and row["independent_complete_budget_fraction"] <= 1
                for row in value["rows"]
            )
        figure = p / plan["accepted_figure_run"] / "results/covered_cad_tail.svg"
        root = element_tree.parse(figure).getroot()
        text = root.findall(".//{http://www.w3.org/2000/svg}text")
        images = root.findall(".//{http://www.w3.org/2000/svg}image")
        assert len(text) > 20 and not images
        export_qa = json.loads((p / plan["export_qa"]).read_text())
        assert export_qa["passed"] and export_qa["editable_svg_sha256"] == digest(figure)[0]
        scientific_checks = {
            "complete_original_dynamic_source_states": 15,
            "all_probe_partitions_recompose_complete_tensor_change": True,
            "actual_native_meshes_two_load_modes_each": len(native),
            "figure_editable_text_nodes": len(text),
            "figure_embedded_raster_images": 0,
        }
    elif plan["kind"] == "native_followup":
        native = summaries["egg-covered-cad-native-linearization-level3-v1"]
        assert native["all_existing_gates_passed"] and len(native["rows"]) == 2
        assert {row["mode"] for row in native["rows"]} == {"full_inertia", "rigid_relief"}
        assert all(
            row["tetrahedra"] == 171228
            and row["all_existing_gates_passed"]
            and row["independent_complete_budget_fraction"] <= 1
            and row["cached_corner_stress_max_existing_budget_fraction"] <= 1
            for row in native["rows"]
        )
        meshes = summaries["egg-thickness-local-bulk-four-level-meshes-v1"]
        assert [row["tetrahedra"] for row in meshes["rows"]] == [502230, 516287, 554574, 653088]
        assert all(
            row["parent_partition"]["all_child_tetrahedra_contained_in_declared_parent"]
            and row["parent_partition"]["each_parent_volume_preserved"]
            for row in meshes["rows"][1:]
        )
        proof = json.loads((p / "meta/frozen-traction-fourth-local-bulk-mesh-qualification-v1.json").read_text())
        assert proof["source_closure_matched"] and len(proof["original_three_meshes_byte_identical"]) == 3
        for row in proof["original_three_meshes_byte_identical"]:
            retain(Path(row["original"]), row["sha256"])
            retain(Path(row["new"]), row["sha256"])
        scientific_checks = {
            "actual_finest_native_original_load_modes_passed": 2,
            "fourth_local_mesh_preserves_original_three_meshes_exactly": True,
            "fourth_mesh_tetrahedra": 653088,
            "fourth_mesh_equilibrium_fields_included": False,
        }
    elif plan["kind"] == "fixed_distance":
        fields = summaries["covered-cad-fixed-distance-fields-v1"]
        assert fields["all_geometry_and_contact_map_arrays_exactly_equal"]
        assert fields["every_original_five_state_force_audit_and_raw_bounds_matched"]
        assert len(fields["rows"]) == 15 and len(fields["comparisons"]) == 10
        assert {row["tetrahedra"] for row in fields["rows"]} == {31668}
        assert sorted({row["barrier_distance_m"] for row in fields["rows"]}, reverse=True) == [2.5e-6, 1.25e-6, 6.25e-7]
        source = summaries["covered-cad-fixed-distance-sources-v1"]
        assert source["selector_mesh_reconstruction_qualified"] and len(source["rows"]) == 15
        assert all(
            max(
                row[key]
                for key in [
                    "normal_reconstruction_max_error_N",
                    "shear_reconstruction_max_error_N",
                    "full_force_pullback_max_error_N",
                ]
            )
            <= 1e-12
            for row in source["rows"]
        )
        diagnostic = summaries["covered-cad-fixed-distance-source-diagnostics-v1"]
        assert len(diagnostic["rows"]) == 15 and len(diagnostic["comparisons"]) == 10
        assert all(
            row["first_run"] != row["second_run"] and row["first_barrier_distance_m"] > row["second_barrier_distance_m"]
            for row in diagnostic["comparisons"]
        )
        probes = summaries["covered-cad-contact-depth-probes-v1"]
        assert probes["all_shared_points_strictly_inside_every_mesh_and_evaluator_qualified"]
        assert (
            len(probes["rows"]) == 3
            and len(probes["original158340_probe_tensor_changes_reproduced_without_replacement"]) == 2
        )
        assert (
            probes["supplemental_probes"] == 576
            and probes["supplemental_probes_within25um_of_baseline_normal_samples"] > 0
        )
        matched = summaries["covered-cad-fixed-distance-depth-probes-v1"]
        assert matched["qualified_baseline_final_probe_tensor_reproduced"]
        assert matched["exact_original576_probe_coordinates_and_depths_retained"]
        assert len(matched["rows"]) == 15 and len(matched["adjacent_depth_changes"]) == 180
        figure = p / plan["accepted_figure_run"] / "results/contact_depth_distance.svg"
        root = element_tree.parse(figure).getroot()
        assert len(root.findall(".//{http://www.w3.org/2000/svg}text")) > 20
        assert not root.findall(".//{http://www.w3.org/2000/svg}image")
        export_qa = json.loads((p / plan["export_qa"]).read_text())
        assert export_qa["passed"] and export_qa["editable_svg_sha256"] == digest(figure)[0]
        scientific_checks = {
            "fixed_mesh_distance_full_dynamic_fields": 15,
            "original_dynamic_source_states": 15,
            "original_common_probes_retained": 158340,
            "fixed_additional_material_points": 576,
            "continuum_contact_convergence_accepted": False,
        }
    elif plan["kind"] == "bulk_inputs":
        qualification = summaries["egg-frozen-traction-load-qualification-v2"]
        assert qualification["qualify_only"] and qualification["all_original_five_cases_qualified"]
        assert qualification["original_traction_samples_unchanged"] and not qualification["quadrature_points_relocated"]
        assert qualification["tetrahedra"] == 502230 and len(qualification["rows"]) == 5
        assert all(
            row["all_original_corner_vm_fields_preserved"]
            and row["all_original_quadrature_fields_qualified"]
            and row["original_complete_budget_fraction"] <= 1
            for row in qualification["rows"]
        )
        assert plan["closed_jobs"]["egg-frozen-traction-load-qualification-v1"]["returncode"] == 1
        scientific_checks = {
            "unchanged_original_quadrature_load_cases_qualified": 5,
            "original_boundary_finder_failure_preserved": True,
            "accepted_finer_equilibrium_fields": 0,
        }
    elif plan["kind"] == "bulk_fields":
        producers = [summaries[f"egg-frozen-traction-local-bulk-level{level}-v2"] for level in [1, 2]]
        assert [value["tetrahedra"] for value in producers] == [516287, 554574]
        assert all(
            not value["qualify_only"]
            and value["original_traction_samples_unchanged"]
            and not value["quadrature_points_relocated"]
            and len(value["rows"]) == 5
            for value in producers
        )
        assert all(row["original_complete_budget_fraction"] <= 1 for value in producers for row in value["rows"])
        assert any(parent["label"] == "frozen-traction-inputs-v1" for parent in parents)
        scientific_checks = {
            "complete_original_force_budget_producer_fields": 10,
            "complete_large_arrays_retained": True,
            "independent_audit_requires_separate_closed_packet": True,
            "physical_tail_accepted": False,
        }
    elif plan["kind"] == "axis_setup":
        geometry = summaries["egg-covered-fixed-trace-thickness-geometry-v1"]
        trace = summaries["egg-covered-fixed-thickness-actual-trace-v1"]
        assert [row["through_thickness_layers"] for row in geometry["rows"]] == [2, 4, 8]
        assert [row["tetrahedra"] for row in geometry["rows"]] == [31668, 63336, 126672]
        assert geometry["new_equilibrium_states_solved"] == trace["new_equilibrium_states_solved"] == 0
        assert all(
            row["all_original_vertices_byte_preserved"]
            and row["both_original_boundary_triangle_sets_identical"]
            and row["canonical_boundary_P2_coordinates_byte_identical"]
            and row["original_initializer_all_anchors_retained"]
            for row in geometry["rows"]
        )
        assert len(trace["rows"]) == 3 and all(
            row["surface_vertices"] == 21294
            and row["surface_triangles"] == 42532
            and row["triangle_topology_identical_under_bijective_vertex_map"]
            and row["all_vertex_lumped_areas_checked"]
            and row["IPC_zero_and_affine_source_displacements_checked"]
            and not row["IPC_initial_intersections"]
            for row in trace["rows"]
        )
        equivalence = summaries["egg-cad1920-record-initializer-fields-v1"]
        assert equivalence["all_original_arrays_and_complete_fields_qualified"]
        assert equivalence["states_checked"] == len(equivalence["rows"]) == 5
        assert equivalence["initializer_comparison"]["byte_identical"]
        assert not equivalence["initializer_comparison"]["independently_resolved_initializer_allowed"]
        assert all(
            row["all_corner_tensor_relative_change"]
            == row["all_quadrature_tensor_relative_change"]
            == row["raw_global_peak_relative_change"]
            == 0
            and all(
                change["maximum_absolute_difference"] == change["relative_L2_difference"] == 0
                for change in row["all_saved_array_changes"].values()
            )
            for row in equivalence["rows"]
        )
        history = summaries["egg-cad1920-record-initializer-all64-audit-v3"]
        assert history["all_original64_initializer_states_checked"]
        assert history["states_checked"] == len(history["rows"]) == 64
        assert all(row["independent_complete_held_residual_N"] <= 1e-9 for row in history["rows"])
        for version in [1, 2]:
            assert plan["closed_jobs"][f"egg-cad1920-record-initializer-all64-audit-v{version}"]["returncode"] == 1
        assert (
            len(
                list((p / "egg-cad1920-record-initializer-all64-audit-v3/results").glob("step[0-9][0-9][0-9][0-9].npz"))
            )
            == 64
        )
        loads = summaries["egg-continuous-traction-load-axis-v1"]
        assert loads["new_equilibrium_states_solved"] == 0 and len(loads["fixed_mesh_rows"]) == 15
        assert len(loads["quadrature_load_changes"]) == 10
        original = loads["original_five_case_continuous_law_and_full_nodal_loads_qualified"]
        assert len(original) == 5 and all(row["full_original_nodal_load_maximum_error_N"] == 0 for row in original)
        scientific_checks = {
            "fixed_boundary_thickness_meshes_and_actual_IPC_traces": 3,
            "recording_extension_original_dynamic_states_byte_preserved": 5,
            "original_initializer_states_independently_reconstructed": 64,
            "both_original_audit_failures_and_partial_arrays_preserved": True,
            "fixed_mesh_continuous_traction_integrations": 15,
            "new_fine_thickness_or_quadrature_stress_fields_accepted": 0,
            "physical_convergence_accepted": False,
        }
    elif plan["kind"] == "matrix_figures":
        figures = summaries["extended-eight-level-reference-figures-v3"]
        assert figures["full_original_rows_retained"] == 240
        assert figures["worst_adjacent_case_observable_rows"] == 210 and len(figures["figures"]) == 2
        qa_path = p / "meta/extended-eight-level-reference-export-qa-v3.json"
        retain(qa_path)
        qa = json.loads(qa_path.read_text())
        assert qa["passed"] and len(qa["figures"]) == 2
        assert all(
            row["exports_after_exact_visual_source_qa"] and not row["text_qa_issues"] for row in figures["figures"]
        )
        output = p / "extended-eight-level-reference-figures-v3/results"
        for row in qa["figures"]:
            stem = row["stem"]
            assert digest(output / f"{stem}.svg")[0] == row["accepted_editable_svg_sha256"]
            assert digest(output / f"{stem}_preview.png")[0] == row["accepted_preview_sha256"]
            assert digest(output / f"{stem}.pdf")[0] == row["PDF_sha256"]
            assert digest(output / f"{stem}.png")[0] == row["PNG_sha256"]
            assert row["PNG_pixels"] == [7204, 6692] and row["PDF_raster_images"] == 0
            assert row["PDF_actual_decoded_TrueType_streams"] and row["ToUnicode_present"]
            tree = element_tree.fromstring((output / f"{stem}.svg").read_bytes())
            assert sum(node.tag.endswith("}text") for node in tree.iter()) > 50
            assert not any(node.tag.endswith("}image") for node in tree.iter())
        assert plan["closed_jobs"]["extended-eight-level-reference-figures-v1"]["returncode"] == 1
        scientific_checks = {
            "complete_original_raw_geometry_load_level_rows_retained": 240,
            "explicit_worst_case_adjacent_observable_rows": 210,
            "accepted_editable_and_colour_monochrome_previews": 2,
            "matched_PDF_fonts_and_full1000dpi_rasters_after_exact_source_acceptance": True,
            "initial_actual_text_QA_failure_retained": True,
            "physical_convergence_accepted": False,
        }
    elif plan["kind"] == "fixed_polyhedron_setup":
        meshes, qualified = [], []
        for shape in ["ellipsoid", "capsule"]:
            run = p / f"fixed-polyhedron-{shape}-three-meshes-v1"
            rows = json.loads((run / "meshes/summary.json").read_text())["cases"]
            assert len(rows) == 3
            qualifier = summaries[f"fixed-polyhedron-{shape}-original-qualification-v1"]
            assert qualifier["qualify_only"] and qualifier["new_equilibrium_states_solved"] == 0
            assert qualifier["all_original_five_loads_full_fields_and_probes_qualified"]
            assert len(qualifier["rows"]) == 5
            assert all(
                row["original_full_nodal_force_maximum_difference_N"] == 0
                and row["original_budget_fraction"] <= 1
                and row["all_original_probe_coordinates_and_validity_retained"]
                for row in qualifier["rows"]
            )
            assert [row["tetrahedra"] for row in rows] == [qualifier["tetrahedra"] * 8**i for i in range(3)]
            for i, row in enumerate(rows):
                assert row["seed_sha256"] == qualifier["mesh_sha256"]
                assert row["minimum_tetrahedron_volume_m3"] > 0
                assert "no CAD projection" in row["geometry_mode"]
                assert digest(run / f"meshes/level{i}.npz")[0] == row["mesh_sha256"]
            meshes.extend(rows)
            qualified.extend(qualifier["rows"])
        scientific_checks = {
            "full_polyhedral_meshes_with_volume_area_COM_checks": 6,
            "original_five_case_fields_freshly_requalified": len(qualified),
            "full_original_force_and_probe_budgets_retained": True,
            "all_original_nodal_loads_exactly_reproduced": True,
            "new_fine_equilibrium_states_accepted_in_this_archive": 0,
            "physical_convergence_accepted": False,
        }
    elif plan["kind"] == "local_finite_setup":
        for shape in ["ellipsoid", "capsule"]:
            generator = summaries[f"fixed-polyhedron-{shape}-local-hotspot-meshes-v1"]
            qualifier = summaries[f"fixed-polyhedron-{shape}-local-original-qualification-v1"]
            assert generator["new_equilibrium_states_solved"] == qualifier["new_equilibrium_states_solved"] == 0
            assert len(generator["rows"]) == 3 and len(qualifier["rows"]) == 5
            assert qualifier["all_original_five_loads_full_fields_and_probes_qualified"]
            assert all(
                row["original_full_nodal_force_maximum_difference_N"] == 0
                and row["original_budget_fraction"] <= 1
                and row["all_original_probe_coordinates_and_validity_retained"]
                and row["all_quadrature_tensor_components_checked"] == 99 * qualifier["tetrahedra"]
                for row in qualifier["rows"]
            )
            for level, row in enumerate(generator["rows"]):
                assert row["unchanged_physical_load_radius_m"] == 0.003
                assert row["passes_per_level"] == 2 and "no CAD projection" in row["geometry_mode"]
                assert (
                    digest(p / f"fixed-polyhedron-{shape}-local-hotspot-meshes-v1/results/level{level}.npz")[0]
                    == row["mesh_sha256"]
                )
                assert all(
                    item["all_child_tetrahedra_contained_in_declared_parent"]
                    and item["each_parent_volume_preserved"]
                    and item["maximum_parent_volume_relative_difference"] <= 1e-10
                    for item in row["parent_partition_records"]
                )
            assert any(parent["label"] == f"fixed-polyhedron-{shape}-fields-v1" for parent in parents)
        scientific_checks = {
            "full_local_meshes_with_original_domain_and_parent_partitions": 6,
            "original_million_cell_fields_freshly_requalified": 10,
            "original_nodal_loads_exact_and_full_force_corner_quadrature_probe_budgets_retained": True,
            "new_finer_equilibria_accepted_in_this_archive": 0,
            "physical_convergence_accepted": False,
        }
    elif plan["kind"] == "u_shared_anchor_diagnosis":
        old = summaries["u-shared-four-comparator-three-level-qualification-v1"]
        assert len(old["rows"]) == 3 and old["common_material_points"] == 75
        proof_path = p / "meta/u-shared-four-comparator-original-three-exact-records-v1.json"
        retain(proof_path)
        proof = json.loads(proof_path.read_text())
        assert (
            proof["all_original_three_level_rows_exactly_equal"]
            and proof["all_original_adjacent_changes_exactly_equal"]
        )
        assert proof["original_common_material_points_unchanged"]
        transfer = summaries["u-shared-initialization-fourth-anchor-transfer-qualification-v2"]
        assert len(transfer["rows"]) == 6 and transfer["new_equilibrium_states_solved"] == 0
        assert not transfer["all_original_anchors_are_existing_P2_nodes_at_original_1e_minus12_m_check"]
        for name in [
            "u-shared-initialization-fourth-v1",
            "u-shared-initialization-fourth-anchor-transfer-qualification-v1",
        ]:
            assert plan["closed_jobs"][name]["returncode"] == 1
        assert not (p / "u-shared-initialization-fourth-v1/accepted-contact-history/manifest.json").exists()
        generator = summaries["u-shared-initialization-anchor-conforming-mesh-v1"]
        assert generator["new_equilibrium_states_solved"] == 0 and len(generator["rows"]) == 1
        row = generator["rows"][0]
        assert row["tetrahedra"] == 16329 and row["same_U_CAD_constructor"]
        assert row["CAD_volume_before_m3"] == row["CAD_volume_after_m3"]
        assert max(row["original_anchor_used_mesh_vertex_distances_m"]) <= 1e-12
        assert digest(p / "u-shared-initialization-anchor-conforming-mesh-v1/results/mesh.npz")[0] == row["mesh_sha256"]
        scientific_checks = {
            "original_three_level_rows_adjacent_changes_and75_probes_exactly_reproduced": True,
            "original_six_anchor_coordinates_and_components_preserved": True,
            "original_node_check_failure_and_isolation_import_failure_retained": True,
            "same_U_CAD_volume_COM_and_original_anchor_vertices_qualified": True,
            "new_contact_equilibria_or_fourth_level_stages_accepted": 0,
            "physical_convergence_accepted": False,
        }
    elif plan["kind"] == "fixed_polyhedron_fields":
        shape = plan["geometry"]
        audit = summaries[f"fixed-polyhedron-{shape}-three-level-audit-v1"]
        assert audit["geometry"] == shape and audit["all_original_and_new_fields_freshly_audited"]
        expected_counts = [14916, 119328, 954624] if shape == "ellipsoid" else [15979, 127832, 1022656]
        assert audit["tetrahedra"] == expected_counts
        assert len(audit["rows"]) == 15 and len(audit["adjacent_changes"]) == 10
        assert all(
            row["full_nodal_load_reintegration_maximum_error_N"] == 0
            and row["original_budget_fraction"] <= 1
            and row["all_quadrature_tensor_components_checked"] == 99 * row["tetrahedra"]
            for row in audit["rows"]
        )
        for level in [1, 2]:
            producer = summaries[f"fixed-polyhedron-{shape}-level{level}-v1"]
            assert producer["new_equilibrium_states_solved"] == 5 and len(producer["rows"]) == 5
            assert producer["tetrahedra"] == expected_counts[level]
        assert any(parent["label"] == "fixed-polyhedron-convex-setup-v1" for parent in parents)
        for level in range(3):
            path = p / f"fixed-polyhedron-{shape}-three-meshes-v1/meshes/level{level}.json"
            retain(path)
            metadata = json.loads(path.read_text())
            assert "no CAD projection" in metadata["geometry_mode"]
            assert metadata["tetrahedra"] == expected_counts[level]
            assert digest(path.with_suffix(".npz"))[0] == metadata["mesh_sha256"]
        scientific_checks = {
            "geometry": shape,
            "new_finer_equilibria": 10,
            "original_and_new_fields_freshly_audited": 15,
            "exact_original_nodal_reintegration_and_unchanged_complete_budgets": True,
            "explicit_quadrature_tensor_components_checked": sum(
                row["all_quadrature_tensor_components_checked"] for row in audit["rows"]
            ),
            "fixed_seed_polyhedral_domain_volume_area_and_COM_qualified_in_parent": True,
            "volume_surface_basis_and_boundary_quadrature_still_coupled": True,
            "physical_convergence_accepted": False,
        }
    elif plan["kind"] == "fixed_polyhedron_peak_cells":
        diagnosis = summaries["fixed-polyhedron-raw-peak-cell-diagnosis-v1"]
        assert diagnosis["family"] == "fixed_polyhedron" and len(diagnosis["rows"]) == 30
        assert diagnosis["manufactured_symmetric_stress_control_passed"]
        assert diagnosis["all_normal_only_trace_controls_passed"]
        for shape in ["ellipsoid", "capsule"]:
            assert any(parent["label"] == f"fixed-polyhedron-{shape}-fields-v1" for parent in parents)
            path = p / f"fixed-polyhedron-{shape}-three-level-audit-v1/results/summary.json"
            retain(path)
            audit = json.loads(path.read_text())
            for row in [item for item in diagnosis["rows"] if item["geometry"] == shape]:
                original = next(
                    item for item in audit["rows"] if item["case"] == row["case"] and item["level"] == row["level"]
                )
                assert row["raw_global_peak_Pa"] == original["raw_global_peak_Pa"]
                assert row["raw_global_peak_position_m"] == original["raw_global_peak_position_m"]
                assert row["incident_cells"] > 0 and 0 < row["peak_cell_mean_ratio_quality"] <= 1 + 1e-12
        scientific_checks = {
            "all_original_raw_maxima_and_locations_preserved": 30,
            "every_cell_quality_and_unsmoothed_same_vertex_incident_tensor_retained": True,
            "original_boundary_traction_compatibility_and_manufactured_controls_retained": True,
            "new_equilibria_solved": 0,
            "physical_convergence_causal_explanation_or_singularity_accepted": False,
        }
    elif plan["kind"] == "native_matched_axis_fields":
        shape = plan["geometry"]
        configurations = [(f"extended-native-{shape}-level0", 8, 10, "v2")]
        configurations += [
            (f"native-{shape}-fixed-original-p{ports}-q{quadrature}", ports, quadrature, "v1")
            for ports, quadrature in [(4, 10), (16, 10), (8, 14)]
        ]
        checked_components = 0
        for prefix, ports, quadrature, version in configurations:
            producer = summaries[f"{prefix}-models-{version}"]
            audit = summaries[f"{prefix}-independent-audit-{version}"]
            assert audit["accepted_fields_freshly_audited"] == 19
            assert audit["unsupported_or_rejected_inputs_retained"] == 1 and len(audit["rows"]) == 20
            assert len(producer["cases"]) == 5
            for case in producer["cases"]:
                assert case["patch_radius_m"] == 0.003 and len(case["models"]) == 4
                assert all(row["ports"] == ports and row["native_quadrature"] == quadrature for row in case["models"])
            accepted = [row for row in audit["rows"] if row["accepted"]]
            assert all(
                row["original_budget_fraction"] <= 1
                and row["all_original_corner_and_independent_quadrature_gates_passed"]
                and row["all_quadrature_tensor_components_checked"] == 99 * audit["tetrahedra"]
                for row in accepted
            )
            checked_components += sum(row["all_quadrature_tensor_components_checked"] for row in accepted)
        assert any(parent["label"] == "native-intrinsic-capacity-v1" for parent in parents)
        scientific_checks = {
            "geometry": shape,
            "accepted_native_fields_freshly_audited": 76,
            "unsupported_original_pressure_rows_retained": 4,
            "all_five_loads_four_models_and_ports4_8_16_quadrature10_14_retained": True,
            "explicit_quadrature_tensor_components_checked": checked_components,
            "original_force_scatter_corners_energy_source_gates_retained": True,
            "physical_convergence_or_warmed_cost_accepted": False,
        }
    elif plan["kind"] == "native_matched_axes_summary":
        study = summaries["native-convex-matched-numerical-axes-comparison-v1"]
        assert study["manufactured_exact_volume_norm_control_passed"]
        assert study["independently_audited_native_fields"] == 228 and study["unsupported_inputs_retained"] == 12
        assert len(study["qualified_native_runs"]) == 12
        for key in ["same_mesh_numerical_axis_comparisons", "three_spatial_level_model_bias_rows"]:
            assert len(study[key]) == 120 and sum(row["accepted"] for row in study[key]) == 114
        for row in study["same_mesh_numerical_axis_comparisons"]:
            assert len(row["identical_frozen_inputs_sha256"]) == 64
        for shape in ["ellipsoid", "capsule"]:
            assert any(parent["label"] == f"native-{shape}-matched-axes-fields-v1" for parent in parents)
            for level in [1, 2]:
                assert any(parent["label"] == f"extended-native-{shape}-level{level}-fields-v1" for parent in parents)
        scientific_checks = {
            "independently_audited_actual_native_fields": 228,
            "original_unsupported_rows_retained": 12,
            "same_mesh_numeric_control_rows": 120,
            "same_mesh_model_bias_rows_across_three_spatial_levels": 120,
            "exact_volume_symmetric_tensor_norm_and_constant_four_point_controls_passed": True,
            "original_frozen_input_bytes_identical_for_each_numerical_control": True,
            "physical_convergence_or_warmed_accuracy_cost_accepted": False,
        }
    elif plan["kind"] == "native_finite_fields":
        shape, level = plan["geometry"], plan["level"]
        producer = summaries[f"extended-native-{shape}-level{level}-models-v2"]
        audit = summaries[f"extended-native-{shape}-level{level}-independent-audit-v2"]
        assert len(producer["cases"]) == 5 and len(audit["rows"]) == 20
        accepted = [row for row in audit["rows"] if row["accepted"]]
        assert len(accepted) == audit["accepted_fields_freshly_audited"] > 0
        assert len(audit["rows"]) - len(accepted) == audit["unsupported_or_rejected_inputs_retained"]
        assert all(
            row["all_original_corner_and_independent_quadrature_gates_passed"]
            and row["original_budget_fraction"] <= 1
            and row["all_quadrature_tensor_components_checked"] == 99 * audit["tetrahedra"]
            for row in accepted
        )
        for case in producer["cases"]:
            assert case["patch_radius_m"] == 0.003 and case["reference_quadrature_level"] == 3
            assert len(case["models"]) == 4
            assert {row["model"] for row in case["models"]} == {
                "pressure",
                "componentwise",
                "pair_union",
                "body_source_prior",
            }
            assert all(row["ports"] == 8 and row["native_quadrature"] == 10 for row in case["models"])
        assert any(parent["label"] == f"extended-finite-{shape}-fields-v1" for parent in parents)
        assert any(parent["label"] == "native-intrinsic-capacity-v1" for parent in parents)
        scientific_checks = {
            "geometry": shape,
            "tetrahedra": audit["tetrahedra"],
            "all_original_five_loads_four_models_retained": 20,
            "accepted_native_fields_freshly_audited": len(accepted),
            "unsupported_or_rejected_inputs_retained": len(audit["rows"]) - len(accepted),
            "explicit_quadrature_tensor_components_checked": sum(
                row["all_quadrature_tensor_components_checked"] for row in accepted
            ),
            "actual_physical_point_scatter_full_corners_equilibrium_energy_and_source_diagnostics": True,
            "physical_convergence_or_warmed_accuracy_cost_accepted": False,
        }
    elif plan["kind"] == "path_model_regression":
        for backend in ["cpu", "gpu"]:
            run = p / f"intrinsic-path-heap-models-{backend}-regression-v1"
            suites = list(element_tree.fromstring((run / "junit.xml").read_bytes()).iter("testsuite"))
            assert sum(int(row.attrib["tests"]) for row in suites) == 8
            assert all(int(row.attrib["failures"]) == int(row.attrib["errors"]) == 0 for row in suites)
            for source in Path("genesis/engine/solvers/rigid/stress").rglob("*.py"):
                assert digest(source)[0] == digest(run / "source" / source)[0]
            assets = json.loads((run / "reconstructed-input-assets.json").read_text())
            assert len(assets["assets"]) == 3 and assets["all_base_revision_blobs_match_current_assets"]
            for row in assets["assets"]:
                assert digest(run / row["reconstructed_copy"])[0] == row["sha256"]
        scientific_checks = {
            "additional_source_closed_CPU_CUDA_cases": 16,
            "both_distance_algorithms_self_balanced_source_prior_and_pair_union_original_gates": True,
            "current_worktree_engine_matches_tested_copies": True,
            "input_assets_reconstructed_from_unchanged_recorded_Git_revisions": True,
            "fine_physical_model_or_convergence_accepted": False,
        }
    elif plan["kind"] == "peak_boundary":
        diagnosis = summaries["extended-finite-raw-peak-boundary-diagnosis-v1"]
        assert diagnosis["new_equilibrium_states_solved"] == 0 and len(diagnosis["rows"]) == 90
        assert diagnosis["manufactured_symmetric_stress_control_passed"]
        assert diagnosis["all_normal_only_trace_controls_passed"]
        for shape in ["ellipsoid", "capsule", "thin_neck", "u", "narrow_gap", "cavity"]:
            assert any(parent["label"] == f"extended-finite-{shape}-fields-v1" for parent in parents)
            source = p / f"extended-finite-{shape}-three-level-audit-v1/results/summary.json"
            retain(source)
            audit = json.loads(source.read_text())
            original = {(row["case"], row["level"]): row for row in audit["rows"]}
            for row in [row for row in diagnosis["rows"] if row["geometry"] == shape]:
                old = original[(row["case"], row["level"])]
                assert row["raw_global_peak_Pa"] == old["raw_global_peak_Pa"]
                assert row["raw_global_peak_position_m"] == old["raw_global_peak_position_m"]
        scientific_checks = {
            "original_raw_peak_positions_and_values_preserved": 90,
            "boundary_topology_normals_and_original_prescribed_tractions_retained": True,
            "manufactured_stress_and_normal_only_algebraic_controls_passed": True,
            "new_equilibrium_states_solved": 0,
            "physical_convergence_or_singularity_accepted": False,
        }
    elif plan["kind"] == "matrix_summary":
        combined = summaries["extended-six-shape-eight-level-summary-v1"]
        assert len(combined["rows"]) == 240 and len(combined["tails"]) == 30
        shapes = ["ellipsoid", "capsule", "thin_neck", "u", "narrow_gap", "cavity"]
        for shape in shapes:
            assert any(parent["label"] == f"extended-finite-{shape}-fields-v1" for parent in parents)
            rows = [row for row in combined["rows"] if row["geometry"] == shape]
            assert len(rows) == 40 and sorted({row["level"] for row in rows}) == list(range(8))
            assert sum(row["separately_fresh_complete_field_audit"] for row in rows) == 15
        assert all(not row["physical_convergence_accepted"] for row in combined["tails"])
        scientific_checks = {
            "original_and_new_geometry_load_level_rows": 240,
            "complete_original_probe_coordinates_validity_raw_peaks_and_energies_retained": True,
            "fresh_latest_three_level_complete_field_audits": 90,
            "whole_matrix_physical_convergence_accepted": False,
        }
    elif plan["kind"] == "matrix_fields":
        shape = plan["geometry"]
        audit = summaries[f"extended-finite-{shape}-three-level-audit-v1"]
        assert audit["geometry"] == shape and audit["all_original_and_new_fields_freshly_audited"]
        assert len(audit["rows"]) == 15 and len(audit["adjacent_changes"]) == 10
        assert all(
            row["full_nodal_load_reintegration_maximum_error_N"] == 0
            and row["original_budget_fraction"] <= 1
            and row["all_quadrature_tensor_components_checked"] == 99 * row["tetrahedra"]
            for row in audit["rows"]
        )
        for level in [1, 2]:
            producer = summaries[f"extended-finite-{shape}-level{level}-v1"]
            assert producer["new_equilibrium_states_solved"] == 5 and len(producer["rows"]) == 5
            assert producer["tetrahedra"] == audit["tetrahedra"][level]
        assert any(parent["label"] == "extended-six-shape-qualification-v1" for parent in parents)
        assert any(parent["label"] == f"fine-reference-{shape}-v2" for parent in parents)
        scientific_checks = {
            "geometry": shape,
            "new_finer_equilibrium_states_solved": 10,
            "original_and_new_fields_freshly_audited": 15,
            "every_original_nodal_load_exactly_reproduced": True,
            "explicit_quadrature_tensor_components_checked": sum(
                row["all_quadrature_tensor_components_checked"] for row in audit["rows"]
            ),
            "all_original_probes_validity_raw_global_peaks_and_energies_retained": True,
            "physical_convergence_accepted": False,
        }
    elif plan["kind"] == "matrix_setup":
        shapes = ["ellipsoid", "capsule", "thin_neck", "u", "narrow_gap", "cavity"]
        qualified_rows, mesh_rows = [], []
        for shape in shapes:
            qualification = summaries[f"extended-finite-{shape}-original-qualification-v1"]
            assert qualification["qualify_only"] and qualification["new_equilibrium_states_solved"] == 0
            assert qualification["all_original_five_loads_full_fields_and_probes_qualified"]
            assert qualification["geometry"] == shape and len(qualification["rows"]) == 5
            assert all(
                row["original_full_nodal_force_maximum_difference_N"] == 0
                and row["original_budget_fraction"] <= 1
                and row["all_original_probe_coordinates_and_validity_retained"]
                and row["all_quadrature_tensor_components_checked"] == 99 * qualification["tetrahedra"]
                for row in qualification["rows"]
            )
            qualified_rows.extend(qualification["rows"])
            generator = p / f"extended-matrix-mesh-{shape}-v1/source/research/rigid_stress/validation_meshes.py"
            baseline_generator = p / f"fine-ref-{shape}-l2-v1/source/research/rigid_stress/validation_meshes.py"
            retain(baseline_generator)
            assert generator.read_bytes() == baseline_generator.read_bytes()
            paths = sorted((p / f"extended-matrix-mesh-{shape}-v1/meshes").glob("*.json"))
            assert len(paths) == 2
            for path in paths:
                mesh = json.loads(path.read_text())
                assert mesh["geometry"]["name"] == shape and mesh["tetrahedra"] > qualification["tetrahedra"]
                assert mesh["minimum_volume_m3"] > 0 and mesh["scaled_jacobian_percentiles"][0] > 0
                assert digest(path.with_suffix(".npz"))[0] == mesh["mesh_sha256"]
                mesh_rows.append(mesh)
        scientific_checks = {
            "original_finest_finite_load_fields_freshly_qualified": len(qualified_rows),
            "every_original_nodal_load_exactly_reproduced": True,
            "explicit_quadrature_tensor_components_checked": sum(
                row["all_quadrature_tensor_components_checked"] for row in qualified_rows
            ),
            "complete_original_probes_and_validity_retained": True,
            "new_meshes_with_identical_original_CAD_generator": len(mesh_rows),
            "new_finer_equilibrium_states_accepted": 0,
            "physical_tail_accepted": False,
        }
    else:
        audit = summaries["egg-frozen-traction-local-bulk-quadrature-audit-v2"]
        qualification = summaries["nested-bulk-observable-qualification-v1"]
        assert qualification["qualification_passed"]
        assert len(audit["rows"]) == 15 and len(audit["adjacent_changes"]) == 10
        assert audit["every_field_full_quadrature_directly_reconstructed"]
        assert all(
            row["complete_residual_N"] <= row["original_budget_N"]
            and row["all_quadrature_tensor_components_checked"] == 99 * row["tetrahedra"]
            for row in audit["rows"]
        )
        assert any(parent["label"] == "frozen-traction-bulk-fields-v1" for parent in parents)
        scientific_checks = {
            "independently_reconstructed_original_fields": 15,
            "explicit_full_quadrature_tensor_components_checked": sum(
                row["all_quadrature_tensor_components_checked"] for row in audit["rows"]
            ),
            "original36_probes_and_raw_extrema_retained": True,
            "old_corner_only_audit_scope_retained_separately": True,
            "physical_tail_accepted": False,
        }

    # Keep the physical capture and fixed geometry assets explicit, regardless
    # of whether an immediate parent already retained their identical bytes.
    for name in plan.get("physical_configurations", []):
        path = p / name
        retain(path)
        config = json.loads(path.read_text())
        closure_path = Path(config["closure_run"]) / "results/configuration.json"
        retain(closure_path)
        capture = Path(json.loads(closure_path.read_text())["capture"])
        report_path = capture / "results/summary.json"
        retain(report_path)
        captured = json.loads(report_path.read_text())
        frame = [row for row in captured["captures"] if row["mode"] == "stationary_lift_hold"][-1]
        retain(capture / "results" / frame["geometry_snapshot"])
        fingers = {row["link"] for row in frame["fingers"]}
        for row in captured["geometry"]:
            if row["link"] in fingers:
                retain(capture / "results" / row["mesh"])
        retain(capture / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz")
    for name in plan.get("extra_files", []):
        retain(p / name)
    retain(args.plan)
    retain(Path(__file__))
    summary = {
        "scope": __doc__,
        "kind": plan["kind"],
        "closed_runs": records,
        "parents": parents,
        "source_identity_records": identities,
        "actual_inputs_reverified": input_records,
        "scientific_checks": scientific_checks,
        "closed_run_summaries": summaries,
        "pending_work_excluded_from_successful_evidence": plan["pending_work"],
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.update([meta / "summary.json", meta / "jobs.txt", meta / "job-commands.txt"])
    assert all(path.stat().st_size < 50_000_000 for path in extras)
    assert not any(any(path.is_relative_to(run) for run in runs) for path in extras)
    destination = args.destination_root / label
    publish_bounded(
        args.runtime_root, destination, runs, sorted(extras), summary, plan["limitations"], p / "publication-staging"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.rigid_stress.scripts.verify_runtime_publication",
            "--manifest",
            str(destination / "manifest.json"),
            "--runtime-root",
            str(args.runtime_root),
            "--staging-root",
            str(p / "publication-staging" / label),
            "--output",
            str(meta / "runtime-verification.json"),
        ],
        check=True,
    )
    assert all(digest(Path(path))[0] == item["sha256"] for path, item in input_records.items())


if __name__ == "__main__":
    main()
