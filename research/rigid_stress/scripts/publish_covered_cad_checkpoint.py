"""Archive closed three-level CAD fields or complete source/localization diagnostics without accepting physical convergence."""

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
    assert plan["kind"] in ["fields", "sources"] and plan["goal_completion"] is False
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
    else:
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
