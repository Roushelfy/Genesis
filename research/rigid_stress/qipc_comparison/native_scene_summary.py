"""Bind existing scene timing, complete conformance and separate sampled memory."""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .artifacts import file_id, runtime_identity, write_json


COHORTS = (
    ("canonical", "native-production-canonical-pilot-v1", "native-production-canonical-audit-v1",
     "native-portal-point-combined-v1", "native-portal-point-combined-audit-v1", (1, 8, 32)),
    ("scaling128-512", "native-production-scaling128-512-v1", "native-production-scaling128-512-audit-v1",
     "native-portal-point-b128-b512-v1", "native-portal-point-b128-b512-audit-v1", (128, 512)),
    ("scaling2048-8192", "native-production-scaling2048-8192-v1", "native-production-scaling2048-8192-audit-v1",
     "native-portal-point-b2048-b8192-v1", "native-portal-point-b2048-b8192-audit-v1", (2048, 8192)),
)
MODES = ("rigid", "dpr-max", "dpr-full", "scr-max", "scr-full")


class Reader:
    def __init__(self, root, proof):
        self.root, self.proof = root, proof
        self.receipts, self.checks, self.sources = {}, [], {}
        self.canonical = None

    def check(self, name, passed):
        self.checks.append({"name": name, "passed": bool(passed)})
        assert passed, name

    def identify(self, path):
        path = Path(path)
        if path not in self.receipts:
            self.receipts[path] = file_id(path)
        return self.receipts[path]

    def read(self, path):
        self.identify(path)
        return json.loads(Path(path).read_text())

    def verify(self, ident):
        self.check("hash/" + ident["path"], self.identify(ident["path"]) == ident)

    def audit(self, name):
        path = self.root / name / "group0/audit/result.json"
        result = self.read(path)
        self.check(name + "/admitted", result.get("passed", result.get("timing_receipts_passed", False)))
        self.check(name + "/all_checks", all(c["passed"] for c in result.get("checks", [])))
        for ident in result["receipts"]:
            self.verify(ident)
        return result, self.identify(path)

    def source(self, campaign):
        campaign = Path(campaign)
        if campaign in self.sources:
            return self.sources[campaign]
        source = self.read(campaign / "source.json")
        spec = {r["path"]: {k: r[k] for k in ("bytes", "sha256")} for r in source["files"]}
        self.check(str(campaign) + "/858_sources", len(spec) == len(source["files"]) == 858)
        patch = self.identify(campaign / "source.patch")
        self.check(str(campaign) + "/patch", patch["sha256"] == source["patch_sha256"])
        if self.canonical is None:
            self.check("canonical_revision", source["revision"] == "1ff39b0775200120737225fdfb8631c9b1fc5fa7"
                       and patch["bytes"] == 0)
            self.canonical = spec
        differences = {p for p in spec.keys() | self.canonical.keys() if spec.get(p) != self.canonical.get(p)}
        override = None
        if differences:
            self.check("known_B8_comment_only", campaign.name == "native-portal-point-conformance-v1"
                       and differences == {"genesis/utils/array_class.py"})
            proof = self.read(self.proof)
            self.check("B8_original_manifest", Path(proof["captured_manifest"]) == campaign / "source.json")
            override = Path(proof["reconstructed_file"])
            current = Path(source["source"]) / "genesis/utils/array_class.py"
            self.check("B8_original_hash", spec["genesis/utils/array_class.py"]["sha256"]
                       == "c5e4a0e115597fb9c80c10b77e73ae9cd5df501e0c066b7c5e006d207dc58da3")
            self.check("B8_AST_equivalent", ast.dump(ast.parse(override.read_text()), include_attributes=False)
                       == ast.dump(ast.parse(current.read_text()), include_attributes=False))
        else:
            self.check(str(campaign) + "/clean_patch", patch["bytes"] == 0)
        for rel, expected in spec.items():
            path = override if override is not None and rel == "genesis/utils/array_class.py" else Path(source["source"]) / rel
            ident = self.identify(path)
            self.check("source/" + str(path), all(ident[k] == expected[k] for k in ("bytes", "sha256")))
        for row in self.read(campaign / "analysis-files.json"):
            ident = self.identify(campaign / "analysis-source" / row["path"])
            self.check("captured/" + str(campaign / row["path"]), all(ident[k] == row[k] for k in ("bytes", "sha256")))
        receipt = {"manifest": self.identify(campaign / "source.json"), "revision": source["revision"], "patch": patch,
                   "comment_equivalence": self.identify(self.proof) if differences else None}
        self.sources[campaign] = receipt
        return receipt

    def config(self, value):
        if value is None:
            return None
        value = dict(value)
        mesh = self.identify(value["mesh"])
        value["mesh"] = {k: mesh[k] for k in ("bytes", "sha256")}
        return value

    def stress_links(self, primary):
        if primary["scope"] == "rigid":
            self.check("rigid_has_no_stress_options", primary.get("stress_configurations", []) == [])
            return []
        configs = primary["stress_configurations"]
        self.check("live_has_exactly_one_stress_link", len(configs) == 1)
        return configs

    def bind(self, label, primary, control, cadence):
        self.check(label + "/control", control["passed"] and all(control["checks"].values()))
        if cadence:
            keys = ("method", "envs", "output_mode", "stress_stride", "dt_s", "substeps", "warmup", "steps")
            self.check(label + "/simulation", all(primary[k] == control[k] for k in keys))
            config, bank = primary["stress_configuration"], primary["conditions"]
        else:
            keys = (("envs", "envs"), ("output_mode", "output_mode"), ("dt", "dt_s"),
                    ("substeps", "substeps"), ("warmup_steps", "warmup"))
            self.check(label + "/simulation", all(primary[a] == control[b] for a, b in keys))
            configs = self.stress_links(primary)
            self.check(label + "/stress_link", len(configs) == int(control["method"] != "rigid"))
            config = configs[0] if configs else None
            bank = self.identify(primary["conditions"])
            self.check(label + "/bank", bank["sha256"] == primary["conditions_sha256"])
        self.verify(bank)
        self.verify(control["conditions"])
        self.check(label + "/bank_identity", bank["sha256"] == control["conditions"]["sha256"])
        config = self.config(config)
        self.check(label + "/all_stress_options", config == self.config(control["stress_configuration"]))
        self.check(label + "/horizon", control["steps"] == 1200 and control["warmup"] == 900
                   and control["dt_s"] == 0.01 and control["substeps"] == 1)
        self.check(label + "/fusion", primary["effective_rigid_post_fusion"] == control["effective_rigid_post_fusion"])
        return config, bank


def measure(reader, label, primary, batch, cadence):
    repeats = primary["repeats"]
    reader.check(label + "/three_repeats", len(repeats) == 3)
    rate_key, step_key, invalid_key = (("env_scene_steps_per_second", "scene_steps", "invalid_environment_updates")
                                     if cadence else ("transitions_per_second", "steps", "invalid_environment_steps"))
    rates = []
    for repeat in repeats:
        seconds = repeat["seconds"]
        reader.check(label + "/complete_valid", repeat[step_key] == 1200 and repeat[invalid_key] == 0
                     and math.isfinite(seconds) and seconds > 0)
        reader.check(label + "/rate", math.isclose(repeat[rate_key], batch * 1200 / seconds, rel_tol=1e-12))
        rates.append(repeat[rate_key])
    return {"median_env_scene_steps_per_s": statistics.median(rates), "minimum": min(rates), "maximum": max(rates),
            "median_scene_step_ms": statistics.median(r["seconds"] for r in repeats) * 1000 / 1200,
            "timed_environment_steps": batch * 3600, "invalid_environment_steps": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--b8-equivalence", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reader, rows = Reader(args.root, args.b8_equivalence), []
    cohorts = [(*c, False) for c in COHORTS] + [
        ("cadence", "native-observation-cadence-primary-v1", "native-observation-cadence-audit-v1",
         "native-observation-cadence-conformance-v1", "native-observation-cadence-audit-v1", (8, 128), True)]
    for cohort, campaign_name, audit_name, control_name, control_audit_name, batches, cadence in cohorts:
        campaign, controls = args.root / campaign_name, args.root / control_name
        audit, audit_id = reader.audit(audit_name)
        conformance, conformance_id = reader.audit(control_audit_name)
        source = reader.source(campaign)
        for group, batch in enumerate(batches):
            baseline = None
            entries = [r for r in audit["timing_rows" if cadence else "rows"] if r["batch"] == batch]
            reader.check(f"{cohort}/B{batch}/mode_count", len(entries) == (13 if cadence else 5))
            for admitted in entries:
                if cadence:
                    method, output, stride = admitted["method"], admitted["output"], admitted["stride"]
                    name = f"{method}-{output}-s{stride}"
                    mode = "rigid" if method == "rigid" else f"{method}-{output}"
                    p = campaign / f"group{group}" / name / "result.json"
                    c = controls / f"group{group}" / name
                    control_source = reader.source(controls)
                else:
                    mode, stride = admitted["mode"], 1
                    method = "rigid" if mode == "rigid" else mode.split("-")[0]
                    p = campaign / f"group{group}" / (mode + ".json")
                    c = controls / f"group{group}" / ("rigid-max" if mode == "rigid" else mode)
                    start = reader.read(c / "start.json")
                    control_source = reader.source(Path(start["args"]["output"]).parents[1])
                primary, control = reader.read(p), reader.read(c / "result.json")
                label = f"{cohort}/B{batch}/{mode}/s{stride}"
                config, bank = reader.bind(label, primary, control, cadence)
                reader.check(label + "/method", control["method"] == method)
                if cadence:
                    reader.check(label + "/primary", primary["passed"] and not primary["policy_transitions_measured"])
                else:
                    reader.check(label + "/scope", primary["scope"] == ("rigid" if method == "rigid" else "live"))
                metrics = measure(reader, label, primary, batch, cadence)
                reader.check(label + "/audit_rate", metrics["median_env_scene_steps_per_s"]
                             == admitted["median_env_scene_steps_per_second" if cadence else "median_env_steps_per_s"])
                if baseline is None:
                    reader.check(label + "/paired_rigid_first", method == "rigid")
                    baseline = metrics | {"gpu_uuid": primary["gpu_uuid"]}
                reader.check(label + "/paired_gpu", baseline["gpu_uuid"] == primary["gpu_uuid"])
                signature = {"stress": config, "bank_sha256": bank["sha256"], "batch": batch, "dt": 0.01,
                             "substeps": 1, "stride": stride, "engine_assets": reader.canonical}
                rows.append({
                    "cohort": cohort, "batch": batch, "mode": mode, "stress_stride": stride, **metrics,
                    "added_scene_step_ms": metrics["median_scene_step_ms"] - baseline["median_scene_step_ms"],
                    "added_scene_latency_fraction": metrics["median_scene_step_ms"] / baseline["median_scene_step_ms"] - 1,
                    "gpu": primary["gpu_name" if cadence else "gpu"], "gpu_uuid": primary["gpu_uuid"],
                    "physical_error": None, "physical_error_status": "pending for this native-collider load configuration",
                    "policy_executed": False, "sampled_process_memory": None,
                    "setup_seconds": primary["setup_seconds_including_JIT" if cadence else "setup_seconds"],
                    "setup_scope": "Separate driver construction/JIT timer; excludes warmup; not whole-process cold start.",
                    "native_logical_buffer_bytes": None if cadence else primary.get("native_buffer_bytes", 0),
                    "torch_peak_allocated_bytes": None if cadence else primary["torch_peak_allocated_bytes"],
                    "conditions": bank, "stress_configuration": config,
                    "configuration_signature": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                    "primary": reader.identify(p), "timing_audit": audit_id,
                    "conformance": reader.identify(c / "result.json"), "conformance_audit": conformance_id,
                    "source": source, "conformance_source": control_source,
                })
    for suffix, cohort in (("128-512", "scaling128-512"), ("2048-8192", "scaling2048-8192")):
        campaign = args.root / f"native-production-memory{suffix}-v1"
        audit, audit_id = reader.audit(f"native-production-memory{suffix}-audit-v1")
        reader.source(campaign)
        batches = sorted({r["batch"] for r in audit["rows"]})
        for admitted in audit["rows"]:
            batch, mode = admitted["batch"], admitted["mode"]
            row = next(r for r in rows if (r["cohort"], r["batch"], r["mode"]) == (cohort, batch, mode))
            path = campaign / f"group{batches.index(batch)}" / mode
            profile, benchmark = reader.read(path / "result.json"), reader.read(path / "benchmark.json")
            primary = reader.read(row["primary"]["path"])
            reader.check(str(path) + "/exact_stress_options",
                         [reader.config(c) for c in reader.stress_links(benchmark)]
                         == [reader.config(c) for c in reader.stress_links(primary)])
            samples = profile["memory"]["samples"]
            gaps = [b["seconds"] - a["seconds"] for a, b in zip(samples[:-1], samples[1:])]
            reader.check(str(path) + "/gaps", bool(gaps) and all(math.isfinite(g) and g > 0 for g in gaps))
            peak = max(s["process_mib"] for s in samples)
            reader.check(str(path) + "/peak", peak == admitted["sampled_process_peak_mib"] == profile["memory"]["peak_mib"])
            reader.check(str(path) + "/primary_rate", row["median_env_scene_steps_per_s"] == admitted["primary_env_scene_steps_per_s"])
            row["sampled_process_memory"] = {
                **admitted, "first_sample_seconds": samples[0]["seconds"], "last_sample_seconds": samples[-1]["seconds"],
                "maximum_observed_sample_gap_seconds": max(gaps), "median_observed_sample_gap_seconds": statistics.median(gaps),
                "profile": reader.identify(path / "result.json"), "audit": audit_id,
            }
    reader.check("61_scene_rows", len(rows) == 61)
    reader.check("20_separate_memory_profiles", sum(r["sampled_process_memory"] is not None for r in rows) == 20)
    reader.check("unique_cohort_rows", len({(r["cohort"], r["batch"], r["mode"], r["stress_stride"]) for r in rows}) == 61)
    columns = ("cohort", "batch", "mode", "stress_stride", "median_env_scene_steps_per_s", "minimum", "maximum",
               "median_scene_step_ms", "added_scene_step_ms", "added_scene_latency_fraction", "gpu", "gpu_uuid",
               "physical_error", "policy_executed", "configuration_signature")
    with (args.output / "native-scenes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "result.json", {
        "identity": runtime_identity(require_gpu_inventory=False), "passed": True, "rows": rows,
        "checks": reader.checks, "receipts": list(reader.receipts.values()), "sources": list(reader.sources.values()),
        "physical_reference_qualified": False, "whole_goal_complete": False,
        "scope": "35 original-cohort and26 cadence primary rows with complete conformance, same-cohort/batch/GPU paired overheads and20 separate sampled-memory profiles. Three warm repeats, not independent processes; repeated eight-condition bank, not diverse-input reliability. No policy execution in rigid/live benchmark scopes. Sample times/gaps do not bound missed memory peaks or establish maximum capacity; sparse rows inherit no memory. Full/max outputs remain GPU resident. Pad physical errors are not inherited. This positive-table join does not supersede original motion failures, separate diverse SCR failures, every-step pad failures or pending friction accuracy.",
    })


if __name__ == "__main__":
    main()
