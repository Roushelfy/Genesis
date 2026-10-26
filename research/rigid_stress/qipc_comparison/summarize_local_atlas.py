"""Build a compact metadata packet from local-atlas JSON evidence using only stdlib.

This command performs no scientific execution and may run on a login node.
It reads result JSON and execution receipts, hashes their bytes, and writes a new
packet directory. Array payloads, captured source trees, samples and traces are
never copied. Reported numerical, spatial and execution states stay separate.
"""

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


OMIT_KEYS = {
    "trace", "samples", "polls", "native_buffers", "device_kernels", "by_field",
    "environment_tiling_indices", "conditions", "conditions_data", "per_environment_errors",
    "frame", "frames_data", "source_trace", "arrays",
}
SKIP_DIRECTORIES = {"analysis-source", "analysis-fields", "__pycache__", ".git", "fields"}
KINDS = ("accuracy", "geometry", "benchmarks", "profiles", "joint", "components", "selector", "controls", "unknown")


@dataclass
class Moments:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    nonzero_count: int = 0
    nonzero_total: float = 0.0
    nonfinite: int = 0

    def add(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        if not math.isfinite(value):
            self.nonfinite += 1
            return
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if value != 0:
            self.nonzero_count += 1
            self.nonzero_total += value

    def result(self):
        result = {
            "count": self.count, "maximum": self.maximum,
            "mean": self.total / self.count if self.count else None,
        }
        if 0 < self.nonzero_count < self.count:
            result["nonzero_mean"] = self.nonzero_total / self.nonzero_count
        if self.nonfinite:
            result["nonfinite_count"] = self.nonfinite
        return result


def numeric_values(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from numeric_values(item)


def numeric_summary(value):
    moments = Moments()
    for item in numeric_values(value):
        moments.add(item)
    return moments.result()


def compact(value, depth=0):
    """Retain small metadata and summarize vectors without copying sampled fields."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            return {key: value[key] for key in ("path", "bytes", "sha256") if key in value}
        if depth > 6:
            return {"keys": sorted(value), "omitted_nested_payload": True}
        return {
            key: compact(item, depth + 1)
            for key, item in value.items()
            if key not in OMIT_KEYS
        }
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, bool) for item in value):
            return {"count": len(value), "true": sum(value), "false": len(value) - sum(value)}
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return value if len(value) <= 8 else numeric_summary(value)
        if len(value) <= 16 and depth < 6:
            return [compact(item, depth + 1) for item in value]
        return {"count": len(value), "omitted_payload": True}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str) and len(value) > 1500:
        return {"characters": len(value), "prefix": value[:300], "omitted_text": True}
    return value


def selected(data, keys):
    return {key: compact(data[key]) for key in keys if key in data}


def file_reference(path, run, raw=None):
    if raw is None:
        raw = path.read_bytes()
    return {
        "path": str(path.relative_to(run)), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def reported_state(data):
    return selected(data, (
        "complete", "passed", "numerical_passed", "spatial_passed", "native_complete", "native_passed",
        "native_qualification_passed", "source_variant_spatial_passed", "curved_shell_accuracy_qualified",
    ))


def metadata(data):
    result = selected(data, (
        "stress_backend", "level", "lifecycle", "options", "original_options", "source_options",
        "reconstruction_options", "basis", "basis_provenance", "mesh", "dimensions", "tensor_budget",
        "peak_budget", "load_budget", "force_budget", "moment_budget", "field_metric", "metric",
        "mother", "physical_nodes", "mother_blocks", "coarse_nodes", "genuine_reduction", "identity",
    ))
    if "conditions" in data:
        value = data["conditions"]
        result["conditions"] = {"count": len(value)} if isinstance(value, list) else compact(value)
    if "atlas" in data:
        result["atlas"] = compact(data["atlas"])
    if isinstance(data.get("identity"), dict):
        identity = data["identity"]
        result["identity"] = selected(identity, ("hostname", "python_version", "environment"))
    return result


def flag_counts(values):
    values = list(values)
    return {"recorded": len(values), "true": sum(value is True for value in values),
            "false": sum(value is False for value in values),
            "unknown": sum(value is not True and value is not False for value in values)}


def metric_rows(rows, key):
    collected = {}
    for row in rows:
        metrics = row.get(key, {})
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if name not in collected:
                collected[name] = Moments()
            for item in numeric_values(value):
                collected[name].add(item)
    return {name: values.result() for name, values in sorted(collected.items())}


def row_summary(rows):
    counts = {}
    for key in ("loaded", "coverage_fallbacks", "active_groups", "active_reduced_groups", "numerical_fallbacks"):
        moments = Moments()
        for row in rows:
            moments.add(row.get(key))
        if moments.count:
            counts[key] = {**moments.result(), "sum": moments.total}
    residuals = {}
    for key in ("mother_spatial_residual", "mother_spatial_residual_relative", "feature_numerical_residual_norm"):
        values = [row[key] for row in rows if key in row]
        if values:
            residuals[key] = numeric_summary(values)
    return {
        "row_count": len(rows), "first_step": rows[0].get("step") if rows else None,
        "last_step": rows[-1].get("step") if rows else None,
        "row_reported_errors": metric_rows(rows, "errors"),
        "per_environment_errors": metric_rows(rows, "per_environment_errors"),
        "mean_scope": "All recorded frame/environment entries, including empty-environment zero outputs; nonzero_mean is separate.",
        "reported_spatial_flags": flag_counts(row["spatial_passed"] for row in rows if "spatial_passed" in row),
        "reported_numerical_flags": flag_counts(row["numerical_passed"] for row in rows if "numerical_passed" in row),
        "work_counts": counts, "residual_diagnostics": residuals,
    }


def native_memory(data):
    result = {}
    for key, value in data.items():
        is_memory = any(term in key for term in ("bytes", "memory", "rss", "allocation_count"))
        if is_memory and key not in {"native_buffers", "by_field"}:
            result[key] = compact(value)
    buffers = data.get("native_buffers", [])
    if isinstance(buffers, list) and buffers:
        kinds = {}
        for entry in buffers:
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("kind", "unknown"))
            if kind not in kinds:
                kinds[kind] = {"entries": 0, "logical_bytes": 0, "unaliased_bytes": 0}
            kinds[kind]["entries"] += 1
            size = entry.get("bytes", 0)
            if isinstance(size, (int, float)):
                kinds[kind]["logical_bytes"] += size
                if entry.get("alias_of") is None:
                    kinds[kind]["unaliased_bytes"] += size
        result["native_buffer_inventory"] = {"entries": len(buffers), "by_kind": kinds,
            "note": "No buffer rows copied. Explicit reported shared/per-environment totals take precedence over shape inference."}
    return result


def accuracy(data):
    rows = row_summary(data.get("rows", []))
    if rows["per_environment_errors"]:
        rows["row_maximum_reported_errors"] = {
            name: values["maximum"] for name, values in rows["row_reported_errors"].items()
        }
        rows["row_reported_errors"] = {}
    return {**metadata(data), **reported_state(data), **rows,
            **selected(data, ("covered_contacts", "active_reduced_groups"))}


def geometry(data):
    result = {**metadata(data), **reported_state(data), **row_summary(data.get("rows", []))}
    result.update(selected(data, (
        "name", "geometry", "qualification_variant", "qualification_checks", "partial_footprint_probe",
        "targeted_selection_search", "junction_probe", "material_sides", "aggregation", "opposed_distant_sources",
        "continuity", "intentional_fallback", "junction_observed_three", "fixed_physical_nodes", "failure_policy",
        "radius_m", "transition_m", "complete_boundary",
    )))
    result["remote_neck"] = [
        {**selected(row, ("name", "numerical_passed", "spatial_passed", "neck_witness", "errors",
                          "tensor_error_norm", "reference_tensor_norm", "coverage_fallbacks"))}
        for row in data.get("rows", []) if "neck_witness" in row
    ]
    if "variants" in data:
        result["variants"] = [
            {**selected(item, ("name", "complete", "passed", "qualification_checks")),
             "variant_file": str(Path(item.get("name", "unknown")) / "result.json")}
            for item in data["variants"] if isinstance(item, dict)
        ]
    return result


def benchmarks(data):
    execution_keys = ("host", "slurm_job_id", "source_revision", "gpu", "gpu_uuid", "torch", "quadrants")
    observation_keys = ("setup_seconds", "surface_quadrature_grid_seconds_including_JIT", "dofs", "tetrahedra",
                        "shell_quadrilaterals", "stress_quantity", "method_selected", "factor_block_entries",
                        "surface_inverse_selected", "shell_recovery_graph_selected", "native_allocation_count")
    parameters = {key: value for key, value in data.items()
                  if isinstance(value, (str, int, float, bool)) and key not in {"command", "arguments"}
                  and key not in execution_keys and key not in observation_keys
                  and not any(term in key for term in ("bytes", "memory", "rss"))}
    case_keys = {"scope", "envs", "level", "precision", "quadrature", "pressure_quadrature", "dt", "substeps",
                 "seed", "varied", "condition_count", "conditions_sha256", "method", "output_mode", "warmup_steps"}
    engine_flags = {key: value for key, value in parameters.items() if key not in case_keys and not key.startswith("atlas_")}
    parameters = {key: value for key, value in parameters.items() if key in case_keys or key.startswith("atlas_")}
    parameters["engine_flags"] = engine_flags
    repeats = []
    for index, repeat in enumerate(data.get("repeats", [])):
        row = {key: value for key, value in repeat.items() if isinstance(value, (str, int, float, bool))}
        row["index"] = index
        if repeat.get("steps", 0) and isinstance(repeat.get("seconds"), (int, float)):
            row["milliseconds_per_step"] = repeat["seconds"] * 1000 / repeat["steps"]
        repeats.append(row)
    times = [item["milliseconds_per_step"] for item in repeats if "milliseconds_per_step" in item]
    source_hashes = data.get("source_sha256", {})
    source_inventory = {
        "entry_count": len(source_hashes),
        "canonical_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "scope": "The complete source-hash inventory stays in this referenced result and campaign source manifest.",
    }
    return {
        "parameters": compact(parameters), "execution": selected(data, execution_keys),
        "model_observations": selected(data, observation_keys), "reported_state": reported_state(data),
        "conditions_sha256": data.get("conditions_sha256"), "repeats": repeats,
        "median_milliseconds_per_step": median(times) if times else None,
        "milliseconds_per_step": numeric_summary(times), "memory": native_memory(data),
        "source_hash_inventory": source_inventory,
        "warmup_quality_entry_count": len(data.get("warmup_quality", [])),
        **selected(data, ("recovery_input", "recovery_snapshot_check", "stress_configurations",
                          "build_timings_including_JIT", "setup_seconds")),
    }


def profiles(data):
    result = {
        **metadata(data), **reported_state(data), "memory": native_memory(data),
        **selected(data, ("primary_timing", "scope_note", "category_rules", "device_categories",
                          "intrusive_wall_seconds", "setup_seconds_including_JIT", "frozen_input_reproduction_passed")),
        "trace_omitted": True, "device_kernel_row_count": len(data.get("device_kernels", [])),
    }
    for key in ("warmup_checks", "final_checks"):
        if key in data:
            result[key] = diagnostic_summary(data[key])
    return result


def diagnostic_summary(data):
    """Aggregate repeated check scalars by field path, without exporting sampled rows."""
    numbers, flags = {}, {}

    def visit(value, field):
        if isinstance(value, dict):
            for key, item in value.items():
                if key not in OMIT_KEYS:
                    visit(item, field + "." + key if field else key)
        elif isinstance(value, list):
            for item in value:
                visit(item, field)
        elif isinstance(value, bool):
            if field not in flags:
                flags[field] = {"true": 0, "false": 0}
            flags[field]["true" if value else "false"] += 1
        elif isinstance(value, (int, float)):
            if field not in numbers:
                numbers[field] = Moments()
            numbers[field].add(value)

    visit(data, "")
    return {"flags": flags, "numeric": {key: value.result() for key, value in numbers.items()}}


def timing_path(data):
    result = {key: compact(value) for key, value in data.items()
              if key not in {"repetitions", "validation", "warmup", "signed_cancellation_vs_reference"}}
    result["repetitions"] = []
    for repeat in data.get("repetitions", []):
        if not isinstance(repeat, dict):
            continue
        item = selected(repeat, ("index", "order", "seconds", "seconds_per_iteration", "passed"))
        validation = repeat.get("validation", {})
        item["validation"] = validation_summary(validation)
        item.update(selected(repeat, ("native_valid_all_iterations", "native_valid_across_iterations")))
        if "field_errors_against_audited_control" in repeat:
            item["field_error_maxima_against_audited_control"] = {
                key: numeric_summary(value)["maximum"]
                for key, value in repeat["field_errors_against_audited_control"].items()
            }
        if "native_numerical_fallbacks_last_iteration" in repeat:
            item["native_numerical_fallbacks_last_iteration"] = numeric_summary(repeat["native_numerical_fallbacks_last_iteration"])
        result["repetitions"].append(item)
    if "validation" in data:
        result["validation"] = validation_summary(data["validation"])
    if isinstance(data.get("warmup"), dict):
        result["warmup"] = validation_summary(data["warmup"])
        if "validation" in data["warmup"]:
            result["warmup"]["validation"] = validation_summary(data["warmup"]["validation"])
    if "warmup_field_errors" in data:
        result["warmup_field_error_maxima"] = {
            key: numeric_summary(value)["maximum"] for key, value in data["warmup_field_errors"].items()
        }
    if "signed_cancellation_vs_reference" in data:
        result["signed_cancellation_vs_reference"] = compact(data["signed_cancellation_vs_reference"])
    return result


def validation_summary(data):
    result = selected(data, ("passed", "finite", "native_valid", "native_valid_all_iterations"))
    if "relative_errors" in data:
        result["relative_error_maxima"] = {
            key: numeric_summary(value)["maximum"] for key, value in data["relative_errors"].items()
        }
    return result


def indexed_joint_checks(items):
    result = []
    for item in items:
        environments = item.get("environments", [])
        record = selected(item, ("operator", "name", "environments", "active_groups", "numerical_passed", "passed"))
        for key in ("source_sum_joint_galerkin", "native_residual", "native_valid"):
            if key not in item:
                continue
            record[key] = restrict_environments(item[key], environments)
        result.append(record)
    return result


def restrict_environments(value, environments):
    """Filter full-batch residual arrays to the environments belonging to a joint tuple."""
    if isinstance(value, dict):
        return {key: restrict_environments(item, environments) for key, item in value.items()}
    if isinstance(value, list) and all(isinstance(item, (int, float, bool)) for item in value):
        if all(isinstance(index, int) and 0 <= index < len(value) for index in environments):
            chosen = [value[index] for index in environments]
            return compact(chosen)
        return {"unfiltered_shape": len(value), "reason": "Environment indices unavailable or out of range."}
    return compact(value)


def joint(data):
    result = {**metadata(data), **reported_state(data)}
    result.update(selected(data, ("joint_control", "fallback_control", "preparation_scope", "timing",
                                 "independent_source_sum_is_not_automatically_joint", "spaces", "joint_cache", "native_atlas")))
    rows = []
    for row in data.get("rows", []):
        item = selected(row, ("step", "input", "active_groups", "active_groups_match_load", "grouped_load_error",
                              "load_relief_linearity", "mother_saved_reconstruction", "mother_numerical_residual",
                              "fallback_spin_numerical_residual", "source_sum_saved_reconstruction",
                              "spatial_budget_passed", "numerical_passed"))
        for key in ("spatial_errors", "source_sum_vs_joint", "mother_spatial_residual"):
            if key in row:
                item[key] = summarize_metric_tree(row[key])
        item["joint_checks_for_own_environments"] = indexed_joint_checks(row.get("joint", []))
        item["timing"] = {
            name: timing_path(path) for name, path in row.get("timing", {}).get("paths", {}).items()
        }
        item["timing_state"] = selected(row.get("timing", {}), ("complete", "passed", "setup_seconds"))
        rows.append(item)
    result["keyframes"] = rows
    return result


def summarize_metric_tree(value):
    if isinstance(value, dict):
        return {key: summarize_metric_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        if all(isinstance(item, bool) for item in value):
            return flag_counts(value)
        stats = numeric_summary(value)
        return {key: stats[key] for key in ("count", "maximum", "mean")}
    return compact(value)


def components(data):
    result = {**metadata(data), **reported_state(data)}
    result.update(selected(data, ("criterion", "solver", "iterations", "repetitions", "warmup", "scope",
                                 "multi_rhs_layout", "aggregation_scope", "empty_work_policy", "spin_policy", "criterion_implementation",
                                 "timing_protocol", "scheduling", "incremental_controls", "signed_cancellation",
                                 "preparation_seconds", "state_and_load_bank_setup_seconds", "input_checks")))
    result["paths"] = {name: timing_path(path) for name, path in data.get("paths", {}).items()}
    result["memory"] = compact(data.get("memory", {}))
    return result


def selector(data):
    result = {**metadata(data), **reported_state(data)}
    result.update(selected(data, ("native", "source_variant_spatial_passed", "environments", "source_environments",
                                 "budget", "timing_scope", "synchronization", "quadrants_reductions", "widths")))
    rows = []
    for row in data.get("rows", []):
        item = {key: compact(value) for key, value in row.items()
                if key not in {"scalar_seconds", "cooperative_seconds", "frame"}}
        item["input_frame"] = compact(row.get("frame"))
        count = row.get("calls_per_repeat", 0)
        for schedule in ("scalar", "cooperative"):
            raw = row.get(schedule + "_seconds", [])
            ms = [value * 1000 / count for value in raw] if count else []
            item[schedule + "_milliseconds_per_call"] = ms
            item[schedule + "_median_milliseconds_per_call"] = median(ms) if ms else None
        rows.append(item)
    result["cases"] = rows
    return result


def controls(data):
    return {**metadata(data), **reported_state(data),
            **selected(data, ("checks", "spaces", "patches", "kinematics", "tolerance", "absolute_tolerance",
                              "structural_tolerance", "source", "identity", "scope", "mother_dofs"))}


def classify(data):
    if not isinstance(data, dict):
        return "unknown"
    rows = data.get("rows")
    row_dicts = isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
    if isinstance(data.get("device_categories"), list) and "trace" in data:
        return "profiles"
    if isinstance(data.get("paths"), dict) and isinstance(data.get("incremental_controls"), dict):
        return "components"
    if "timing_scope" in data and "widths" in data and row_dicts:
        return "selector"
    if "joint_control" in data and row_dicts:
        return "joint"
    if (isinstance(data.get("qualification_checks"), dict) and row_dicts) or (
        isinstance(data.get("variants"), list) and "geometry" in data
    ):
        return "geometry"
    if isinstance(data.get("repeats"), list) and "scope" in data and "envs" in data:
        return "benchmarks"
    if row_dicts and ("per_environment_errors" in next(iter(rows), {}) or (
        "stress_backend" in data and "tensor_budget" in data
    )):
        return "accuracy"
    if any(key in data for key in ("checks", "patches", "spaces")):
        return "controls"
    return "unknown"


def campaign_manifest(directory, run):
    result = {"campaign": directory.name, "manifests": {}, "job_ids": [], "execution_receipts": []}
    for name in ("source.json", "submission.json", "analysis-files.json", "tasks.json"):
        path = directory / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        item = {"file": file_reference(path, run, raw)}
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError) as error:
            item["read_error"] = str(error)[:200]
        else:
            item["keys"] = sorted(data) if isinstance(data, dict) else None
            item["entry_count"] = len(data) if isinstance(data, (list, dict)) else None
            if isinstance(data, dict):
                item.update(selected(data, ("revision", "source", "patch_sha256", "job", "dependency",
                                            "node_local_cache", "kernel_profiler", "disable_cuda_graphs")))
                if "files" in data:
                    item["source_file_count"] = len(data["files"])
                if "job" in data:
                    result["job_ids"].append(str(data["job"]))
        result["manifests"][name] = item
    for path in sorted(directory.glob("group*/*returncode.txt")):
        result["execution_receipts"].append({"file": file_reference(path, run), "returncode_text": path.read_text().strip()})
    return result


def csv_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["case"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: csv_cell(value) for key, value in row.items()} for row in rows)


def csv_rows(kind, records):
    rows = []
    for record in records:
        case = record["file"]["path"]
        base = {"case": case}
        if kind == "benchmarks":
            params = record["parameters"]
            for repeat in record["repeats"]:
                rows.append({**base, **{key: params.get(key) for key in ("scope", "envs", "output_mode", "atlas_solver", "atlas_core_scale")},
                             "conditions_sha256": record.get("conditions_sha256"), **repeat})
        elif kind == "accuracy":
            metrics = record["per_environment_errors"] or record["row_reported_errors"]
            work = {name + "_sum": values["sum"] for name, values in record["work_counts"].items()}
            for name, stats in metrics.items():
                rows.append({**base, **selected(record, ("stress_backend", "level", "complete", "passed", "row_count")),
                             "metric": name, **stats, **work})
        elif kind == "selector":
            rows.extend({**base, **{key: value for key, value in item.items() if not isinstance(value, (dict, list))}}
                        for item in record["cases"])
        elif kind == "components":
            for name, path in record["paths"].items():
                rows.append({**base, "path": name, "complete": record.get("complete"), "reported_passed": record.get("passed"),
                             **{key: value for key, value in path.items() if not isinstance(value, (dict, list))},
                             "ratios_in": "components.json"})
        elif kind == "joint":
            for frame in record["keyframes"]:
                for name in ("coarse", "local", "joint", "mother"):
                    rows.append({**base, "step": frame.get("step"), "path": name,
                                 "numerical_passed": frame.get("numerical_passed"),
                                 "spatial_errors": frame.get("spatial_errors", {}).get(name),
                                 "timing": {key: value for key, value in frame.get("timing", {}).get(name, {}).items()
                                            if not isinstance(value, (dict, list))}})
        elif kind == "profiles":
            rows.extend({**base, **item, "memory_in": "profiles.json"} for item in record.get("device_categories", []))
        elif kind == "geometry":
            rows.append({**base, "name": record.get("name"), "complete": record.get("complete"),
                         "reported_passed": record.get("passed"), "checks": record.get("qualification_checks"),
                         "row_errors": record.get("row_reported_errors"), "partial": record.get("partial_footprint_probe")})
    return rows


def share_metadata(documents):
    """Store repeated configuration objects once, preserving their exact compact values."""
    eligible = {"parameters", "options", "original_options", "source_options", "reconstruction_options",
                "stress_configurations", "identity", "atlas", "memory", "source_hash_inventory", "basis",
                "basis_provenance", "execution", "model_observations", "engine_flags"}
    counts, values = {}, {}

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in eligible and isinstance(item, (dict, list)):
                    encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    if len(encoded) > 160:
                        digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
                        counts[digest] = counts.get(digest, 0) + 1
                        values[digest] = item
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    def replace(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                digest = None
                if key in eligible and isinstance(item, (dict, list)):
                    encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
                result[key] = {"$shared": digest} if counts.get(digest, 0) > 1 else replace(item)
            return result
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value

    collect(documents)
    result = replace(documents)
    result["shared"] = [{"id": digest, "references": counts[digest], "value": replace(values[digest])}
                        for digest in sorted(values) if counts[digest] > 1]
    return result


def summarize(run, output):
    run = run.resolve(strict=True)
    output = output.resolve()
    if output == run or run in output.parents and len(output.relative_to(run).parts) < 2:
        raise ValueError("Use a separate new packet directory, for example RUN/meta/summary-preview.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("The packet directory must be new or empty; existing evidence is never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    packets = {kind: [] for kind in KINDS}
    campaigns = []
    read_errors = []
    parsers = {"accuracy": accuracy, "geometry": geometry, "benchmarks": benchmarks, "profiles": profiles,
               "joint": joint, "components": components, "selector": selector, "controls": controls}
    for directory in sorted(run.iterdir()):
        if not directory.is_dir() or directory.name == "meta" or directory == output or directory.name.startswith("summary"):
            continue
        manifest = campaign_manifest(directory, run)
        campaigns.append(manifest)
        manifest["result_counts"] = {kind: 0 for kind in KINDS}
        manifest["reported_complete_counts"] = {"true": 0, "false": 0, "unknown": 0}
        manifest["incomplete_records"] = []
        for path in sorted(directory.glob("group*/**/*.json")):
            if any(part in SKIP_DIRECTORIES for part in path.relative_to(directory).parts) or output in path.parents:
                continue
            raw = path.read_bytes()
            reference = file_reference(path, run, raw)
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError) as error:
                read_errors.append({"file": reference, "state": "unreadable_snapshot_not_a_final_verdict", "error": str(error)[:200]})
                continue
            kind = classify(data)
            if kind == "unknown":
                summary = {"keys": sorted(data) if isinstance(data, dict) else None,
                           "json_type": type(data).__name__, "classification": "unknown_schema_preserved_by_reference"}
            else:
                summary = parsers[kind](data)
            summary = {"file": reference, "campaign": directory.name, **summary}
            backend = data.get("stress_backend") if isinstance(data, dict) else None
            hints = [name for name in ("shell", "volume") if name in str(path.relative_to(directory)).lower()]
            summary["backend_context"] = {
                "declared": backend, "path_hint": hints[0] if len(hints) == 1 else None,
            }
            packets[kind].append(summary)
            manifest["result_counts"][kind] += 1
            complete = data.get("complete") if isinstance(data, dict) else None
            manifest["reported_complete_counts"]["true" if complete is True else "false" if complete is False else "unknown"] += 1
            if complete is False:
                manifest["incomplete_records"].append(reference["path"])
    metadata_files = [{"file": file_reference(path, run)} for path in sorted((run / "meta").glob("*.json"))]
    documents = {"campaigns": campaigns, "metadata": metadata_files, "read_errors": read_errors, **packets}
    serializable = dict(documents)
    serializable["benchmarks"] = [
        {**{key: value for key, value in item.items() if key != "repeats"},
         "raw_repeats": {"csv": "benchmarks.csv", "case": item["file"]["path"], "count": len(item["repeats"])}}
        for item in packets["benchmarks"]
    ]
    serializable = share_metadata(serializable)
    for name, data in serializable.items():
        lines = [json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for item in data]
        (output / (name + ".json")).write_text("[\n" + ",\n".join(lines) + "\n]\n", encoding="utf-8")
    for kind in ("accuracy", "geometry", "benchmarks", "profiles", "joint", "components", "selector"):
        write_csv(output / (kind + ".csv"), csv_rows(kind, packets[kind]))
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(output.iterdir()) if path.is_file()]
    index = {
        "schema_version": 1, "run": str(run), "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "summarizer": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "counts": {kind: len(records) for kind, records in packets.items()},
        "campaign_count": len(campaigns), "read_error_count": len(read_errors), "files": files,
        "payload_bytes_before_index": sum(item["bytes"] for item in files),
        "interpretation": [
            "No retention or six-cell completion decision is inferred.",
            "Reported passed, numerical, spatial, complete and returncode are separate evidence fields.",
            "A complete spatial rejection can return 1; an incomplete record is not a final negative result.",
            "All means specify their observation population; zero unloaded outputs remain included.",
            "Mother residual is a spatial diagnostic for reduced answers; signed raw loads are not physical-contact qualification.",
            "Source/analysis manifests are referenced, never replaced by a benchmark's unrecorded revision.",
            "JSON inputs may be live snapshots. Each SHA identifies exactly the bytes read; regenerate a new frozen packet after jobs finish.",
        ],
        "excluded": ["Scientific array files", "captured analysis-source trees", "sample/trace/poll payloads", "per-buffer/per-field inventories"],
        "reference_paths": "File-record paths resolve relative to run; absolute paths already recorded by original results are preserved.",
        "shared_metadata": "A $shared value resolves by id in shared.json; repeated options/configurations remain lossless.",
        "csv_provenance": "Join each CSV case to the same file.path in its category JSON for the original SHA and run-relative source path.",
    }
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if total > 2 * 1024 * 1024:
        raise ValueError(f"Packet is {total} bytes, above the 2 MiB budget; inspect compaction before freezing it.")
    return {"output": str(output), "packet_bytes": total, "counts": index["counts"], "read_errors": len(read_errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
