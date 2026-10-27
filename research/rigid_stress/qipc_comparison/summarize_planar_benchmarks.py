"""Audit blocked planar-loading timings using JSON receipts and the Python standard library.

Primary summaries require three complete blocks, one 600-step repetition per process, matched physical inputs and
explicit source revisions. Diagnostic mode retains older artifacts without certifying them as primary measurements.
"""

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

ASSET_NAMES = ("egg_outer.obj", "egg_shell.urdf", "shell.npz", "elastic.npz")
COMMON_SOURCES = (
    "examples/rigid/egg_stress_controller.py",
    "examples/rigid/franka_egg_stress.py",
    "examples/speed_benchmark/rigid_stress.py",
    "genesis/engine/entities/rigid_entity/description.py",
    "genesis/engine/solvers/rigid/rigid_solver.py",
    "genesis/options/morphs.py",
)
COMMON_FIELDS = (
    "envs",
    "output_mode",
    "precision",
    "level",
    "dt",
    "substeps",
    "seed",
    "varied",
    "egg_geometry",
    "collision_method",
    "contact_patch",
    "condition_count",
    "conditions_sha256",
    "contact_model",
    "radius_model",
    "pressure_profile",
    "native_controller",
    "controller_buffers",
    "reset_mode",
    "history",
    "warmup_steps",
    "torch",
    "quadrants",
)
MATERIAL_FIELDS = (
    "young",
    "poisson",
    "density",
    "shell_thickness",
    "shell_drilling_factor",
    "contact_radius",
    "contact_modulus",
    "radius_model",
    "box_edge_policy",
    "pressure_profile",
    "contact_model",
    "source_projection_limit",
    "source_geometry_tolerance",
    "source_uncertainty",
    "tolerance",
    "absolute_tolerance",
)
POLICY_FIELDS = ("layers", "seed", "dtype", "residual_scale_rad", "control_error_budget_rad")
ROUTE_FIELDS = (
    "attempted_loaded_contacts",
    "proxy_contacts",
    "fallback_contacts",
    "analytic_sections",
    "overflow_steps",
    "material_queries",
    "material_rejections",
)


@dataclass(frozen=True)
class TaskSpec:
    group: int
    block: int
    variant: str
    name: str
    scope: str
    batch: int
    output_mode: str


@dataclass
class TaskAudit:
    specification: TaskSpec
    issues: list[str] = field(default_factory=list)
    result_file: dict = field(default_factory=dict)
    returncode: int | None = None
    seconds_samples: list[float] = field(default_factory=list)
    timed_repetitions: list[dict] = field(default_factory=list)
    receipt: dict = field(default_factory=dict)
    route_observations: list[dict] = field(default_factory=list)
    trajectory_metrics: dict | None = None


def file_identity(path: Path) -> dict:
    """Return a file's location, size and SHA256 without decoding its payload."""
    contents = path.read_bytes()
    return {"path": str(path), "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}


def explicit_arguments(arguments: list[str]) -> dict:
    """Parse explicit long options without importing a scientific benchmark module."""
    values = {}
    i_argument = 0
    while i_argument < len(arguments):
        argument = arguments[i_argument]
        if not argument.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {argument}")
        option, separator, value = argument[2:].partition("=")
        option = option.replace("-", "_")
        if option in values:
            raise ValueError(f"Repeated explicit option: {option}")
        if separator:
            values[option] = value
        elif i_argument + 1 < len(arguments) and not arguments[i_argument + 1].startswith("--"):
            i_argument += 1
            values[option] = arguments[i_argument]
        else:
            values[option] = True
        i_argument += 1
    return values


def is_same_argument(expected, actual) -> bool:
    """Compare a command-line literal with its JSON-decoded parser value."""
    if isinstance(actual, bool):
        return expected is actual
    if isinstance(actual, (int, float)):
        try:
            return math.isfinite(actual) and actual == float(expected)
        except (TypeError, ValueError):
            return False
    return expected == actual


def plan_specifications(plan: dict) -> list[TaskSpec]:
    """Resolve flat or grouped raw task order while keeping the plan's block identities."""
    groups = {group.get("group_id", group.get("group")): group for group in plan.get("groups", [])}
    entries = plan.get("raw_task_order")
    if entries is None:
        entries = [entry for group in groups.values() for entry in group["raw_task_order"]]
    specifications = []
    for entry in entries:
        group = groups.get(entry["group"], {})
        scope = entry.get("scope", entry.get("worker_scope", group.get("scope_family")))
        if scope == "policy-rigid":
            scope = "policy"
        specifications.append(
            TaskSpec(
                group=entry["group"],
                block=entry["block"],
                variant=entry["variant"],
                name=entry["name"],
                scope=scope,
                batch=entry.get("batch", group.get("batch")),
                output_mode=entry.get("output_mode", group.get("output_mode")),
            )
        )
    return specifications


def canonical_hashes(hashes: dict, suffixes: tuple[str, ...], issues: list[str], label: str) -> dict:
    """Extract unambiguous hashes independently of checkout directory names."""
    canonical = {}
    for suffix in suffixes:
        matches = [value for name, value in hashes.items() if name == suffix or name.endswith("/" + suffix)]
        if len(matches) != 1:
            issues.append(f"{label}: expected one receipt for {suffix}, found {len(matches)}")
        else:
            value = matches[0]["sha256"] if isinstance(matches[0], dict) else matches[0]
            if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
                issues.append(f"{label}: invalid SHA256 for {suffix}")
            canonical[suffix] = value
    return canonical


def audit_task(specification: TaskSpec, task: dict | None, campaign: Path, source: dict) -> TaskAudit:
    """Read one process's timing, input, source, route and validity receipts."""
    audit = TaskAudit(specification)
    path = campaign / f"group{specification.group}" / specification.name
    try:
        audit.returncode = int(path.with_name(path.name + ".returncode.txt").read_text().strip())
    except (OSError, ValueError) as error:
        audit.issues.append(f"Missing or invalid process return code: {error}")
    if audit.returncode != 0:
        audit.issues.append(f"Process did not terminate successfully: returncode={audit.returncode}")
    try:
        result = json.loads(path.read_text())
        audit.result_file = file_identity(path)
    except (OSError, ValueError) as error:
        audit.issues.append(f"Missing or invalid result JSON: {error}")
        return audit
    if not isinstance(result, dict):
        audit.issues.append("Result JSON is not an object")
        return audit
    if task is None:
        audit.issues.append("Task is absent from captured campaign tasks.json")
        return audit
    if task.get("module") != "examples.speed_benchmark.rigid_stress":
        audit.issues.append("Captured task does not invoke the expected native benchmark module")
    try:
        arguments = explicit_arguments(task["args"])
    except (KeyError, TypeError, ValueError) as error:
        audit.issues.append(f"Invalid captured task arguments: {error}")
        return audit
    actual_arguments = result.get("arguments", {})
    for name, value in arguments.items():
        if name not in actual_arguments or not is_same_argument(value, actual_arguments[name]):
            audit.issues.append(f"Explicit task argument differs from result: {name}")
    revision = arguments.get("source_revision")
    if not isinstance(revision, str) or re.fullmatch("[0-9a-f]{40}", revision) is None:
        audit.issues.append("Task lacks an explicit full source revision")
    if result.get("source_revision") != revision:
        audit.issues.append("Result source_revision differs from explicit task source revision")
    if task.get("source", source.get("source")) == source.get("source") and source.get("revision") != revision:
        audit.issues.append("Explicit task revision differs from captured campaign source revision")
    expected_scope = "policy-rigid" if specification.variant == "pureRigid" else specification.scope
    for name, expected in (
        ("scope", expected_scope),
        ("envs", specification.batch),
        ("output_mode", specification.output_mode),
    ):
        if result.get(name) != expected or actual_arguments.get(name) != expected:
            audit.issues.append(f"Plan/task/result mismatch: {name} expected {expected!r}")
    if specification.scope not in ("recovery", "policy"):
        audit.issues.append(f"Unsupported primary scope: {specification.scope}")
    if result.get("precision") != "64":
        audit.issues.append("Primary physics precision is not 64")
    if not isinstance(result.get("gpu_uuid"), str) or not result["gpu_uuid"].startswith("GPU-"):
        audit.issues.append("GPU UUID receipt is missing")
    if not result.get("host"):
        audit.issues.append("Host receipt is missing")
    if actual_arguments.get("steps") != 600 or actual_arguments.get("repetitions") != 1:
        audit.issues.append("Primary task requires exactly 600 steps and one repetition")
    if actual_arguments.get("minimum_seconds") != 0:
        audit.issues.append("Primary task requires minimum_seconds=0")
    repeats = result.get("repeats", [])
    audit.timed_repetitions = repeats
    if len(repeats) != 1:
        audit.issues.append(f"Expected one timed repetition, found {len(repeats)}")
    for i_repeat, repeat in enumerate(repeats):
        seconds = repeat.get("seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            audit.issues.append(f"Repetition {i_repeat}: elapsed seconds are not finite and positive")
        else:
            audit.seconds_samples.append(seconds)
        if repeat.get("steps") != 600:
            audit.issues.append(f"Repetition {i_repeat}: actual step count is not 600")
        if repeat.get("invalid_environment_steps") != 0:
            audit.issues.append(f"Repetition {i_repeat}: nonzero or missing invalid_environment_steps")
        failed = repeat.get("failed_steps_per_environment")
        if failed is not None and (len(failed) != specification.batch or any(value != 0 for value in failed)):
            audit.issues.append(f"Repetition {i_repeat}: invalid per-environment failure counts")
        overflow = repeat.get("scatter_overflow_calls")
        if type(overflow) is not int or overflow < 0:
            audit.issues.append(f"Repetition {i_repeat}: invalid scatter overflow count")
        routes = repeat.get("planar_routes")
        route_observation = {"repeat": i_repeat, "scatter_overflow_calls": overflow, "planar_routes": routes}
        audit.route_observations.append(route_observation)
        if specification.variant.startswith("planar"):
            if not isinstance(routes, dict) or routes.get("full_interval_counts_available") is not True:
                audit.issues.append(f"Repetition {i_repeat}: complete planar route counts are unavailable")
            else:
                if routes.get("environment_resets_inside_timer") != 0:
                    audit.issues.append(f"Repetition {i_repeat}: reset inside planar timing interval")
                delta = routes.get("per_environment_delta", {})
                totals = {}
                for name in ROUTE_FIELDS:
                    values = delta.get(name, [])
                    if len(values) != specification.batch or any(
                        type(value) is not int or value < 0 for value in values
                    ):
                        audit.issues.append(f"Repetition {i_repeat}: missing/invalid planar route counter {name}")
                    else:
                        totals[name] = sum(values)
                route_observation["totals"] = totals
                if (
                    all(name in totals for name in ROUTE_FIELDS[:3])
                    and totals["attempted_loaded_contacts"] != totals["proxy_contacts"] + totals["fallback_contacts"]
                ):
                    audit.issues.append(f"Repetition {i_repeat}: attempted/proxy/fallback counts do not reconcile")
    conditions = actual_arguments.get("conditions")
    if not isinstance(result.get("conditions_sha256"), str):
        audit.issues.append("Frozen conditions SHA256 is missing")
    elif conditions is None:
        audit.issues.append("Task has no frozen condition bank")
    else:
        try:
            if file_identity(Path(conditions))["sha256"] != result["conditions_sha256"]:
                audit.issues.append("Current frozen condition bytes differ from the result SHA256")
        except OSError as error:
            audit.issues.append(f"Cannot verify frozen condition file: {error}")
    source_hashes = canonical_hashes(result.get("source_sha256", {}), COMMON_SOURCES, audit.issues, "common source")
    assets = canonical_hashes(result.get("physical_assets", {}), ASSET_NAMES, audit.issues, "physical asset")
    policy = result.get("policy", {})
    policy_signature = {name: policy.get(name) for name in POLICY_FIELDS}
    if any(value is None for value in policy_signature.values()):
        audit.issues.append("Policy architecture/seed/precision/control receipt is incomplete")
    policy_error = policy.get("warmup_reference_max_error_rad")
    policy_budget = policy.get("control_error_budget_rad")
    if type(policy_error) not in (int, float) or type(policy_budget) not in (int, float):
        audit.issues.append("Policy warmup reference error or unchanged control budget is missing")
    elif not math.isfinite(policy_error) or not math.isfinite(policy_budget) or not 0 <= policy_error <= policy_budget:
        audit.issues.append("Policy warmup exceeds its unchanged controller error budget")
    material = []
    if specification.variant != "pureRigid":
        configurations = result.get("stress_configurations", [])
        if len(configurations) != 1:
            audit.issues.append("Expected one explicit native stress/material configuration")
        for configuration in configurations:
            if any(name not in configuration for name in MATERIAL_FIELDS):
                audit.issues.append("Native material/acceptance configuration is incomplete")
            material.append({name: configuration.get(name) for name in MATERIAL_FIELDS})
            for name in ("output_mode", "quadrature"):
                if configuration.get(name) != actual_arguments.get(name):
                    audit.issues.append(f"Actual native stress configuration differs from arguments: {name}")
            if configuration.get("backend") != actual_arguments.get("stress_backend"):
                audit.issues.append("Actual native stress backend differs from arguments")
            if configuration.get("backend") == "shell":
                if configuration.get("shell_loading") != actual_arguments.get("shell_loading"):
                    audit.issues.append("Actual shell loading differs from arguments")
                if configuration.get("shell_loading") == "planar" and configuration.get(
                    "loading_subdivisions"
                ) != actual_arguments.get("loading_subdivisions"):
                    audit.issues.append("Actual planar subdivision differs from arguments")
        if specification.variant == "Q4" and actual_arguments.get("shell_loading") != "q4":
            audit.issues.append("Q4 variant does not request original Q4 loading")
        if specification.variant == "volume" and actual_arguments.get("stress_backend") != "volume":
            audit.issues.append("Volume variant does not request volume elasticity")
        if specification.variant.startswith("planar"):
            subdivision = specification.variant.removeprefix("planar")
            if (
                actual_arguments.get("shell_loading") != "planar"
                or str(actual_arguments.get("loading_subdivisions")) != subdivision
            ):
                audit.issues.append("Planar variant differs from actual loading/subdivision options")
    snapshot = result.get("recovery_snapshot_check")
    if specification.scope == "recovery":
        if not isinstance(snapshot, dict):
            audit.issues.append("Module timing lacks an untimed snapshot acceptance receipt")
        else:
            for name in ("normalized_force_error_max", "normalized_moment_error_max"):
                value = snapshot.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1e-8:
                    audit.issues.append(f"Module snapshot fails original 1e-8 source-point gate: {name}")
            physical_source = snapshot.get("physical_source_identity", {})
            if re.fullmatch("[0-9a-f]{64}", str(physical_source.get("sha256"))) is None:
                audit.issues.append("Module snapshot lacks the explicit physical_source_identity SHA256")
            shape = physical_source.get("shape", [])
            if len(shape) != 2 or shape[0] <= 0:
                audit.issues.append("Module physical-source snapshot is empty or has no explicit shape")
    audit.trajectory_metrics = result.get("trajectory_metrics")
    if specification.scope == "policy" and not isinstance(audit.trajectory_metrics, dict):
        audit.issues.append("Policy timing lacks trajectory outcome metrics")
    common = {name: result.get(name) for name in COMMON_FIELDS}
    if any(value is None for value in common.values()):
        audit.issues.append("Common physical/controller/runtime receipt is incomplete")
    audit.receipt = {
        "host": result.get("host"),
        "gpu_uuid": result.get("gpu_uuid"),
        "gpu": result.get("gpu"),
        "slurm_job_id": result.get("slurm_job_id"),
        "source_revision": result.get("source_revision"),
        "task_source": task.get("source", source.get("source")),
        "explicit_task_arguments": arguments,
        "common": common,
        "common_source_sha256": source_hashes,
        "physical_assets": assets,
        "material": material,
        "policy": policy_signature,
        "snapshot_check": snapshot,
        "actual_stress_arguments": {
            name: actual_arguments.get(name)
            for name in (
                "stress_backend",
                "shell_loading",
                "loading_subdivisions",
                "planar_integration",
                "quadrature",
                "pressure_quadrature",
                "scatter_tasks_per_env",
                "method",
                "inverse_precision",
                "cooperative_anchor",
            )
        },
    }
    return audit


def summarize(plan_path: Path, campaign: Path, is_diagnostic: bool) -> dict:
    """Audit every planned process and retain invalid blocks in paired summaries."""
    plan = json.loads(plan_path.read_text())
    specifications = plan_specifications(plan)
    tasks = json.loads((campaign / "tasks.json").read_text())
    source = json.loads((campaign / "source.json").read_text())
    task_map = {(i_group, task["name"]): task for i_group, group in enumerate(tasks) for task in group}
    audits = [
        audit_task(specification, task_map.get((specification.group, specification.name)), campaign, source)
        for specification in specifications
    ]
    issues = []
    if len(task_map) != sum(len(group) for group in tasks):
        issues.append("Captured campaign repeats a task identity")
    identities = [(specification.group, specification.name) for specification in specifications]
    if len(identities) != len(set(identities)):
        issues.append("Plan repeats a process identity")
    if set(identities) != set(task_map):
        issues.append(
            "Plan and captured campaign task identities differ; no unplanned or missing process may be dropped"
        )
    blocks = []
    cohorts = []
    for group in sorted({specification.group for specification in specifications}):
        group_audits = [audit for audit in audits if audit.specification.group == group]
        group_blocks = []
        for block in sorted({audit.specification.block for audit in group_audits}):
            members = [audit for audit in group_audits if audit.specification.block == block]
            block_issues = [f"{audit.specification.name}: {issue}" for audit in members for issue in audit.issues]
            variants = [audit.specification.variant for audit in members]
            planar_variants = [variant for variant in variants if variant.startswith("planar")]
            scopes = {audit.specification.scope for audit in members}
            expected = {"Q4", "volume", *planar_variants}
            if "policy" in scopes:
                expected.add("pureRigid")
            if len(planar_variants) != 1 or len(variants) != len(set(variants)) or set(variants) != expected:
                block_issues.append("Block lacks exactly one member of every required variant")
            if len(scopes) != 1:
                block_issues.append("Block mixes module and policy timing scopes")
            for field_name in ("host", "gpu_uuid", "common", "common_source_sha256", "physical_assets", "policy"):
                values = [audit.receipt.get(field_name) for audit in members]
                if not values or any(value is None for value in values) or any(value != values[0] for value in values):
                    block_issues.append(f"Paired members differ or lack receipt: {field_name}")
            materials = [
                audit.receipt.get("material") for audit in members if audit.specification.variant != "pureRigid"
            ]
            if materials and any(material != materials[0] for material in materials):
                block_issues.append("Paired stress members differ in material or acceptance configuration")
            if scopes == {"recovery"}:
                source_identities = [
                    (audit.receipt.get("snapshot_check") or {}).get("physical_source_identity") for audit in members
                ]
                signatures = [
                    {name: identity.get(name) for name in ("sha256", "shape", "dtype", "columns")}
                    if isinstance(identity, dict)
                    else None
                    for identity in source_identities
                ]
                if any(value is None for value in signatures) or any(value != signatures[0] for value in signatures):
                    block_issues.append(
                        "Module physical sources are not explicitly identical; slot/anchor hashes are not substitutes"
                    )
            times = {
                audit.specification.variant: statistics.median(audit.seconds_samples)
                for audit in members
                if audit.seconds_samples
            }
            ratios = {
                f"{numerator}/{denominator}": times[numerator] / times[denominator]
                for numerator, denominator in itertools.permutations(sorted(times), 2)
            }
            block_record = {
                "group": group,
                "block": block,
                "scope": members[0].specification.scope,
                "batch": members[0].specification.batch,
                "output_mode": members[0].specification.output_mode,
                "raw_order": [audit.specification.name for audit in members],
                "issues": block_issues,
                "metadata_checks_passed": not block_issues,
                "primary_qualified": not is_diagnostic and not block_issues,
                "raw_elapsed_seconds_median_of_internal_repeats": times,
                "raw_elapsed_ratios_numerator_over_denominator": ratios,
                "qualified_elapsed_ratios_numerator_over_denominator": ratios
                if not is_diagnostic and not block_issues
                else None,
            }
            blocks.append(block_record)
            group_blocks.append(block_record)
        cohort_issues = []
        if len(group_blocks) != 3:
            cohort_issues.append(f"Expected exactly three independent process blocks, found {len(group_blocks)}")
        for name in ("scope", "batch", "output_mode"):
            if len({block[name] for block in group_blocks}) != 1:
                cohort_issues.append(f"Group mixes {name}")
        for name in ("host", "gpu_uuid", "common", "common_source_sha256", "physical_assets", "policy"):
            values = [audit.receipt.get(name) for audit in group_audits]
            if any(value is None for value in values) or any(value != values[0] for value in values):
                cohort_issues.append(f"Group receipts differ across blocks: {name}")
        for variant in {audit.specification.variant for audit in group_audits}:
            variant_audits = [audit for audit in group_audits if audit.specification.variant == variant]
            for name in ("source_revision", "task_source", "actual_stress_arguments", "material"):
                values = [audit.receipt.get(name) for audit in variant_audits]
                if any(value is None for value in values) or any(value != values[0] for value in values):
                    cohort_issues.append(f"Variant {variant} differs across blocks: {name}")
        ratio_names = sorted(
            {name for block in group_blocks for name in block["raw_elapsed_ratios_numerator_over_denominator"]}
        )
        statistics_rows = []
        for name in ratio_names:
            samples = [
                {
                    "block": block["block"],
                    "value": block["raw_elapsed_ratios_numerator_over_denominator"].get(name),
                    "metadata_checks_passed": block["metadata_checks_passed"],
                }
                for block in group_blocks
            ]
            is_complete = len(samples) == 3 and all(sample["value"] is not None for sample in samples)
            is_qualified = (
                is_complete and not cohort_issues and all(block["primary_qualified"] for block in group_blocks)
            )
            values = [sample["value"] for sample in samples] if is_complete else []
            statistics_rows.append(
                {
                    "elapsed_ratio_numerator_over_denominator": name,
                    "raw_blocks_including_failures": samples,
                    "all_three_blocks_qualified": is_qualified,
                    "qualified_median": statistics.median(values) if is_qualified else None,
                    "qualified_range": [min(values), max(values)] if is_qualified else None,
                    "diagnostic_observed_median": statistics.median(values) if is_complete else None,
                    "diagnostic_observed_range": [min(values), max(values)] if is_complete else None,
                }
            )
        cohorts.append({"group": group, "issues": cohort_issues, "ratios": statistics_rows})
    is_metadata_valid = (
        bool(blocks)
        and not issues
        and all(block["metadata_checks_passed"] for block in blocks)
        and all(not cohort["issues"] for cohort in cohorts)
    )
    return {
        "schema": "genesis.planar-benchmark-summary.v1",
        "diagnostic_only": is_diagnostic,
        "metadata_checks_passed": is_metadata_valid,
        "primary_qualified": is_metadata_valid and not is_diagnostic,
        "plan": file_identity(plan_path),
        "campaign": str(campaign),
        "captured_tasks": file_identity(campaign / "tasks.json"),
        "captured_source": file_identity(campaign / "source.json"),
        "issues": issues,
        "tasks": [asdict(audit) for audit in audits],
        "blocks": blocks,
        "cohorts": cohorts,
        "interpretation": {
            "ratios": "Elapsed numerator/denominator for the same 600-step batch. Q4/planar32 is the planar speedup relative to original Q4. Ratios of unqualified blocks are diagnostic only.",
            "repeats": "Primary estimates use exactly three fresh-process blocks. Median and observed range describe these three pairs; no confidence interval is inferred. Failed pairs remain in raw_blocks_including_failures and prevent a qualified aggregate.",
            "module": "Identical sorted physical source SHA256, shape, dtype and columns are mandatory. Contact-slot and original-anchor hashes are retained in each snapshot and are not required to match independently anchored volume elasticity.",
            "policy": "Initial condition bytes, rigid/controller code, assets, material, seed and multilayer perceptron architecture/seed/precision match. Stress feedback may change later actions and contacts, so policy rates describe initial-condition-matched closed-loop workflows; their future physical sources need not be identical.",
            "acceptance": "Successful process exit retains the benchmark's native errno, original unscaled residual and source-point acceptance. This metadata tool does not replace independent spatial qualification. Fallback and overflow counts are reported and their work remains inside elapsed time.",
        },
    }


def finite_json(value):
    """Keep nonfinite failed-artifact values visible in standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_raw_value": str(value)}
    if isinstance(value, dict):
        return {name: finite_json(item) for name, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Report unsupported older receipts without certifying primary timing; exit zero after writing the report.",
    )
    args = parser.parse_args()
    summary = summarize(args.plan, args.campaign, args.diagnostic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "primary_qualified": summary["primary_qualified"],
                "metadata_checks_passed": summary["metadata_checks_passed"],
                "tasks": len(summary["tasks"]),
                "blocks": len(summary["blocks"]),
            }
        )
    )
    if not args.diagnostic and not summary["primary_qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
