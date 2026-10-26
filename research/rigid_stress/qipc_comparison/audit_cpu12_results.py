"""Audit saved CPU12 parity fields, resultants, boundaries and spatial residuals.

This postprocessor reads immutable artifacts only. It never assembles an
operator, solves a system, changes a displacement, or imports the engine.
Spatial rejection is reported separately from CPU/GPU numerical equivalence.
Scientific imports require Slurm rtx-mid and image 1_26.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from .artifacts import file_id, runtime_identity, write_json

STRATEGIES = ("continuous", "nearest", "narrow")
BANKS = ("published43", "boundary18", "static16", "static64")
REPOSITORY = Path(__file__).resolve().parents[3]
BUDGETS = dict(field=0.10, peak=0.05, N=0.10, M=0.10, Q=0.10, mean_loaded_field=0.05)


def _api():
    import numpy as np
    import scipy
    from scipy import sparse
    from threadpoolctl import threadpool_limits

    return SimpleNamespace(**locals())


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


class Checks:
    def __init__(self, api):
        self.np, self.rows = api.np, []

    def add(self, name, passed, *, category="consistency", gating=True, **details):
        self.rows.append(dict(name=name, passed=bool(passed), category=category, gating=gating, **details))

    def close(self, name, actual, expected, *, rtol=1e-8, atol=1e-13, category="equivalence"):
        np = self.np
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            self.add(
                name, False, category=category, actual_shape=list(actual.shape), expected_shape=list(expected.shape)
            )
            return
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        difference, scale = np.linalg.norm(actual - expected), np.linalg.norm(expected)
        finite = finite and bool(np.isfinite(difference) and np.isfinite(scale))
        self.add(
            name,
            finite and difference <= atol + rtol * scale,
            category=category,
            absolute_l2=_number(difference),
            reference_l2=_number(scale),
            relative_l2=_number(difference / max(scale, 1e-30)),
            rtol=rtol,
            absolute_l2_floor=atol,
        )

    def passed(self):
        return all(row["passed"] for row in self.rows if row["gating"])


class Reader:
    def __init__(self, api, root, inventory, checks, label):
        self.api, self.root, self.expected, self.checks, self.label = api, root, inventory, checks, label
        self.consumed = {}

    def verify(self, relative):
        path = self.root / relative
        identity = file_id(path)
        expected = self.expected[relative]
        self.checks.add(
            f"identity/{self.label}/{relative}",
            identity["sha256"] == expected["sha256"] and identity["bytes"] == expected["bytes"],
            category="identity",
            actual_sha256=identity["sha256"],
            expected_sha256=expected["sha256"],
        )
        self.consumed[relative] = identity
        return path

    def arrays(self, relative):
        with self.api.np.load(self.verify(relative), allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}

    def json(self, relative):
        return json.loads(self.verify(relative).read_text())

    def matrix(self, relative):
        return self.api.sparse.load_npz(self.verify(relative)).tocsr()

    def unchanged(self):
        for relative, identity in self.consumed.items():
            self.checks.add(
                f"unchanged/{self.label}/{relative}", file_id(self.root / relative) == identity, category="identity"
            )


class Output:
    def __init__(self, api, root):
        self.np, self.root, self.files = api.np, root, {}

    def arrays(self, relative, **values):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: self.np.asarray(value) for key, value in values.items()}
        if any(value.dtype.hasobject for value in arrays.values()):
            raise TypeError("Audit arrays must not contain objects or pickle.")
        self.np.savez_compressed(path, **arrays)
        self.files[relative] = file_id(path)

    def json(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
        self.files[relative] = file_id(path)


def _reference_bank(reader, bank):
    alias_path = f"oracles/{bank}/alias.npz"
    if alias_path in reader.expected:
        alias = reader.arrays(alias_path)
        return str(alias["reference_bank"]), alias["case_indices"]
    return bank, None


def _reference_output(reader, bank, strategy):
    original, indices = _reference_bank(reader, bank)
    values = reader.arrays(f"oracles/{original}/{strategy}/output.npz")
    if indices is not None:
        values = {key: value[..., indices] for key, value in values.items()}
    return values


def _reference_cases(reader, bank, relative):
    original, indices = _reference_bank(reader, bank)
    values = reader.arrays(f"oracles/{original}/{relative}")
    case_keys = (
        "groups_per_case",
        "contributions_per_case",
        "relative_field_error",
        "relative_peak_error",
        "grouped_vs_source_superposition",
    )
    # weights/group CSR fields use source/group axes and must never be sliced
    # with environment aliases from a static bank.
    values = {key: values[key] for key in case_keys}
    if indices is not None:
        values = {key: value[indices] for key, value in values.items()}
    return values


def _weighted_error(np, actual, reference, weights):
    """Leading axes are samples/components; the final axis is environment."""
    if actual.shape != reference.shape or actual.shape[:-1] != weights.shape:
        raise ValueError(f"Weighted norm layout mismatch: {actual.shape}, {reference.shape}, {weights.shape}.")
    scale = weights[..., None]
    difference = np.linalg.norm((scale * (actual - reference)).reshape((-1, actual.shape[-1])), axis=0)
    reference_norm = np.linalg.norm((scale * reference).reshape((-1, reference.shape[-1])), axis=0)
    return difference / np.maximum(reference_norm, 1e-30), difference, reference_norm


def _metrics(np, actual, mother, weights):
    metrics, absolute, reference_norm = {}, {}, {}
    for name, key in (("field", "stress_local"), ("N", "N"), ("M", "M"), ("Q", "Q")):
        metrics[name], absolute[name], reference_norm[name] = _weighted_error(
            np, actual[key], mother[key], weights[name]
        )
    absolute["peak"] = np.abs(actual["peak"] - mother["peak"])
    reference_norm["peak"] = np.abs(mother["peak"])
    metrics["peak"] = absolute["peak"] / np.maximum(reference_norm["peak"], 1e-30)
    return metrics, absolute, reference_norm


def _layouts(np, output, n_elements, n_cases):
    expected = {
        "stress_local": (2 * n_elements * 4 * 3, n_cases),
        "stress_full9": (2 * n_elements * 4, 9, n_cases),
        "N": (n_elements, 4, 3, n_cases),
        "M": (n_elements, 4, 3, n_cases),
        "Q": (n_elements, 4, 2, n_cases),
        "peak": (n_cases,),
    }
    for name, shape in expected.items():
        if output[name].shape != shape:
            raise ValueError(f"{name} has layout {output[name].shape}; declared _outputs layout is {shape}.")
        if not np.isfinite(output[name]).all():
            raise ArithmeticError(f"Saved {name} includes nonfinite entries.")


def _summary(np, metrics, loaded):
    maximum = {key: float(values[loaded].max(initial=0.0)) for key, values in metrics.items()}
    mean_field = float(metrics["field"][loaded].mean()) if loaded.any() else 0.0
    return dict(
        loaded_cases=int(loaded.sum()),
        total_cases=len(loaded),
        maximum_errors=maximum,
        mean_loaded_field_error=mean_field,
        casewise_budget_passed=all(maximum[key] <= BUDGETS[key] for key in metrics),
        mean_field_budget_passed=mean_field <= BUDGETS["mean_loaded_field"],
        passed=all(maximum[key] <= BUDGETS[key] for key in metrics) and mean_field <= BUDGETS["mean_loaded_field"],
    )


def _numerical_equivalence(np, checks, prefix, actual, expected, weights):
    error, absolute, reference_norm = _metrics(np, actual, expected, weights)
    # This repeats the parity driver's declared whole-array tolerances without
    # substituting spatial approximation budgets for numerical acceptance.
    for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
        checks.close(f"{prefix}/{key}", actual[key], expected[key])
    return error, absolute, reference_norm


def _bank_strategy(api, reference, native, output, checks, bank, strategy, weights, matrix, free, n_elements):
    np = api.np
    prefix = f"{bank}/{strategy}"
    cases = reference.arrays(f"cases/{bank}.npz")
    loaded = np.diff(cases["case_offsets"]) > 0
    n_cases = len(cases["case_names"])
    actual = native.arrays(f"{prefix}/output.npz")
    mother = native.arrays(f"{prefix}/mother/output.npz")
    cpu = _reference_output(reference, bank, strategy)
    cpu_mother = _reference_output(reference, bank, "mother")
    for values in (actual, mother, cpu, cpu_mother):
        _layouts(np, values, n_elements, n_cases)
    checks.add(f"{prefix}/native_valid", actual["valid"].all(), category="equivalence")
    checks.add(f"{prefix}/matching_mother_valid", mother["valid"].all(), category="equivalence")
    checks.add(
        f"{prefix}/identical_current_force", np.array_equal(actual["force"], mother["force"]), category="equivalence"
    )
    gpu_error, gpu_absolute, gpu_norm = _metrics(np, actual, mother, weights)
    cpu_error, cpu_absolute, cpu_norm = _metrics(np, cpu, cpu_mother, weights)
    equivalence, equivalence_absolute, equivalence_norm = _numerical_equivalence(
        np, checks, f"{prefix}/cpu_gpu", actual, cpu, weights
    )
    mother_equivalence, _, _ = _numerical_equivalence(
        np, checks, f"{prefix}/mother_cpu_gpu", mother, cpu_mother, weights
    )
    recorded_spatial = native.arrays(f"{prefix}/spatial.npz")
    cpu_selection = _reference_cases(reference, bank, f"{strategy}/selection.npz")
    native_selection = native.arrays(f"{prefix}/selection.npz")
    for key in ("groups_per_case", "contributions_per_case"):
        checks.add(f"{prefix}/{key}", np.array_equal(native_selection[key], cpu_selection[key]), category="equivalence")
    # Selection archives have source-indexed arrays as well as case-indexed
    # arrays. Only the explicitly case-indexed statistics are used here.
    checks.close(
        f"{prefix}/saved_spatial_field",
        gpu_error["field"],
        recorded_spatial["relative_field_error"],
        rtol=1e-12,
        atol=1e-13,
        category="consistency",
    )
    checks.close(
        f"{prefix}/saved_spatial_peak",
        gpu_error["peak"],
        recorded_spatial["relative_peak_error"],
        rtol=1e-12,
        atol=1e-13,
        category="consistency",
    )
    checks.close(
        f"{prefix}/cpu_reported_field",
        cpu_error["field"],
        cpu_selection["relative_field_error"],
        rtol=1e-12,
        atol=1e-13,
        category="consistency",
    )
    checks.close(
        f"{prefix}/cpu_reported_peak",
        cpu_error["peak"],
        cpu_selection["relative_peak_error"],
        rtol=1e-12,
        atol=1e-13,
        category="consistency",
    )
    common = native.arrays(f"{prefix}/common_load.npz")
    original, case_indices = _reference_bank(reference, bank)
    cpu_loads = reference.arrays(f"oracles/{original}/loads.npz")
    cpu_balanced = cpu_loads["balanced_environment"]
    if case_indices is not None:
        cpu_balanced = cpu_balanced[:, case_indices]
    checks.close(f"{prefix}/common_balanced_cpu_gpu", common["balanced"], cpu_balanced, rtol=1e-10, atol=1e-14)
    checks.close(f"{prefix}/mother_rhs_common_balanced", mother["rhs"], common["balanced"], rtol=1e-10, atol=1e-14)
    residual = (matrix @ actual["q"] - common["balanced"])[free]
    cpu_residual = (matrix @ cpu["q"] - cpu_balanced)[free]
    mother_residual = (matrix @ mother["q"] - common["balanced"])[free]
    rhs_norm = np.linalg.norm(common["balanced"][free], axis=0)
    cpu_rhs_norm = np.linalg.norm(cpu_balanced[free], axis=0)
    residual_norm = np.linalg.norm(residual, axis=0)
    residual_relative = residual_norm / np.maximum(rhs_norm, 1e-30)
    cpu_residual_relative = np.linalg.norm(cpu_residual, axis=0) / np.maximum(cpu_rhs_norm, 1e-30)
    mother_residual_relative = np.linalg.norm(mother_residual, axis=0) / np.maximum(rhs_norm, 1e-30)
    checks.add(
        f"{prefix}/mother_numerical_residual",
        float(mother_residual_relative.max(initial=0.0)) <= 1e-8,
        category="equivalence",
        maximum=float(mother_residual_relative.max(initial=0.0)),
        tolerance=1e-8,
    )
    summary = dict(
        bank=bank,
        strategy=strategy,
        native=_summary(np, gpu_error, loaded),
        cpu=_summary(np, cpu_error, loaded),
        mean_loaded_groups=float(native_selection["groups_per_case"][loaded].mean()) if loaded.any() else 0.0,
        mean_contributions_per_source=float(
            (native_selection["contributions_per_case"][loaded] / np.diff(cases["case_offsets"])[loaded]).mean()
        )
        if loaded.any()
        else 0.0,
        maximum_mother_spatial_residual=float(residual_relative.max(initial=0.0)),
        maximum_cpu_mother_spatial_residual=float(cpu_residual_relative.max(initial=0.0)),
        maximum_matching_mother_numerical_residual=float(mother_residual_relative.max(initial=0.0)),
    )
    checks.add(
        f"{prefix}/spatial_budgets_native",
        summary["native"]["passed"],
        category="spatial",
        gating=False,
        summary=summary["native"],
    )
    checks.add(
        f"{prefix}/spatial_budgets_cpu",
        summary["cpu"]["passed"],
        category="spatial",
        gating=False,
        summary=summary["cpu"],
    )
    rows = []
    arrays = dict(
        case_names=cases["case_names"],
        case_families=cases["case_families"],
        source_counts=np.diff(cases["case_offsets"]),
        loaded=loaded,
        groups_per_case=native_selection["groups_per_case"],
        contributions_per_case=native_selection["contributions_per_case"],
        mother_residual_free=residual,
        cpu_mother_residual_free=cpu_residual,
        mother_spatial_residual_norm=residual_norm,
        mother_spatial_rhs_norm=rhs_norm,
        mother_spatial_residual_relative=residual_relative,
        cpu_mother_spatial_residual_relative=cpu_residual_relative,
        matching_mother_numerical_residual_relative=mother_residual_relative,
    )
    for key in gpu_error:
        arrays.update(
            {
                f"native_{key}_error": gpu_error[key],
                f"cpu_{key}_error": cpu_error[key],
                f"native_{key}_absolute_weighted_error": gpu_absolute[key],
                f"native_{key}_reference_weighted_norm": gpu_norm[key],
                f"cpu_{key}_absolute_weighted_error": cpu_absolute[key],
                f"cpu_{key}_reference_weighted_norm": cpu_norm[key],
                f"cpu_gpu_{key}_error": equivalence[key],
                f"cpu_gpu_{key}_absolute_error": equivalence_absolute[key],
                f"cpu_gpu_{key}_reference_norm": equivalence_norm[key],
                f"mother_cpu_gpu_{key}_error": mother_equivalence[key],
            }
        )
    for index in range(n_cases):
        native_values = {key: float(value[index]) for key, value in gpu_error.items()}
        cpu_values = {key: float(value[index]) for key, value in cpu_error.items()}
        rows.append(
            dict(
                index=index,
                case=str(cases["case_names"][index]),
                family=str(cases["case_families"][index]),
                loaded=bool(loaded[index]),
                sources=int(np.diff(cases["case_offsets"])[index]),
                native_errors=native_values,
                cpu_errors=cpu_values,
                groups=int(native_selection["groups_per_case"][index]),
                contributions=int(native_selection["contributions_per_case"][index]),
                native_budget_passed={key: value <= BUDGETS[key] for key, value in native_values.items()},
                cpu_budget_passed={key: value <= BUDGETS[key] for key, value in cpu_values.items()},
                cpu_gpu_relative_errors={key: float(value[index]) for key, value in equivalence.items()},
                mother_spatial_residual=float(residual_relative[index]),
                cpu_mother_spatial_residual=float(cpu_residual_relative[index]),
                matching_mother_numerical_residual=float(mother_residual_relative[index]),
            )
        )
    for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
        checks.add(
            f"{prefix}/release_zero/{key}", np.count_nonzero(actual[key][..., ~loaded]) == 0, category="equivalence"
        )
    output.arrays(f"{prefix}/casewise.npz", **arrays)
    output.json(f"{prefix}/casewise.json", rows)
    output.json(f"{prefix}/summary.json", summary)
    boundary = None
    if bank == "boundary18":
        boundary = dict(cases=cases, native=actual, mother=mother, cpu=cpu, cpu_mother=cpu_mother)
    return summary, boundary


def _boundary(api, reference, output, checks, saved, weights):
    np = api.np
    published = reference.json("oracles/boundary18/boundary_jumps.json")
    rows = []
    for strategy, values in saved.items():
        cases = values["cases"]
        for left, right in cases["pair_case_indices"]:
            left, right = int(left), int(right)
            family, epsilon = str(cases["case_families"][left]), float(cases["epsilon"][left])
            checks.add(
                f"boundary18/pair/{left}-{right}/{strategy}",
                cases["case_families"][left] == cases["case_families"][right]
                and cases["epsilon"][left] == cases["epsilon"][right]
                and cases["crossing_side"][left] == 0
                and cases["crossing_side"][right] == 1,
                category="identity",
            )
            fields = {}
            for label, field, mother in (
                ("native", values["native"], values["mother"]),
                ("cpu", values["cpu"], values["cpu_mother"]),
            ):
                field_jump, _, _ = _weighted_error(
                    np,
                    field["stress_local"][:, right : right + 1],
                    field["stress_local"][:, left : left + 1],
                    weights["field"],
                )
                mother_jump, _, _ = _weighted_error(
                    np,
                    mother["stress_local"][:, right : right + 1],
                    mother["stress_local"][:, left : left + 1],
                    weights["field"],
                )
                scale = max(float((mother["peak"][left] + mother["peak"][right]) / 2.0), 1e-30)
                fields[label] = dict(
                    relative_field_jump=float(field_jump[0]),
                    relative_sampled_peak_jump=float(abs(field["peak"][right] - field["peak"][left]) / scale),
                    reference_relative_field_change=float(mother_jump[0]),
                    reference_relative_peak_change=float(abs(mother["peak"][right] - mother["peak"][left]) / scale),
                )
            matches = [
                row
                for row in published
                if row["boundary"] == family and row["strategy"] == strategy and row["epsilon_rad"] == epsilon
            ]
            checks.add(f"boundary18/oracle_row/{strategy}/{family}/{epsilon}", len(matches) == 1, category="identity")
            if len(matches) != 1:
                raise ValueError("Expected exactly one matching frozen boundary-jump oracle row.")
            for key in ("relative_field_jump", "relative_sampled_peak_jump", "reference_relative_field_change"):
                checks.close(
                    f"boundary18/frozen_cpu/{strategy}/{family}/{epsilon}/{key}",
                    fields["cpu"][key],
                    matches[0][key],
                    rtol=1e-10,
                    atol=1e-13,
                    category="consistency",
                )
                checks.close(
                    f"boundary18/cpu_gpu/{strategy}/{family}/{epsilon}/{key}",
                    fields["native"][key],
                    fields["cpu"][key],
                    rtol=1e-8,
                    atol=1e-10,
                )
            rows.append(
                dict(
                    boundary=family, strategy=strategy, epsilon_rad=epsilon, left_case=left, right_case=right, **fields
                )
            )
    convergence = []
    for strategy in STRATEGIES:
        for family in sorted({row["boundary"] for row in rows}):
            current = sorted(
                (row for row in rows if row["strategy"] == strategy and row["boundary"] == family),
                key=lambda row: row["epsilon_rad"],
                reverse=True,
            )
            eps = [row["epsilon_rad"] for row in current]
            native_jumps = [row["native"]["relative_field_jump"] for row in current]
            cpu_jumps = [row["cpu"]["relative_field_jump"] for row in current]
            checks.add(
                f"boundary18/epsilon_population/{strategy}/{family}", eps == [1e-4, 1e-5, 1e-6], category="identity"
            )
            orders = [
                math.log(max(native_jumps[i], 1e-30) / max(native_jumps[i + 1], 1e-30)) / math.log(eps[i] / eps[i + 1])
                for i in range(len(current) - 1)
            ]
            decreases = all(native_jumps[i + 1] < native_jumps[i] for i in range(len(current) - 1))
            nearest_boundary = strategy == "nearest" and family in ("nearest_pair_boundary", "triple_junction")
            preserved = (
                nearest_boundary and native_jumps[-1] > 0.5 * native_jumps[0] and native_jumps[-1] > 0.5 * cpu_jumps[-1]
            )
            row = dict(
                boundary=family,
                strategy=strategy,
                epsilon_rad=eps,
                native_field_jump=native_jumps,
                cpu_field_jump=cpu_jumps,
                observed_field_orders=orders,
                field_decreases=decreases,
                nearest_jump_preserved=preserved if nearest_boundary else None,
                note="Nearest pair/triple nonzero limiting jumps are retained; no continuity requirement is imposed on nearest.",
            )
            if strategy in ("continuous", "narrow"):
                checks.add(
                    f"boundary18/decreasing_field/{strategy}/{family}",
                    decreases,
                    category="boundary_behavior",
                    gating=False,
                )
            if nearest_boundary:
                checks.add(
                    f"boundary18/nearest_jump_preserved/{family}", preserved, category="boundary_behavior", gating=False
                )
            convergence.append(row)
    output.json("boundary18/boundary_jumps.json", rows)
    output.json("boundary18/convergence.json", convergence)
    output.arrays(
        "boundary18/boundary_jumps.npz",
        boundary=np.array([row["boundary"] for row in rows]),
        strategy=np.array([row["strategy"] for row in rows]),
        epsilon_rad=np.array([row["epsilon_rad"] for row in rows]),
        native_field_jump=np.array([row["native"]["relative_field_jump"] for row in rows]),
        cpu_field_jump=np.array([row["cpu"]["relative_field_jump"] for row in rows]),
        native_peak_jump=np.array([row["native"]["relative_sampled_peak_jump"] for row in rows]),
        cpu_peak_jump=np.array([row["cpu"]["relative_sampled_peak_jump"] for row in rows]),
    )
    return convergence


def audit(api, args, result):
    np = api.np
    checks, output = Checks(api), Output(api, args.output)
    reference_manifest = json.loads((args.reference / "manifest.json").read_text())
    native_result = json.loads((args.native / "result.json").read_text())
    reference = Reader(api, args.reference, reference_manifest["files"], checks, "reference")
    native = Reader(api, args.native, native_result["files"], checks, "native")
    start = native.json("start.json")
    native_checks = native.json("checks.json")
    checks.add(
        "identity/reference_schema", reference_manifest["schema"] == "genesis.cpu12-reference.v1", category="identity"
    )
    checks.add(
        "identity/reference_passed",
        reference_manifest["passed"] and reference_manifest["source_unchanged"],
        category="identity",
    )
    for path, identity in reference_manifest["source_files"].items():
        checks.add(
            f"identity/primary/{path}",
            identity["matches_primary"] and identity["sha256"] == identity["primary_sha256"],
            category="identity",
        )
    reference_hash = file_id(args.reference / "manifest.json")["sha256"]
    checks.add(
        "identity/matching_reference",
        start["reference_manifest_sha256"] == reference_hash
        and native_result["reference_manifest_sha256"] == reference_hash,
        category="identity",
    )
    checks.add(
        "identity/all_banks_stage", start["stage"] == "all" and tuple(start["banks"]) == BANKS, category="identity"
    )
    checks.add(
        "identity/native_sources_start_end", start["source_files"] == native_result["source_files"], category="identity"
    )
    for path, digest in start["source_files"].items():
        available = Path(path).is_file()
        checks.add(
            f"identity/native_source/{path}",
            available and file_id(path)["sha256"] == digest,
            category="identity",
            available=available,
            expected_sha256=digest,
        )
    checks.add(
        "numerical/native_driver_passed",
        native_result["passed"],
        category="equivalence",
        failed=native_result["failed"],
    )
    for row in native_checks:
        checks.add(f"native_check/{row['name']}", row["passed"], category="equivalence")
    numerics = reference.arrays("oracles/operators/mother/numerics.npz")
    area = numerics["gauss_weights"]
    if area.ndim != 2 or area.shape[1] != 4:
        raise ValueError(f"Expected gauss_weights[element,4], got {area.shape}.")
    n_elements = len(area)
    weights = dict(
        field=numerics["field_weights"],
        N=np.sqrt(area[..., None] * np.array([1.0, 1.0, 2.0])),
        M=np.sqrt(area[..., None] * np.array([1.0, 1.0, 2.0])),
        Q=np.sqrt(np.broadcast_to(area[..., None], (*area.shape, 2))),
    )
    matrix = reference.matrix("oracles/operators/mother/K.npz")
    free = numerics["free"]
    summaries, boundary = [], {}
    for bank in BANKS:
        for strategy in STRATEGIES:
            summary, saved_boundary = _bank_strategy(
                api, reference, native, output, checks, bank, strategy, weights, matrix, free, n_elements
            )
            summaries.append(summary)
            if saved_boundary is not None:
                boundary[strategy] = saved_boundary
            output.json("checks.json", checks.rows)
            output.json("summaries.json", summaries)
    convergence = _boundary(api, reference, output, checks, boundary, weights)
    reference.unchanged()
    native.unchanged()
    for name, path, identity in (
        ("reference_manifest", args.reference / "manifest.json", result["reference_manifest"]),
        ("native_result", args.native / "result.json", result["native_result"]),
        ("audit_worker", Path(__file__), result["worker"]),
    ):
        checks.add(f"identity/unchanged_anchor/{name}", file_id(path) == identity, category="identity")
    output.json("checks.json", checks.rows)
    output.json("summaries.json", summaries)
    spatial_passed = all(row["native"]["passed"] for row in summaries)
    cpu_spatial_passed = all(row["cpu"]["passed"] for row in summaries)
    result.update(
        complete=True,
        passed=checks.passed(),
        cpu_gpu_equivalence_passed=all(
            row["passed"] for row in checks.rows if row["category"] in ("identity", "equivalence")
        ),
        spatial_passed=spatial_passed,
        cpu_spatial_passed=cpu_spatial_passed,
        spatial_decision="qualified_within_declared_budgets"
        if spatial_passed
        else "rejected_within_this_cpu12_feature_space",
        spatial_failure_preserves_equivalence=True,
        summaries=summaries,
        boundary_convergence=convergence,
        failed_consistency_checks=[row["name"] for row in checks.rows if row["gating"] and not row["passed"]],
        failed_spatial_checks=[
            row["name"] for row in checks.rows if row["category"] == "spatial" and not row["passed"]
        ],
        inputs=dict(reference=reference.consumed, native=native.consumed),
        files=output.files,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific postprocessing requires Slurm rtx-mid with GENESIS_IMAGE_VER=1_26.")
    args.reference, args.native, args.output = (
        path.expanduser().resolve() for path in (args.reference, args.native, args.output)
    )
    if (
        args.output.is_relative_to(REPOSITORY)
        or args.output.is_relative_to(args.reference)
        or args.output.is_relative_to(args.native)
    ):
        parser.error("Use a separate project data output directory outside all source/input trees.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh output directory; existing measurements are immutable.")
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(
        schema="genesis.cpu12-results-audit.v1",
        complete=False,
        passed=False,
        identity=runtime_identity(require_gpu_inventory=False),
        worker=file_id(Path(__file__)),
        reference_manifest=file_id(args.reference / "manifest.json"),
        native_result=file_id(args.native / "result.json"),
        budgets=BUDGETS,
        numerical_tolerances=dict(rtol=1e-8, absolute_l2_floor=1e-13),
        norm_definitions=dict(
            field="Original CPU field_weights: both faces, all four Gauss points, (xx,yy,2*xy) squared multiplicity.",
            N_M="sqrt(sum(GaussArea*(dxx^2+dyy^2+2*dxy^2)))/max(reference weighted L2,1e-30). Layout [element,Gauss,component,case].",
            Q="sqrt(sum(GaussArea*(dQ1^2+dQ2^2)))/max(reference weighted L2,1e-30). Layout [element,Gauss,component,case].",
            mean="Arithmetic mean over loaded cases separately in each bank; static repetitions are not independent trajectories.",
            boundary="Right-minus-left weighted field norm / left field norm; peak jump / mean of the matching mother peaks.",
            mother_residual="(CPU K @ saved approximate q - saved native common balanced force)[CPU free]; spatial diagnostic, never a correction or reduced-solve gate.",
        ),
        interpretation="Same-space CPU/GPU equivalence and spatial approximation budgets are separate. CPU12 resultant spatial failures are valid negative findings and do not invalidate numerical equivalence.",
    )
    write_json(args.output / "result.json", result)
    started = time.perf_counter()
    try:
        api = _api()
        result["versions"] = dict(numpy=api.np.__version__, scipy=api.scipy.__version__)
        with api.threadpool_limits(limits=1):
            audit(api, args, result)
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["seconds_including_readback_and_exports"] = time.perf_counter() - started
        write_json(args.output / "result.json", result)
    assert result["passed"], (
        "Saved-artifact consistency/equivalence audit failed; spatial rejection alone does not fail this command."
    )


if __name__ == "__main__":
    main()
