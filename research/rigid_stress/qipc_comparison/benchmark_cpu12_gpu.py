"""Paired complete-module wall timings for the matched native CPU12 construction.

Run only through Slurm rtx-mid/image 1_26. Each sample prepares fresh material
caps and recovers current forces. CPU oracles, audit exports, validity readback
and memory queries stay outside timing. This is not a complete-scene benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from .check_cpu12_gpu import (
    Checks,
    _api,
    _hash,
    _integer,
    _load,
    _outputs,
    _reference_array,
    _scalar_vectors,
    _source_inputs,
    _vector,
    build_native_space,
)


def _timing_api():
    api = _api()
    from genesis.engine.solvers.rigid.stress.atlas_coverage import AtlasCoverage
    from genesis.engine.solvers.rigid.stress.material_cap import kernel_material_cap_aggregate
    from genesis.engine.solvers.rigid.stress.prepared_atlas import kernel_admit_environments

    return SimpleNamespace(
        api=api, AtlasCoverage=AtlasCoverage, aggregate=kernel_material_cap_aggregate, admit=kernel_admit_environments
    )


def _tiled_cases(np, cases, tile):
    source_keys = (
        "direction",
        "radius",
        "force",
        "has_force",
        "normal_force",
        "friction",
        "tangent",
        "has_tangent",
        "cap_tangent",
        "has_cap_tangent",
        "side",
        "source_names",
    )
    tiled = {key: np.concatenate([cases[key]] * tile) for key in source_keys}
    n_envs = len(cases["case_names"])
    tiled["source_environment"] = np.concatenate([cases["source_environment"] + i * n_envs for i in range(tile)])
    tiled["case_names"] = np.concatenate([cases["case_names"]] * tile)
    tiled["case_families"] = np.concatenate([cases["case_families"]] * tile)
    tiled["case_offsets"] = np.r_[0, np.cumsum(np.tile(np.diff(cases["case_offsets"]), tile))]
    return tiled


@dataclass
class Routes:
    """Keep device load preparation and exact/reduced recovery workspaces distinct."""

    timing_api: object
    space: object
    sources: object
    geometry: object
    cap: object
    selector: object
    coverage: object
    prepared: object
    mother_state: object
    atlas_state: object
    mother_valid: object
    omega: object
    strategy: str

    def run(self, method, *, profile=False):
        api = self.timing_api.api
        stages = {}
        begin = time.perf_counter()
        self.cap.assemble(self.geometry, self.sources)
        if profile:
            api.qd.sync()
            stages["material_cap"] = time.perf_counter() - begin
            begin = time.perf_counter()
        if method == "mother":
            self.timing_api.aggregate(self.sources, self.cap.state, self.mother_state, self.mother_valid)
            if profile:
                api.qd.sync()
                stages["source_aggregation_admission"] = time.perf_counter() - begin
                begin = time.perf_counter()
            self.space.model.recover(self.omega, self.mother_state)
            self.timing_api.admit(self.mother_valid, self.mother_state)
        else:
            config = self.space.selection_input
            selected = self.selector.run(
                self.sources,
                self.cap.state,
                self.space.library.info.centers,
                strategy=self.strategy,
                support_angle=float(config["support_angle"]),
                narrow_fraction=float(config["narrow_fraction"]),
                maximum_radius=float(config["maximum_radius"]),
            )
            if profile:
                api.qd.sync()
                stages["selection_aggregation"] = time.perf_counter() - begin
                begin = time.perf_counter()
            covered = self.coverage.check(self.sources, self.cap.state, selected)
            if profile:
                api.qd.sync()
                stages["finite_footprint_coverage"] = time.perf_counter() - begin
                begin = time.perf_counter()
            self.prepared.recover(selected.grouped_force, self.atlas_state, covered.environment_covered)
        if profile:
            api.qd.sync()
            stages[
                "relief_project_solve_lift_recovery_acceptance"
                if method == "atlas"
                else "mother_relief_solve_recovery_acceptance"
            ] = time.perf_counter() - begin
        return stages

    def snapshot(self, method):
        api, np = self.timing_api.api, self.timing_api.api.np
        state = self.atlas_state if method == "atlas" else self.mother_state
        valid = api.qd_to_numpy(state.valid)
        cap_valid = api.qd_to_numpy(self.cap.state.valid)
        if method == "atlas":
            coverage_failures = int(api.qd_to_numpy(self.coverage.state.coverage_failures))
            groups = int(api.qd_to_numpy(self.selector.state.group_count))
            contributions = int(api.qd_to_numpy(self.selector.state.contribution_count))
            invalid_sources = int(api.qd_to_numpy(self.selector.state.invalid_source_count))
            fallbacks = sum(int(api.qd_to_numpy(item.fallbacks).sum()) for item in self.prepared.states)
        else:
            coverage_failures, contributions = 0, len(cap_valid)
            groups = int(np.count_nonzero(api.qd_to_numpy(state.rhs_norm_squared)))
            invalid_sources = int(np.count_nonzero(~cap_valid))
            fallbacks = int(api.qd_to_numpy(state.fallbacks).sum())
        return {
            "valid_environments": int(np.count_nonzero(valid)),
            "environments": len(valid),
            "rejected_environments": int(np.count_nonzero(~valid)),
            "invalid_sources": invalid_sources,
            "cap_invalid_sources": int(np.count_nonzero(~cap_valid)),
            "active_groups": groups,
            "source_feature_contributions": contributions,
            "coverage_failures": coverage_failures,
            "fallback_environments": fallbacks,
        }


def _memory_snapshot():
    command = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    own = []
    for line in command.stdout.splitlines():
        values = line.split(",")
        if len(values) == 2 and values[0].strip() == str(os.getpid()):
            memory = values[1].strip()
            if memory.isdecimal():
                own.append(int(memory) * 1024 * 1024)
    return {
        "pid": os.getpid(),
        "own_process_device_bytes": sum(own) if own else None,
        "raw_nvidia_processes": command.stdout,
        "nvidia_returncode": command.returncode,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "scope": "Point device snapshots and process peak RSS; not a sampled peak-device-memory measurement.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--bank", choices=("static16", "static64"), default="static64")
    parser.add_argument("--tile", type=int, default=1)
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--strategy", choices=("continuous", "nearest", "narrow"), default="continuous")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument(
        "--profile", action="store_true", help="Run intrusive stage fences, separately from wall timing."
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Use Slurm rtx-mid with image 1_26.")
    if min(args.tile, args.warmup, args.repeats) < 1:
        parser.error("tile, warmup and repeats must be positive.")
    output, reference = args.output.expanduser().resolve(), args.reference.expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[3]) or (output.exists() and any(output.iterdir())):
        parser.error("Use a new/empty data-side output directory outside the source checkout.")
    output.mkdir(parents=True, exist_ok=True)
    timing_api = _timing_api()
    api, np = timing_api.api, timing_api.api.np
    writer = api.Writer(output, api)
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    assert reference_manifest["passed"] and reference_manifest["schema"] == "genesis.cpu12-reference.v1"
    for relative, identity in reference_manifest["files"].items():
        assert _hash(reference / relative) == identity["sha256"], relative
    source_files = {
        str(path): _hash(path)
        for path in sorted((Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress").glob("*.py"))
    }
    source_files[str(Path(__file__).resolve())] = _hash(Path(__file__))
    source_files[str(Path(build_native_space.__code__.co_filename).resolve())] = _hash(
        Path(build_native_space.__code__.co_filename)
    )
    identity = {
        "kind": "cpu12_complete_module_intrusive_profile" if args.profile else "cpu12_complete_module_paired_wall",
        "reference_manifest_sha256": _hash(reference / "manifest.json"),
        "bank": args.bank,
        "tile": args.tile,
        "output_mode": args.output_mode,
        "strategy": args.strategy,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "precision": "64",
        "source_files": source_files,
        "native_package": str(api.gs.__file__),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": "Frozen-contact complete native material-cap modules; no complete Franka/controller/rigid timing claim.",
        "timed": "Fresh cap.assemble, native aggregation/selection/coverage, relief, solve, recovery and device validity gates.",
        "outside_timing": "CPU oracle checks, exports, device synchronization before start, validity readback, memory queries.",
        "fallback_policy": "No mother correction or approximation fallback; failed admission is reported as rejected work.",
        "passed_semantics": "Independent numerical equivalence, native admission and unchanged source identity.",
        "spatial_qualification": "Requires a separate source-matched all-bank spatial audit and declared tensor/peak/resultant budgets.",
        "scene_throughput_qualification": "Complete-module measurements do not establish complete-scene accuracy or throughput gains.",
    }
    writer.json("start.json", identity)
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    before_memory = _memory_snapshot()
    setup_started = time.perf_counter()
    space = build_native_space(api, reference, output, output_mode=args.output_mode)
    cases = _tiled_cases(np, _load(np, reference / f"cases/{args.bank}.npz"), args.tile)
    writer.arrays("cases.npz", **cases)
    n_envs, n_sources = len(cases["case_names"]), len(cases["radius"])
    sources = _source_inputs(api, cases)
    geometry = api.build_material_cap_geometry(
        space.model.info.vertices,
        _integer(api, space.model_input["face_tables"]),
        space.model.shell.directors,
        ncoarse=int(space.model_input["ncoarse"]),
        subdiv=int(space.model_input["subdiv"]),
        vertex_stride=2,
    )
    cap = api.MaterialCap(
        int(reference_manifest["configuration"]["quadrature"]),
        n_sources,
        866,
        ncoarse=int(space.model_input["ncoarse"]),
        subdiv=int(space.model_input["subdiv"]),
        thickness=float(space.model_input["thickness"]),
        vertex_stride=2,
        audit=False,
    )
    selector = api.AtlasSelection(n_sources, n_envs, space.model.info.vertices.shape[0])
    coverage = timing_api.AtlasCoverage(space.basis.coarse_index, space.library.info.local_mask, n_sources, n_envs)
    prepared = api.PreparedAtlas(space.model, space.library.maps, n_envs)
    mother_state, atlas_state = (
        space.model.create_state(n_envs, args.output_mode),
        space.model.create_state(n_envs, args.output_mode),
    )
    valid = api.V(dtype=api.gs.qd_int, shape=(n_envs,))
    routes = Routes(
        timing_api,
        space,
        sources,
        geometry,
        cap,
        selector,
        coverage,
        prepared,
        mother_state,
        atlas_state,
        valid,
        _vector(api, np.zeros((n_envs, 3))),
        args.strategy,
    )
    api.qd.sync()
    setup_seconds = time.perf_counter() - setup_started
    setup = {
        "seconds": setup_seconds,
        "model_timings": asdict(space.model.build_timings),
        "atlas_timings": asdict(prepared.build_timings),
        "environments": n_envs,
        "sources": n_sources,
        "mother_raw_dofs": 5196,
        "mother_free_dofs": 5190,
        "feature_raw_dofs": 3522,
        "feature_free_dofs": 3516,
        "mother_factor_lower_blocks": space.model.factor.info.lower.shape[0],
        "feature_factor_lower_blocks": [operator.factor.info.lower.shape[0] for operator in prepared.operators],
        "before_memory": before_memory,
        "after_setup_memory": _memory_snapshot(),
    }
    writer.json("setup.json", setup)
    # Preflight outputs are independently verified before warmup and before any
    # timed sample. These tensors and CPU arrays never enter timed preparation.
    checks = Checks(api)
    for method in ("mother", "atlas"):
        routes.run(method)
        api.qd.sync()
        state = mother_state if method == "mother" else atlas_state
        snapshot = routes.snapshot(method)
        writer.json(f"preflight/{method}/validity.json", snapshot)
        checks.condition(
            f"preflight/{method}/valid",
            snapshot["valid_environments"] == n_envs
            and snapshot["invalid_sources"] == 0
            and snapshot["coverage_failures"] == 0,
        )
        oracle_strategy = "mother" if method == "mother" else args.strategy
        actual = (
            _outputs(api, space.model, state)
            if args.output_mode == "full"
            else {
                "q": _scalar_vectors(api, state.displacement),
                "peak": api.qd_to_numpy(state.peak),
                "force": _scalar_vectors(api, state.force),
                "rhs": _scalar_vectors(api, state.rhs),
            }
        )
        writer.arrays(f"preflight/{method}/output.npz", **actual)
        for key in (
            ("q", "peak", "stress_local", "stress_full9", "N", "M", "Q")
            if args.output_mode == "full"
            else ("q", "peak")
        ):
            expected = _reference_array(api, reference, args.bank, f"{oracle_strategy}/output.npz", key)
            expected = np.concatenate([expected] * args.tile, axis=-1)
            checks.close(f"preflight/{method}/{key}", actual[key], expected, rtol=1e-8, atol=1e-13)
        expected_raw = _reference_array(api, reference, args.bank, "loads.npz", "raw_environment")
        checks.close(
            f"preflight/{method}/raw", actual["force"], np.concatenate([expected_raw] * args.tile, axis=-1), atol=1e-14
        )
    checks.flush(writer)
    if not all(row["passed"] for row in checks.rows):
        raise ArithmeticError("Independent preflight failed; no performance samples are admitted.")
    for iteration in range(args.warmup):
        for method in ("mother", "atlas") if iteration % 2 == 0 else ("atlas", "mother"):
            routes.run(method)
            api.qd.sync()
    measurements = []
    for pair in range(args.repeats):
        for order, method in enumerate(("mother", "atlas") if pair % 2 == 0 else ("atlas", "mother")):
            api.qd.sync()
            begin = time.perf_counter()
            stages = routes.run(method, profile=args.profile)
            api.qd.sync()
            seconds = time.perf_counter() - begin
            snapshot = routes.snapshot(method)
            measurements.append(
                {
                    "pair": pair,
                    "order": order,
                    "method": method,
                    "seconds": seconds,
                    "stages_seconds": stages,
                    **snapshot,
                    "valid_environment_steps_per_second": snapshot["valid_environments"] / seconds,
                }
            )
    writer.json("measurements.json", measurements)
    writer.arrays(
        "measurements.npz",
        pair=[m["pair"] for m in measurements],
        order=[m["order"] for m in measurements],
        method=[m["method"] for m in measurements],
        seconds=[m["seconds"] for m in measurements],
        valid_environments=[m["valid_environments"] for m in measurements],
        rejected_environments=[m["rejected_environments"] for m in measurements],
    )
    summary = []
    for method in ("mother", "atlas"):
        rows = [item for item in measurements if item["method"] == method]
        latency = np.array([item["seconds"] for item in rows])
        summary.append(
            {
                "method": method,
                "median_seconds": float(np.median(latency)),
                "minimum_seconds": float(latency.min()),
                "maximum_seconds": float(latency.max()),
                "valid_environment_steps_per_second": sum(row["valid_environments"] for row in rows) / latency.sum(),
                "rejected_environment_steps": sum(row["rejected_environments"] for row in rows),
                "coverage_failures": sum(row["coverage_failures"] for row in rows),
                "fallback_environments": sum(row["fallback_environments"] for row in rows),
            }
        )
    passed = all(
        item["valid_environments"] == n_envs and item["invalid_sources"] == 0 and item["coverage_failures"] == 0
        for item in measurements
    )
    source_unchanged = all(_hash(path) == digest for path, digest in source_files.items())
    result = {
        **identity,
        "passed": passed and source_unchanged,
        "numerical_equivalence_passed": all(row["passed"] for row in checks.rows),
        "native_admission_passed": passed,
        "spatial_qualification_established_by_this_benchmark": False,
        "complete_scene_benefit_established_by_this_benchmark": False,
        "source_unchanged": source_unchanged,
        "summary": summary,
        "setup_seconds": setup_seconds,
        "after_measurements_memory": _memory_snapshot(),
        "profile_is_intrusive": args.profile,
        "files": writer.inventory,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "passed": result["passed"], "summary": summary}))
    if not result["passed"]:
        raise ArithmeticError("Native paired benchmark rejected work or changed sources; inspect raw measurements.")


if __name__ == "__main__":
    main()
