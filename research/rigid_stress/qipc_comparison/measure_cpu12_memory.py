"""Measure one CPU12 mother/Atlas/packed process, without performance claims.

Each invocation prepares one configuration and executes three fresh current-load
calls. A separate standard-library collector requests NVIDIA process-memory
samples every 50 ms, covering imports, preparation/JIT and runtime. Scientific
imports require Slurm rtx-mid/image 1_26. No CPU numerical oracle is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def _pid_mapping():
    pid = os.getpid()
    mapping = dict(container_pid=pid, nspid=[], candidates=[pid], preferred_pid=pid, status_error=None)
    try:
        status = Path("/proc/self/status").read_text()
        line = next((line for line in status.splitlines() if line.startswith("NSpid:")), None)
        if line is not None:
            identifiers = [int(value) for value in line.split()[1:]]
            mapping.update(nspid=identifiers, raw_nspid=line)
            if identifiers:
                mapping["preferred_pid"] = identifiers[0]
                mapping["candidates"] = list(dict.fromkeys([identifiers[0], *identifiers, pid]))
    except (OSError, ValueError) as error:
        mapping["status_error"] = str(error)
    mapping["policy"] = (
        "Prefer the outermost NSpid; otherwise accept a unique matching candidate. Ambiguous/unmatched samples remain unavailable."
    )
    return mapping


def _collector_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_collector", action="store_true")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--pid-candidates", required=True)
    parser.add_argument("--preferred-pid", type=int, required=True)
    parser.add_argument("--started", type=float, required=True)
    args = parser.parse_args()
    candidates = {int(value) for value in args.pid_candidates.split(",")}
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        json.dumps(
            dict(
                event="ready",
                pid=os.getpid(),
                target_parent_pid=args.parent_pid,
                scientific_modules_loaded=[
                    name for name in ("numpy", "torch", "quadrants", "genesis", "scipy") if name in sys.modules
                ],
            )
        ),
        flush=True,
    )
    command = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"]
    while not stopping and os.getppid() == args.parent_pid:
        before = time.perf_counter()
        record = dict(
            event="poll", started_seconds=before - args.started, command=command, matches=[], selected_pid=None
        )
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
            record.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
            matches = []
            for row in csv.reader(completed.stdout.splitlines()):
                if len(row) < 3:
                    continue
                pid, memory = row[0].strip(), row[2].strip()
                if pid.isdecimal() and int(pid) in candidates and memory.isdecimal():
                    matches.append(dict(pid=int(pid), gpu_uuid=row[1].strip(), bytes=int(memory) * 1024 * 1024))
            matched_pids = {row["pid"] for row in matches}
            selected = (
                args.preferred_pid
                if args.preferred_pid in matched_pids
                else next(iter(matched_pids))
                if len(matched_pids) == 1
                else None
            )
            record.update(
                candidate_matches=matches,
                selected_pid=selected,
                matches=[row for row in matches if row["pid"] == selected]
                if selected is not None and completed.returncode == 0
                else [],
                pid_match_status="preferred"
                if selected == args.preferred_pid
                else "unique_candidate"
                if selected is not None
                else "ambiguous_or_unmatched",
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            record.update(returncode=None, error=dict(type=type(error).__name__, message=str(error)))
        record["seconds"] = time.perf_counter() - args.started
        print(json.dumps(record), flush=True)
        remaining = 0.05 - (time.perf_counter() - before)
        if not stopping and remaining > 0:
            time.sleep(remaining)


class ProcessSampler:
    def __init__(self, output, mapping):
        self.output, self.mapping = output, mapping
        self.process, self.records, self.errors = None, None, None
        self.started = time.perf_counter()
        self.start_error, self.ready = None, None
        self.raw_path = output / "nvidia-process-memory.jsonl"
        self.error_path = output / "memory-collector.stderr.log"

    def start(self):
        self.records = self.raw_path.open("w", encoding="utf-8")
        self.errors = self.error_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_collector",
                    "--parent-pid",
                    str(os.getpid()),
                    "--pid-candidates",
                    ",".join(str(pid) for pid in self.mapping["candidates"]),
                    "--preferred-pid",
                    str(self.mapping["preferred_pid"]),
                    "--started",
                    repr(self.started),
                ],
                stdout=self.records,
                stderr=self.errors,
            )
            deadline = time.perf_counter() + 5.0
            while time.perf_counter() < deadline:
                first = os.pread(self.records.fileno(), 8192, 0).split(b"\n", 1)
                if len(first) == 2:
                    self.ready = json.loads(first[0])
                    if self.ready["event"] != "ready" or self.ready["scientific_modules_loaded"]:
                        raise RuntimeError("The memory collector must be a separate standard-library-only process.")
                    return
                if self.process.poll() is not None:
                    raise RuntimeError("Memory collector exited before its readiness receipt.")
                time.sleep(0.01)
            raise RuntimeError("Memory collector did not report ready within5s.")
        except (OSError, ValueError, RuntimeError) as error:
            self.start_error = dict(type=type(error).__name__, message=str(error))

    def finish(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.records is not None:
            self.records.close()
        if self.errors is not None:
            self.errors.close()
        rows, errors = [], []
        if self.raw_path.exists():
            for line_number, line in enumerate(self.raw_path.read_text().splitlines(), start=1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    errors.append(dict(line=line_number, error=str(error)))
        polls = [row for row in rows if row["event"] == "poll"]
        matched = [row for row in polls if row["matches"]]
        per_device = {}
        for row in matched:
            for device in row["matches"]:
                per_device[device["gpu_uuid"]] = max(per_device.get(device["gpu_uuid"], 0), device["bytes"])
        return dict(
            available=bool(matched),
            sampled_peak_device_bytes=max(
                (sum(device["bytes"] for device in row["matches"]) for row in matched), default=None
            ),
            sampled_peak_bytes_by_gpu=per_device,
            requested_interval_seconds=0.05,
            observed_poll_count=len(polls),
            matched_poll_count=len(matched),
            unmatched_poll_count=len(polls) - len(matched),
            first_matching_seconds=matched[0]["seconds"] if matched else None,
            last_matching_seconds=matched[-1]["seconds"] if matched else None,
            ready=self.ready,
            start_error=self.start_error,
            parse_errors=errors,
            collector_returncode=self.process.returncode if self.process is not None else None,
            pid_mapping=self.mapping,
            raw_log=str(self.raw_path),
            collector_stderr=str(self.error_path),
            scope="Sampled own-process NVIDIA memory, including CUDA/JIT/allocator residency; not a guaranteed allocator or hardware peak. Sampling cadence includes nvidia-smi command latency.",
        )


def _api():
    from genesis.utils.array_class import DataItem, DataKind, iter_data

    from .benchmark_cpu12_gpu import Routes, _tiled_cases, _timing_api
    from .check_cpu12_gpu import (
        NativeSpace,
        _hash,
        _integer,
        _load,
        _source_inputs,
        _vector,
        build_native_mother,
        build_native_space,
    )

    timing_api = _timing_api()
    return dict(
        timing=timing_api,
        api=timing_api.api,
        Routes=Routes,
        tiled_cases=_tiled_cases,
        NativeSpace=NativeSpace,
        hash=_hash,
        integer=_integer,
        load=_load,
        source_inputs=_source_inputs,
        vector=_vector,
        build_mother=build_native_mother,
        build_space=build_native_space,
        DataItem=DataItem,
        DataKind=DataKind,
        iter_data=iter_data,
    )


def _items(context, routes, method):
    DataItem, DataKind, iter_data = context["DataItem"], context["DataKind"], context["iter_data"]
    model = routes.space.model
    yield from model.iter_info_data("mother")
    if model.factor is not None:
        yield from iter_data(model.factor.info, "mother.factor")
    if model.inverse is not None:
        yield from iter_data(model.inverse.info, "mother.inverse")
    if model.surface_inverse is not None:
        yield from iter_data(model.surface_inverse.info, "mother.surface_inverse")
    yield from iter_data(routes.geometry, "material_cap.geometry")
    yield from iter_data(routes.cap.rule, "material_cap.rule")
    yield from iter_data(routes.cap.state, "material_cap.state")
    yield from iter_data(routes.sources, "physical_sources")
    yield DataItem("current_omega", routes.omega, DataKind.SCRATCH)
    if method == "mother":
        yield from iter_data(routes.mother_state, "mother.current_state")
        yield DataItem("mother.environment_valid", routes.mother_valid, DataKind.SCRATCH)
    else:
        yield from iter_data(routes.atlas_state, "atlas.current_state")
        yield from iter_data(routes.space.library.info, "library.info")
        yield DataItem("basis.coarse_transform_values", routes.space.basis.transform.values, DataKind.CONSTANT)
        for index, mapping in enumerate(routes.space.library.maps):
            yield DataItem(f"library.maps.{index}.values", mapping.values, DataKind.CONSTANT)
        yield from iter_data(routes.selector.state, "selector.state")
        yield from iter_data(routes.coverage.state, "coverage.state")
        yield DataItem("coverage.coarse", routes.coverage.coarse, DataKind.CONSTANT)
        yield DataItem("coverage.local_mask", routes.coverage.local_mask, DataKind.CONSTANT)
        yield DataItem("coverage.source_maximum", routes.coverage.source_maximum, DataKind.SCRATCH)
        yield DataItem("coverage.feature_failed", routes.coverage.feature_failed, DataKind.SCRATCH)
        yield from routes.prepared.iter_info_data("prepared")
        yield from routes.prepared.iter_state_data("prepared")
        yield DataItem("prepared.zero_omega", routes.prepared.omega, DataKind.SCRATCH)


def _inventory(context, routes, method, n_environments):
    api, DataKind = context["api"], context["DataKind"]
    allocations, buffers = {}, []
    for item in _items(context, routes, method):
        if item.kind == DataKind.CONFIG:
            continue
        if math.prod(item.value.shape) == 0:
            buffers.append(
                dict(
                    name=item.name,
                    kind=item.kind.name,
                    shape=list(item.value.shape),
                    logical_bytes=0,
                    storage_bytes=0,
                    alias_of=None,
                )
            )
            continue
        view = api.qd_to_torch(item.value, copy=False)
        storage = view.untyped_storage()
        pointer, storage_bytes = int(storage.data_ptr()), int(storage.nbytes())
        key = (str(view.device), pointer)
        alias_of = allocations[key]["first_name"] if key in allocations else None
        if key not in allocations:
            allocations[key] = dict(
                first_name=item.name,
                device=str(view.device),
                storage_pointer=pointer,
                storage_bytes=storage_bytes,
                aliases=[],
                kinds=[],
            )
        allocation = allocations[key]
        allocation["storage_bytes"] = max(allocation["storage_bytes"], storage_bytes)
        allocation["aliases"].append(item.name)
        if item.kind.name not in allocation["kinds"]:
            allocation["kinds"].append(item.kind.name)
        buffers.append(
            dict(
                name=item.name,
                kind=item.kind.name,
                shape=list(view.shape),
                strides=list(view.stride()),
                dtype=str(view.dtype),
                device=str(view.device),
                data_pointer=int(view.data_ptr()),
                storage_pointer=pointer,
                logical_bytes=int(view.numel() * view.element_size()),
                storage_bytes=storage_bytes,
                alias_of=alias_of,
            )
        )
    unique = list(allocations.values())
    for item in unique:
        item["classification"] = "CONSTANT" if item["kinds"] == ["CONSTANT"] else "NONCONSTANT"
    constant = sum(item["storage_bytes"] for item in unique if item["classification"] == "CONSTANT")
    nonconstant = sum(item["storage_bytes"] for item in unique if item["classification"] == "NONCONSTANT")
    summary = dict(
        unique_native_bytes=constant + nonconstant,
        shared_constant_bytes=constant,
        nonconstant_bytes=nonconstant,
        nonconstant_bytes_per_environment=nonconstant / n_environments,
        environments=n_environments,
        unique_allocation_count=len(unique),
        exposed_buffer_count=len(buffers),
        logical_buffer_bytes_before_alias_dedup=sum(item["logical_bytes"] for item in buffers),
        constant_nonconstant_alias_conflicts=[
            item["first_name"] for item in unique if "CONSTANT" in item["kinds"] and len(item["kinds"]) > 1
        ],
        definition="Resident exposed QD buffers, deduplicated by device and zero-copy Torch storage pointer. Per-environment division includes fixed scratch and per-source buffers; compiler, allocator pools and CUDA context memory are outside this inventory.",
    )
    return summary, buffers, unique


def _experiment(context, args, manifest, result, phase):
    api, np = context["api"], context["api"].np
    torch_reset_error = None
    try:
        api.torch.cuda.reset_peak_memory_stats()
    except RuntimeError as error:
        torch_reset_error = str(error)
    phase("before_genesis_init")
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    phase("after_genesis_init")
    if args.method == "mother":
        model_input, selection_input, model = context["build_mother"](
            api, args.reference, args.output, output_mode=args.output_mode
        )
        space = context["NativeSpace"](model, None, None, model_input, selection_input)
    else:
        space = context["build_space"](api, args.reference, args.output, output_mode=args.output_mode)
    phase("after_space_preparation")
    cases = context["tiled_cases"](np, context["load"](np, args.reference / f"cases/{args.bank}.npz"), args.tile)
    np.savez_compressed(args.output / "cases.npz", **cases)
    n_envs, n_sources = len(cases["case_names"]), len(cases["radius"])
    sources = context["source_inputs"](api, cases)
    geometry = api.build_material_cap_geometry(
        space.model.info.vertices,
        context["integer"](api, space.model_input["face_tables"]),
        space.model.shell.directors,
        ncoarse=int(space.model_input["ncoarse"]),
        subdiv=int(space.model_input["subdiv"]),
        vertex_stride=2,
    )
    cap = api.MaterialCap(
        int(manifest["configuration"]["quadrature"]),
        n_sources,
        len(space.model_input["vertices"]),
        ncoarse=int(space.model_input["ncoarse"]),
        subdiv=int(space.model_input["subdiv"]),
        thickness=float(space.model_input["thickness"]),
        vertex_stride=2,
        audit=False,
    )
    selector, coverage, prepared, mother_state, atlas_state, mother_valid = None, None, None, None, None, None
    if args.method == "mother":
        mother_state = space.model.create_state(n_envs, args.output_mode)
        mother_valid = api.V(dtype=api.gs.qd_int, shape=(n_envs,))
    else:
        selector = api.AtlasSelection(n_sources, n_envs, space.model.info.vertices.shape[0])
        coverage = context["timing"].AtlasCoverage(
            space.basis.coarse_index, space.library.info.local_mask, n_sources, n_envs
        )
        prepared = api.PreparedAtlas(space.model, space.library.maps, n_envs, packed_solve=args.method == "packed")
        atlas_state = space.model.create_state(n_envs, args.output_mode)
    routes = context["Routes"](
        context["timing"],
        space,
        sources,
        geometry,
        cap,
        selector,
        coverage,
        prepared,
        mother_state,
        atlas_state,
        mother_valid,
        context["vector"](api, np.zeros((n_envs, 3))),
        args.strategy,
    )
    api.qd.sync()
    phase("after_load_and_recovery_preparation")
    route_method = "mother" if args.method == "mother" else "atlas"
    snapshots = []
    for call in range(3):
        phase(f"before_current_load_call_{call + 1}")
        routes.run(route_method)
        api.qd.sync()
        snapshot = routes.snapshot(route_method)
        snapshots.append(dict(call=call + 1, **snapshot))
        phase(f"after_current_load_call_{call + 1}")
    phase("before_buffer_inventory")
    inventory, buffers, allocations = _inventory(context, routes, args.method, n_envs)
    phase("after_buffer_inventory")
    for name, payload in (("buffers.json", buffers), ("allocations.json", allocations)):
        (args.output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    result.update(
        model_options=space.model.options.model_dump(mode="json"),
        environments=n_envs,
        sources=n_sources,
        released_environments=int(np.count_nonzero(np.diff(cases["case_offsets"]) == 0)),
        current_load_calls=3,
        admission_snapshots=snapshots,
        resident_inventory=inventory,
        mother_scalar_dofs=space.model.info.vertices.shape[0] * 3,
        mother_factor_lower_entries=space.model.factor.info.lower.shape[0],
        feature_spaces_constructed=0 if prepared is None else len(prepared.operators),
        feature_scalar_dofs=[]
        if prepared is None
        else [operator.info.vertices.shape[0] * 3 for operator in prepared.operators],
        feature_free_dofs=[]
        if prepared is None
        else [int(mapping.reduced_free.sum()) for mapping in space.library.maps],
        feature_factor_lower_entries=[]
        if prepared is None
        else [operator.factor.info.lower.shape[0] for operator in prepared.operators],
        original_mother_factor_retained=True,
        original_feature_factors_retained=prepared is not None,
        packed_extra_inventory=asdict(prepared.packed.memory_inventory())
        if prepared is not None and prepared.packed is not None
        else None,
        peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        torch_peak_allocated_bytes=int(api.torch.cuda.max_memory_allocated()),
        torch_peak_reserved_bytes=int(api.torch.cuda.max_memory_reserved()),
        torch_peak_reset_error=torch_reset_error,
        torch_peak_scope="Torch allocator counters only, not total QD/device/process allocation.",
        native_admission_passed=all(
            snapshot["rejected_environments"] == 0
            and snapshot["invalid_sources"] == 0
            and snapshot["cap_invalid_sources"] == 0
            and snapshot["coverage_failures"] == 0
            for snapshot in snapshots
        ),
    )


def _main():
    from .artifacts import file_id, runtime_identity, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mother", "atlas", "packed"), required=True)
    parser.add_argument("--bank", choices=("static16", "static64"), default="static64")
    parser.add_argument("--tile", type=int, default=1)
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--strategy", choices=("continuous", "nearest", "narrow"), default="continuous")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Use Slurm rtx-mid with GENESIS_IMAGE_VER=1_26.")
    if args.tile < 1:
        parser.error("tile must be positive.")
    args.reference, args.output, args.git_source = (
        path.expanduser().resolve() for path in (args.reference, args.output, args.git_source)
    )
    if args.output.is_relative_to(Path(__file__).resolve().parents[3]) or args.output.is_relative_to(args.reference):
        parser.error("Use a separate project data output directory outside source/reference trees.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh output directory; prior measurements remain immutable.")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GS_CACHE_FILE_PATH", str(args.output / "cache/genesis"))
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if manifest["schema"] != "genesis.cpu12-reference.v1" or not manifest["passed"]:
        parser.error("Use a successfully frozen CPU12 reference.")
    production_inputs = {}
    for relative in ("inputs/model.npz", "inputs/selection.npz", f"cases/{args.bank}.npz"):
        identity = file_id(args.reference / relative)
        expected = manifest["files"][relative]
        if identity["sha256"] != expected["sha256"] or identity["bytes"] != expected["bytes"]:
            raise ArithmeticError(f"Frozen physical input changed: {relative}")
        production_inputs[relative] = identity
    sampler = ProcessSampler(args.output, _pid_mapping())
    sampler.start()
    events = []

    def phase(name):
        events.append(dict(name=name, seconds=time.perf_counter() - sampler.started))

    phase("before_scientific_imports")
    result = dict(
        schema="genesis.cpu12-process-memory.v1",
        complete=False,
        passed=False,
        method=args.method,
        bank=args.bank,
        tile=args.tile,
        output_mode=args.output_mode,
        strategy=args.strategy,
        process_id=os.getpid(),
        identity=runtime_identity(),
        worker=file_id(Path(__file__)),
        reference_manifest=file_id(args.reference / "manifest.json"),
        production_inputs=production_inputs,
        precision="64",
        isolated_method_process=True,
        performance_measurement=False,
        passed_semantics="Native current-load admission and unchanged source/input identities; GPU peak sampling availability is independent and may be unavailable.",
        scope="One process/configuration: preparation and JIT, then three fresh current-load calls. No CPU numerical oracle, throughput or latency claim.",
        comparison_policy="Mother-only never constructs P/B maps or twelve feature factors. Atlas/packed retain their actual mother and feature inventories; their sampled peaks are not independent-mother peaks.",
        spatial_qualification="Memory/admission observations do not qualify approximation errors or applicable resultant budgets.",
    )
    write_json(args.output / "result.json", result)
    try:
        context = _api()
        phase("after_scientific_imports")
        api = context["api"]
        native_source = Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress"
        paths = sorted(native_source.glob("*.py")) + [
            Path(context["build_mother"].__code__.co_filename),
            Path(context["Routes"].run.__code__.co_filename),
            Path(__file__),
        ]
        source_files = {str(path.resolve()): file_id(path)["sha256"] for path in paths}
        result.update(
            source_files=source_files,
            native_package=str(api.gs.__file__),
            checkout_commit=subprocess.check_output(
                ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
            ).strip(),
        )
        _experiment(context, args, manifest, result, phase)
        result["source_unchanged"] = all(file_id(path)["sha256"] == digest for path, digest in source_files.items())
        result["physical_inputs_unchanged"] = all(
            file_id(args.reference / relative) == identity for relative, identity in production_inputs.items()
        )
        result["reference_manifest_unchanged"] = (
            file_id(args.reference / "manifest.json") == result["reference_manifest"]
        )
        result.update(
            complete=True,
            passed=result["native_admission_passed"]
            and result["source_unchanged"]
            and result["physical_inputs_unchanged"]
            and result["reference_manifest_unchanged"],
        )
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        phase("before_sampler_stop")
        result["process_gpu_sampling"] = sampler.finish()
        result["memory_metric_availability"] = dict(
            sampled_own_gpu_peak=result["process_gpu_sampling"]["available"],
            process_peak_rss=True,
            torch_peak="torch_peak_allocated_bytes" in result,
        )
        result["stage_events"] = events
        result["peak_process_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        result["peak_process_rss_scope"] = (
            "Linux process ru_maxrss, including imports, preparation, JIT, current loads and inventory export."
        )
        result["raw_sampler_log"] = file_id(sampler.raw_path)
        result["raw_sampler_stderr"] = file_id(sampler.error_path)
        write_json(args.output / "result.json", result)
    assert result["passed"], "Memory workload admission/source checks failed; retain measurements and raw sampler logs."


if __name__ == "__main__":
    if "--_collector" in sys.argv:
        _collector_main()
    else:
        _main()
