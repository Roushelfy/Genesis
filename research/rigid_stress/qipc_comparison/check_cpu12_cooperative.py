"""Check the existing cooperative factor kernel at actual and boundary block sizes.

Run through Slurm rtx-mid/image 1_26. This private probe does not change the
production dispatcher or certify complete-module or complete-scene speedups.
"""

import argparse
import inspect
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .benchmark_cpu12_gpu import _tiled_cases
from .check_cpu12_gpu import (
    SOURCE,
    Checks,
    _api,
    _hash,
    _integer,
    _load,
    _matrix,
    _scalar_vectors,
    _source_inputs,
    _vector,
    build_native_space,
)


def _native_api():
    api = _api()
    import torch

    from genesis.engine.solvers.rigid.stress.data import allocate_stress_info, create_stress_state
    from genesis.engine.solvers.rigid.stress.factor import StressFactor, kernel_solve_cooperative
    from genesis.engine.solvers.rigid.stress.material_cap import kernel_material_cap_aggregate
    from genesis.engine.solvers.rigid.stress.prepared_atlas import kernel_group_wrench, kernel_project_group
    from genesis.engine.solvers.rigid.stress.solve import kernel_direct_init, kernel_full_residual

    qd, gs = api.qd, api.gs

    @qd.kernel
    def kernel_boundary_matrix(row_start: qd.Tensor, columns: qd.Tensor, stiffness: qd.Tensor):
        for block in range(row_start.shape[0] - 1):
            a = 1.0 + qd.cast(block % 17, gs.qd_float) / 32.0
            b = 1.5 + qd.cast(block % 7, gs.qd_float) / 16.0
            c = 2.0 + qd.cast(block % 11, gs.qd_float) / 32.0
            lower = qd.Matrix([[a, 0.0, 0.0], [0.125, b, 0.0], [-0.25, 0.0625, c]])
            for entry in range(row_start[block], row_start[block + 1]):
                value = 0.02 * qd.Matrix.identity(gs.qd_float, 3)
                if columns[entry] == block:
                    value = lower @ lower.transpose()
                stiffness[entry] = value

    @qd.kernel
    def kernel_boundary_rhs(
        row_start: qd.Tensor,
        columns: qd.Tensor,
        stiffness: qd.Tensor,
        rhs: qd.Tensor,
        exact_q: qd.Tensor,
        young: float,
    ):
        for block, environment in qd.ndrange(rhs.shape[0], rhs.shape[1]):
            value = qd.Vector.zero(gs.qd_float, 3)
            if environment < rhs.shape[1] - 1:
                scale = qd.cast(environment + 1, gs.qd_float)
                value = qd.Vector(
                    [
                        qd.cast(block % 13 - 6, gs.qd_float) / 16.0 + scale / 8.0,
                        qd.cast(block % 7 - 3, gs.qd_float) / 8.0 - scale / 16.0,
                        qd.cast(block % 5 - 2, gs.qd_float) / 4.0 + scale / 32.0,
                    ]
                )
            exact_q[block, environment] = value
        for block, environment in qd.ndrange(rhs.shape[0], rhs.shape[1]):
            value = qd.Vector.zero(gs.qd_float, 3)
            for entry in range(row_start[block], row_start[block + 1]):
                other = columns[entry]
                value += stiffness[entry] @ exact_q[other, environment]
            rhs[block, environment] = young * value

    return SimpleNamespace(**locals())


@dataclass
class System:
    name: str
    info: object
    factor: object
    state: object
    stress_state: object
    operator: object = None
    matrix: object = None


def _hardware(native):
    torch = native.torch
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "torch_device_index": device,
        "properties": {
            name: getattr(properties, name, None)
            for name in (
                "name",
                "major",
                "minor",
                "total_memory",
                "multi_processor_count",
                "warp_size",
                "max_threads_per_block",
                "shared_memory_per_block",
                "shared_memory_per_block_optin",
                "shared_memory_per_multiprocessor",
            )
        },
        "torch_properties_repr": str(properties),
        "nvidia_smi_returncode": query.returncode,
        "nvidia_smi": query.stdout,
        "nvidia_smi_stderr": query.stderr,
        "shared_bytes": {"1174": 1174 * 3 * 8, "1732": 1732 * 3 * 8, "2048": 2048 * 3 * 8},
        "shared_memory_scope": "Declared SharedArray bytes; compiler allocation/occupancy are not measured here.",
    }


def _dispatch(native, system, method, young):
    if method == "default":
        system.factor.solve(young, system.state, system.info, cooperative=True, only_failed=False)
    else:
        native.kernel_solve_cooperative(
            young, system.state, system.info, system.factor.info, system.info.vertices.shape[0], False
        )


def _snapshot(native, model, system):
    api, np = native.api, native.api.np
    native.kernel_full_residual(
        model.options.young,
        model.options.tolerance,
        model.options.absolute_tolerance,
        system.state,
        system.info,
        False,
        False,
    )
    if system.operator is not None:
        system.stress_state.displacement.fill(0)
        system.operator.lift(system.state.displacement, system.stress_state.displacement)
    model.evaluate_stress(model.options, system.stress_state)
    q = _scalar_vectors(api, system.state.displacement)
    rhs = _scalar_vectors(api, system.state.rhs)
    free = np.flatnonzero(api.qd_to_numpy(system.info.is_free).reshape(-1))
    residual = (system.matrix @ q - rhs)[free]
    return {
        "q": q,
        "rhs": rhs,
        "peak": api.qd_to_numpy(system.stress_state.peak),
        "valid": api.qd_to_numpy(system.state.valid),
        "native_residual_norm_squared": api.qd_to_numpy(system.state.residual_norm_squared),
        "unscaled_residual_absolute": np.linalg.norm(residual, axis=0),
        "rhs_free_norm": np.linalg.norm(rhs[free], axis=0),
    }


def _check_snapshot(api, checks, name, value, options, reference=None, *, zero=False):
    np = api.np
    budget = np.maximum(options.absolute_tolerance, options.tolerance * value["rhs_free_norm"])
    checks.condition(f"{name}/native_valid", value["valid"].all())
    checks.condition(
        f"{name}/unscaled_original_residual",
        np.isfinite(value["unscaled_residual_absolute"]).all()
        and (value["unscaled_residual_absolute"] <= budget).all(),
        maximum_absolute=float(np.max(value["unscaled_residual_absolute"])),
        tolerance=options.tolerance,
        absolute_tolerance=options.absolute_tolerance,
    )
    if reference is not None:
        checks.close(f"{name}/q", value["q"], reference["q"], rtol=1e-8, atol=1e-14)
        if "peak" in value:
            checks.close(f"{name}/peak", value["peak"], reference["peak"], rtol=1e-8, atol=1e-12)
    if zero:
        for key in ("q", "peak", "rhs", "unscaled_residual_absolute"):
            if key in value:
                checks.condition(f"{name}/zero/{key}", np.count_nonzero(value[key]) == 0)


def _boundary_snapshot(native, system, options):
    api, np = native.api, native.api.np
    native.kernel_full_residual(
        options.young, options.tolerance, options.absolute_tolerance, system.state, system.info, False, False
    )
    q, rhs = _scalar_vectors(api, system.state.displacement), _scalar_vectors(api, system.state.rhs)
    residual = system.matrix @ q - rhs
    return {
        "q": q,
        "rhs": rhs,
        "valid": api.qd_to_numpy(system.state.valid),
        "native_residual_norm_squared": api.qd_to_numpy(system.state.residual_norm_squared),
        "unscaled_residual_absolute": np.linalg.norm(residual, axis=0),
        "rhs_free_norm": np.linalg.norm(rhs, axis=0),
    }


def _run_boundary(native, writer, checks, hardware, args):
    """Probe declared shared capacity with a separate, nonphysical native SPD system."""
    api, np = native.api, native.api.np
    n_blocks, n_envs = args.boundary_blocks, args.boundary_envs
    prefix = f"boundary-{n_blocks}"
    shared_bytes = n_blocks * 3 * 8
    metadata = {
        "blocks": n_blocks,
        "raw_scalar_dofs": 3 * n_blocks,
        "environments": n_envs,
        "declared_shared_bytes": shared_bytes,
        "default_shared_limit": hardware["properties"]["shared_memory_per_block"],
        "optin_shared_limit": hardware["properties"]["shared_memory_per_block_optin"],
        "compile_launch_status": "not_attempted",
        "cooperative_compile_launch_supported": False,
        "physical_qualification": False,
        "construction": "Host chain CSR only; QD SPD diagonal blocks L@L.T, weak native neighbor couplings, QD rhs=young*K@known_q, final RHS column zero.",
        "only_failed": False,
    }
    writer.json(f"{prefix}/launch.json", metadata)
    if metadata["default_shared_limit"] is None or shared_bytes > metadata["default_shared_limit"]:
        raise RuntimeError(f"Default shared-memory limit does not admit the declared {shared_bytes}-byte boundary.")
    options = SimpleNamespace(young=7.0, tolerance=1e-10, absolute_tolerance=1e-12)
    nodes = np.arange(n_blocks, dtype=api.gs.np_int)
    pairs = np.unique(
        np.concatenate(
            (
                np.column_stack((nodes, nodes)),
                np.column_stack((nodes[:-1], nodes[1:])),
                np.column_stack((nodes[1:], nodes[:-1])),
            )
        ),
        axis=0,
    )
    row_start = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=n_blocks))]
    info = native.allocate_stress_info(n_blocks, 0, 0, len(pairs), 0)
    info.vertices.fill(0)
    info.pins.fill(-1)
    info.is_free.fill(1)
    info.row_start.from_numpy(row_start.astype(api.gs.np_int, copy=False))
    info.columns.from_numpy(pairs[:, 1])
    native.kernel_boundary_matrix(info.row_start, info.columns, info.stiffness)
    factor = native.StressFactor(info, scaling=True)
    state = native.create_stress_state(n_blocks, n_envs)
    exact_q = api.V_VEC(3, dtype=api.gs.qd_float, shape=(n_blocks, n_envs))
    native.kernel_boundary_rhs(info.row_start, info.columns, info.stiffness, state.rhs, exact_q, options.young)
    system = System(prefix, info, factor, state, None, matrix=_matrix(api, info, young=options.young))
    expected_q = _scalar_vectors(api, exact_q)
    writer.matrix(f"{prefix}/native_original_K.npz", system.matrix)
    writer.arrays(f"{prefix}/known_solution.npz", q=expected_q)
    writer.json("default_dispatch_source.json", {"source": inspect.getsource(type(factor).solve)})
    start_check = len(checks.rows)
    baseline = None
    for method in ("default", "cooperative"):
        native.kernel_boundary_rhs(info.row_start, info.columns, info.stiffness, state.rhs, exact_q, options.young)
        native.kernel_direct_init(state)
        api.qd.sync()
        begin = time.perf_counter()
        try:
            _dispatch(native, system, method, options.young)
            api.qd.sync()
        except BaseException:
            metadata.update(
                compile_launch_status=f"{method}_failed",
                failed_method=method,
                traceback=traceback.format_exc(),
            )
            writer.json(f"{prefix}/launch.json", metadata)
            raise
        metadata[f"{method}_first_dispatch_seconds_including_JIT"] = time.perf_counter() - begin
        metadata["compile_launch_status"] = f"{method}_completed"
        if method == "cooperative":
            metadata["cooperative_compile_launch_supported"] = True
        writer.json(f"{prefix}/launch.json", metadata)
        current = _boundary_snapshot(native, system, options)
        writer.arrays(f"{prefix}/{method}/current.npz", **current)
        _check_snapshot(api, checks, f"{prefix}/{method}/current", current, options, baseline)
        checks.close(f"{prefix}/{method}/known_q", current["q"], expected_q, rtol=1e-12, atol=1e-13)
        checks.condition(f"{prefix}/{method}/zero_column", np.count_nonzero(current["q"][:, -1]) == 0)
        if baseline is None:
            baseline = current
        native.kernel_direct_init(state)
        _dispatch(native, system, method, options.young)
        repeated = _boundary_snapshot(native, system, options)
        _check_snapshot(api, checks, f"{prefix}/{method}/repeat", repeated, options, baseline)
        state.rhs.fill(0)
        native.kernel_direct_init(state)
        _dispatch(native, system, method, options.young)
        released = _boundary_snapshot(native, system, options)
        writer.arrays(f"{prefix}/{method}/released_rhs.npz", **released)
        _check_snapshot(api, checks, f"{prefix}/{method}/release", released, options, zero=True)
        native.kernel_boundary_rhs(info.row_start, info.columns, info.stiffness, state.rhs, exact_q, options.young)
        native.kernel_direct_init(state)
        _dispatch(native, system, method, options.young)
        restored = _boundary_snapshot(native, system, options)
        _check_snapshot(api, checks, f"{prefix}/{method}/restore", restored, options, baseline)
        checks.flush(writer)
    metadata["numerical_passed"] = all(row["passed"] for row in checks.rows[start_check:])
    metadata["compile_launch_status"] = "cooperative_completed"
    writer.json(f"{prefix}/launch.json", metadata)
    if not metadata["numerical_passed"]:
        raise ArithmeticError("Native SPD boundary numerical checks failed; no boundary timings are admitted.")
    timings = []
    if args.timing:
        for _ in range(args.warmup):
            for method in ("default", "cooperative"):
                native.kernel_direct_init(state)
                _dispatch(native, system, method, options.young)
                api.qd.sync()
        for pair in range(args.repeats):
            order = ("default", "cooperative") if pair % 2 == 0 else ("cooperative", "default")
            for position, method in enumerate(order):
                native.kernel_direct_init(state)
                api.qd.sync()
                begin = time.perf_counter()
                _dispatch(native, system, method, options.young)
                api.qd.sync()
                seconds = time.perf_counter() - begin
                value = _boundary_snapshot(native, system, options)
                _check_snapshot(api, checks, f"timing/{prefix}/{pair}/{method}", value, options)
                timings.append(
                    {"system": prefix, "pair": pair, "order": position, "method": method, "seconds": seconds}
                )
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--bank", choices=("published43", "static16", "static64"), default="published43")
    parser.add_argument("--tile", type=int, default=1)
    parser.add_argument("--feature", type=int, choices=range(12), default=0)
    parser.add_argument("--timing", action="store_true", help="Also time paired factor solves, excluding preparation.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--boundary-blocks", type=int, choices=(2048,))
    parser.add_argument("--boundary-envs", type=int, default=5)
    parser.add_argument(
        "--boundary-only", action="store_true", help="Run only the nonphysical 2048-block boundary probe."
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Use Slurm rtx-mid with image 1_26; scientific imports follow this guard.")
    if min(args.tile, args.warmup, args.repeats) < 1 or args.boundary_envs < 2:
        parser.error("tile, warmup and repeats must be positive; boundary-envs requires a nonzero and a zero column.")
    if args.boundary_only:
        args.boundary_blocks = 2048
    output, reference = args.output.expanduser().resolve(), args.reference.expanduser().resolve()
    if output.is_relative_to(SOURCE) or (output.exists() and any(output.iterdir())):
        parser.error("Use a new/empty project data output directory outside the source checkout.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((reference / "manifest.json").read_text())
    if not manifest["passed"] or manifest["schema"] != "genesis.cpu12-reference.v1":
        parser.error("Use a successful frozen CPU12 v1 input export.")
    for relative, record in manifest["files"].items():
        if _hash(reference / relative) != record["sha256"]:
            raise ArithmeticError(f"Frozen reference artifact changed: {relative}")
    native = _native_api()
    api, np = native.api, native.api.np
    writer, checks = api.Writer(output, api), Checks(api)
    source_files = {
        str(path): _hash(path)
        for path in sorted((Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress").glob("*.py"))
    }
    for path in (
        Path(__file__),
        Path(build_native_space.__code__.co_filename),
        Path(_tiled_cases.__code__.co_filename),
    ):
        source_files[str(path.resolve())] = _hash(path)
    identity = {
        "kind": "native_cpu12_existing_cooperative_kernel_probe",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files": source_files,
        "reference_manifest_sha256": _hash(reference / "manifest.json"),
        "bank": args.bank,
        "tile": args.tile,
        "feature": args.feature,
        "boundary_blocks": args.boundary_blocks,
        "boundary_only": args.boundary_only,
        "precision": "64",
        "scope": "Actual 1174/1732 blocks unless boundary-only; optional nonphysical 2048 boundary. No production change.",
        "timing_scope": "Factor dispatch plus completion fence; preparation, residuals, peak, transfers and JIT excluded.",
        "oracle_policy": "No CPU numerical matrices or solutions enter the device; residual matrices are native readbacks.",
    }
    writer.json("start.json", identity)
    try:
        api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
        hardware = _hardware(native)
        writer.json("hardware.json", hardware)
        shared_limit = hardware["properties"]["shared_memory_per_block"]
        if shared_limit is None or shared_limit < 1732 * 3 * 8:
            raise RuntimeError("The reported default per-block shared-memory limit cannot admit the 1732-block probe.")
        timings = []
        if not args.boundary_only:
            space = build_native_space(api, reference, output, output_mode="max")
            cases = _tiled_cases(np, _load(np, reference / f"cases/{args.bank}.npz"), args.tile)
            writer.arrays("cases.npz", **cases)
            n_envs, n_sources = len(cases["case_names"]), len(cases["radius"])
            sources = _source_inputs(api, cases)
            cap = api.MaterialCap(
                int(manifest["configuration"]["quadrature"]),
                n_sources,
                len(space.model_input["vertices"]),
                ncoarse=int(space.model_input["ncoarse"]),
                subdiv=int(space.model_input["subdiv"]),
                thickness=float(space.model_input["thickness"]),
                vertex_stride=2,
            )
            geometry = api.build_material_cap_geometry(
                space.model.info.vertices,
                _integer(api, space.model_input["face_tables"]),
                space.model.shell.directors,
                ncoarse=int(space.model_input["ncoarse"]),
                subdiv=int(space.model_input["subdiv"]),
                vertex_stride=2,
            )
            selector = api.AtlasSelection(n_sources, n_envs, space.model.info.vertices.shape[0])
            coverage = api.AtlasCoverage(space.basis.coarse_index, space.library.info.local_mask, n_sources, n_envs)
            operator = api.FeatureOperator(space.model.info, *space.library.maps[args.feature])
            projected_relief = api.PreparedFeatureInfo(
                projected_relief=api.V_MAT(3, 6, dtype=api.gs.qd_float, shape=(operator.info.vertices.shape[0],))
            )
            api.kernel_project_common_relief(projected_relief, space.model.info, operator.transform)
            mother_state = space.model.create_state(n_envs, "max")
            feature_state = native.create_stress_state(operator.info.vertices.shape[0], n_envs)
            systems = (
                System("mother", space.model.info, space.model.factor, mother_state, mother_state),
                System(
                    "feature", operator.info, operator.factor, feature_state, space.model.create_state(n_envs), operator
                ),
            )
            assert tuple(system.info.vertices.shape[0] for system in systems) == (1732, 1174)
            omega = _vector(api, np.zeros((n_envs, 3)))
            environment_valid = api.V(dtype=api.gs.qd_int, shape=(n_envs,))
            writer.json("default_dispatch_source.json", {"source": inspect.getsource(type(space.model.factor).solve)})

            def prepare():
                cap.assemble(geometry, sources)
                config = space.selection_input
                selected = selector.run(
                    sources,
                    cap.state,
                    space.library.info.centers,
                    support_angle=float(config["support_angle"]),
                    narrow_fraction=float(config["narrow_fraction"]),
                    maximum_radius=float(config["maximum_radius"]),
                )
                covered = coverage.check(sources, cap.state, selected)
                native.kernel_material_cap_aggregate(sources, cap.state, mother_state, environment_valid)
                api.kernel_balance(omega, mother_state, space.model.info, space.model.options.cooperative_balance)
                native.kernel_group_wrench(args.feature, selected.grouped_force, feature_state, space.model.info)
                native.kernel_project_group(
                    args.feature, selected.grouped_force, feature_state, projected_relief, operator.transform
                )
                if (
                    not api.qd_to_numpy(cap.state.valid).all()
                    or not api.qd_to_numpy(selected.source_valid).all()
                    or not api.qd_to_numpy(selected.environment_valid).all()
                    or not api.qd_to_numpy(covered.environment_covered).all()
                    or not api.qd_to_numpy(environment_valid).all()
                ):
                    raise ArithmeticError("Native cap, selection, coverage or source ownership was rejected.")

            prepare()
            for system in systems:
                system.matrix = _matrix(api, system.info, young=space.model.options.young)
                writer.matrix(f"{system.name}/native_original_K.npz", system.matrix)
            for system in systems:
                baseline = None
                for method in ("default", "cooperative"):
                    prepare()
                    native.kernel_direct_init(system.state)
                    _dispatch(native, system, method, space.model.options.young)
                    value = _snapshot(native, space.model, system)
                    writer.arrays(f"{system.name}/{method}/current.npz", **value)
                    checks.condition(f"{system.name}/{method}/nonzero_current", np.count_nonzero(value["rhs"]) > 0)
                    _check_snapshot(
                        api, checks, f"{system.name}/{method}/current", value, space.model.options, baseline
                    )
                    if baseline is None:
                        baseline = value
                    # Reuse the same nonzero RHS and existing coordinates, without clearing displacement.
                    native.kernel_direct_init(system.state)
                    _dispatch(native, system, method, space.model.options.young)
                    repeated = _snapshot(native, space.model, system)
                    _check_snapshot(
                        api, checks, f"{system.name}/{method}/repeat", repeated, space.model.options, baseline
                    )
                    # An empty-current-RHS control follows a nonzero solve in the same state and environment slots.
                    system.state.rhs.fill(0)
                    native.kernel_direct_init(system.state)
                    _dispatch(native, system, method, space.model.options.young)
                    released = _snapshot(native, space.model, system)
                    writer.arrays(f"{system.name}/{method}/released_rhs.npz", **released)
                    _check_snapshot(
                        api, checks, f"{system.name}/{method}/release", released, space.model.options, zero=True
                    )
                    prepare()
                    native.kernel_direct_init(system.state)
                    _dispatch(native, system, method, space.model.options.young)
                    restored = _snapshot(native, space.model, system)
                    _check_snapshot(
                        api, checks, f"{system.name}/{method}/restore", restored, space.model.options, baseline
                    )
                checks.flush(writer)
            if not all(row["passed"] for row in checks.rows):
                raise ArithmeticError("Cooperative numerical smoke failed; no timing samples are admitted.")
            timings = []
            if args.timing:
                prepare()
                for system in systems:
                    for _ in range(args.warmup):
                        for method in ("default", "cooperative"):
                            native.kernel_direct_init(system.state)
                            _dispatch(native, system, method, space.model.options.young)
                            api.qd.sync()
                    for pair in range(args.repeats):
                        order = ("default", "cooperative") if pair % 2 == 0 else ("cooperative", "default")
                        for position, method in enumerate(order):
                            native.kernel_direct_init(system.state)
                            api.qd.sync()
                            begin = time.perf_counter()
                            _dispatch(native, system, method, space.model.options.young)
                            api.qd.sync()
                            seconds = time.perf_counter() - begin
                            snapshot = _snapshot(native, space.model, system)
                            _check_snapshot(
                                api, checks, f"timing/{system.name}/{pair}/{method}", snapshot, space.model.options
                            )
                            timings.append(
                                {
                                    "system": system.name,
                                    "pair": pair,
                                    "order": position,
                                    "method": method,
                                    "seconds": seconds,
                                }
                            )
        if args.boundary_blocks is not None:
            timings.extend(_run_boundary(native, writer, checks, hardware, args))
        writer.json("factor_timings.json", timings)
        for path, digest in source_files.items():
            checks.condition(f"source_unchanged/{path}", _hash(path) == digest)
        checks.flush(writer)
        result = {
            **identity,
            "passed": all(row["passed"] for row in checks.rows),
            "checks": len(checks.rows),
            "failed": [row["name"] for row in checks.rows if not row["passed"]],
            "tested_blocks": ([] if args.boundary_only else [1732, 1174])
            + ([] if args.boundary_blocks is None else [args.boundary_blocks]),
            "timing_enabled": args.timing,
            "candidate_cap_2048": (
                "Nonphysical SPD shared boundary executed; consult launch status and numerical checks."
                if args.boundary_blocks is not None
                else "Not executed; actual-size checks do not certify the 2048-block boundary."
            ),
            "files": writer.inventory,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"output": str(output), "passed": result["passed"], "checks": result["checks"]}))
        if not result["passed"]:
            raise ArithmeticError("Native cooperative probe rejected work or sources changed; inspect checks.json.")
    except BaseException:
        checks.flush(writer)
        writer.json("failure.json", {"traceback": traceback.format_exc(), "completed_checks": len(checks.rows)})
        raise


if __name__ == "__main__":
    main()
