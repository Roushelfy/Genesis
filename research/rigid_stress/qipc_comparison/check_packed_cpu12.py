"""Compare sequential and packed CPU12 feature-factor solves on native current loads.

Run only in Slurm rtx-mid/image 1_26. Optional paired timings cover the factor
component, including packed gather/scatter, and do not establish full-path speed.
"""

import argparse
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from .check_cpu12_cooperative import _hardware
from .check_cpu12_gpu import (
    SOURCE,
    Checks,
    _api,
    _hash,
    _integer,
    _load,
    _matrix,
    _outputs,
    _scalar_vectors,
    _source_inputs,
    build_native_space,
)


def _packed_api():
    api = _api()
    import torch

    from genesis.engine.solvers.rigid.stress.factor import kernel_solve_cooperative
    from genesis.engine.solvers.rigid.stress.packed_factor import PackedStressFactor
    from genesis.engine.solvers.rigid.stress.prepared_atlas import (
        kernel_admit_environments,
        kernel_group_wrench,
        kernel_merge_group_acceptance,
        kernel_prepare_group_solve,
        kernel_project_group,
    )
    from genesis.engine.solvers.rigid.stress.solve import kernel_direct_init, kernel_full_residual

    return SimpleNamespace(**locals())


@dataclass
class Experiment:
    native: object
    space: object
    prepared: object
    packed: object
    sources: object
    zero_sources: object
    cap: object
    geometry: object
    selector: object
    coverage: object
    parent: object
    matrices: tuple
    strategy: str
    selected: object = None
    covered: object = None
    expected_active: object = None

    def prepare(self, *, zero=False):
        api, np = self.native.api, self.native.api.np
        sources = self.zero_sources if zero else self.sources
        self.cap.assemble(self.geometry, sources)
        config = self.space.selection_input
        self.selected = self.selector.run(
            sources,
            self.cap.state,
            self.space.library.info.centers,
            strategy=self.strategy,
            support_angle=float(config["support_angle"]),
            narrow_fraction=float(config["narrow_fraction"]),
            maximum_radius=float(config["maximum_radius"]),
        )
        self.covered = self.coverage.check(sources, self.cap.state, self.selected)
        if (
            not api.qd_to_numpy(self.cap.state.valid).all()
            or not api.qd_to_numpy(self.selected.source_valid).all()
            or not api.qd_to_numpy(self.selected.environment_valid).all()
            or not api.qd_to_numpy(self.covered.environment_covered).all()
        ):
            raise ArithmeticError("Native cap, selector or coverage rejected current loads.")
        for i_f, (operator, feature, state) in enumerate(
            zip(self.prepared.operators, self.prepared.features, self.prepared.states, strict=True)
        ):
            self.native.kernel_group_wrench(i_f, self.selected.grouped_force, state, self.space.model.info)
            self.native.kernel_project_group(i_f, self.selected.grouped_force, state, feature, operator.transform)
            self.native.kernel_direct_init(state)
            self.native.kernel_prepare_group_solve(state)
        self.expected_active = np.stack([~api.qd_to_numpy(state.valid) for state in self.prepared.states])

    def solve(self, method):
        young = self.space.model.options.young
        if method == "packed":
            self.packed.solve(young)
        else:
            for operator, state in zip(self.prepared.operators, self.prepared.states, strict=True):
                if method == "default":
                    operator.factor.solve(young, state, operator.info, cooperative=True, only_failed=True)
                else:
                    self.native.kernel_solve_cooperative(
                        young, state, operator.info, operator.factor.info, operator.info.vertices.shape[0], True
                    )

    def capture(self, checks, prefix, reference=None, *, zero=False):
        api, np = self.native.api, self.native.api.np
        options = self.space.model.options
        group_q, group_rhs, residuals, native_residuals = [], [], [], []
        active = np.stack([api.qd_to_numpy(state.active) for state in self.prepared.states])
        fallbacks = np.stack([api.qd_to_numpy(state.fallbacks) for state in self.prepared.states])
        checks.exact(f"{prefix}/active", active, self.expected_active.astype(api.gs.np_int))
        checks.exact(f"{prefix}/primary_factor_metadata", fallbacks, self.expected_active.astype(api.gs.np_int))
        for i_f, (operator, state, matrix) in enumerate(
            zip(self.prepared.operators, self.prepared.states, self.matrices, strict=True)
        ):
            self.native.kernel_full_residual(
                options.young, options.tolerance, options.absolute_tolerance, state, operator.info, False, False
            )
            q, rhs = _scalar_vectors(api, state.displacement), _scalar_vectors(api, state.rhs)
            norm = np.linalg.norm(matrix @ q - rhs, axis=0)
            budget = np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))
            checks.condition(f"{prefix}/feature-{i_f:02d}/valid", api.qd_to_numpy(state.valid).all())
            checks.condition(
                f"{prefix}/feature-{i_f:02d}/original_residual",
                np.isfinite(norm).all() and (norm <= budget).all(),
                maximum_absolute=float(norm.max()),
                tolerance=options.tolerance,
                absolute_tolerance=options.absolute_tolerance,
            )
            if reference is not None:
                checks.close(f"{prefix}/feature-{i_f:02d}/q", q, reference["group_q"][i_f], rtol=1e-8, atol=1e-14)
                checks.close(f"{prefix}/feature-{i_f:02d}/rhs", rhs, reference["group_rhs"][i_f], atol=1e-14)
            group_q.append(q)
            group_rhs.append(rhs)
            residuals.append(norm)
            native_residuals.append(api.qd_to_numpy(state.residual_norm_squared))
        api.kernel_sum_feature_forces(self.selected.grouped_force, self.parent)
        api.kernel_balance(self.prepared.omega, self.parent, self.space.model.info, options.cooperative_balance)
        self.native.kernel_direct_init(self.parent)
        self.parent.displacement.fill(0)
        for operator, state in zip(self.prepared.operators, self.prepared.states, strict=True):
            operator.lift(state.displacement, self.parent.displacement, add=True)
            self.native.kernel_merge_group_acceptance(state, self.parent)
        self.native.kernel_admit_environments(self.covered.environment_covered, self.parent)
        self.space.model.evaluate_stress(options, self.parent)
        output = _outputs(api, self.space.model, self.parent)
        checks.condition(f"{prefix}/merged_valid", output["valid"].all())
        if reference is not None:
            for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
                checks.close(f"{prefix}/merged/{key}", output[key], reference["output"][key], rtol=1e-8, atol=1e-13)
        if zero:
            for key in ("q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
                checks.condition(f"{prefix}/released/{key}", np.count_nonzero(output[key]) == 0)
            checks.condition(f"{prefix}/released/group_q", all(np.count_nonzero(q) == 0 for q in group_q))
        return {
            "group_q": tuple(group_q),
            "group_rhs": tuple(group_rhs),
            "original_residual_absolute": np.stack(residuals),
            "native_residual_norm_squared": np.stack(native_residuals),
            "active": active,
            "fallbacks": fallbacks,
            "output": output,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--strategy", choices=("continuous", "nearest", "narrow"), default="continuous")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Use Slurm rtx-mid/image 1_26; scientific imports follow this guard.")
    if min(args.warmup, args.repeats) < 1:
        parser.error("warmup and repeats must be positive.")
    output, reference = args.output.expanduser().resolve(), args.reference.expanduser().resolve()
    if output.is_relative_to(SOURCE) or (output.exists() and any(output.iterdir())):
        parser.error("Use a new/empty project data output directory outside the source checkout.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((reference / "manifest.json").read_text())
    if not manifest["passed"] or manifest["schema"] != "genesis.cpu12-reference.v1":
        parser.error("Use a successful frozen CPU12 v1 export.")
    for relative, record in manifest["files"].items():
        if _hash(reference / relative) != record["sha256"]:
            raise ArithmeticError(f"Frozen reference changed: {relative}")
    native = _packed_api()
    api, np = native.api, native.api.np
    writer, checks = api.Writer(output, api), Checks(api)
    source_files = {
        str(path): _hash(path)
        for path in sorted((Path(api.gs.__file__).resolve().parent / "engine/solvers/rigid/stress").glob("*.py"))
    }
    for path in (Path(__file__), Path(build_native_space.__code__.co_filename), Path(_hardware.__code__.co_filename)):
        source_files[str(path.resolve())] = _hash(path)
    identity = {
        "kind": "native_cpu12_packed_factor_component_control",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files": source_files,
        "reference_manifest_sha256": _hash(reference / "manifest.json"),
        "bank": "published43",
        "strategy": args.strategy,
        "precision": "64",
        "methods": ["default", "sequential_cooperative", "packed"],
        "scope": "Factor component only; no complete-module or complete-scene speed claim.",
        "timing_scope": "Dispatch and completion including packed gather/scatter; loads, projection, residuals and recovery excluded.",
        "release_control": "Nonzero native sources, then zero current source forces, then original sources in reused buffers.",
        "oracle_policy": "Only physical frozen input and integer topology enter native execution; matrices are native readbacks.",
    }
    writer.json("start.json", identity)
    try:
        api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
        writer.json("hardware.json", _hardware(native))
        space = build_native_space(api, reference, output)
        cases = _load(np, reference / "cases/published43.npz")
        n_envs, n_sources = len(cases["case_names"]), len(cases["radius"])
        prepared = api.PreparedAtlas(space.model, space.library.maps, n_envs)
        sources, zero_sources = _source_inputs(api, cases), _source_inputs(api, cases)
        zero_sources.force.fill(0)
        zero_sources.normal_force.fill(0)
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
        started = time.perf_counter()
        packed = native.PackedStressFactor(prepared.operators, prepared.states)
        api.qd.sync()
        writer.json(
            "packing.json",
            {
                "seconds_including_copy_JIT": time.perf_counter() - started,
                "supported": packed.supported,
                "unsupported_reason": packed.unsupported_reason,
                "bucket": packed.bucket,
                "memory": asdict(packed.memory_inventory()),
            },
        )
        if not packed.supported:
            raise RuntimeError(packed.unsupported_reason)
        matrices = tuple(
            _matrix(api, operator.info, young=space.model.options.young) for operator in prepared.operators
        )
        experiment = Experiment(
            native,
            space,
            prepared,
            packed,
            sources,
            zero_sources,
            cap,
            geometry,
            api.AtlasSelection(n_sources, n_envs, space.model.info.vertices.shape[0]),
            api.AtlasCoverage(space.basis.coarse_index, space.library.info.local_mask, n_sources, n_envs),
            space.model.create_state(n_envs, "full"),
            matrices,
            args.strategy,
        )
        baselines = {}
        for method in identity["methods"]:
            for phase in ("loaded", "zero", "reload"):
                experiment.prepare(zero=phase == "zero")
                if method == "packed" and phase == "zero":
                    packed.state.work.fill(1)
                experiment.solve(method)
                target = baselines.get("loaded" if phase == "reload" else phase)
                value = experiment.capture(checks, f"{method}/{phase}", target, zero=phase == "zero")
                writer.arrays(
                    f"{method}/{phase}/groups.npz",
                    q=np.concatenate(value["group_q"]),
                    rhs=np.concatenate(value["group_rhs"]),
                    original_residual_absolute=value["original_residual_absolute"],
                    native_residual_norm_squared=value["native_residual_norm_squared"],
                    active=value["active"],
                    fallbacks=value["fallbacks"],
                    selection_weights=api.qd_to_numpy(experiment.selected.weights),
                    selected_positive_groups=api.qd_to_numpy(experiment.selected.active_groups),
                    source_feature_contributions=api.qd_to_numpy(experiment.selected.contribution_count),
                )
                writer.arrays(f"{method}/{phase}/output.npz", **value["output"])
                if method == "default":
                    baselines[phase] = value
                checks.flush(writer)
        if not all(row["passed"] for row in checks.rows):
            raise ArithmeticError("Packed factor numerical controls failed; no component timings are admitted.")
        timings = []
        if args.timing:
            for _ in range(args.warmup):
                for method in identity["methods"]:
                    experiment.prepare()
                    experiment.solve(method)
                    api.qd.sync()
            for pair in range(args.repeats):
                methods = identity["methods"] if pair % 2 == 0 else list(reversed(identity["methods"]))
                for position, method in enumerate(methods):
                    experiment.prepare()
                    api.qd.sync()
                    begin = time.perf_counter()
                    experiment.solve(method)
                    api.qd.sync()
                    seconds = time.perf_counter() - begin
                    experiment.capture(checks, f"timing/{pair}/{method}", baselines["loaded"])
                    timings.append({"pair": pair, "order": position, "method": method, "seconds": seconds})
        writer.json("factor_timings.json", timings)
        for path, digest in source_files.items():
            checks.condition(f"source_unchanged/{path}", _hash(path) == digest)
        checks.flush(writer)
        result = {
            **identity,
            "passed": all(row["passed"] for row in checks.rows),
            "checks": len(checks.rows),
            "failed": [row["name"] for row in checks.rows if not row["passed"]],
            "packing_memory": asdict(packed.memory_inventory()),
            "timing_enabled": args.timing,
            "files": writer.inventory,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"output": str(output), "passed": result["passed"], "checks": result["checks"]}))
        if not result["passed"]:
            raise ArithmeticError("Packed factor checks failed or source changed; inspect checks.json.")
    except BaseException:
        checks.flush(writer)
        writer.json("failure.json", {"traceback": traceback.format_exc(), "completed_checks": len(checks.rows)})
        raise


if __name__ == "__main__":
    main()
