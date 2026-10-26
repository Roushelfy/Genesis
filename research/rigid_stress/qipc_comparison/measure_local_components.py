"""Time four equivalent native local-feature recovery component paths.

Replay four saved, genuinely scattered contact-source loads. This worker never
refits pressure or selects features. Run only in Slurm rtx-mid / image 1_26.
The measurements are fixed-load component controls, not full-scene headlines.
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.engine.solvers.rigid.stress.feature_atlas import (
    FeatureAtlas,
    FeatureAtlasInfo,
    kernel_atlas_begin,
    kernel_fallback_add,
    kernel_fallback_rhs,
    kernel_group_begin,
    kernel_group_diagnostics,
    kernel_group_lift,
    kernel_group_rhs,
    kernel_group_wrench,
    kernel_primary_factor_done,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.solve import kernel_balance, kernel_direct_init, kernel_full_residual
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V, iter_data
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from .artifacts import file_id, runtime_identity, write_json
from .audit_local_keyframes import kernel_timing_retain_valid, relative_per_environment, resolve_asset
from .check_local_geometry import tensor_norm, tensor_peak


@qd.kernel
def kernel_load_source(index: int, raw: qd.Tensor, grouped: qd.Tensor, target: StressState, groups: qd.Tensor):
    for i_n, i_b in qd.ndrange(target.force.shape[0], target.force.shape[1]):
        target.force[i_n, i_b] = raw[index, i_n, i_b]
    for i_n, i_g, i_b in qd.ndrange(groups.shape[0], groups.shape[1], groups.shape[2]):
        groups[i_n, i_g, i_b] = grouped[index, i_n, i_g, i_b]


@qd.kernel
def kernel_aggregate_sources(raw: qd.Tensor, grouped: qd.Tensor, target: StressState, groups: qd.Tensor):
    for i_n, i_b in qd.ndrange(target.force.shape[0], target.force.shape[1]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for i_c in range(raw.shape[0]):
            value += raw[i_c, i_n, i_b]
        target.force[i_n, i_b] = value
    for i_n, i_g, i_b in qd.ndrange(groups.shape[0], groups.shape[1], groups.shape[2]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for i_c in range(grouped.shape[0]):
            value += grouped[i_c, i_n, i_g, i_b]
        groups[i_n, i_g, i_b] = value


@qd.kernel
def kernel_clear_displacement(state: StressState):
    for i_b in range(state.valid.shape[0]):
        state.valid[i_b] = True
        state.active[i_b] = 0
        state.iterations[i_b] = 0
        state.corrections[i_b] = 0
        state.fallbacks[i_b] = 0
        state.residual_norm_squared[i_b] = 0.0
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.valid.shape[0]):
        state.displacement[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_clear_linear_fields(state: StressState):
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.valid.shape[0]):
        state.displacement[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_e, i_q, i_b in qd.ndrange(state.stress_tensor.shape[0], 4, state.valid.shape[0]):
        state.stress_tensor[i_e, i_q, i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_e, i_q, i_b in qd.ndrange(state.membrane_resultant.shape[0], 4, state.valid.shape[0]):
        state.membrane_resultant[i_e, i_q, i_b] = qd.Vector.zero(gs.qd_float, 6)
        state.bending_resultant[i_e, i_q, i_b] = qd.Vector.zero(gs.qd_float, 6)
        state.transverse_resultant[i_e, i_q, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_b in range(state.valid.shape[0]):
        state.valid[i_b] = True
        state.iterations[i_b] = 0
        state.corrections[i_b] = 0
        state.fallbacks[i_b] = 0


@qd.kernel
def kernel_add_linear_fields(source: StressState, target: StressState):
    for i_n, i_b in qd.ndrange(target.displacement.shape[0], target.valid.shape[0]):
        target.displacement[i_n, i_b] += source.displacement[i_n, i_b]
    for i_e, i_q, i_b in qd.ndrange(target.stress_tensor.shape[0], 4, target.valid.shape[0]):
        target.stress_tensor[i_e, i_q, i_b] += source.stress_tensor[i_e, i_q, i_b]
    for i_e, i_q, i_b in qd.ndrange(target.membrane_resultant.shape[0], 4, target.valid.shape[0]):
        target.membrane_resultant[i_e, i_q, i_b] += source.membrane_resultant[i_e, i_q, i_b]
        target.bending_resultant[i_e, i_q, i_b] += source.bending_resultant[i_e, i_q, i_b]
        target.transverse_resultant[i_e, i_q, i_b] += source.transverse_resultant[i_e, i_q, i_b]
    for i_b in range(target.valid.shape[0]):
        target.valid[i_b] = target.valid[i_b] and source.valid[i_b]
        target.iterations[i_b] += source.iterations[i_b]
        target.corrections[i_b] += source.corrections[i_b]
        target.fallbacks[i_b] += source.fallbacks[i_b]


@qd.kernel
def kernel_add_displacement(source: StressState, target: StressState):
    for i_n, i_b in qd.ndrange(target.displacement.shape[0], target.valid.shape[0]):
        target.displacement[i_n, i_b] += source.displacement[i_n, i_b]
    for i_b in range(target.valid.shape[0]):
        target.valid[i_b] = target.valid[i_b] and source.valid[i_b]
        target.iterations[i_b] += source.iterations[i_b]
        target.corrections[i_b] += source.corrections[i_b]
        target.fallbacks[i_b] += source.fallbacks[i_b]


@qd.kernel
def kernel_spin_displacement(young: float, omega: qd.Tensor, atlas: FeatureAtlasInfo, target: StressState):
    for i_n, i_b in qd.ndrange(target.displacement.shape[0], target.valid.shape[0]):
        w = omega[i_b]
        terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        target.displacement[i_n, i_b] = atlas.centrifugal[i_n] @ terms / young


@qd.kernel
def kernel_tensor_von_mises(state: StressState):
    """Use the native six-component von Mises formula after tensor summation.

    Native solve/shell scans inline this same criterion; neither exposes a
    standalone tensor-input function. This bounded control supports only VM.
    """
    for i_b in range(state.valid.shape[0]):
        state.peak[i_b] = 0.0
    for i_e, i_q, i_b in qd.ndrange(state.stress_tensor.shape[0], 4, state.valid.shape[0]):
        s = state.stress_tensor[i_e, i_q, i_b]
        squared = 0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2)
        squared += 3.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
        vm = qd.sqrt(qd.max(0.0, squared))
        state.von_mises[i_e, i_q, i_b] = qd.Vector([vm])
        qd.atomic_max(state.peak[i_b], vm)
        if not state.valid[i_b]:
            state.von_mises[i_e, i_q, i_b] = qd.Vector([float("nan")])


def find_frame(native, metadata, label):
    matches = [row for row in metadata["rows"] if row["name"] == label]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one saved {label}; four real saved contact sources are required.")
    recorded = Path(matches[0]["frame"]["path"])
    candidates = (native / recorded.name, recorded, native / f"{label}.npz")
    for path in candidates:
        if path.is_file():
            if file_id(path)["sha256"] != matches[0]["frame"]["sha256"]:
                raise ValueError(f"Saved source hash changed: {path}.")
            with np.load(path, allow_pickle=False) as arrays:
                frame = {name: arrays[name].copy() for name in arrays.files}
            return path, frame
    raise FileNotFoundError(f"Cannot locate saved real scatter for {label} in {native}.")


def fields(state):
    names = ["displacement", "stress_tensor", "peak", "von_mises"]
    if state.membrane_resultant.shape[0]:
        names += ["membrane_resultant", "bending_resultant", "transverse_resultant"]
    return {name: qd_to_numpy(getattr(state, name), transpose=True, copy=True) for name in names}


def compare(actual, expected):
    errors = {name: relative_per_environment(value, expected[name]).tolist() for name, value in actual.items()}
    finite = all(np.isfinite(value).all() for value in actual.values())
    return {
        "relative_errors": errors,
        "finite": finite,
        "passed": bool(finite and all(max(v) <= 1e-8 for v in errors.values())),
    }


def tile(array, n_envs, axis=0):
    return np.take(array, np.arange(n_envs) % array.shape[axis], axis=axis)


def linear_field_norm(name, value, weights):
    if name == "stress_tensor":
        return tensor_norm(value, weights)
    return np.linalg.norm(value.reshape((len(value), -1)), axis=1)


def signed_field_diagnostics(actual, reference, source_norms, weights):
    linear = {}
    for name, source_norm in source_norms.items():
        error = linear_field_norm(name, actual[name] - reference[name], weights)
        reference_norm = linear_field_norm(name, reference[name], weights)
        linear[name] = {
            "absolute_error_norm": error.tolist(),
            "total_reference_norm": reference_norm.tolist(),
            "source_norm_sum": source_norm.tolist(),
            "total_relative_error": (error / np.maximum(reference_norm, 1e-30)).tolist(),
            "source_normalized_error": (error / np.maximum(source_norm, 1e-30)).tolist(),
            "cancellation_ratio": (reference_norm / np.maximum(source_norm, 1e-30)).tolist(),
        }
    peak_error = np.abs(actual["peak"] - reference["peak"])
    return {
        "linear_fields": linear,
        "peak_absolute_error": peak_error.tolist(),
        "peak_total_relative_error": (peak_error / np.maximum(np.abs(reference["peak"]), 1e-30)).tolist(),
        "spatial_budgets_diagnostic_only": {
            "tensor_0_10": (np.asarray(linear["stress_tensor"]["total_relative_error"]) <= 0.10).tolist(),
            "peak_0_05": (peak_error <= 0.05 * np.abs(reference["peak"])).tolist(),
        },
    }


def storage_inventory(mother, atlas, states, tensors):
    entries = list(mother.iter_info_data("mother"))
    entries += list(iter_data(mother.factor.info, "mother.factor"))
    entries += list(atlas.data("atlas"))
    for name, state in states.items():
        entries += list(iter_data(state, name))
    sizes, seen = {}, set()
    for item in entries:
        if id(item.value) in seen or 0 in item.value.shape:
            continue
        seen.add(id(item.value))
        view = qd_to_torch(item.value)
        sizes[item.name] = view.numel() * view.element_size()
    for name, tensor in tensors.items():
        view = qd_to_torch(tensor)
        sizes[name] = view.numel() * view.element_size()
    return {
        "declared_native_bytes": sum(sizes.values()),
        "by_field": sizes,
        "scope": "Logical persistent tensor storage; excludes constructor temporaries, allocator overhead and process GPU peak.",
    }


def measure(args, result):
    native = args.native.resolve()
    saved = json.loads((native / "result.json").read_text())
    labels = [args.label, *[f"{args.label}_separate_source_{i}" for i in range(4)], f"{args.label}_separate_spin"]
    loaded = [find_frame(native, saved, label) for label in labels]
    combined, *source_and_spin = [frame for _path, frame in loaded]
    sources, spin = source_and_spin[:4], source_and_spin[4]
    options_data = dict(saved["options"])
    options_data["mesh"] = resolve_asset(options_data["mesh"], native)
    if options_data.get("atlas_basis"):
        options_data["atlas_basis"] = resolve_asset(options_data["atlas_basis"], native)
    original_options = RigidStressOptions(**options_data)
    options = original_options.model_copy(
        update={"atlas_solver": args.solver, "method": "direct", "surface_inverse": False}
    )
    assert options.tolerance == 1e-8 and options.absolute_tolerance == 1e-11
    assert options.atlas_regions > 0 and options.history_size == 0
    assert all(np.count_nonzero(frame["contact_omega"]) == 0 for frame in sources)
    assert np.count_nonzero(spin["raw_force"]) == 0 and np.count_nonzero(spin["grouped_force"]) == 0
    np.testing.assert_array_equal(spin["contact_omega"], combined["contact_omega"])
    n_original = combined["raw_force"].shape[0]
    assert args.envs % n_original == 0, "Tile complete original environment banks without dropping original cases."
    n_boundary, n_groups, bank_envs, xyz = combined["grouped_force"].shape
    assert bank_envs == n_original and n_groups == options.atlas_regions + 1 and xyz == 3
    linear_names = [
        key[len("candidate_") :] for key in combined if key.startswith("candidate_") and key != "candidate_peak"
    ]
    expected = {name: tile(combined[f"candidate_{name}"], args.envs) for name in linear_names}
    expected["peak"] = tile(combined["candidate_peak"], args.envs)
    _peak, vm = tensor_peak(expected["stress_tensor"])
    expected["von_mises"] = vm[..., None]
    summed = {name: sum(frame[f"candidate_{name}"] for frame in source_and_spin) for name in linear_names}
    summed_peak, summed_vm = tensor_peak(summed["stress_tensor"])
    summed.update(peak=summed_peak, von_mises=summed_vm[..., None])
    untiled_expected = {name: value[np.arange(n_original)] for name, value in expected.items()}
    input_checks = {
        "saved_source_field_superposition": compare(summed, untiled_expected),
        "raw_force_sum": relative_per_environment(
            sum(frame["raw_force"] for frame in sources), combined["raw_force"]
        ).tolist(),
        "grouped_force_sum": relative_per_environment(
            sum(frame["grouped_force"] for frame in sources).transpose(2, 0, 1, 3),
            combined["grouped_force"].transpose(2, 0, 1, 3),
        ).tolist(),
    }
    assert input_checks["saved_source_field_superposition"]["passed"]
    assert max(input_checks["raw_force_sum"] + input_checks["grouped_force_sum"]) <= 1e-10
    coefficients = (1.0, -1.0 / 1.13, 0.0, 0.0) if args.signed_cancellation else (1.0, 1.0, 1.0, 1.0)
    raw_target = combined["raw_force"]
    signed_reference, reference_source_norms = None, None
    if args.signed_cancellation:
        raw_target = sum(
            coefficient * frame["raw_force"] for coefficient, frame in zip(coefficients, sources, strict=True)
        )

        def signed_fields(prefix):
            values = {
                name: tile(
                    sum(
                        coefficient * frame[f"{prefix}_{name}"]
                        for coefficient, frame in zip(coefficients, sources, strict=True)
                    ),
                    args.envs,
                )
                for name in linear_names
            }
            peak, vm = tensor_peak(values["stress_tensor"])
            return {**values, "peak": peak, "von_mises": vm[..., None]}

        expected = signed_fields("candidate")
        signed_reference = signed_fields("reference")
        weights = combined["sample_weights"]
        reference_source_norms = {
            name: tile(
                sum(
                    abs(coefficient) * linear_field_norm(name, frame[f"reference_{name}"], weights)
                    for coefficient, frame in zip(coefficients, sources, strict=True)
                ),
                args.envs,
            )
            for name in linear_names
        }
        candidate_source_norms = {
            name: tile(
                sum(
                    abs(coefficient) * linear_field_norm(name, frame[f"candidate_{name}"], weights)
                    for coefficient, frame in zip(coefficients, sources, strict=True)
                ),
                args.envs,
            )
            for name in linear_names
        }
        diagnostics = signed_field_diagnostics(expected, signed_reference, reference_source_norms, weights)
        ratios = np.asarray(diagnostics["linear_fields"]["stress_tensor"]["cancellation_ratio"])
        result["signed_cancellation"].update(
            coefficients=list(coefficients),
            reference_cancellation_ratio=ratios.tolist(),
            candidate_cancellation_ratio=(
                tensor_norm(expected["stress_tensor"], weights)
                / np.maximum(candidate_source_norms["stress_tensor"], 1e-30)
            ).tolist(),
            reference_cancellation_ratio_by_field={
                name: values["cancellation_ratio"] for name, values in diagnostics["linear_fields"].items()
            },
            candidate_cancellation_ratio_by_field={
                name: (linear_field_norm(name, expected[name], weights) / np.maximum(norm, 1e-30)).tolist()
                for name, norm in candidate_source_norms.items()
            },
            below_0_25=(ratios < 0.25).tolist(),
            near_cancellation_observed=bool((ratios < 0.25).any()),
            all_environments_below_0_25=bool((ratios < 0.25).all()),
            saved_candidate_vs_reference=diagnostics,
        )
        np.savez_compressed(
            args.output / "signed-algebra-targets.npz",
            coefficients=np.asarray(coefficients),
            raw_force=tile(raw_target, args.envs),
            **{f"candidate_{name}": value for name, value in expected.items()},
            **{f"reference_{name}": value for name, value in signed_reference.items()},
        )
    result.update(
        inputs=[{"label": label, **file_id(path)} for label, (path, _frame) in zip(labels, loaded, strict=True)],
        native_result=file_id(native / "result.json"),
        input_checks=input_checks,
        original_options=original_options.model_dump(mode="json"),
        options=options.model_dump(mode="json"),
        mesh=file_id(options.mesh),
        basis=file_id(options.atlas_basis) if options.atlas_basis is not None else None,
        native_qualification_passed=saved.get("passed"),
        original_environments=n_original,
        environment_tiling_indices=(np.arange(args.envs) % n_original).tolist(),
    )
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    started = time.perf_counter()
    mother = ShellModel(options) if options.backend == "shell" else StressModel(options)
    atlas = FeatureAtlas(mother, create_contacts(4, args.envs, options.contact_radius), options)
    qd.sync()
    result["preparation_seconds"] = time.perf_counter() - started
    result["atlas"] = asdict(atlas.metadata())
    started = time.perf_counter()
    assert atlas.metadata().mother_blocks == saved["atlas"]["mother_blocks"]
    assert atlas.metadata().coarse_nodes == saved["atlas"]["coarse_nodes"]
    for old, new in zip(saved["atlas"]["groups"], atlas.metadata().groups, strict=True):
        for name in ("blocks", "local_nodes", "stiffness_entries", "transform_entries", "factor_entries"):
            assert old[name] == getattr(new, name), (name, old, new)
    boundary = qd_to_numpy(atlas.regions.info.boundary_nodes)
    n_blocks = mother.info.vertices.shape[0]
    assert n_boundary == len(boundary)
    raw_arrays = np.stack(
        [
            coefficient * tile(frame["raw_force"], args.envs).transpose(1, 0, 2)
            for coefficient, frame in zip(coefficients, sources, strict=True)
        ]
    )
    grouped_arrays = np.stack(
        [
            coefficient * tile(frame["grouped_force"], args.envs, axis=2)
            for coefficient, frame in zip(coefficients, sources, strict=True)
        ]
    )
    assert raw_arrays.shape == (4, n_blocks, args.envs, 3)
    assert grouped_arrays.shape == (4, n_boundary, n_groups, args.envs, 3)
    expanded = np.zeros((args.envs, n_blocks, 3))
    expanded[:, boundary] = grouped_arrays.sum(axis=(0, 2)).transpose(1, 0, 2)
    assert max(relative_per_environment(expanded, tile(raw_target, args.envs))) <= 1e-10
    raw_bank = V_VEC(3, dtype=gs.qd_float, shape=(4, n_blocks, args.envs))
    grouped_bank = V_VEC(3, dtype=gs.qd_float, shape=(4, n_boundary, n_groups, args.envs))
    raw_bank.from_numpy(np.ascontiguousarray(raw_arrays, dtype=gs.np_float))
    grouped_bank.from_numpy(np.ascontiguousarray(grouped_arrays, dtype=gs.np_float))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    zero_omega = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    spin_omega = np.zeros_like(spin["contact_omega"]) if args.signed_cancellation else spin["contact_omega"]
    omega.from_numpy(np.ascontiguousarray(tile(spin_omega, args.envs), dtype=gs.np_float))
    zero_omega.fill(0)
    part = mother.create_state(args.envs, "full")
    outputs = {
        name: mother.create_state(args.envs, "full")
        for name in ("A_separate_sources", "B_grouped_tensors", "C_merged_displacement", "per_contact_common_q")
    }
    retained = {name: V(dtype=gs.qd_bool, shape=(args.envs,)) for name in outputs}
    coalesced = options.coalesced_residual if options.coalesced_residual is not None else args.envs >= 8192
    qd.sync()
    result["state_and_load_bank_setup_seconds"] = time.perf_counter() - started

    def solve_group(i_group, target):
        group = atlas.groups[i_group - 1] if i_group else None
        state = group.state if group else atlas.fallback_state
        info = group.operator.info if group else mother.info
        factor = group.operator.factor if group else mother.factor
        response = group.response if group else None
        kernel_group_wrench(i_group, state, atlas.state, mother.info, atlas.regions.info, atlas.regions.state)
        if group:
            kernel_group_rhs(
                i_group, state, group.info, group.operator.transform, atlas.regions.info, atlas.regions.state
            )
        else:
            kernel_fallback_rhs(state, mother.info, atlas.regions.info, atlas.regions.state)
        kernel_direct_init(state)
        kernel_group_begin(i_group, state, atlas.state)
        if response is not None:
            response.apply(options.young, i_group, atlas.regions.state.force, state)
        else:
            factor.solve(options.young, state, info, options.cooperative_solve, only_failed=True)
            kernel_primary_factor_done(state)
        kernel_full_residual(
            options.young,
            options.tolerance,
            options.absolute_tolerance,
            state,
            info,
            only_active=True,
            coalesced=coalesced,
        )
        if response is not None:
            factor.solve(options.young, state, info, options.cooperative_solve, only_failed=True)
            kernel_full_residual(
                options.young,
                options.tolerance,
                options.absolute_tolerance,
                state,
                info,
                only_active=True,
                coalesced=coalesced,
            )
        if group:
            kernel_group_lift(i_group, target, state, atlas.state, group.operator.transform)
        else:
            kernel_fallback_add(target, state, atlas.state)
        kernel_group_diagnostics(i_group, target, state, atlas.state)

    def recover_displacement(target):
        kernel_balance(zero_omega, target, mother.info, options.cooperative_balance)
        kernel_direct_init(target)
        kernel_atlas_begin(options.young, zero_omega, target, atlas.state, atlas.info, atlas.regions.state)
        for i_group in range(n_groups):
            solve_group(i_group, target)

    def add_spin(output):
        kernel_clear_displacement(part)
        kernel_spin_displacement(options.young, omega, atlas.info, part)
        mother.evaluate_stress(options, part)
        kernel_add_linear_fields(part, output)

    def replay(name):
        output = outputs[name]
        if name in ("A_separate_sources", "per_contact_common_q"):
            if name == "A_separate_sources":
                kernel_clear_linear_fields(output)
            else:
                kernel_clear_displacement(output)
            for i_source in range(4):
                kernel_load_source(i_source, raw_bank, grouped_bank, part, atlas.regions.state.force)
                recover_displacement(part)
                if name == "A_separate_sources":
                    mother.evaluate_stress(options, part)
                    kernel_add_linear_fields(part, output)
                else:
                    kernel_add_displacement(part, output)
            if name == "A_separate_sources":
                add_spin(output)
                kernel_tensor_von_mises(output)
            else:
                kernel_clear_displacement(part)
                kernel_spin_displacement(options.young, omega, atlas.info, part)
                kernel_add_displacement(part, output)
                mother.evaluate_stress(options, output)
        else:
            kernel_aggregate_sources(raw_bank, grouped_bank, output, atlas.regions.state.force)
            if name == "C_merged_displacement":
                atlas.recover(omega, output, options)
            else:
                kernel_clear_linear_fields(output)
                kernel_balance(zero_omega, output, mother.info, options.cooperative_balance)
                kernel_direct_init(output)
                kernel_atlas_begin(options.young, zero_omega, output, atlas.state, atlas.info, atlas.regions.state)
                for i_group in range(n_groups):
                    kernel_clear_displacement(part)
                    solve_group(i_group, part)
                    mother.evaluate_stress(options, part)
                    kernel_add_linear_fields(part, output)
                add_spin(output)
                kernel_tensor_von_mises(output)
        kernel_timing_retain_valid(output, retained[name])

    active = np.any(grouped_arrays != 0, axis=(1, 4))
    aggregate_active = np.any(grouped_arrays.sum(axis=0) != 0, axis=(0, 3))
    result.update(
        dimensions={
            "source_count": 4,
            "envs": args.envs,
            "mother_blocks": n_blocks,
            "boundary_blocks": n_boundary,
            "groups_including_fallback": n_groups,
            "sampled_elements": mother.n_output_elements,
            "samples_per_element": 4,
        },
        scheduling={
            "source_active_groups_per_environment": active.astype(int).tolist(),
            "grouped_active_groups_per_environment": aggregate_active.astype(int).tolist(),
            "source_fallback_active_environments": active[:, 0].sum(axis=1).tolist(),
            "grouped_fallback_active_environments": int(aggregate_active[0].sum()),
            "source_group_calls_per_iteration": {
                "A": 4 * n_groups,
                "B": n_groups,
                "C": n_groups,
                "per_contact_common_q": 4 * n_groups,
            },
            "full_stress_scans_per_iteration": {"A": 5, "B": n_groups + 1, "C": 1, "per_contact_common_q": 1},
        },
        memory=storage_inventory(
            mother,
            atlas,
            {"part": part, **outputs},
            {"raw_bank": raw_bank, "grouped_bank": grouped_bank, "omega": omega, "zero_omega": zero_omega, **retained},
        ),
    )
    names = tuple(outputs)
    result["paths"] = {name: {"repetitions": []} for name in names}
    write_json(args.output / "result.json", result)
    for name in names:
        retained[name].fill(True)
        for _ in range(args.warmup):
            replay(name)
        qd.sync()
        validation = compare(fields(outputs[name]), expected)
        validation["native_valid"] = qd_to_numpy(retained[name]).tolist()
        validation["passed"] = bool(validation["passed"] and all(validation["native_valid"]))
        result["paths"][name]["warmup"] = validation
        write_json(args.output / "result.json", result)
        assert validation["passed"], (name, validation)
    for repetition in range(args.repetitions):
        order = names[repetition % len(names) :] + names[: repetition % len(names)]
        for name in order:
            retained[name].fill(True)
            qd.sync()
            started = time.perf_counter()
            for _ in range(args.iterations):
                replay(name)
            qd.sync()
            elapsed = time.perf_counter() - started
            actual = fields(outputs[name])
            validation = compare(actual, expected)
            valid = qd_to_numpy(retained[name]).tolist()
            result["paths"][name]["repetitions"].append(
                {
                    "index": repetition,
                    "order": list(order),
                    "seconds": elapsed,
                    "seconds_per_iteration": elapsed / args.iterations,
                    "native_valid_all_iterations": valid,
                    "native_numerical_fallbacks_last_iteration": qd_to_numpy(outputs[name].fallbacks).tolist(),
                    "validation": validation,
                    "passed": bool(validation["passed"] and all(valid)),
                }
            )
            write_json(args.output / "result.json", result)
    for name in names:
        entries = result["paths"][name]["repetitions"]
        times = [entry["seconds_per_iteration"] for entry in entries]
        result["paths"][name].update(
            median_seconds_per_iteration=median(times),
            minimum_seconds_per_iteration=min(times),
            maximum_seconds_per_iteration=max(times),
            environment_fields_per_second=args.envs / median(times),
            passed=all(entry["passed"] for entry in entries),
        )
        np.savez_compressed(args.output / f"{name}.npz", **fields(outputs[name]))
        if args.signed_cancellation:
            result["paths"][name]["signed_cancellation_vs_reference"] = signed_field_diagnostics(
                fields(outputs[name]), signed_reference, reference_source_norms, combined["sample_weights"]
            )
    result["incremental_controls"] = {
        "per_contact_common_q_over_C_grouping_latency_ratio": result["paths"]["per_contact_common_q"][
            "median_seconds_per_iteration"
        ]
        / result["paths"]["C_merged_displacement"]["median_seconds_per_iteration"],
        "A_over_B_latency_ratio": result["paths"][names[0]]["median_seconds_per_iteration"]
        / result["paths"][names[1]]["median_seconds_per_iteration"],
        "B_over_C_merged_recovery_latency_ratio": result["paths"][names[1]]["median_seconds_per_iteration"]
        / result["paths"][names[2]]["median_seconds_per_iteration"],
        "grouping_control": "per_contact_common_q and C both recover the complete stress once: four independent source solves/merges versus on-device feature RHS grouping and grouped solves.",
        "merged_recovery_control": "B and C use the same grouped feature solve/gate/lift workload; B scans each group and spin separately and adds tensors, while C adds q and scans once.",
        "scope": "Observed four-path component differences at identical saved loads/samples; A/B also changes the declared number of stress scans, so its ratio is not an isolated aggregation-kernel speedup. No rigid or pressure/selection timing.",
    }
    result.update(complete=True, passed=all(value["passed"] for value in result["paths"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, choices=(4, 256), default=4)
    parser.add_argument("--solver", choices=("factor", "response"), default="factor")
    parser.add_argument("--criterion", choices=("von_mises",), default="von_mises")
    parser.add_argument("--label", default="same_feature_aggregation")
    parser.add_argument(
        "--signed-cancellation",
        action="store_true",
        help="Offline signed raw-load algebra control; not physical contact qualification.",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if min(args.iterations, args.repetitions, args.warmup) < 1:
        parser.error("Iterations, repetitions and warmup must all be positive.")
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid"
    assert os.environ.get("GENESIS_IMAGE_VER") == "1_26"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "identity": runtime_identity(),
        "worker_source": file_id(Path(__file__)),
        "complete": False,
        "passed": False,
        "paths": {},
        "criterion": args.criterion,
        "solver": args.solver,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "tolerance": 1e-8,
        "absolute_tolerance": 1e-11,
        "equivalence_tolerance": 1e-8,
        "scope": "Four-source fixed-load native component control; not a full-module or full-scene headline.",
        "multi_rhs_layout": "Each contact is a separate native solve call with B independent RHS environments; four sources are never folded into one aggregated RHS in A or per_contact_common_q.",
        "aggregation_scope": "B/C sum all four saved weighted native scatter loads on device inside every timed iteration; pressure, feature selection and topology/factor/response setup are excluded.",
        "empty_work_policy": "All four contact slots and every prepared group execute; only native active-environment masks skip factor/product work. No host empty-source skipping.",
        "criterion_implementation": "A/B use a research QD kernel with the same six-component symmetric tensor VM formula in native solve/shell scans; current engine has no standalone tensor-input criterion function. Never sum per-source VM/peaks.",
        "spin_policy": "Six native mother spin response columns are prepared once; physical replay adds the saved spin once per iteration, while signed-cancellation replay sets omega=0 in every path.",
        "timing_protocol": "Warm every path, rotate repetition order, sync before/after timed batches; all readback, equivalence checks and exports occur afterwards.",
        "signed_cancellation": {
            "enabled": args.signed_cancellation,
            "physical_contact_qualification": False,
            "scope": "Optional algebra diagnostic on saved native weighted raw loads only; original pressure/contact files and their physical baseline validation are unchanged.",
            "field_rule": "Use [1,-1/1.13,0,0] times the four saved source candidate/reference linear fields and zero spin; derive VM/peak only after total tensor summation. Never combine source peaks.",
            "acceptance": "Only original native solve gates and four-path equivalence determine passed. Near-cancellation observation and unchanged spatial budget flags are reported diagnostics.",
            "norm_metric": "Stress tensor uses native sample area/volume-weighted Frobenius norm with shear multiplicity two; q/N/M/Q use their complete sampled Euclidean norms. Source denominator is sum(abs(coefficient) * saved source reference norm).",
        },
    }
    write_json(args.output / "result.json", result)
    try:
        measure(args, result)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(args.output / "result.json", result)
        raise
    write_json(args.output / "result.json", result)
    assert result["passed"], "Component equivalence/numerical acceptance failed; preserve all raw results."


if __name__ == "__main__":
    main()
