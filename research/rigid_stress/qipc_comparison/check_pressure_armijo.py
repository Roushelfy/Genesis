"""Compare the two saved serial-pressure failures, saved warp loads and corrected serial loads.

The original immutable readbacks supply old behavior; the corrected serial
route replays the same physical sources and original Q4 mother. Scientific
execution requires Slurm rtx-mid and image 1_26.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-run", type=Path, required=True)
    parser.add_argument("--case-names", nargs="+", default=("tip-p0-s1-normal", "tip-p1-s1-normal"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific execution requires Slurm rtx-mid and image 1_26.")
    output, saved = args.output.resolve(), args.saved_run.resolve()
    if not output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output.")
    output.mkdir(parents=True, exist_ok=False)
    from .check_planar_footprint import _check_book_run, _finite_json, _identity, _load, _native_api

    api = _native_api()
    np = api.np
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    writer, checks = api.Writer(output, api), api.Checks(api)
    cpu = saved / "cpu"
    cases = json.loads((cpu / "inputs/book/cases.json").read_text())
    indices = [next(i for i, case in enumerate(cases) if case["name"] == name) for name in args.case_names]
    selected = np.zeros(len(cases), dtype=bool)
    selected[indices] = True
    result = {
        "schema": "genesis.pressure-armijo-check.v1",
        "complete": False,
        "passed": False,
        "worker": _identity(Path(__file__).resolve()),
        "saved_run": str(saved),
        "case_names": args.case_names,
        "environment_indices": indices,
        "source_wrench_limit": 1e-8,
        "original_K_relative": 1e-8,
        "original_K_absolute": 1e-11,
        "same_space_field_limit": 0.001,
        "scope": "Saved oldserial/warp evidence and one newserial solve, same two stored physical sources, original mother, quadrature48, no rigid advancement.",
    }
    writer.json("result.json", result)
    try:
        old_serial_path, old_warp_path = saved / "book/serial/cap1/initial", saved / "book/faces/cap1/initial"
        inputs = [
            cpu / "inputs/book/model.npz",
            cpu / "inputs/book/cases.json",
            cpu / "oracles/q48.npz",
            cpu / "oracles/K.npz",
            cpu / "oracles/numerics.npz",
            old_serial_path / "outputs.npz",
            old_serial_path / "routing.npz",
            old_warp_path / "outputs.npz",
            old_warp_path / "routing.npz",
        ]
        identities = [_identity(path) for path in inputs]
        result["inputs"] = identities
        old_serial, old_warp = _load(np, old_serial_path / "outputs.npz"), _load(np, old_warp_path / "outputs.npz")
        serial_routing, warp_routing = (
            _load(np, old_serial_path / "routing.npz"),
            _load(np, old_warp_path / "routing.npz"),
        )
        checks.condition(
            "old_serial/stored_same_sources_rejected",
            (serial_routing["contact_status"][0, indices] == 3).all(),
            status=serial_routing["contact_status"][0, indices].tolist(),
        )
        checks.condition(
            "old_serial/rejected_sources_scatter_no_load", not np.count_nonzero(old_serial["force"][:, indices])
        )
        checks.condition(
            "old_warp/same_sources_admitted",
            not warp_routing["contact_status"][0, indices].any() and old_warp["valid"][indices].all(),
        )
        native = api.build_native_mother(api, cpu, output, "book", method="inverse", quadrature=48)
        model = native.model
        contacts, _cases, valid, _sides = api.prepare_physical_contacts(
            api, cpu, "book", model.options.young, model.options.poisson
        )
        current = valid & selected[None, :]
        contacts.valid.from_numpy(current)
        physical = api._snapshot_contacts(api, contacts)
        surface = api.StressSurface(48, model.load_info, angular=True)
        loading = api.PlanarLoading(surface.info, model.load_info, contacts, 16, tasks_per_env=1)
        state = model.create_state(len(cases), "full")
        workspace = api.create_scatter_workspace(len(cases), 0, False)
        omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(len(cases),))
        omega.fill(0)
        options = model.options.model_copy(update={"cooperative_pressure": False, "cooperative_scatter": False})
        api.kernel_anchor(float(np.finfo(api.gs.np_float).eps), contacts, surface.info)
        started = time.perf_counter()
        loading.scatter(state, model.load_info, surface.info, options, workspace, False)
        model.recover(omega, state)
        api.qd.sync()
        result["intrusive_single_call_seconds"] = time.perf_counter() - started
        actual = api.native_outputs(api, model, state)
        status = api.unbatched_array(api, contacts.status)
        force_error, moment_error = (
            api.unbatched_array(api, contacts.force_error),
            api.unbatched_array(api, contacts.moment_error),
        )
        magnitude = np.linalg.norm(physical["force"], axis=-1)
        checks.condition("new_serial/contact_status", not status[current].any(), status=status[current].tolist())
        checks.condition(
            "new_serial/original_source_wrench",
            (force_error[current] <= 1e-8 * magnitude[current]).all()
            and (moment_error[current] <= 1e-8 * magnitude[current] * physical["radius"][current]).all(),
        )
        checks.condition(
            "new_serial/fallback_material_success",
            not api.unbatched_array(api, loading.material_state.status)[current].any(),
        )
        expected = _load(np, cpu / "oracles/q48.npz")
        numerics = _load(np, cpu / "oracles/numerics.npz")
        matrix = api.sparse.load_npz(cpu / "oracles/K.npz")
        keys = ("raw", "balanced", "q", "stress_local", "stress_full9", "peak", "N", "M", "Q")
        expected = {key: expected[key].copy() for key in keys}
        for value in expected.values():
            value[..., ~selected] = 0
        free = api.unbatched_array(api, model.info.is_free).reshape(-1).astype(bool)
        result["original_Q4_reference"] = _check_book_run(
            api, writer, checks, "new_serial", actual, expected, expected, numerics, free, matrix, selected, 48
        )
        checks.close(
            "new_serial_vs_stored_warp/raw",
            actual["force"][:, indices],
            old_warp["force"][:, indices],
            rtol=1e-8,
            atol=1e-13,
        )
        warp_reference = {key: old_warp[key] for key in ("stress_local", "stress_full9", "peak", "N", "M", "Q")}
        errors = api._errors(np, actual, warp_reference, numerics["gauss_weights"])
        maxima = {key: float(value[selected].max(initial=0)) for key, value in errors.items()}
        checks.condition(
            "new_serial_vs_stored_warp/field",
            all(value <= 0.001 for value in maxima.values()),
            maxima=maxima,
            tolerance=0.001,
        )
        nodal = actual["force"].reshape((-1, 6, len(cases)))
        for index in indices:
            source = cases[index]["contacts"][0]
            resultant = nodal[:, :3, index].sum(axis=0)
            moment = np.cross(native.inputs["vertices"] - source["point"], nodal[:, :3, index]).sum(axis=0) + nodal[
                :, 3:, index
            ].sum(axis=0)
            force_norm = np.linalg.norm(source["force"])
            checks.condition(
                f"new_serial/raw_original_point_wrench/{cases[index]['name']}",
                np.linalg.norm(resultant - source["force"]) <= 1e-8 * force_norm
                and np.linalg.norm(moment) <= 1e-8 * force_norm * source["radius"],
            )
        for name in ("position", "force", "normal", "radius", "friction", "valid"):
            checks.exact(
                f"new_serial/physical_inputs_unchanged/{name}",
                api.unbatched_array(api, vars(contacts)[name]),
                physical[name],
            )
        checks.condition("provenance/saved_inputs_unchanged", [_identity(path) for path in inputs] == identities)
        writer.arrays(
            "pressure.npz",
            status=status,
            coefficient=api.unbatched_array(api, contacts.coefficient),
            weight_sum=api.unbatched_array(api, contacts.weight_sum),
            evaluations=api.unbatched_array(api, contacts.evaluations),
            candidate_count=api.unbatched_array(api, contacts.candidate_count),
            force_error=force_error,
            moment_error=moment_error,
            valid=current,
        )
        result.update(
            complete=True,
            passed=all(row["passed"] for row in checks.rows),
            checks=len(checks.rows),
            old_serial_status=serial_routing["contact_status"][0, indices].tolist(),
            old_warp_status=warp_routing["contact_status"][0, indices].tolist(),
            new_serial_status=status[0, indices].tolist(),
            new_serial_evaluations=api.unbatched_array(api, contacts.evaluations)[0, indices].tolist(),
        )
    finally:
        if sys.exc_info()[0] is not None:
            (output / "failure.txt").write_text(traceback.format_exc())
        safe, count = _finite_json(checks.rows)
        if count:
            result["passed"] = False
            result["nonfinite_diagnostic_count"] = count
        writer.json("checks.json", safe)
        safe_result, _count = _finite_json(result)
        writer.json("result.json", safe_result)
    if not result["passed"]:
        raise ArithmeticError("The pressure Armijo regression failed. Preserve every old and new negative.")


if __name__ == "__main__":
    main()
