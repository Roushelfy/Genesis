"""Locate one serial planar-loading call using durable kernel-boundary receipts.

Replay saved physical inputs only. The kernel order matches PlanarLoading's
serial path; no CPU oracle array enters native preparation or loading. This
untimed scientific diagnostic requires Slurm rtx-mid and image 1_26.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _api():
    from .check_planar_footprint import _native_api

    base = _native_api()
    from genesis.engine.solvers.rigid.stress.contact import kernel_pressure, kernel_scatter
    from genesis.engine.solvers.rigid.stress.planar_loading import (
        kernel_planar_fit,
        kernel_planar_moments,
        kernel_planar_query,
        kernel_planar_scatter,
        kernel_planar_validate,
    )

    return SimpleNamespace(
        **(vars(base) | {name: value for name, value in locals().items() if name not in ("base", "_native_api")})
    )


def _stage(api, output, events, name, function, *args):
    api.qd.sync()
    start = time.time()
    events.append({"name": name, "started_unix": start, "finished": False})
    _json(output / "stages.json", events)
    function(*args)
    api.qd.sync()
    events[-1].update(finished=True, finished_unix=time.time(), diagnostic_wall_seconds=time.time() - start)
    _json(output / "stages.json", events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-inputs", type=Path, required=True, help="A CPU worker directory containing inputs/book only."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quadrature", type=int, choices=(16, 48), default=48)
    parser.add_argument("--refinement", type=int, choices=(16, 32), default=16)
    parser.add_argument("--capacity", type=int, choices=(1, 4096), default=1)
    parser.add_argument("--method", choices=("direct", "inverse"), default="inverse")
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific diagnostics require Slurm rtx-mid and image 1_26.")
    output, inputs = args.output.resolve(), args.physical_inputs.resolve()
    if not output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh explicit data-side output.")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "complete": False,
        "passed": False,
        "schema": "genesis.planar-serial-diagnostic.v1",
        "worker": _identity(Path(__file__).resolve()),
        "physical_inputs": [_identity(inputs / "inputs/book" / name) for name in ("model.npz", "cases.json")],
        "scope": "One serial call, no rigid advance, no oracle input and no primary timing claim.",
    }
    _json(output / "result.json", result)
    api = _api()
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    events = []
    try:
        _json(output / "stages.json", [{"name": "build_original_mother", "finished": False}])
        native = api.build_native_mother(api, inputs, output, "book", method=args.method, quadrature=args.quadrature)
        model = native.model
        contacts, cases, _valid, _sides = api.prepare_physical_contacts(
            api, inputs, "book", model.options.young, model.options.poisson
        )
        surface = api.StressSurface(args.quadrature, model.load_info, angular=True)
        loading = api.PlanarLoading(
            surface.info, model.load_info, contacts, args.refinement, tasks_per_env=args.capacity
        )
        state = model.create_state(len(cases), "full")
        omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(len(cases),))
        omega.fill(0)
        epsilon = float(api.np.finfo(api.gs.np_float).eps)
        _stage(api, output, events, "anchor", api.kernel_anchor, epsilon, contacts, surface.info)
        _stage(api, output, events, "query_serial", api.kernel_planar_query, contacts, loading.info, loading.state)
        _stage(
            api,
            output,
            events,
            "moments_serial",
            api.kernel_planar_moments,
            contacts,
            loading.state,
            surface.info,
            True,
        )
        _stage(api, output, events, "fit", api.kernel_planar_fit, contacts, loading.state, surface.info, True)
        _stage(
            api,
            output,
            events,
            "validate",
            api.kernel_planar_validate,
            contacts,
            loading.state,
            model.load_info,
            surface.info,
            True,
        )
        _stage(
            api,
            output,
            events,
            "material",
            api.kernel_planar_material,
            epsilon,
            contacts,
            loading.state.fallback,
            surface.info,
            loading.info,
            loading.material_state,
        )
        _stage(
            api,
            output,
            events,
            "original_pressure_serial",
            api.kernel_pressure,
            loading.fallback_contacts,
            loading.material_surface,
            False,
        )
        _stage(
            api,
            output,
            events,
            "original_scatter_serial",
            api.kernel_scatter,
            loading.fallback_contacts,
            state,
            model.load_info,
            loading.material_surface,
            False,
            model.options.cached_face_bounds,
        )
        _stage(
            api,
            output,
            events,
            "proxy_scatter",
            api.kernel_planar_scatter,
            contacts,
            loading.state,
            model.load_info,
            state,
        )
        _stage(api, output, events, "original_recover", model.recover, omega, state)
        actual = api.native_outputs(api, model, state)
        api.np.savez_compressed(output / "outputs.npz", **actual)
        result.update(complete=True, passed=bool(actual["valid"].all()), stages=events)
    except Exception:
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        _json(output / "result.json", result)
    if not result["passed"]:
        raise ArithmeticError("The original serial replay did not satisfy native admission.")


if __name__ == "__main__":
    main()
