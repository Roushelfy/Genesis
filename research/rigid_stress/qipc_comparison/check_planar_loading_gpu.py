"""Compare unchanged native shell operators and planar-loading outputs with pinned CPU oracles.

Scientific execution requires Slurm rtx-mid/image 1_26. Physical inputs are loaded
before native preparation; derived CPU operators and fields are used only in the
untimed audit after native construction.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class NativeMother:
    model: object
    inputs: dict
    bank: str
    asset: Path


def _api():
    import numpy as np
    import quadrants as qd
    from scipy import sparse
    from scipy.sparse.linalg import splu

    import genesis as gs
    from genesis.engine.solvers.rigid.stress.data import StressState
    from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
    from genesis.utils.array_class import V_VEC
    from genesis.utils.misc import qd_to_numpy

    from .artifacts import runtime_identity
    from .check_cpu12_gpu import Checks
    from .export_planar_loading_reference import Writer

    return SimpleNamespace(**locals())


def _load(np, path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unbatched_array(api, field):
    value = api.qd_to_numpy(field, transpose=True)
    if len(field.shape) > 1:
        value = api.np.moveaxis(value, 0, len(field.shape) - 1)
    return value


def build_native_mother(api, reference, output, bank, *, method="direct", output_mode="full"):
    """Prepare the original shell from physical inputs, preserving its native translation gauge."""
    inputs = _load(api.np, reference / f"inputs/{bank}/model.npz")
    asset = output / f"{bank}-native-shell.npz"
    api.np.savez_compressed(asset, vertices=inputs["vertices"], quads=inputs["quads"], directors=inputs["directors"])
    options = api.gs.options.RigidStressOptions.dpr(
        mesh=asset,
        backend="shell",
        method=method,
        output_mode=output_mode,
        young=float(inputs["young"]),
        poisson=float(inputs["poisson"]),
        density=float(inputs["density"]),
        shell_thickness=float(inputs["thickness"]),
        shell_drilling_factor=float(inputs["drilling_factor"]),
        shell_gauge="translation",
        tolerance=1e-8,
        absolute_tolerance=1e-11,
        inverse_precision="64",
        inverse_max_bytes=64 * 1024 * 1024,
        factor_scaling=True,
        history_size=0,
    )
    return NativeMother(api.ShellModel(options), inputs, bank, asset)


def _tensor(api, components):
    output = api.np.zeros((*components.shape[:-2], 3, 3, components.shape[-1]))
    for index, (first, second) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        output[..., first, second, :] = components[..., index, :]
        output[..., second, first, :] = components[..., index, :]
    return output


def native_outputs(api, model, state):
    """Read native side-interleaved body tensors into the oracle's side-major local/full layout."""
    np = api.np
    count, elements = state.active.shape[0], model.n_shell_elements
    axes = unbatched_array(api, model.shell.axes)
    first, second = axes[..., 0], axes[..., 1]
    components = api.qd_to_numpy(state.stress_tensor, transpose=True)
    components = components.reshape((count, elements, 2, 4, 6)).transpose(2, 1, 3, 4, 0)
    body = _tensor(api, components)
    local = np.stack(
        (
            np.einsum("egi,segijb,egj->segb", first, body, first),
            np.einsum("egi,segijb,egj->segb", second, body, second),
            np.einsum("egi,segijb,egj->segb", first, body, second),
        ),
        axis=3,
    )
    output = {
        key: api.qd_to_numpy(field, transpose=True).reshape((count, -1)).T
        for key, field in (("force", state.force), ("rhs", state.rhs), ("q", state.displacement))
    }
    output.update(
        stress_local=local,
        stress_full9=body.reshape((2, elements, 4, 9, count)),
        peak=api.qd_to_numpy(state.peak, transpose=True),
        valid=api.qd_to_numpy(state.valid, transpose=True),
    )
    for name, field in (("N", state.membrane_resultant), ("M", state.bending_resultant)):
        components = api.qd_to_numpy(field, transpose=True).transpose(1, 2, 3, 0)
        resultant = _tensor(api, components)
        output[name] = np.stack(
            (
                np.einsum("egi,egijb,egj->egb", first, resultant, first),
                np.einsum("egi,egijb,egj->egb", second, resultant, second),
                np.einsum("egi,egijb,egj->egb", first, resultant, second),
            ),
            axis=2,
        )
    transverse = api.qd_to_numpy(state.transverse_resultant, transpose=True).transpose(1, 2, 3, 0)
    output["Q"] = np.einsum("egij,egib->egjb", axes[..., :2], transverse)
    return output


def check_operators(api, writer, checks, reference, native):
    model, inputs, np = native.model, native.inputs, api.np
    prefix = f"oracles/{native.bank}/operators"
    oracle_k = api.sparse.load_npz(reference / f"{prefix}/K.npz")
    oracle_s = api.sparse.load_npz(reference / f"{prefix}/S.npz")
    oracle = _load(np, reference / f"{prefix}/numerics.npz")
    actual_k = api.sparse.bsr_matrix(
        (
            unbatched_array(api, model.info.stiffness) * model.options.young,
            unbatched_array(api, model.info.columns),
            unbatched_array(api, model.info.row_start),
        ),
        shape=oracle_k.shape,
    ).tocsr()
    checks.close("operators/original_K", actual_k, oracle_k)
    checks.exact("operators/elastic_DOF", model.info.vertices.shape[0] * 3, len(inputs["vertices"]) * 6)
    checks.exact("operators/Q4_connectivity", unbatched_array(api, model.shell.quads), inputs["quads"])
    checks.close("operators/authored_directors", unbatched_array(api, model.shell.directors), inputs["directors"])
    for name, field in (
        ("membrane", model.shell.membrane),
        ("bending", model.shell.bending),
        ("shear", model.shell.shear),
    ):
        checks.close(f"operators/{name}", unbatched_array(api, field).transpose(0, 1, 3, 2), oracle[f"B_{name}"])
    checks.close("operators/Gauss_axes", unbatched_array(api, model.shell.axes), oracle["axes"])
    checks.close("operators/Gauss_area_weights", unbatched_array(api, model.shell.weights), oracle["gauss_weights"])
    checks.close(
        "operators/mass_modes", unbatched_array(api, model.info.mass_modes).reshape((-1, 6)), oracle["mass_modes"]
    )
    checks.close(
        "operators/rigid_modes", unbatched_array(api, model.info.modes).reshape((-1, 6)), oracle["rigid_modes"]
    )
    checks.close("operators/mass_gram", unbatched_array(api, model.info.gram), oracle["mass_gram"])
    pins = unbatched_array(api, model.info.pins)
    free = unbatched_array(api, model.info.is_free).reshape(-1).astype(bool)
    checks.condition("operators/native_translation_gauge_rank", np.linalg.matrix_rank(oracle["rigid_modes"][pins]) == 6)
    writer.json(
        "operators/gauge.json",
        {
            "native_pins": pins.tolist(),
            "CPU_pins": oracle["gauge_dofs"].tolist(),
            "identical": bool(np.array_equal(np.sort(pins), np.sort(oracle["gauge_dofs"]))),
            "comparison": "Original native gauge retained; independent CPU K is solved with these same native integer pins for coordinate comparison.",
        },
    )
    state = model.create_state(4, "full")
    omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(4,))
    omega.fill(0.0)
    qd, gs, state_type = api.qd, api.gs, api.StressState

    @qd.kernel
    def kernel_probe(thickness: float, current: state_type):
        for i_n, i_b in qd.ndrange(current.force.shape[0], current.force.shape[1]):
            value = qd.Vector.zero(gs.qd_float, 3)
            if i_b < 3:
                for axis in qd.static(range(3)):
                    value[axis] = 1e-3 * qd.sin(float((3 * i_n + axis + 1) * (i_b + 1)))
                if i_n % 2 == 1:
                    value *= thickness
            current.force[i_n, i_b] = value

    kernel_probe(float(model.options.shell_thickness), state)
    model.recover(omega, state)
    actual = native_outputs(api, model, state)
    balanced = actual["force"] - oracle["mass_modes"] @ np.linalg.solve(
        oracle["mass_gram"], oracle["rigid_modes"].T @ actual["force"]
    )
    expected_q = np.zeros_like(actual["q"])
    expected_q[free] = api.splu(oracle_k[free][:, free].tocsc()).solve(np.asfortranarray(balanced[free]))
    residual = (oracle_k @ actual["q"] - actual["rhs"])[free]
    residual_norm, rhs_norm = np.linalg.norm(residual, axis=0), np.linalg.norm(actual["rhs"][free], axis=0)
    checks.close("probe/common_relief", actual["rhs"], balanced)
    checks.close("probe/original_K_displacement", actual["q"], expected_q, rtol=1e-8, atol=1e-13)
    expected_local = (oracle_s @ expected_q).reshape(actual["stress_local"].shape)
    checks.close("probe/original_S_complete_stress", actual["stress_local"], expected_local, rtol=1e-8, atol=1e-10)
    checks.condition(
        "probe/original_unscaled_free_residual",
        np.all(residual_norm <= np.maximum(1e-11, 1e-8 * rhs_norm)),
        absolute_norms=residual_norm.tolist(),
        relative_norms=(residual_norm / np.maximum(rhs_norm, 1e-30)).tolist(),
    )
    checks.condition("probe/native_admission", actual["valid"].all())
    writer.arrays("operators/probe.npz", **actual, expected_q=expected_q, expected_stress_local=expected_local)
    state.force.fill(0.0)
    model.recover(omega, state)
    released = native_outputs(api, model, state)
    for key in ("force", "rhs", "q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
        checks.condition(f"probe/release/{key}", np.count_nonzero(released[key]) == 0)
    kernel_probe(float(model.options.shell_thickness), state)
    model.recover(omega, state)
    reloaded = native_outputs(api, model, state)
    for key in ("force", "rhs", "q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
        checks.close(f"probe/reload/{key}", reloaded[key], actual[key], rtol=1e-10, atol=1e-13)
    writer.json(
        "operators/inventory.json",
        {
            "raw_scalar_DOF": actual_k.shape[0],
            "free_scalar_DOF": int(free.sum()),
            "native_K_block_entries": model.info.stiffness.shape[0],
            "native_factor_lower_blocks": model.factor.info.lower.shape[0],
            "selected_method": model.selected_method,
            "build_timings_including_JIT": asdict(model.build_timings),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", choices=("egg75", "torus8"), default="egg75")
    parser.add_argument("--method", choices=("direct", "inverse", "auto"), default="direct")
    parser.add_argument("--stage", choices=("operators",), default="operators")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Choose a fresh project data output under /mnt/data/zhaofeng.")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "genesis.planar-loading-reference.v1" or not manifest["passed"]:
        raise ValueError("Use a successful pinned planar-loading reference export.")
    for relative, record in manifest["files"].items():
        if _hash(args.reference / relative) != record["sha256"]:
            raise ValueError(f"Reference payload changed: {relative}.")
    started = time.perf_counter()
    api = _api()
    api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    writer, checks = api.Writer(args.output, api), api.Checks(api)
    writer.json(
        "start.json",
        {
            "identity": api.runtime_identity(),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "reference_manifest_sha256": _hash(manifest_path),
            "native_package": api.gs.__file__,
        },
    )
    native = build_native_mother(api, args.reference, args.output, args.bank, method=args.method)
    check_operators(api, writer, checks, args.reference, native)
    checks.flush(writer)
    passed = all(row["passed"] for row in checks.rows)
    writer.json(
        "result.json",
        {
            "complete": True,
            "passed": passed,
            "stage": args.stage,
            "bank": args.bank,
            "checks": len(checks.rows),
            "failed_checks": [row["name"] for row in checks.rows if not row["passed"]],
            "seconds_including_native_preparation": time.perf_counter() - started,
            "scope": "Unchanged Q4 elasticity, common relief, original native gauge, stress recovery and zero/reload controls; planar current loading is a separate stage.",
        },
    )
    if not passed:
        raise ArithmeticError("Native planar-loading mother/operator controls failed.")


if __name__ == "__main__":
    main()
