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
    from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts, kernel_anchor
    from genesis.engine.solvers.rigid.stress.data import StressState
    from genesis.engine.solvers.rigid.stress.scatter import create_scatter_workspace
    from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
    from genesis.engine.solvers.rigid.stress.surface import StressSurface
    from genesis.utils.array_class import V_VEC, V
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


def build_native_mother(api, reference, output, bank, *, method="direct", output_mode="full", quadrature=24):
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
        quadrature=quadrature,
        pressure_profile="hertz",
        tolerance=1e-8,
        absolute_tolerance=1e-11,
        inverse_precision="64",
        inverse_max_bytes=64 * 1024 * 1024,
        factor_scaling=True,
        history_size=0,
    )
    return NativeMother(api.ShellModel(options), inputs, bank, asset)


def prepare_physical_contacts(api, reference, bank, young, poisson):
    """Load original physical sources and evaluate supplied-curvature Hertz sizes in Quadrants."""
    np = api.np
    rows = json.loads((reference / f"inputs/{bank}/cases.json").read_text())
    n_envs, n_contacts = len(rows), max(len(row["contacts"]) for row in rows)
    positions, forces, normals = (np.zeros((n_contacts, n_envs, 3)) for _ in range(3))
    curves = np.zeros((n_contacts, n_envs, 2))
    radius = np.zeros((n_contacts, n_envs))
    friction = np.ones((n_contacts, n_envs))
    valid = np.zeros((n_contacts, n_envs), dtype=bool)
    manual = np.zeros_like(valid)
    sides = np.zeros((n_contacts, n_envs), dtype=np.int32)
    for i_b, row in enumerate(rows):
        for i_c, source in enumerate(row["contacts"]):
            positions[i_c, i_b], forces[i_c, i_b], normals[i_c, i_b] = (
                source["point"],
                source["force"],
                -np.array(source["normal"]),
            )
            curves[i_c, i_b] = source["curvatures"]
            radius[i_c, i_b] = source["radius"] if source["radius"] is not None else 0.0
            friction[i_c, i_b] = source["friction"] if source["friction"] is not None else 1.0
            valid[i_c, i_b], manual[i_c, i_b], sides[i_c, i_b] = True, source["radius"] is not None, source["side"]
    contacts = api.create_contacts(n_contacts, n_envs, 1e-4, hertz_profile=True)
    contacts.position.from_numpy(positions.astype(api.gs.np_float))
    contacts.force.from_numpy(forces.astype(api.gs.np_float))
    contacts.normal.from_numpy(normals.astype(api.gs.np_float))
    contacts.friction.from_numpy(friction.astype(api.gs.np_float))
    contacts.radius.from_numpy(radius.astype(api.gs.np_float))
    contacts.valid.from_numpy(valid)
    curvature = api.V_VEC(2, dtype=api.gs.qd_float, shape=(n_contacts, n_envs))
    curvature.from_numpy(curves.astype(api.gs.np_float))
    given = api.V(dtype=api.gs.qd_bool, shape=(n_contacts, n_envs))
    given.from_numpy(manual)
    qd, contact_type = api.qd, api.StressContactState

    @qd.kernel
    def kernel_radius(young: float, poisson: float, current: contact_type, curvatures: qd.Tensor, supplied: qd.Tensor):
        for i_c, i_b in qd.ndrange(current.valid.shape[0], current.valid.shape[1]):
            if current.valid[i_c, i_b] and not supplied[i_c, i_b]:
                load = current.force[i_c, i_b].dot(current.normal[i_c, i_b].normalized())
                average = 0.5 * (curvatures[i_c, i_b][0] + curvatures[i_c, i_b][1])
                modulus = young / (1.0 - poisson * poisson)
                current.radius[i_c, i_b] = (3.0 * load / (4.0 * modulus * average)) ** (1.0 / 3.0)
                current.radius_status[i_c, i_b] = 1

    kernel_radius(float(young), float(poisson), contacts, curvature, given)
    api.qd.sync()
    return contacts, rows, valid, sides


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


def check_loading(api, writer, checks, reference, native, refinement, quadrature, capacity, budgets, work_warps):
    """Check primary single-side CPU equivalence, whole-object error and current-load lifecycle."""
    from genesis.engine.solvers.rigid.stress.planar_loading import PlanarLoading

    np, model = api.np, native.model
    surface = api.StressSurface(quadrature, model.load_info)
    contacts, cases, original_valid, sides = prepare_physical_contacts(
        api, reference, native.bank, model.options.young, model.options.poisson
    )
    loading = PlanarLoading(
        surface.info, model.load_info, contacts, refinement, tasks_per_env=capacity, work_warps=work_warps
    )
    loading.state.side_filter.from_numpy(sides.astype(api.gs.np_int))
    state = model.create_state(len(cases), "full")
    workspace = api.create_scatter_workspace(len(cases), model.options.scatter_tasks_per_env, False)
    omega = api.V_VEC(3, dtype=api.gs.qd_float, shape=(len(cases),))
    omega.fill(0.0)
    label = f"tri-k{refinement}-q{quadrature}"
    cpu_load = _load(np, reference / f"oracles/{native.bank}/{label}/loads.npz")
    cpu_output = _load(np, reference / f"oracles/{native.bank}/{label}/output.npz")
    original_output = _load(np, reference / f"oracles/{native.bank}/Q4-q48/output.npz")
    oracle = _load(np, reference / f"oracles/{native.bank}/operators/numerics.npz")
    oracle_k = api.sparse.load_npz(reference / f"oracles/{native.bank}/operators/K.npz")
    free = unbatched_array(api, model.info.is_free).reshape(-1).astype(bool)
    expected_q = np.zeros_like(cpu_output["q"])
    expected_q[free] = api.splu(oracle_k[free][:, free].tocsc()).solve(np.asfortranarray(cpu_load["balanced"][free]))
    api.kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
    loading.scatter(state, model.load_info, surface.info, model.options, workspace, False)
    model.recover(omega, state)
    actual = native_outputs(api, model, state)
    routing = {
        name: api.qd_to_numpy(field, transpose=True)
        for name, field in (
            ("eligible", loading.state.eligible),
            ("fallback", loading.state.fallback),
            ("reason", loading.state.reason),
            ("analytic_count", loading.state.analytic_count),
            ("sample_count", loading.state.sample_count),
            ("fit_evaluations", loading.state.fit_evaluations),
            ("fallback_count", loading.state.fallback_count),
            ("overflow", loading.state.overflow),
            ("task_count", loading.state.task_count),
        )
    }
    writer.arrays(f"{label}/output.npz", **actual, expected_q_in_native_gauge=expected_q)
    writer.arrays(f"{label}/routing.npz", **routing)
    checks.condition(f"{label}/native_admission", actual["valid"].all())
    checks.condition(
        f"{label}/CPU_primary_complete_proxy_port",
        not routing["fallback"].any(),
        fallback_contacts=int(routing["fallback"].sum()),
        scope="Primary parity uses the explicitly selected CPU side. General production defaults retain both sides; fallback controls are separate.",
    )
    checks.condition(
        f"{label}/capacity",
        not routing["overflow"].any(),
        capacity_per_environment=capacity,
        maximum_actual_tasks=int(routing["task_count"].max(initial=0)),
    )
    checks.close(
        f"{label}/original_anchors",
        api.qd_to_numpy(contacts.center, transpose=True)[original_valid.T],
        api.qd_to_numpy(contacts.position, transpose=True)[original_valid.T],
        rtol=1e-10,
        atol=1e-13,
    )
    checks.close(f"{label}/raw_generalized_load", actual["force"], cpu_load["raw"], rtol=1e-8, atol=1e-13)
    checks.close(f"{label}/common_relief", actual["rhs"], cpu_load["balanced"], rtol=1e-8, atol=1e-13)
    checks.close(f"{label}/original_K_displacement", actual["q"], expected_q, rtol=1e-8, atol=1e-13)
    for key in ("stress_local", "stress_full9", "peak", "N", "M", "Q"):
        checks.close(
            f"{label}/CPU_native/{key}",
            actual[key],
            cpu_output[key],
            rtol=1e-8,
            atol=1e-10 if key.startswith("stress") or key == "peak" else 1e-13,
        )
    residual = (oracle_k @ actual["q"] - actual["rhs"])[free]
    residual_norm, rhs_norm = np.linalg.norm(residual, axis=0), np.linalg.norm(actual["rhs"][free], axis=0)
    checks.condition(
        f"{label}/original_unscaled_residual",
        np.all(residual_norm <= np.maximum(1e-11, 1e-8 * rhs_norm)),
        absolute_norms=residual_norm.tolist(),
        relative_norms=(residual_norm / np.maximum(rhs_norm, 1e-30)).tolist(),
    )
    weights, multiplicity = oracle["gauss_weights"], np.array([1.0, 1.0, 2.0])
    errors = {}
    for key in ("stress_local", "N", "M", "Q"):
        expected, delta = original_output[key], actual[key] - original_output[key]
        if key == "stress_local":
            squared = np.einsum("segcb,segcb,eg,c->b", delta, delta, weights, multiplicity)
            reference_squared = np.einsum("segcb,segcb,eg,c->b", expected, expected, weights, multiplicity)
            key = "field"
        elif key in ("N", "M"):
            squared = np.einsum("egcb,egcb,eg,c->b", delta, delta, weights, multiplicity)
            reference_squared = np.einsum("egcb,egcb,eg,c->b", expected, expected, weights, multiplicity)
        else:
            squared = np.einsum("egcb,egcb,eg->b", delta, delta, weights)
            reference_squared = np.einsum("egcb,egcb,eg->b", expected, expected, weights)
        errors[key] = np.sqrt(squared) / np.maximum(np.sqrt(reference_squared), 1e-30)
    errors["peak"] = np.abs(actual["peak"] - original_output["peak"]) / np.maximum(original_output["peak"], 1e-30)
    loaded = original_valid.any(axis=0)
    mean_field = float(errors["field"][loaded].mean())
    maxima = {key: float(value[loaded].max(initial=0)) for key, value in errors.items()}
    original_budget = budgets["original_budgets"]
    original_passed = (
        mean_field <= original_budget["mean_weighted_tensor_field"]
        and maxima["field"] <= original_budget["worst_weighted_tensor_field"]
        and maxima["peak"] <= original_budget["worst_sampled_peak"]
        and all(maxima[key] <= original_budget[key] for key in ("N", "M", "Q"))
    )
    target_applicable = refinement in (8, 16)
    target = budgets["CPU_demonstrated_fidelity_targets"].get(f"k{refinement}") if target_applicable else None
    target_passed = (
        bool(
            target is not None
            and maxima["field"] <= target["worst_field"]
            and maxima["peak"] <= target["worst_peak"]
            and all(maxima[key] <= target[key] for key in ("N", "M", "Q"))
        )
        if target_applicable
        else None
    )
    target_status = ("passed" if target_passed else "failed") if target_applicable else "not_applicable"
    writer.json(
        f"{label}/spatial.json",
        {
            "original_budget_passed": bool(original_passed),
            "CPU_fidelity_target_passed": target_passed,
            "CPU_fidelity_target_status": target_status,
            "mean_loaded_field_error": mean_field,
            "maximum_loaded_errors": maxima,
            "rows": [
                {
                    "name": row["name"],
                    "category": row["category"],
                    "loaded": bool(loaded[i]),
                    **{key: float(value[i]) for key, value in errors.items()},
                }
                for i, row in enumerate(cases)
            ],
            "scope": "CPU primary single-side prescribed-source fixture, matching the original Q4 q48 same-elasticity reference.",
        },
    )
    checks.condition(
        f"{label}/original_spatial_budgets", original_passed, maxima=maxima, mean_loaded_field_error=mean_field
    )
    if target_applicable:
        checks.condition(f"{label}/CPU_fidelity_targets", target_passed, target=target, maxima=maxima)
    else:
        checks.rows.append(
            {
                "name": f"{label}/CPU_fidelity_targets",
                "passed": None,
                "status": "not_applicable",
                "gating": False,
                "reason": "Frozen CPU-demonstrated fidelity targets apply only to k8/k16; k4 retains the original spatial budgets.",
                "target": None,
                "maxima": maxima,
            }
        )
    qd, contact_type = api.qd, api.StressContactState

    @qd.kernel
    def kernel_reverse(current: contact_type, side_filter: qd.Tensor):
        for i_b in range(current.valid.shape[1]):
            for i_c in range(current.valid.shape[0] // 2):
                j_c = current.valid.shape[0] - i_c - 1
                position, force, normal = current.position[i_c, i_b], current.force[i_c, i_b], current.normal[i_c, i_b]
                radius, friction, valid = current.radius[i_c, i_b], current.friction[i_c, i_b], current.valid[i_c, i_b]
                radius_status, side = current.radius_status[i_c, i_b], side_filter[i_c, i_b]
                current.position[i_c, i_b], current.force[i_c, i_b], current.normal[i_c, i_b] = (
                    current.position[j_c, i_b],
                    current.force[j_c, i_b],
                    current.normal[j_c, i_b],
                )
                current.radius[i_c, i_b], current.friction[i_c, i_b], current.valid[i_c, i_b] = (
                    current.radius[j_c, i_b],
                    current.friction[j_c, i_b],
                    current.valid[j_c, i_b],
                )
                current.radius_status[i_c, i_b], side_filter[i_c, i_b] = (
                    current.radius_status[j_c, i_b],
                    side_filter[j_c, i_b],
                )
                current.position[j_c, i_b], current.force[j_c, i_b], current.normal[j_c, i_b] = position, force, normal
                current.radius[j_c, i_b], current.friction[j_c, i_b], current.valid[j_c, i_b] = radius, friction, valid
                current.radius_status[j_c, i_b], side_filter[j_c, i_b] = radius_status, side

    phases = ("slot_reverse", "release_all", "reload", "release_env0", "reload_env0")
    for phase in phases:
        if phase == "slot_reverse":
            kernel_reverse(contacts, loading.state.side_filter)
        elif phase == "release_all":
            contacts.valid.fill(False)
        elif phase == "reload":
            kernel_reverse(contacts, loading.state.side_filter)
            contacts.valid.from_numpy(original_valid)
        elif phase == "release_env0":
            current_valid = original_valid.copy()
            current_valid[:, 0] = False
            contacts.valid.from_numpy(current_valid)
        else:
            contacts.valid.from_numpy(original_valid)
        api.kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
        loading.scatter(state, model.load_info, surface.info, model.options, workspace, False)
        model.recover(omega, state)
        current = native_outputs(api, model, state)
        for key in ("force", "rhs", "q", "stress_local", "stress_full9", "peak", "N", "M", "Q"):
            expected = actual[key].copy()
            if phase == "release_all":
                expected.fill(0)
            elif phase == "release_env0":
                expected[..., 0] = 0
            checks.close(
                f"{label}/{phase}/{key}",
                current[key],
                expected,
                rtol=1e-9,
                atol=1e-10 if key.startswith("stress") or key == "peak" else 1e-13,
            )
        checks.condition(f"{label}/{phase}/native_admission", current["valid"].all())
        if phase == "release_all":
            checks.condition(
                f"{label}/{phase}/query_clear",
                not api.qd_to_numpy(loading.state.eligible, transpose=True).any()
                and not api.qd_to_numpy(loading.state.fallback, transpose=True).any(),
            )
    return {
        "method": label,
        "work_warps": work_warps,
        "original_spatial_budgets_passed": bool(original_passed),
        "CPU_fidelity_targets_passed": target_passed,
        "CPU_fidelity_targets_status": target_status,
        "mean_loaded_field_error": mean_field,
        "maximum_loaded_errors": maxima,
        "fallback_contacts": int(routing["fallback"].sum()),
        "maximum_tasks": int(routing["task_count"].max(initial=0)),
        "analytic_contact_sections": int(routing["analytic_count"].sum()),
        "samples": int(routing["sample_count"].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", choices=("egg75", "torus8"), default="egg75")
    parser.add_argument("--method", choices=("direct", "inverse", "auto"), default="direct")
    parser.add_argument("--stage", choices=("operators", "loads", "all"), default="operators")
    parser.add_argument("--quadrature", type=int, choices=(24, 48), default=24)
    parser.add_argument("--subdivisions", type=int, nargs="+", choices=(4, 8, 16), default=[8, 16])
    parser.add_argument("--tasks-per-env", type=int, default=2000)
    parser.add_argument("--work-warps", type=int, choices=(512, 2048, 4096), default=2048)
    parser.add_argument("--budgets", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Choose a fresh project data output under /mnt/data/zhaofeng.")
    if args.stage != "operators" and (args.budgets is None or args.tasks_per_env < 1):
        parser.error("Load qualification requires --budgets and a positive --tasks-per-env.")
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
    native = build_native_mother(
        api, args.reference, args.output, args.bank, method=args.method, quadrature=args.quadrature
    )
    if args.stage in ("operators", "all"):
        check_operators(api, writer, checks, args.reference, native)
    summaries = []
    if args.stage in ("loads", "all"):
        budgets = json.loads(args.budgets.read_text())
        if not budgets["frozen_before_native_tuning"]:
            raise ValueError("Use the task's budgets frozen before native tuning.")
        writer.json("budgets.json", {"input_sha256": _hash(args.budgets), "budgets": budgets})
        summaries = [
            check_loading(
                api,
                writer,
                checks,
                args.reference,
                native,
                count,
                args.quadrature,
                args.tasks_per_env,
                budgets,
                args.work_warps,
            )
            for count in args.subdivisions
        ]
    checks.flush(writer)
    gating_checks = [row for row in checks.rows if row.get("gating", True)]
    passed = all(row["passed"] for row in gating_checks)
    writer.json(
        "result.json",
        {
            "complete": True,
            "passed": passed,
            "stage": args.stage,
            "bank": args.bank,
            "checks": len(checks.rows),
            "gating_checks": len(gating_checks),
            "loading_summaries": summaries,
            "failed_checks": [row["name"] for row in gating_checks if not row["passed"]],
            "seconds_including_native_preparation": time.perf_counter() - started,
            "scope": "Original Q4 elasticity and numerical gates; explicitly selected CPU side for primary planar-loading parity. Production both-side/actual collision qualification is separate.",
        },
    )
    if not passed:
        raise ArithmeticError("Native planar-loading mother/operator controls failed.")


if __name__ == "__main__":
    main()
