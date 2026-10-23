"""Compose captured native sharp-edge inputs with native complete-shell recovery."""

import argparse
import json
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.edge import StressEdge
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.hertz_tangential import StressHertzShear
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_curved_shell_operators import locate_curved
from .check_native_hertz_recovery import array, matrix
from .study_edge_fields import field_metrics
from .study_hertz_residual import field_difference


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    for name in ["binding", "mesh", "reference", "reference_geometry", "output"]:
        cli.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    cli.add_argument("--moving-probes", action="store_true")
    cli.add_argument("--check-max", action="store_true")
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(p) for p in [args.binding, args.mesh, args.reference, args.reference_geometry]],
            "targets": {"volume_tensor_relative": 1e-3, "common_probe_tensor_relative": 1e-3},
            "scope": "Native complete-field composition from captured native pressure inputs. Retains all contact slots and all environments. Same approximate point boundary correction and proportional edge shear as the cached field reference. This checks composition and input conformance, not independent physical accuracy or live-scene throughput.",
            "moving_probes": args.moving_probes,
            "check_max": args.check_max,
            "edge_peak": "Ideal sharp-edge pressure has an unbounded continuum limit. Finite sampled tensors use declared positive depths. Automatic edge probes start at 0.01 widths, with 89 samples per loaded source. Full query positions are retained.",
        },
    )
    with np.load(args.binding) as data:
        binding = {key: data[key] for key in data.files}
    with np.load(args.reference) as data:
        reference = {key: data[key] for key in data.files}
    cases = json.loads(args.reference_geometry.read_text())["cases"]
    selected = [i for i, case in enumerate(cases) if case["kind"] == "allocated"]
    expected = reference["complete_pa"][selected].sum(axis=0)
    batch, regions = binding["source_count"].shape
    np.testing.assert_array_equal(binding["status"], 0)
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions.hertz(mesh=args.mesh, method="direct"))
    count = batch * regions
    law, edge = StressHertz(count), StressEdge(count)
    for field, key in [
        (law.state.force, "normal_force_n"),
        (law.state.axes, "axes_m"),
        (law.state.direction, "direction"),
        (law.state.peak_pressure, "amplitude_pa"),
        (law.state.status, "hertz_status"),
        (edge.state.exponent, "edge_exponent"),
        (edge.state.amplitude, "amplitude_pa"),
        (edge.state.status, "edge_status"),
    ]:
        field.from_numpy(binding[key])
    edge.state.force.from_numpy(binding["normal_force_n"])
    edge.state.width.from_numpy(binding["axes_m"][:, 1].copy())
    edge.state.half_length.from_numpy(binding["axes_m"][:, 0].copy())
    shear = StressHertzShear(
        friction=V(dtype=gs.qd_float, shape=(count,)),
        force=V_VEC(2, dtype=gs.qd_float, shape=(count,)),
        coefficients=V_VEC(2, dtype=gs.qd_float, shape=(16, count)),
    )
    forces = binding["force_n"].transpose(1, 0, 2).reshape(count, 3)
    direction = binding["direction"]
    rotation = np.zeros((count, 3, 3))
    rotation[:, 0, 0], rotation[:, 1, 0] = direction[:, 0], direction[:, 1]
    rotation[:, 0, 1], rotation[:, 1, 1] = -direction[:, 1], direction[:, 0]
    rotation[:, 2, 2] = 1
    basis = binding["basis"] @ rotation
    local_force = np.einsum("ni,nij->nj", forces, basis)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", basis, local_force), forces, rtol=1e-12, atol=1e-14)
    shear.force.from_numpy(local_force[:, :2].copy())
    coefficients = np.zeros((16, count, 2))
    loaded = binding["normal_force_n"] > 0
    shear.friction.fill(1)
    coefficients[15, loaded] = local_force[loaded, :2] / binding["normal_force_n"][loaded, None]
    shear.coefficients.from_numpy(coefficients)
    positions = reference["points_m"]
    vertices, elements = array(model.info.vertices), array(model.info.elements).T
    owners, barycentric, valid, _ = locate_curved(vertices[elements], positions)
    assert valid.all(), np.flatnonzero(~valid)
    module = StressHertzRecovery(
        batch,
        regions,
        len(positions),
        output_mode="full",
        moving_probes=args.moving_probes,
        shear=shear,
        edge=edge.state,
    )
    for field, key in [
        (module.frames.center, "center_m"),
        (module.frames.basis, "basis"),
        (module.frames.anchor, "anchor"),
    ]:
        field.from_numpy(binding[key])
    module.queries.position.from_numpy(positions)
    module.queries.element.from_numpy(owners.astype(gs.np_int))
    module.queries.barycentric.from_numpy(barycentric)
    surface = StressSurface(24, model.info)
    state = model.create_state(batch, "full")
    omega = V_VEC(3, dtype=gs.qd_float, shape=(batch,))
    errno = V(dtype=gs.qd_int, shape=(batch,))
    omega.fill(0)
    errno.fill(0)
    kernel_begin_step(state)
    module.scatter(law.state, state, model.info, surface.info, 0.3, 1e-12, errno)
    write_json(
        args.output / "scatter.json",
        {
            "status": array(module.state.status).tolist(),
            "errno": array(errno).tolist(),
            "wrench": array(module.state.wrench_error).tolist(),
        },
    )
    np.testing.assert_array_equal(array(module.state.status), 0)
    np.testing.assert_array_equal(array(errno), 0)
    model.recover(omega, state, surface_load=True, compute_peak=False)
    module.evaluate(law.state, state, model, 1e-12, errno)
    qd.sync()
    actual = matrix(array(module.state.tensor))
    np.savez_compressed(
        args.output / "fields.npz",
        points_m=positions,
        complete_pa=actual,
        corner_pa=matrix(array(state.stress_tensor)),
        displacement_m=array(state.displacement),
        correction_forces_n=array(state.force),
        probe_positions_m=array(module.probes.position),
        probe_elements=array(module.probes.element),
        probe_pa=matrix(array(module.probes.tensor)),
        sampled_vm_pa=array(state.peak),
        sampled_tensile_pa=array(module.state.tensile_peak),
    )
    weights = reference["volume_weights_m3"]
    rows = []
    for env in range(batch):
        rows.append(
            {
                "environment": env,
                "volume_tensor_relative": float(
                    field_difference(actual[env, : len(weights)], expected[: len(weights)], weights)
                ),
                "common_probes": field_metrics(actual[env, len(weights) :], expected[len(weights) :]),
                "all_samples": field_metrics(actual[env], expected),
            }
        )
    passed = bool(
        np.all(array(module.state.status) == 0)
        and np.all(array(errno) == 0)
        and all(row["volume_tensor_relative"] < 1e-3 and row["common_probes"]["tensor_relative"] < 1e-3 for row in rows)
    )
    maximum = None
    if args.check_max:
        max_module = StressHertzRecovery(
            batch,
            regions,
            len(positions),
            output_mode="max",
            moving_probes=args.moving_probes,
            shear=shear,
            edge=edge.state,
        )
        for field, key in [
            (max_module.frames.center, "center_m"),
            (max_module.frames.basis, "basis"),
            (max_module.frames.anchor, "anchor"),
        ]:
            field.from_numpy(binding[key])
        max_module.queries.position.from_numpy(positions)
        max_module.queries.element.from_numpy(owners.astype(gs.np_int))
        max_module.queries.barycentric.from_numpy(barycentric)
        max_state = model.create_state(batch, "max")
        kernel_begin_step(max_state)
        errno.fill(0)
        max_module.scatter(law.state, max_state, model.info, surface.info, 0.3, 1e-12, errno)
        model.recover(omega, max_state, surface_load=True, compute_peak=False)
        max_module.evaluate(law.state, max_state, model, 1e-12, errno)
        np.testing.assert_array_equal(array(max_module.state.status), 0)
        np.testing.assert_array_equal(array(errno), 0)
        np.testing.assert_allclose(array(max_state.peak), array(state.peak), rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(
            array(max_module.state.tensile_peak), array(module.state.tensile_peak), rtol=1e-10, atol=1e-6
        )
        maximum = {
            "status": array(max_module.state.status).tolist(),
            "sampled_vm_pa": array(max_state.peak).tolist(),
            "sampled_tensile_pa": array(max_module.state.tensile_peak).tolist(),
            "agrees_with_full": True,
        }
    write_json(
        args.output / "result.json",
        {
            "passed": passed,
            "status": array(module.state.status).tolist(),
            "errno": array(errno).tolist(),
            "step_valid": array(state.step_valid).tolist(),
            "rows": rows,
            "sampled_vm_pa": array(state.peak).tolist(),
            "sampled_tensile_pa": array(module.state.tensile_peak).tolist(),
            "moving_probes": args.moving_probes,
            "max_output": maximum,
            "whole_goal_complete": False,
        },
    )
    assert passed


if __name__ == "__main__":
    main()
