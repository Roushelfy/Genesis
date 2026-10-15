"""Compare complete spatial fields at one unchanged physical time, retaining recovery provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_articulated_contact import quadrature_fields
from .audit_objective_frame import finite_frame
from .audit_original_history_fields import wide_volume_bounds
from .audit_precise_spatial_state import assemble_state
from .compare_spatial_franka import common_probes, point_fields, qualify, weighted_change
from .qualify_articulated_precision import tensor_from_f
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--recovery-run", type=Path, required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument("--step", type=int, default=26)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) >= 3 and args.step > 1
    for run in [*args.run, args.recovery_run, args.recovery_audit]:
        closed_source(run)
    assert int((args.recovery_audit / "returncode.txt").read_text()) == 0
    audited = json.loads((args.recovery_audit / "results/summary.json").read_text())
    assert audited["recovered_step"] == args.step
    assert Path(audited["source_run"]).resolve() == args.run[-1].resolve()
    assert Path(audited["recovery_run"]).resolve() == args.recovery_run.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, rows, fields, hashes = None, [], [], {}
    for index, run in enumerate(args.run):
        static, config = restore_static(run)
        physical = {k: v for k, v in config.items() if k not in ["local_levels", "output"]}
        if baseline is None:
            baseline = physical
        assert physical == baseline
        if index == 0:
            points, weights = common_probes(static.body)
        body = static.body
        audit_points = qualify(body, points)
        path = (
            args.recovery_run / "results/recovered.npz"
            if index == len(args.run) - 1
            else run / f"results/step{args.step:04d}.npz"
        )
        with np.load(path) as saved:
            u, force = saved["displacement"].copy(), saved["contact_force_N"].copy()
        with np.load(run / f"results/step{args.step - 1:04d}.npz") as saved:
            previous, velocity = saved["displacement"].copy(), saved["velocity"].copy()
        metrics, checked = assemble_state(static, config, u, previous, velocity, None)
        n = body.basis.N
        vector = np.empty(n, dtype=np.longdouble)
        vector[static.parts[0].permutation] = u[:n]
        bulk = WidePiolaBulk(body)
        f = bulk.corner_deformation_gradients(vector)
        tensor = tensor_from_f(f, body.lame)
        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))
        j_lower, upper = wide_volume_bounds(f, bulk.mu)
        rotation, _, local = finite_frame(body, np.asarray(vector, dtype=float))
        sampled, probe_vm, probe_u = point_fields(body, vector, points)
        sampled = rotation.T @ sampled @ rotation
        _, energy = quadrature_fields(bulk, np.asarray(vector, dtype=float))
        cell, corner = np.unravel_index(np.argmax(vm), vm.shape)
        row = {
            "run": str(run),
            "state_source": str(path),
            "step": args.step,
            "time_s": args.step * config["dt"],
            "local_levels": config["local_levels"],
            "tetrahedra": body.mesh.nelements,
            **metrics,
            "original_saved_contact_force_difference_N": float(np.linalg.norm(checked["contact_force_N"] - force)),
            "raw_peak_lower_Pa": float(vm.max()),
            "raw_peak_upper_Pa": upper,
            "raw_peak_reference_position_m": body.corner_positions[cell, corner].tolist(),
            "whole_volume_J_lower_bound": j_lower,
            "nonlinear_energy_J": energy,
            "point_evaluator_qualification": audit_points,
        }
        rows.append(row)
        fields.append({"tensor": sampled, "vm": probe_vm, "displacement": probe_u})
        np.savez_compressed(
            args.output / f"case{index:02d}.npz",
            vertices=body.mesh.p.T,
            tetrahedra=body.mesh.t.T,
            raw_corner_tensor_Pa=tensor,
            raw_corner_vm_Pa=vm,
            objective_displacement_m=local,
            probe_tensor_Pa=sampled,
            probe_vm_Pa=probe_vm,
            probe_displacement_m=probe_u,
            **checked,
        )
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    comparisons = []
    for index in range(1, len(rows)):
        row = {"coarse": index - 1, "fine": index}
        for key in ["tensor", "vm", "displacement"]:
            row[key] = weighted_change(fields[index - 1][key][None], fields[index][key][None], weights)
        for key in ["raw_peak_lower_Pa", "nonlinear_energy_J"]:
            row[key] = abs(rows[index - 1][key] / rows[index][key] - 1)
        comparisons.append(row)
    np.savez_compressed(args.output / "common_probes.npz", points_m=points, volume_weights_m3=weights)
    result = {
        "scope": __doc__,
        "rows": rows,
        "comparisons": comparisons,
        "common_probes": len(points),
        "physical_configuration": baseline,
        "input_sha256": hashes,
        "goal_completion": False,
        "limitations": "One matched physical time with fixed original physical inputs and complete independently reconstructed1e-9N body/mapped force gates. The finest recovered state follows its original accepted25-state prefix, not a completed new80-step trajectory. Lumped support areas are quadrature diagnostics, not independently resolved physical contact boundaries. All raw peaks and common physical points remain visible. Whole histories, credible spatial/contact/geometry tails, application/model/feedback and verified warmed costs remain incomplete.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": rows, "comparisons": comparisons, "common_probes": len(points)}))


if __name__ == "__main__":
    main()
