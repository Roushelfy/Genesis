"""Time admitted sharp-edge Franka rigid/max/full windows with a separate every-step audit."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


def observe(workload, mode):
    result = {"position_m": tensor_to_array(workload.egg.get_pos()).copy()}
    if mode != "rigid":
        entry = workload.scene.rigid_solver.stress_recovery.links[0]
        native = entry.hertz
        assert native is not None
        result.update(
            sampled_vm_pa=tensor_to_array(workload.link.get_max_sampled_stress()).copy(),
            vm_pa=tensor_to_array(workload.link.get_max_stress()).copy(),
            principal_pa=tensor_to_array(workload.link.get_max_principal_stress()).copy(),
            input_status=array(native.state.status),
            recovery_status=array(native.recovery.state.status),
            valid=array(entry.state.step_valid),
            loaded=np.any(
                array(entry.contacts.valid) & (np.linalg.norm(array(entry.contacts.force), axis=2) > 0), axis=1
            ),
            sharp_edge=np.any(array(native.friction_regions.profile) == 2, axis=1),
        )
        assert result["valid"].all() and (result["input_status"] == 0).all()
        assert (result["recovery_status"] <= 1).all()
        assert np.isfinite(result["sampled_vm_pa"]).all() and np.isfinite(result["principal_pa"]).all()
        np.testing.assert_array_equal(np.isposinf(result["vm_pa"]), result["sharp_edge"])
    return result


def compare(left, right):
    np.testing.assert_allclose(right["position_m"], left["position_m"], rtol=0, atol=1e-12)
    if "sampled_vm_pa" in left and "sampled_vm_pa" in right:
        for key in ["valid", "loaded", "sharp_edge", "input_status", "recovery_status"]:
            np.testing.assert_array_equal(right[key], left[key])
        for key in ["vm_pa", "sampled_vm_pa", "principal_pa"]:
            np.testing.assert_allclose(right[key], left[key], rtol=1e-7, atol=1e-6)


def build_workload(args, batch, mode):
    workload = FrankaEgg(
        batch,
        stress=mode != "rigid",
        seed=623016,
        varied=True,
        conditions=args.conditions,
        output_mode="full" if mode == "full" else "max",
        fused_rigid_post=False,
        contact_model="hertz",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        stress_mesh=args.stress_mesh,
        collision_urdf=args.collision_urdf,
    )
    workload.stress = False
    return workload


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ["stress-mesh", "collision-urdf", "conditions", "output"]:
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--batches", type=int, nargs="+", default=[4])
    cli.add_argument("--start-step", type=int, default=100)
    cli.add_argument("--steps", type=int, default=30)
    cli.add_argument("--repeats", type=int, default=3)
    args = cli.parse_args()
    assert args.start_step >= 0 and args.steps > 0 and args.repeats > 0
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "inputs": [file_id(path) for path in [args.stress_mesh, args.collision_urdf, args.conditions]],
            "collision_assets": [
                file_id(path)
                for path in sorted(
                    {
                        args.collision_urdf.parent / item.attrib["filename"]
                        for item in ElementTree.parse(args.collision_urdf).iter("mesh")
                    }
                )
            ],
            "batches": args.batches,
            "start_step": args.start_step,
            "steps": args.steps,
            "repeats": args.repeats,
            "scope": "Same-allocation pure-rigid/max/full, actual moving Franka and matched sharp collision/material geometry. Each timed window has continuous history from the same initial state. Native runtime acceptance, controller, dispatch and final synchronization are included. Audit reads/exports, resets and pre-window steps are outside timing. Admission is required; no physical accuracy claim. A short window does not establish full-cycle throughput. Larger batches tile the supplied condition bank.",
        },
    )
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    rows = []
    horizon = args.start_step + args.steps
    for batch in args.batches:
        audits = {}
        for mode in ["rigid", "max", "full"]:
            begun = perf_counter()
            workload = build_workload(args, batch, mode)
            preparation = perf_counter() - begun
            begun = perf_counter()
            snapshots = []
            for step in range(horizon):
                workload.step()
                snapshots.append(observe(workload, mode))
            workload.scene.rigid_solver.check_errno()
            audit_seconds = perf_counter() - begun
            audit = {key: np.stack([item[key] for item in snapshots]) for key in snapshots[0]}
            np.savez_compressed(args.output / f"{mode}-b{batch}-audit.npz", **audit)
            if mode != "rigid":
                compare(audits["rigid"], audit)
            if mode == "full":
                compare(audits["max"], audit)
            audits[mode] = audit
            samples = []
            for repeat in range(args.repeats):
                workload.restart()
                for step in range(args.start_step):
                    workload.step()
                qd.sync()
                begun = perf_counter()
                for step in range(args.steps):
                    workload.step()
                qd.sync()
                samples.append((perf_counter() - begun) / args.steps)
                workload.scene.rigid_solver.check_errno()
                compare(snapshots[-1], observe(workload, mode))
            entry = workload.scene.rigid_solver.stress_recovery.links[0] if mode != "rigid" else None
            median = float(np.median(samples))
            row = {
                "batch": batch,
                "mode": mode,
                "seconds_per_scene_step": samples,
                "median_seconds": median,
                "environment_steps_per_second": batch / median,
                "preparation_seconds": preparation,
                "audit_and_warmup_seconds": audit_seconds,
                "model_preparation": asdict(entry.model.build_timings) if entry is not None else None,
                "audit_environment_steps": horizon * batch,
                "admitted_timed_environment_steps": args.steps * batch * args.repeats if entry is not None else None,
                "surface_quadrature": entry.surface.info.gauss.shape[0] if entry is not None else None,
                "friction_penalty": entry.hertz.friction.penalty if entry is not None else None,
                "loaded_timed_environment_steps": int(audit["loaded"][args.start_step :].sum())
                if entry is not None
                else None,
                "sharp_timed_environment_steps": int(audit["sharp_edge"][args.start_step :].sum())
                if entry is not None
                else None,
            }
            rows.append(row)
            write_json(args.output / "progress.json", {"rows": rows})
            workload.scene.destroy()
        rigid = rows[-3]["median_seconds"]
        for row in rows[-2:]:
            row["added_seconds_per_scene_step"] = row["median_seconds"] - rigid
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
