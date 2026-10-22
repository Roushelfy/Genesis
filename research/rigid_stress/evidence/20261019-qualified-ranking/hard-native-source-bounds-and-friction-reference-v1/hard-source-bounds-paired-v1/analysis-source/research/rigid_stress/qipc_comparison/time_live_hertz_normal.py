"""Measure paired headless rigid/max/full steps through the actual native route."""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import quadrants as qd
from skfem import MeshTet

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_live_hertz_normal import build_fixture
from .check_native_hertz_recovery import array
from ..shell_comparison.submit import capture_source


def paired_sources(args):
    """Compare source variants on one allocation with isolated child caches."""
    current = Path(gs.__file__).resolve().parents[1]
    assert args.reference_source.resolve() != current
    capture_source(args.reference_source.resolve(), args.output, "reference-source")
    records = []
    for label, source in [("reference", args.reference_source.resolve()), ("current", current)]:
        with tempfile.TemporaryDirectory(prefix=f"live-hertz-{label}-", dir=os.environ["TMPDIR"]) as temporary:
            scratch = Path(temporary)
            environment = os.environ.copy()
            paths = [part for part in environment["PYTHONPATH"].split(os.pathsep) if Path(part).resolve() != current]
            environment["PYTHONPATH"] = os.pathsep.join([*paths, str(source)])
            cache_paths = {
                "GS_CACHE_FILE_PATH": "genesis",
                "QD_OFFLINE_CACHE_FILE_PATH": "quadrants",
                "CUDA_CACHE_PATH": "cuda",
                "NUMBA_CACHE_DIR": "numba",
                "MPLCONFIGDIR": "matplotlib",
            }
            for name, suffix in cache_paths.items():
                environment[name] = str(scratch / suffix)
            command = [
                sys.executable,
                "-m",
                "research.rigid_stress.qipc_comparison.time_live_hertz_normal",
                "--batches",
                *[str(batch) for batch in args.batches],
                "--output",
                str(args.output / label),
            ]
            with (args.output / f"{label}.log").open("w") as log:
                result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            records.append({"label": label, "source": str(source), "command": command, "returncode": result.returncode})
            write_json(args.output / "children.json", {"children": records})
            assert result.returncode == 0, records[-1]
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "children": records,
            "same_allocation": True,
            "separate_processes_and_private_compiler_caches": True,
            "whole_goal_complete": False,
        },
    )


def update(fixture, step, phase):
    position = np.zeros((len(phase), 3))
    position[:, 2] = 1e-7 * np.sin(0.3 * step + phase)
    fixture.sphere.set_pos(position)
    fixture.scene.step(update_visualizer=False, refresh_visualizer=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256, 1024])
    cli.add_argument("--reference-source", type=Path)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "scope": "Actual centered sphere/flat-link normal contact. Complete headless rigid/max/full scene steps, including vector input motion, native association, curvature/region prediction, acceptance, solve, moving probes and reductions. Two physical substeps per scene step. No research recovery producer or audit in primary timing. These are normal box scene rates, not Franka or general friction qualification.",
        },
    )
    if args.reference_source is not None:
        paired_sources(args)
        return
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(np.array([-0.004, 0, 0.004]), np.array([-0.004, 0, 0.004]), np.array([-0.002, 0, 0.002]))
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=mesh.p.T, tetrahedra=mesh.t.T, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    rows = []
    for batch in args.batches:
        audits = {}
        phase = 0.11 * np.arange(batch)
        for mode in ["rigid", "max", "full"]:
            begun = perf_counter()
            fixture = build_fixture(path, batch=batch, dt=2e-4, substeps=2, output_mode=mode)
            preparation = perf_counter() - begun
            for step in range(4):
                update(fixture, step, phase)
            qd.sync()
            elapsed = []
            for repeat in range(3):
                begun = perf_counter()
                for frame in range(8):
                    update(fixture, repeat * 8 + frame, phase)
                qd.sync()
                elapsed.append((perf_counter() - begun) / 8)
            # The native errno latches all rejected timed updates, even when a
            # later mechanical step clears derived field validity.
            fixture.scene.rigid_solver.check_errno()
            pose_rows, vm_rows, principal_rows, input_rows, field_rows, loaded_rows = [], [], [], [], [], []
            for step in range(24):
                update(fixture, step, phase)
                pose_rows.append(tensor_to_array(fixture.target.get_pos()).copy())
                if fixture.entry is not None:
                    entry = fixture.entry
                    assert entry.hertz is not None
                    normal = entry.hertz
                    vm = tensor_to_array(fixture.target.base_link.get_max_stress())
                    principal = tensor_to_array(fixture.target.base_link.get_max_principal_stress())
                    inputs, fields = array(normal.state.status), array(normal.recovery.state.status)
                    loaded = np.any(
                        array(entry.contacts.valid) & (np.linalg.norm(array(entry.contacts.force), axis=2) > 0), axis=1
                    )
                    assert (inputs == 0).all() and (fields <= 1).all()
                    assert array(entry.state.step_valid).all() and array(entry.state.valid).all()
                    assert np.isfinite(vm).all() and np.isfinite(principal).all()
                    vm_rows.append(vm.copy())
                    principal_rows.append(principal.copy())
                    input_rows.append(inputs)
                    field_rows.append(fields)
                    loaded_rows.append(loaded)
            fixture.scene.rigid_solver.check_errno()
            pose = np.stack(pose_rows)
            if mode == "rigid":
                np.savez_compressed(args.output / f"{mode}-b{batch}-audit.npz", positions_m=pose)
            else:
                data = {
                    "positions_m": pose,
                    "vm_pa": np.stack(vm_rows),
                    "principal_pa": np.stack(principal_rows),
                    "input_status": np.stack(input_rows),
                    "field_status": np.stack(field_rows),
                    "loaded": np.stack(loaded_rows),
                }
                np.savez_compressed(args.output / f"{mode}-b{batch}-audit.npz", **data)
                audits[mode] = data
                np.testing.assert_allclose(pose, audits["rigid"]["positions_m"], rtol=0, atol=1e-12)
                if mode == "full":
                    np.testing.assert_allclose(data["vm_pa"], audits["max"]["vm_pa"], rtol=1e-7, atol=1e-7)
                    np.testing.assert_allclose(
                        data["principal_pa"], audits["max"]["principal_pa"], rtol=1e-7, atol=1e-7
                    )
            if mode == "rigid":
                audits[mode] = {"positions_m": pose}
            seconds = float(np.median(elapsed))
            entry = fixture.entry
            rows.append(
                {
                    "batch": batch,
                    "mode": mode,
                    "seconds_per_scene_step": elapsed,
                    "median_seconds": seconds,
                    "environment_scene_steps_per_second": batch / seconds,
                    "physical_substeps_per_scene_step": 2,
                    "preparation_seconds": preparation,
                    "model_preparation": asdict(entry.model.build_timings) if entry is not None else None,
                    "contact_slots_per_environment": entry.contacts.valid.shape[0] if entry is not None else None,
                    "audit_environment_steps": 24 * batch,
                    "admitted_audit_environment_steps": 24 * batch if entry is not None else None,
                    "admitted_loaded_audit_steps": int(np.stack(loaded_rows).sum()) if entry is not None else None,
                }
            )
            write_json(args.output / "progress.json", {"rows": rows})
            fixture.scene.destroy()
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "mesh": file_id(path),
            "actual_native_route_without_research_recovery_callback": True,
            "scene_input_generation_included": True,
            "research_audits_outside_primary_timing": True,
            "same_allocation_rigid_max_full": True,
            "general_friction_curved_or_franka_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
