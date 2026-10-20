"""Compare the generic native boundary transfer to retained physical pad fields."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

import genesis as gs
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.engine.solvers.rigid.stress.resolved import kernel_scatter_resolved
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import voigt_tensor
from .native_pad_recovery import NativePadRecovery
from .plates import plate_xml


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--cpu", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.run / "start.json").read_text())
    terminal = json.loads((args.run / "result.json").read_text())
    cpu = json.loads((args.cpu / "result.json").read_text())
    assert terminal["completed"] and terminal["mechanical_history_passed"] and cpu["passed"]
    assert all(w["mechanical_window_passed"] for w in terminal["windows"])
    params = SimpleNamespace(**start["args"])
    params.assets, params.output = Path(params.assets), args.output
    params.stress_output, params.primary_cost = "full", False
    assert params.method == "dpr" and params.source_reduction == "none"
    assert file_id(Path(params.mesh)) == start["mesh"]
    with np.load(params.mesh) as saved:
        outer, faces = saved["outer_vertices"], saved["mid_faces"]
    write_json(args.output / "start.json", {
        "identity": runtime_identity(), "mesh": start["mesh"],
        "original_start": file_id(args.run / "start.json"), "original_result": file_id(args.run / "result.json"),
        "cpu_result": file_id(args.cpu / "result.json"), "field_relative_target": 1e-6,
        "scope": "Generic material-boundary resolver and production P2 scatter on actual qualified pad inputs. The old research adapter only supplies its established scene/input buffers and a paired field; the new route resolves all native boundary faces and transfers independently.",
    })
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    xml = plate_xml(float(outer[:, 2].min() - params.clearance), float(outer[:, 2].max() + params.clearance))
    baseline = NativePadRecovery.allocate(params, outer, faces, outer[faces].mean(axis=1), xml, "resolved_outer_p2")
    entry = baseline.entry
    sources = create_sources(*entry.contacts.valid.shape)
    sources.geom.fill(entry.link.geoms[0].idx)
    sources.pair.fill(0)
    rows = []
    for case in cpu["cases"]:
        level = case["level"]
        physical = args.run / f"frames/level{level}.npz"
        original = args.run / f"frames/stress-level{level}.npz"
        assert file_id(original) == case["original"]
        with np.load(physical) as saved:
            force, rotation, angular = saved["contact_force_n"], saved["rotation"], saved["angular_velocity_rad_per_s"]
        with np.load(original) as saved:
            load = saved["load_n"].item()
        reference_path = Path(case["variants"]["distributed_outer"]["field"]["path"])
        assert file_id(reference_path) == case["variants"]["distributed_outer"]["field"]
        old = baseline.sample(force, rotation, angular, params.friction, load, physical_law_admitted=True)
        assert old["native_valid"]
        sources.witness.from_numpy(np.swapaxes(qd_to_numpy(entry.contacts.position, transpose=True), 0, 1).copy())
        kernel_begin_step(entry.state)
        epsilon = float(np.finfo(gs.np_float).eps)
        kernel_resolve_sources(entry.options.source_projection_limit, epsilon, entry.contacts, sources,
                               entry.model.info, entry.surface.info)
        kernel_scatter_resolved(epsilon, entry.contacts, sources, entry.state, entry.model.info, entry.surface.info)
        state_path = args.output / f"level{level}-sources.npz"
        np.savez_compressed(state_path,
            position=qd_to_numpy(entry.contacts.position, transpose=True),
            force=qd_to_numpy(entry.contacts.force, transpose=True),
            normal=qd_to_numpy(entry.contacts.normal, transpose=True),
            valid=qd_to_numpy(entry.contacts.valid, transpose=True),
            contact_status=qd_to_numpy(entry.contacts.status, transpose=True),
            source_status=qd_to_numpy(sources.status, transpose=True),
            anchor_face=qd_to_numpy(entry.contacts.anchor_face, transpose=True),
            moment_error=qd_to_numpy(entry.contacts.moment_error, transpose=True))
        status = qd_to_numpy(entry.contacts.status, transpose=True)
        assert not status.any(), (level, np.unique(status, return_counts=True))
        entry.model.recover(entry.omega, entry.state, surface_load=True)
        kernel_accept(entry.state, entry.contacts, baseline.errno)
        assert qd_to_numpy(entry.state.step_valid, transpose=True).all()
        actual = {
            "stress_pa": voigt_tensor(qd_to_numpy(entry.state.stress_tensor, transpose=True)[0]),
            "forces_n": qd_to_numpy(entry.state.force, transpose=True)[0],
            "rhs_n": qd_to_numpy(entry.state.rhs, transpose=True)[0],
        }
        with np.load(reference_path) as saved:
            distance, permutation = cKDTree(saved["p2_coordinates_m"]).query(baseline.coordinates)
            assert distance.max() < 1e-14 and len(np.unique(permutation)) == len(baseline.coordinates)
            errors = {key: float(np.linalg.norm(value - (saved[key] if key == "stress_pa" else saved[key][permutation])) /
                                np.linalg.norm(saved[key])) for key, value in actual.items()}
        assert max(errors.values()) <= 1e-6, errors
        path = args.output / f"level{level}-field.npz"
        np.savez_compressed(path, **actual)
        rows.append({"level": level, "relative_errors": errors, "physical": file_id(physical),
                     "cpu_field": file_id(reference_path), "field": file_id(path), "sources": file_id(state_path)})
        write_json(args.output / "progress.json", {"rows": rows})
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False,
        "scope": "Complete same-load numerical transfer comparison. Physical model accuracy is inherited only within the existing declared pad problem; no automatic support, new dynamics or matched production cost is established."})


if __name__ == "__main__":
    main()
