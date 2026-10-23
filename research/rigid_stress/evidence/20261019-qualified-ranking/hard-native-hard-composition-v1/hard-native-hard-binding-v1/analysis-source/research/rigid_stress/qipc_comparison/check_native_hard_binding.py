"""Replay actual rigid inputs through native grouped sharp-edge pressure and force updates."""

import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.hard_binding import StressHardBinding
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegions
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


def snapshot(module):
    return {
        "active": array(module.state.is_active),
        "status": array(module.state.status),
        "profile": array(module.state.profile),
        "geometry_status": array(module.state.geometry_status),
        "leader": array(module.state.leader),
        "source_count": array(module.state.source_count),
        "force_n": array(module.state.force),
        "original_moment_nm": array(module.state.original_moment),
        "minimum_m": array(module.state.minimum),
        "geometry_basis": array(module.state.basis),
        "center_m": array(module.frames.center),
        "basis": array(module.frames.basis),
        "anchor": array(module.frames.anchor),
        "axes_m": array(module.hertz.state.axes),
        "normal_force_n": array(module.hertz.state.force),
        "direction": array(module.hertz.state.direction),
        "amplitude_pa": array(module.hertz.state.peak_pressure),
        "edge_exponent": array(module.edge.state.exponent),
        "edge_status": array(module.edge.state.status),
        "hertz_status": array(module.hertz.state.status),
        "force_error": array(module.state.force_error),
        "moment_error": array(module.state.moment_error),
        "cone_margin": array(module.state.cone_margin),
        "allocation_status": array(module.allocation.state.status),
        "maximum_change": array(module.allocation.state.maximum_change),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    for name in ["frame", "boundaries", "geometry", "raw-contacts", "mesh", "reference", "output"]:
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [
                file_id(p)
                for p in [args.frame, args.boundaries, args.geometry, args.raw_contacts, args.mesh, args.reference]
            ],
            "targets": {"center_m": 1e-9, "force_relative": 1e-5, "support_relative": 5e-4, "wrench_relative": 1e-8},
            "scope": "Captured actual source provenance, current finite-face roots and native material anchoring feed grouped pressure laws and three native force/support feedback updates. Compare the independent grouped offline result in environment0; retain every environment status. Merged collision-sample invariance, original total wrench, friction cones, unload/isolation and restoration are checked. This is input conformance, not recovered stress or accepted full-scene throughput.",
        },
    )
    data = []
    for path in [args.frame, args.boundaries, args.geometry, args.raw_contacts]:
        with np.load(path) as saved:
            data.append({key: saved[key] for key in saved.files})
    frame, bounds, geometry, raw = data
    batch, count = frame["contact_valid"].shape
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions.hertz(mesh=args.mesh, method="direct"))
    surface = StressSurface(4, model.info)
    np.testing.assert_array_equal(array(model.info.surface_nodes), frame["material_surface_nodes"])
    contacts, sources = create_contacts(count, batch, 0.001), create_sources(count, batch)
    for field, key in [
        (contacts.force, "contact_force_n"),
        (contacts.position, "contact_position_m"),
        (contacts.normal, "contact_normal"),
        (sources.witness, "source_witness_m"),
    ]:
        field.from_numpy(frame[key].transpose(1, 0, 2))
    for field, key in [(contacts.valid, "contact_valid"), (sources.geom, "source_geom"), (sources.pair, "source_pair")]:
        field.from_numpy(frame[key].T)
    contacts.frame_position.from_numpy(frame["frame_position_m"])
    contacts.frame_quaternion.from_numpy(frame["frame_quaternion"])
    indices = raw["sort_idx"][:, :count]
    envs = np.arange(batch)[:, None]
    first, second = raw["geom_a"][envs, indices], raw["geom_b"][envs, indices]
    other = np.where(first == frame["source_geom"], second, first)
    opponent_type = geometry["geom_type"][np.maximum(other, 0)]
    opponent_type[~frame["contact_valid"]] = -1
    contacts.friction.from_numpy(raw["friction"][envs, indices].T)
    region = StressHertzRegions(contacts).state
    for field, values in [
        (region.center, frame["region_center_m"]),
        (region.normal, bounds["normal"]),
        (region.tangent, bounds["tangent"]),
        (region.curvature, frame["region_curvature_per_m"]),
        (region.boundary_planes, bounds["planes"]),
    ]:
        field.from_numpy(np.swapaxes(values, 0, 1))
    region.status.from_numpy(frame["region_status"].T)
    region.n_boundaries.from_numpy(bounds["count"].T)
    region.opponent_type.from_numpy(opponent_type.T)
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-4, epsilon, contacts, sources, model.info, surface.info)
    module = StressHardBinding(contacts, model.info)
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    result = snapshot(module)
    np.savez_compressed(args.output / "binding.npz", **result)
    write_json(
        args.output / "numerical.json",
        {
            key: result[key].tolist()
            for key in [
                "active",
                "status",
                "geometry_status",
                "profile",
                "leader",
                "source_count",
                "force_error",
                "moment_error",
                "cone_margin",
                "allocation_status",
                "maximum_change",
            ]
        },
    )
    assert result["status"][0] == 0, result["status"]
    references = [row for row in json.loads(args.reference.read_text())["cases"] if row["kind"] == "allocated"]
    expected_leaders = sorted(row["contact"] for row in references)
    np.testing.assert_array_equal(np.flatnonzero(result["source_count"][0]), expected_leaders)
    rows = []
    for row in references:
        contact = row["contact"]
        index = contact * batch
        np.testing.assert_array_equal(np.flatnonzero(result["leader"][0] == contact), row.get("contacts", [contact]))
        reference_force = np.array(row["basis"]) @ row["force"]
        force_error = np.linalg.norm(result["force_n"][0, contact] - reference_force) / np.linalg.norm(reference_force)
        center_error = np.linalg.norm(result["center_m"][index] - row["center"])
        axes_error = np.max(np.abs(result["axes_m"][index] / [row["along"], row["width"]] - 1))
        amplitude_error = abs(result["amplitude_pa"][index] / row["amplitude"] - 1)
        rows.append(
            {
                "contact": contact,
                "center_error_m": float(center_error),
                "force_relative": float(force_error),
                "axes_relative": float(axes_error),
                "amplitude_relative": float(amplitude_error),
            }
        )
    write_json(args.output / "comparison.json", {"rows": rows})
    assert max(row["center_error_m"] for row in rows) < 1e-9, rows
    assert max(row["force_relative"] for row in rows) < 1e-5, rows
    assert max(max(row["axes_relative"], row["amplitude_relative"]) for row in rows) < 5e-4, rows
    wrench_rows = []
    for env in np.flatnonzero(result["status"] == 0):
        leaders = np.flatnonzero(result["source_count"][env] > 0)
        selected = leaders * batch + env
        forces = result["force_n"][env, leaders]
        raw_force, raw_position = frame["contact_force_n"][env], frame["contact_position_m"][env]
        active = frame["contact_valid"][env] & (np.linalg.norm(raw_force, axis=1) > 0)
        force_error = (
            np.linalg.norm(forces.sum(axis=0) - raw_force[active].sum(axis=0))
            / np.linalg.norm(raw_force[active], axis=1).sum()
        )
        moment = np.cross(result["center_m"][selected], forces).sum(axis=0)
        original = np.cross(raw_position[active], raw_force[active]).sum(axis=0)
        moment_error = np.linalg.norm(moment - original) / np.sum(
            result["normal_force_n"][selected] * result["axes_m"][selected, 0]
        )
        assert max(force_error, moment_error) < 1e-8
        wrench_rows.append(
            {"environment": int(env), "force_error": float(force_error), "moment_error": float(moment_error)}
        )
    # Merge the two samples of each refined gripper region and retain their free couple.
    merged_force, merged_valid = frame["contact_force_n"].copy(), frame["contact_valid"].copy()
    couples = np.zeros_like(merged_force)
    for row in references:
        members = row.get("contacts", [row["contact"]])
        leader = row["contact"]
        force = frame["contact_force_n"][0, members]
        position = frame["contact_position_m"][0, members]
        merged_force[0, leader] = force.sum(axis=0)
        couples[0, leader] = np.cross(position - frame["contact_position_m"][0, leader], force).sum(axis=0)
        for contact in members[1:]:
            merged_force[0, contact], merged_valid[0, contact] = 0, False
    contacts.force.from_numpy(merged_force.transpose(1, 0, 2))
    contacts.valid.from_numpy(merged_valid.T)
    sources.free_moment.from_numpy(couples.transpose(1, 0, 2))
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    merged = snapshot(module)
    np.testing.assert_array_equal(merged["status"], result["status"])
    np.testing.assert_allclose(
        merged["force_n"][0, expected_leaders], result["force_n"][0, expected_leaders], rtol=1e-9, atol=1e-13
    )
    for key in ["center_m", "axes_m", "amplitude_pa"]:
        np.testing.assert_allclose(
            merged[key][np.array(expected_leaders) * batch],
            result[key][np.array(expected_leaders) * batch],
            rtol=1e-9,
            atol=1e-13,
        )
    merged_valid[0] = False
    contacts.valid.from_numpy(merged_valid.T)
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    unloaded = snapshot(module)
    assert unloaded["status"][0] == 1 and not unloaded["active"][0]
    np.testing.assert_array_equal(unloaded["source_count"][0], 0)
    np.testing.assert_array_equal(unloaded["force_n"][1:], result["force_n"][1:])
    contacts.valid.from_numpy(frame["contact_valid"].T)
    contacts.force.from_numpy(frame["contact_force_n"].transpose(1, 0, 2))
    sources.free_moment.fill(0)
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    restored = snapshot(module)
    np.testing.assert_array_equal(restored["status"], result["status"])
    np.testing.assert_array_equal(restored["force_n"], result["force_n"])
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "comparison": rows,
            "wrenches": wrench_rows,
            "environment_status": result["status"].tolist(),
            "merged_source_invariance": True,
            "unload_isolation_and_restore": True,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
