"""Separate sampled-cone infeasibility from numerical failure in saved native contacts.

Replay preserves the snapshot's anchor, force point, radius and actual friction.
The native profile is reconstructed using the declared port-path law. Independent
CPU LPs use inscribed and circumscribed friction polygons: inner feasibility is
a constructive Coulomb witness, while outer infeasibility bounds the exact
sampled cones. Neither conclusion certifies a continuous footprint model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_surface_fan_tractions,
    create_footprints,
    kernel_surface_fan_profiles,
    kernel_footprint_ports,
    kernel_footprint_profiles,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info,
    create_traction_samples,
    create_tractions,
    kernel_fit_tractions,
    kernel_traction_samples,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy


def cone_lp(relative, normals, friction, target, sides, outer):
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    first = np.cross(normals, axis)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(normals, first)
    angles = np.arange(sides) * (2.0 * np.pi / sides)
    coefficient = friction / np.cos(np.pi / sides) if outer else friction
    rays = normals[:, None] + coefficient * (
        first[:, None] * np.cos(angles)[None, :, None] + second[:, None] * np.sin(angles)[None, :, None]
    )
    # Add the exact target force direction wherever it is inside the true cone.
    # A fixed angular polygon alone can reject a force arbitrarily close to
    # sliding; these additional rays still provide constructive inner witnesses.
    force = target[:3]
    compression = normals @ force
    tangent = force - compression[:, None] * normals
    admissible = (compression >= 0.0) & (np.linalg.norm(tangent, axis=1) <= friction * compression)
    aligned = np.where(admissible[:, None], force, 0.0)
    rays = np.concatenate((rays, aligned[:, None]), axis=1)
    columns = np.concatenate((rays, np.cross(relative[:, None], rays)), axis=2).reshape(-1, 6).T
    result = linprog(
        np.zeros(columns.shape[1]),
        A_eq=columns,
        b_eq=target,
        bounds=(0.0, None),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    record = {"sides": sides, "outer": outer, "status": int(result.status), "message": result.message}
    forces = None
    if result.success:
        forces = np.sum(result.x.reshape(len(relative), sides + 1, 1) * rays, axis=1)
        wrench = np.concatenate((forces.sum(axis=0), np.cross(relative, forces).sum(axis=0)))
        compression = np.einsum("ij,ij->i", forces, normals)
        tangent = forces - compression[:, None] * normals
        record["wrench_error"] = float(np.linalg.norm(wrench - target))
        record["minimum_compression"] = float(compression.min())
        record["maximum_cone_violation"] = float((np.linalg.norm(tangent, axis=1) - friction * compression).max())
        if not outer:
            assert record["wrench_error"] <= 1e-8
            assert record["minimum_compression"] >= -1e-12
            assert record["maximum_cone_violation"] <= 1e-12
    return record, forces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--mesh", type=Path, default=Path("examples/rigid/assets/hollow_egg/level1/elastic.npz"))
    parser.add_argument("--anchor-fan", action="store_true")
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    if args.packed and not args.anchor_fan:
        parser.error("--packed requires --anchor-fan.")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions(mesh=args.mesh, method="direct"))
    surface = StressSurface(10, model.info)
    paths = StressSurfacePaths(model.info, 8)
    info = create_traction_info(surface.info)
    positions = qd_to_numpy(info.positions, transpose=True, copy=True)
    normals = qd_to_numpy(info.inward_normals, transpose=True, copy=True)
    contacts = create_contacts(1, 1, 0.001)
    sources = create_sources(1, 1)
    footprints = create_footprints(paths.info.positions.shape[0], 1, 1)
    tractions = (
        create_surface_fan_tractions(surface.info, 1, 1)
        if args.anchor_fan
        else create_tractions(len(positions), 1, 1)
    )
    epsilon = float(np.finfo(gs.np_float).eps)
    exported = create_traction_samples(tractions) if args.anchor_fan else None
    records = []
    for snapshot in sorted(args.run_dir.glob("snapshot-*.npz")):
        with np.load(snapshot) as saved:
            loaded = saved["valid"] & (np.linalg.norm(saved["force"], axis=2) > 1e-12)
            failed = loaded & (saved["traction_status"] == 3) & (saved["source_status"] == 0)
            for i_b, i_c in zip(*np.nonzero(failed)):
                position, anchor, force = (saved[key][i_b, i_c] for key in ["position", "anchor", "force"])
                radius = float(saved["radius"][i_b, i_c])
                friction = float(saved["friction"][i_b, i_c])
                anchor_face = int(saved["anchor_face"][i_b, i_c])
                moment = np.cross(position - anchor, force)
                contacts.valid.fill(True)
                contacts.position.from_numpy(position[None, None])
                contacts.center.from_numpy(anchor[None, None])
                contacts.force.from_numpy(force[None, None])
                contacts.radius.fill(radius)
                contacts.friction.fill(friction)
                contacts.anchor_face.fill(anchor_face)
                sources.status.fill(0)
                sources.anchor_moment.from_numpy(moment[None, None])
                kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
                if args.anchor_fan:
                    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info)
                    kernel_traction_samples(exported, tractions, surface.info, info)
                    positions = qd_to_numpy(exported.positions, transpose=True, copy=True).reshape(-1, 3)
                    normals = qd_to_numpy(exported.normals, transpose=True, copy=True).reshape(-1, 3)
                else:
                    kernel_footprint_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info)
                profile = qd_to_numpy(tractions.profile, transpose=True, copy=True).reshape(-1)
                count = int(qd_to_numpy(tractions.sample_count, transpose=True).item())
                eligible = (np.arange(len(profile)) < count) & (profile > 0.0)
                if not args.anchor_fan:
                    assert eligible.sum() == saved["samples"][i_b, i_c]
                relative = (positions[eligible] - anchor) / radius
                scale = max(np.linalg.norm(force), np.linalg.norm(moment) / radius)
                target = np.concatenate((force, moment / radius)) / scale
                record = {
                    "snapshot": snapshot.name,
                    "env": int(i_b),
                    "contact": int(i_c),
                    "samples": int(eligible.sum()),
                    "radius_m": radius,
                    "friction": friction,
                    "target": target.tolist(),
                    "saved_force_error": float(saved["fit_force_error"][i_b, i_c]),
                    "saved_moment_error": float(saved["fit_moment_error"][i_b, i_c]),
                    "lp": [],
                    "native": [],
                }
                witnesses = {}
                for sides in [32, 128]:
                    for outer in [False, True]:
                        lp, fitted = cone_lp(relative, normals[eligible], friction, target, sides, outer)
                        record["lp"].append(lp)
                        if fitted is not None:
                            witnesses[f"lp_{sides}_{'outer' if outer else 'inner'}"] = fitted
                for tolerance, evaluations in [(8e-9, 128), (8e-9, 512)]:
                    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, tolerance, evaluations)
                    record["native"].append(
                        {
                            "tolerance": tolerance,
                            "evaluation_budget": evaluations,
                            "status": int(qd_to_numpy(tractions.status, transpose=True).item()),
                            "evaluations": int(qd_to_numpy(tractions.evaluations, transpose=True).item()),
                            "force_error": float(qd_to_numpy(tractions.force_error, transpose=True).item()),
                            "moment_error": float(qd_to_numpy(tractions.moment_error, transpose=True).item()),
                            "dual": qd_to_numpy(tractions.dual, transpose=True).reshape(-1).tolist(),
                        }
                    )
                np.savez_compressed(
                    args.output.parent / f"{snapshot.stem}-env{i_b}-contact{i_c}.npz",
                    positions=positions[eligible],
                    inward_normals=normals[eligible],
                    profile=profile[eligible],
                    relative=relative,
                    target=target,
                    force=force,
                    position=position,
                    anchor=anchor,
                    radius=radius,
                    friction=friction,
                    **witnesses,
                )
                records.append(record)
                print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps({"scope": __doc__, "contacts": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
