"""Force-line full-shell refinement, minimum footprints, real contact snapshots and wall-layer checks."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .convergence import prescribed_contacts
from .cpu import ContactBatch, EggConfig, EggRecoveryCPU
from .mechanics import SurfaceGeometry
from .wrench import FinitePatchMapper


@dataclass(frozen=True)
class PhysicalCases:
    contacts: ContactBatch
    omega_rad_s: np.ndarray
    gravity_m_s2: np.ndarray
    labels: tuple[str, ...]


def cases_from_replay(path: Path) -> PhysicalCases:
    source = np.load(path)
    frames, environments = source["omega_rad_s"].shape[:2]
    capacity = int(source["ids"][:, 1].max(initial=0)) + 1
    companion = path.with_name(path.name.removesuffix(".contacts.npz") + ".json")
    phases = np.array([row["phase"] for row in json.loads(companion.read_text())["records"]])
    phases = np.repeat(phases, frames // len(phases), axis=0)
    if phases.shape != (frames, environments):
        raise ValueError("Replay requires its matching every-step phase records")
    candidates = []
    # Choose complete actual wrenches at each trajectory stage by force magnitude, never by a stress hotspot.
    for lo, hi, label in (
        (0.0, 0.15, "approach/table"),
        (0.15, 0.3, "clamp"),
        (0.3, 0.5, "lift"),
        (0.5, 0.6, "hold"),
        (0.7, 0.76, "weak/sliding"),
        (0.9, 1.0, "release/table"),
    ):
        best = (0.0, 0, 0)
        for frame in range(frames):
            first, last = source["offsets"][frame : frame + 2]
            ids, forces = source["ids"][first:last], source["force_n"][first:last]
            norm = np.linalg.norm(forces, axis=1)
            total = np.bincount(ids[:, 0], weights=norm, minlength=environments)
            total = np.where((phases[frame] >= lo) & (phases[frame] < hi), total, -1)
            environment = int(np.argmax(total))
            if total[environment] > best[0]:
                best = (float(total[environment]), frame, environment)
        candidates.append((best[1], best[2], label))
    count = len(candidates)
    position, force, normal = (np.zeros((count, capacity, 3)) for _ in range(3))
    radius, friction = np.zeros((count, capacity)), np.zeros((count, capacity))
    valid = np.zeros((count, capacity), dtype=bool)
    omega, gravity = np.zeros((count, 3)), np.zeros((count, 3))
    labels = []
    for case, (frame, environment, label) in enumerate(candidates):
        first, last = source["offsets"][frame : frame + 2]
        selected = np.flatnonzero(source["ids"][first:last, 0] == environment) + first
        slots = source["ids"][selected, 1]
        position[case, slots], force[case, slots] = source["position_m"][selected], source["force_n"][selected]
        normal[case, slots], friction[case, slots] = source["inward_normal"][selected], source["friction"][selected]
        radius[case, slots], valid[case, slots] = source["radius_m"][selected], True
        omega[case], gravity[case] = (
            source["omega_rad_s"][frame, environment],
            source["gravity_m_s2"][frame, environment],
        )
        labels.append(f"{label}: frame {frame}, environment {environment}")
    return PhysicalCases(
        ContactBatch(position, force, radius, valid, friction, normal, float(source["source_epsilon"])),
        omega,
        gravity,
        tuple(labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--wall-layers", type=int, nargs="+")
    parser.add_argument("--quadratures", type=int, nargs="+", default=[6, 10, 16])
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    if len(args.levels) < 3 and (args.wall_layers is None or len(args.wall_layers) < 3):
        raise ValueError("At least three surface refinements or three wall-layer counts are required")
    live = cases_from_replay(args.replay)
    prescribed = prescribed_contacts(12)
    radius = prescribed.radius_m.copy()
    # Declared radius range: 6 mm * [0.8, 1.2] * [0.85, 1.15]. Include the minimum explicitly.
    radius[:4] = 0.00408
    prescribed = replace(prescribed, radius_m=radius)
    sets = (
        PhysicalCases(
            prescribed,
            np.zeros((12, 3)),
            np.zeros((12, 3)),
            tuple(f"asymmetric finite pad {case}" for case in range(12)),
        ),
        live,
    )
    fingerprint = hashlib.sha256()
    for suite in sets:
        for array in (
            suite.contacts.position_m,
            suite.contacts.force_n,
            suite.contacts.radius_m,
            suite.contacts.inward_normal,
            suite.contacts.friction,
            suite.contacts.valid,
            suite.omega_rad_s,
            suite.gravity_m_s2,
        ):
            fingerprint.update(array.tobytes())
    fingerprint = fingerprint.hexdigest()
    rows, configurations = [], []
    if args.reuse is not None:
        previous = json.loads(args.reuse.read_text())
        if previous["input_sha256"] != fingerprint:
            raise ValueError("Reused convergence results require exactly identical physical loads")
        rows, configurations = previous["results"], previous["configurations"]
    specs = (
        [(level, 2) for level in args.levels]
        if args.wall_layers is None
        else [(args.levels[-1], layers) for layers in args.wall_layers]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "input_sha256": fingerprint,
        "replay": str(args.replay),
        "load_law": "Force-line exterior anchor, unchanged wrench, nonnegative compact Gaussian pad pressure",
        "radius_min_m": 0.00408,
        "sets": [suite.labels for suite in sets],
        "results": rows,
        "configurations": configurations,
        "criterion": 0.02,
        "wall_layers": args.wall_layers,
    }
    with threadpool_limits(limits=1):
        for level, layers in specs:
            if sum(row["level"] == level and row["layers"] == layers for row in rows) == 18 * len(args.quadratures):
                continue
            start = perf_counter()
            config = EggConfig(level=level, layers=layers, ordering="column-nd", factor_backend="cholmod")
            model = EggRecoveryCPU(len(sets[0].labels), config, history=0, direct=True, anchor_to_surface=True)
            configurations.append(
                {
                    "config": asdict(config),
                    "dofs": model.fem.ndof,
                    "tetrahedra": model.fem.ne,
                    "build_seconds": perf_counter() - start,
                    "factor_seconds": model.fem.factor_s,
                }
            )
            for quadrature in args.quadratures:
                model.surface = SurfaceGeometry(model.fem, quadrature)
                model.mapper = FinitePatchMapper(model.surface, anchor_to_surface=True)
                for suite_index, suite in enumerate(sets):
                    # Padding is only batch bookkeeping; every active source wrench is preserved.
                    count = len(suite.labels)
                    contacts = suite.contacts
                    padding = model.environments - count
                    padded = ContactBatch(
                        np.pad(contacts.position_m, ((0, padding), (0, 0), (0, 0))),
                        np.pad(contacts.force_n, ((0, padding), (0, 0), (0, 0))),
                        np.pad(contacts.radius_m, ((0, padding), (0, 0))),
                        np.pad(contacts.valid, ((0, padding), (0, 0))),
                        np.pad(contacts.friction, ((0, padding), (0, 0))),
                        np.pad(contacts.inward_normal, ((0, padding), (0, 0), (0, 0))),
                        contacts.source_epsilon,
                    )
                    loads = model.map_contacts(padded)
                    gravity = np.pad(suite.gravity_m_s2, ((0, padding), (0, 0)))
                    omega = np.pad(suite.omega_rad_s, ((0, padding), (0, 0)))
                    external = loads.nodal_force_n + model.fem.mr[:, :3] @ gravity.T
                    rhs = model.compatible_rhs(external, omega)
                    displacement = np.zeros_like(rhs)
                    displacement[model.fem.free] = model.fem.factor.solve(rhs[model.fem.free])
                    residual = np.linalg.norm(model.fem.k @ displacement - rhs, axis=0)
                    norm = np.linalg.norm(rhs, axis=0)
                    assert np.all(residual <= np.maximum(1e-11, 1e-6 * norm))
                    for case, label in enumerate(suite.labels):
                        peak, element, corner = model.peak(displacement[:, case])
                        rows.append(
                            {
                                "level": level,
                                "layers": layers,
                                "quadrature": quadrature,
                                "suite": suite_index,
                                "case": case,
                                "label": label,
                                "peak_pa": peak,
                                "hot_position_m": model.fem.base_xyz[model.fem.base_tets[element, corner]].tolist(),
                                "full_residual_absolute_n": float(residual[case]),
                                "full_residual_relative": float(residual[case] / max(norm[case], 1e-11)),
                                "moment_error_nm": float(
                                    np.linalg.norm(loads.resultant_moment_nm[case] - loads.input_moment_nm[case])
                                ),
                            }
                        )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"level": level, "layers": layers, "quadrature": quadrature}), flush=True)
            del model
    lookup = {
        (row["level"], row["layers"], row["quadrature"], row["suite"], row["case"]): row["peak_pa"] for row in rows
    }
    comparisons = []
    for suite, physical in enumerate(sets):
        for case in range(len(physical.labels)):
            coarse = lookup[(*specs[-2], args.quadratures[-1], suite, case)]
            fine = lookup[(*specs[-1], args.quadratures[-1], suite, case)]
            lower_quad = lookup[(*specs[-1], args.quadratures[-2], suite, case)]
            comparisons.append(
                {
                    "suite": suite,
                    "case": case,
                    "final_mesh_relative_change": abs(fine - coarse) / max(fine, 1),
                    "final_quadrature_relative_change": abs(fine - lower_quad) / max(fine, 1),
                }
            )
    report.update(
        status="complete",
        comparisons=comparisons,
        accepted=all(
            max(row["final_mesh_relative_change"], row["final_quadrature_relative_change"]) <= 0.02
            for row in comparisons
        ),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"accepted": report["accepted"], "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
