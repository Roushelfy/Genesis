"""Measure full-shell mesh and quadrature convergence with fixed physical pad wrenches."""

import argparse
import csv
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import ContactBatch, EggConfig, EggRecoveryCPU, reference
from .wrench import FinitePatchMapper


def prescribed_contacts(environments: int) -> ContactBatch:
    rng = np.random.default_rng(202610091)
    unit = rng.normal(size=(environments, 3, 3))
    unit /= np.linalg.norm(unit, axis=2)[..., None]
    scale = 1 - 0.18 * unit[..., 2]
    position = unit * np.stack((0.022 * scale, 0.022 * scale, np.full_like(scale, 0.030)), axis=-1)
    gradient = np.stack(
        (
            2 * unit[..., 0] / (0.022 * scale),
            2 * unit[..., 1] / (0.022 * scale),
            2 / 0.030 * (unit[..., 2] + 0.18 * (unit[..., 0] ** 2 + unit[..., 1] ** 2) / scale),
        ),
        axis=-1,
    )
    normal = -gradient / np.linalg.norm(gradient, axis=2)[..., None]
    tangent = np.cross(normal, np.roll(normal, 1, axis=2))
    tangent /= np.linalg.norm(tangent, axis=2)[..., None]
    compression = rng.uniform(0.5, 3, (environments, 3))
    friction = rng.uniform(0.4, 1, (environments, 3))
    force = compression[..., None] * (
        normal + rng.uniform(-0.7, 0.7, (environments, 3, 1)) * friction[..., None] * tangent
    )
    radius = rng.uniform(0.006, 0.009, (environments, 3))
    valid = np.arange(3)[None] < (1 + np.arange(environments) % 3)[:, None]
    return ContactBatch(position, force, radius, valid, friction, normal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--quadratures", type=int, nargs="+", default=[6, 10, 16])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--criterion", type=float, default=0.02)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.levels) < 3 or len(args.quadratures) < 2:
        raise ValueError("At least three full mesh levels and two quadrature refinements are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contacts = prescribed_contacts(args.cases)
    rows = []
    configurations = []
    if args.reuse is not None:
        previous = json.loads(args.reuse.read_text())
        for name, values in (
            ("position_m", contacts.position_m),
            ("force_n", contacts.force_n),
            ("inward_normal", contacts.inward_normal),
            ("friction", contacts.friction),
            ("radius_m", contacts.radius_m),
            ("valid", contacts.valid),
        ):
            np.testing.assert_array_equal(values, previous["contacts"][name])
        configurations = [
            item
            for item in previous["configurations"]
            if item["level"] in args.levels and item["layers"] == args.layers
        ]
        cached_levels = {item["level"] for item in configurations}
        rows = [
            item
            for item in previous["results"]
            if item["level"] in cached_levels and item["quadrature"] in args.quadratures
        ]
    report = {
        "status": "running",
        "physical_law": "Compact Gaussian, nonnegative pressure, constant pad traction ratio, exact point wrench",
        "contacts": {
            "position_m": contacts.position_m.tolist(),
            "force_n": contacts.force_n.tolist(),
            "inward_normal": contacts.inward_normal.tolist(),
            "friction": contacts.friction.tolist(),
            "radius_m": contacts.radius_m.tolist(),
            "valid": contacts.valid.tolist(),
        },
        "configurations": configurations,
        "results": rows,
    }
    with threadpool_limits(limits=1):
        for level in args.levels:
            if sum(item["level"] == level for item in rows) == len(args.quadratures) * args.cases:
                continue
            start = perf_counter()
            model = EggRecoveryCPU(
                args.cases,
                EggConfig(level=level, layers=args.layers, surface_quadrature=min(args.quadratures)),
                history=0,
                direct=True,
            )
            configurations.append(
                {
                    "level": level,
                    "layers": args.layers,
                    "dofs": model.fem.ndof,
                    "tetrahedra": model.fem.ne,
                    "factor_nnz": model.fem.factor.L.nnz + model.fem.factor.U.nnz,
                    "build_seconds": perf_counter() - start,
                }
            )
            for quadrature in args.quadratures:
                model.surface = reference.SurfaceGeometry(model.fem, quadrature)
                model.mapper = FinitePatchMapper(model.surface)
                loads = model.map_contacts(contacts)
                rhs = model.compatible_rhs(loads.nodal_force_n)
                recovered = model.recover(rhs)
                for i_case in range(args.cases):
                    peak, element, corner = model.peak(recovered.displacement_m[:, i_case])
                    rows.append(
                        {
                            "level": level,
                            "quadrature": quadrature,
                            "case": i_case,
                            "peak_pa": float(peak),
                            "hot_position_m": model.fem.base_xyz[model.fem.base_tets[element, corner]].tolist(),
                            "full_residual_relative": float(recovered.relative_residual[i_case]),
                            "force_error_n": float(
                                np.linalg.norm(
                                    loads.resultant_force_n[i_case]
                                    - contacts.force_n[i_case, contacts.valid[i_case]].sum(axis=0)
                                )
                            ),
                            "moment_error_nm": float(
                                np.linalg.norm(loads.resultant_moment_nm[i_case] - loads.input_moment_nm[i_case])
                            ),
                        }
                    )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {"level": level, "quadrature": quadrature, "maximum_peak_pa": float(recovered.peak_pa.max())}
                    ),
                    flush=True,
                )
            del model
            gc.collect()
    peak_lookup = {(row["level"], row["quadrature"], row["case"]): row["peak_pa"] for row in rows}
    comparisons = []
    for i_case in range(args.cases):
        fine = peak_lookup[args.levels[-1], args.quadratures[-1], i_case]
        previous_mesh = peak_lookup[args.levels[-2], args.quadratures[-1], i_case]
        previous_quadrature = peak_lookup[args.levels[-1], args.quadratures[-2], i_case]
        comparisons.append(
            {
                "case": i_case,
                "final_mesh_relative_change": abs(fine - previous_mesh) / max(fine, 1),
                "final_quadrature_relative_change": abs(fine - previous_quadrature) / max(fine, 1),
                "peak_sequence_pa": [peak_lookup[level, args.quadratures[-1], i_case] for level in args.levels],
            }
        )
    report.update(
        status="complete",
        criterion=args.criterion,
        accepted=all(
            max(row["final_mesh_relative_change"], row["final_quadrature_relative_change"]) <= args.criterion
            for row in comparisons
        ),
        comparisons=comparisons,
        applicability="Synthetic asymmetric finite pad suite. Live replays and wall-layer convergence require separate checks",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"accepted": report["accepted"], "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
