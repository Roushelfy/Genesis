"""Compare radial memory with independent incremental vector surface contact."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_history_cone import ConeFrictionHistory
from .hertz_tangential import cerruti_stress, elliptic_tangential_stress
from .study_hertz_nearfield import stress_measures
from .study_hertz_residual import elliptic_stress
from .tangential_contact_reference import TangentialContactReference


def probes(radius):
    ring, angle, depth = np.meshgrid([0.0, 0.5, 0.95, 1.2], np.arange(8) * np.pi / 4, [0.15, 0.5, 1.0], indexing="ij")
    return radius * np.column_stack(
        [(ring * np.cos(angle)).reshape(-1), (ring * np.sin(angle)).reshape(-1), depth.reshape(-1)]
    )


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def compare(reference, candidate, normal, friction, poisson, output, label, targets, analytical_ratio=None):
    radius = normal ** (1 / 3)
    peak = 3 * normal / (2 * np.pi * radius**2)
    points = probes(radius)
    source, force = reference.sources(order=8)
    tensor = cerruti_stress(points, source, force, poisson)
    high_source, high_force = reference.sources(order=16)
    high = cerruti_stress(points, high_source, high_force, poisson)
    quadrature_change = relative(tensor, high)
    write_json(output / f"{label}-quadrature.json", {"relative_change": quadrature_change, "target": 1e-4})
    assert quadrature_change < 1e-4, quadrature_change
    actual = candidate.stress(points, np.array([radius, radius]), peak, friction, poisson=poisson, order=48)
    normal_tensor = elliptic_stress(points, np.array([radius, radius]), peak=peak, poisson=poisson, order=64)
    tensile, vm = stress_measures(normal_tensor + high)
    actual_tensile, actual_vm = stress_measures(normal_tensor + actual)
    analytical = np.zeros_like(high)
    if analytical_ratio is not None:
        stick = radius * (1 - analytical_ratio) ** (1 / 3)
        analytical = friction * (
            elliptic_tangential_stress(points, np.array([radius, radius]), np.array([1.0, 0.0]), peak, poisson)
            - elliptic_tangential_stress(
                points, np.array([stick, stick]), np.array([1.0, 0.0]), peak * stick / radius, poisson
            )
        )
    np.savez_compressed(
        output / f"{label}.npz",
        points=points,
        normal_tensor=normal_tensor,
        reference_tangent=high,
        reference_lower_rule_tangent=tensor,
        candidate_tangent=actual,
        analytical_tangent=analytical,
        cell_centers=reference.centers,
        reference_cell_force=reference.force,
        candidate_line=candidate.line,
        candidate_coefficients=candidate.coefficients,
    )
    tensor_error = relative(actual, high)
    combined_error = relative(actual + normal_tensor, high + normal_tensor)
    tensile_error = float(abs(actual_tensile.max() - tensile.max()) / max(abs(tensile.max()), 1e-30))
    vm_error = float(abs(actual_vm.max() - vm.max()) / max(abs(vm.max()), 1e-30))
    return {
        "label": label,
        "tangent_tensor_relative": tensor_error,
        "combined_tensor_relative": combined_error,
        "combined_sampled_tensile_peak_relative": tensile_error,
        "combined_sampled_vm_peak_relative": vm_error,
        "reference_quadrature_relative": quadrature_change,
        "analytical_collinear_tensor_relative": relative(high, analytical) if analytical_ratio is not None else None,
        "candidate_engineering_targets_met_on_this_bank": max(tensor_error, combined_error) < targets["tensor"]
        and max(tensile_error, vm_error) < targets["peak"],
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--grid", type=int, default=17)
    cli.add_argument("--poisson", type=float, default=0.3)
    cli.add_argument("--subdivisions", type=int, default=8)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"tensor": 0.05, "peak": 0.10, "reference_spatial_or_time_change": 0.02}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "grid": args.grid,
            "poisson": args.poisson,
            "subdivisions": args.subdivisions,
            "targets": targets,
            "scope": "Independent two-dimensional vector surface discretization of a three-dimensional similar-material circular Hertz contact. Normal pressure is supplied; tangential traction and its history are solved from Cerruti compliance and vector Coulomb disks. Candidate radial history is compared at positive-depth tensors and sampled combined peaks. Spatial/time qualification requires the separate matched inspection. This is an offline discriminating friction reference, not a runtime pressure solver, shell/friction qualification or a full-volume/surface global-peak reference.",
        },
    )
    friction = 0.4
    waypoints = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.65, 0.0],
            [0.65, 0.2, 0.3],
            [1.2, -0.35, 0.25],
            [0.85, -0.25, -0.3],
            [1.05, 0.4, -0.2],
            [1.0, 0.0, 0.0],
        ]
    )
    begun = perf_counter()
    reference = TangentialContactReference(args.grid, waypoints[:, 0].max() ** (1 / 3), args.poisson)
    setup_seconds = perf_counter() - begun
    rows, history, forces, capacities, lines = [], [], [], [], []
    for ratio in [0.25, 0.65]:
        reference.force.fill(0)
        candidate = ConeFrictionHistory(16)
        force = np.array([friction * ratio, 0.0])
        state = reference.update(1.0, 1.0, friction, force)
        candidate.update(1.0, 1.0, friction, force)
        label = f"monotonic-{ratio}"
        rows.append(compare(reference, candidate, 1.0, friction, args.poisson, args.output, label, targets, ratio))
        record = asdict(state)
        for key in ["force", "capacity", "rigid_increment"]:
            record.pop(key)
        write_json(args.output / f"{label}-numerics.json", record)
    reference.force.fill(0)
    candidate = ConeFrictionHistory(16)
    for segment, (first, second) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        for part in range(1, args.subdivisions + 1):
            values = first + (second - first) * part / args.subdivisions
            normal, force = values[0], friction * values[1:]
            radius = normal ** (1 / 3)
            state = reference.update(radius, normal, friction, force)
            candidate.update(radius, normal, friction, force)
            record = asdict(state)
            for key in ["force", "capacity", "rigid_increment"]:
                record.pop(key)
            history.append({"segment": segment, "part": part, "normal": normal, "force": force.tolist(), **record})
            forces.append(state.force)
            capacities.append(state.capacity)
            lines.append(candidate.line.copy())
            if part == args.subdivisions:
                rows.append(
                    compare(
                        reference,
                        candidate,
                        normal,
                        friction,
                        args.poisson,
                        args.output,
                        f"waypoint-{segment + 1}",
                        targets,
                    )
                )
            write_json(args.output / "progress.json", {"rows": rows, "history": history})
    np.savez_compressed(
        args.output / "history.npz",
        reference_cell_force=np.stack(forces),
        reference_cell_capacity=np.stack(capacities),
        candidate_line=np.stack(lines),
        cell_centers=reference.centers,
        waypoints=waypoints,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "numerical_reference_execution_passed": True,
            "independent_spatial_temporal_reference_qualified": False,
            "candidate_engineering_targets_met_on_this_bank": all(
                row["candidate_engineering_targets_met_on_this_bank"] for row in rows
            ),
            "setup_seconds": setup_seconds,
            "cells": len(reference.centers),
            "poisson": args.poisson,
            "grid": args.grid,
            "subdivisions": args.subdivisions,
            "targets": targets,
            "rows": rows,
            "history": history,
            "outputs": [file_id(path) for path in sorted(args.output.glob("*.npz"))],
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
