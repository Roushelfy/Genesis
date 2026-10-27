"""Compare planar loading proxies against one unchanged Q4 elastic model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from research.rigid_stress.shell_cpu.contacts import Contact, _basis, _shape
from research.rigid_stress.shell_cpu.core import ShellModel

from .checks import run_checks
from .reference import Q4LoadTransfer
from .proxy import TriangularLoadTransfer


@dataclass(frozen=True)
class LoadCase:
    name: str
    category: str
    contacts: tuple[Contact, ...]


def physical_contact(model, face, coordinate, side=1, shear=0.3, radius=None, force=None):
    """Construct a source at the actual offset Q4 material position."""
    shape, derivative_u, derivative_v = _shape(*coordinate)
    nodes = model.quads[face]
    vertices = model.vertices[nodes] + side * 0.5 * model.h * model.directors[nodes]
    point = shape @ vertices
    normal = side * np.cross(derivative_u @ vertices, derivative_v @ vertices)
    normal /= np.linalg.norm(normal)
    if force is None:
        force = -0.05 * normal + shear * 0.05 * _basis(normal)[:, 0]
    return Contact(point, np.array(force), normal, (50.0, 70.0), side, radius, 0.5, int(face))


def make_cases(model):
    """Freeze surface, triangle-seam, material-edge and multiple-source cases."""
    cases = []
    faces = np.linspace(3, len(model.quads) - 4, 8, dtype=int)
    for side in (1, -1):
        for index, face in enumerate(faces):
            coordinate = (0.19, -0.27) if index % 2 else (-0.31, 0.23)
            shear = 0.0 if index % 3 == 0 else 0.3
            contact = physical_contact(model, face, coordinate, side, shear)
            cases.append(LoadCase(f"interior-s{side}-f{face}", "interior-auto", (contact,)))
    face = int(faces[3])
    for side in (1, -1):
        fixed = physical_contact(model, face, (0.0, 0.0), side).force
        for index, parameter in enumerate(np.linspace(-0.65, 0.65, 13)):
            contact = physical_contact(model, face, (float(parameter), 0.07), side, force=fixed)
            cases.append(LoadCase(f"diagonal-s{side}-{index:02}", "triangle-seam-auto", (contact,)))
    edge_map = {}
    for face_idx, nodes in enumerate(model.quads):
        for edge_idx in range(4):
            edge = tuple(sorted((int(nodes[edge_idx]), int(nodes[(edge_idx + 1) % 4]))))
            edge_map.setdefault(edge, []).append((face_idx, edge_idx))
    witness = next(value for value in edge_map.values() if len(value) == 2 and value[0][0] == face)
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for side in (1, -1):
        edge_face, edge_idx = witness[0]
        midpoint = 0.5 * (corners[edge_idx] + corners[(edge_idx + 1) % 4])
        first_normal = physical_contact(model, edge_face, midpoint, side).normal
        other_face, other_edge = witness[1]
        other_midpoint = 0.5 * (corners[other_edge] + corners[(other_edge + 1) % 4])
        second_normal = physical_contact(model, other_face, other_midpoint, side).normal
        shared_normal = first_normal + second_normal
        shared_normal /= np.linalg.norm(shared_normal)
        fixed = -0.05 * shared_normal + 0.005 * _basis(shared_normal)[:, 0]
        for index, signed in enumerate((-0.1, -0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01, 0.1)):
            entry = witness[0] if signed <= 0 else witness[1]
            edge_face, edge_idx = entry
            midpoint = 0.5 * (corners[edge_idx] + corners[(edge_idx + 1) % 4])
            coordinate = (1.0 - abs(signed)) * midpoint
            contact = physical_contact(model, edge_face, coordinate, side, force=fixed)
            cases.append(LoadCase(f"edge-s{side}-{index:02}", "material-edge-auto", (contact,)))
    for side in (1, -1):
        for radius in (0.0005, 0.002):
            for index, coordinate in enumerate(((0.19, -0.27), (0.0, 0.0), (0.95, 0.07))):
                contact = physical_contact(model, face, coordinate, side, radius=radius)
                cases.append(LoadCase(f"finite-s{side}-a{radius:g}-{index}", "finite-radius", (contact,)))
    first = physical_contact(model, int(faces[1]), (0.19, -0.27), shear=0.23)
    second = physical_contact(model, int(faces[6]), (-0.31, 0.23), shear=-0.17)
    third = physical_contact(model, int(faces[4]), (0.12, 0.31), side=-1, shear=0.11)
    cases.append(LoadCase("asymmetric-two", "multiple", (first, second)))
    cases.append(LoadCase("asymmetric-three", "multiple", (first, second, third)))
    cases.append(LoadCase("release", "release", ()))
    return cases


def case_payload(cases):
    rows = []
    for case in cases:
        sources = []
        for contact in case.contacts:
            sources.append(dict(point=contact.point.tolist(), force=contact.force.tolist(),
                                normal=contact.normal.tolist(), curvatures=list(contact.curvatures),
                                side=contact.side, radius=contact.radius, friction=contact.friction,
                                face_hint=contact.face_hint))
        rows.append(dict(name=case.name, category=case.category, contacts=sources))
    return rows


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def assemble_case(transfer, case, order, **options):
    force = np.zeros(transfer.model.ndof)
    records = []
    for contact in case.contacts:
        current, info = transfer.transfer(contact, order=order, **options)
        force += current
        records.append(info)
    return force, records


def tensor_norm(field, weights):
    multiplicity = np.array([1.0, 1.0, 2.0])
    return float(np.sqrt(np.einsum("egc,egc,eg,c->", field, field, weights, multiplicity)))


def field_errors(model, candidate, reference):
    weights = model.gauss_weights
    norm_reference = np.hypot(tensor_norm(reference["local_top"], weights),
                              tensor_norm(reference["local_bottom"], weights))
    norm_difference = np.hypot(tensor_norm(candidate["local_top"] - reference["local_top"], weights),
                               tensor_norm(candidate["local_bottom"] - reference["local_bottom"], weights))
    output = dict(field=float(norm_difference / max(norm_reference, 1e-30)))
    peaks = []
    for fields in (candidate, reference):
        values = []
        for key in ("local_top", "local_bottom"):
            xx, yy, xy = np.moveaxis(fields[key], -1, 0)
            values.append(np.sqrt(np.maximum(0, xx**2 + yy**2 - xx * yy + 3 * xy**2)).max())
        peaks.append(max(values))
    output["peak"] = float(abs(peaks[0] - peaks[1]) / max(peaks[1], 1e-30))
    output["reference_peak_pa"] = float(peaks[1])
    for key in ("N", "M"):
        output[key] = tensor_norm(candidate[key] - reference[key], weights) / max(tensor_norm(reference[key], weights), 1e-30)
    output["Q"] = float(np.sqrt(np.einsum("egc,egc,eg->", candidate["Q"] - reference["Q"],
                                                 candidate["Q"] - reference["Q"], weights)) /
                        max(np.sqrt(np.einsum("egc,egc,eg->", reference["Q"], reference["Q"], weights)), 1e-30))
    return output


def numeric_study(model, cases, reference, proxies, output, order=24, reference_order=48):
    records = []
    reference_forces = {}
    reference_fields = {}
    selected_fields = {}
    for case in cases:
        try:
            upper, upper_info = assemble_case(reference, case, reference_order)
            lower, lower_info = assemble_case(reference, case, order)
            displacement = model.solve(np.column_stack((upper, lower)))
            fields = model.stress(displacement)
            upper_fields = {key: value[..., 0] for key, value in fields.items()}
            lower_fields = {key: value[..., 1] for key, value in fields.items()}
            errors = field_errors(model, lower_fields, upper_fields)
            reference_forces[case.name] = upper
            reference_fields[case.name] = upper_fields
            records.append(dict(case=case.name, category=case.category, method="Q4", valid=True,
                                load_error=float(np.linalg.norm(lower-upper)/max(np.linalg.norm(upper), 1e-30)),
                                **errors, diagnostics=lower_info))
        except ValueError as error:
            records.append(dict(case=case.name, category=case.category, method="Q4", valid=False,
                                error=str(error)))
            continue
        for refinement, transfer in proxies.items():
            for variant, options in ((f"tri-k{refinement}", dict(analytic=True, anchor_mode="fixed")),):
                try:
                    force, info = assemble_case(transfer, case, order, **options)
                    displacement = model.solve(force)
                    fields = model.stress(displacement)
                    errors = field_errors(model, fields, upper_fields)
                    records.append(dict(case=case.name, category=case.category, method=variant, valid=True,
                                        load_error=float(np.linalg.norm(force-upper)/max(np.linalg.norm(upper), 1e-30)),
                                        **errors, diagnostics=info))
                    if case.name == "asymmetric-two":
                        selected_fields[variant] = np.stack((fields["local_top"], fields["local_bottom"]))
                except ValueError as error:
                    records.append(dict(case=case.name, category=case.category, method=variant, valid=False,
                                        error=str(error)))
        if case.name == "asymmetric-two":
            selected_fields["Q4"] = np.stack((upper_fields["local_top"], upper_fields["local_bottom"]))
        if len(reference_forces) % 10 == 0:
            print(f"numerical cases completed: {len(reference_forces)}/{len(cases)}", flush=True)
    write_json(output / "numeric.json", records)
    summaries = []
    for method in sorted({row["method"] for row in records}):
        rows = [row for row in records if row["method"] == method]
        valid = [row for row in rows if row["valid"] and row["category"] != "release"]
        summary = dict(method=method, cases=len(rows), rejected=sum(not row["valid"] for row in rows),
                       loaded_valid=len(valid))
        if valid:
            summary.update(mean_field=float(np.mean([row["field"] for row in valid])),
                           **{f"max_{key}": max(row[key] for row in valid) for key in ("field", "peak", "N", "M", "Q", "load_error")})
            summary["worst_field_case"] = max(valid, key=lambda row: row["field"])["case"]
            summary["declared_budget_passed"] = (summary["rejected"] == 0 and summary["mean_field"] <= 0.05
                and summary["max_field"] <= 0.1 and summary["max_peak"] <= 0.05
                and max(summary["max_N"], summary["max_M"], summary["max_Q"]) <= 0.1)
        summaries.append(summary)
    write_json(output / "summary.json", summaries)
    columns = ["case", "category", "method", "valid", "field", "peak", "N", "M", "Q", "load_error", "error"]
    with (output / "errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / "selected_fields.npz", vertices=model.vertices, quads=model.quads,
                        **{key.replace("-", "_"): value for key, value in selected_fields.items()})
    return summaries


def benchmark(model, cases, reference, proxies, output, order=24, repeats=7):
    selected = []
    for side in (1, -1):
        selected += [case for case in cases if case.category == "interior-auto" and case.contacts[0].side == side][:3]
        selected += [case for case in cases if case.category == "triangle-seam-auto" and case.contacts[0].side == side][3:5]
        selected += [case for case in cases if case.category == "material-edge-auto" and case.contacts[0].side == side][3:5]
    selected += [next(case for case in cases if case.name == "asymmetric-two"), cases[-1]]
    methods = [("Q4", reference, {})] + [(f"tri-k{k}", transfer, dict(analytic=True, anchor_mode="fixed"))
                                            for k, transfer in proxies.items()]
    rows = []
    for repeat in range(-2, repeats):
        order_methods = methods if repeat % 2 == 0 else list(reversed(methods))
        for method, transfer, options in order_methods:
            start = time.perf_counter()
            columns = []
            diagnostics = []
            errors = []
            for case in selected:
                try:
                    force, info = assemble_case(transfer, case, order, **options)
                except ValueError as error:
                    errors.append(dict(case=case.name, error=str(error)))
                    force, info = np.zeros(model.ndof), []
                columns.append(force)
                diagnostics.extend(info)
            assembled = time.perf_counter()
            forces = np.column_stack(columns)
            displacements = model.solve(forces)
            solved = time.perf_counter()
            fields = model.stress(displacements)
            recovered = time.perf_counter()
            checksum = float(np.linalg.norm(fields["local_top"]) + np.linalg.norm(fields["local_bottom"]))
            if repeat >= 0:
                rows.append(dict(repeat=repeat, method=method, batch=len(selected), order=order,
                    load_seconds=assembled-start, solve_seconds=solved-assembled,
                    stress_seconds=recovered-solved, total_seconds=recovered-start,
                    valid=len(selected)-len(errors), rejected=errors, checksum=checksum,
                    analytic_contacts=sum(info.get("analytic_count", 0) for info in diagnostics),
                    samples=sum(info.get("n_samples", info.get("samples", 0)) for info in diagnostics),
                    stage_seconds={key: sum(info.get("stages", {}).get(key, info.get(key+"_seconds", 0)) for info in diagnostics)
                                   for key in ("query", "integration", "fit", "scatter")}))
    write_json(output / "timings.json", rows)
    summaries = []
    for method, _, _ in methods:
        chosen = [row for row in rows if row["method"] == method]
        ratio = []
        load_ratio = []
        for row in chosen:
            baseline = next(item for item in rows if item["method"] == "Q4" and item["repeat"] == row["repeat"])
            ratio.append(baseline["total_seconds"] / row["total_seconds"])
            load_ratio.append(baseline["load_seconds"] / row["load_seconds"])
        summaries.append(dict(method=method, batch=len(selected), repeats=repeats,
            total_ms=float(1000*np.median([row["total_seconds"] for row in chosen])),
            load_ms=float(1000*np.median([row["load_seconds"] for row in chosen])),
            solve_ms=float(1000*np.median([row["solve_seconds"] for row in chosen])),
            stress_ms=float(1000*np.median([row["stress_seconds"] for row in chosen])),
            paired_speedup=float(np.median(ratio)), paired_load_speedup=float(np.median(load_ratio)),
            paired_speedup_min=float(min(ratio)), paired_speedup_max=float(max(ratio)),
            faster_repeats=sum(value > 1 for value in ratio),
            total_ms_min=float(1000*min(row["total_seconds"] for row in chosen)),
            total_ms_max=float(1000*max(row["total_seconds"] for row in chosen)),
            rejected=sum(len(row["rejected"]) for row in chosen),
            analytic_contacts=int(np.median([row["analytic_contacts"] for row in chosen])),
            samples=int(np.median([row["samples"] for row in chosen]))))
    write_json(output / "timing_summary.json", summaries)
    write_json(output / "benchmark_cases.json", [case.name for case in selected])
    return summaries


def run(output, phase="all"):
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    asset = repository / "examples/rigid/assets/hollow_egg/level1/shell.npz"
    started = time.perf_counter()
    with np.load(asset) as data:
        model = ShellModel(data["vertices"], data["quads"], E=1e10, nu=0.3, h=0.0005,
                           density=2000, directors=data["directors"], gauge="translation")
    model.factorize()
    reference = Q4LoadTransfer(model)
    proxies = {}
    preparation = []
    for refinement in (1, 2, 4, 8, 16):
        before = time.perf_counter()
        proxies[refinement] = TriangularLoadTransfer(model, subdivisions=refinement)
        preparation.append(dict(refinement=refinement, seconds=time.perf_counter()-before,
                                triangles_per_side=len(model.quads)*2*refinement**2,
                                native_array_bytes=proxies[refinement].array_bytes))
    cases = make_cases(model)
    fixtures = case_payload(cases)
    write_json(output / "fixtures.json", fixtures)
    protocol = dict(source_revision="cf40684b1c170aacadcab3990e9d57e69882e4cc",
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        fixtures_sha256=hashlib.sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest(),
        before_results=True, cpu_model_dofs=model.ndof, q4_elements=len(model.quads),
        primary_order=24, reference_order=48, proxy_refinements=[1, 2, 4, 8, 16],
        unchanged="Q4 K, mass, gauge, inertia relief, stress sampling, source center/force/radius/side",
        pressure_law="3D surface-ball Hertz density plus constant-direction nonnegative affine fit",
        budgets=dict(mean_field=0.05, field=0.1, peak=0.05, N=0.1, M=0.1, Q=0.1),
        strict_secondary_target=dict(field=0.01, peak=0.01),
        scope="CPU fixed-contact load-adapter study, not GPU, Franka dynamics, independent contact physics or convergence",
        preparation=preparation, total_preparation_seconds=time.perf_counter()-started)
    environment = dict(platform=platform.platform(), processor=platform.processor(),
        cpu_model=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")), "unknown"),
        python=platform.python_version(), numpy=np.__version__, threadpools=threadpool_info())
    if phase == "benchmark":
        write_json(output / "benchmark_protocol.json", protocol)
        write_json(output / "benchmark_environment.json", environment)
    else:
        write_json(output / "protocol.json", protocol)
        write_json(output / "environment.json", environment)
    sources_before = {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in Path(__file__).parent.glob("*.py")}
    if phase in ("all", "numeric"):
        print(json.dumps(numeric_study(model, cases, reference, proxies, output), indent=2), flush=True)
        checks = run_checks(model, output)
        print(json.dumps(dict(independent_checks=checks["checks"], passed=checks["passed"])), flush=True)
    if phase in ("all", "benchmark"):
        print(json.dumps(benchmark(model, cases, reference, proxies, output), indent=2), flush=True)
    sources_after = {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in Path(__file__).parent.glob("*.py")}
    write_json(output / f"source_{phase}.json", dict(before=sources_before, after=sources_after,
        source_unchanged=sources_before == sources_after))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "numeric", "benchmark"), default="all")
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        run(arguments.output, arguments.phase)
