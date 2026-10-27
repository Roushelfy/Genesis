"""Freeze pinned planar-loading CPU inputs and independent numerical oracles.

Scientific work requires Slurm rtx-mid, image 1_26. The outer process copies the
primary Git bytes; a fresh interpreter imports only that frozen reference tree.
Native preparation and timing may consume inputs/, while oracles/ is audit-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

PRIMARY = "797203553f246ca155bbef909c1bb33e7cd7b90c"
SCHEMA = "genesis.planar-loading-reference.v1"
REFERENCE_DIRECTORY = "research/rigid_stress/triangular_load_cpu"
ASSET = "examples/rigid/assets/hollow_egg/level1/shell.npz"
REFERENCE_FILES = (
    "research/rigid_stress/__init__.py",
    "research/rigid_stress/shell_cpu/__init__.py",
    "research/rigid_stress/shell_cpu/core.py",
    "research/rigid_stress/shell_cpu/contacts.py",
    "research/rigid_stress/wrench.py",
    *(f"{REFERENCE_DIRECTORY}/{name}.py" for name in ("proxy", "reference", "study", "checks", "geometry_controls")),
    ASSET,
)
PUBLISHED_FILES = ("protocol.json", "fixtures.json", "summary.json", "benchmark_cases.json", "checks.json")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_api():
    import numpy as np
    import scipy
    from scipy import sparse
    from threadpoolctl import threadpool_limits

    from research.rigid_stress.shell_cpu.contacts import Contact
    from research.rigid_stress.shell_cpu.core import ShellModel
    from research.rigid_stress.triangular_load_cpu.checks import SurfaceGeometry, _affine_model, _shape, run_checks
    from research.rigid_stress.triangular_load_cpu.geometry_controls import torus_model
    from research.rigid_stress.triangular_load_cpu.proxy import TriangularLoadTransfer
    from research.rigid_stress.triangular_load_cpu.reference import Q4LoadTransfer
    from research.rigid_stress.triangular_load_cpu.study import (
        LoadCase,
        case_payload,
        field_errors,
        make_cases,
        physical_contact,
    )

    return SimpleNamespace(**locals())


class Writer:
    def __init__(self, output, api):
        self.output, self.api, self.inventory = output, api, {}

    def path(self, relative):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def register(self, path, arrays=None):
        record = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if arrays is not None:
            record["arrays"] = {
                key: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256_c_order": hashlib.sha256(self.api.np.ascontiguousarray(value).tobytes()).hexdigest(),
                }
                for key, value in arrays.items()
            }
        self.inventory[str(path.relative_to(self.output))] = record

    def arrays(self, relative, **arrays):
        values = {key: self.api.np.asarray(value) for key, value in arrays.items()}
        if any(value.dtype.hasobject for value in values.values()):
            raise TypeError("Frozen numerical arrays require pickle-free scalar dtypes.")
        path = self.path(relative)
        self.api.np.savez_compressed(path, **values)
        self.register(path, values)

    def matrix(self, relative, matrix):
        path = self.path(relative)
        self.api.sparse.save_npz(path, matrix.tocsr(), compressed=True)
        self.register(path)

    def json(self, relative, value):
        path = self.path(relative)
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self.register(path)


def _cases_from_payload(api, payload):
    cases = []
    for row in payload:
        contacts = []
        for source in row["contacts"]:
            contacts.append(
                api.Contact(
                    api.np.array(source["point"]),
                    api.np.array(source["force"]),
                    api.np.array(source["normal"]),
                    tuple(source["curvatures"]),
                    source["side"],
                    source["radius"],
                    source["friction"],
                    source["face_hint"],
                )
            )
        cases.append(api.LoadCase(row["name"], row["category"], tuple(contacts)))
    return cases


def _export_inputs(writer, bank, model, cases):
    np = writer.api.np
    writer.arrays(
        f"inputs/{bank}/model.npz",
        vertices=model.vertices,
        quads=model.quads,
        directors=model.directors,
        young=model.E,
        poisson=model.nu,
        thickness=model.h,
        density=model.density,
        drilling_factor=model.drilling_factor,
    )
    sources, environments, slots, offsets = [], [], [], [0]
    for environment, case in enumerate(cases):
        for slot, source in enumerate(case.contacts):
            sources.append(source)
            environments.append(environment)
            slots.append(slot)
        offsets.append(len(sources))
    writer.arrays(
        f"inputs/{bank}/sources.npz",
        point=np.array([source.point for source in sources]),
        force=np.array([source.force for source in sources]),
        normal=np.array([source.normal for source in sources]),
        curvatures=np.array([source.curvatures for source in sources]),
        side=np.array([source.side for source in sources]),
        radius=np.array([source.radius if source.radius is not None else 0.0 for source in sources]),
        radius_given=np.array([source.radius is not None for source in sources]),
        friction=np.array([source.friction if source.friction is not None else 1.0 for source in sources]),
        face_hint=np.array([source.face_hint if source.face_hint is not None else -1 for source in sources]),
        environment=np.array(environments, dtype=np.int64),
        slot=np.array(slots, dtype=np.int64),
        case_offsets=np.array(offsets, dtype=np.int64),
    )
    writer.json(f"inputs/{bank}/cases.json", writer.api.case_payload(cases))
    return sources, np.array(environments, dtype=np.int64)


def _export_operators(writer, bank, model):
    prefix = f"oracles/{bank}/operators"
    factor = model.factorize()
    writer.matrix(f"{prefix}/K.npz", model.K)
    writer.matrix(f"{prefix}/K_free.npz", model.K[factor.free_dofs][:, factor.free_dofs])
    writer.matrix(f"{prefix}/S.npz", model.stress_matrix())
    writer.arrays(
        f"{prefix}/numerics.npz",
        gauge_dofs=model.gauge_dofs,
        free_dofs=factor.free_dofs,
        rigid_modes=model.rigid_modes,
        mass_modes=model.mass_modes,
        mass_gram=model.mass_gram,
        mass_diagonal=model.mass_diagonal,
        rotary_mass=model.rotary_mass,
        nodal_mass=model.nodal_mass,
        center_of_mass=model.center_of_mass,
        gauss_points=model.gauss_points,
        gauss_weights=model.gauss_weights,
        axes=model.axes,
        B_membrane=model.B_membrane,
        B_bending=model.B_bending,
        B_shear=model.B_shear,
        element_dofs=model.element_dofs,
    )


def _export_outputs(writer, bank, label, model, cases, sources, environments, transfer, order, proxy=False):
    np = writer.api.np
    raw_source = np.zeros((model.ndof, len(sources)))
    rows = []
    for index, source in enumerate(sources):
        kwargs = {"analytic": True, "anchor_mode": "fixed"} if proxy else {}
        raw_source[:, index], diagnostics = transfer.transfer(source, order=order, **kwargs)
        rows.append({"source": index, "environment": int(environments[index]), "diagnostics": diagnostics})
    raw = np.zeros((model.ndof, len(cases)))
    for source, environment in enumerate(environments):
        raw[:, environment] += raw_source[:, source]
    balanced, acceleration = model.balance_load(raw)
    q = model.solve(raw)
    fields = model.stress(q)
    free = model.factorize().free_dofs
    residual = np.linalg.norm((model.K @ q - balanced)[free], axis=0)
    residual /= np.maximum(np.linalg.norm(balanced[free], axis=0), 1e-30)
    local = np.stack((fields["local_top"], fields["local_bottom"]))
    xx, yy, xy = local[:, :, :, 0], local[:, :, :, 1], local[:, :, :, 2]
    peak = np.sqrt(np.maximum(0, xx * xx + yy * yy - xx * yy + 3 * xy * xy)).max(axis=(0, 1, 2))
    writer.arrays(
        f"oracles/{bank}/{label}/loads.npz",
        raw_source=raw_source,
        raw=raw,
        balanced=balanced,
        relief_acceleration=acceleration,
        computed_radius=np.array([row["diagnostics"]["radius_m"] for row in rows]),
    )
    writer.arrays(
        f"oracles/{bank}/{label}/output.npz",
        q=q,
        stress_local=local,
        stress_full9=np.stack((fields["sigma_top"], fields["sigma_bottom"])).reshape(
            (2, len(model.quads), 4, 9, len(cases))
        ),
        peak=peak,
        N=fields["N"],
        M=fields["M"],
        Q=fields["Q"],
        unscaled_free_residual=residual,
    )
    writer.json(f"oracles/{bank}/{label}/sources.json", rows)
    release = np.array([not case.contacts for case in cases])
    if residual.max(initial=0.0) > 1e-8 or np.any(q[:, release]) or np.any(local[..., release]):
        raise ArithmeticError(f"CPU residual/release self-check failed for {bank}/{label}.")
    print(f"exported {bank}/{label}: {len(cases)} cases, {len(sources)} sources", flush=True)
    return fields, raw, residual


def _export_bank(writer, bank, model, cases, refinements):
    api, np = writer.api, writer.api.np
    sources, environments = _export_inputs(writer, bank, model, cases)
    _export_operators(writer, bank, model)
    original = api.Q4LoadTransfer(model)
    reference, _, _ = _export_outputs(writer, bank, "Q4-q48", model, cases, sources, environments, original, 48)
    rows = []
    for refinement in (0, *refinements):
        transfer = original if refinement == 0 else api.TriangularLoadTransfer(model, refinement)
        if refinement:
            for name, surface in (("outer", transfer.outer), ("inner", transfer.inner)):
                writer.arrays(
                    f"oracles/{bank}/proxy-k{refinement}/{name}.npz",
                    vertices=surface.vertices,
                    parent_shapes=surface.parent_shapes,
                    parent_faces=surface.parent_faces,
                    normals=surface.normals,
                    dual=surface.dual,
                    centers=surface.centers,
                    extents=surface.extents,
                )
        for order in (24, 48) if refinement else (24,):
            label = f"tri-k{refinement}-q{order}" if refinement else "Q4-q24"
            fields, _raw, residual = _export_outputs(
                writer, bank, label, model, cases, sources, environments, transfer, order, proxy=bool(refinement)
            )
            errors = []
            for index, case in enumerate(cases):
                if not case.contacts:
                    continue
                measured = api.field_errors(
                    model,
                    {key: value[..., index] for key, value in fields.items()},
                    {key: value[..., index] for key, value in reference.items()},
                )
                errors.append({"case": case.name, **measured})
            summary = {
                "method": label,
                "cases": len(cases),
                "sources": len(sources),
                "loaded_cases": len(errors),
                "mean_field": float(np.mean([row["field"] for row in errors])),
                "maximum_unscaled_free_residual": float(residual.max(initial=0.0)),
                **{f"max_{key}": max(row[key] for row in errors) for key in ("field", "peak", "N", "M", "Q")},
                "rows": errors,
            }
            writer.json(f"oracles/{bank}/{label}/spatial.json", summary)
            rows.append({key: value for key, value in summary.items() if key != "rows"})
    writer.json(f"oracles/{bank}/summary.json", rows)
    return rows


def _export_load_controls(writer, refinements):
    api, np = writer.api, writer.api.np
    controls = []
    affine = api._affine_model()
    for side in (-1, 1):
        point = api._shape([0.23, -0.31])[0] @ (affine.vertices + side * 0.5 * affine.h * affine.directors)
        normal = np.array([0.0, 0.0, float(side)])
        contact = api.Contact(
            point,
            -0.05 * normal + np.array([0.012, 0.004, 0.0]),
            normal,
            radius=0.0025,
            friction=0.5,
            side=side,
            face_hint=0,
        )
        controls.append((f"affine-side-{side}", affine, contact, False))
    vertices = np.array([[-0.01, -0.01, 0.002], [0.01, -0.01, -0.002], [0.01, 0.01, 0.002], [-0.01, 0.01, -0.002]])
    warped = api.SurfaceGeometry(vertices, np.array([[0, 1, 2, 3]]), np.tile([0.0, 0.0, 1.0], (4, 1)))
    contact = api.Contact(
        np.array([0.0, 0.0, 0.5 * warped.h]),
        np.array([0.0, 0.0, -0.05]),
        np.array([0.0, 0.0, 1.0]),
        radius=0.0002,
        friction=0.5,
        face_hint=0,
    )
    controls.append(("warped-miss", warped, contact, True))
    tilted = api.SurfaceGeometry(vertices * np.array([1.0, 1.0, 0.5]), warped.quads, warped.directors)
    coordinate = np.array([0.4, -0.35])
    point = api._shape(coordinate)[0] @ (tilted.vertices + 0.5 * tilted.h * tilted.directors)
    normal = np.array([-0.1 * coordinate[1], -0.1 * coordinate[0], 1.0])
    normal /= np.linalg.norm(normal)
    contact = api.Contact(
        point, -0.05 * normal + np.array([0.012, 0.0, 0.0]), normal, radius=0.002, friction=0.5, face_hint=0
    )
    controls.append(("tilted-affine", tilted, contact, False))
    for name, model, contact, expected_miss in controls:
        writer.arrays(
            f"inputs/controls/{name}/model.npz",
            vertices=model.vertices,
            quads=model.quads,
            directors=model.directors,
            thickness=model.h,
            young=model.E,
            poisson=model.nu,
        )
        writer.json(f"inputs/controls/{name}/cases.json", api.case_payload([api.LoadCase(name, name, (contact,))]))
        original, info = api.Q4LoadTransfer(model).transfer(contact, order=48)
        writer.arrays(f"oracles/controls/{name}/Q4-q48.npz", raw=original)
        result = {"expected_proxy_k1_miss": expected_miss, "original": info, "variants": []}
        # Keep the required coarse warped-miss witness, plus only the requested candidates.
        for refinement in dict.fromkeys((1, *refinements)):
            transfer = api.TriangularLoadTransfer(model, refinement)
            try:
                raw, diagnostics = transfer.transfer(contact, order=48, analytic=True, anchor_mode="fixed")
                writer.arrays(f"oracles/controls/{name}/tri-k{refinement}-q48.npz", raw=raw)
                result["variants"].append({"refinement": refinement, "accepted": True, "diagnostics": diagnostics})
            except ValueError as error:
                result["variants"].append({"refinement": refinement, "accepted": False, "reason": str(error)})
        if expected_miss and (not np.linalg.norm(original) > 0 or result["variants"][0]["accepted"]):
            raise ArithmeticError("The pinned warped-miss control did not reproduce its guarded coarse proxy miss.")
        writer.json(f"oracles/controls/{name}/expected.json", result)


def _worker(args):
    started = time.perf_counter()
    api = _cpu_api()
    writer = Writer(args.output, api)
    snapshot = json.loads((args.output / "source-snapshot.json").read_text())
    if _sha256(Path(__file__)) != snapshot["exporter"]["sha256"]:
        raise ArithmeticError("The executing exporter differs from its frozen launch receipt.")
    writer.json(
        "inputs/proxy.json",
        {
            "refinements": args.refinements,
            "triangles_per_parent_side": [2 * value * value for value in args.refinements],
            "subcell_diagonal": [[0, 1, 2], [0, 2, 3]],
            "inner_vertex_permutation": [0, 2, 1],
            "side_order": [1, -1],
            "anchor": "original physical point",
        },
    )
    with api.threadpool_limits(limits=1):
        protocol = json.loads((args.output / "published/protocol.json").read_text())
        if _sha256(args.reference_root / ASSET) != protocol["asset_sha256"]:
            raise ArithmeticError("The primary asset differs from its published protocol.")
        with api.np.load(args.reference_root / ASSET, allow_pickle=False) as asset:
            model = api.ShellModel(
                asset["vertices"],
                asset["quads"],
                E=1e10,
                nu=0.3,
                h=0.0005,
                density=2000.0,
                directors=asset["directors"],
                gauge="translation",
            )
        published = json.loads((args.output / "published/fixtures.json").read_text())
        fixture_sha256 = hashlib.sha256(json.dumps(published, sort_keys=True).encode()).hexdigest()
        if fixture_sha256 != protocol["fixtures_sha256"]:
            raise ArithmeticError("Published physical source inputs differ from their original protocol.")
        generated = api.case_payload(api.make_cases(model))
        if [(row["name"], row["category"]) for row in generated] != [
            (row["name"], row["category"]) for row in published
        ]:
            raise ArithmeticError("The pinned primary no longer generates the published case identities.")
        cases = _cases_from_payload(api, published)
        if len(cases) != 75:
            raise ArithmeticError("The primary bank requires all 75 published cases.")
        for generated_case, frozen_case in zip(generated, published):
            for actual, expected in zip(generated_case["contacts"], frozen_case["contacts"]):
                for key in ("point", "force", "normal", "curvatures"):
                    if not api.np.allclose(actual[key], expected[key], rtol=1e-13, atol=1e-15):
                        raise ArithmeticError(f"The pinned case generator differs from frozen physical inputs: {key}.")
        summaries = _export_bank(writer, "egg75", model, cases, args.refinements)
        lookup = {case.name: index for index, case in enumerate(cases)}
        names = json.loads((args.output / "published/benchmark_cases.json").read_text())
        indices = api.np.array([lookup[name] for name in names], dtype=api.np.int64)
        writer.arrays("inputs/static16/case_indices.npz", reference_case_indices=indices)
        writer.json("inputs/static16/cases.json", [published[index] for index in indices])
        independent = api.run_checks(model, args.output / "oracles/independent")
        if not independent["passed"]:
            raise ArithmeticError("The pinned independent CPU controls failed.")
        _export_load_controls(writer, args.refinements)
        torus = api.torus_model()
        definitions = (
            ("inner-outside-normal", 65, (0.23, -0.17), 1, 0.0, 0.0005),
            ("inner-outside-friction", 65, (0.23, -0.17), 1, 0.3, 0.001),
            ("inner-inside-friction", 70, (-0.31, 0.23), -1, -0.25, 0.001),
            ("inner-material-edge", 73, (1.0, 0.07), 1, 0.2, 0.0005),
            ("inner-triangle-seam", 76, (0.07, 0.07), 1, -0.2, 0.001),
            ("outer-outside-friction", 2, (0.19, -0.27), 1, 0.3, 0.0005),
            ("outer-inside-friction", 8, (-0.31, 0.23), -1, 0.3, 0.001),
            ("top-friction", 36, (0.13, -0.29), 1, -0.2, 0.001),
        )
        torus_cases = [
            api.LoadCase(
                name,
                "nonconvex-prescribed",
                (api.physical_contact(torus, face, coordinate, side=side, shear=shear, radius=radius),),
            )
            for name, face, coordinate, side, shear, radius in definitions
        ]
        _export_bank(writer, "torus8", torus, torus_cases, args.refinements)
        expected = json.loads((args.output / "published/summary.json").read_text())
        comparisons = []
        for row in summaries:
            if row["method"].endswith("q48"):
                continue
            label = "Q4" if row["method"] == "Q4-q24" else row["method"].removesuffix("-q24")
            original = next(value for value in expected if value["method"] == label)
            difference = {
                key: abs(row[key] - original[key])
                for key in ("mean_field", "max_field", "max_peak", "max_N", "max_M", "max_Q")
            }
            comparisons.append(
                {"method": label, "absolute_differences": difference, "passed": max(difference.values()) <= 1e-8}
            )
        writer.json(
            "published-reproduction.json",
            {
                "comparisons": comparisons,
                "absolute_tolerance": 1e-8,
                "passed": all(row["passed"] for row in comparisons),
            },
        )
    for row in snapshot["files"]:
        path = args.output / row["snapshot_path"]
        if _sha256(path) != row["sha256"]:
            raise ArithmeticError(f"Frozen reference bytes changed during execution: {path}.")
    if _sha256(Path(__file__)) != snapshot["exporter"]["sha256"]:
        raise ArithmeticError("The exporter source changed during scientific execution.")
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            relative = str(path.relative_to(args.output))
            if relative not in writer.inventory:
                writer.register(path)
    imports = {
        key: str(Path(module.__file__).resolve())
        for key, module in sys.modules.items()
        if key.startswith("research.rigid_stress.") and module.__file__
    }
    if not all(Path(path).is_relative_to(args.reference_root) for path in imports.values()):
        raise ArithmeticError("A CPU reference module was imported outside the frozen primary tree.")
    passed = independent["passed"] and all(row["passed"] for row in comparisons)
    manifest = {
        "schema": SCHEMA,
        "primary_commit": PRIMARY,
        "checkout_commit": snapshot["checkout_commit"],
        "complete": True,
        "passed": passed,
        "source_unchanged": True,
        "reference_imports": imports,
        "configuration": {
            "refinements": args.refinements,
            "orders": [24, 48],
            "dtype": "float64",
            "egg_raw_dofs": model.ndof,
            "egg_cases": 75,
            "torus_cases": 8,
            "independent_checks": independent["checks"],
        },
        "production_inputs": "Only physical geometry/material, integer topology, physical source/case data and refinement integers under inputs/.",
        "validation_only": "All oracles/ payloads, including CPU proxy geometry/maps, operators, raw loads, solutions, stress and mass/gauge numerics.",
        "norms": {
            "local": "(side,element,Gauss,component=(xx,yy,xy),environment)",
            "full9": "(side,element,Gauss,row-major tensor9,environment)",
            "field": "Two-side Gauss-area weighted tensor Frobenius norm, shear multiplicity two.",
            "N_M": "Gauss-area weighted symmetric tensor norm, shear multiplicity two.",
            "Q": "Gauss-area weighted vector L2 norm.",
            "residual": "Original physical K, common inertia relief and free translation-gauge rows, unscaled.",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": api.np.__version__,
            "scipy": api.scipy.__version__,
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "required_image": "1_26",
        },
        "seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "files": writer.inventory,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": passed,
                "files": len(writer.inventory),
                "seconds": manifest["seconds"],
            }
        ),
        flush=True,
    )
    if not passed:
        raise ArithmeticError("The independent CPU export differs from the published primary results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument(
        "--refinements",
        type=int,
        nargs="+",
        choices=(1, 2, 4, 8, 16),
        default=[8, 16],
        help="Proxy subdivisions to export; e.g. --refinements 4 exports a separate k4 candidate bank.",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reference-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires a Slurm rtx-mid allocation using image 1_26.")
    args.output = args.output.expanduser().resolve()
    if args._worker:
        try:
            _worker(args)
        except BaseException:
            (args.output / "worker-failure.txt").write_text(traceback.format_exc())
            raise
        return
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Choose a fresh output under the user's project data directory /mnt/data/zhaofeng.")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / "reference-source"
    rows = []
    for relative in (*REFERENCE_FILES, *(f"{REFERENCE_DIRECTORY}/results/{name}" for name in PUBLISHED_FILES)):
        content = subprocess.check_output(["git", "-C", str(args.git_source), "show", f"{PRIMARY}:{relative}"])
        destination = (
            frozen / relative if relative in REFERENCE_FILES else args.output / "published" / Path(relative).name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        rows.append(
            {
                "primary_path": relative,
                "snapshot_path": str(destination.relative_to(args.output)),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    checkout = subprocess.check_output(["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True).strip()
    (args.output / "source-snapshot.json").write_text(
        json.dumps(
            {
                "primary_commit": PRIMARY,
                "checkout_commit": checkout,
                "exporter": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
                "files": rows,
                "execution": "Fresh interpreter with PYTHONPATH restricted to the frozen primary tree.",
            },
            indent=2,
        )
        + "\n"
    )
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(frozen),
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker",
            "--output",
            str(args.output),
            "--reference-root",
            str(frozen),
            "--refinements",
            *map(str, args.refinements),
        ],
        cwd=frozen,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
