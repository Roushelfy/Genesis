"""Separate same-mesh port changes from combined remeshing changes in native histories."""

import argparse
import hashlib
import json
import shlex
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from skfem import Basis, ElementTetP2, ElementVector, MeshTet

from .profile_reference_sensitivity import tensor_l2
from .sample_load_fields import common_physical_probes, locate_probes, rigid_gauge, sample_field
from .summarize_fine_references import closed_source, relative_change


def tensor_energy(tensor, volumes):
    """Integrate sigma:C^-1:sigma/2 for the declared 10GPa, nu=0.3 material."""
    trace = np.trace(tensor, axis1=-2, axis2=-1)
    trace_square = np.sum(volumes / 20 * (np.sum(trace**2, axis=1) + trace.sum(axis=1) ** 2))
    shear_modulus = 1e10 / (2 * 1.3)
    return float((tensor_l2(tensor, volumes) ** 2 - 0.3 / 1.3 * trace_square) / (4 * shear_modulus))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, sources, layouts = [], [], {}
    samples = {}
    for run in args.run:
        closed_source(run)
        configuration_file = run / "results/configuration.json"
        if configuration_file.exists():
            configuration = json.loads(configuration_file.read_text())
        else:
            argv = shlex.split((run / "arguments.txt").read_text())
            configuration = {"mesh": argv[argv.index("--mesh") + 1], "surface_edge_subdivisions": 8}
        mesh_path = Path(configuration["mesh"])
        mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        ports = configuration["surface_edge_subdivisions"]
        if mesh_hash not in layouts:
            with np.load(mesh_path) as saved:
                vertices, tets = saved["vertices"], saved["tetrahedra"]
            mesh = MeshTet(vertices.T, tets.T, sort_t=False)
            volumes = abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
            com = np.sum(volumes[:, None] * vertices[tets].mean(axis=1), axis=0) / volumes.sum()
            basis = Basis(mesh, ElementVector(ElementTetP2()), intorder=4)
            probes = common_physical_probes("u")
            elements, selected, barycentric = locate_probes(mesh, probes)
            modes, mass_modes, gram = rigid_gauge(basis, com)
            layouts[mesh_hash] = (
                mesh,
                volumes,
                basis,
                probes,
                elements,
                selected,
                barycentric,
                modes,
                mass_modes,
                gram,
            )
        mesh, volumes, basis, probes, elements, selected, barycentric, modes, mass_modes, gram = layouts[mesh_hash]
        sources.append(
            {
                "run": str(run),
                "returncode": int((run / "returncode.txt").read_text()),
                "configuration": configuration,
                "path_initialization": json.loads((run / "results/path_initialization.json").read_text())
                if (run / "results/path_initialization.json").exists()
                else None,
            }
        )
        seen = set()
        for snapshot in sorted((run / "results").glob("*_snapshot.json")):
            row = json.loads(snapshot.read_text())
            seen.add((row["model"], row["step"]))
            row.update(
                run=str(run),
                mesh_sha256=mesh_hash,
                tetrahedra=mesh.nelements,
                ports=ports,
                snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            )
            inputs_path = run / "results" / f"{row['label']}_inputs.npz"
            row["rigid_inputs_sha256"] = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
            field_path = snapshot.with_name(snapshot.name.replace("_snapshot.json", "_field.npz"))
            if row.get("classification") == "numerically_qualified" and field_path.exists():
                with np.load(field_path) as field:
                    tensor, vm = field["tensor_Pa"], field["vm_Pa"]
                    reconstructed = np.sqrt(
                        1.5
                        * np.sum(
                            (tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3) ** 2,
                            axis=(-2, -1),
                        )
                    )
                    np.testing.assert_allclose(vm, reconstructed, rtol=1e-12, atol=1e-8)
                    np.testing.assert_allclose(row["raw_global_peak_Pa"], vm.max(), rtol=1e-4, atol=1e-3)
                    element, corner = np.unravel_index(vm.argmax(), vm.shape)
                    aligned = field["displacement"] - modes @ np.linalg.solve(
                        gram, mass_modes.T @ field["displacement"]
                    )
                    sampled = sample_field(
                        basis, elements, selected, barycentric, probes, {"tensor_Pa": tensor, "displacement": aligned}
                    )
                    row.update(
                        raw_global_peak_position_m=mesh.p[:, mesh.t[corner, element]].tolist(),
                        energy_J=tensor_energy(tensor, volumes),
                        field_sha256=hashlib.sha256(field_path.read_bytes()).hexdigest(),
                        outside_probe_indices=np.flatnonzero(~selected).tolist(),
                    )
                    samples[str(field_path)] = (*sampled, selected)
                    np.savez_compressed(
                        args.output / f"{run.name}_{field_path.stem}_probes.npz",
                        points_m=probes,
                        tensor_Pa=sampled[0],
                        vm_Pa=sampled[1],
                        displacement_m=sampled[2],
                        valid=selected,
                    )
            else:
                row.setdefault("classification", "incomplete_qualification")
            rows.append(row)
        complete_summary = run / "results/summary.json"
        if complete_summary.exists():
            summary = json.loads(complete_summary.read_text())
            sources[-1]["warmed_costs"] = summary["warmed_costs"]
            for row in summary["rows"]:
                if (row["model"], row["step"]) not in seen:
                    assert row["classification"] != "numerically_qualified"
                    row.update(run=str(run), mesh_sha256=mesh_hash, tetrahedra=mesh.nelements, ports=ports)
                    rows.append(row)
    qualified = [r for r in rows if r["classification"] == "numerically_qualified"]
    port_groups, mesh_groups = defaultdict(list), defaultdict(list)
    for row in qualified:
        port_groups[row["mesh_sha256"], row["model"], row["label"]].append(row)
        mesh_groups[row["ports"], row["model"], row["label"]].append(row)
    port_changes, mesh_changes = [], []
    for group in port_groups.values():
        for first, second in pairwise(sorted(group, key=lambda r: r["ports"])):
            assert first["rigid_inputs_sha256"] == second["rigid_inputs_sha256"]
            prefix = f"{first['model']}_{first['step']}_field.npz"
            with (
                np.load(Path(first["run"]) / "results" / prefix) as earlier,
                np.load(Path(second["run"]) / "results" / prefix) as later,
            ):
                volumes = layouts[first["mesh_sha256"]][1]
                denominator = tensor_l2(later["tensor_Pa"], volumes)
                absolute = tensor_l2(later["tensor_Pa"] - earlier["tensor_Pa"], volumes)
            port_changes.append(
                {
                    "first_run": first["run"],
                    "second_run": second["run"],
                    "model": first["model"],
                    "label": first["label"],
                    "tetrahedra": first["tetrahedra"],
                    "ports": [first["ports"], second["ports"]],
                    "volume_l2_tensor_absolute_change_Pa_m1_5": absolute,
                    "volume_l2_tensor_relative_change": absolute / denominator if denominator else None,
                    "raw_peak_absolute_change_Pa": abs(second["raw_global_peak_Pa"] - first["raw_global_peak_Pa"]),
                    "energy_absolute_change_J": abs(second["energy_J"] - first["energy_J"]),
                }
            )
    for group in mesh_groups.values():
        ordered = sorted(group, key=lambda r: r["tetrahedra"])
        for first, second in pairwise(ordered):
            if first["mesh_sha256"] == second["mesh_sha256"]:
                continue
            prefix = f"{first['model']}_{first['step']}_field.npz"
            earlier = samples[str(Path(first["run"]) / "results" / prefix)]
            later = samples[str(Path(second["run"]) / "results" / prefix)]
            common = earlier[3] & later[3]
            change = {
                "first_run": first["run"],
                "second_run": second["run"],
                "model": first["model"],
                "label": first["label"],
                "ports": first["ports"],
                "tetrahedra": [first["tetrahedra"], second["tetrahedra"]],
                "excluded_probe_indices": np.flatnonzero(~common).tolist(),
                "raw_peaks_Pa": [first["raw_global_peak_Pa"], second["raw_global_peak_Pa"]],
                "raw_peak_positions_m": [first["raw_global_peak_position_m"], second["raw_global_peak_position_m"]],
            }
            assert common.any()
            for index, label in enumerate(("tensor", "vm", "displacement")):
                relative, absolute = relative_change(earlier[index][common], later[index][common])
                change[f"probe_{label}_relative_change"] = relative
                change[f"probe_{label}_absolute_change"] = absolute
            with (
                np.load(Path(first["run"]) / "results" / f"{first['label']}_inputs.npz") as a,
                np.load(Path(second["run"]) / "results" / f"{second['label']}_inputs.npz") as b,
            ):
                change["input_force_absolute_change_N"] = float(np.linalg.norm(a["forces_N"] - b["forces_N"]))
                change["input_points_absolute_change_m"] = float(np.linalg.norm(a["points_m"] - b["points_m"]))
                moments_a = a["free_moments_Nm"] + np.cross(a["points_m"], a["forces_N"])
                moments_b = b["free_moments_Nm"] + np.cross(b["points_m"], b["forces_N"])
                change["input_origin_moment_absolute_change_Nm"] = float(np.linalg.norm(moments_a - moments_b))
            mesh_changes.append(change)
    report = {
        "scope": __doc__,
        "source_closures": sources,
        "rows": rows,
        "classifications": dict(Counter(r["classification"] for r in rows)),
        "same_mesh_port_changes": port_changes,
        "combined_remeshing_changes": mesh_changes,
        "limitations": "Same-mesh port comparisons freeze original input bytes. Remeshing also changes geometry and continuous-traction integration; original force-point/wrench changes are retained. Raw peaks remain global and unsmoothed. Prescribed-reference traction admissibility and resolved-contact acceptance remain separate. No new physical acceptance or paired GPU speedup is claimed.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "classifications": report["classifications"],
                "port_comparisons": len(port_changes),
                "remeshing_comparisons": len(mesh_changes),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
