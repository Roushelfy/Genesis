"""Compare cheap shell residuals with the independently solved normal pressure.

Reuse completed scalar normal-source responses and the fixed material domain.
No new elastic factorization, default runtime pressure solve or vector contact
basis is introduced. All complete arrays and individual reference gates remain
separate from native scene timing.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skfem import Basis, ElementTetP2, ElementVector, MeshTet

from .artifacts import file_id, runtime_identity, write_json
from .bilinear_normal_stress import BilinearNormalStress
from .hertz_reference import ellipse
from .study_hertz_nearfield import stress_measures
from .study_hertz_residual import field_difference
from .study_hertz_shell_residual import column_resultants, particular
from .study_shell_contact_pressure import hat_source_weights, points_on_grid


class P2Readout:
    """Interpolate retained displacements without assembling a new elastic solve."""

    def __init__(self, vertices, tetrahedra, points):
        self.mesh = MeshTet(vertices.T, tetrahedra.T, sort_t=False)
        scalar = Basis(self.mesh, ElementTetP2(), intorder=4)
        self.coordinates = scalar.doflocs.T
        self.basis = Basis(
            self.mesh,
            ElementVector(ElementTetP2()),
            quadrature=(np.eye(4)[1:], np.ones(4)),
        )
        dofs = np.stack(self.basis.split_indices(), axis=-1)
        np.testing.assert_array_equal(dofs.reshape(-1), np.arange(self.basis.N))
        elements, weights = [], []
        for start in range(0, len(points), 256):
            current = points[start : start + 256]
            index = self.mesh.element_finder()(*current.T)
            local = self.basis.mapping.invF(current.T[:, :, None], tind=index)[:, :, 0]
            elements.append(index)
            weights.append(np.column_stack((1 - local.sum(axis=0), local.T)))
        self.elements, self.weights = np.concatenate(elements), np.concatenate(weights)

    def field(self, displacement):
        gradient = self.basis.interpolate(displacement.reshape(-1)).grad.transpose(2, 3, 0, 1)
        shear, lame = 1e10 / 2.6, 1e10 * 0.3 / (1.3 * 0.4)
        corners = shear * (gradient + gradient.swapaxes(-1, -2)) + lame * np.trace(gradient, axis1=-2, axis2=-1)[
            ..., None, None
        ] * np.eye(3)
        common = np.einsum("nc,ncij->nij", self.weights, corners[self.elements])
        return common, corners.reshape(-1, 3, 3)


def metrics(field, points, volumes, column_slice, projection, thickness, depths, weights, areas, common_count):
    volume = field[: len(volumes)]
    columns = field[column_slice].reshape(len(areas), len(depths), 3, 3)
    membrane, bending = column_resultants(columns, projection, thickness, depths, weights)
    tension, vm = stress_measures(field)
    ti, vi = int(tension.argmax()), int(vm.argmax())
    return {
        "volume": volume,
        "membrane": membrane,
        "bending": bending,
        "tensile_max_pa": float(tension[ti]),
        "tensile_location_m": points[ti].tolist(),
        "von_mises_max_pa": float(vm[vi]),
        "von_mises_location_m": points[vi].tolist(),
        "common_tensile_max_pa": float(tension[:common_count].max()),
        "common_von_mises_max_pa": float(vm[:common_count].max()),
    }


def compare(first, second, volumes, areas):
    result = {
        "complete_tensor_relative": float(field_difference(first["volume"], second["volume"], volumes)),
        "membrane_relative": float(field_difference(first["membrane"], second["membrane"], areas)),
        "bending_relative": float(field_difference(first["bending"], second["bending"], areas)),
    }
    for name in ["tensile", "von_mises"]:
        result[name + "_peak_relative"] = abs(first[name + "_max_pa"] / second[name + "_max_pa"] - 1)
        result["common_" + name + "_peak_relative"] = abs(
            first["common_" + name + "_max_pa"] / second["common_" + name + "_max_pa"] - 1
        )
        result[name + "_peak_location_distance_m"] = float(
            np.linalg.norm(np.array(first[name + "_location_m"]) - second[name + "_location_m"])
        )
    return result


def evaluate_pressure(pressure, axes, unique, length, scale, output, label, check):
    model = BilinearNormalStress(axes, pressure, length, scale)
    local = unique.copy()
    local[:, 2] = 0.03 - local[:, 2]
    assert local[:, 2].min() >= 0
    near = np.linalg.norm(local, axis=1) <= 4 * length
    high, low = np.empty((len(unique), 3, 3)), None
    started = perf_counter()
    high[~near] = model.far(local[~near], 17)
    indices = np.flatnonzero(near)
    for start in range(0, len(indices), 128):
        selected = indices[start : start + 128]
        high[selected] = model.near(local[selected], 512)
        write_json(
            output / "particular-progress.json",
            {
                "label": label,
                "near_completed": min(start + 128, len(indices)),
                "near_total": len(indices),
                "far_total": int((~near).sum()),
            },
        )
    checks = {}
    if check:
        low = np.empty_like(high)
        low[~near] = model.far(local[~near], 9)
        for start in range(0, len(indices), 128):
            selected = indices[start : start + 128]
            low[selected] = model.near(local[selected], 256)
        checks = {
            "near256_512_maximum_over_pressure": float(np.max(np.abs(high[near] - low[near])) / scale),
            "far9_17_maximum_over_pressure": float(np.max(np.abs(high[~near] - low[~near])) / scale),
        }
    for field in [high] + ([low] if low is not None else []):
        field[:, 2, :2] *= -1
        field[:, :2, 2] *= -1
    path = output / (label + "-particular.npz")
    np.savez_compressed(
        path, unique_tensor_pa=high, **({"unique_lower_rule_tensor_pa": low} if low is not None else {})
    )
    checks.update(
        seconds=perf_counter() - started,
        near_points=int(near.sum()),
        far_points=int((~near).sum()),
        fields=file_id(path),
    )
    return high, low, checks


def hertz_weights(source_axes, axes, load):
    node, weight = np.polynomial.legendre.leggauss(24)
    t, weight = (node + 1) / 2, weight / 2
    angle = 2 * np.pi * (np.arange(512) + 0.5) / 512
    positions = (
        np.sqrt(1 - t[:, None] ** 2)[..., None] * (np.column_stack((np.cos(angle), np.sin(angle))) * axes)[None]
    ).reshape(-1, 2)
    interpolation = RegularGridInterpolator(source_axes, np.eye(25).reshape(5, 5, 25))(positions)
    return np.einsum("n,ng->g", np.repeat(3 * t * t * weight * load / 512, 512), interpolation)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parents", type=Path, nargs=3, required=True)
    cli.add_argument("--fine-refinement", type=Path, required=True)
    cli.add_argument("--mesh-study", type=Path, required=True)
    cli.add_argument("--pressure-inspection", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load = 0.2
    axes, _, _, peak = ellipse(load, 1e10 / 0.91, np.diag([50.0, 80.0]))
    targets = {
        "reference_field_change": 0.02,
        "reference_peak_change": 0.05,
        "candidate_field_error": 0.05,
        "candidate_peak_error": 0.10,
        "near_integration_over_pressure": 1e-3,
        "far_integration_over_pressure": 1e-5,
    }
    inspection = json.loads(args.pressure_inspection.read_text())
    assert inspection["tested_reference_pressure_variations_within_targets"]
    assert (args.fine_refinement.parent / "returncode.txt").read_text().strip() == "0"
    parents = []
    for parent in args.parents:
        assert (parent.parent / "returncode.txt").read_text().strip() == "0"
        start = json.loads((parent / "start.json").read_text())
        result = json.loads((parent / "result.json").read_text())
        assert len(result["normal_source_checks"]) == 25 and result["execution_completed"]
        parents.append((parent, start))
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "targets": targets,
            "pressure_inspection": file_id(args.pressure_inspection),
            "parents": [file_id(parent / "result.json") for parent, _ in parents],
            "fine_refinement": file_id(args.fine_refinement / "result.json"),
            "scope": "One matched fixed hollow-shell normal case. Independently solved pressure hats feed independently integrated Boussinesq tensors and retained scalar normal elastic responses. Shared-pressure body refinement, pressure65/97, source3/5 and near/far integration gates remain separate. Native throughput, friction and arbitrary-contact qualification are excluded.",
        },
    )
    with np.load(args.mesh_study / "coarse-fields.npz") as data:
        sets = {
            name: data[key]
            for name, key in [
                ("volume", "volume_points_m"),
                ("columns", "column_points_m"),
                ("probes", "probe_points_m"),
            ]
        }
        volumes, areas = data["volume_weights_m3"], data["column_areas_m2"]
    sets["columns"] = sets["columns"].reshape(-1, 3)
    ring_angles = 2 * np.pi * np.arange(160) / 160
    ring_radius = np.linspace(0.75, 1.25, 33)
    ring = (
        ring_radius[:, None, None]
        * np.column_stack((np.cos(ring_angles) * axes[0], np.sin(ring_angles) * axes[1]))[None]
    ).reshape(-1, 2)
    sets["ring"] = np.column_stack((ring, np.full(len(ring), 0.03)))
    common = np.concatenate(list(sets.values()))
    column_slice = slice(len(sets["volume"]), len(sets["volume"]) + len(sets["columns"]))
    meshes = {}
    for name in ["coarse", "medium", "fine"]:
        with np.load(args.mesh_study / (name + "-mesh.npz")) as data:
            meshes[name] = {key: data[key] for key in data.files}
        mesh = meshes[name]
        sets["corners-" + name] = mesh["vertices"][mesh["tetrahedra"]].reshape(-1, 3)
    unique, inverse = np.unique(np.concatenate(list(sets.values())), axis=0, return_inverse=True)
    offsets = np.r_[0, np.cumsum([len(value) for value in sets.values()])]
    mappings = {key: inverse[offsets[i] : offsets[i + 1]] for i, key in enumerate(sets)}
    common_mapping = np.concatenate([mappings[key] for key in ["volume", "columns", "probes", "ring"]])
    np.savez_compressed(
        args.output / "queries.npz",
        unique_points_m=unique,
        common_points_m=common,
        volume_weights_m3=volumes,
        column_areas_m2=areas,
        **{"map_" + key: value for key, value in mappings.items()},
    )
    coarse = meshes["coarse"]
    faces = coarse["mid_faces"]
    outer, inner = coarse["outer_vertices"][faces].mean(axis=1), coarse["inner_vertices"][faces].mean(axis=1)
    gauss, weights = np.polynomial.legendre.leggauss(8)
    depths, weights = (gauss + 1) / 2, weights / 2
    columns = outer[:, None] * (1 - depths[None, :, None]) + inner[:, None] * depths[None, :, None]
    np.testing.assert_allclose(columns.reshape(-1, 3), sets["columns"], rtol=0, atol=1e-16)
    mid = (coarse["outer_vertices"] + coarse["inner_vertices"])[faces] / 2
    vector = np.cross(mid[:, 1] - mid[:, 0], mid[:, 2] - mid[:, 0])
    normals = vector / np.linalg.norm(vector, axis=1)[:, None]
    np.testing.assert_allclose(np.linalg.norm(vector, axis=1) / 2, areas, rtol=1e-12)
    projection = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    thickness = np.abs(np.einsum("fi,fi->f", outer - inner, normals))
    pressure_data = {}
    for count, path in [(65, parents[-1][0] / "pressures.npz"), (97, args.fine_refinement / "pressures.npz")]:
        with np.load(path) as data:
            pressure_data[count] = (data[f"grid{count}-sources5-pressure_pa"], data[f"grid{count}-sources5-force_n"])
    particular_fields, numerical = {}, {}
    lower = None
    for count in [97, 65]:
        pressure, _ = pressure_data[count]
        grid = [np.linspace(-1.3 * width, 1.3 * width, count) for width in axes]
        field, low, checks = evaluate_pressure(
            pressure, grid, unique, axes[0], peak, args.output, f"pressure{count}", count == 97
        )
        particular_fields[count] = field
        numerical[count] = checks
        if count == 97:
            lower = low
    started = perf_counter()
    hertz = particular(unique, axes, peak, np.array([0, 0, 0.03]), 16)
    np.savez_compressed(args.output / "hertz16-particular.npz", unique_tensor_pa=hertz)
    numerical["hertz16_seconds"] = perf_counter() - started
    write_json(args.output / "particular-result.json", numerical)
    source_axes = [np.linspace(-1.5 * width, 1.5 * width, 5) for width in axes]
    source_selected = np.array([i * 5 + j for i in [0, 2, 4] for j in [0, 2, 4]])
    coefficient = {}
    for count in [65, 97]:
        grid = [np.linspace(-1.3 * width, 1.3 * width, count) for width in axes]
        points = points_on_grid(grid)
        widths = np.array([axis[1] - axis[0] for axis in grid])
        coefficient[count] = hat_source_weights(source_axes, points, widths) @ pressure_data[count][1]
    grid = [np.linspace(-1.3 * width, 1.3 * width, 97) for width in axes]
    coefficient3 = (
        hat_source_weights(
            [axis[::2] for axis in source_axes], points_on_grid(grid), np.array([axis[1] - axis[0] for axis in grid])
        )
        @ pressure_data[97][1]
    )
    hertz_coefficient = hertz_weights(source_axes, axes, load)
    rows, records, legacy_checks = [], {}, {}
    for name, (parent, start) in zip(["coarse", "medium", "fine"], parents):
        mesh = meshes[name]
        assert file_id(args.mesh_study / (name + "-mesh.npz")) == start["mesh"]
        readout = P2Readout(mesh["vertices"], mesh["tetrahedra"], common)
        affine = np.array([[1e-5, 2e-6, 0], [-1e-6, -0.5e-5, 1e-6], [0.2e-5, 0, 1.1e-5]])
        expected = (1e10 / 2.6) * (affine + affine.T) + (1e10 * 0.3 / (1.3 * 0.4)) * np.trace(affine) * np.eye(3)
        affine_common, affine_corners = readout.field(readout.coordinates @ affine.T)
        readout_error = max(
            float(np.max(np.abs(affine_common - expected))), float(np.max(np.abs(affine_corners - expected)))
        ) / np.linalg.norm(expected)
        assert readout_error < 1e-7, readout_error
        with np.load(parent / "normal-source-table.npz") as table:
            np.testing.assert_array_equal(readout.coordinates, table["coordinates_m"])
            displacement = table["displacement_m_per_n"]
        variants = {
            "reference97": (np.einsum("nig,g->ni", displacement, coefficient[97]), particular_fields[97]),
            "hertz-point": (load * displacement[:, :, 12], hertz),
            "hertz-distributed": (np.einsum("nig,g->ni", displacement, hertz_coefficient), hertz),
            "hertz-only": (np.zeros_like(displacement[:, :, 0]), hertz),
        }
        if name == "fine":
            variants.update(
                {
                    "reference65": (np.einsum("nig,g->ni", displacement, coefficient[65]), particular_fields[65]),
                    "reference97-sources3": (
                        np.einsum("nig,g->ni", displacement[:, :, source_selected], coefficient3),
                        particular_fields[97],
                    ),
                    "reference97-lower-integration": (variants["reference97"][0], lower),
                }
            )
        arrays = {}
        points = np.concatenate((common, sets["corners-" + name]))
        metrics_per_variant = {}
        for label, (nodal, part) in variants.items():
            value, corners = readout.field(nodal)
            value += part[common_mapping]
            corners += part[mappings["corners-" + name]]
            complete = np.concatenate((value, corners))
            summary = metrics(
                complete, points, volumes, column_slice, projection, thickness, depths, weights, areas, len(common)
            )
            metrics_per_variant[label] = summary
            arrays[label + "-common_tensor_pa"] = value
            arrays[label + "-corner_tensor_pa"] = corners
            arrays[label + "-displacement_m"] = nodal
            arrays[label + "-membrane_n_per_m"] = summary["membrane"]
            arrays[label + "-bending_n"] = summary["bending"]
        with np.load(args.mesh_study / (name + "-fields.npz")) as old:
            legacy = old["corrected_volume_stress_pa"]
            legacy_checks[name] = float(
                field_difference(metrics_per_variant["hertz-distributed"]["volume"], legacy, volumes)
            )
        path = args.output / (name + "-fields.npz")
        np.savez_compressed(path, **arrays)
        record = {
            "mesh": name,
            "affine_readout_relative_error": readout_error,
            "fields": file_id(path),
            "variants": {
                label: {key: value for key, value in summary.items() if not isinstance(value, np.ndarray)}
                for label, summary in metrics_per_variant.items()
            },
        }
        rows.append(record)
        records[name] = metrics_per_variant
        write_json(args.output / "field-progress.json", {"rows": rows})
    reference = records["fine"]["reference97"]
    reference_changes = {
        "body_medium_fine": compare(records["medium"]["reference97"], reference, volumes, areas),
        "body_coarse_fine": compare(records["coarse"]["reference97"], reference, volumes, areas),
        "pressure65_97": compare(records["fine"]["reference65"], reference, volumes, areas),
        "source3_5": compare(records["fine"]["reference97-sources3"], reference, volumes, areas),
        "integration_lower_higher": compare(
            records["fine"]["reference97-lower-integration"], reference, volumes, areas
        ),
    }
    comparisons = {
        name: {
            label: compare(records[name][label], reference, volumes, areas)
            for label in ["hertz-only", "hertz-point", "hertz-distributed"]
        }
        for name in records
    }
    reference_met = all(
        max(row[key] for key in ["complete_tensor_relative", "membrane_relative", "bending_relative"])
        <= targets["reference_field_change"]
        and max(
            row[key]
            for key in [
                "tensile_peak_relative",
                "von_mises_peak_relative",
                "common_tensile_peak_relative",
                "common_von_mises_peak_relative",
            ]
        )
        <= targets["reference_peak_change"]
        for key, row in reference_changes.items()
        if key != "body_coarse_fine"
    )
    reference_met = (
        reference_met
        and numerical[97]["near256_512_maximum_over_pressure"] <= targets["near_integration_over_pressure"]
        and numerical[97]["far9_17_maximum_over_pressure"] <= targets["far_integration_over_pressure"]
        and max(legacy_checks.values()) <= targets["reference_field_change"]
    )
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": rows,
            "reference_changes": reference_changes,
            "candidate_errors": comparisons,
            "normal_source_table_versus_exact_prescribed_hertz_volume_changes": legacy_checks,
            "numerical_checks": numerical,
            "targets": targets,
            "conditional_single_case_reference_targets_met": bool(reference_met),
            "conditional_candidate_targets_met": {
                name: {
                    label: bool(
                        reference_met
                        and max(
                            row[key] for key in ["complete_tensor_relative", "membrane_relative", "bending_relative"]
                        )
                        <= targets["candidate_field_error"]
                        and max(
                            row[key]
                            for key in [
                                "tensile_peak_relative",
                                "von_mises_peak_relative",
                                "common_tensile_peak_relative",
                                "common_von_mises_peak_relative",
                            ]
                        )
                        <= targets["candidate_peak_error"]
                    )
                    for label, row in entries.items()
                }
                for name, entries in comparisons.items()
            },
            "native_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
            "scope": "Conditional single normal polar case, shared independent fine97 pressure through body refinement. Complete volume tensors, membrane/bending and global sampled tensile/VM maxima include common volume/columns/probes, an added contact-edge ring and every own element corner. Reference grid/source/integration changes and prescribed-Hertz transfer controls are explicit. Candidate point/distributed shell residuals use retained elastic normal responses; no native scene cost, general geometry/friction or continuum extremum certification follows.",
        },
    )


if __name__ == "__main__":
    main()
