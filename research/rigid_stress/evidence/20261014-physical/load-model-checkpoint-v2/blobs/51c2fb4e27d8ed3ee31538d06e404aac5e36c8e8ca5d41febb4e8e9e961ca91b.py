"""Sample saved unsmoothed P2 fields at fixed physical probes across independent meshes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from skfem import Basis, ElementTetP2, ElementVector, MeshTet

from .validation_meshes import transform_shell


def common_physical_probes(geometry):
    """Declare interior coordinates independently of mesh nodes and contact samples."""
    if geometry in ("ellipsoid", "thin_neck", "capsule"):
        if geometry == "ellipsoid":
            axes = ([-0.004, 0, 0.004], [-0.005, 0, 0.005], [-0.003, 0, 0.003])
        elif geometry == "thin_neck":
            axes = ([-0.003, 0, 0.003], [-0.0005, 0, 0.0005], [-0.0005, 0, 0.0005])
        else:
            axes = ([-0.003, 0, 0.003], [-0.003, 0, 0.003], [-0.014, -0.008, 0, 0.008, 0.014])
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if geometry in ("u", "narrow_gap"):
        radius = 0.008 if geometry == "u" else 0.0045
        angles, offsets, heights = np.meshgrid(
            np.linspace(np.pi, 2 * np.pi, 9)[1:-1], [-0.0006, 0, 0.0006], [-0.0006, 0, 0.0006], indexing="ij"
        )
        bend = np.stack(((radius + offsets) * np.cos(angles), (radius + offsets) * np.sin(angles), heights), axis=-1)
        arms = np.stack(np.meshgrid([-radius, radius], [0.008, 0.018], [-0.0006, 0, 0.0006]), axis=-1)
        return np.concatenate((bend.reshape(-1, 3), arms.reshape(-1, 3)))
    directions = np.random.default_rng(2026101404).normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions = np.concatenate((directions, np.eye(3), -np.eye(3)))
    if geometry == "cavity":
        return 0.0096 * directions
    if geometry == "egg":
        return transform_shell("egg", 0.96 * directions)
    raise ValueError(f"No declared physical probes for {geometry}")


def locate_probes(mesh, probes):
    """Keep points outside an approximate domain visible; never move or replace them."""
    finder = mesh.element_finder()
    elements = np.full(len(probes), -1, dtype=int)
    for index, point in enumerate(probes):
        try:
            elements[index] = finder(*point[:, None])[0]
        except ValueError as error:
            if str(error) != "Point is outside of the mesh.":
                raise
    selected = elements >= 0
    local = mesh.mapping().invF(probes[selected].T[:, :, None], tind=elements[selected])[:, :, 0]
    barycentric = np.column_stack((1 - local.sum(axis=0), local.T))
    return elements, selected, barycentric


def sample_field(basis, elements, selected, barycentric, probes, field):
    """Interpolate the exact linear tensor within each affine P2 tetrahedron."""
    corners = field["tensor_Pa"]
    tensor = np.full((len(probes), 3, 3), np.nan)
    tensor[selected] = np.einsum("nc,ncij->nij", barycentric, corners[elements[selected]])
    deviator = tensor - np.trace(tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1)))
    displacement = np.full((len(probes), 3), np.nan)
    displacement[selected] = (basis.probes(probes[selected].T) @ field["displacement"]).reshape(3, -1).T
    return tensor, vm, displacement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.mesh) as saved:
        mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
    basis = Basis(mesh, ElementVector(ElementTetP2()), intorder=4)
    probes = common_physical_probes(args.geometry)
    elements, selected, barycentric = locate_probes(mesh, probes)
    np.savez_compressed(args.output / "probe_coordinates.npz", points_m=probes, elements=elements)
    rows = []
    for case in sorted(args.run.glob("*/snapshot.json")):
        snapshot = json.loads(case.read_text())
        assert snapshot["mesh_sha256"] == hashlib.sha256(args.mesh.read_bytes()).hexdigest()
        assert snapshot["geometry"] == args.geometry
        expected_files = [case.parent / "reference_field.npz"]
        expected_files.extend(
            case.parent / f"{row['model']}_field.npz" for row in snapshot["models"] if row["accepted"]
        )
        reference_tensor, reference_vm, reference_u = None, None, None
        for field_path in expected_files:
            with np.load(field_path) as field:
                tensor, vm, displacement = sample_field(basis, elements, selected, barycentric, probes, field)
                corner_tensor = field["tensor_Pa"]
                corner_deviator = (
                    corner_tensor - np.trace(corner_tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
                )
                corner_vm = np.sqrt(1.5 * np.sum(corner_deviator * corner_deviator, axis=(-2, -1)))
                element, corner = np.unravel_index(np.argmax(corner_vm), corner_vm.shape)
                raw_peak = corner_vm[element, corner]
                peak_position = mesh.p[:, mesh.t[corner, element]]
                raw_native_peak, raw_native_position = None, None
                if "raw_native_tensor_Pa" in field:
                    native = field["raw_native_tensor_Pa"]
                    native_vm = np.sqrt(
                        0.5
                        * (
                            (native[..., 0] - native[..., 1]) ** 2
                            + (native[..., 1] - native[..., 2]) ** 2
                            + (native[..., 2] - native[..., 0]) ** 2
                        )
                        + 3 * np.sum(native[..., 3:] ** 2, axis=-1)
                    )
                    native_element, native_corner = np.unravel_index(np.argmax(native_vm), native_vm.shape)
                    raw_native_peak = float(native_vm[native_element, native_corner])
                    raw_native_position = mesh.p[:, mesh.t[native_corner, native_element]].tolist()
            destination = args.output / case.parent.name
            destination.mkdir(exist_ok=True)
            np.savez_compressed(
                destination / field_path.name,
                points_m=probes,
                valid=selected,
                tensor_Pa=tensor,
                vm_Pa=vm,
                displacement_m=displacement,
            )
            row = {
                "case": case.parent.name,
                "field": field_path.stem,
                "input_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
                "raw_global_peak_Pa": float(raw_peak),
                "raw_global_peak_position_m": peak_position.tolist(),
                "raw_native_peak_Pa": raw_native_peak,
                "raw_native_peak_position_m": raw_native_position,
            }
            if reference_tensor is None:
                reference_tensor, reference_vm, reference_u = tensor, vm, displacement
            else:
                for label, actual, expected in (
                    ("tensor", tensor, reference_tensor),
                    ("vm", vm, reference_vm),
                    ("displacement", displacement, reference_u),
                ):
                    row[f"probe_{label}_relative_error"] = float(
                        np.linalg.norm(actual[selected] - expected[selected]) / np.linalg.norm(expected[selected])
                    )
            rows.append(row)
    report = {
        "geometry": args.geometry,
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "probe_coordinate_sha256": hashlib.sha256(probes.tobytes()).hexdigest(),
        "probe_count": len(probes),
        "valid_count": int(selected.sum()),
        "outside_probe_indices": np.flatnonzero(~selected).tolist(),
        "tetrahedra": mesh.nelements,
        "probe_definition": "Fixed physical interiors; no smoothing, relocating or omitted outside coordinates",
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
