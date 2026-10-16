"""Reconstruct actual IPC surface meshes for fixed-boundary thickness controls without solving elasticity."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import ipctk
import numpy as np
from scipy.spatial import cKDTree
from skfem import Basis, ElementTetP2, MeshTet

from .audit_franka_reference import rotation
from .contact_reference import quadratic_boundary_triangles
from .refine_egg_thickness import canonical_rows
from .refined_franka_contact import refine_contact
from .summarize_fine_references import closed_source


def lumped_areas(positions, faces):
    triangles = positions[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    result = np.zeros(len(positions))
    np.add.at(result, faces.reshape(-1), np.repeat(area / 3, 3))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for run in [args.geometry_run, args.source_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    configuration = args.source_run / "results/configuration.json"
    metadata = args.geometry_run / "results/summary.json"
    for path in [configuration, metadata]:
        retain(path)
    config = json.loads(configuration.read_text())
    closure = Path(config["closure_run"]) / "results/configuration.json"
    retain(closure)
    capture = Path(json.loads(closure.read_text())["capture"]) / "results/summary.json"
    retain(capture)
    frame = [r for r in json.loads(capture.read_text())["captures"] if r["mode"] == "stationary_lift_hold"][-1]
    transform = rotation(frame["egg_quaternion_wxyz"])
    shift = np.array(frame["egg_position_m"]) - frame["egg_world_com_m"]
    baseline_path = args.source_run / "results/contact_geometry.npz"
    retain(baseline_path)
    with np.load(baseline_path) as saved:
        original = {name: saved[name].copy() for name in ["rest_positions", "faces", "groups"]}
    finger_ids = np.flatnonzero(original["groups"] > 0)
    finger_inverse = np.full(len(original["groups"]), -1, dtype=int)
    finger_inverse[finger_ids] = np.arange(len(finger_ids))
    finger_faces = original["faces"][np.all(original["groups"][original["faces"]] > 0, axis=1)]
    original_area = lumped_areas(original["rest_positions"], original["faces"])
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for geometric in json.loads(metadata.read_text())["rows"]:
        path = args.geometry_run / f"results/cad-level{geometric['level']}.npz"
        retain(path)
        assert inputs[str(path)] == geometric["mesh_sha256"]
        with np.load(path) as saved:
            vertices = saved["vertices"] @ transform.T
            vertices += shift
            mesh = MeshTet(vertices.T, saved["tetrahedra"].T, sort_t=False)
        # A one-element scalar basis still builds the actual global P2 DOF
        # coordinates/identities; no whole-volume operator or Hessian is built.
        basis = Basis(mesh, ElementTetP2(), elements=np.array([0]), intorder=4)
        body = SimpleNamespace(mesh=mesh, scalar_basis=basis, coordinates=basis.doflocs.T)
        count = len(body.coordinates)
        system = SimpleNamespace(
            body=body,
            rest=np.concatenate((body.coordinates, original["rest_positions"][finger_ids])),
            faces=np.concatenate((quadratic_boundary_triangles(body), finger_inverse[finger_faces] + count)),
            groups=np.r_[np.zeros(count, dtype=int), original["groups"][finger_ids]],
        )
        system = refine_contact(system, 0)
        assert not ipctk.has_intersections(system.mesh, system.contact_surface_rest)
        ids = np.full(len(system.contact_surface_rest), -1, dtype=int)
        maximum_error = 0.0
        for group in np.unique(original["groups"]):
            old = np.flatnonzero(original["groups"] == group)
            new = np.flatnonzero(system.contact_surface_groups == group)
            assert len(old) == len(new)
            if group == 0:
                distances, selected = cKDTree(original["rest_positions"][old]).query(system.contact_surface_rest[new])
                assert distances.max() < 1e-14 and len(np.unique(selected)) == len(old)
            else:
                # Finger nodes retain their original compact order, including
                # geometrically coincident but topologically distinct nodes.
                selected = np.arange(len(old))
                distances = np.linalg.norm(system.contact_surface_rest[new] - original["rest_positions"][old], axis=1)
                assert distances.max() == 0
            ids[new] = old[selected]
            maximum_error = max(maximum_error, float(distances.max()))
        assert np.array_equal(np.sort(ids), np.arange(len(ids)))
        np.testing.assert_array_equal(
            canonical_rows(ids[system.contact_surface_faces]), canonical_rows(original["faces"])
        )
        areas = lumped_areas(system.contact_surface_rest, system.contact_surface_faces)
        np.testing.assert_allclose(areas, original_area[ids], rtol=1e-11, atol=1e-24)
        zero = system.mesh.displace_vertices(np.zeros_like(system.rest))
        np.testing.assert_allclose(zero, original["rest_positions"][ids], rtol=0, atol=1e-14)
        for scale in [0.0, 0.01]:
            values = scale * system.rest + np.array([1e-6, -2e-6, 3e-6])
            mapped = system.mesh.displace_vertices(values)
            expected = original["rest_positions"][ids] * (1 + scale) + np.array([1e-6, -2e-6, 3e-6])
            np.testing.assert_allclose(mapped, expected, rtol=0, atol=1e-14)
        output = args.output / f"level{geometric['level']}"
        output.mkdir()
        np.savez_compressed(
            output / "contact_geometry.npz",
            rest_positions=system.contact_surface_rest,
            faces=system.contact_surface_faces,
            groups=system.contact_surface_groups,
            corresponding_original_compact_surface_ids=ids,
            lumped_areas_m2=areas,
        )
        row = {
            "level": geometric["level"],
            "through_thickness_layers": geometric["through_thickness_layers"],
            "tetrahedra": mesh.nelements,
            "surface_vertices": len(ids),
            "surface_triangles": len(system.contact_surface_faces),
            "maximum_actual_surface_position_difference_m": maximum_error,
            "triangle_topology_identical_under_bijective_vertex_map": True,
            "all_vertex_lumped_areas_checked": True,
            "IPC_zero_and_affine_source_displacements_checked": True,
            "IPC_initial_intersections": False,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert len(rows) == 3
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": inputs,
        "new_equilibrium_states_solved": 0,
        "goal_completion": False,
        "limitations": "The actual existing quadratic boundary subdivision and IPC interpolation reconstruct each volume mesh's contact trace. Body coordinates are matched bijectively to the original compact trace; coincident finger vertices retain distinct identities. Geometry, triangles, groups and lumped areas match, and zero/affine IPC source displacements are checked. The interior basis and interpolation source column count change with thickness. No stiffness, constitutive response, contact-force/state,64-control initializer or five-step dynamics is solved or accepted by this geometry-only check.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
