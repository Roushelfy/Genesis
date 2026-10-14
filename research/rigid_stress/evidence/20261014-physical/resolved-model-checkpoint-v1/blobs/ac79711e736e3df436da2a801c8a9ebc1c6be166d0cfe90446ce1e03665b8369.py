"""Compare native finite-footprint laws against immutable resolved contact loads."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .finite_traction_reference import FinitePatch, PatchTractionReference, triangle_closest_points
from .load_model_validation import map_native, native_cloud, traction_observables
from .native_physical_sequence import build_scene
from .profile_reference_sensitivity import tensor_l2
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--ports", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.replay)
    assert int((args.replay / "returncode.txt").read_text()) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.replay / "results/summary.json").read_text())
    reference_run = Path(report["reference"])
    material = json.loads((reference_run / "solution/result.json").read_text())
    with np.load(args.mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    reference = SkfemElasticReference(
        vertices,
        tets,
        material["material"]["body_young_Pa"],
        material["material"]["body_poisson"],
        material["material"]["body_density_kg_m3"],
    )
    volumes = abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
    with np.load(args.replay / "results/contact_hold.npz") as saved:
        sources = (saved["source_normal_force_N"] + saved["source_shear_force_N"])[:, : len(reference.coordinates)]
        body_force = saved["body_force_N"].copy()
        body_tensor = saved["body_tensor_Pa"].copy()
        body_u = saved["body_displacement"].copy()
        current_positions = (
            saved["rest_positions"][: len(reference.coordinates)]
            + saved["displacement"].reshape(-1, 3)[: len(reference.coordinates)]
        )
    np.testing.assert_allclose(sources.sum(axis=0), body_force[reference.component_dofs], rtol=1e-10, atol=1e-12)
    boundary = reference.mesh.boundary_facets()
    triangles = vertices[reference.mesh.facets[:, boundary].T]
    patches, source_rows = [], []
    for source in range(2):
        normals = [r for r in report["normal_stencils"] if r["shoe"] == source]
        shear = [r for r in report["friction_stencils"] if r["shoe"] == source]
        assert len(normals) == len(shear)
        positions, force_samples, inward_samples, areas = [], [], [], []
        for first, second in zip(normals, shear, strict=True):
            assert first["nodes"] == second["nodes"]
            point = np.array(first["reference_application_point_m"])
            np.testing.assert_allclose(point, second["reference_application_point_m"], rtol=0, atol=1e-10)
            normal_force = np.array(first["body_resultant_N"])
            force_samples.append(normal_force + second["body_resultant_N"])
            inward_samples.append(normal_force / np.linalg.norm(normal_force))
            positions.append(point)
            areas.append(first["weight"])
        positions, force_samples, inward_samples, areas = map(
            np.array, (positions, force_samples, inward_samples, areas)
        )
        force = sources[source].sum(axis=0)
        normal_weights = np.array([np.linalg.norm(r["body_resultant_N"]) for r in normals])
        point = normal_weights @ positions / normal_weights.sum()
        closest, distance = triangle_closest_points(triangles, point)
        witness = closest[distance.argmin()]
        moment = np.cross(reference.coordinates - point, sources[source]).sum(axis=0)
        np.testing.assert_allclose(force_samples.sum(axis=0), force, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.cross(positions - point, force_samples).sum(axis=0), moment, rtol=0, atol=1e-12)
        normal = np.array([r["body_resultant_N"] for r in normals]).sum(axis=0)
        normal /= np.linalg.norm(normal)
        specification = FinitePatch(
            f"shoe{source}",
            witness,
            force / np.linalg.norm(force),
            float(np.linalg.norm(force)),
            args.radius,
            material["friction"],
        )
        load = np.zeros(reference.basis.N)
        load[reference.component_dofs] = sources[source]
        patches.append(
            PatchTractionReference(
                specification,
                witness,
                point,
                force,
                moment,
                normal,
                positions,
                inward_samples,
                areas,
                force_samples / areas[:, None],
                load,
                float(areas.sum()),
                boundary,
            )
        )
        observations = traction_observables(
            positions, inward_samples, force_samples, areas, np.full(len(areas), material["friction"]), reference.com
        )
        observations.update(
            physical_source=source,
            point_m=point.tolist(),
            witness_m=witness.tolist(),
            free_moment_Nm=moment.tolist(),
            pressure_sample_maximum_distance_from_point_m=float(np.linalg.norm(positions - point, axis=1).max()),
            deformed_minus_reference_origin_moment_Nm=np.cross(
                current_positions - reference.coordinates, sources[source]
            )
            .sum(axis=0)
            .tolist(),
            area_definition="IPC positive weighted stencil areas, not a reconstructed continuum patch support",
        )
        source_rows.append(observations)
    np.savez_compressed(
        args.output / "frozen_inputs.npz",
        points_m=[p.point_m for p in patches],
        witnesses_m=[p.witness_m for p in patches],
        forces_N=[p.force_n for p in patches],
        free_moments_Nm=[p.free_moment_nm for p in patches],
        normals=[p.inward_normal for p in patches],
        radii_m=[p.specification.radius_m for p in patches],
        friction=[p.specification.friction for p in patches],
        omega_rad_s=np.zeros(3),
        frame_position_m=np.zeros(3),
        frame_quaternion=[1, 0, 0, 0],
    )
    inputs_hash = hashlib.sha256((args.output / "frozen_inputs.npz").read_bytes()).hexdigest()
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    _scene, link = build_scene(reference, args.output)
    base_options = RigidStressOptions(
        mesh=args.mesh,
        young=reference.young,
        poisson=reference.poisson,
        density=reference.density,
        method="pcg",
        output_mode="full",
        max_iterations=20000,
    )
    model = StressModel(base_options)
    surface = StressSurface(10, model.info)
    paths = None
    permutation = reference.native_dof_permutation(qd_to_numpy(model.info.vertices, transpose=True, copy=True))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    omega.fill(0)
    rows = []
    for name in ("pressure", "componentwise", "pair_union", "body_source_prior"):
        row = {"model": name, "frozen_inputs_sha256": inputs_hash}
        if name == "pressure" and any(
            np.linalg.norm(p.free_moment_nm) > 1e-8 * np.linalg.norm(p.force_n) * args.radius for p in patches
        ):
            row["classification"] = "unsupported_free_contact_moment"
            rows.append(row)
            continue
        if name != "pressure" and paths is None:
            paths = StressSurfacePaths(model.info, args.ports)
        options = RigidStressOptions(
            mesh=args.mesh,
            young=reference.young,
            poisson=reference.poisson,
            density=reference.density,
            method="pcg",
            output_mode="full",
            max_iterations=20000,
            quadrature=10,
            contact_radius=args.radius,
            surface_edge_subdivisions=args.ports,
            contact_model="pressure" if name == "pressure" else "intrinsic",
            intrinsic_support="pair_union" if name == "pair_union" else "contact",
            intrinsic_load_partition="body_source_prior" if name == "body_source_prior" else "componentwise",
        )
        state, contacts, mapped, errno = map_native(
            model, surface, paths, link, patches, options, np.arange(2, dtype=gs.np_int)
        )
        status = qd_to_numpy(contacts.status, transpose=True, copy=True)[0]
        row.update(contact_status=status.tolist(), errno=errno)
        np.savez_compressed(
            args.output / f"{name}_mapped.npz", force_N=qd_to_numpy(state.force, transpose=True, copy=True)
        )
        if errno or status.any():
            row["classification"] = "native_load_rejected"
        else:
            model.recover(omega, state, options)
            valid = bool(qd_to_numpy(state.valid, transpose=True)[0])
            native_u, native_force = np.empty(reference.basis.N), np.empty(reference.basis.N)
            native_u[permutation] = qd_to_numpy(state.displacement, transpose=True, copy=True)[0].reshape(-1)
            native_force[permutation] = qd_to_numpy(state.force, transpose=True, copy=True)[0].reshape(-1)
            raw = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
            tensor, vm = reference.corner_stress(native_u)
            np.savez_compressed(
                args.output / f"{name}_field.npz",
                force_N=native_force,
                displacement=native_u,
                tensor_Pa=tensor,
                vm_Pa=vm,
                raw_native_tensor_Pa=raw,
                native_valid=valid,
            )
            if not valid:
                row["classification"] = "native_elastic_rejected"
            else:
                rhs = reference.balanced_rhs(native_force)
                budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
                residual = np.linalg.norm(reference.stiffness @ native_u - rhs)
                assert residual <= budget
                np.testing.assert_allclose(
                    raw, tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]], rtol=1e-9, atol=1e-5
                )
                np.testing.assert_allclose(qd_to_numpy(state.peak)[0], vm.max(), rtol=1e-4, atol=1e-3)
                np.testing.assert_allclose(
                    reference.rigid_modes.T @ native_force, reference.rigid_modes.T @ body_force, rtol=1e-8, atol=1e-10
                )
                cloud = native_cloud(model, surface, contacts, mapped, np.arange(2), patches)
                source_observations = []
                for source in range(2):
                    chosen = cloud.source == source
                    obs = traction_observables(
                        cloud.positions_m[chosen],
                        cloud.normals[chosen],
                        cloud.forces_n[chosen],
                        cloud.areas_m2[chosen],
                        cloud.friction[chosen],
                        reference.com,
                    )
                    obs["force_shift_from_resolved_prior_N"] = (
                        np.array(obs["resultant_N"]) - patches[source].force_n
                    ).tolist()
                    prior_moment = np.cross(reference.coordinates - reference.com, sources[source]).sum(axis=0)
                    obs["moment_shift_from_resolved_prior_Nm"] = (np.array(obs["moment_Nm"]) - prior_moment).tolist()
                    source_observations.append(obs)
                np.savez_compressed(
                    args.output / f"{name}_traction.npz",
                    positions_m=cloud.positions_m,
                    normals=cloud.normals,
                    forces_N=cloud.forces_n,
                    areas_m2=cloud.areas_m2,
                    source=cloud.source,
                    faces=cloud.faces,
                )
                element, corner = np.unravel_index(vm.argmax(), vm.shape)
                row.update(
                    classification="numerically_qualified",
                    complete_budget_fraction=float(residual / budget),
                    raw_global_peak_Pa=float(vm.max()),
                    raw_global_peak_position_m=reference.corner_positions[element, corner].tolist(),
                    volume_l2_tensor_relative_difference_from_resolved_contact=tensor_l2(tensor - body_tensor, volumes)
                    / tensor_l2(body_tensor, volumes),
                    energy_J=float(0.5 * native_u @ (reference.stiffness @ native_u)),
                    reference_energy_J=float(0.5 * body_u @ (reference.stiffness @ body_u)),
                    source_charts=source_observations,
                )
        rows.append(row)
        (args.output / f"{name}_snapshot.json").write_text(json.dumps(row, indent=2) + "\n")
    summary = {
        "scope": __doc__,
        "replay": str(args.replay),
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "physical_radius_m": args.radius,
        "ports": args.ports,
        "reference_sources": source_rows,
        "rows": rows,
        "limitations": "Resolved nodal forces are reduced to source wrenches in linear-FEM material coordinates. Original force/moment inputs remain fixed across models; deformed-space moment differences are recorded. Native profile choice, finite footprint and IPC stencil integration weights are different declared measures. Spatial/regularization/nonlinear pad convergence, actual Franka matching and rigid-contact feedback remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
