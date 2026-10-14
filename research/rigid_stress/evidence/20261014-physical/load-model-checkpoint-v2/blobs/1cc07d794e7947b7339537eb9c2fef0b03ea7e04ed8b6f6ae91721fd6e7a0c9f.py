"""Compare four native load models with independent finite-traction FEM on frozen snapshots."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.intrinsic import IntrinsicContacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionSamples,
    create_traction_samples,
    kernel_traction_samples,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy

from .finite_traction_reference import assemble_finite_patch, physical_grip_patches
from .native_load_samples import kernel_intrinsic_load_samples, kernel_pressure_load_samples
from .qualify_reference_scaling import AMGPinFactor
from .skfem_reference import SkfemElasticReference


@dataclass(frozen=True)
class TractionCloud:
    positions_m: np.ndarray
    normals: np.ndarray
    forces_n: np.ndarray
    areas_m2: np.ndarray
    source: np.ndarray
    friction: np.ndarray
    faces: np.ndarray
    basis: np.ndarray


def traction_observables(positions, normals, forces, areas, friction, origin):
    """Measure traction, wrench and concentration with physical area weights."""
    density = forces / areas[:, None]
    pressure = np.einsum("qi,qi->q", density, normals)
    shear = density - pressure[:, None] * normals
    shear_norm = np.linalg.norm(shear, axis=1)
    magnitude = np.linalg.norm(forces, axis=1)
    effective_area = magnitude.sum() ** 2 / np.sum(magnitude * magnitude / areas)
    return {
        "resultant_N": forces.sum(axis=0).tolist(),
        "moment_Nm": np.cross(positions - origin, forces).sum(axis=0).tolist(),
        "footprint_area_m2": float(areas.sum()),
        "loaded_area_m2": float(areas[magnitude > 0].sum()),
        "effective_traction_area_m2": float(effective_area),
        "maximum_pressure_Pa": float(pressure.max()),
        "minimum_pressure_Pa": float(pressure.min()),
        "maximum_shear_Pa": float(shear_norm.max()),
        "integrated_pressure_N": float(pressure @ areas),
        "integrated_shear_norm_N": float(shear_norm @ areas),
        "integrated_traction_norm_N": float(magnitude.sum()),
        "integrated_traction_squared_N2_per_m2": float(np.sum(magnitude * magnitude / areas)),
        "maximum_coulomb_violation_Pa": float(max(0, np.maximum(shear_norm - friction * pressure, -pressure).max())),
    }


def map_native(model, surface, paths, link, patches, options, pairs, output_mode="full"):
    """Run the production load mapper for a complete frozen set of contacts."""
    contacts = create_contacts(len(patches), 1, options.contact_radius)
    contacts.valid.fill(True)
    contacts.position.from_numpy(np.stack([patch.point_m for patch in patches])[:, None])
    contacts.force.from_numpy(np.stack([patch.force_n for patch in patches])[:, None])
    contacts.normal.from_numpy(np.stack([patch.inward_normal for patch in patches])[:, None])
    contacts.radius.from_numpy(np.array([patch.specification.radius_m for patch in patches])[:, None])
    contacts.friction.from_numpy(np.array([patch.specification.friction for patch in patches])[:, None])
    contacts.frame_position.fill(0)
    contacts.frame_quaternion.from_numpy(np.array([[1.0, 0, 0, 0]]))
    state = model.create_state(1, output_mode)
    kernel_begin_step(state)
    epsilon = float(np.finfo(gs.np_float).eps)
    mapped = None
    errno = V(dtype=gs.qd_int, shape=(1,))
    errno.fill(0)
    if options.contact_model == "pressure":
        kernel_anchor(epsilon, contacts, surface.info)
        kernel_pressure(contacts, surface.info)
        kernel_scatter(contacts, state, model.info, surface.info)
    else:
        mapped = IntrinsicContacts(link, contacts, model.info, surface.info, options, paths)
        mapped.raw_sources.geom.fill(link.geoms[0].idx)
        mapped.raw_sources.pair.from_numpy(pairs[:, None])
        mapped.raw_sources.witness.from_numpy(np.stack([patch.witness_m for patch in patches])[:, None])
        mapped.raw_sources.free_moment.from_numpy(np.stack([patch.free_moment_nm for patch in patches])[:, None])
        mapped.scatter(contacts, state, model.info, surface.info, options, epsilon, errno)
    return state, contacts, mapped, int(qd_to_numpy(errno, transpose=True)[0])


def native_cloud(model, surface, contacts, mapped, source_ids, patches):
    """Capture every destination sample for one separable production mapping group."""
    if mapped is None:
        shape = (surface.info.positions.shape[0] + 7 * surface.info.shape.shape[0], len(patches), 1)
        samples = StressTractionSamples(
            positions=V_VEC(3, dtype=gs.qd_float, shape=shape),
            normals=V_VEC(3, dtype=gs.qd_float, shape=shape),
            faces=V(dtype=gs.qd_int, shape=shape),
            shape=V_VEC(6, dtype=gs.qd_float, shape=shape),
        )
        forces = V_VEC(3, dtype=gs.qd_float, shape=shape)
        areas = V(dtype=gs.qd_float, shape=shape)
        kernel_pressure_load_samples(samples, forces, areas, contacts, surface.info)
        order = np.arange(len(patches))
    else:
        samples = create_traction_samples(mapped.tractions)
        kernel_traction_samples(samples, mapped.tractions, surface.info, mapped.traction_info)
        shape = mapped.tractions.profile.shape
        forces = V_VEC(3, dtype=gs.qd_float, shape=shape)
        areas = V(dtype=gs.qd_float, shape=shape)
        kernel_intrinsic_load_samples(
            forces, areas, mapped.contacts, mapped.tractions, mapped.traction_info, surface.info
        )
        order = qd_to_numpy(mapped.packing.indices, transpose=True, copy=True)[0, : len(patches)]
    positions = qd_to_numpy(samples.positions, transpose=True, copy=True)[0]
    normals = qd_to_numpy(samples.normals, transpose=True, copy=True)[0]
    actual_force = qd_to_numpy(forces, transpose=True, copy=True)[0]
    area_weights = qd_to_numpy(areas, transpose=True, copy=True)[0]
    sample_faces = qd_to_numpy(samples.faces, transpose=True, copy=True)[0]
    basis = qd_to_numpy(samples.shape, transpose=True, copy=True)[0]
    selected = (area_weights > 0) & (sample_faces >= 0)
    destination = np.broadcast_to(source_ids[order][None], selected.shape)
    friction = np.broadcast_to(
        np.array([patch.specification.friction for patch in patches])[order][None], selected.shape
    )
    return TractionCloud(
        positions[selected],
        normals[selected],
        actual_force[selected],
        area_weights[selected],
        destination[selected],
        friction[selected],
        sample_faces[selected],
        basis[selected],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--cases", nargs="+", default=["opposing", "bending", "asymmetric", "near_slip", "multiple"])
    parser.add_argument("--models", nargs="+", default=["pressure", "componentwise", "pair_union", "body_source_prior"])
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--method", choices=("direct", "pcg"), default="direct")
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--reference-quadrature-level", type=int, default=3)
    parser.add_argument("--reference-solver", choices=("superlu", "amg-cg"), default="superlu")
    parser.add_argument("--native-quadrature", type=int, default=10)
    parser.add_argument("--ports", type=int, default=8)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.mesh) as saved:
        vertices, tetrahedra = saved["vertices"], saved["tetrahedra"]
        boundary = saved["surface_triangles"]
    reference = SkfemElasticReference(vertices, tetrahedra, 1e10, 0.3, 2000)
    if args.reference_solver == "amg-cg":
        operator = (reference.stiffness[reference.free_dofs][:, reference.free_dofs] / reference.young).tocsr()
        modes = reference.rigid_modes[reference.free_dofs]
        reference.factor = AMGPinFactor(operator, modes / np.linalg.norm(modes, axis=0))
    model, surface, paths, entity = None, None, None, None
    if not args.reference_only:
        gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
        mesh_file = args.output / "surface.obj"
        boundary_facets = reference.mesh.boundary_facets()
        oriented_boundary = reference.mesh.facets[:, boundary_facets].T.copy()
        triangles = vertices[oriented_boundary]
        material_centres = vertices[reference.mesh.t[:, reference.mesh.f2t[0, boundary_facets]].T].mean(axis=1)
        inward = (
            np.einsum(
                "fi,fi->f",
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                material_centres - triangles.mean(axis=1),
            )
            > 0
        )
        oriented_boundary[inward, 1:3] = oriented_boundary[inward, 2:0:-1]
        triangles = vertices[oriented_boundary]
        signed_volume = np.einsum("fi,fi->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6
        np.testing.assert_allclose(signed_volume, reference.total_mass / 2000, rtol=1e-12, atol=1e-20)
        (args.output / "surface_geometry_audit.json").write_text(
            json.dumps(
                {
                    "outward_boundary_triangles": len(oriented_boundary),
                    "signed_surface_volume_m3": float(signed_volume),
                    "independent_volume_m3": float(reference.total_mass / 2000),
                },
                indent=2,
            )
            + "\n"
        )
        with mesh_file.open("w") as stream:
            np.savetxt(stream, vertices, fmt="v %.17g %.17g %.17g")
            np.savetxt(stream, oriented_boundary + 1, fmt="f %d %d %d")
        robot = ET.Element("robot", name="physical_validation")
        link = ET.SubElement(robot, "link", name="material")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(f"{value:.17g}" for value in reference.com))
        ET.SubElement(inertial, "mass", value=f"{reference.total_mass:.17g}")
        inertia = reference.gram[3:, 3:]
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{inertia[0, 0]:.17g}",
            iyy=f"{inertia[1, 1]:.17g}",
            izz=f"{inertia[2, 2]:.17g}",
            ixy=f"{inertia[0, 1]:.17g}",
            ixz=f"{inertia[0, 2]:.17g}",
            iyz=f"{inertia[1, 2]:.17g}",
        )
        for name in ("visual", "collision"):
            geometry = ET.SubElement(ET.SubElement(link, name), "geometry")
            ET.SubElement(geometry, "mesh", filename=str(mesh_file.resolve()))
        scene = gs.Scene(
            show_viewer=False,
        )
        entity = scene.add_entity(
            gs.morphs.URDF(
                file=ET.tostring(robot, encoding="unicode"),
                convexify=False,
                align=False,
                preserve_mesh_precision=True,
            ),
        )
        scene.build(n_envs=1)
        options = RigidStressOptions(
            mesh=args.mesh,
            output_mode="full",
            quadrature=args.native_quadrature,
            surface_edge_subdivisions=args.ports,
            method=args.method,
        )
        model = StressModel(options)
        surface = StressSurface(args.native_quadrature, model.info)
        paths = StressSurfacePaths(model.info, args.ports)
    reports = []
    for case in args.cases:
        output = args.output / case
        output.mkdir()
        specifications = physical_grip_patches(args.geometry, case, args.radius)
        patches = tuple(
            assemble_finite_patch(reference, patch, args.reference_quadrature_level) for patch in specifications
        )
        load = sum((patch.load_vector_n for patch in patches), np.zeros(reference.basis.N))
        omega = np.array([0.4, -0.7, 0.9]) if case == "asymmetric" else np.zeros(3)
        np.savez_compressed(
            output / "frozen_inputs.npz",
            points_m=np.stack([p.point_m for p in patches]),
            forces_N=np.stack([p.force_n for p in patches]),
            witnesses_m=np.stack([p.witness_m for p in patches]),
            free_moments_Nm=np.stack([p.free_moment_nm for p in patches]),
            collision_normals=np.stack([p.inward_normal for p in patches]),
            radii_m=np.array([p.specification.radius_m for p in patches]),
            friction=np.array([p.specification.friction for p in patches]),
            omega_rad_s=omega,
        )
        solved = reference.solve(load, omega)
        assert solved.residual_budget_fraction <= 1
        np.savez_compressed(
            output / "reference_field.npz",
            displacement=solved.displacement,
            rhs_N=solved.rhs,
            force_N=load,
            tensor_Pa=solved.tensor_pa,
            vm_Pa=solved.von_mises_pa,
        )
        snapshot = {
            "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
            "geometry": args.geometry,
            "case": case,
            "material": {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000},
            "mass_kg": float(reference.total_mass),
            "com_m": reference.com.tolist(),
            "pose": {"position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
            "omega_rad_s": omega.tolist(),
            "patch_radius_m": args.radius,
            "reference_quadrature_level": args.reference_quadrature_level,
            "reference_peak_Pa": float(solved.peak_pa),
            "reference_peak_position_m": solved.peak_position_m.tolist(),
            "reference_energy_J": float(solved.energy_j),
            "reference_complete_budget_fraction": float(solved.residual_budget_fraction),
            "patches": [],
            "models": [],
        }
        for i_p, patch in enumerate(patches):
            np.savez_compressed(
                output / f"reference_patch{i_p}.npz",
                positions_m=patch.positions_m,
                normals=patch.normals,
                area_weights_m2=patch.area_weights_m2,
                tractions_Pa=patch.tractions_pa,
            )
            snapshot["patches"].append(
                {
                    "name": patch.specification.name,
                    "physical_anchor_m": patch.specification.centre_m.tolist(),
                    "witness_m": patch.witness_m.tolist(),
                    "point_m": patch.point_m.tolist(),
                    "force_N": patch.force_n.tolist(),
                    "integrated_minus_declared_force_N": (
                        np.sum(patch.tractions_pa * patch.area_weights_m2[:, None], axis=0) - patch.force_n
                    ).tolist(),
                    "free_moment_Nm": patch.free_moment_nm.tolist(),
                    "normal": patch.inward_normal.tolist(),
                    "friction": patch.specification.friction,
                    "profile_integral_m2": float(patch.profile_integral_m2),
                    "observables": traction_observables(
                        patch.positions_m,
                        patch.normals,
                        patch.tractions_pa * patch.area_weights_m2[:, None],
                        patch.area_weights_m2,
                        patch.specification.friction,
                        reference.com,
                    ),
                }
            )
        (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        if args.reference_only:
            reports.append(snapshot)
            continue
        coordinates = qd_to_numpy(model.info.vertices, transpose=True, copy=True)
        permutation = reference.native_dof_permutation(coordinates)
        angular_velocity = V_VEC(3, dtype=gs.qd_float, shape=(1,))
        angular_velocity.from_numpy(omega[None])
        for name in args.models:
            started = perf_counter()
            options = RigidStressOptions(
                mesh=args.mesh,
                output_mode="full",
                contact_radius=args.radius,
                contact_model="pressure" if name == "pressure" else "intrinsic",
                intrinsic_support="pair_union" if name == "pair_union" else "contact",
                intrinsic_load_partition="body_source_prior" if name == "body_source_prior" else "componentwise",
                warm_start_tractions=False,
                quadrature=args.native_quadrature,
                surface_edge_subdivisions=args.ports,
                method=args.method,
            )
            row = {"model": name, "native_quadrature": args.native_quadrature, "ports": args.ports}
            if name == "pressure" and any(
                np.linalg.norm(patch.free_moment_nm) > 1e-8 * np.linalg.norm(patch.force_n) * args.radius
                for patch in patches
            ):
                row.update({"accepted": False, "classification": "unsupported_free_contact_moment"})
                snapshot["models"].append(row)
                continue
            pairs = np.arange(len(patches), dtype=gs.np_int)
            if case == "multiple":
                pairs[-1] = 0
            state, contacts, mapped, errno = map_native(
                model, surface, paths, entity.base_link, patches, options, pairs
            )
            status = qd_to_numpy(contacts.status, transpose=True, copy=True)[0]
            row["contact_status"] = status.tolist()
            row["errno"] = errno
            row["selected_anchor_m"] = qd_to_numpy(contacts.center, transpose=True, copy=True)[0].tolist()
            if mapped is not None:
                np.savez_compressed(
                    output / f"{name}_native_workspace.npz",
                    **{
                        item.name: qd_to_numpy(item.value, transpose=True, copy=True)
                        for item in mapped.data("mapped")
                        if isinstance(item.value, (qd.Tensor, qd.Field, qd.Ndarray))
                    },
                )
            native_load = qd_to_numpy(state.force, transpose=True, copy=True)[0].reshape(-1)
            row["mapping_seconds_with_cold_compile"] = perf_counter() - started
            np.savez_compressed(output / f"{name}_mapped_load.npz", force_N=native_load, contact_status=status)
            if errno != 0 or (status != 0).any():
                row.update({"accepted": False, "classification": "native_load_rejected"})
                snapshot["models"].append(row)
                continue
            recovery_started = perf_counter()
            model.recover(angular_velocity, state, options)
            row["accepted"] = bool(qd_to_numpy(state.valid, transpose=True)[0])
            row["recovery_seconds_with_cold_compile"] = perf_counter() - recovery_started
            if not row["accepted"]:
                row["classification"] = "native_elastic_rejected"
                snapshot["models"].append(row)
                continue
            oracle_load = np.empty(reference.basis.N)
            oracle_load[permutation] = native_load
            oracle = reference.solve(oracle_load, omega)
            displacement = np.empty(reference.basis.N)
            displacement[permutation] = qd_to_numpy(state.displacement, transpose=True, copy=True)[0].reshape(-1)
            raw_displacement = displacement.copy()
            displacement = reference.align_rigid_gauge(displacement)
            tensor = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
            vm = qd_to_numpy(state.von_mises, transpose=True, copy=True)[0, ..., 0]
            rhs = qd_to_numpy(state.rhs, transpose=True, copy=True)[0].reshape(-1)
            expected_tensor = oracle.tensor_pa[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
            np.testing.assert_allclose(tensor, expected_tensor, rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(vm, oracle.von_mises_pa, rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(rhs, oracle.rhs[permutation], rtol=1e-10, atol=1e-13)
            residual = np.linalg.norm(reference.stiffness @ displacement - oracle.rhs)
            budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(oracle.rhs))
            assert residual <= budget
            energy = 0.5 * displacement @ (reference.stiffness @ displacement)
            np.testing.assert_allclose(energy, oracle.energy_j, rtol=1e-8, atol=1e-20)
            peak = float(qd_to_numpy(state.peak, transpose=True)[0])
            np.testing.assert_allclose(peak, oracle.peak_pa, rtol=1e-4, atol=1e-3)
            row.update(
                {
                    "classification": "numerically_qualified",
                    "peak_Pa": peak,
                    "peak_position_m": oracle.peak_position_m.tolist(),
                    "energy_J": float(energy),
                    "complete_budget_fraction": float(residual / budget),
                    "tensor_relative_error_vs_continuous_load": float(
                        np.linalg.norm(oracle.tensor_pa - solved.tensor_pa) / np.linalg.norm(solved.tensor_pa)
                    ),
                    "peak_relative_error_vs_continuous_load": float(abs(peak / solved.peak_pa - 1)),
                    "energy_relative_error_vs_continuous_load": float(abs(energy / solved.energy_j - 1)),
                    "full_original_wrench": (reference.rigid_modes.T @ oracle_load).tolist(),
                    "analysis_elapsed_seconds": perf_counter() - started,
                }
            )
            np.savez_compressed(
                output / f"{name}_field.npz",
                displacement=displacement,
                raw_native_displacement=raw_displacement,
                tensor_Pa=oracle.tensor_pa,
                vm_Pa=vm,
                raw_native_tensor_Pa=tensor,
                rhs_N=rhs,
            )
            if name == "pair_union" and int(qd_to_numpy(mapped.packing.maximum, transpose=True)) > 1:
                clouds = []
                replayed = np.zeros_like(native_load)
                for pair in np.unique(pairs):
                    selected = np.flatnonzero(pairs == pair)
                    members = tuple(patches[i_p] for i_p in selected)
                    replay, replay_contacts, replay_mapped, replay_errno = map_native(
                        model, surface, paths, entity.base_link, members, options, pairs[selected], output_mode="max"
                    )
                    assert replay_errno == 0
                    assert (qd_to_numpy(replay_contacts.status, transpose=True) == 0).all()
                    replayed += qd_to_numpy(replay.force, transpose=True, copy=True)[0].reshape(-1)
                    clouds.append(native_cloud(model, surface, replay_contacts, replay_mapped, selected, members))
                np.testing.assert_allclose(replayed, native_load, rtol=1e-8, atol=1e-12)
                cloud = TractionCloud(
                    np.concatenate([c.positions_m for c in clouds]),
                    np.concatenate([c.normals for c in clouds]),
                    np.concatenate([c.forces_n for c in clouds]),
                    np.concatenate([c.areas_m2 for c in clouds]),
                    np.concatenate([c.source for c in clouds]),
                    np.concatenate([c.friction for c in clouds]),
                    np.concatenate([c.faces for c in clouds]),
                    np.concatenate([c.basis for c in clouds]),
                )
            else:
                cloud = native_cloud(model, surface, contacts, mapped, np.arange(len(patches)), patches)
            surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True, copy=True).T
            scattered = np.zeros((len(coordinates), 3))
            np.add.at(
                scattered,
                surface_nodes[cloud.faces].reshape(-1),
                (cloud.basis[:, :, None] * cloud.forces_n[:, None]).reshape(-1, 3),
            )
            np.testing.assert_allclose(scattered.reshape(-1), native_load, rtol=1e-8, atol=1e-12)
            row["sample_scatter_max_difference_N"] = float(abs(scattered.reshape(-1) - native_load).max())
            row["islands"] = []
            for i_p, patch in enumerate(patches):
                selected = cloud.source == i_p
                statistics = traction_observables(
                    cloud.positions_m[selected],
                    cloud.normals[selected],
                    cloud.forces_n[selected],
                    cloud.areas_m2[selected],
                    cloud.friction[selected],
                    reference.com,
                )
                statistics["force_shift_from_prior_N"] = (
                    np.sum(cloud.forces_n[selected], axis=0) - patch.force_n
                ).tolist()
                original_moment = patch.free_moment_nm + np.cross(patch.point_m - reference.com, patch.force_n)
                statistics["moment_shift_from_prior_Nm"] = (
                    np.array(statistics["moment_Nm"]) - original_moment
                ).tolist()
                row["islands"].append(statistics)
            row["maximum_coulomb_violation_Pa"] = max(s["maximum_coulomb_violation_Pa"] for s in row["islands"])
            np.savez_compressed(
                output / f"{name}_traction.npz",
                positions_m=cloud.positions_m,
                normals=cloud.normals,
                forces_N=cloud.forces_n,
                areas_m2=cloud.areas_m2,
                source=cloud.source,
                friction=cloud.friction,
                faces=cloud.faces,
                basis=cloud.basis,
            )
            snapshot["models"].append(row)
            (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            print(json.dumps({"geometry": args.geometry, "case": case, **row}), flush=True)
        reports.append(snapshot)
        (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        (args.output / "summary.json").write_text(json.dumps({"cases": reports}, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps({"cases": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
