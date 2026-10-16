"""Audit changing-contact histories and warmed native stress cost against independent FEM."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
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
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy

from .finite_traction_reference import assemble_finite_patch, physical_grip_patches
from .load_model_validation import TractionCloud, map_native, native_cloud, traction_observables
from .profile_reference_sensitivity import tensor_l2
from .skfem_reference import SkfemElasticReference


def build_scene(reference, directory):
    mesh_file = directory / "surface.obj"
    boundary = reference.mesh.boundary_facets()
    faces = reference.mesh.facets[:, boundary].T.copy()
    vertices = reference.mesh.p.T
    triangles = vertices[faces]
    owners = vertices[reference.mesh.t[:, reference.mesh.f2t[0, boundary]].T].mean(axis=1)
    inward = (
        np.einsum(
            "fi,fi->f",
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            owners - triangles.mean(axis=1),
        )
        > 0
    )
    faces[inward, 1:3] = faces[inward, 2:0:-1]
    with mesh_file.open("w") as stream:
        np.savetxt(stream, vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(stream, faces + 1, fmt="f %d %d %d")
    robot = ET.Element("robot", name="physical_sequence")
    link = ET.SubElement(robot, "link", name="material")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(str, reference.com)))
    ET.SubElement(inertial, "mass", value=str(reference.total_mass))
    inertia = reference.gram[3:, 3:]
    ET.SubElement(
        inertial,
        "inertia",
        **{
            name: str(inertia[i, j])
            for name, i, j in (("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2), ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2))
        },
    )
    for name in ("visual", "collision"):
        ET.SubElement(ET.SubElement(ET.SubElement(link, name), "geometry"), "mesh", filename=str(mesh_file.resolve()))
    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        gs.morphs.URDF(
            file=ET.tostring(robot, encoding="unicode"), convexify=False, align=False, preserve_mesh_precision=True
        )
    )
    scene.build(n_envs=1)
    return scene, entity.base_link


class ReusedMapper:
    def __init__(self, model, surface, paths, link, options):
        self.model, self.surface, self.link, self.options = model, surface, link, options
        self.contacts = create_contacts(3, 1, options.contact_radius)
        self.contacts.frame_position.fill(0)
        self.contacts.frame_quaternion.from_numpy(np.array([[1.0, 0.0, 0.0, 0.0]]))
        self.state = model.create_state(1, "full")
        self.errno = V(dtype=gs.qd_int, shape=(1,))
        self.omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
        self.mapped = None
        if options.contact_model != "pressure":
            self.mapped = IntrinsicContacts(link, self.contacts, model.info, surface.info, options, paths)
            self.mapped.raw_sources.geom.fill(link.geoms[0].idx)

    def run(self, patches, pairs, omega):
        count = len(patches)
        valid = np.zeros((3, 1), dtype=bool)
        valid[:count] = True
        self.contacts.valid.from_numpy(valid)
        self.contacts.status.fill(0)
        for name, values in (
            ("position", [p.point_m for p in patches]),
            ("force", [p.force_n for p in patches]),
            ("normal", [p.inward_normal for p in patches]),
        ):
            array = np.zeros((3, 1, 3))
            if count:
                array[:count, 0] = values
            getattr(self.contacts, name).from_numpy(array)
        radii, friction = np.full((3, 1), self.options.contact_radius), np.zeros((3, 1))
        if count:
            radii[:count, 0] = [p.specification.radius_m for p in patches]
            friction[:count, 0] = [p.specification.friction for p in patches]
        self.contacts.radius.from_numpy(radii)
        self.contacts.friction.from_numpy(friction)
        kernel_begin_step(self.state)
        self.errno.fill(0)
        epsilon = float(np.finfo(gs.np_float).eps)
        if self.mapped is None:
            kernel_anchor(epsilon, self.contacts, self.surface.info)
            kernel_pressure(self.contacts, self.surface.info)
            kernel_scatter(self.contacts, self.state, self.model.info, self.surface.info)
        else:
            group_ids = np.zeros((3, 1), dtype=gs.np_int)
            witnesses, moments = np.zeros((3, 1, 3)), np.zeros((3, 1, 3))
            if count:
                group_ids[:count, 0] = pairs
                witnesses[:count, 0] = [p.witness_m for p in patches]
                moments[:count, 0] = [p.free_moment_nm for p in patches]
            self.mapped.raw_sources.pair.from_numpy(group_ids)
            self.mapped.raw_sources.witness.from_numpy(witnesses)
            self.mapped.raw_sources.free_moment.from_numpy(moments)
            self.mapped.scatter(
                self.contacts, self.state, self.model.info, self.surface.info, self.options, epsilon, self.errno
            )
        self.omega.from_numpy(omega[None])
        status = qd_to_numpy(self.contacts.status, transpose=True)[0]
        if int(qd_to_numpy(self.errno, transpose=True)[0]) == 0 and not np.any(status[:count]):
            self.model.recover(self.omega, self.state, self.options)
        qd.sync()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--ports", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("pressure", "componentwise", "pair_union", "body_source_prior"),
        default=("pressure", "componentwise", "pair_union", "body_source_prior"),
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ports < 1 or len(args.models) != len(set(args.models)):
        parser.error("Ports must be positive and each model may be selected only once.")
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "mesh": str(args.mesh.resolve()),
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "geometry": args.geometry,
        "backend": args.backend,
        "surface_edge_subdivisions": args.ports,
        "models": list(args.models),
        "quadrature": 10,
        "max_iterations": args.max_iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    (args.output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    with np.load(args.mesh) as saved:
        vertices, tetrahedra = saved["vertices"], saved["tetrahedra"]
    reference = SkfemElasticReference(vertices, tetrahedra, 1e10, 0.3, 2000)
    volumes = abs(np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])) / 6
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    _scene, link = build_scene(reference, args.output)
    base_options = RigidStressOptions(
        mesh=args.mesh,
        method="pcg",
        output_mode="full",
        quadrature=10,
        surface_edge_subdivisions=args.ports,
        max_iterations=args.max_iterations,
    )
    model = StressModel(base_options)
    surface = StressSurface(10, model.info)
    paths = None
    coordinates = qd_to_numpy(model.info.vertices, transpose=True, copy=True)
    permutation = reference.native_dof_permutation(coordinates)
    bank = {}
    for case in ("opposing", "multiple", "near_slip", "asymmetric"):
        bank[case] = tuple(assemble_finite_patch(reference, p, 3) for p in physical_grip_patches(args.geometry, case))
    scaled = tuple(
        replace(
            p,
            force_n=0.8 * p.force_n,
            free_moment_nm=0.8 * p.free_moment_nm,
            load_vector_n=0.8 * p.load_vector_n,
            tractions_pa=0.8 * p.tractions_pa,
        )
        for p in bank["opposing"]
    )
    multiple = bank["multiple"]
    sequence = [
        ("empty", (), [], np.zeros(3)),
        ("opposing", bank["opposing"], [0, 1], np.zeros(3)),
        ("scaled", scaled, [0, 1], np.zeros(3)),
        ("birth", multiple, [0, 1, 0], np.zeros(3)),
        ("reordered", tuple(multiple[i] for i in (2, 0, 1)), [0, 0, 1], np.zeros(3)),
        ("death", bank["opposing"][:1], [0], np.zeros(3)),
        ("near_slip", bank["near_slip"], [0, 1], np.zeros(3)),
        ("asymmetric", bank["asymmetric"], [0, 1], np.array([0.4, -0.7, 0.9])),
        ("return", bank["opposing"], [0, 1], np.zeros(3)),
        ("release", (), [], np.zeros(3)),
    ]
    references = {}
    for label, patches, pairs, omega in sequence:
        np.savez_compressed(
            args.output / f"{label}_inputs.npz",
            points_m=np.array([p.point_m for p in patches]).reshape(-1, 3),
            forces_N=np.array([p.force_n for p in patches]).reshape(-1, 3),
            free_moments_Nm=np.array([p.free_moment_nm for p in patches]).reshape(-1, 3),
            witnesses_m=np.array([p.witness_m for p in patches]).reshape(-1, 3),
            normals=np.array([p.inward_normal for p in patches]).reshape(-1, 3),
            radii_m=np.array([p.specification.radius_m for p in patches]),
            friction=np.array([p.specification.friction for p in patches]),
            pairs=np.array(pairs),
            omega_rad_s=omega,
        )
        load = sum((p.load_vector_n for p in patches), np.zeros(reference.basis.N))
        references[label] = reference.solve(load, omega)
        assert references[label].residual_budget_fraction <= 1
        np.savez_compressed(
            args.output / f"{label}_reference.npz",
            displacement=references[label].displacement,
            tensor_Pa=references[label].tensor_pa,
            vm_Pa=references[label].von_mises_pa,
            rhs_N=references[label].rhs,
        )
    rows, costs = [], []
    for name in args.models:
        if name != "pressure" and paths is None:
            started = perf_counter()
            paths = StressSurfacePaths(model.info, args.ports)
            n_ports = paths.info.positions.shape[0]
            (args.output / "path_initialization.json").write_text(
                json.dumps(
                    {
                        "seconds": perf_counter() - started,
                        "ports": n_ports,
                        "dense_distance_bytes": 8 * n_ports * n_ports,
                    },
                    indent=2,
                )
                + "\n"
            )
        options = RigidStressOptions(
            mesh=args.mesh,
            method="pcg",
            output_mode="full",
            quadrature=10,
            surface_edge_subdivisions=args.ports,
            max_iterations=args.max_iterations,
            contact_block_size=3,
            warm_start_tractions=True,
            contact_model="pressure" if name == "pressure" else "intrinsic",
            intrinsic_support="pair_union" if name == "pair_union" else "contact",
            intrinsic_load_partition="body_source_prior" if name == "body_source_prior" else "componentwise",
        )
        mapper = ReusedMapper(model, surface, paths, link, options)
        for index, (label, patches, pairs, omega) in enumerate(sequence):
            row = {"model": name, "step": index, "label": label, "contact_count": len(patches)}
            reference_cones = [
                traction_observables(
                    p.positions_m,
                    p.normals,
                    p.tractions_pa * p.area_weights_m2[:, None],
                    p.area_weights_m2,
                    p.specification.friction,
                    reference.com,
                )
                for p in patches
            ]
            row["reference_traction_observables"] = reference_cones
            row["reference_traction_admissible"] = all(
                c["maximum_coulomb_violation_Pa"] <= 1e-10 * max(c["maximum_pressure_Pa"], c["maximum_shear_Pa"], 1)
                for c in reference_cones
            )
            if name == "pressure" and any(
                np.linalg.norm(p.free_moment_nm) > 1e-8 * np.linalg.norm(p.force_n) * options.contact_radius
                for p in patches
            ):
                row["classification"] = "unsupported_free_contact_moment"
                rows.append(row)
                continue
            started = perf_counter()
            mapper.run(patches, pairs, omega)
            row["stress_evaluation_seconds_including_input_upload_and_sync"] = perf_counter() - started
            status = qd_to_numpy(mapper.contacts.status, transpose=True, copy=True)[0]
            errno = int(qd_to_numpy(mapper.errno, transpose=True)[0])
            valid = bool(qd_to_numpy(mapper.state.valid, transpose=True)[0])
            row.update({"contact_status": status.tolist(), "errno": errno, "native_valid": valid})
            captured = {
                key: qd_to_numpy(getattr(mapper.state, key), transpose=True, copy=True)
                for key in ("force", "rhs", "displacement", "stress_tensor", "von_mises", "valid", "iterations", "peak")
            }
            np.savez_compressed(args.output / f"{name}_{index}_native_state.npz", **captured)
            (args.output / f"{name}_{index}_snapshot.json").write_text(json.dumps(row, indent=2) + "\n")
            if errno or np.any(status[: len(patches)]):
                row["classification"] = "native_load_rejected"
            elif not valid:
                row["classification"] = "native_elastic_rejected"
            else:
                native_load = qd_to_numpy(mapper.state.force, transpose=True, copy=True)[0].reshape(-1)
                native_u = qd_to_numpy(mapper.state.displacement, transpose=True, copy=True)[0].reshape(-1)
                independent_load, independent_u = np.empty(reference.basis.N), np.empty(reference.basis.N)
                independent_load[permutation], independent_u[permutation] = native_load, native_u
                rhs = reference.balanced_rhs(independent_load, omega)
                residual = float(np.linalg.norm(reference.stiffness @ independent_u - rhs))
                budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
                assert residual <= budget, (name, label, residual, budget)
                tensor, vm = reference.corner_stress(independent_u)
                peak = float(qd_to_numpy(mapper.state.peak, transpose=True)[0])
                np.testing.assert_allclose(peak, vm.max(), rtol=1e-4, atol=1e-3)
                expected = references[label]
                row.update(
                    {
                        "classification": "numerically_qualified",
                        "complete_budget_fraction": residual / budget,
                        "raw_global_peak_Pa": peak,
                        "native_iterations": int(qd_to_numpy(mapper.state.iterations, transpose=True)[0]),
                        "volume_l2_tensor_difference_from_finite_reference": tensor_l2(
                            tensor - expected.tensor_pa, volumes
                        ),
                        "volume_l2_tensor_reference_norm": tensor_l2(expected.tensor_pa, volumes),
                    }
                )
                if not patches:
                    assert peak <= 1e-3 and np.linalg.norm(native_load) <= 1e-13
                else:
                    fresh, _, _, fresh_errno = map_native(
                        model, surface, paths, link, patches, options, np.array(pairs, dtype=gs.np_int)
                    )
                    fresh_omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
                    fresh_omega.from_numpy(omega[None])
                    model.recover(fresh_omega, fresh, options)
                    assert fresh_errno == 0 and bool(qd_to_numpy(fresh.valid, transpose=True)[0])
                    fresh_u = np.empty(reference.basis.N)
                    fresh_u[permutation] = qd_to_numpy(fresh.displacement, transpose=True, copy=True)[0].reshape(-1)
                    np.savez_compressed(
                        args.output / f"{name}_{index}_fresh_state.npz",
                        displacement=fresh_u,
                        force_N=qd_to_numpy(fresh.force, transpose=True, copy=True)[0].reshape(-1),
                    )
                    fresh_tensor, _ = reference.corner_stress(fresh_u)
                    difference = tensor_l2(tensor - fresh_tensor, volumes) / tensor_l2(fresh_tensor, volumes)
                    assert difference < 1e-4, (name, label, difference)
                    row["history_vs_fresh_tensor_relative_difference"] = difference
                np.savez_compressed(
                    args.output / f"{name}_{index}_field.npz",
                    force_N=independent_load,
                    displacement=independent_u,
                    tensor_Pa=tensor,
                    vm_Pa=vm,
                )
                if patches:
                    if name == "pair_union":
                        clouds, replayed = [], np.zeros_like(native_load)
                        pairs_array = np.array(pairs, dtype=gs.np_int)
                        for pair in np.unique(pairs_array):
                            selected = np.flatnonzero(pairs_array == pair)
                            members = tuple(patches[i] for i in selected)
                            replay, replay_contacts, replay_mapped, replay_errno = map_native(
                                model, surface, paths, link, members, options, pairs_array[selected]
                            )
                            assert replay_errno == 0 and not np.any(qd_to_numpy(replay_contacts.status, transpose=True))
                            replayed += qd_to_numpy(replay.force, transpose=True, copy=True)[0].reshape(-1)
                            clouds.append(
                                native_cloud(model, surface, replay_contacts, replay_mapped, selected, members)
                            )
                        replay_difference = float(np.linalg.norm(replayed - native_load))
                        replay_budget = 1e-8 * sum(np.linalg.norm(p.force_n) for p in patches) + 1e-13
                        row["group_replay_nodal_difference_N"] = replay_difference
                        row["group_replay_budget_N"] = replay_budget
                        np.savez_compressed(
                            args.output / f"{name}_{index}_group_replay.npz",
                            actual_force_native_order_N=native_load,
                            replayed_force_native_order_N=replayed,
                        )
                        assert replay_difference <= replay_budget, (name, label, replay_difference, replay_budget)
                        cloud = TractionCloud(
                            **{
                                key: np.concatenate([getattr(c, key) for c in clouds])
                                for key in TractionCloud.__dataclass_fields__
                            }
                        )
                    else:
                        cloud = native_cloud(
                            model, surface, mapper.contacts, mapper.mapped, np.arange(len(patches)), patches
                        )
                    np.savez_compressed(
                        args.output / f"{name}_{index}_traction.npz",
                        positions_m=cloud.positions_m,
                        normals=cloud.normals,
                        forces_N=cloud.forces_n,
                        areas_m2=cloud.areas_m2,
                        source=cloud.source,
                        friction=cloud.friction,
                        faces=cloud.faces,
                        basis=cloud.basis,
                    )
                    row["source_charts"] = []
                    for source_index, patch in enumerate(patches):
                        selected = cloud.source == source_index
                        observed = traction_observables(
                            cloud.positions_m[selected],
                            cloud.normals[selected],
                            cloud.forces_n[selected],
                            cloud.areas_m2[selected],
                            cloud.friction[selected],
                            reference.com,
                        )
                        observed["force_shift_from_prior_N"] = (
                            np.array(observed["resultant_N"]) - patch.force_n
                        ).tolist()
                        original_moment = patch.free_moment_nm + np.cross(patch.point_m - reference.com, patch.force_n)
                        observed["moment_shift_from_prior_Nm"] = (
                            np.array(observed["moment_Nm"]) - original_moment
                        ).tolist()
                        row["source_charts"].append(observed)
            rows.append(row)
            (args.output / f"{name}_{index}_snapshot.json").write_text(json.dumps(row, indent=2) + "\n")
        patches = bank["opposing"]
        for _ in range(args.warmup):
            mapper.run(patches, [0, 1], np.zeros(3))
        samples = []
        for _ in range(args.repeats):
            started = perf_counter()
            mapper.run(patches, [0, 1], np.zeros(3))
            elapsed = perf_counter() - started
            assert (
                bool(qd_to_numpy(mapper.state.valid, transpose=True)[0])
                and int(qd_to_numpy(mapper.errno, transpose=True)[0]) == 0
            )
            assert not np.any(qd_to_numpy(mapper.contacts.status, transpose=True)[0, :2])
            samples.append(elapsed)
        costs.append(
            {
                "configuration": configuration,
                "model": name,
                "scope": "One full-field stress evaluation, input uploads, load mapping, recovery, history, raw global peak and device sync; validation downloads and FEM excluded",
                "warmup": args.warmup,
                "seconds": samples,
                "minimum_seconds": min(samples),
                "maximum_seconds": max(samples),
                "median_seconds": float(np.median(samples)),
            }
        )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "geometry": args.geometry,
                "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                "tetrahedra": len(tetrahedra),
                "max_iterations": args.max_iterations,
                "rows": rows,
                "warmed_costs": costs,
                "limitations": "Single-environment stress scope; rigid dynamics/policy and FEM are outside timing. Prescribed traction references can be locally inadmissible and require separate interpretation. Full physical/contact acceptance remains required.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "qualified": sum(r["classification"] == "numerically_qualified" for r in rows),
                "costs": costs,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
