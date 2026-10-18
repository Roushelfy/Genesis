"""Compare native finite-footprint reconstruction with existing Franka R/L fields.

R is the qualified nonlinear contact snapshot; L reuses its linear full-inertia
RHS. M changes only the contact part of that RHS to the production finite
footprint. Actual pulled-back application points and enabled moments are kept.
Their small deformation-induced moment difference from reference nodal points
is reported separately, so M/L is not called a pure profile error.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .finite_traction_reference import triangle_closest_points
from .load_model_validation import map_native, native_cloud, traction_observables
from .native_physical_sequence import build_scene
from .neohookean_reference import NeoHookeanBulk
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


@dataclass(frozen=True)
class PacketFootprint:
    radius_m: float
    friction: float


@dataclass(frozen=True)
class PacketContact:
    point_m: np.ndarray
    witness_m: np.ndarray
    force_n: np.ndarray
    free_moment_nm: np.ndarray
    inward_normal: np.ndarray
    specification: PacketFootprint


def linear_quadrature(bulk, displacement):
    gradient = np.einsum("aie,ajeq->ijeq", displacement[bulk.dofs], bulk.gradients, optimize=True).transpose(2, 3, 0, 1)
    return bulk.mu * (gradient + gradient.swapaxes(-1, -2)) + bulk.lam * np.trace(gradient, axis1=-2, axis2=-1)[
        ..., None, None
    ] * np.eye(3)


def weighted_error(first, second, weights):
    assert first.shape == second.shape == (*weights.shape, 3, 3)
    absolute = float(np.sqrt(np.sum(weights[..., None, None] * (first - second) ** 2)))
    norm = float(np.sqrt(np.sum(weights[..., None, None] * second**2)))
    return {
        "absolute_L2_Pa_m1p5": absolute,
        "reference_L2_Pa_m1p5": norm,
        "relative_L2": absolute / norm if norm else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--radii", nargs="+", type=float, default=[0.003, 0.006])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["pressure", "componentwise", "pair_union", "body_source_prior"],
        default=["pressure", "componentwise", "pair_union", "body_source_prior"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.packets)
    assert int((args.packets / "returncode.txt").read_text()) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def record(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    summary = json.loads(record(args.packets / "results/summary.json").read_text())
    mesh = record(args.packets / "results/body_mesh.npz")
    with np.load(mesh) as saved:
        reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    bulk = NeoHookeanBulk(reference)
    assert np.all(bulk.basis.dx > 0)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    _scene, link = build_scene(reference, args.output)
    base = RigidStressOptions(mesh=mesh, method="pcg", output_mode="full", max_iterations=20000)
    model = StressModel(base)
    surface = StressSurface(10, model.info)
    paths = StressSurfacePaths(model.info, 8)
    permutation = reference.native_dof_permutation(qd_to_numpy(model.info.vertices, transpose=True, copy=True))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    omega.fill(0)
    triangles = reference.mesh.p.T[reference.mesh.facets[:, reference.mesh.boundary_facets()].T]
    rows = []
    for snapshot in summary["rows"]:
        step = snapshot["step"]
        packets = json.loads(record(args.packets / f"results/step{step:05d}-packets.json").read_text())
        with np.load(record(args.packets / f"results/step{step:05d}.npz")) as saved:
            resolved_force = saved["resolved_contact_N"].copy()
            full_rhs = saved["resolved_full_inertia_rhs_N"].copy()
            linear_u = saved["resolved_linear_displacement_m"].copy()
            nonlinear = saved["resolved_nonlinear_quadrature_tensor_Pa"].copy()
        linear = linear_quadrature(bulk, linear_u)
        _linear_tensor, linear_vm = reference.corner_stress(linear_u)
        frame_error = weighted_error(linear, nonlinear, bulk.basis.dx)
        pairs = np.array([p["source"] for p in packets], dtype=gs.np_int)
        for radius in args.radii:
            contacts = []
            distances = []
            for packet in packets:
                point = np.array(packet["point_m"])
                nearest, distance = triangle_closest_points(triangles, point)
                witness = nearest[distance.argmin()]
                distances.append(float(distance.min()))
                contacts.append(
                    PacketContact(
                        point,
                        witness,
                        np.array(packet["force_N"]),
                        np.array(packet["free_moment_Nm"]),
                        np.array(packet["inward_normal"]),
                        PacketFootprint(radius, packet["friction"]),
                    )
                )
            for name in args.models:
                row = {
                    "step": step,
                    "time_s": snapshot["time_s"],
                    "radius_m": radius,
                    "model": name,
                    "contact_count": len(contacts),
                    "maximum_point_to_reference_boundary_distance_m": max(distances),
                    "current_minus_reference_application_moment_Nm": snapshot["current_minus_reference_wrench_Nm"],
                    "L_vs_R": frame_error,
                    "L_raw_global_peak_Pa": float(linear_vm.max()),
                }
                rows.append(row)
                label = f"step{step:05d}-r{round(radius * 1e6)}-{name}"
                if name == "pressure" and any(
                    np.linalg.norm(p.free_moment_nm) > 1e-8 * np.linalg.norm(p.force_n) * radius for p in contacts
                ):
                    row["classification"] = "unsupported_enabled_free_moment"
                    continue
                options = base.model_copy(
                    update={
                        "quadrature": 10,
                        "contact_radius": radius,
                        "surface_edge_subdivisions": 8,
                        "contact_model": "pressure" if name == "pressure" else "intrinsic",
                        "intrinsic_support": "pair_union" if name == "pair_union" else "contact",
                        "intrinsic_load_partition": "body_source_prior"
                        if name == "body_source_prior"
                        else "componentwise",
                    }
                )
                state, contact_data, mapped, errno = map_native(model, surface, paths, link, contacts, options, pairs)
                status = qd_to_numpy(contact_data.status, transpose=True, copy=True)[0]
                row.update(
                    contact_status=status.tolist(),
                    errno=errno,
                    reported_force_error_N=qd_to_numpy(contact_data.force_error, transpose=True, copy=True)[0].tolist(),
                    reported_moment_error_Nm=qd_to_numpy(contact_data.moment_error, transpose=True, copy=True)[
                        0
                    ].tolist(),
                )
                if errno or status.any():
                    row["classification"] = "native_load_rejected"
                    print(json.dumps({"case": label, "status": status.tolist()}), flush=True)
                    continue
                mapped_force = np.empty(reference.basis.N)
                mapped_force[permutation] = qd_to_numpy(state.force, transpose=True, copy=True)[0].reshape(-1)
                cloud = native_cloud(model, surface, contact_data, mapped, np.arange(len(contacts)), contacts)
                surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True, copy=True).T
                replay = np.zeros((len(permutation) // 3, 3))
                for node in range(6):
                    np.add.at(replay, surface_nodes[cloud.faces, node], cloud.basis[:, node, None] * cloud.forces_n)
                np.testing.assert_allclose(replay.reshape(-1), mapped_force[permutation], rtol=1e-8, atol=1e-12)
                # The qualified resolved inertia/support term is kept identical.
                reconstructed_rhs = full_rhs + mapped_force - resolved_force
                state.force.from_numpy(reconstructed_rhs[permutation].reshape(-1, 1, 3))
                model.recover(omega, state, options)
                if not bool(qd_to_numpy(state.valid, transpose=True)[0]):
                    row["classification"] = "native_elastic_rejected"
                    continue
                displacement = np.empty(reference.basis.N)
                displacement[permutation] = qd_to_numpy(state.displacement, transpose=True, copy=True)[0].reshape(-1)
                tensor, vm = reference.corner_stress(displacement)
                raw = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
                np.testing.assert_allclose(
                    raw, tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]], rtol=1e-9, atol=1e-5
                )
                peak = float(qd_to_numpy(state.peak)[0])
                np.testing.assert_allclose(peak, vm.max(), rtol=1e-4, atol=1e-3)
                balanced = reference.balanced_rhs(reconstructed_rhs, np.zeros(3))
                residual = float(np.linalg.norm(reference.stiffness @ displacement - balanced))
                budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(balanced))
                assert residual <= budget
                reconstructed = linear_quadrature(bulk, displacement)
                location = np.unravel_index(vm.argmax(), vm.shape)
                row.update(
                    classification="accepted_conditional_snapshot",
                    M_vs_L=weighted_error(reconstructed, linear, bulk.basis.dx),
                    M_vs_R=weighted_error(reconstructed, nonlinear, bulk.basis.dx),
                    M_raw_global_peak_Pa=peak,
                    M_raw_global_peak_reference_position_m=reference.mesh.p.T[
                        reference.mesh.t.T[location[0], location[1]]
                    ].tolist(),
                    native_complete_budget_fraction=residual / budget,
                    inertia_rhs_change_from_reference_contact_N=float(np.linalg.norm(reconstructed_rhs - full_rhs)),
                    additional_balance_projection_N=float(np.linalg.norm(balanced - reconstructed_rhs)),
                    packet_redistribution=[],
                )
                for index, packet in enumerate(contacts):
                    selected = cloud.source == index
                    if not selected.any():
                        row["packet_redistribution"].append(
                            {"packet": index, "physical_source": int(pairs[index]), "loaded": False}
                        )
                        continue
                    observation = traction_observables(
                        cloud.positions_m[selected],
                        cloud.normals[selected],
                        cloud.forces_n[selected],
                        cloud.areas_m2[selected],
                        cloud.friction[selected],
                        reference.com,
                    )
                    target_moment = np.cross(packet.point_m - reference.com, packet.force_n) + packet.free_moment_nm
                    observation.update(
                        packet=index,
                        physical_source=int(pairs[index]),
                        force_shift_N=(np.array(observation["resultant_N"]) - packet.force_n).tolist(),
                        moment_shift_Nm=(np.array(observation["moment_Nm"]) - target_moment).tolist(),
                    )
                    row["packet_redistribution"].append(observation)
                np.savez_compressed(
                    args.output / f"{label}.npz",
                    native_tensor_Pa=raw,
                    displacement_m=displacement,
                    mapped_contact_force_N=mapped_force,
                    full_inertia_rhs_N=reconstructed_rhs,
                    corner_tensor_Pa=tensor,
                    corner_vm_Pa=vm,
                    quadrature_tensor_Pa=reconstructed,
                    traction_positions_m=cloud.positions_m,
                    traction_normals=cloud.normals,
                    traction_forces_N=cloud.forces_n,
                    traction_areas_m2=cloud.areas_m2,
                    packet=cloud.source,
                    physical_source=pairs[cloud.source],
                    faces=cloud.faces,
                )
                print(json.dumps({"case": label, "M_L": row["M_vs_L"]["relative_L2"], "peak_Pa": peak}), flush=True)
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "packets": str(args.packets),
                "radii_m": args.radii,
                "models": args.models,
                "tetrahedra": len(reference.mesh.t.T),
                "rows": rows,
                "input_sha256": inputs,
                "goal_completion": False,
                "limitations": "Four compact snapshots of the original480-tet numerical reference. Radius/profile are physical model assumptions, not refinement or observed contact area. R/L geometry/contact uncertainty remains. Full-inertia forcing is held fixed while pulled-back actual contact points/moments replace reference nodal loading; measured moment-coordinate differences and balance projection are explicit. No universal accuracy or fracture threshold is claimed. Unsupported inputs and native rejection statuses are retained.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
