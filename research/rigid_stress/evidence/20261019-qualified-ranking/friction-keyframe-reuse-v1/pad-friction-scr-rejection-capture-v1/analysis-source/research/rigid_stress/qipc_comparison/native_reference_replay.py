"""Replay resolved or compressed historical loads through the actual native Quadrants stress kernels."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.contact import kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import CommonColumns, field_error, tangent, voigt_tensor


@dataclass
class ContactViews:
    position: torch.Tensor
    force: torch.Tensor
    normal: torch.Tensor
    friction: torch.Tensor
    radius: torch.Tensor
    valid: torch.Tensor
    frame_position: torch.Tensor
    frame_quaternion: torch.Tensor

    @classmethod
    def allocate(cls, contacts):
        return cls(
            qd_to_torch(contacts.position, transpose=True, copy=False),
            qd_to_torch(contacts.force, transpose=True, copy=False),
            qd_to_torch(contacts.normal, transpose=True, copy=False),
            qd_to_torch(contacts.friction, transpose=True, copy=False),
            qd_to_torch(contacts.radius, transpose=True, copy=False),
            qd_to_torch(contacts.valid, transpose=True, copy=False),
            qd_to_torch(contacts.frame_position, transpose=True, copy=False),
            qd_to_torch(contacts.frame_quaternion, transpose=True, copy=False),
        )

    def load(self, packet):
        count = len(packet["compressed_forces_material_n"])
        assert count <= self.valid.shape[1]
        self.valid.zero_()
        self.position.zero_()
        self.force.zero_()
        self.normal.zero_()
        self.position[0, :count] = torch.as_tensor(packet["compressed_positions_material_m"], device=gs.device)
        self.force[0, :count] = torch.as_tensor(packet["compressed_forces_material_n"], device=gs.device)
        self.normal[0, :count] = torch.as_tensor(packet["compressed_normals"], device=gs.device)
        self.friction.fill_(packet["friction"].item())
        self.radius.fill_(packet["radius_m"].item())
        self.valid[0, :count] = True
        self.frame_position.zero_()
        self.frame_quaternion.zero_()
        self.frame_quaternion[..., 0] = 1
        return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--route", choices=("core", "dpr", "scr"), required=True)
    parser.add_argument("--body-loads", choices=("measured", "rigid"), default="measured")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.route != "core" or args.body_loads == "measured"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    packet_receipt = json.loads((args.packets / "result.json").read_text())
    oracle_receipt = json.loads((args.oracle / "result.json").read_text())
    assert packet_receipt["reference"]["sha256"] == oracle_receipt["reference"]["sha256"]
    reference = Path(packet_receipt["reference"]["path"]).parent
    inputs = json.loads((reference / "inputs.json").read_text())
    with np.load(args.oracle / "operator.npz") as saved:
        coordinates, component_dofs = saved["p2_coordinates_m"], saved["p2_component_dofs"]
        transfer = sparse.csr_matrix(
            (saved["transfer_data"], saved["transfer_indices"], saved["transfer_indptr"]),
            shape=tuple(saved["transfer_shape"]),
        )
    with np.load(args.columns) as saved:
        columns = CommonColumns(**{key: saved[key] for key in saved.files})
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    workload = FrankaEgg(
        1,
        level=1,
        conditions=Path(inputs["conditions"]["path"]),
        output_mode="full",
        native_controller=False,
        contact_model="intrinsic" if args.route == "scr" else "pressure",
        intrinsic_load_partition="body_source_prior",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        fused_pipeline=False,
    )
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    assert file_id(workload.link.stress_options.mesh)["sha256"] == oracle_receipt["native_mesh"]["sha256"]
    native_coordinates = qd_to_numpy(entry.model.info.vertices, transpose=True)
    # Static connectivity has a corner axis, with no environment axis.
    native_elements = qd_to_numpy(entry.model.info.elements, transpose=True).T
    assert native_elements.shape == (entry.state.stress_tensor.shape[0], 10)
    distance, permutation = cKDTree(coordinates).query(native_coordinates)
    assert distance.max() < 1e-14 and len(np.unique(permutation)) == len(coordinates)
    np.savez_compressed(
        args.output / "rest.npz",
        coordinates_m=native_coordinates,
        p2_elements=native_elements,
        independent_p2_node_permutation=permutation,
    )
    force_view = qd_to_torch(entry.state.force, transpose=True, copy=False)
    omega_view = qd_to_torch(entry.omega, transpose=True, copy=False)
    errno_view = qd_to_torch(workload.scene.rigid_solver._errno, transpose=True, copy=False)
    contacts = ContactViews.allocate(entry.contacts)
    epsilon = np.finfo(gs.np_float).eps.item()
    invalid = 0
    executed = 0
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "genesis_import": gs.__file__,
            "route": args.route,
            "body_loads": args.body_loads,
            "options": entry.options.model_dump(mode="json"),
            "packets": file_id(args.packets / "result.json"),
            "oracle": file_id(args.oracle / "result.json"),
            "columns": file_id(args.columns),
            "source_inference": "Point-only reconstruction uses the compressed physical point as an inferred boundary witness, original source uncertainty/projection limits, and provider-local support-component ids. No actual native collider observation or exact material witness is asserted.",
        },
    )
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for path in sorted((args.packets / "frames").glob("step*.npz")):
            with np.load(path) as saved:
                packet = {key: saved[key] for key in saved.files}
            with np.load(args.oracle / "frames" / path.name) as saved:
                expected_cpu = saved["p2_linear_corner_stress_pa"]
                expected_reference = tangent(
                    saved["reference_corotated_cauchy_pa"][columns.reference_elements], columns
                )
            errno_view.zero_()
            kernel_begin_step(entry.state)
            force_view.zero_()
            omega_view.zero_()
            count = 0
            if args.route == "core":
                mapped = transfer @ (
                    packet["contact_forces_material_n"] - packet["inertial_reaction_material_n"]
                ).reshape(-1)
                force_view[0] = torch.as_tensor(mapped[component_dofs][permutation], device=gs.device)
            else:
                count = contacts.load(packet)
                if entry.intrinsic is not None:
                    sources = entry.intrinsic.raw_sources
                    sources.geom.fill(workload.link.geoms[0].idx)
                    sources.method.fill(0)
                    sources.vertex.fill(-1)
                    sources.witness_face.fill(-1)
                    sources.has_vertex_ambiguity.fill(False)
                    sources.free_moment.fill(0)
                    pair_view = qd_to_torch(sources.pair, transpose=True, copy=False)
                    witness_view = qd_to_torch(sources.witness, transpose=True, copy=False)
                    pair_view.fill_(-1)
                    witness_view.fill_(float("nan"))
                    pair_view[0, :count] = torch.as_tensor(
                        packet["support_components"], dtype=gs.tc_int, device=gs.device
                    )
                    witness_view[0, :count] = contacts.position[0, :count]
                    entry.intrinsic.scatter(
                        entry.contacts,
                        entry.state,
                        entry.model.info,
                        entry.surface.info,
                        entry.options,
                        epsilon,
                        workload.scene.rigid_solver._errno,
                    )
                else:
                    kernel_anchor(epsilon, entry.contacts, entry.surface.info)
                    kernel_pressure(entry.contacts, entry.surface.info, entry.options.cooperative_pressure)
                    kernel_scatter(
                        entry.contacts,
                        entry.state,
                        entry.model.info,
                        entry.surface.info,
                        entry.options.cooperative_scatter,
                        entry.options.cached_face_bounds,
                    )
                if args.body_loads == "measured":
                    mapped = transfer @ packet["inertial_reaction_material_n"].reshape(-1)
                    force_view[0] -= torch.as_tensor(mapped[component_dofs][permutation], device=gs.device)
                else:
                    omega_view[0] = torch.as_tensor(packet["omega_material_rad_per_s"], device=gs.device)
            entry.model.recover(entry.omega, entry.state, surface_load=False)
            if args.route != "core":
                kernel_accept(entry.state, entry.contacts, workload.scene.rigid_solver._errno)
            stress = qd_to_numpy(entry.state.stress_tensor, transpose=True)[0]
            residual = qd_to_numpy(entry.state.residual, transpose=True)[0]
            rhs = qd_to_numpy(entry.state.rhs, transpose=True)[0]
            error_code = qd_to_numpy(workload.scene.rigid_solver._errno, transpose=True)[0].item()
            valid = (
                qd_to_numpy(entry.state.valid, transpose=True)[0].item()
                and error_code == 0
                and np.isfinite(stress).all()
            )
            invalid += not valid
            tensor = voigt_tensor(stress)
            residual_norm = np.linalg.norm(residual).item()
            rhs_norm = np.linalg.norm(rhs).item()
            vm = qd_to_numpy(entry.state.von_mises, transpose=True)[0, ..., 0]
            peak_element, peak_corner = np.unravel_index(vm.argmax(), vm.shape) if valid else (0, 0)
            sampled = tangent(
                np.einsum("qc,qcij->qij", columns.native_barycentric, tensor[columns.native_elements]), columns
            )
            cpu_sampled = tangent(
                np.einsum("qc,qcij->qij", columns.native_barycentric, expected_cpu[columns.native_elements]), columns
            )
            row = {
                "time_s": packet["time_s"].item(),
                "valid": bool(valid),
                "errno": error_code,
                "contacts": count,
                "contact_status": qd_to_numpy(entry.contacts.status, transpose=True)[0, :count].tolist(),
                "complete_force_residual_l2_n": residual_norm if np.isfinite(residual_norm) else None,
                "complete_force_relative_residual": residual_norm / max(rhs_norm, 1e-12)
                if np.isfinite(residual_norm) and np.isfinite(rhs_norm)
                else None,
                "raw_vm_peak_pa": vm.max().item() if valid else None,
                "raw_maximum_principal_pa": np.linalg.eigvalsh(tensor)[..., -1].max().item() if valid else None,
                "raw_vm_peak_material_position_m": native_coordinates[
                    native_elements[peak_element, peak_corner]
                ].tolist()
                if valid
                else None,
                "common_tangent_error": field_error(sampled, expected_reference, columns) if valid else None,
                "independent_p2_core_error": field_error(sampled, cpu_sampled, columns)
                if valid and args.route == "core"
                else None,
            }
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            np.savez_compressed(
                args.output / "frames" / path.name,
                time_s=packet["time_s"],
                stress_voigt_link_pa=stress,
                balanced_rhs_n=rhs,
                residual_n=residual,
                applied_nodal_force_n=qd_to_numpy(entry.state.force, transpose=True)[0],
            )
            executed += 1
    write_json(
        args.output / "result.json",
        {
            "kind": "native-reference-load-replay",
            "route": args.route,
            "body_loads": args.body_loads,
            "executed_frames": executed,
            "invalid_frames": invalid,
            "reference_qualified": False,
            "scope": "Actual production Quadrants scatter/balance/elastic/full-field kernels, without advancing a contact simulation. Resolved loads and measured inertia use the documented material-node P2 transfer; its physical-versus-undeformed moment difference remains in the oracle. Compressed packets retain physical application points and component wrenches. Point-only inferred-source failures and complete residuals are retained; these interface ablations do not substitute for full contact/deformation rollouts.",
        },
    )


if __name__ == "__main__":
    main()
