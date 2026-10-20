"""Observe native DPR/SCR from the current rigid-pad trajectory's actual loads."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import torch
import trimesh

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.intrinsic import IntrinsicContacts
from genesis.engine.solvers.rigid.stress.recovery import StressLink, kernel_accept, kernel_begin_step
from genesis.engine.solvers.rigid.stress.resolved import ResolvedContacts
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from .artifacts import file_id, write_json
from .common_metric import voigt_tensor
from .distributed_pad_transfer import ResolvedPadTransfer
from .native_reference_replay import ContactViews
from .source_roundoff import reconcile_source_roundoff
from .wrench_compression import compress


@dataclass
class NativePadRecovery:
    scene: object
    entry: StressLink
    errno: qd.Tensor
    points: np.ndarray
    adjacency: sparse.csr_matrix
    coordinates: np.ndarray
    tetrahedra: np.ndarray
    radius: float
    source_reduction: str
    phase_timing: bool = True
    failure_path: Path | None = None
    distributed: ResolvedPadTransfer | None = None
    roundoff_reconciliation: bool = False

    @classmethod
    def allocate(cls, args, outer, faces, points, xml, load_transfer="fixed_radius", source_roundoff=False):
        if load_transfer not in ("fixed_radius", "resolved_outer_p2", "resolved_boundary_p2"):
            raise ValueError("Unknown pad load transfer")
        if load_transfer != "fixed_radius" and (args.method != "dpr" or args.source_reduction != "none"):
            raise ValueError("Resolved transfer requires DPR with complete physical centroid sources")
        if source_roundoff and (load_transfer == "resolved_outer_p2" or args.source_reduction != "none"):
            raise ValueError("Source reconciliation requires complete sources and fixed or native resolved transfer")
        # The motion scene accepts external wrenches; the independent observer
        # receives their complete surface-source packet. Keep the native guard
        # against observing an unspecified external wrench intact.
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0)),
            rigid_options=gs.options.RigidOptions(enable_collision=False),
            show_viewer=False,
        )
        scene.add_entity(morph=gs.morphs.MJCF(file=xml, default_armature=0.0))
        egg = scene.add_entity(
            morph=gs.morphs.URDF(
                file=args.assets / "egg_shell.urdf", align=False, decimate=False, preserve_mesh_precision=True
            )
        )
        egg.base_link.configure_stress_recovery(
            gs.options.RigidStressOptions(
                mesh=args.assets / "elastic.npz",
                method="pcg",
                max_iterations=10000,
                contact_model="pressure" if args.method == "dpr" else "intrinsic",
                load_transfer="resolved" if load_transfer == "resolved_boundary_p2" else "patch",
                contact_radius=args.radius,
                intrinsic_load_partition="body_source_prior",
                output_mode=getattr(args, "stress_output", "full"),
                update_mode="manual",
                fused_pipeline=False,
                surface_inverse=False,
            )
        )
        envs = getattr(args, "envs", 1)
        scene.build(n_envs=envs)
        entry = scene.rigid_solver.stress_recovery.links[0]
        if args.source_reduction == "none":
            # Own a research input buffer large enough for every physical
            # centroid. Do not manufacture collider geometry to obtain capacity.
            entry.contacts = create_contacts(
                len(points),
                envs,
                args.radius,
                args.method == "dpr" and entry.options.cooperative_pressure,
                args.method == "dpr" and entry.options.cooperative_scatter,
            )
            if entry.intrinsic is not None:
                entry.intrinsic = IntrinsicContacts(
                    entry.link,
                    entry.contacts,
                    entry.model.info,
                    entry.surface.info,
                    entry.options,
                    entry.intrinsic.paths,
                )
            if entry.resolved is not None:
                entry.resolved = ResolvedContacts(entry.link, entry.contacts, entry.model.info, entry.options)
        pairs = trimesh.Trimesh(outer, faces, process=False).face_adjacency
        adjacency = sparse.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(faces), len(faces))
        ).tocsr()
        adjacency = adjacency + adjacency.T
        coordinates = qd_to_numpy(entry.model.info.vertices, transpose=True)
        tetrahedra = qd_to_numpy(entry.model.info.elements, transpose=True).T[:, :4]
        return cls(
            scene,
            entry,
            scene.rigid_solver._errno,
            points,
            adjacency,
            coordinates,
            tetrahedra,
            args.radius,
            args.source_reduction,
            not getattr(args, "primary_cost", False),
            args.output / "frames" / "rejected-stress.npz",
            ResolvedPadTransfer.allocate(outer, faces, coordinates, envs) if load_transfer == "resolved_outer_p2" else None,
            source_roundoff,
        )

    @property
    def load_transfer(self):
        if self.entry.resolved is not None:
            return "resolved_boundary_p2"
        return "resolved_outer_p2" if self.distributed is not None else "fixed_radius"

    def sample(self, force_world, rotation, angular_world, friction, load, path=None, *, physical_law_admitted=False):
        if self.load_transfer != "fixed_radius" and not physical_law_admitted:
            raise ValueError("Resolved pressure transfer requires separate physical law and momentum admission")
        clock = perf_counter if self.phase_timing else lambda: 0.0
        started = clock()
        force = force_world @ rotation
        original_force = force
        normals = np.zeros_like(force_world)
        normals[:, 2] = np.where(self.points[:, 2] > 0, -1.0, 1.0)
        roundoff = None
        if self.roundoff_reconciliation:
            try:
                roundoff = reconcile_source_roundoff(self.points, force, normals @ rotation, friction, load)
            except ValueError as error:
                if self.failure_path is not None:
                    self.failure_path.parent.mkdir(parents=True, exist_ok=True)
                    selected = np.flatnonzero(np.any(force != 0, axis=1))
                    np.savez_compressed(self.failure_path, source_points_m=self.points[selected],
                        source_forces_n=force[selected], source_normals_material=(normals @ rotation)[selected],
                        original_contact_forces_n=force, friction=friction, load_n=load)
                    write_json(self.failure_path.with_suffix(".json"), {
                        "accepted": False, "error": str(error), "input": file_id(self.failure_path),
                        "scope": "Unchanged actual input refused before recovery because its cone conflict exceeds the physical source-roundoff budget."})
                raise
            force = roundoff.forces
        if self.source_reduction == "wrench":
            selected, coefficient, components, defect = compress(self.points, force, self.adjacency, 0.06)
        else:
            selected = np.flatnonzero(np.any(force != 0, axis=1))
            coefficient = np.ones(len(selected))
            _, components = connected_components(self.adjacency[selected][:, selected], directed=False)
            defect = 0.0
        entry = self.entry
        kernel_begin_step(entry.state)
        errno = qd_to_torch(self.errno, transpose=True, copy=False)
        errno.zero_()
        force_view = qd_to_torch(entry.state.force, transpose=True, copy=False)
        force_view.zero_()
        contacts = ContactViews.allocate(entry.contacts)
        count = contacts.load(
            {
                "compressed_positions_material_m": self.points[selected],
                "compressed_forces_material_n": force[selected] * coefficient[:, None],
                "compressed_normals": (normals @ rotation)[selected],
                "friction": np.array(friction),
                "radius_m": np.array(self.radius),
            }
        )
        compression_seconds = clock() - started
        if self.distributed is not None:
            self.distributed.scatter(selected, entry.contacts, entry.state)
        elif entry.intrinsic is not None or entry.resolved is not None:
            transfer = entry.intrinsic if entry.intrinsic is not None else entry.resolved
            sources = transfer.raw_sources
            sources.geom.fill(entry.link.geoms[0].idx)
            sources.method.fill(0)
            sources.vertex.fill(-1)
            sources.witness_face.fill(-1)
            sources.has_vertex_ambiguity.fill(False)
            sources.free_moment.fill(0)
            pair = qd_to_torch(sources.pair, transpose=True, copy=False)
            witness = qd_to_torch(sources.witness, transpose=True, copy=False)
            pair.fill_(-1)
            witness.fill_(float("nan"))
            pair[0, :count] = torch.as_tensor(components, dtype=gs.tc_int, device=gs.device)
            witness[0, :count] = contacts.position[0, :count]
            extra = (self.errno,) if entry.intrinsic is not None else ()
            transfer.scatter(
                entry.contacts,
                entry.state,
                entry.model.info,
                entry.surface.info,
                entry.options,
                np.finfo(gs.np_float).eps.item(),
                *extra,
            )
        else:
            kernel_anchor(np.finfo(gs.np_float).eps.item(), entry.contacts, entry.surface.info)
            kernel_pressure(entry.contacts, entry.surface.info, entry.options.cooperative_pressure)
            kernel_scatter(
                entry.contacts,
                entry.state,
                entry.model.info,
                entry.surface.info,
                entry.options.cooperative_scatter,
                entry.options.cached_face_bounds,
            )
        omega = qd_to_torch(entry.omega, transpose=True, copy=False)
        omega[0] = torch.as_tensor(angular_world @ rotation, device=gs.device)
        entry.model.recover(entry.omega, entry.state, surface_load=False)
        kernel_accept(entry.state, entry.contacts, self.errno)
        torch.cuda.synchronize()
        voigt = (
            qd_to_numpy(entry.state.stress_tensor, transpose=True)[0]
            if entry.output_mode == "full"
            else np.empty((0, 6))
        )
        stress = voigt_tensor(voigt)
        residual = qd_to_numpy(entry.state.residual, transpose=True)[0]
        rhs = qd_to_numpy(entry.state.rhs, transpose=True)[0]
        error_code = int(qd_to_numpy(self.errno, transpose=True)[0])
        peak = float(qd_to_numpy(entry.state.step_peak, transpose=True)[0])
        status = qd_to_numpy(entry.contacts.status, transpose=True)[0]
        valid = bool(
            qd_to_numpy(entry.state.valid, transpose=True)[0]
            and error_code == 0
            and np.isfinite(stress).all()
            and np.isfinite(peak)
            and np.isfinite(rhs).all()
            and np.isfinite(residual).all()
        )
        observation_seconds = clock() - started
        if not valid and path is None:
            path = self.failure_path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                vertices_m=self.coordinates,
                tetrahedra=self.tetrahedra,
                stress_pa=stress,
                forces_n=qd_to_numpy(entry.state.force, transpose=True)[0],
                rhs_n=rhs,
                equation_residual_n=residual,
                load_n=load,
                raw_vm_peak_pa=peak,
                source_points_m=self.points[selected],
                source_forces_n=force[selected] * coefficient[:, None],
                source_components=components,
                original_contact_points_m=self.points,
                original_contact_forces_n=original_force,
                source_roundoff_correction_n=force - original_force,
                original_contact_forces_world_n=force_world,
                rotation_material_to_world=rotation,
                angular_velocity_world_rad_per_s=angular_world,
                source_normals_material=(normals @ rotation)[selected],
                friction=friction,
                native_valid=valid,
                errno=error_code,
                contact_status=status[:count],
                load_transfer=self.load_transfer,
            )
        relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30))
        return {
            "native_valid": valid,
            "errno": error_code,
            "source_count": count,
            "source_capacity": entry.contacts.valid.shape[0],
            "source_reduction": self.source_reduction,
            "load_transfer": self.load_transfer,
            "source_roundoff": None if roundoff is None else {
                "corrected_sources": roundoff.corrected_sources,
                "force_change_bound_n": roundoff.force_change_bound_n,
                "moment_change_bound_nm": roundoff.moment_change_bound_nm,
                "physical_force_budget_n": roundoff.physical_force_budget_n,
            },
            "source_compression_wrench_error": defect,
            "source_compression_seconds": compression_seconds,
            "observation_seconds_excluding_frame_IO": observation_seconds,
            "observation_seconds_including_IO": clock() - started,
            "phase_timing_enabled": self.phase_timing,
            "output_mode": entry.output_mode,
            # Nonfinite values remain in the rejected NPZ, while the strict
            # journal records missing diagnostics and reaches the rejection.
            "raw_vm_peak_pa": peak if np.isfinite(peak) else None,
            "field_host_bytes": voigt.nbytes,
            "residual_host_bytes": residual.nbytes + rhs.nbytes,
            "contact_status": status[:count].tolist(),
            "elastic_equation_relative_residual": relative_residual if np.isfinite(relative_residual) else None,
            "field": file_id(path) if path is not None else None,
        }
