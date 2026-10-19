"""Observe native DPR/SCR from the current rigid-pad trajectory's actual loads."""

from dataclasses import dataclass
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
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from .artifacts import file_id
from .common_metric import voigt_tensor
from .native_reference_replay import ContactViews
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

    @classmethod
    def allocate(cls, args, outer, faces, points, xml):
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
                contact_radius=args.radius,
                intrinsic_load_partition="body_source_prior",
                output_mode="full",
                update_mode="manual",
                fused_pipeline=False,
                surface_inverse=False,
            )
        )
        scene.build(n_envs=1)
        entry = scene.rigid_solver.stress_recovery.links[0]
        if args.source_reduction == "none":
            # Own a research input buffer large enough for every physical
            # centroid. Do not manufacture collider geometry to obtain capacity.
            entry.contacts = create_contacts(
                len(points),
                1,
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
        )

    def sample(self, force_world, rotation, angular_world, friction, load, path):
        started = perf_counter()
        force = force_world @ rotation
        if self.source_reduction == "wrench":
            selected, coefficient, components, defect = compress(self.points, force, self.adjacency, 0.06)
        else:
            selected = np.flatnonzero(np.any(force != 0, axis=1))
            coefficient = np.ones(len(selected))
            _, components = connected_components(self.adjacency[selected][:, selected], directed=False)
            defect = 0.0
        normals = np.zeros_like(force_world)
        normals[:, 2] = np.where(self.points[:, 2] > 0, -1.0, 1.0)
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
        compression_seconds = perf_counter() - started
        if entry.intrinsic is not None:
            sources = entry.intrinsic.raw_sources
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
            entry.intrinsic.scatter(
                entry.contacts,
                entry.state,
                entry.model.info,
                entry.surface.info,
                entry.options,
                np.finfo(gs.np_float).eps.item(),
                self.errno,
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
        stress = voigt_tensor(qd_to_numpy(entry.state.stress_tensor, transpose=True)[0])
        residual = qd_to_numpy(entry.state.residual, transpose=True)[0]
        rhs = qd_to_numpy(entry.state.rhs, transpose=True)[0]
        error_code = int(qd_to_numpy(self.errno, transpose=True)[0])
        valid = bool(
            qd_to_numpy(entry.state.valid, transpose=True)[0] and error_code == 0 and np.isfinite(stress).all()
        )
        np.savez_compressed(
            path,
            vertices_m=self.coordinates,
            tetrahedra=self.tetrahedra,
            stress_pa=stress,
            forces_n=qd_to_numpy(entry.state.force, transpose=True)[0],
            rhs_n=rhs,
            equation_residual_n=residual,
            load_n=load,
            source_points_m=self.points[selected],
            source_forces_n=force[selected] * coefficient[:, None],
            source_components=components,
            original_contact_points_m=self.points,
            original_contact_forces_n=force,
        )
        return {
            "native_valid": valid,
            "errno": error_code,
            "source_count": count,
            "source_capacity": entry.contacts.valid.shape[0],
            "source_reduction": self.source_reduction,
            "source_compression_wrench_error": defect,
            "source_compression_seconds": compression_seconds,
            "observation_seconds_including_IO": perf_counter() - started,
            "contact_status": qd_to_numpy(entry.contacts.status, transpose=True)[0, :count].tolist(),
            "elastic_equation_relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30)),
            "field": file_id(path),
        }
