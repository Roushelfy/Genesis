"""Read original QIPC elastic and assembled contact fields without modifying solver state."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from qipc import Scene
from qipc._src.native.solver import ContactSystem, GlobalVertexManager
from qipc.constitution import StableNeoHookean, StvkHencky

from .elastic import SolidField, evaluate, nodal_elastic_gradient


@dataclass(frozen=True)
class ReferenceConfig:
    dt: float = 0.001
    young: float = 1e10
    poisson: float = 0.3
    density: float = 2000.0
    velocity_tolerance: float = 1e-5
    linear_tolerance: float = 1e-7
    max_newton: int = 256
    max_pcg: int = 4096
    d_hat: float = 2e-5
    friction: float = 0.6
    friction_epsilon_velocity: float = 1e-4
    resistance: float = 1e10
    energy: str = "snh"
    mas_precision: str = "float"
    linear_solver: str = "partition_pcg"

    def scene(self, gravity=(0.0, 0.0, -9.81), contact=True, diagnostics=False):
        scene = Scene(
            dt=self.dt,
            gravity=gravity,
            **{
                "contact/enable": int(contact),
                "contact/d_hat": self.d_hat,
                "contact/intersection_check": 1,
                "newton/velocity_tol": self.velocity_tolerance,
                "newton/contact_tol_scale": 1.0,
                "newton/max_iter": self.max_newton,
                "linear_system/max_iter": self.max_pcg,
                "linear_system/tol_rate": self.linear_tolerance,
                "linear_system/newton_aware_pcg/enable": 0,
                "linear_system/preconditioner/mas/invert_precision": self.mas_precision,
                "linear_system/solver": self.linear_solver,
                "line_search/max_iter": 64,
                "fem/assemble_fp32": 0,
                "friction/eps_v": self.friction_epsilon_velocity,
                "extras/pcg_diagnostics/enable": int(diagnostics),
                "extras/pcg_diagnostics/capacity": 16386 if diagnostics else 2048,
                "extras/pcg_diagnostics/record": "r",
                "extras/pcg_diagnostics/system_capacity": 8,
                "extras/pcg_diagnostics/nnz_capacity": 100000 if diagnostics else 0,
            },
        )
        scene.contact_tabular.default_model(friction_rate=self.friction, resistance=self.resistance)
        return scene

    def material(self):
        if self.energy == "snh":
            return StableNeoHookean()
        if self.energy == "hencky":
            return StvkHencky()
        raise ValueError(self.energy)


@dataclass
class ContactField:
    fem_forces_n: torch.Tensor
    global_vertex_ids: torch.Tensor
    global_forces_n: torch.Tensor
    global_positions_m: torch.Tensor


@dataclass
class SolidStep:
    field: SolidField
    contact: ContactField
    inertial_reaction_n: torch.Tensor
    residual_n: torch.Tensor
    positions_m: torch.Tensor
    kinetic_j: torch.Tensor
    potential_j: torch.Tensor
    elastic_j: torch.Tensor

    def metrics(self):
        load = torch.linalg.norm(self.contact.fem_forces_n).item()
        residual = torch.linalg.norm(self.residual_n).item()
        return {
            "minimum_det_F": self.field.determinant.min().item(),
            "all_fields_finite": bool(torch.isfinite(self.field.cauchy_pa).all()),
            "raw_vm_peak_pa": self.field.von_mises_pa.max().item(),
            "raw_maximum_principal_pa": self.field.maximum_principal_pa.max().item(),
            "minimum_surface_height_m": self.positions_m[:, 2].min().item(),
            "contact_resultant_n": self.contact.fem_forces_n.sum(dim=0).tolist(),
            "contact_moment_world_origin_nm": torch.cross(self.positions_m, self.contact.fem_forces_n, dim=1)
            .sum(dim=0)
            .tolist(),
            "force_residual_l2_n": residual,
            "force_residual_relative_contact": residual / load if load > 1e-12 else None,
            "force_residual_resultant_n": self.residual_n.sum(dim=0).tolist(),
            "egg_kinetic_j": self.kinetic_j.item(),
            "egg_potential_j": self.potential_j.item(),
            "egg_elastic_j": self.elastic_j.item(),
        }


@dataclass
class SolidObserver:
    scene: Scene
    config: ReferenceConfig
    rest: torch.Tensor
    tetrahedra: torch.Tensor
    inverse_rest: torch.Tensor
    volumes: torch.Tensor
    masses: torch.Tensor
    mu: torch.Tensor
    lame: torch.Tensor
    global_to_fem: torch.Tensor
    fem_global: torch.Tensor

    @classmethod
    def allocate(cls, scene, config):
        fe = scene.finite_element
        rest, tets = fe.x_bar.clone(), fe.tet_indices.clone().reshape(-1, 4)
        inverse_rest = torch.linalg.inv((rest[tets[:, 1:]] - rest[tets[:, :1]]).transpose(-1, -2))
        torch.testing.assert_close(fe.Dm_inv.reshape(-1, 3, 3), inverse_rest, rtol=2e-12, atol=2e-10)
        constitution = StableNeoHookean if config.energy == "snh" else StvkHencky
        live = scene._constitution_data[constitution]
        torch.testing.assert_close(
            live["tet_indices"],
            torch.arange(len(tets), dtype=live["tet_indices"].dtype, device=rest.device),
            rtol=0,
            atol=0,
        )
        gvm = scene.sim_systems[GlobalVertexManager]
        positions = gvm.positions_view().cpu().numpy()
        distance, indices = cKDTree(positions).query(fe.x.cpu().numpy())
        assert distance.max() < 1e-13 and len(np.unique(indices)) == len(indices)
        fem_global = torch.as_tensor(indices, device=rest.device)
        global_to_fem = torch.full((len(positions),), -1, dtype=fem_global.dtype, device=rest.device)
        global_to_fem[fem_global] = torch.arange(len(rest), dtype=fem_global.dtype, device=rest.device)
        return cls(
            scene,
            config,
            rest,
            tets,
            inverse_rest,
            fe.rest_volumes.clone(),
            fe.masses.clone(),
            live["mu"].clone(),
            live["lambda"].clone(),
            global_to_fem,
            fem_global,
        )

    def update(self) -> SolidField:
        current = self.scene.finite_element.x
        deformation = (current[self.tetrahedra[:, 1:]] - current[self.tetrahedra[:, :1]]).transpose(
            -1, -2
        ) @ self.inverse_rest
        return evaluate(deformation, self.mu, self.lame, self.config.energy)

    def elastic_gradient(self, field):
        return nodal_elastic_gradient(field, self.inverse_rest, self.volumes, self.tetrahedra, len(self.rest))

    def read_step(self, previous_velocity):
        fe = self.scene.finite_element
        with torch.cuda.stream(self.scene.solver.cuda_stream):
            field = self.update()
            gradient = self.elastic_gradient(field)
            inertial = self.masses[:, None] * (
                (fe.velocities - previous_velocity) / self.config.dt - fe.gravity - fe.external_acc
            )
            kinetic = (self.masses[:, None] * fe.velocities**2).sum() / 2
            potential = -(self.masses[:, None] * fe.gravity * fe.x).sum()
            elastic = (self.volumes * field.energy_density_relative_rest_j_per_m3).sum()
        torch.cuda.synchronize()
        contact = self.contact()
        return SolidStep(
            field, contact, inertial, inertial + gradient - contact.fem_forces_n, fe.x, kinetic, potential, elastic
        )

    def contact(self):
        native = self.scene.sim_systems[ContactSystem]
        count = native.n_unique_doublets_host()
        forces = self.rest.new_zeros((len(self.rest), 3))
        global_positions = self.scene.sim_systems[GlobalVertexManager].positions_view()
        torch.testing.assert_close(global_positions[self.fem_global], self.scene.finite_element.x, rtol=0, atol=1e-12)
        if count:
            global_ids = native.unique_doublet_vert_view()[:count].clone()
            global_forces = -native.unique_doublet_grad_view()[:count].clone() / self.config.dt**2
            assert (global_ids >= 0).all() and (global_ids < len(self.global_to_fem)).all()
            local = self.global_to_fem[global_ids]
            valid = local >= 0
            forces.index_add_(0, local[valid], global_forces[valid])
        else:
            global_ids = torch.empty((0,), dtype=self.tetrahedra.dtype, device=self.rest.device)
            global_forces = self.rest.new_empty((0, 3))
        return ContactField(forces, global_ids, global_forces, global_positions.clone())


def solver_statistics(scene):
    solver = scene.solver
    enabled = getattr(solver, "_et_check_enabled", None)
    reader = getattr(solver, "read_et_intersections", None)
    if reader is None:
        reader = getattr(getattr(solver, "_native", None), "read_et_intersections", None)
    pairs = reader() if reader is not None and enabled else None
    return {
        "newton": solver.newton_iters,
        "pcg_max": solver.max_pcg_iters,
        "pcg_total": solver.total_pcg_iters,
        "line_search_max": solver.max_ls_iters,
        "newton_last_max_displacement_m": solver.newton_max_disp_free,
        "newton_last_contact_displacement_m": solver.newton_max_disp_contact,
        "ccd_minimum_alpha": solver.min_ccd_alpha,
        "edge_triangle_check_enabled": enabled,
        "edge_triangle_intersection_pairs": [list(pair) for pair in pairs] if pairs is not None else None,
        "edge_triangle_intersection_count": len(pairs) if pairs is not None else None,
    }


def preserve_contact_failure(scene, output):
    """Keep a real native dump and static contact indices at an already rejected state."""
    from .artifacts import file_id, write_json

    native = scene.solver._native
    outcomes = {}
    try:
        native.contact_closure_stage_static()
        arrays = {key: np.asarray(value) for key, value in native.contact_closure_get_static().items()}
        path = output / "failure-contact-static.npz"
        np.savez_compressed(path, **arrays)
        outcomes["static_contact"] = file_id(path)
    except Exception as error:
        outcomes["static_contact_export_error"] = f"{type(error).__name__}: {error}"
    try:
        scene.workspace = output / "failure-native-checkpoint"
        scene.dump()
        outcomes["native_checkpoint"] = str(scene.workspace)
    except Exception as error:
        outcomes["native_checkpoint_export_error"] = f"{type(error).__name__}: {error}"
    outcomes["scope"] = (
        "Rejected-state forensic capture only. A real native dump does not validate continuation through an intersection. No solver, contact law, history, tolerance or state is repaired."
    )
    write_json(output / "contact-failure-capture.json", outcomes)
