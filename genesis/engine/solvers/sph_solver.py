import math
from dataclasses import dataclass

import numpy as np
import torch

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.couplers import LegacyCoupler
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_solve_mass_batch
from genesis.engine.states.solvers import SPHSolverState
from genesis.utils import array_class
from genesis.utils.misc import qd_to_torch, tensor_to_array
from genesis.utils.point_cloud import sample_mesh_point_cloud

from .base_solver import GravityMixin, Solver, TimeBasedMixin


# Number of rigid links whose boundary a fluid particle caches (see init_particle_fields).
N_BOUNDARY_SLOTS = 3


@dataclass
class SPHSolveStats:
    """Iteration counts and residuals of the solves of the SPH solver, summed over the substeps of the last step.

    A residual is the final error of a solve over its tolerance, so a solve has converged when it is at most 1. A solve
    is unconverged when it stops at its iteration cap with a residual above 1, and a substep holding one is no valid
    accuracy or performance sample.
    """

    n_substeps: int = 0
    n_divergence_iters: int = 0
    n_density_iters: int = 0
    n_viscosity_iters: int = 0
    n_coupling_passes: int = 0
    n_unconverged_divergence: int = 0
    n_unconverged_density: int = 0
    n_unconverged_viscosity: int = 0
    n_unconverged_coupling: int = 0
    max_divergence_residual: float = 0.0
    max_density_residual: float = 0.0
    max_viscosity_residual: float = 0.0
    max_coupling_residual: float = 0.0
    # fluid particles near more rigid links than they cache, whose surplus links they ignore (see N_BOUNDARY_SLOTS)
    n_boundary_slot_overflows: int = 0


@qd.data_oriented
class SPHSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = SPH.Base
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius
        self._pressure_solver = options.pressure_solver
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        # Divergence-free SPH (DFSPH) parameters
        self._df_max_error_div = options.max_divergence_error
        self._df_max_error_den = options.max_density_error
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations
        self._df_min_div_iters = options.min_divergence_solver_iterations
        self._df_min_den_iters = options.min_density_solver_iterations
        self._df_skip_div = options.skip_divergence_solver
        self._df_warm_start = options.warm_start
        self._micropolar = options.micropolar
        self._visc_max_error = options.max_viscosity_error
        self._visc_max_iters = options.max_viscosity_solver_iterations
        self._max_coupling_iters = options.max_coupling_iterations
        # Whether any material has a viscosity, which enables the implicit viscosity solve. Resolved at build time.
        self._has_viscosity = False
        self._df_eps = 1e-5
        # A particle with fewer neighbors sits at a free surface, where the divergence estimate is unreliable.
        self._df_min_div_neighbors = 20

        self._particle_volume = options.particle_volume_factor * self._particle_size**3

        # Boundary particles sampled on the surface of the rigid geoms. The spacing is the one of the regular triangle
        # sampling of Akinci et al. 2012 boundaries, denser than the fluid so that the fluid cannot leak through.
        self._boundary_spacing = 0.75 * self._particle_size
        # Upper bound on the number of boundary particles, which grows with the area of the simulation domain.
        self._max_n_boundary_particles = 5_000_000
        self._boundary_offset = options.boundary_offset
        self._n_boundary_particles = 0
        self._has_boundary = False
        # Links with degrees of freedom that carry boundary particles, which exchange momentum with the fluid in both
        # directions (SPH_SOLVER_DESIGN.md section 2). Resolved at build time.
        self._coupled_links_idx = np.zeros(0, dtype=gs.np_int)
        self._has_rigid_coupling = False
        self._solve_stats = SPHSolveStats()

        # spatial hasher
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )
        # boundary
        self.boundary = CubeBoundary(lower=self._lower_bound, upper=self._upper_bound)

    def init_particle_fields(self):
        # dynamic particle state
        # TODO: Some fields may not need to be in both reordered and un-reoredered.
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            acc=gs.qd_vec3,  # non-pressure acceleration
            material=gs.qd_int,  # material index
            reordered_idx=gs.qd_int,  # TODO: see above.
            active=gs.qd_bool,
            # Micro-rotation (Bender et al. 2017, "Turbulent Micropolar SPH Fluids with Foam"), allocated on demand.
            **(dict(angular_vel=gs.qd_vec3, angular_acc=gs.qd_vec3) if self._micropolar else {}),
            # --- DFSPH fields ---
            # 1 / (|sum_j V_j grad W_ij|^2 + sum_j |V_j grad W_ij|^2), i.e. alpha * rho_0^2 / rho
            alpha_factor_adj=gs.qd_float,
            # sum_j V_j (a_i - a_j) . grad W_ij, the density change rate induced by the current pressure
            predicted_source=gs.qd_float,
            # p / rho^2 of the density solve, kept across substeps for warm starting
            pressure_density2_norm_p=gs.qd_float,
            # p / rho^2 of the divergence solve, kept across substeps for warm starting
            pressure_density2_norm_d=gs.qd_float,
            # pressure acceleration
            pressure_acc=gs.qd_vec3,
            # predicted density rho* / rho_0 (density solve), or density change rate (divergence solve)
            density_adv_norm=gs.qd_float,
            # rho / rho_0 = sum_j V_j W_ij
            density_norm=gs.qd_float,
            # number of neighbors within the support radius
            n_neighbors=gs.qd_int,
        )

        # static material info
        struct_material_info = qd.types.struct(
            rest_volume=gs.qd_float,
            mass=gs.qd_float,
            rest_density=gs.qd_float,  # mass / rest_volume
            # mu = rest_density * nu, in Pa s (the material option is the kinematic viscosity nu, in m^2/s)
            dynamic_viscosity=gs.qd_float,
            surface_tension=gs.qd_float,
            vorticity=gs.qd_float,
            angular_viscosity=gs.qd_float,
            inverse_microinertia=gs.qd_float,  # 1 / microinertia
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            active=gs.qd_bool,
        )

        # construct fields
        self.particles = struct_particle_state.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_reordered = struct_particle_state.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        # One material per entity.
        self.materials = struct_material_info.field(shape=(len(self.entities),), needs_grad=False, layout=qd.Layout.AOS)

        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

        # Per-environment reductions of the DFSPH solver iterations.
        self.envs_n_active = qd.field(gs.qd_int, shape=(self._B,))
        self.envs_solver_error = qd.field(gs.qd_float, shape=(self._B,))

        if self._has_viscosity:
            # State of the implicit viscosity solve (M + h D) v = M v_rhs, by preconditioned conjugate gradient, in
            # hashed (reordered) particle order. See SPH_SOLVER_DESIGN.md section 3.4.
            struct_viscosity_state = qd.types.struct(
                # velocity after the explicit accelerations of the substep (v*)
                vel_star=gs.qd_vec3,
                # velocity change of the last density solve, and its change since the previous coupling pass
                pressure_vel=gs.qd_vec3,
                pressure_vel_delta=gs.qd_vec3,
                residual=gs.qd_vec3,
                residual_precond=gs.qd_vec3,
                direction=gs.qd_vec3,
                # image of the search direction by the system matrix
                direction_image=gs.qd_vec3,
                # inverse of the 3x3 diagonal block of the system matrix (block-Jacobi preconditioner)
                precond_inv=gs.qd_mat3,
                # sum of the wall viscosity tensors of the particle (see _func_wall_viscosity_tensor)
                wall_tensor=gs.qd_mat3,
                # renormalization of the viscosity operator of the particle (see _kernel_compute_viscosity_geometry)
                viscosity_scale=gs.qd_float,
                # Chebyshev acceleration of the coupling passes (see _kernel_accelerate_coupling): the pressures the
                # current and the previous pass started from
                pressure_in=gs.qd_float,
                pressure_in_prev=gs.qd_float,
            )
            self.particles_viscosity = struct_viscosity_state.field(
                shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
            )
            struct_viscosity_env = qd.types.struct(
                # preconditioned squared residual norm r^T z, and its value after the current update
                residual_norm2=gs.qd_float,
                residual_norm2_next=gs.qd_float,
                # d^T A d of the current search direction
                direction_norm2=gs.qd_float,
                # squared preconditioned norm of the residual at the start of the solve, the scale of the tolerance
                residual_norm2_init=gs.qd_float,
                # squared preconditioned norm of the right-hand side, the scale of the floor of the tolerance
                rhs_norm2=gs.qd_float,
                # residual norm of the viscosity-pressure agreement
                coupling_norm2=gs.qd_float,
                # Upper bound of the eigenvalues of M^-1 (M + h D), from the Gershgorin disks of the viscosity system
                viscous_stiffness_bound=gs.qd_float,
                # Chebyshev step, squared spectrum ratio and weight of the current coupling pass
                chebyshev_step=gs.qd_float,
                chebyshev_ratio2=gs.qd_float,
                chebyshev_weight=gs.qd_float,
                # squared preconditioned norms of the viscous impulse of the first coupling pass of the substep and of
                # the gravity impulse, the scales of the tolerance of the viscosity-pressure agreement
                viscous_norm2=gs.qd_float,
                gravity_norm2=gs.qd_float,
            )
            self.envs_viscosity = struct_viscosity_env.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.AOS)

        if self._has_boundary:
            # Fluid load on each rigid link, accumulated over the substeps of the current step: impulses in world frame,
            # the torque about the center of mass of the link. See SPH_SOLVER_DESIGN.md section 8.
            n_links = self.sim.rigid_solver.n_links
            self.links_fluid_impulse = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            self.links_fluid_angular_impulse = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            # Boundary of every fluid particle, per rigid link near it, in hashed (reordered) particle order: the
            # gradient of its density constraint (see _kernel_compute_density_and_factor) and the wall viscosity tensor
            # (see _kernel_compute_viscosity_geometry). A slot holds the link index -1 when unused. The pressure and
            # viscosity solves read these aggregates instead of the boundary particles.
            struct_boundary_slot = qd.types.struct(
                link_idx=gs.qd_int,
                grad=gs.qd_vec3,
                wall_tensor=gs.qd_mat3,
            )
            self.particles_boundary_slots = struct_boundary_slot.field(
                shape=(self._n_particles, self._B, N_BOUNDARY_SLOTS), needs_grad=False, layout=qd.Layout.SOA
            )
            self.envs_boundary_slot_overflows = qd.field(gs.qd_int, shape=(self._B,))

        if self._has_rigid_coupling:
            # Rigid side of the coupling, in the space of the spatial velocities of the coupled links (angular then
            # linear part, about the center of mass of their kinematic tree, see func_vel_at_point), per environment.
            # The mobility Lambda = S M^-1 S^T maps a wrench on any coupled link to the change of the spatial
            # acceleration of every coupled link through the articulation, S stacking the motion subspaces of the
            # degrees of freedom of each link and its ancestors. See SPH_SOLVER_DESIGN.md section 5.
            rigid_solver = self.sim.rigid_solver
            n_coupled_6 = 6 * len(self._coupled_links_idx)
            self.links_coupled_idx = qd.field(gs.qd_int, shape=(rigid_solver.n_links,))
            self.coupled_links_idx = qd.field(gs.qd_int, shape=(len(self._coupled_links_idx),))
            self.coupled_mobility = qd.field(gs.qd_float, shape=(n_coupled_6, n_coupled_6, self._B))
            # spatial velocity seen by the fluid, its value before the impulses of the substep plus the smooth
            # acceleration, and its value at the start of the current solve
            self.coupled_vel = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_vel_star = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_vel_base = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            # spatial acceleration of the response to the pressure of the current iteration, and its wrench
            self.coupled_acc = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_wrench = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            # impulses of the fluid on the coupled links in the substep, from the divergence, viscosity and density
            # solves (wrenches times the substep)
            self.coupled_impulse_div = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_impulse_visc = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_impulse_den = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            # wall viscosity of the coupled links in their spatial velocity space, K (block diagonal, 6 x 6 per link)
            # and the wrench of the fluid velocity, f, and the dense system (I + h Lambda K) xi = rhs
            self.coupled_visc_stiffness = qd.field(gs.qd_float, shape=(n_coupled_6, 6, self._B))
            self.coupled_visc_force = qd.field(gs.qd_float, shape=(n_coupled_6, self._B))
            self.coupled_system = qd.field(gs.qd_float, shape=(n_coupled_6, n_coupled_6 + 1, self._B))
            # largest change of the coupled velocities over the last viscosity pass, relative to their magnitude
            self.envs_coupled_vel_change = qd.field(gs.qd_float, shape=(self._B,))
            # right-hand side and solution of the solves with the rigid mass matrix
            self.dofs_coupling_rhs = qd.field(gs.qd_float, shape=(rigid_solver.n_dofs_, self._B))
            self.dofs_coupling_sol = qd.field(gs.qd_float, shape=(rigid_solver.n_dofs_, self._B))
            # center of mass of each link in world frame at the current substep, the reference point of the torque
            self.links_com = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            # spatial velocity of each link at the current substep (see func_vel_at_point), which gives the rigid
            # velocity anywhere in space, as the wall viscosity needs it at the fluid particles
            self.links_root_com = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            self.links_vel = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))
            self.links_ang = qd.Vector.field(3, dtype=gs.qd_float, shape=(n_links, self._B))

        if self._has_boundary:
            # Boundary particles, sampled on the rigid geoms and moved with their link. They carry the static
            # sampling data as well, so that their hashed (reordered) copy is self-contained.
            struct_boundary_particle_state = qd.types.struct(
                pos_local=gs.qd_vec3,  # position in link frame
                normal_local=gs.qd_vec3,  # outward unit normal of the geom surface at the sample, in link frame
                normal=gs.qd_vec3,
                area=gs.qd_float,  # share of the geom surface the sample stands for
                link_idx=gs.qd_int,
                is_fixed=gs.qd_bool,  # whether the link is fixed wrt the world
                pos=gs.qd_vec3,
                vel=gs.qd_vec3,
                # Akinci et al. 2012 volume 1 / sum_l W_bl, which weighs a boundary particle by its sampling density.
                volume=gs.qd_float,
                reordered_idx=gs.qd_int,
                active=gs.qd_bool,
            )
            self.boundary_particles = struct_boundary_particle_state.field(
                shape=(self._n_boundary_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
            )
            self.boundary_particles_reordered = struct_boundary_particle_state.field(
                shape=(self._n_boundary_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
            )

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles

        self._coupler = self.sim._coupler

        if self.is_active:
            self.sh.build(self._B)

            # Sample the boundary particles of the rigid geoms coupled with the fluid, in link frame.
            boundaries_pos_local, boundaries_link_idx, boundaries_is_fixed, boundaries_active = [], [], [], []
            boundaries_normal_local, boundaries_area = [], []
            rigid_solver = self.sim.rigid_solver
            if isinstance(self._coupler, LegacyCoupler) and self._coupler.options.rigid_sph and rigid_solver.is_active:
                # Fluid stays within the domain, so boundary particles further away than the support radius are
                # never anyone's neighbor.
                area_lower = self._lower_bound - self._support_radius
                area_upper = self._upper_bound + self._support_radius
                n_boundary_particles = 0
                for entity in rigid_solver.entities:
                    for link in entity.links:
                        link_pos = np.atleast_2d(tensor_to_array(link.get_pos(relative=False)))[0]
                        link_quat = np.atleast_2d(tensor_to_array(link.get_quat(relative=False)))[0]
                        for geom in link.geoms:
                            if not geom.needs_coup:
                                continue
                            if geom.type == gs.GEOM_TYPE.PLANE:
                                # A plane collides as an infinite half-space while its mesh is a large finite quad,
                                # so it is sampled over its cut through the boundary area instead.
                                plane_pos = gu.transform_by_trans_quat(geom.init_pos, link_pos, link_quat)
                                plane_normal = gu.transform_by_quat(
                                    gu.transform_by_quat(geom.data[:3], geom.init_quat), link_quat
                                )
                                verts, faces = gu.plane_aabb_intersection(
                                    plane_pos, plane_normal, area_lower, area_upper
                                )
                                verts = gu.inv_transform_by_trans_quat(verts, link_pos, link_quat)
                                # The winding of the cut polygon is arbitrary, so its face normal may point inward.
                                surface_normal = gu.transform_by_quat(geom.data[:3], geom.init_quat)
                            else:
                                verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
                                faces = geom.init_faces
                            if len(faces) == 0:
                                continue

                            # Number of points of a regular triangular lattice with the boundary spacing.
                            area = trimesh.Trimesh(vertices=verts, faces=faces, process=False).area
                            n_points = math.ceil(2.0 * area / (math.sqrt(3.0) * self._boundary_spacing**2))
                            n_boundary_particles += n_points
                            if n_boundary_particles > self._max_n_boundary_particles:
                                gs.raise_exception(
                                    f"Sampling the SPH boundary of geom {geom.idx} of entity {entity.idx} requires "
                                    f"{n_points} particles, which brings the total above "
                                    f"{self._max_n_boundary_particles}. Please tighten the SPH domain "
                                    "(`lower_bound` / `upper_bound`) around the fluid, or shrink the geom."
                                )
                            # Furthest point sampling only spaces the samples evenly given enough candidates. Holes
                            # in the sampling lower the boundary density locally, and fluid leaks through them.
                            pos_local, normals_local = sample_mesh_point_cloud(
                                verts, faces, n_points, n_candidates=16 * n_points, seed=0, return_normals=True
                            )
                            if geom.type == gs.GEOM_TYPE.PLANE:
                                normals_local = np.tile(surface_normal, (len(pos_local), 1))
                            pos_local = pos_local - self._boundary_offset * normals_local

                            # A fixed link never moves, so the samples outside the boundary area can be dropped.
                            if link.is_fixed:
                                pos_world = gu.transform_by_trans_quat(pos_local, link_pos, link_quat)
                                is_inside = np.all((pos_world > area_lower) & (pos_world < area_upper), axis=1)
                                pos_local = pos_local[is_inside]
                                normals_local = normals_local[is_inside]

                            if geom.active_envs_mask is None:
                                active = np.ones((self._B, len(pos_local)), dtype=gs.np_int)
                            else:
                                envs_active = tensor_to_array(geom.active_envs_mask, dtype=gs.np_int)
                                active = np.tile(envs_active[:, None], (1, len(pos_local)))
                            boundaries_pos_local.append(pos_local)
                            boundaries_normal_local.append(normals_local)
                            # each sample stands for an equal share of the surface of the geom
                            boundaries_area.append(np.full(len(pos_local), area / n_points, dtype=gs.np_float))
                            boundaries_link_idx.append(np.full(len(pos_local), link.idx, dtype=gs.np_int))
                            boundaries_is_fixed.append(np.full(len(pos_local), link.is_fixed, dtype=gs.np_int))
                            boundaries_active.append(active)
            self._n_boundary_particles = sum(len(pos_local) for pos_local in boundaries_pos_local)
            self._has_boundary = self._n_boundary_particles > 0
            if self._has_boundary:
                boundaries_link_idx_all = np.concatenate(boundaries_link_idx)
                boundaries_is_fixed_all = np.concatenate(boundaries_is_fixed)
                self._coupled_links_idx = np.unique(boundaries_link_idx_all[boundaries_is_fixed_all == 0])
            self._has_rigid_coupling = len(self._coupled_links_idx) > 0
            self._has_viscosity = any(entity.material.viscosity > 0.0 for entity in self.entities)
            if self._has_boundary:
                gs.logger.info(f"SPH boundary sampled with ~<{self._n_boundary_particles}>~ particles.")
                self.sh_boundary = gu.SpatialHasher(cell_size=self.sh.cell_size, grid_res=self.sh.grid_res)
                self.sh_boundary.build(self._B)

            self.init_particle_fields()
            self.init_ckpt()
            if self._has_rigid_coupling:
                links_coupled_idx = np.full(rigid_solver.n_links, -1, dtype=gs.np_int)
                links_coupled_idx[self._coupled_links_idx] = np.arange(len(self._coupled_links_idx), dtype=gs.np_int)
                self._kernel_set_coupled_links(links_coupled_idx, self._coupled_links_idx)

            for entity in self.entities:
                entity._add_to_solver()

            if self._has_boundary:
                self._kernel_add_boundary_particles(
                    np.concatenate(boundaries_pos_local),
                    np.concatenate(boundaries_normal_local),
                    np.concatenate(boundaries_area),
                    np.concatenate(boundaries_link_idx),
                    np.concatenate(boundaries_is_fixed),
                    np.concatenate(boundaries_active, axis=1),
                )
                self._kernel_update_boundary_particles(rigid_solver.dyn_state.links)
                self._kernel_compute_boundary_volume()

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity(as_field=True)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "SPHEntity":
        entity = SPHEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            name=name,
        )

        self.entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles.active, self.particles.reordered_idx
        )

        # copy to reordered
        self.particles_reordered.active.fill(False)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                reordered_idx = self.particles[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                # only need to copy back dynamic state, i.e. self.particles
                self.particles[i_p, i_b] = self.particles_reordered[self.particles[i_p, i_b].reordered_idx, i_b]

    @qd.func
    def _task_compute_non_pressure_forces(self, i, j, ret: qd.template(), i_b):
        i_m_i = self.particles_reordered[i, i_b].material
        i_m_j = self.particles_reordered[j, i_b].material
        mass_i = self.materials[i_m_i].mass
        mass_j = self.materials[i_m_j].mass

        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        dist = x_ij.norm()

        # Surface tension (cohesion). The kernel is evaluated no closer than one particle size so that the attraction
        # vanishes as particles touch instead of peaking at the kernel center.
        effective_dist = qd.max(dist, self._particle_size)
        ret.acc -= self.materials[i_m_i].surface_tension / mass_i * mass_j * x_ij * self.cubic_kernel(effective_dist)

        # The viscosity is implicit, see _kernel_init_viscosity_solve.
        if qd.static(self._micropolar):
            # The micro-rotation field only couples particles of the same material.
            if i_m_i == i_m_j:
                rho_j = self.particles_reordered[j, i_b].density_norm * self.materials[i_m_j].rest_density
                grad_W = self.cubic_kernel_derivative(x_ij)
                vorticity = self.materials[i_m_i].vorticity
                inverse_microinertia = self.materials[i_m_i].inverse_microinertia
                rho_i = self.particles_reordered[i, i_b].density_norm * self.materials[i_m_i].rest_density
                omega_ij = self.particles_reordered[i, i_b].angular_vel - self.particles_reordered[j, i_b].angular_vel
                vel_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel

                # Smoothing of the micro-rotation field between neighbors (XSPH-like angular viscosity).
                ret.angular_acc -= (
                    inverse_microinertia
                    * self.materials[i_m_i].angular_viscosity
                    / self._substep_dt
                    * (mass_j / rho_j)
                    * omega_ij
                    * self.cubic_kernel(dist)
                )
                # Exchange between the curl of the velocity field and the micro-rotation field.
                ret.acc += vorticity / rho_i * mass_j * omega_ij.cross(grad_W)
                ret.angular_acc += vorticity / rho_i * inverse_microinertia * mass_j * vel_ij.cross(grad_W)

    @qd.kernel
    def _kernel_compute_non_pressure_forces(self, f: qd.i32, t: qd.f32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                ret = qd.Struct(acc=self._gravity[i_b], angular_acc=qd.Vector.zero(gs.qd_float, 3))
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_non_pressure_forces,
                    i_b,
                )

                for i_ff in qd.static(range(len(self._ffs))):
                    ret.acc += self._ffs[i_ff].get_acc(
                        self.particles_reordered[i_p, i_b].pos, self.particles_reordered[i_p, i_b].vel, t, i_p
                    )
                self.particles_reordered[i_p, i_b].acc = ret.acc

                if qd.static(self._micropolar):
                    # Decay of the micro-rotation towards zero, which balances the vorticity injected by the flow.
                    i_m = self.particles_reordered[i_p, i_b].material
                    ret.angular_acc -= (
                        2.0
                        * self.materials[i_m].inverse_microinertia
                        * self.materials[i_m].vorticity
                        * self.particles_reordered[i_p, i_b].angular_vel
                    )
                    self.particles_reordered[i_p, i_b].angular_acc = ret.angular_acc

        # Velocities and micro-rotations are integrated in a separate pass, since the accelerations above read the
        # velocities and micro-rotations of the neighbors.
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].acc
                if qd.static(self._has_viscosity):
                    self.particles_viscosity[i_p, i_b].vel_star = self.particles_reordered[i_p, i_b].vel
                    self.particles_viscosity[i_p, i_b].pressure_vel = qd.Vector.zero(gs.qd_float, 3)
                if qd.static(self._micropolar):
                    self.particles_reordered[i_p, i_b].angular_vel += (
                        self._substep_dt * self.particles_reordered[i_p, i_b].angular_acc
                    )

    @qd.kernel
    def _kernel_advect_position(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                pos = self.particles_reordered[i_p, i_b].pos
                new_vel = self.particles_reordered[i_p, i_b].vel

                new_pos = pos + self._substep_dt * new_vel

                # Clamp to the simulation domain.
                corrected_pos, corrected_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                self.particles_reordered[i_p, i_b].vel = corrected_vel
                self.particles_reordered[i_p, i_b].pos = corrected_pos

    # ------------------------------------------------------------------------------------
    # ------------------------------------ boundary --------------------------------------
    # ------------------------------------------------------------------------------------
    # Particle-based rigid boundaries (Akinci et al. 2012, "Versatile Rigid-Fluid Coupling for Incompressible SPH"),
    # entering the DFSPH solve as in Bender et al. 2023, "Consistent SPH Rigid-Fluid Coupling" (BWJ23). Boundary
    # particles only add constant contributions to the density constraints of the fluid particles, so they never
    # need a pressure value of their own.

    @qd.kernel
    def _kernel_add_boundary_particles(
        self,
        pos_local: qd.types.ndarray(),
        normal_local: qd.types.ndarray(),
        area: qd.types.ndarray(),
        link_idx: qd.types.ndarray(),
        is_fixed: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            for i in qd.static(range(3)):
                self.boundary_particles[i_bp, i_b].pos_local[i] = pos_local[i_bp, i]
                self.boundary_particles[i_bp, i_b].normal_local[i] = normal_local[i_bp, i]
            self.boundary_particles[i_bp, i_b].area = area[i_bp]
            self.boundary_particles[i_bp, i_b].link_idx = link_idx[i_bp]
            self.boundary_particles[i_bp, i_b].is_fixed = qd.cast(is_fixed[i_bp], gs.qd_bool)
            self.boundary_particles[i_bp, i_b].active = qd.cast(active[i_b, i_bp], gs.qd_bool)

    @qd.kernel
    def _kernel_update_boundary_particles(self, links_state: array_class.LinksState):
        for i_l, i_b in qd.ndrange(self.links_com.shape[0], self._B):
            self.links_com[i_l, i_b] = links_state.i_pos[i_l, i_b] + links_state.root_COM[i_l, i_b]
            self.links_root_com[i_l, i_b] = links_state.root_COM[i_l, i_b]
            self.links_vel[i_l, i_b] = links_state.cd_vel[i_l, i_b]
            self.links_ang[i_l, i_b] = links_state.cd_ang[i_l, i_b]

        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            if self.boundary_particles[i_bp, i_b].active:
                i_l = self.boundary_particles[i_bp, i_b].link_idx
                pos = gu.qd_transform_by_trans_quat(
                    self.boundary_particles[i_bp, i_b].pos_local, links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
                self.boundary_particles[i_bp, i_b].pos = pos
                self.boundary_particles[i_bp, i_b].normal = gu.qd_transform_by_quat(
                    self.boundary_particles[i_bp, i_b].normal_local, links_state.quat[i_l, i_b]
                )
                self.boundary_particles[i_bp, i_b].vel = self.sim.rigid_solver._func_vel_at_point(
                    i_l, i_b, pos, links_state
                )

        self.sh_boundary.compute_reordered_idx(
            self._n_boundary_particles,
            self.boundary_particles.pos,
            self.boundary_particles.active,
            self.boundary_particles.reordered_idx,
        )

        self.boundary_particles_reordered.active.fill(False)
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            if self.boundary_particles[i_bp, i_b].active:
                i_r = self.boundary_particles[i_bp, i_b].reordered_idx
                self.boundary_particles_reordered[i_r, i_b] = self.boundary_particles[i_bp, i_b]

    @qd.func
    def _task_compute_boundary_volume(self, i, j, ret: qd.template(), i_b):
        # The sampling of a moving link is rigid, so only its own particles count and its volumes stay valid as it
        # moves. The fixed links never move with respect to each other, so they form a single boundary.
        is_same_boundary = self.boundary_particles_reordered[j, i_b].link_idx == ret.link_idx
        if ret.is_fixed and self.boundary_particles_reordered[j, i_b].is_fixed:
            is_same_boundary = True
        if is_same_boundary:
            x_ij = self.boundary_particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
            ret.sum_w += self.cubic_kernel(x_ij.norm())

    @qd.kernel
    def _kernel_compute_boundary_volume(self):
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            if self.boundary_particles[i_bp, i_b].active:
                i_r = self.boundary_particles[i_bp, i_b].reordered_idx
                ret = qd.Struct(
                    sum_w=self.cubic_kernel(0.0),
                    link_idx=self.boundary_particles[i_bp, i_b].link_idx,
                    is_fixed=self.boundary_particles[i_bp, i_b].is_fixed,
                )
                self.sh_boundary.for_all_neighbors(
                    i_r,
                    self.boundary_particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_boundary_volume,
                    i_b,
                )
                self.boundary_particles[i_bp, i_b].volume = 1.0 / ret.sum_w

    @qd.func
    def _task_compute_boundary_density_and_factor(self, i, j, ret: qd.template(), i_b):
        V_b = self.boundary_particles_reordered[j, i_b].volume
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        W_ib = self.cubic_kernel(x_ib.norm())
        ret.density_norm += V_b * W_ib
        ret.n_neighbors += 1

        # Accumulate in the slot of the link of the boundary particle, the first unused one taking a new link.
        i_l = self.boundary_particles_reordered[j, i_b].link_idx
        i_s = -1
        for k in range(N_BOUNDARY_SLOTS):
            i_l_slot = self.particles_boundary_slots[i, i_b, k].link_idx
            if i_s == -1 and (i_l_slot == i_l or i_l_slot == -1):
                i_s = k
        if i_s == -1:
            ret.n_overflows += 1
        else:
            # The wall only resists the motion of the fluid along its normal, as a smooth impermeable wall does: the
            # kernel gradient is projected on the outward normal of the geom at the boundary particle. The full gradient
            # keeps a tangential part that varies as the fluid slides past the discrete samples, and acts as a roughness
            # that drags the fluid along any wall. The force along the projected gradient is not central, so its
            # reaction applies to the link at the fluid particle (see _func_slot_wall_vel), which keeps linear and
            # angular momentum balanced.
            normal = self.boundary_particles_reordered[j, i_b].normal
            self.particles_boundary_slots[i, i_b, i_s].link_idx = i_l
            self.particles_boundary_slots[i, i_b, i_s].grad += (
                V_b * self.cubic_kernel_derivative(x_ib).dot(normal) * normal
            )

    @qd.func
    def _func_slot_wall_vel(self, i_p, i_b, k):
        """Return the rigid velocity of the link of boundary slot k of fluid particle i_p, at the fluid particle."""
        i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
        return self._func_link_vel_at_pos(i_l, i_b, self.particles_reordered[i_p, i_b].pos)

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------
    # See https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp for reference.

    @qd.func
    def _task_compute_density_and_factor(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        V_j = self.materials[i_m].rest_volume
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos

        ret.density_norm += V_j * self.cubic_kernel(x_ij.norm())
        grad_p_j = -V_j * self.cubic_kernel_derivative(x_ij)
        ret.sum_grad_p_k += grad_p_j.norm_sqr()
        ret.grad_p_i -= grad_p_j
        ret.n_neighbors += 1

    @qd.kernel
    def _kernel_compute_density_and_factor(self, f: qd.i32):
        for i_b in range(self._B):
            self.envs_n_active[i_b] = 0

        if qd.static(self._has_boundary):
            for i_b in range(self._B):
                self.envs_boundary_slot_overflows[i_b] = 0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                i_m = self.particles_reordered[i_p, i_b].material
                ret = qd.Struct(
                    density_norm=self.materials[i_m].rest_volume * self.cubic_kernel(0.0),
                    sum_grad_p_k=gs.qd_float(0.0),
                    grad_p_i=qd.Vector.zero(gs.qd_float, 3),
                    n_neighbors=0,
                    n_overflows=0,
                )
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_density_and_factor,
                    i_b,
                )

                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        self.particles_boundary_slots[i_p, i_b, k].link_idx = -1
                        self.particles_boundary_slots[i_p, i_b, k].grad = qd.Vector.zero(gs.qd_float, 3)
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        ret,
                        self._task_compute_boundary_density_and_factor,
                        i_b,
                    )
                    if ret.n_overflows > 0:
                        self.envs_boundary_slot_overflows[i_b] += 1
                    # BWJ23 eq. (32): boundary particles carry no constraint of their own, so they only enter the
                    # gradient of the constraint of particle i, and not the sum of squared neighbor gradients.
                    for k in range(N_BOUNDARY_SLOTS):
                        i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                        if i_l != -1:
                            grad = self.particles_boundary_slots[i_p, i_b, k].grad
                            ret.grad_p_i += grad
                            if qd.static(self._has_rigid_coupling):
                                # The pressure of the particle also moves the coupled link it pushes, which feeds back
                                # on its own constraint through the mobility of the link (BWJ23 for dynamic bodies).
                                i_c = self.links_coupled_idx[i_l]
                                if i_c != -1:
                                    torque = (
                                        self.particles_reordered[i_p, i_b].pos - self.links_root_com[i_l, i_b]
                                    ).cross(grad)
                                    wrench = qd.Vector([torque[0], torque[1], torque[2], grad[0], grad[1], grad[2]])
                                    response = gs.qd_float(0.0)
                                    for a in qd.static(range(6)):
                                        for b in qd.static(range(6)):
                                            response += (
                                                wrench[a]
                                                * self.coupled_mobility[6 * i_c + a, 6 * i_c + b, i_b]
                                                * wrench[b]
                                            )
                                    ret.sum_grad_p_k += self.materials[i_m].mass * response

                sum_grad_p_k = ret.sum_grad_p_k + ret.grad_p_i.norm_sqr()
                factor = gs.qd_float(0.0)
                if sum_grad_p_k > self._df_eps:
                    factor = 1.0 / sum_grad_p_k

                self.particles_reordered[i_p, i_b].density_norm = ret.density_norm
                self.particles_reordered[i_p, i_b].alpha_factor_adj = factor
                self.particles_reordered[i_p, i_b].n_neighbors = ret.n_neighbors
                self.envs_n_active[i_b] += 1

    @qd.func
    def _func_pressure_density2_norm(self, i_p, i_b, is_divergence):
        res = self.particles_reordered[i_p, i_b].pressure_density2_norm_p
        if is_divergence:
            res = self.particles_reordered[i_p, i_b].pressure_density2_norm_d
        return res

    @qd.func
    def _task_compute_pressure_acc(self, i, j, ret: qd.template(), i_b):
        i_m_i = self.particles_reordered[i, i_b].material
        i_m_j = self.particles_reordered[j, i_b].material
        density_ratio = self.materials[i_m_j].rest_density / self.materials[i_m_i].rest_density
        p_sum = ret.p_rho2_i + density_ratio * self._func_pressure_density2_norm(j, i_b, ret.is_divergence)
        if qd.abs(p_sum) > self._df_eps:
            x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
            ret.acc -= p_sum * self.materials[i_m_j].rest_volume * self.cubic_kernel_derivative(x_ij)

    @qd.func
    def _func_compute_pressure_acc(self, i_p, i_b, is_divergence: qd.template()):
        ret = qd.Struct(
            acc=qd.Vector.zero(gs.qd_float, 3),
            p_rho2_i=self._func_pressure_density2_norm(i_p, i_b, is_divergence),
            is_divergence=is_divergence,
        )
        self.sh.for_all_neighbors(
            i_p, self.particles_reordered.pos, self._support_radius, ret, self._task_compute_pressure_acc, i_b
        )

        if qd.static(self._has_boundary):
            # BWJ23 eq. (33): a single term of the fluid particle pressure, with no pressure value at the boundary.
            for k in range(N_BOUNDARY_SLOTS):
                if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                    ret.acc -= ret.p_rho2_i * self.particles_boundary_slots[i_p, i_b, k].grad

        self.particles_reordered[i_p, i_b].pressure_acc = ret.acc

    @qd.func
    def _task_compute_density_change(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        v_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        ret += self.materials[i_m].rest_volume * v_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.kernel
    def _kernel_init_pressure_solve(self, is_divergence: qd.template(), keep_pressure: bool):
        """Compute the source term of the divergence or density solve and initialize its pressure.

        keep_pressure starts the density solve from the pressure it holds, which a later coupling pass of a substep does
        (SPH_SOLVER_DESIGN.md section 5.2).
        """
        inv_dt = 1.0 / self._substep_dt
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                density_change = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    density_change,
                    self._task_compute_density_change,
                    i_b,
                )

                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                            vel_rel = self.particles_reordered[i_p, i_b].vel - self._func_slot_wall_vel(i_p, i_b, k)
                            density_change += vel_rel.dot(self.particles_boundary_slots[i_p, i_b, k].grad)

                factor = self.particles_reordered[i_p, i_b].alpha_factor_adj
                if qd.static(is_divergence):
                    self.particles_reordered[i_p, i_b].density_adv_norm = density_change

                    # Only compression is corrected, and never at the free surface. This clamped source only
                    # seeds the solve: the iterations use the raw density change rate.
                    source = qd.max(density_change, 0.0)
                    if self.particles_reordered[i_p, i_b].n_neighbors < self._df_min_div_neighbors:
                        source = 0.0

                    p_rho2 = source * factor * inv_dt
                    if qd.static(self._df_warm_start):
                        p_rho2 = 0.0
                        if source > 0.0:
                            p_rho2 = 0.5 * qd.min(
                                self.particles_reordered[i_p, i_b].pressure_density2_norm_d, 0.5 * inv_dt
                            )
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_d = p_rho2
                else:
                    density_adv = self.particles_reordered[i_p, i_b].density_norm + self._substep_dt * density_change
                    self.particles_reordered[i_p, i_b].density_adv_norm = density_adv

                    p_rho2 = -qd.min(1.0 - density_adv, 0.0) * factor * inv_dt * inv_dt
                    if qd.static(self._df_warm_start):
                        p_rho2 = 0.0
                        if density_adv > 1.0:
                            # The clamp bounds p * dt^2, which is what limits the warm-started velocity change.
                            p_rho2 = 0.5 * qd.min(
                                self.particles_reordered[i_p, i_b].pressure_density2_norm_p, 0.00025 * inv_dt * inv_dt
                            )
                    if keep_pressure:
                        p_rho2 = self.particles_reordered[i_p, i_b].pressure_density2_norm_p
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_p = p_rho2

    @qd.func
    def _task_compute_predicted_source(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        a_ij = self.particles_reordered[i, i_b].pressure_acc - self.particles_reordered[j, i_b].pressure_acc
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        ret += self.materials[i_m].rest_volume * a_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.kernel
    def _kernel_pressure_solve_iteration(self, is_divergence: qd.template()) -> float:
        """
        Run one Jacobi iteration of the divergence or density solve and return the worst average error over envs.

        The error is the average density deviation (normalized by the rest density) for the density solve, and the
        average normalized density change rate for the divergence solve.
        """
        # dt for the divergence solve, dt^2 for the density solve.
        dt_pow = self._substep_dt
        if qd.static(not is_divergence):
            dt_pow = self._substep_dt * self._substep_dt

        for i_b in range(self._B):
            self.envs_solver_error[i_b] = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self._func_compute_pressure_acc(i_p, i_b, is_divergence)

        if qd.static(self._has_rigid_coupling):
            # Response of the coupled links to the boundary pressure of the iteration (SPH_SOLVER_DESIGN.md section 5.1)
            n_coupled_6 = 6 * self.coupled_links_idx.shape[0]
            for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
                self.coupled_wrench[i_r, i_b] = 0.0
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles_reordered[i_p, i_b].active:
                    i_m = self.particles_reordered[i_p, i_b].material
                    mass_kappa = self.materials[i_m].mass * self._func_pressure_density2_norm(i_p, i_b, is_divergence)
                    self._func_add_coupled_wrench(i_p, i_b, mass_kappa)
            for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
                acc = gs.qd_float(0.0)
                for i_col in range(n_coupled_6):
                    acc += self.coupled_mobility[i_r, i_col, i_b] * self.coupled_wrench[i_col, i_b]
                self.coupled_acc[i_r, i_b] = acc

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                predicted_source = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    predicted_source,
                    self._task_compute_predicted_source,
                    i_b,
                )

                if qd.static(self._has_boundary):
                    # BWJ23 eq. (34): the pressure acceleration of a boundary particle is the one of its link, zero for
                    # a link that is not coupled.
                    for k in range(N_BOUNDARY_SLOTS):
                        if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                            acc_rel = self.particles_reordered[i_p, i_b].pressure_acc
                            if qd.static(self._has_rigid_coupling):
                                acc_rel -= self._func_coupled_point_acc(i_p, i_b, k)
                            predicted_source += acc_rel.dot(self.particles_boundary_slots[i_p, i_b, k].grad)

                self.particles_reordered[i_p, i_b].predicted_source = predicted_source

                source = -self.particles_reordered[i_p, i_b].density_adv_norm
                if qd.static(not is_divergence):
                    source = 1.0 + source
                residual = source - dt_pow * predicted_source
                error = -qd.min(residual, 0.0)

                p_rho2 = self._func_pressure_density2_norm(i_p, i_b, is_divergence)
                factor = self.particles_reordered[i_p, i_b].alpha_factor_adj
                p_rho2 = qd.max(p_rho2 - 0.5 * residual * factor / dt_pow, 0.0)
                if qd.static(is_divergence):
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_d = p_rho2
                    if self.particles_reordered[i_p, i_b].n_neighbors < self._df_min_div_neighbors:
                        error = 0.0
                else:
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_p = p_rho2

                self.envs_solver_error[i_b] += error

        max_avg_error = gs.qd_float(0.0)
        for i_b in range(self._B):
            n_active = qd.max(self.envs_n_active[i_b], 1)
            qd.atomic_max(max_avg_error, self.envs_solver_error[i_b] / n_active)
        return max_avg_error

    @qd.kernel
    def _kernel_apply_pressure_acc(self, is_divergence: qd.template()):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self._func_compute_pressure_acc(i_p, i_b, is_divergence)
                pressure_vel = self._substep_dt * self.particles_reordered[i_p, i_b].pressure_acc
                self.particles_reordered[i_p, i_b].vel += pressure_vel
                if qd.static(self._has_viscosity and not is_divergence):
                    self.particles_viscosity[i_p, i_b].pressure_vel_delta = (
                        pressure_vel - self.particles_viscosity[i_p, i_b].pressure_vel
                    )
                    self.particles_viscosity[i_p, i_b].pressure_vel = pressure_vel

        if qd.static(self._has_rigid_coupling):
            # The impulse of the converged pressure on the coupled links, applied once to their velocities.
            n_coupled_6 = 6 * self.coupled_links_idx.shape[0]
            for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
                self.coupled_wrench[i_r, i_b] = 0.0
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles_reordered[i_p, i_b].active:
                    i_m = self.particles_reordered[i_p, i_b].material
                    mass_kappa = self.materials[i_m].mass * self._func_pressure_density2_norm(i_p, i_b, is_divergence)
                    self._func_add_coupled_wrench(i_p, i_b, mass_kappa)
            for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
                impulse = self._substep_dt * self.coupled_wrench[i_r, i_b]
                if qd.static(is_divergence):
                    self.coupled_impulse_div[i_r, i_b] = impulse
                else:
                    self.coupled_impulse_den[i_r, i_b] = impulse
            for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
                vel = self.coupled_vel_base[i_r, i_b]
                for i_col in range(n_coupled_6):
                    vel += self._substep_dt * self.coupled_mobility[i_r, i_col, i_b] * self.coupled_wrench[i_col, i_b]
                self.coupled_vel[i_r, i_b] = vel
            for i_c, i_b in qd.ndrange(self.coupled_links_idx.shape[0], self._B):
                self._func_set_coupled_links_vel(i_c, i_b)

    def _pressure_solve(self, is_divergence, keep_pressure=False):
        """
        Run the divergence solve or the density solve and apply the resulting pressure accelerations to velocities.

        Iterations continue until the average error of every environment drops below the tolerance, within the
        min / max iteration bounds. keep_pressure starts the density solve from the pressure it holds instead of the
        warm start, which a later coupling pass of a substep does. Such a pass also has to reduce its starting error
        tenfold, since the coupling passes only converge while their density solves keep reducing their own error
        (inexact iteration with a forcing term of 0.1), down to a floor of a thousandth of the tolerance. Returns the
        number of iterations and the final error over the tolerance (at most 1 once converged).
        """
        if is_divergence:
            n_iters_min, n_iters_max = self._df_min_div_iters, self._df_max_div_iters
            # The divergence error is a rate, so its tolerance is the density tolerance per substep.
            max_error = self._df_max_error_div / self._substep_dt
        else:
            n_iters_min, n_iters_max = self._df_min_den_iters, self._df_max_den_iters
            max_error = self._df_max_error_den

        self._kernel_init_pressure_solve(is_divergence, keep_pressure)

        n_iters, avg_error, is_converged = 0, 0.0, False
        target_error = max_error
        while (not is_converged or n_iters < n_iters_min) and n_iters < n_iters_max:
            avg_error = self._kernel_pressure_solve_iteration(is_divergence)
            if keep_pressure and n_iters == 0:
                target_error = max(min(max_error, 0.1 * avg_error), 1e-3 * max_error)
            is_converged = avg_error <= target_error
            n_iters += 1

        gs.logger.debug(
            f"DFSPH {'divergence' if is_divergence else 'density'} solve: {n_iters} iterations, avg error {avg_error:.4g}"
        )

        self._kernel_apply_pressure_acc(is_divergence)
        return n_iters, avg_error / max_error

    # ------------------------------------------------------------------------------------
    # -------------------------------- implicit viscosity --------------------------------
    # ------------------------------------------------------------------------------------
    # Backward Euler step of the Rayleigh dissipation function of SPH_SOLVER_DESIGN.md section 3.3: the viscous pair
    # force between i and j is -k_ij ((v_i - v_j) . x_ij) x_ij / r_ij^2, along x_ij, so the operator conserves linear
    # and angular momentum and its null modes are the rigid motions. The solve (M + h D) v = M v_rhs is symmetric
    # positive definite, so it runs as conjugate gradient with the 3x3 diagonal blocks as preconditioner. The walls
    # enter with the velocity of their boundary particles, which gives the no-slip condition.

    @qd.func
    def _func_viscosity_pair_coefficient(self, i, j, x_ij, i_b):
        """Return k_ij / r_ij^2 of the fluid pair (i, j), in N s / m^3, with the harmonic mean of their viscosities."""
        i_m_i = self.particles_reordered[i, i_b].material
        i_m_j = self.particles_reordered[j, i_b].material
        mu_i = self.materials[i_m_i].dynamic_viscosity
        mu_j = self.materials[i_m_j].dynamic_viscosity
        res = gs.qd_float(0.0)
        r2 = x_ij.norm_sqr()
        if mu_i * mu_j > 0.0 and r2 > self._df_eps * self._support_radius**2:
            mu_ij = 2.0 * mu_i * mu_j / (mu_i + mu_j)
            scale_ij = 0.5 * (
                self.particles_viscosity[i, i_b].viscosity_scale + self.particles_viscosity[j, i_b].viscosity_scale
            )
            res = (
                2.0
                * (3.0 + 2.0)
                * scale_ij
                * mu_ij
                * self._particle_volume**2
                * (-self.cubic_kernel_derivative(x_ij).dot(x_ij))
                / ((r2 + 0.01 * self._support_radius**2) * r2)
            )
        return res

    @qd.func
    def _func_wall_viscosity_tensor(self, i, j, i_b):
        """Return the wall viscosity tensor of fluid particle i due to boundary particle j, in N s / m.

        A boundary particle stands for the column of solid behind its share of the geom surface, which a no-slip wall
        fills with ghost fluid whose velocity extrapolates linearly the velocity of the fluid particle relative to the
        wall, to the opposite value at the mirrored distance: the ghost at depth s behind the surface, with the fluid
        particle at distance d in front of it, moves relative to the wall by -(s / d) (v_i - v_wall). The column is
        integrated by the midpoint rule over the kernel support, each ghost acting on particle i through the pair
        dissipation of the fluid (SPH_SOLVER_DESIGN.md section 3.3) with its relative velocity (1 + s / d) (v_i -
        v_wall). The extrapolation holds for the tangential velocity alone: at an impermeable no-slip wall the normal
        velocity of an incompressible flow grows quadratically with the distance, so its viscous coupling to the wall
        vanishes to leading order, and the pressure carries the impermeability. The tensor is therefore projected on the
        tangent plane of the wall at the boundary particle. It is symmetric positive semi-definite. Returns the second
        moment of the ghost column as well, for the renormalization of the viscosity operator (see
        _kernel_compute_viscosity_geometry).
        """
        normal = self.boundary_particles_reordered[j, i_b].normal
        surface_pos = self.boundary_particles_reordered[j, i_b].pos + self._boundary_offset * normal
        dist = (self.particles_reordered[i, i_b].pos - surface_pos).dot(normal)
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        moment = gs.qd_float(0.0)
        # Columns of surfaces the fluid particle lies behind (around a convex edge) do not face it.
        if dist > 0.0:
            # A fluid particle pressed against the wall gets the friction of a particle at a twentieth of the support
            # radius, which bounds the extrapolation factor.
            dist_eff = qd.max(dist, 0.05 * self._support_radius)
            i_m = self.particles_reordered[i, i_b].material
            mu_i = self.materials[i_m].dynamic_viscosity
            column_volume = self.boundary_particles_reordered[j, i_b].area * self._support_radius / 8.0
            for i_q in range(8):
                depth = (i_q + 0.5) / 8.0 * self._support_radius
                x_ig = self.particles_reordered[i, i_b].pos - (surface_pos - depth * normal)
                r2 = x_ig.norm_sqr()
                if r2 > self._df_eps * self._support_radius**2:
                    weight = (
                        column_volume
                        * (-self.cubic_kernel_derivative(x_ig).dot(x_ig))
                        / (r2 + 0.01 * self._support_radius**2)
                    )
                    moment += weight * r2
                    tensor += (
                        2.0
                        * (3.0 + 2.0)
                        * mu_i
                        * self._particle_volume
                        * weight
                        / r2
                        * (1.0 + depth / dist_eff)
                        * x_ig.outer_product(x_ig)
                    )
            tangent_proj = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
            tensor = tangent_proj @ tensor @ tangent_proj
        return tensor, moment

    @qd.func
    def _task_compute_viscosity_moment(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        r2 = x_ij.norm_sqr()
        ret += (
            self._particle_volume
            * (-self.cubic_kernel_derivative(x_ij).dot(x_ij))
            * r2
            / (r2 + 0.01 * self._support_radius**2)
        )

    @qd.func
    def _task_compute_boundary_viscosity_geometry(self, i, j, ret: qd.template(), i_b):
        tensor, moment = self._func_wall_viscosity_tensor(i, j, i_b)
        ret += moment
        i_l = self.boundary_particles_reordered[j, i_b].link_idx
        for k in range(N_BOUNDARY_SLOTS):
            if self.particles_boundary_slots[i, i_b, k].link_idx == i_l:
                self.particles_boundary_slots[i, i_b, k].wall_tensor += tensor

    @qd.kernel
    def _kernel_compute_viscosity_geometry(self):
        """Compute the renormalization of the viscosity operator and the wall viscosity tensors of every particle.

        The pair dissipation of SPH_SOLVER_DESIGN.md section 3.3 reproduces the Laplacian of a quadratic velocity field
        only if the fourth moment of the kernel gradient over the neighbors, sum_j V_j (-grad W_ij . x_ij) r_ij^2 /
        (r_ij^2 + eta^2), equals its continuum value 3, the ghost fluid of the wall columns (see
        _func_wall_viscosity_tensor) completing the neighborhood. A kernel support of two particle spacings holds too
        few neighbors for this. The scale of a particle is 3 times its normalized density over that moment, so that the
        operator is consistent in trace for any arrangement of the particles, and the density in the numerator keeps the
        scale of a particle at the free surface at its value in the bulk, since its missing neighbors carry no velocity
        difference. The scale multiplies the viscous coefficients of the particle.

        The wall viscosity tensor of each boundary slot is the sum of the ghost column tensors of the link (see
        _func_wall_viscosity_tensor) times the scale, and the wall force on the particle is -M (v_i - v_wall(x_i)) per
        slot.
        """
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                moment = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    moment,
                    self._task_compute_viscosity_moment,
                    i_b,
                )
                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        self.particles_boundary_slots[i_p, i_b, k].wall_tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        moment,
                        self._task_compute_boundary_viscosity_geometry,
                        i_b,
                    )
                scale = gs.qd_float(1.0)
                if moment > self._df_eps:
                    scale = 3.0 * self.particles_reordered[i_p, i_b].density_norm / moment
                self.particles_viscosity[i_p, i_b].viscosity_scale = scale

                wall_tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        slot_tensor = scale * self.particles_boundary_slots[i_p, i_b, k].wall_tensor
                        self.particles_boundary_slots[i_p, i_b, k].wall_tensor = slot_tensor
                        wall_tensor += slot_tensor
                self.particles_viscosity[i_p, i_b].wall_tensor = wall_tensor

    @qd.func
    def _func_link_vel_at_pos(self, i_l, i_b, pos):
        """Return the rigid velocity of link i_l at a point of space, at the current substep."""
        return self.links_vel[i_l, i_b] + self.links_ang[i_l, i_b].cross(pos - self.links_root_com[i_l, i_b])

    @qd.func
    def _task_init_viscosity_solve(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        k_ij = self._func_viscosity_pair_coefficient(i, j, x_ij, i_b)
        v_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel
        ret.image += self._substep_dt * k_ij * v_ij.dot(x_ij) * x_ij
        ret.block += self._substep_dt * k_ij * x_ij.outer_product(x_ij)
        # The diagonal and the off-diagonal blocks of the pair both have the norm h k_ij r_ij^2.
        ret.row_norm += 2.0 * self._substep_dt * k_ij * x_ij.norm_sqr()

    @qd.kernel
    def _kernel_init_viscosity_solve(self, is_first_pass: bool):
        """Start the conjugate gradient solve of the viscosity from the current velocity.

        The right-hand side velocity is v* plus the velocity change of the last density solve (SPH_SOLVER_DESIGN.md
        section 5.2). is_first_pass marks the first coupling pass of the substep, whose initial residual (the viscous
        impulse of the substep) scales the tolerance of the agreement between viscosity and pressure.
        """
        for i_b in range(self._B):
            self.envs_viscosity[i_b].residual_norm2 = 0.0
            self.envs_viscosity[i_b].residual_norm2_init = 0.0
            self.envs_viscosity[i_b].rhs_norm2 = 0.0
            if is_first_pass:
                self.envs_viscosity[i_b].gravity_norm2 = 0.0
                self.envs_viscosity[i_b].viscous_stiffness_bound = 1.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                i_m = self.particles_reordered[i_p, i_b].material
                mass = self.materials[i_m].mass
                vel_rhs = self.particles_viscosity[i_p, i_b].vel_star + self.particles_viscosity[i_p, i_b].pressure_vel
                ret = qd.Struct(
                    image=mass * self.particles_reordered[i_p, i_b].vel,
                    block=mass * qd.Matrix.identity(gs.qd_float, 3),
                    rhs=mass * vel_rhs,
                    row_norm=gs.qd_float(0.0),
                )
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, ret, self._task_init_viscosity_solve, i_b
                )
                # The wall acts on the particle alone (see _kernel_compute_viscosity_geometry), so it only adds to the
                # diagonal block, and the wall velocity is known in the solve, so its part goes to the right-hand side.
                wall_block = self._substep_dt * self.particles_viscosity[i_p, i_b].wall_tensor
                ret.block += wall_block
                ret.image += wall_block @ self.particles_reordered[i_p, i_b].vel
                ret.row_norm += wall_block.trace()
                if qd.static(self._has_boundary):
                    for k in range(N_BOUNDARY_SLOTS):
                        if self.particles_boundary_slots[i_p, i_b, k].link_idx != -1:
                            ret.rhs += self._substep_dt * (
                                self.particles_boundary_slots[i_p, i_b, k].wall_tensor
                                @ self._func_slot_wall_vel(i_p, i_b, k)
                            )
                precond_inv = ret.block.inverse()
                residual = ret.rhs - ret.image
                residual_precond = precond_inv @ residual
                self.particles_viscosity[i_p, i_b].precond_inv = precond_inv
                self.particles_viscosity[i_p, i_b].residual = residual
                self.particles_viscosity[i_p, i_b].residual_precond = residual_precond
                self.particles_viscosity[i_p, i_b].direction = residual_precond
                self.envs_viscosity[i_b].residual_norm2 += residual.dot(residual_precond)
                self.envs_viscosity[i_b].rhs_norm2 += ret.rhs.dot(precond_inv @ ret.rhs)
                if is_first_pass:
                    # The velocity change of gravity over the substep, in the norm of the mass, so that the scale does
                    # not shrink with the viscosity.
                    self.envs_viscosity[i_b].gravity_norm2 += mass * (self._substep_dt * self._gravity[i_b]).norm_sqr()
                    qd.atomic_max(self.envs_viscosity[i_b].viscous_stiffness_bound, 1.0 + ret.row_norm / mass)

        for i_b in range(self._B):
            self.envs_viscosity[i_b].residual_norm2_init = self.envs_viscosity[i_b].residual_norm2
            if is_first_pass:
                self.envs_viscosity[i_b].viscous_norm2 = self.envs_viscosity[i_b].residual_norm2

    @qd.func
    def _task_apply_viscosity_operator(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        k_ij = self._func_viscosity_pair_coefficient(i, j, x_ij, i_b)
        d_ij = self.particles_viscosity[i, i_b].direction - self.particles_viscosity[j, i_b].direction
        ret += self._substep_dt * k_ij * d_ij.dot(x_ij) * x_ij

    @qd.func
    def _func_viscosity_tolerance2(self, i_b):
        """Return the squared tolerance on the preconditioned residual norm of the viscosity solve of env i_b.

        The tolerance is relative to the initial residual, which is the viscous impulse of the substep, with a floor
        relative to the momentum that single precision resolves.
        """
        return qd.max(
            self._visc_max_error**2 * self.envs_viscosity[i_b].residual_norm2_init,
            1e-12 * self.envs_viscosity[i_b].rhs_norm2,
        )

    @qd.kernel
    def _kernel_viscosity_solve_iteration(self) -> float:
        """Run one conjugate gradient iteration of the viscosity solve and return the worst relative residual over envs.

        The relative residual is the residual norm over its tolerance, so the solve has converged once it is at most 1.
        An env that has converged keeps its velocity.
        """
        for i_b in range(self._B):
            self.envs_viscosity[i_b].direction_norm2 = 0.0
            self.envs_viscosity[i_b].residual_norm2_next = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            is_iterating = self.envs_viscosity[i_b].residual_norm2 > self._func_viscosity_tolerance2(i_b)
            if self.particles_reordered[i_p, i_b].active and is_iterating:
                direction = self.particles_viscosity[i_p, i_b].direction
                i_m = self.particles_reordered[i_p, i_b].material
                image = self.materials[i_m].mass * direction
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    image,
                    self._task_apply_viscosity_operator,
                    i_b,
                )
                if qd.static(self._has_boundary):
                    image += self._substep_dt * (self.particles_viscosity[i_p, i_b].wall_tensor @ direction)
                self.particles_viscosity[i_p, i_b].direction_image = image
                self.envs_viscosity[i_b].direction_norm2 += direction.dot(image)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            is_iterating = self.envs_viscosity[i_b].residual_norm2 > self._func_viscosity_tolerance2(i_b)
            if self.particles_reordered[i_p, i_b].active and is_iterating:
                step = gs.qd_float(0.0)
                if self.envs_viscosity[i_b].direction_norm2 > 0.0:
                    step = self.envs_viscosity[i_b].residual_norm2 / self.envs_viscosity[i_b].direction_norm2
                self.particles_reordered[i_p, i_b].vel += step * self.particles_viscosity[i_p, i_b].direction
                residual = (
                    self.particles_viscosity[i_p, i_b].residual
                    - step * self.particles_viscosity[i_p, i_b].direction_image
                )
                residual_precond = self.particles_viscosity[i_p, i_b].precond_inv @ residual
                self.particles_viscosity[i_p, i_b].residual = residual
                self.particles_viscosity[i_p, i_b].residual_precond = residual_precond
                self.envs_viscosity[i_b].residual_norm2_next += residual.dot(residual_precond)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            is_iterating = self.envs_viscosity[i_b].residual_norm2 > self._func_viscosity_tolerance2(i_b)
            if self.particles_reordered[i_p, i_b].active and is_iterating:
                beta = self.envs_viscosity[i_b].residual_norm2_next / self.envs_viscosity[i_b].residual_norm2
                self.particles_viscosity[i_p, i_b].direction = (
                    self.particles_viscosity[i_p, i_b].residual_precond
                    + beta * self.particles_viscosity[i_p, i_b].direction
                )

        max_rel_residual = gs.qd_float(0.0)
        for i_b in range(self._B):
            if self.envs_viscosity[i_b].residual_norm2 > self._func_viscosity_tolerance2(i_b):
                self.envs_viscosity[i_b].residual_norm2 = self.envs_viscosity[i_b].residual_norm2_next
            qd.atomic_max(
                max_rel_residual,
                qd.sqrt(self.envs_viscosity[i_b].residual_norm2 / qd.max(self._func_viscosity_tolerance2(i_b), 1e-30)),
            )
        return max_rel_residual

    @qd.kernel
    def _kernel_viscosity_residual(self) -> float:
        """Return the worst relative residual over envs of the viscosity solve as it stands, without iterating."""
        max_rel_residual = gs.qd_float(0.0)
        for i_b in range(self._B):
            qd.atomic_max(
                max_rel_residual,
                qd.sqrt(self.envs_viscosity[i_b].residual_norm2 / qd.max(self._func_viscosity_tolerance2(i_b), 1e-30)),
            )
        return max_rel_residual

    def _viscosity_solve(self, is_first_pass):
        """Solve the implicit viscosity of the substep from the current velocity, in place.

        Returns the number of iterations and the final relative residual (at most 1 once converged).
        """
        self._kernel_init_viscosity_solve(is_first_pass)
        rel_residual = self._kernel_viscosity_residual()
        n_iters = 0
        while rel_residual > 1.0 and n_iters < self._visc_max_iters:
            rel_residual = self._kernel_viscosity_solve_iteration()
            n_iters += 1
        return n_iters, rel_residual

    @qd.func
    def _task_apply_viscosity_operator_to_pressure_delta(self, i, j, ret: qd.template(), i_b):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        k_ij = self._func_viscosity_pair_coefficient(i, j, x_ij, i_b)
        d_ij = self.particles_viscosity[i, i_b].pressure_vel_delta - self.particles_viscosity[j, i_b].pressure_vel_delta
        ret += self._substep_dt * k_ij * d_ij.dot(x_ij) * x_ij

    @qd.kernel
    def _kernel_coupling_residual(self) -> float:
        """Return the worst relative disagreement over envs between the viscosity and the last density solve.

        A density solve that changed the pressure velocity by delta leaves the residual -h D delta in the viscosity
        system of the combined velocity (SPH_SOLVER_DESIGN.md section 5.2). It is measured relative to the larger of the
        viscous and the gravity impulses of the substep, with the floor of the viscosity solve, so the coupling has
        converged once the result is at most 1.
        """
        for i_b in range(self._B):
            self.envs_viscosity[i_b].coupling_norm2 = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                residual = qd.Vector.zero(gs.qd_float, 3)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    residual,
                    self._task_apply_viscosity_operator_to_pressure_delta,
                    i_b,
                )
                if qd.static(self._has_boundary):
                    residual += self._substep_dt * (
                        self.particles_viscosity[i_p, i_b].wall_tensor
                        @ self.particles_viscosity[i_p, i_b].pressure_vel_delta
                    )
                self.envs_viscosity[i_b].coupling_norm2 += residual.dot(
                    self.particles_viscosity[i_p, i_b].precond_inv @ residual
                )

        max_rel_residual = gs.qd_float(0.0)
        for i_b in range(self._B):
            impulse_norm2 = qd.max(self.envs_viscosity[i_b].viscous_norm2, self.envs_viscosity[i_b].gravity_norm2)
            tolerance2 = qd.max(self._visc_max_error**2 * impulse_norm2, 1e-12 * self.envs_viscosity[i_b].rhs_norm2)
            qd.atomic_max(
                max_rel_residual, qd.sqrt(self.envs_viscosity[i_b].coupling_norm2 / qd.max(tolerance2, 1e-30))
            )
        return max_rel_residual

    @qd.kernel
    def _kernel_store_coupling_pressure(self, is_first_pass: bool):
        """Record the pressure a coupling pass starts from, which the viscosity solve of the pass reads.

        The first pass starts from zero pressure, since its viscosity solve sees no pressure velocity.
        """
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                pressure_in = self.particles_reordered[i_p, i_b].pressure_density2_norm_p
                if is_first_pass:
                    pressure_in = 0.0
                self.particles_viscosity[i_p, i_b].pressure_in = pressure_in

    @qd.kernel
    def _kernel_accelerate_coupling(self, i_pass: int):
        """Extrapolate the pressure of the next coupling pass by Chebyshev acceleration, and its pressure velocity.

        A coupling pass maps the pressure it starts from to the pressure of its density solve. With the set of compressed
        particles fixed, this map is a Richardson step on the pressure system G A^-1 G^T of the coupled viscosity and
        pressure, preconditioned by the density solve G M^-1 G^T. The eigenvalues of the preconditioned system lie in
        [1 / L, 1], where L bounds the eigenvalues of M^-1 A (Gershgorin disks of the viscosity system), since
        M / L <= A^-1 <= M^-1 in the ordering of symmetric matrices. Chebyshev semi-iteration on that interval converges
        at the rate (sqrt(L) - 1) / (sqrt(L) + 1) instead of 1 - 1 / L, and reduces to the plain iteration as L tends to
        1 for weakly viscous fluids. i_pass is the index of the pass that just ran. The pressure is clamped to stay
        nonnegative.
        """
        for i_b in range(self._B):
            if i_pass == 0:
                bound = self.envs_viscosity[i_b].viscous_stiffness_bound
                self.envs_viscosity[i_b].chebyshev_step = 2.0 / (1.0 + 1.0 / bound)
                self.envs_viscosity[i_b].chebyshev_ratio2 = ((1.0 - 1.0 / bound) / (1.0 + 1.0 / bound)) ** 2
                self.envs_viscosity[i_b].chebyshev_weight = 1.0
            elif i_pass == 1:
                self.envs_viscosity[i_b].chebyshev_weight = 1.0 / (
                    1.0 - 0.5 * self.envs_viscosity[i_b].chebyshev_ratio2
                )
            else:
                self.envs_viscosity[i_b].chebyshev_weight = 1.0 / (
                    1.0 - 0.25 * self.envs_viscosity[i_b].chebyshev_ratio2 * self.envs_viscosity[i_b].chebyshev_weight
                )

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                pressure_in = self.particles_viscosity[i_p, i_b].pressure_in
                pressure_in_prev = self.particles_viscosity[i_p, i_b].pressure_in_prev
                if i_pass == 0:
                    pressure_in_prev = pressure_in
                correction = self.envs_viscosity[i_b].chebyshev_step * (
                    self.particles_reordered[i_p, i_b].pressure_density2_norm_p - pressure_in
                )
                pressure = (
                    self.envs_viscosity[i_b].chebyshev_weight * (pressure_in - pressure_in_prev + correction)
                    + pressure_in_prev
                )
                self.particles_viscosity[i_p, i_b].pressure_in_prev = pressure_in
                self.particles_reordered[i_p, i_b].pressure_density2_norm_p = qd.max(pressure, 0.0)

        # The pressure velocity of the extrapolated pressure, which the viscosity solve of the next pass reads
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self._func_compute_pressure_acc(i_p, i_b, False)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self.particles_viscosity[i_p, i_b].pressure_vel = (
                    self._substep_dt * self.particles_reordered[i_p, i_b].pressure_acc
                )

    @qd.kernel
    def _kernel_remove_pressure_vel(self):
        """Remove the velocity change of the last density solve, so that the next one starts from the viscous velocity."""
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                self.particles_reordered[i_p, i_b].vel -= self.particles_viscosity[i_p, i_b].pressure_vel

    # ------------------------------------------------------------------------------------
    # ------------------------------------ fluid load ------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_accumulate_boundary_load(self, is_step_start: bool):
        """Add the fluid impulse of the substep on every rigid link, from the final pressures and velocities.

        The pressure impulse of the divergence and the density solves (SPH_SOLVER_DESIGN.md section 3.2) and the viscous
        impulse of each boundary slot act at the fluid particle (see _kernel_compute_density_and_factor and
        _kernel_compute_viscosity_geometry). is_step_start resets the accumulation, which spans the substeps of one
        scene step.
        """
        if is_step_start:
            for i_l, i_b in qd.ndrange(self.links_fluid_impulse.shape[0], self._B):
                self.links_fluid_impulse[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
                self.links_fluid_angular_impulse[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                kappa = self.particles_reordered[i_p, i_b].pressure_density2_norm_p
                if qd.static(not self._df_skip_div):
                    kappa += self.particles_reordered[i_p, i_b].pressure_density2_norm_d
                i_m = self.particles_reordered[i_p, i_b].material
                mass_kappa = self.materials[i_m].mass * kappa
                pos_i = self.particles_reordered[i_p, i_b].pos
                for k in range(N_BOUNDARY_SLOTS):
                    i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                    if i_l != -1:
                        impulse = self._substep_dt * mass_kappa * self.particles_boundary_slots[i_p, i_b, k].grad
                        if qd.static(self._has_viscosity):
                            vel_rel = self.particles_reordered[i_p, i_b].vel - self._func_slot_wall_vel(i_p, i_b, k)
                            impulse += self._substep_dt * (
                                self.particles_boundary_slots[i_p, i_b, k].wall_tensor @ vel_rel
                            )
                        self.links_fluid_impulse[i_l, i_b] += impulse
                        self.links_fluid_angular_impulse[i_l, i_b] += (pos_i - self.links_com[i_l, i_b]).cross(impulse)

    # ------------------------------------------------------------------------------------
    # ---------------------------------- rigid coupling ----------------------------------
    # ------------------------------------------------------------------------------------
    # The fluid stage of a substep runs inside the substep of the rigid solver, between its forward dynamics and its
    # constraint solve (SPH_SOLVER_DESIGN.md section 4): the pressure and viscosity solves see the response of the coupled
    # links through their mobility, and their net impulse enters the smooth force of the rigid solver, so that the
    # constraint solve and the integration of the rigid solver take the fluid load of the same substep.

    @qd.kernel
    def _kernel_set_coupled_links(self, links_coupled_idx: qd.types.ndarray(), coupled_links_idx: qd.types.ndarray()):
        for i_l in range(self.links_coupled_idx.shape[0]):
            self.links_coupled_idx[i_l] = links_coupled_idx[i_l]
        for i_c in range(self.coupled_links_idx.shape[0]):
            self.coupled_links_idx[i_c] = coupled_links_idx[i_c]

    @qd.func
    def _func_chain_spatial_vel(
        self,
        i_c: int,
        i_b: int,
        vec: qd.Tensor,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_config: qd.template(),
    ):
        """Return the spatial velocity (angular and linear part) of coupled link i_c for the dof velocities vec."""
        ang = qd.Vector.zero(gs.qd_float, 3)
        lin = qd.Vector.zero(gs.qd_float, 3)
        i_l = self.coupled_links_idx[i_c]
        for _ in range(self.links_coupled_idx.shape[0]):
            if i_l != -1:
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                    ang += dyn_state.dofs.cdof_ang[i_d, i_b] * vec[i_d, i_b]
                    lin += dyn_state.dofs.cdof_vel[i_d, i_b] * vec[i_d, i_b]
                i_l = dyn_info.links.parent_idx[I_l]
        return ang, lin

    @qd.func
    def _func_chain_add_wrench(
        self,
        i_c: int,
        i_b: int,
        torque: qd.types.vector(3),
        force: qd.types.vector(3),
        vec: qd.Tensor,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_config: qd.template(),
    ):
        """Add to the dof forces vec the generalized force of a wrench on coupled link i_c (torque about the center of
        mass of its kinematic tree)."""
        i_l = self.coupled_links_idx[i_c]
        for _ in range(self.links_coupled_idx.shape[0]):
            if i_l != -1:
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                    vec[i_d, i_b] += dyn_state.dofs.cdof_ang[i_d, i_b].dot(torque) + dyn_state.dofs.cdof_vel[
                        i_d, i_b
                    ].dot(force)
                i_l = dyn_info.links.parent_idx[I_l]

    @qd.kernel
    def _kernel_init_rigid_coupling(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        rigid_config: qd.template(),
    ):
        """Compute the mobility of the coupled links and take their spatial velocities at the start of the substep.

        Column (c, s) of the mobility is the spatial acceleration of every coupled link for a unit wrench component s
        on link c, solved with the mass matrix factor of the forward dynamics of the substep.
        """
        n_coupled = self.coupled_links_idx.shape[0]
        n_dofs = self.dofs_coupling_rhs.shape[0]
        for i_b in range(self._B):
            for i_col in range(6 * n_coupled):
                i_c = i_col // 6
                i_s = i_col % 6
                for i_d in range(n_dofs):
                    self.dofs_coupling_rhs[i_d, i_b] = 0.0
                torque = qd.Vector(
                    [qd.cast(i_s == 0, gs.qd_float), qd.cast(i_s == 1, gs.qd_float), qd.cast(i_s == 2, gs.qd_float)]
                )
                force = qd.Vector(
                    [qd.cast(i_s == 3, gs.qd_float), qd.cast(i_s == 4, gs.qd_float), qd.cast(i_s == 5, gs.qd_float)]
                )
                self._func_chain_add_wrench(
                    i_c, i_b, torque, force, self.dofs_coupling_rhs, dyn_state, dyn_info, rigid_config
                )
                func_solve_mass_batch(
                    i_b, self.dofs_coupling_rhs, self.dofs_coupling_sol, dyn_state, rigid_info, rigid_config
                )
                for i_r in range(n_coupled):
                    ang, lin = self._func_chain_spatial_vel(
                        i_r, i_b, self.dofs_coupling_sol, dyn_state, dyn_info, rigid_config
                    )
                    for k in qd.static(range(3)):
                        self.coupled_mobility[6 * i_r + k, i_col, i_b] = ang[k]
                        self.coupled_mobility[6 * i_r + 3 + k, i_col, i_b] = lin[k]

        for i_c, i_b in qd.ndrange(n_coupled, self._B):
            i_l = self.coupled_links_idx[i_c]
            for k in qd.static(range(3)):
                self.coupled_vel[6 * i_c + k, i_b] = dyn_state.links.cd_ang[i_l, i_b][k]
                self.coupled_vel[6 * i_c + 3 + k, i_b] = dyn_state.links.cd_vel[i_l, i_b][k]
            for k in range(6):
                self.coupled_vel_base[6 * i_c + k, i_b] = self.coupled_vel[6 * i_c + k, i_b]
                self.coupled_impulse_div[6 * i_c + k, i_b] = 0.0
                self.coupled_impulse_visc[6 * i_c + k, i_b] = 0.0
                self.coupled_impulse_den[6 * i_c + k, i_b] = 0.0

    @qd.func
    def _func_set_coupled_links_vel(self, i_c, i_b):
        """Copy the spatial velocity the fluid sees for coupled link i_c into the velocity fields of the links."""
        i_l = self.coupled_links_idx[i_c]
        self.links_ang[i_l, i_b] = qd.Vector(
            [self.coupled_vel[6 * i_c, i_b], self.coupled_vel[6 * i_c + 1, i_b], self.coupled_vel[6 * i_c + 2, i_b]]
        )
        self.links_vel[i_l, i_b] = qd.Vector(
            [self.coupled_vel[6 * i_c + 3, i_b], self.coupled_vel[6 * i_c + 4, i_b], self.coupled_vel[6 * i_c + 5, i_b]]
        )

    @qd.kernel
    def _kernel_set_coupled_vel_star(
        self, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo, rigid_config: qd.template()
    ):
        """Set the coupled velocities to their value after the divergence impulse and the smooth acceleration."""
        for i_c, i_b in qd.ndrange(self.coupled_links_idx.shape[0], self._B):
            ang, lin = self._func_chain_spatial_vel(
                i_c, i_b, dyn_state.dofs.acc_smooth, dyn_state, dyn_info, rigid_config
            )
            for k in qd.static(range(3)):
                self.coupled_vel[6 * i_c + k, i_b] += self._substep_dt * ang[k]
                self.coupled_vel[6 * i_c + 3 + k, i_b] += self._substep_dt * lin[k]
            for k in range(6):
                self.coupled_vel_star[6 * i_c + k, i_b] = self.coupled_vel[6 * i_c + k, i_b]
            self._func_set_coupled_links_vel(i_c, i_b)

    @qd.kernel
    def _kernel_set_coupled_vel_base(self, with_viscosity: bool):
        """Set the coupled velocities to the base of the density solve: v* plus the response to the viscous impulse."""
        n_coupled_6 = 6 * self.coupled_links_idx.shape[0]
        for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
            vel = self.coupled_vel_star[i_r, i_b]
            if with_viscosity:
                for i_col in range(n_coupled_6):
                    vel += self.coupled_mobility[i_r, i_col, i_b] * self.coupled_impulse_visc[i_col, i_b]
            self.coupled_vel_base[i_r, i_b] = vel
            self.coupled_vel[i_r, i_b] = vel
        for i_c, i_b in qd.ndrange(self.coupled_links_idx.shape[0], self._B):
            self._func_set_coupled_links_vel(i_c, i_b)

    @qd.func
    def _func_add_coupled_wrench(self, i_p, i_b, mass_kappa):
        """Add the reaction of the boundary pressure of fluid particle i_p to the wrenches of the coupled links."""
        pos_i = self.particles_reordered[i_p, i_b].pos
        for k in range(N_BOUNDARY_SLOTS):
            i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
            if i_l != -1:
                i_c = self.links_coupled_idx[i_l]
                if i_c != -1:
                    force = mass_kappa * self.particles_boundary_slots[i_p, i_b, k].grad
                    torque = (pos_i - self.links_root_com[i_l, i_b]).cross(force)
                    for j in qd.static(range(3)):
                        self.coupled_wrench[6 * i_c + j, i_b] += torque[j]
                        self.coupled_wrench[6 * i_c + 3 + j, i_b] += force[j]

    @qd.func
    def _func_coupled_point_acc(self, i_p, i_b, k):
        """Return the acceleration of the link of boundary slot k of fluid particle i_p at the particle, due to the
        pressure of the current iteration (zero for a link that is not coupled)."""
        res = qd.Vector.zero(gs.qd_float, 3)
        i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
        i_c = self.links_coupled_idx[i_l]
        if i_c != -1:
            ang = qd.Vector(
                [self.coupled_acc[6 * i_c, i_b], self.coupled_acc[6 * i_c + 1, i_b], self.coupled_acc[6 * i_c + 2, i_b]]
            )
            lin = qd.Vector(
                [
                    self.coupled_acc[6 * i_c + 3, i_b],
                    self.coupled_acc[6 * i_c + 4, i_b],
                    self.coupled_acc[6 * i_c + 5, i_b],
                ]
            )
            res = lin + ang.cross(self.particles_reordered[i_p, i_b].pos - self.links_root_com[i_l, i_b])
        return res

    @qd.kernel
    def _kernel_solve_coupled_viscosity(self) -> float:
        """Solve the velocities of the coupled links under the wall viscosity of the fluid velocities of the pass.

        The wall dissipation of the slots (see _kernel_compute_viscosity_geometry) is quadratic in the spatial
        velocities xi of the coupled links: the wrench of the fluid on them is f - K xi, with K the sum over the slots of
        G^T M G and f that of G^T M v_i, G mapping a spatial velocity to the velocity at the fluid particle. The
        backward Euler step of the links, xi = xi* + Lambda (I_den + h (f - K xi)), is the dense system
        (I + h Lambda K) xi = xi* + Lambda I_den + h Lambda f, solved per environment by Gaussian elimination (its
        matrix is similar to a symmetric positive definite one). Returns the largest change of the coupled velocities
        over envs, relative to their magnitude.
        """
        n_coupled = self.coupled_links_idx.shape[0]
        n_coupled_6 = 6 * n_coupled
        for i_r, i_b in qd.ndrange(n_coupled_6, self._B):
            self.coupled_visc_force[i_r, i_b] = 0.0
            for j in range(6):
                self.coupled_visc_stiffness[i_r, j, i_b] = 0.0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                pos_i = self.particles_reordered[i_p, i_b].pos
                vel_i = self.particles_reordered[i_p, i_b].vel
                for k in range(N_BOUNDARY_SLOTS):
                    i_l = self.particles_boundary_slots[i_p, i_b, k].link_idx
                    if i_l != -1:
                        i_c = self.links_coupled_idx[i_l]
                        if i_c != -1:
                            tensor = self.particles_boundary_slots[i_p, i_b, k].wall_tensor
                            r = pos_i - self.links_root_com[i_l, i_b]
                            # G = [-[r]x, I] maps (angular, linear) to the point velocity, G^T y = (r x y, y)
                            cross_r = qd.Matrix([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
                            force = tensor @ vel_i
                            torque = r.cross(force)
                            block_aa = -(cross_r @ tensor @ cross_r)
                            block_al = cross_r @ tensor
                            for a in qd.static(range(3)):
                                self.coupled_visc_force[6 * i_c + a, i_b] += torque[a]
                                self.coupled_visc_force[6 * i_c + 3 + a, i_b] += force[a]
                                for b in qd.static(range(3)):
                                    self.coupled_visc_stiffness[6 * i_c + a, b, i_b] += block_aa[a, b]
                                    self.coupled_visc_stiffness[6 * i_c + a, 3 + b, i_b] += block_al[a, b]
                                    self.coupled_visc_stiffness[6 * i_c + 3 + a, b, i_b] += block_al[b, a]
                                    self.coupled_visc_stiffness[6 * i_c + 3 + a, 3 + b, i_b] += tensor[a, b]

        max_change = gs.qd_float(0.0)
        for i_b in range(self._B):
            h = self._substep_dt
            # system matrix and right-hand side, Lambda K being block-structured by the column link
            for i_r in range(n_coupled_6):
                rhs = self.coupled_vel_star[i_r, i_b]
                for i_col in range(n_coupled_6):
                    rhs += self.coupled_mobility[i_r, i_col, i_b] * (
                        self.coupled_impulse_den[i_col, i_b] + h * self.coupled_visc_force[i_col, i_b]
                    )
                self.coupled_system[i_r, n_coupled_6, i_b] = rhs
                for i_col in range(n_coupled_6):
                    i_c_col = i_col // 6
                    value = gs.qd_float(0.0)
                    for j in range(6):
                        value += (
                            self.coupled_mobility[i_r, 6 * i_c_col + j, i_b]
                            * self.coupled_visc_stiffness[6 * i_c_col + j, i_col % 6, i_b]
                        )
                    self.coupled_system[i_r, i_col, i_b] = h * value + qd.cast(i_r == i_col, gs.qd_float)
            for i_k in range(n_coupled_6):
                pivot = self.coupled_system[i_k, i_k, i_b]
                for i_r in range(i_k + 1, n_coupled_6):
                    ratio = self.coupled_system[i_r, i_k, i_b] / pivot
                    for i_col in range(i_k, n_coupled_6 + 1):
                        self.coupled_system[i_r, i_col, i_b] -= ratio * self.coupled_system[i_k, i_col, i_b]
            vel_scale = gs.qd_float(0.0)
            change = gs.qd_float(0.0)
            for i_k_ in range(n_coupled_6):
                i_k = n_coupled_6 - 1 - i_k_
                value = self.coupled_system[i_k, n_coupled_6, i_b]
                for i_col in range(i_k + 1, n_coupled_6):
                    value -= self.coupled_system[i_k, i_col, i_b] * self.coupled_system[i_col, n_coupled_6, i_b]
                value = value / self.coupled_system[i_k, i_k, i_b]
                self.coupled_system[i_k, n_coupled_6, i_b] = value
                change = qd.max(change, qd.abs(value - self.coupled_vel[i_k, i_b]))
                vel_scale = qd.max(vel_scale, qd.abs(value))
            for i_r in range(n_coupled_6):
                self.coupled_vel[i_r, i_b] = self.coupled_system[i_r, n_coupled_6, i_b]
            # viscous impulse h (f - K xi) on each link, with its own block of K
            for i_r in range(n_coupled_6):
                i_c = i_r // 6
                value = self.coupled_visc_force[i_r, i_b]
                for j in range(6):
                    value -= self.coupled_visc_stiffness[i_r, j, i_b] * self.coupled_vel[6 * i_c + j, i_b]
                self.coupled_impulse_visc[i_r, i_b] = h * value
            self.envs_coupled_vel_change[i_b] = change / qd.max(vel_scale, gs.EPS)
            qd.atomic_max(max_change, self.envs_coupled_vel_change[i_b])

        for i_c, i_b in qd.ndrange(n_coupled, self._B):
            self._func_set_coupled_links_vel(i_c, i_b)
        return max_change

    @qd.kernel
    def _kernel_apply_coupled_impulse(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        rigid_config: qd.template(),
    ):
        """Add the net impulse of the fluid of the substep to the smooth force and acceleration of the rigid solver."""
        n_coupled = self.coupled_links_idx.shape[0]
        n_dofs = self.dofs_coupling_rhs.shape[0]
        inv_dt = 1.0 / self._substep_dt
        for i_b in range(self._B):
            for i_d in range(n_dofs):
                self.dofs_coupling_rhs[i_d, i_b] = 0.0
            for i_c in range(n_coupled):
                torque = qd.Vector.zero(gs.qd_float, 3)
                force = qd.Vector.zero(gs.qd_float, 3)
                for j in qd.static(range(3)):
                    torque[j] = inv_dt * (
                        self.coupled_impulse_div[6 * i_c + j, i_b]
                        + self.coupled_impulse_visc[6 * i_c + j, i_b]
                        + self.coupled_impulse_den[6 * i_c + j, i_b]
                    )
                    force[j] = inv_dt * (
                        self.coupled_impulse_div[6 * i_c + 3 + j, i_b]
                        + self.coupled_impulse_visc[6 * i_c + 3 + j, i_b]
                        + self.coupled_impulse_den[6 * i_c + 3 + j, i_b]
                    )
                self._func_chain_add_wrench(
                    i_c, i_b, torque, force, self.dofs_coupling_rhs, dyn_state, dyn_info, rigid_config
                )
            func_solve_mass_batch(
                i_b, self.dofs_coupling_rhs, self.dofs_coupling_sol, dyn_state, rigid_info, rigid_config
            )
            for i_d in range(n_dofs):
                dyn_state.dofs.force[i_d, i_b] += self.dofs_coupling_rhs[i_d, i_b]
                dyn_state.dofs.qf_smooth[i_d, i_b] += self.dofs_coupling_rhs[i_d, i_b]
                dyn_state.dofs.acc_smooth[i_d, i_b] += self.dofs_coupling_sol[i_d, i_b]

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def cubic_kernel(self, r_norm):
        """
        Cubic spline smoothing kernel.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q**2
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
            else:
                res = 2 * k * (1.0 - q) ** 3
        return res

    @qd.func
    def cubic_kernel_derivative(self, r):
        """
        Derivative of cubic spline smoothing kernel.
        """
        res = qd.Vector.zero(gs.qd_float, 3)

        r_norm = r.norm()
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
            else:
                res = -6.0 * k * (1.0 - q) ** 2 * grad_q
        return res

    @qd.func
    def cubic_kernel_plane_integral(self, dist):
        """
        Integral of the cubic spline smoothing kernel over the plane at distance dist from its center.

        Closed form of 2 * pi * int_{|dist|}^{h} W(r) r dr. Its own integral along the plane normal equals
        half the kernel normalization: int_{0}^{h} of this quantity over dist is exactly 1/2, which makes
        forces weighted by it integrate to a per-unit-area magnitude over a covering layer of particles.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        q = qd.abs(dist) / h
        if q <= 1.0:
            q2 = q**2
            q3 = q2 * q
            if q <= 0.5:
                res = 0.0875 - (1.2 * q3 * q2 - 1.5 * q2**2 + 0.5 * q2)
            else:
                res = 0.1 - (q2 - 2.0 * q3 + 1.5 * q2**2 - 0.4 * q3 * q2)
            res = res * 16.0 / h
        return res

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self.entities[::-1]:
            entity.process_input_grad()

    @property
    def is_rigid_coupled(self):
        """Whether the fluid stage of a substep runs inside the substep of the rigid solver (see rigid coupling)."""
        return self.is_active and self._has_rigid_coupling

    def substep_coupled_with_rigid(self, f):
        """Run the fluid stage of a substep from the substep of the rigid solver, after its forward dynamics."""
        self._run_fluid_stage(f)

    def substep_pre_coupling(self, f):
        if self.is_active and not self._has_rigid_coupling:
            self._run_fluid_stage(f)

    def _run_fluid_stage(self, f):
        """Run the boundary update, the divergence solve, the explicit forces and the coupled viscosity and density
        solves of a substep (SPH_SOLVER_DESIGN.md section 4), and the transfer of the fluid impulse to the coupled
        links."""
        # f indexes the substep in the memory of the simulator, so the first substep of a step is counted globally.
        is_step_start = self._sim.cur_substep_global % self._sim.substeps == 0
        if is_step_start:
            self._solve_stats = SPHSolveStats()
        stats = self._solve_stats
        stats.n_substeps += 1

        rigid_solver = self.sim.rigid_solver
        if self._has_boundary:
            self._kernel_update_boundary_particles(rigid_solver.dyn_state.links)
        if self._has_rigid_coupling:
            self._kernel_init_rigid_coupling(
                rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_info, rigid_solver.rigid_config
            )
        self._kernel_reorder_particles(f)
        self._kernel_compute_density_and_factor(f)
        if not self._df_skip_div:
            n_iters, residual = self._pressure_solve(is_divergence=True)
            stats.n_divergence_iters += n_iters
            stats.n_unconverged_divergence += residual > 1.0
            stats.max_divergence_residual = max(stats.max_divergence_residual, residual)
        self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
        if self._has_rigid_coupling:
            self._kernel_set_coupled_vel_star(rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_config)

        # Coupled viscosity and density solves, alternated until they agree (SPH_SOLVER_DESIGN.md section 5.2).
        n_passes = 1 if not self._has_viscosity else self._max_coupling_iters
        if self._has_viscosity:
            self._kernel_compute_viscosity_geometry()
        for i_pass in range(n_passes):
            rigid_change = 0.0
            if self._has_viscosity:
                n_iters, residual = self._viscosity_solve(is_first_pass=i_pass == 0)
                stats.n_viscosity_iters += n_iters
                stats.n_unconverged_viscosity += residual > 1.0
                stats.max_viscosity_residual = max(stats.max_viscosity_residual, residual)
                if self._has_rigid_coupling:
                    rigid_change = self._kernel_solve_coupled_viscosity()
                if i_pass > 0:
                    self._kernel_remove_pressure_vel()
                self._kernel_store_coupling_pressure(i_pass == 0)
            if self._has_rigid_coupling:
                self._kernel_set_coupled_vel_base(self._has_viscosity)
            n_iters, residual = self._pressure_solve(is_divergence=False, keep_pressure=i_pass > 0)
            stats.n_density_iters += n_iters
            stats.n_unconverged_density += residual > 1.0
            stats.max_density_residual = max(stats.max_density_residual, residual)
            stats.n_coupling_passes += 1
            if self._has_viscosity:
                # The rigid velocities converge with the fluid ones: their change over the pass, relative to their
                # magnitude, has the tolerance of the viscosity solve.
                coupling_residual = max(self._kernel_coupling_residual(), rigid_change / self._visc_max_error)
                gs.logger.debug(
                    f"SPH coupling pass {i_pass}: viscosity {stats.n_viscosity_iters} iterations in total, density "
                    f"residual {residual:.3g}, coupling residual {coupling_residual:.3g}"
                )
                if coupling_residual <= 1.0 or i_pass == n_passes - 1:
                    stats.n_unconverged_coupling += coupling_residual > 1.0
                    stats.max_coupling_residual = max(stats.max_coupling_residual, coupling_residual)
                    break
                self._kernel_accelerate_coupling(i_pass)

        if self._has_rigid_coupling:
            self._kernel_apply_coupled_impulse(
                rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_info, rigid_solver.rigid_config
            )
        if self._has_boundary:
            self._kernel_accumulate_boundary_load(is_step_start)
            self._solve_stats.n_boundary_slot_overflows += int(qd_to_torch(self.envs_boundary_slot_overflows).sum())

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_advect_position(f)
            self._kernel_copy_from_reordered(f)

    def substep_post_coupling_grad(self, f):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.active)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
            self.particles[i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        if self.is_active:
            state = SPHSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
            active[i_b, i_p] = self.particles[i_p, i_b].active

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = self.particles[i_p, i_b].active

    @qd.kernel
    def _kernel_add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        material_idx: qd.i32,
        pos: qd.types.ndarray(),
        material: array_class.SPHMaterialInfo,
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles[i_p, i_b].active = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].material = material_idx
            self.particles[i_p, i_b].pressure_density2_norm_p = 0.0
            self.particles[i_p, i_b].pressure_density2_norm_d = 0.0
            if qd.static(self._micropolar):
                self.particles[i_p, i_b].angular_vel = qd.Vector.zero(gs.qd_float, 3)

        self.materials[material_idx].rest_volume = self._particle_volume
        self.materials[material_idx].mass = self._particle_volume * material.rest_density
        self.materials[material_idx].rest_density = material.rest_density
        self.materials[material_idx].dynamic_viscosity = material.rest_density * material.viscosity
        self.materials[material_idx].surface_tension = material.surface_tension
        self.materials[material_idx].vorticity = material.vorticity
        self.materials[material_idx].angular_viscosity = material.angular_viscosity
        self.materials[material_idx].inverse_microinertia = 1.0 / material.microinertia

    # ----------------------------------------------------------------------

    @qd.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
            self.particles[i_p, i_b].vel.fill(0.0)
            self.particles[i_p, i_b].acc.fill(0.0)

    @qd.kernel
    def _kernel_get_particles_pos(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[i_p, i_b].pos[i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]
            self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_get_particles_vel(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[i_p, i_b].vel[i]

    @qd.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].active = actives[i_b_, i_p_]

    @qd.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles[i_p, i_b].active

    @qd.kernel
    def _kernel_get_mass(
        self, particle_start: qd.i32, n_particles: qd.i32, mass: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            i_m = self.particles[i_p, i_b].material
            mass[i_b_] += self.materials[i_m].mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def n_boundary_particles(self):
        return self._n_boundary_particles

    @property
    def solve_stats(self):
        """Iteration counts and residuals of the solves of the last step (see SPHSolveStats)."""
        return self._solve_stats

    def get_links_fluid_wrench(self, envs_idx=None):
        """Return the force and the torque of the fluid on every rigid link, averaged over the substeps of the last step.

        The force is in world frame, in N. The torque is about the center of mass of the link, in world frame, in N m.
        Both hold the pressure and viscous load of the fluid alone, without gravity, actuators or contacts. For a link
        driven along a prescribed trajectory, they are the load its actuators carry against the fluid. Links away from
        the fluid get zero. The tensors have shape (n_envs, n_links, 3), without the leading dimension for a scene that
        is not batched.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        n_links = self.sim.rigid_solver.n_links
        if not self._has_boundary:
            zeros = torch.zeros((len(envs_idx), n_links, 3), dtype=gs.tc_float, device=gs.device)
            force, torque = zeros, zeros.clone()
        else:
            inv_step_dt = 1.0 / self._sim.dt
            force = qd_to_torch(self.links_fluid_impulse, envs_idx, transpose=True) * inv_step_dt
            torque = qd_to_torch(self.links_fluid_angular_impulse, envs_idx, transpose=True) * inv_step_dt
        if self._scene.n_envs == 0:
            force, torque = force[0], torque[0]
        return force, torque

    @property
    def boundary_particle_radius(self):
        return self._boundary_spacing / 2.0

    @property
    def support_radius(self):
        return self._support_radius

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
