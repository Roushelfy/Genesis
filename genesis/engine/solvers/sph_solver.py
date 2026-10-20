import math

import numpy as np

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.couplers import LegacyCoupler
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine.states.solvers import SPHSolverState
from genesis.utils import array_class
from genesis.utils.misc import tensor_to_array
from genesis.utils.point_cloud import sample_mesh_point_cloud

from .base_solver import GravityMixin, Solver, TimeBasedMixin


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
        self._df_eps = 1e-5
        # A particle with fewer neighbors sits at a free surface, where the divergence estimate is unreliable.
        self._df_min_div_neighbors = 20

        self._particle_volume = 0.8 * self._particle_size**3  # 0.8 is an empirical value

        # Boundary particles sampled on the surface of the rigid geoms. The spacing is the one of the regular triangle
        # sampling of Akinci et al. 2012 boundaries, denser than the fluid so that the fluid cannot leak through.
        self._boundary_spacing = 0.75 * self._particle_size
        # Upper bound on the number of boundary particles, which grows with the area of the simulation domain.
        self._max_n_boundary_particles = 5_000_000
        self._n_boundary_particles = 0
        self._has_boundary = False

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
            # TODO: Add micropolar rotation and angular velocity
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
            viscosity=gs.qd_float,
            surface_tension=gs.qd_float,
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

        if self._has_boundary:
            # Boundary particles, sampled on the rigid geoms and moved with their link. They carry the static
            # sampling data as well, so that their hashed (reordered) copy is self-contained.
            struct_boundary_particle_state = qd.types.struct(
                pos_local=gs.qd_vec3,  # position in link frame
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
                            pos_local = sample_mesh_point_cloud(
                                verts, faces, n_points, n_candidates=16 * n_points, seed=0
                            )

                            # A fixed link never moves, so the samples outside the boundary area can be dropped.
                            if link.is_fixed:
                                pos_world = gu.transform_by_trans_quat(pos_local, link_pos, link_quat)
                                is_inside = np.all((pos_world > area_lower) & (pos_world < area_upper), axis=1)
                                pos_local = pos_local[is_inside]

                            if geom.active_envs_mask is None:
                                active = np.ones((self._B, len(pos_local)), dtype=gs.np_int)
                            else:
                                envs_active = tensor_to_array(geom.active_envs_mask, dtype=gs.np_int)
                                active = np.tile(envs_active[:, None], (1, len(pos_local)))
                            boundaries_pos_local.append(pos_local)
                            boundaries_link_idx.append(np.full(len(pos_local), link.idx, dtype=gs.np_int))
                            boundaries_is_fixed.append(np.full(len(pos_local), link.is_fixed, dtype=gs.np_int))
                            boundaries_active.append(active)
            self._n_boundary_particles = sum(len(pos_local) for pos_local in boundaries_pos_local)
            self._has_boundary = self._n_boundary_particles > 0
            if self._has_boundary:
                gs.logger.info(f"SPH boundary sampled with ~<{self._n_boundary_particles}>~ particles.")
                self.sh_boundary = gu.SpatialHasher(cell_size=self.sh.cell_size, grid_res=self.sh.grid_res)
                self.sh_boundary.build(self._B)

            self.init_particle_fields()
            self.init_ckpt()

            for entity in self.entities:
                entity._add_to_solver()

            if self._has_boundary:
                self._kernel_add_boundary_particles(
                    np.concatenate(boundaries_pos_local),
                    np.concatenate(boundaries_link_idx),
                    np.concatenate(boundaries_is_fixed),
                    np.concatenate(boundaries_active, axis=1),
                )
                self._kernel_update_boundary_particles(rigid_solver.dyn_state.links)
                self._kernel_compute_boundary_volume()

            # The explicit viscosity term in [_task_compute_non_pressure_forces] is only stable below this timestep.
            viscosity_max = max(entity.material.viscosity for entity in self.entities)
            stable_dt = 0.01 * self._support_radius**2 / viscosity_max if viscosity_max > 0.0 else float("inf")
            if self.substep_dt > stable_dt:
                gs.logger.warning(
                    f"Current timestep ({self.substep_dt:.4g}) is larger than the stable timestep "
                    f"({stable_dt:.4g}) for SPH solver. "
                    "Please consider reducing the timestep, liquid viscosity, or increasing the particle size."
                )

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
        ret -= self.materials[i_m_i].surface_tension / mass_i * mass_j * x_ij * self.cubic_kernel(effective_dist)

        # Viscosity, using the 2 * (d + 2) Laplacian approximation in 3D with a regularized denominator.
        v_ij = (self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel).dot(x_ij)
        rho_j = self.particles_reordered[j, i_b].density_norm * self.materials[i_m_j].rest_density
        ret += (
            2.0
            * (3.0 + 2.0)
            * self.materials[i_m_i].viscosity
            * (mass_j / rho_j)
            * v_ij
            / (dist**2 + 0.01 * self._support_radius**2)
            * self.cubic_kernel_derivative(x_ij)
        )

    @qd.kernel
    def _kernel_compute_non_pressure_forces(self, f: qd.i32, t: qd.f32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                acc = self._gravity[i_b]
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    acc,
                    self._task_compute_non_pressure_forces,
                    i_b,
                )

                for i_ff in qd.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(
                        self.particles_reordered[i_p, i_b].pos, self.particles_reordered[i_p, i_b].vel, t, i_p
                    )
                self.particles_reordered[i_p, i_b].acc = acc
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * acc

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
        link_idx: qd.types.ndarray(),
        is_fixed: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            for i in qd.static(range(3)):
                self.boundary_particles[i_bp, i_b].pos_local[i] = pos_local[i_bp, i]
            self.boundary_particles[i_bp, i_b].link_idx = link_idx[i_bp]
            self.boundary_particles[i_bp, i_b].is_fixed = qd.cast(is_fixed[i_bp], gs.qd_bool)
            self.boundary_particles[i_bp, i_b].active = qd.cast(active[i_b, i_bp], gs.qd_bool)

    @qd.kernel
    def _kernel_update_boundary_particles(self, links_state: array_class.LinksState):
        for i_bp, i_b in qd.ndrange(self._n_boundary_particles, self._B):
            if self.boundary_particles[i_bp, i_b].active:
                i_l = self.boundary_particles[i_bp, i_b].link_idx
                pos = gu.qd_transform_by_trans_quat(
                    self.boundary_particles[i_bp, i_b].pos_local, links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
                )
                self.boundary_particles[i_bp, i_b].pos = pos
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
        ret.density_norm += V_b * self.cubic_kernel(x_ib.norm())
        # BWJ23 eq. (32): boundary particles carry no constraint of their own, so they only enter the gradient of
        # the constraint of particle i, and not the sum of squared neighbor gradients.
        ret.grad_p_i += V_b * self.cubic_kernel_derivative(x_ib)
        ret.n_neighbors += 1

    @qd.func
    def _task_compute_boundary_density_change(self, i, j, ret: qd.template(), i_b):
        V_b = self.boundary_particles_reordered[j, i_b].volume
        v_ib = self.particles_reordered[i, i_b].vel - self.boundary_particles_reordered[j, i_b].vel
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret += V_b * v_ib.dot(self.cubic_kernel_derivative(x_ib))

    @qd.func
    def _task_compute_boundary_pressure_acc(self, i, j, ret: qd.template(), i_b):
        # BWJ23 eq. (33): a single term of the fluid particle pressure, with no pressure value at the boundary.
        V_b = self.boundary_particles_reordered[j, i_b].volume
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret.acc -= ret.p_rho2_i * V_b * self.cubic_kernel_derivative(x_ib)

    @qd.func
    def _task_compute_boundary_predicted_source(self, i, j, ret: qd.template(), i_b):
        # BWJ23 eq. (34): the pressure acceleration of a boundary particle is zero.
        V_b = self.boundary_particles_reordered[j, i_b].volume
        x_ib = self.particles_reordered[i, i_b].pos - self.boundary_particles_reordered[j, i_b].pos
        ret += V_b * self.particles_reordered[i, i_b].pressure_acc.dot(self.cubic_kernel_derivative(x_ib))

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

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                i_m = self.particles_reordered[i_p, i_b].material
                ret = qd.Struct(
                    density_norm=self.materials[i_m].rest_volume * self.cubic_kernel(0.0),
                    sum_grad_p_k=gs.qd_float(0.0),
                    grad_p_i=qd.Vector.zero(gs.qd_float, 3),
                    n_neighbors=0,
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
            if qd.abs(ret.p_rho2_i) > self._df_eps:
                self.sh_boundary.for_all_neighbors_of_pos(
                    i_p,
                    self.particles_reordered[i_p, i_b].pos,
                    -1,
                    self.boundary_particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_boundary_pressure_acc,
                    i_b,
                )

        self.particles_reordered[i_p, i_b].pressure_acc = ret.acc

    @qd.func
    def _task_compute_density_change(self, i, j, ret: qd.template(), i_b):
        i_m = self.particles_reordered[j, i_b].material
        v_ij = self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        ret += self.materials[i_m].rest_volume * v_ij.dot(self.cubic_kernel_derivative(x_ij))

    @qd.kernel
    def _kernel_init_pressure_solve(self, is_divergence: qd.template()):
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
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        density_change,
                        self._task_compute_boundary_density_change,
                        i_b,
                    )

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
                            p_rho2 = 0.5 * qd.min(self.particles_reordered[i_p, i_b].pressure_density2_norm_d, 0.5 * inv_dt)
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
                    self.sh_boundary.for_all_neighbors_of_pos(
                        i_p,
                        self.particles_reordered[i_p, i_b].pos,
                        -1,
                        self.boundary_particles_reordered.pos,
                        self._support_radius,
                        predicted_source,
                        self._task_compute_boundary_predicted_source,
                        i_b,
                    )

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
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].pressure_acc

    def _pressure_solve(self, is_divergence):
        """
        Run the divergence solve or the density solve and apply the resulting pressure accelerations to velocities.

        Iterations continue until the average error of every environment drops below the tolerance, within the
        min / max iteration bounds.
        """
        if is_divergence:
            n_iters_min, n_iters_max = self._df_min_div_iters, self._df_max_div_iters
            # The divergence error is a rate, so its tolerance is the density tolerance per substep.
            max_error = self._df_max_error_div / self._substep_dt
        else:
            n_iters_min, n_iters_max = self._df_min_den_iters, self._df_max_den_iters
            max_error = self._df_max_error_den

        self._kernel_init_pressure_solve(is_divergence)

        n_iters, avg_error, is_converged = 0, 0.0, False
        while (not is_converged or n_iters < n_iters_min) and n_iters < n_iters_max:
            avg_error = self._kernel_pressure_solve_iteration(is_divergence)
            is_converged = avg_error <= max_error
            n_iters += 1

        gs.logger.debug(
            f"DFSPH {'divergence' if is_divergence else 'density'} solve: {n_iters} iterations, avg error {avg_error:.4g}"
        )

        self._kernel_apply_pressure_acc(is_divergence)

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

    def substep_pre_coupling(self, f):
        if self.is_active:
            # The rigid solver runs its substep before this one, so the fluid sees the boundary at the end of the
            # substep, with the velocity it reaches there.
            if self._has_boundary:
                self._kernel_update_boundary_particles(self.sim.rigid_solver.dyn_state.links)
            self._kernel_reorder_particles(f)
            self._kernel_compute_density_and_factor(f)
            if not self._df_skip_div:
                self._pressure_solve(is_divergence=True)
            self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
            self._pressure_solve(is_divergence=False)

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
        mat_rest_density: qd.f32,
        # The stiffness and exponent of the equation of state only apply to the legacy 'WCSPH' pressure solver.
        mat_stiffness: qd.f32,
        mat_exponent: qd.f32,
        mat_viscosity: qd.f32,
        mat_surface_tension: qd.f32,
        pos: qd.types.ndarray(),
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

        self.materials[material_idx].rest_volume = self._particle_volume
        self.materials[material_idx].mass = self._particle_volume * mat_rest_density
        self.materials[material_idx].rest_density = mat_rest_density
        self.materials[material_idx].viscosity = mat_viscosity
        self.materials[material_idx].surface_tension = mat_surface_tension

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
