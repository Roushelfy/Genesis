from collections.abc import Iterator
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.entities.shell_entity import ShellEntity, ShellEntityDescription
from genesis.engine.materials.shell import Shell
from genesis.engine.states.solvers import ShellSolverState
from genesis.utils.misc import broadcast_tensor, qd_to_torch

from .base_solver import GravityMixin, Solver, TimeBasedMixin
from .rigid.abd.forward_dynamics import func_vel_at_point

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


# Edge of the grid the position of every vertex is anchored to, in m (see verts_pos_cell in array_class.py). A power of
# two keeps the cells exact in floating point, and its size bounds the offsets, whose precision it sets.
POS_GRID = 2.0**-10

# Floor of the norms the shell kernels divide by, guarding degenerate geometry alone. An additive epsilon (as in
# 'norm(gs.EPS)') would bias the edges, areas and normals of fine meshes, whose squared magnitudes approach it in single
# precision.
NORM_FLOOR = 1e-30

# Fraction of its squared norm the true residual of a linear solve must lose between two restarts of its conjugate
# gradient (see func_pcg_decide), the solve having reached the floor of the floating-point precision otherwise
STAGNATION_RATIO = 0.25

# Distance a contact is detected within, as a multiple of the distance the face and the geom can close over a substep
# at their velocities at its start. Such a speculative contact only pushes once the solve would make it penetrate.
CONTACT_MARGIN_RATIO = 2.0

# Smallest relative velocity below which the friction of a contact sticks, in m/s. The friction impulse grows smoothly
# from zero to the Coulomb bound as the slip velocity reaches the stick velocity of the contact, which widens past this
# floor until the stuck friction is no stiffer than the normal penalty (see func_contact_stick_velocity).
FRICTION_STICK_VELOCITY = 1e-4

# Maximum Newton iterations of the contact solve of a substep (see kernel_shell_rigid_contact_solve), a safety limit for
# the environments whose contacts keep changing regime
N_CONTACT_NEWTON_ITERATIONS = 32

# Evenly spaced step lengths at which every bracketing pass of the line search of the contact solve evaluates the slope
# of the potential along the step, the bracket of its minimum narrowing N_LINE_SEARCH_POINTS-fold per pass, then the
# refining passes and their stopping slope, relative to the initial one (see func_contact_line_search_update)
N_LINE_SEARCH_POINTS = 16
N_LINE_SEARCH_LEVELS = 2
N_LINE_SEARCH_REFINEMENTS = 4
LINE_SEARCH_SLOPE_TOLERANCE = 1e-2

# Fraction of the initial slope of the potential along the step that its slope at the full step may reach for the full
# step to be taken, a Newton step overshooting the minimum along it by a tenth at most
FULL_STEP_SLOPE_RATIO = 0.1

# Terms of the line search accumulated per environment (see func_contact_line_search_start), and values of the bracket
# of its minimum (see func_contact_line_search_update)
N_LINE_SEARCH_TERMS = 6
N_LINE_SEARCH_BRACKET = 6

# Iterations of the projected gradient descent locating the deepest point of a face in a geom, the step halving from
# half the longest edge of the face, which bounds the error on the position of the point to its 2^-N.
N_DEEPEST_POINT_ITERATIONS = 10

# Width of the soft minimum blending the vertices of a face into its first estimate of the deepest point, relative to
# its longest edge. A face lying flat on a geom starts at its centroid, and one tilted by more than this slope at its
# deepest vertex.
DEEPEST_POINT_SOFTMIN_WIDTH = 1e-3

# Barycentric coordinate below which the deepest point of a face lies on the edge opposite that corner, and sine of the
# angle by which the push of a geom on an edge may lean past the normal cone of the surface there (see
# func_contact_is_in_normal_cone)
EDGE_BARY_TOLERANCE = 1e-3
NORMAL_CONE_TOLERANCE = 1e-2


class PCG_MODE(IntEnum):
    """What the next system product of the linear solve of an environment applies to (see func_pcg_decide)."""

    # The search direction of a conjugate gradient iteration
    SOLVE = 0
    # The solution itself, whose true residual b - A dv checks the convergence the recursive residual claims
    CHECK = 1


class SHELL_SOLVE_STATUS(IntEnum):
    """Outcome of the linear solve of an environment, as bit flags so that a history can hold their union."""

    CONVERGED = 0
    # The iteration limit stopped the solve before its true residual reached the tolerance
    MAX_ITERATIONS = 1
    # The curvature p^T A p of a search direction was not positive, which a positive definite system excludes
    BREAKDOWN = 2
    # The residual stopped being finite, the solution being discarded
    NON_FINITE = 4
    # The true residual stopped decreasing above the tolerance, at the floor the floating-point precision of the system
    # resolves. The solve keeps its last iterate, as accurate as that precision allows, which is no failure.
    STAGNATION = 8


# Statuses an environment counts as a failure of its solver (see ShellSolver.get_envs_solver_failure)
SHELL_SOLVE_FAILURE = SHELL_SOLVE_STATUS.MAX_ITERATIONS | SHELL_SOLVE_STATUS.BREAKDOWN | SHELL_SOLVE_STATUS.NON_FINITE


class ShellSolver(GravityMixin, TimeBasedMixin, Solver):
    """
    Solver of thin elastoplastic sheets that tear and crack.

    Each substep integrates the sheets by one linearized backward Euler step: membrane stretching (Saint
    Venant-Kirchhoff on the Green strain, plane stress) and hinge bending, both with stiffness-proportional damping. The
    linear system is solved by matrix-free preconditioned conjugate gradient (PCG), every environment iterating until
    the residual of its own solve, in the norm of the preconditioner, falls below the tolerances of the options, the
    whole loop running on the device. The preconditioner adds to block-Jacobi a coarse correction where each patch of
    vertices moves affinely, which resolves the stiff membranes (paper, metal, glass) whose block-Jacobi iterations
    would only converge after hundreds of iterations. Positions are anchored to a fine grid, keeping strains precise in
    single precision wherever the sheet is. Contacts with rigid geoms join the same linear system (see
    solve_rigid_contact), before the positions advance, the material yields plastically, the damage index of every face
    is evaluated, and the vertices whose surrounding stress exceeds the tensile strength split along the mesh edges that
    relieve the most stress.
    """

    material_cls = Shell

    def __init__(self, scene: "Scene", sim: "Simulator", options):
        super().__init__(scene, sim, options)

        self._pcg_tolerance = options.pcg_tolerance
        self._pcg_velocity_tolerance = options.pcg_velocity_tolerance
        self._contact_stiffness = options.contact_stiffness
        self._pcg_max_iterations = options.pcg_max_iterations
        self._fracture_capacity = options.fracture_capacity
        self._n_coarse_patches = options.n_coarse_patches
        self._coarse_update_interval = options.coarse_update_interval

        self._static_config: array_class.ShellStaticConfig | None = None
        self._shell_info: array_class.ShellInfo | None = None
        self._shell_state: array_class.ShellState | None = None
        self._shell_scratch: array_class.ShellScratch | None = None
        # Allocated by the coupler when the sheets touch rigid geoms, whose contacts then join the linear solve
        self._shell_contact: array_class.ShellContactScratch | None = None
        # The environments whose solve produced non-finite values since their last reset (see check_errno)
        self._errno: qd.Tensor | None = None

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None, desc=None) -> ShellEntity:
        """Create a shell entity from its description, resolved from the other arguments when none is given."""
        if desc is None:
            desc = ShellEntityDescription.resolve(morph, material, surface, name)
        entity = ShellEntity(
            scene=self._scene,
            solver=self,
            idx=idx,
            desc=desc,
            vert_start=self.n_verts,
            face_start=self.n_faces,
            hinge_start=self.n_hinges,
        )
        self._entities.append(entity)
        return entity

    def build(self):
        super().build()
        self._n_verts = self.n_verts
        self._n_faces = self.n_faces
        self._n_hinges = self.n_hinges

        if self.is_active:
            materials = [entity.material for entity in self._entities]
            self._static_config = array_class.ShellStaticConfig(
                has_fracture=any(material.tensile_strength is not None and material.fracture for material in materials),
                has_damage=any(material.tensile_strength is not None for material in materials),
                has_plasticity=any(
                    material.yield_stress is not None or material.yield_curvature is not None for material in materials
                ),
                has_coarse_space=self._n_coarse_patches > 0,
                coarse_block_dim=1 if gs.backend == gs.cpu else 32,
            )
            entities_coarse_dim = [
                9 * entity.patches.n_patches if entity.patches is not None else 0 for entity in self._entities
            ]
            self._entities_coarse_dof_start = np.cumsum([0, *entities_coarse_dim])[:-1]
            self._entities_coarse_matrix_start = np.cumsum([0, *(dim**2 for dim in entities_coarse_dim)])[:-1]
            self._entities_coarse_dim = np.array(entities_coarse_dim)
            self._shell_info = array_class.get_shell_info(
                len(self._entities), self._n_verts, self._n_faces, self._n_hinges, int(self._entities_coarse_dim.sum())
            )
            self._shell_state = array_class.get_shell_state(
                len(self._entities), self._n_verts, self._n_faces, self._n_hinges, self._B
            )
            self._shell_scratch = array_class.get_shell_scratch(
                self._n_verts,
                self._n_faces,
                self._n_hinges,
                int(self._entities_coarse_dim.sum()),
                int(np.square(self._entities_coarse_dim).sum()),
                self._B,
                self._static_config.has_fracture,
                self._static_config.has_damage,
            )
            self._init_info_and_state()

        self._build_gravity()

    def build_rigid_contact(
        self,
        geoms_bound_center: np.ndarray,
        geoms_bound_radius: np.ndarray,
        n_trees: int,
        n_dofs: int,
        max_tree_dofs: int,
    ):
        """Allocate the buffers of the contacts between the sheets and the rigid geoms, which then join the linear solve
        of every substep, the coupler driving it (see solve_rigid_contact).

        Every geom is bounded by the sphere of radius geoms_bound_radius around geoms_bound_center in its own frame, a
        negative radius standing for an unbounded geom. The rigid solver has n_trees kinematic trees and n_dofs degrees
        of freedom, at most max_tree_dofs in a tree.
        """
        self._shell_contact = array_class.get_shell_contact_scratch(
            self._n_verts,
            self._n_faces,
            len(geoms_bound_radius),
            n_trees,
            n_dofs,
            max_tree_dofs,
            N_LINE_SEARCH_TERMS,
            N_LINE_SEARCH_POINTS,
            N_LINE_SEARCH_BRACKET,
            self._B,
        )
        self._shell_contact.geoms_bound_center.from_numpy(geoms_bound_center)
        self._shell_contact.geoms_bound_radius.from_numpy(geoms_bound_radius)
        self._shell_contact.contacts_geom.fill(-1)
        self._shell_contact.verts_contact_force.fill(0.0)
        self._shell_contact.dofs_dv.fill(0.0)

    def _init_info_and_state(self):
        """Fill the rest mesh of every entity and the initial state of every environment."""
        n_verts, n_faces, n_hinges, B = self._n_verts, self._n_faces, self._n_hinges, self._B
        info, state = self._shell_info, self._shell_state

        entities_material = np.array(
            [
                (
                    material.E / (1.0 - material.nu**2),
                    material.nu,
                    material.bending_scale * material.E / (12.0 * (1.0 - material.nu**2)),
                    material.damping,
                    material.tensile_strength or 0.0,
                    material.bending_fracture_scale,
                    material.yield_stress or 0.0,
                    material.plastic_flow_rate,
                    material.yield_curvature or 0.0,
                )
                for material in (entity.material for entity in self._entities)
            ],
            dtype=gs.np_float,
        )
        for i, tensor in enumerate(
            (
                info.entities_stretching_modulus,
                info.entities_nu,
                info.entities_bending_modulus,
                info.entities_damping,
                info.entities_tensile_strength,
                info.entities_bending_fracture_scale,
                info.entities_yield_stress,
                info.entities_plastic_flow_rate,
                info.entities_yield_curvature,
            )
        ):
            tensor.from_numpy(np.ascontiguousarray(entities_material[:, i]))
        info.entities_is_fracturable.from_numpy(
            np.array([e.material.tensile_strength is not None and e.material.fracture for e in self._entities])
        )
        info.entities_vert_start.from_numpy(np.array([e.vert_start for e in self._entities], dtype=gs.np_int))
        info.entities_vert_end.from_numpy(
            np.array([e.vert_start + e.n_verts_max for e in self._entities], dtype=gs.np_int)
        )
        info.entities_face_start.from_numpy(np.array([e.face_start for e in self._entities], dtype=gs.np_int))
        info.entities_face_end.from_numpy(np.array([e.face_start + e.n_faces for e in self._entities], dtype=gs.np_int))
        info.entities_coarse_dof_start.from_numpy(self._entities_coarse_dof_start.astype(gs.np_int))
        info.entities_coarse_dim.from_numpy(self._entities_coarse_dim.astype(gs.np_int))
        info.entities_coarse_matrix_start.from_numpy(self._entities_coarse_matrix_start.astype(gs.np_int))
        coarse_dofs_entity = np.repeat(np.arange(len(self._entities), dtype=gs.np_int), self._entities_coarse_dim)
        info.coarse_dofs_entity.from_numpy(coarse_dofs_entity if len(coarse_dofs_entity) else np.zeros(1, gs.np_int))

        faces_entity = np.zeros(n_faces, dtype=gs.np_int)
        faces_mass = np.zeros(n_faces, dtype=gs.np_float)
        faces_rest_area = np.zeros(n_faces, dtype=gs.np_float)
        faces_Dm = np.zeros((n_faces, 2, 2), dtype=gs.np_float)
        faces_basis = np.zeros((n_faces, 3, 2), dtype=gs.np_float)
        faces_hinge = np.full((n_faces, 3), -1, dtype=gs.np_int)
        hinges_entity = np.zeros(max(n_hinges, 1), dtype=gs.np_int)
        # A scene without interior edge keeps one hinge whose endpoints never match across its faces, so that it never
        # reads as intact.
        hinges_corner = np.tile(np.array([0, 1, 0, 0], dtype=gs.np_int), (max(n_hinges, 1), 1))
        hinges_opposite_corner = np.zeros((max(n_hinges, 1), 2), dtype=gs.np_int)
        hinges_rest_angle = np.zeros(max(n_hinges, 1), dtype=gs.np_float)
        hinges_rest_len = np.ones(max(n_hinges, 1), dtype=gs.np_float)
        hinges_rest_area = np.ones(max(n_hinges, 1), dtype=gs.np_float)
        verts_coarse_dof = np.full(n_verts, -1, dtype=gs.np_int)
        verts_coarse_phi = np.zeros((n_verts, 3), dtype=gs.np_float)
        verts_fan_start = np.zeros(n_verts, dtype=gs.np_int)
        verts_fan_len = np.zeros(n_verts, dtype=gs.np_int)
        verts_is_fan_closed = np.zeros(n_verts, dtype=np.bool_)
        fans_corner = np.zeros(3 * n_faces, dtype=gs.np_int)
        fans_next_hinge = np.full(3 * n_faces, -1, dtype=gs.np_int)
        verts_pos = np.zeros((n_verts, 3), dtype=gs.np_float)
        verts_origin = np.full(n_verts, -1, dtype=gs.np_int)
        corners_vert = np.zeros(3 * n_faces, dtype=gs.np_int)
        entities_n_verts = np.zeros(len(self._entities), dtype=gs.np_int)

        for i_e, entity in enumerate(self._entities):
            topology = entity.topology
            v_start, f_start, h_start = entity.vert_start, entity.face_start, entity.hinge_start
            faces_slice = slice(f_start, f_start + entity.n_faces)
            hinges_slice = slice(h_start, h_start + entity.n_hinges)
            corners_slice = slice(3 * f_start, 3 * (f_start + entity.n_faces))
            verts_slice = slice(v_start, v_start + entity.n_verts)

            faces_entity[faces_slice] = i_e
            faces_mass[faces_slice] = topology.faces_mass
            faces_rest_area[faces_slice] = topology.faces_rest_area
            faces_Dm[faces_slice] = topology.faces_Dm
            faces_basis[faces_slice] = topology.faces_basis
            faces_hinge[faces_slice] = np.where(topology.faces_hinge >= 0, topology.faces_hinge + h_start, -1)
            hinges_entity[hinges_slice] = i_e
            hinges_corner[hinges_slice] = topology.hinges_corner + 3 * f_start
            hinges_opposite_corner[hinges_slice] = topology.hinges_opposite_corner + 3 * f_start
            hinges_rest_angle[hinges_slice] = topology.hinges_rest_angle
            hinges_rest_len[hinges_slice] = topology.hinges_rest_len
            hinges_rest_area[hinges_slice] = topology.hinges_rest_area
            if entity.patches is not None:
                verts_coarse_dof[verts_slice] = self._entities_coarse_dof_start[i_e] + 9 * entity.patches.verts_patch
                verts_coarse_phi[verts_slice] = entity.patches.verts_phi
            verts_fan_start[verts_slice] = topology.verts_fan_start + 3 * f_start
            verts_fan_len[verts_slice] = topology.verts_fan_len
            verts_is_fan_closed[verts_slice] = topology.verts_is_fan_closed
            fans_corner[corners_slice] = topology.fans_corner + 3 * f_start
            fans_next_hinge[corners_slice] = np.where(
                topology.fans_next_hinge >= 0, topology.fans_next_hinge + h_start, -1
            )
            verts_pos[verts_slice] = entity.init_verts
            verts_origin[verts_slice] = np.arange(v_start, v_start + entity.n_verts, dtype=gs.np_int)
            corners_vert[corners_slice] = entity.init_faces.reshape(-1) + v_start
            entities_n_verts[i_e] = entity.n_verts

        info.faces_entity.from_numpy(faces_entity)
        info.faces_mass.from_numpy(faces_mass)
        info.faces_rest_area.from_numpy(faces_rest_area)
        info.faces_Dm.from_numpy(faces_Dm)
        info.faces_Dm_inv.from_numpy(np.linalg.inv(faces_Dm))
        info.faces_basis.from_numpy(faces_basis)
        info.faces_hinge.from_numpy(faces_hinge)
        info.hinges_entity.from_numpy(hinges_entity)
        info.hinges_corner.from_numpy(hinges_corner)
        info.hinges_opposite_corner.from_numpy(hinges_opposite_corner)
        info.hinges_rest_angle.from_numpy(hinges_rest_angle)
        info.hinges_rest_len.from_numpy(hinges_rest_len)
        info.hinges_rest_area.from_numpy(hinges_rest_area)
        info.verts_coarse_dof.from_numpy(verts_coarse_dof)
        info.verts_coarse_phi.from_numpy(verts_coarse_phi)
        info.verts_fan_start.from_numpy(verts_fan_start)
        info.verts_fan_len.from_numpy(verts_fan_len)
        info.verts_is_fan_closed.from_numpy(verts_is_fan_closed)
        info.fans_corner.from_numpy(fans_corner)
        info.fans_next_hinge.from_numpy(fans_next_hinge)

        faces_thickness = np.array([entity.material.thickness for entity in self._entities], dtype=gs.np_float)
        state.verts_pos.from_numpy(np.ascontiguousarray(np.broadcast_to(verts_pos[:, None], (n_verts, B, 3))))
        verts_pos_cell = np.round(verts_pos / POS_GRID).astype(gs.np_int)
        state.verts_pos_cell.from_numpy(np.ascontiguousarray(np.broadcast_to(verts_pos_cell[:, None], (n_verts, B, 3))))
        verts_pos_offset = (verts_pos - verts_pos_cell * POS_GRID).astype(gs.np_float)
        state.verts_pos_offset.from_numpy(
            np.ascontiguousarray(np.broadcast_to(verts_pos_offset[:, None], (n_verts, B, 3)))
        )
        state.verts_vel.from_numpy(np.zeros((n_verts, B, 3), dtype=gs.np_float))
        state.verts_dv.from_numpy(np.zeros((n_verts, B, 3), dtype=gs.np_float))
        state.verts_origin.from_numpy(np.ascontiguousarray(np.broadcast_to(verts_origin[:, None], (n_verts, B))))
        state.verts_is_fixed.from_numpy(np.zeros((n_verts, B), dtype=np.bool_))
        state.corners_vert.from_numpy(np.ascontiguousarray(np.broadcast_to(corners_vert[:, None], (3 * n_faces, B))))
        state.entities_n_verts.from_numpy(
            np.ascontiguousarray(np.broadcast_to(entities_n_verts[:, None], (len(self._entities), B)))
        )
        state.entities_peak_damage.from_numpy(np.zeros((len(self._entities), B), dtype=gs.np_float))
        state.entities_failure_face.from_numpy(np.full((len(self._entities), B), -1, dtype=gs.np_int))
        state.contacts_geom_prev.from_numpy(np.full((2 * n_faces, B), -1, dtype=gs.np_int))
        state.contacts_friction_bound_prev.from_numpy(np.zeros((2 * n_faces, B), dtype=gs.np_float))
        state.envs_solver_failure.from_numpy(np.zeros(B, dtype=gs.np_int))
        state.faces_plastic.from_numpy(
            np.ascontiguousarray(np.broadcast_to(np.eye(2, dtype=gs.np_float), (n_faces, B, 2, 2)))
        )
        state.faces_thickness.from_numpy(
            np.ascontiguousarray(np.broadcast_to(faces_thickness[faces_entity][:, None], (n_faces, B)))
        )
        state.hinges_plastic_angle.from_numpy(np.zeros((max(n_hinges, 1), B), dtype=gs.np_float))
        # The coarse matrices of every environment are due at the first substep
        self._shell_scratch.envs_coarse_age.from_numpy(np.full(B, self._coarse_update_interval, dtype=gs.np_int))
        self._errno = array_class.V(dtype=gs.qd_int, shape=(B,))
        self._errno.from_numpy(np.zeros(B, dtype=gs.np_int))

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        pass

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        kernel_shell_compute_forces(
            self._substep_dt, self._gravity, self._shell_state, self._shell_scratch, self._shell_info
        )
        if self._static_config.has_coarse_space:
            kernel_shell_coarse_assemble(
                self._shell_state, self._shell_scratch, self._shell_info, self._coarse_update_interval
            )
            kernel_shell_coarse_factorize(
                self._shell_state,
                self._shell_scratch,
                self._shell_info,
                self._static_config,
                self._coarse_update_interval,
            )
        # The coupler solves the sheets touching rigid geoms jointly with their contacts (see solve_rigid_contact)
        if self._shell_contact is None:
            kernel_shell_pcg_solve(
                self._shell_scratch.pcg_flag,
                self._shell_state,
                self._shell_scratch,
                self._shell_info,
                self._static_config,
                self._pcg_max_iterations,
                self._pcg_tolerance,
                self._pcg_velocity_tolerance,
                self._errno,
            )
            kernel_shell_apply_dv(self._shell_state, self._shell_scratch)

    def solve_rigid_contact(self, rigid_solver):
        """Solve the velocity update of the sheets jointly with their contacts with the rigid geoms, between the
        constraint solve of the rigid solver and its integration.

        The contacts are detected at the start of the substep, the rigid velocities being the ones the rigid solver
        found for its end, then solved by Newton iterations on the incremental potential of the substep (see
        kernel_shell_rigid_contact_solve). The velocity change the contacts give the rigid degrees of freedom is added
        to their acceleration, which the rigid solver integrates next, and the contact force of every contact is added
        to its rigid link.
        """
        kernel_shell_rigid_contact_detect(
            self._substep_dt,
            self._shell_state,
            self._shell_scratch,
            self._shell_contact,
            rigid_solver.dyn_state,
            self._shell_info,
            rigid_solver.dyn_info,
            rigid_solver.rigid_info,
            rigid_solver.collider.collider_info.sdf,
            rigid_solver.rigid_config,
            rigid_solver.collider.collider_config,
            self._contact_stiffness,
        )
        kernel_shell_rigid_contact_solve(
            self._substep_dt,
            self._shell_contact.newton_flag,
            self._shell_scratch.pcg_flag,
            self._shell_contact.line_flag,
            self._shell_state,
            self._shell_scratch,
            self._shell_contact,
            rigid_solver.dyn_state,
            self._shell_info,
            rigid_solver.dyn_info,
            rigid_solver.rigid_info,
            self._static_config,
            rigid_solver.rigid_config,
            N_CONTACT_NEWTON_ITERATIONS,
            self._pcg_max_iterations,
            self._contact_stiffness,
            self._pcg_tolerance,
            self._pcg_velocity_tolerance,
            self._errno,
        )
        kernel_shell_rigid_contact_finalize(
            self._substep_dt,
            self._shell_state,
            self._shell_scratch,
            self._shell_contact,
            rigid_solver.dyn_state,
            rigid_solver.dyn_info,
        )
        kernel_shell_apply_dv(self._shell_state, self._shell_scratch)

    def substep_post_coupling(self, f):
        if not self.is_active:
            return
        kernel_shell_integrate(self._substep_dt, self._shell_state, self._shell_scratch)
        if self._static_config.has_plasticity:
            kernel_shell_plastic_flow(self._substep_dt, self._shell_state, self._shell_scratch, self._shell_info)
        # The damage index is evaluated on the mesh the substep deformed, before fracture rewires it
        if self._static_config.has_damage:
            kernel_shell_damage(self._shell_state, self._shell_scratch, self._shell_info)
        if self._static_config.has_fracture:
            kernel_shell_fracture(self._shell_state, self._shell_scratch, self._shell_info)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling_grad(self, f):
        pass

    def reset_grad(self):
        pass

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f):
        if not self.is_active:
            return None
        state = self._shell_state
        return ShellSolverState(
            scene=self._scene,
            verts_pos=qd_to_torch(state.verts_pos, transpose=True, copy=True),
            verts_pos_cell=qd_to_torch(state.verts_pos_cell, transpose=True, copy=True),
            verts_pos_offset=qd_to_torch(state.verts_pos_offset, transpose=True, copy=True),
            verts_vel=qd_to_torch(state.verts_vel, transpose=True, copy=True),
            verts_dv=qd_to_torch(state.verts_dv, transpose=True, copy=True),
            verts_origin=qd_to_torch(state.verts_origin, transpose=True, copy=True),
            verts_is_fixed=qd_to_torch(state.verts_is_fixed, transpose=True, copy=True),
            corners_vert=qd_to_torch(state.corners_vert, transpose=True, copy=True),
            contacts_geom_prev=qd_to_torch(state.contacts_geom_prev, transpose=True, copy=True),
            contacts_friction_bound_prev=qd_to_torch(state.contacts_friction_bound_prev, transpose=True, copy=True),
            entities_n_verts=qd_to_torch(state.entities_n_verts, transpose=True, copy=True),
            entities_peak_damage=qd_to_torch(state.entities_peak_damage, transpose=True, copy=True),
            entities_failure_face=qd_to_torch(state.entities_failure_face, transpose=True, copy=True),
            envs_solver_failure=qd_to_torch(state.envs_solver_failure, copy=True),
            faces_plastic=qd_to_torch(state.faces_plastic, transpose=True, copy=True),
            faces_thickness=qd_to_torch(state.faces_thickness, transpose=True, copy=True),
            hinges_plastic_angle=qd_to_torch(state.hinges_plastic_angle, transpose=True, copy=True),
        )

    def set_state(self, f, state: ShellSolverState, envs_idx=None):
        """Restore some environments, the warm start of their next linear solve included.

        The coarse preconditioner of a restored environment is rebuilt at its next substep, so that an environment
        reset to a state starts from the same numerical history as a fresh scene in that state.
        """
        if not self.is_active:
            return
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        kernel_shell_set_state(
            envs_idx,
            state.verts_pos[envs_idx].contiguous(),
            state.verts_pos_cell[envs_idx].contiguous(),
            state.verts_pos_offset[envs_idx].contiguous(),
            state.verts_vel[envs_idx].contiguous(),
            state.verts_dv[envs_idx].contiguous(),
            state.verts_origin[envs_idx].contiguous(),
            state.verts_is_fixed[envs_idx].contiguous(),
            state.corners_vert[envs_idx].contiguous(),
            state.contacts_geom_prev[envs_idx].contiguous(),
            state.contacts_friction_bound_prev[envs_idx].contiguous(),
            state.entities_n_verts[envs_idx].contiguous(),
            state.entities_peak_damage[envs_idx].contiguous(),
            state.entities_failure_face[envs_idx].contiguous(),
            state.envs_solver_failure[envs_idx].contiguous(),
            state.faces_plastic[envs_idx].contiguous(),
            state.faces_thickness[envs_idx].contiguous(),
            state.hinges_plastic_angle[envs_idx].contiguous(),
            self._shell_state,
            self._shell_scratch,
            self._coarse_update_interval,
            self._errno,
        )

    @property
    def data(self) -> Iterator[array_class.DataItem]:
        yield from array_class.iter_data(self._static_config, "static_config")
        yield from array_class.iter_data(self._errno, "errno", array_class.DataKind.STATE)
        yield from array_class.iter_data(self._shell_info, "shell_info")
        yield from array_class.iter_data(self._shell_state, "shell_state")
        yield from array_class.iter_data(self._shell_scratch, "shell_scratch")
        if self._shell_contact is not None:
            yield from array_class.iter_data(self._shell_contact, "shell_contact")

    def check_errno(self):
        """Raise if the linear solve of any environment produced non-finite values since the last reset."""
        if (qd_to_torch(self._errno) > 0).any():
            gs.raise_exception(
                "The linear solve of the shell solver produced non-finite values. Decrease the shell simulation "
                "timestep, or check the material parameters and the contacts of the sheets."
            )

    def _sanitize_verts_idx(self, entity: ShellEntity, verts_idx_local, envs_idx):
        """Return the pool indices of the vertices of an entity, one row per environment of `envs_idx`."""
        if verts_idx_local is None:
            verts_idx_local = torch.arange(entity.n_verts, dtype=gs.tc_int, device=gs.device)
        verts_idx_local = torch.atleast_1d(torch.as_tensor(verts_idx_local, dtype=gs.tc_int, device=gs.device))
        if verts_idx_local.ndim == 1:
            verts_idx_local = verts_idx_local.expand((len(envs_idx), verts_idx_local.shape[0]))
        if ((verts_idx_local < 0) | (verts_idx_local >= entity.n_verts_max)).any():
            gs.raise_exception(f"Vertex indices must lie in [0, {entity.n_verts_max}), got {verts_idx_local}.")
        return (verts_idx_local + entity.vert_start).contiguous()

    def set_verts_pos(self, pos, entity: ShellEntity, verts_idx_local, envs_idx):
        """Set the position of some vertices of an entity."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_verts_idx(entity, verts_idx_local, envs_idx)
        pos = broadcast_tensor(pos, gs.tc_float, (*verts_idx.shape, 3), ("envs_idx", "verts_idx", ""))
        kernel_shell_set_verts_pos(verts_idx, envs_idx, pos.contiguous(), self._shell_state)

    def set_verts_vel(self, vel, entity: ShellEntity, verts_idx_local, envs_idx):
        """Set the velocity of some vertices of an entity."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_verts_idx(entity, verts_idx_local, envs_idx)
        vel = broadcast_tensor(vel, gs.tc_float, (*verts_idx.shape, 3), ("envs_idx", "verts_idx", ""))
        kernel_shell_set_verts_vel(verts_idx, envs_idx, vel.contiguous(), self._shell_state)

    def set_verts_fixed(self, is_fixed: bool, entity: ShellEntity, verts_idx_local, envs_idx):
        """Fix or release some vertices of an entity."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_verts_idx(entity, verts_idx_local, envs_idx)
        kernel_shell_set_verts_fixed(verts_idx, envs_idx, is_fixed, self._shell_state)

    def update_render_fields(self):
        """Refresh the per-corner positions and normals the visualizer reads."""
        kernel_shell_update_render(self._shell_state, self._shell_scratch, self._shell_info)

    @gs.assert_built
    def get_envs_solver_failure(self, envs_idx=None) -> torch.Tensor:
        """
        Get the failures of the linear solves of every environment since its last reset.

        A failure is the numerical one of the solver, which an episode may end on as it would on the damage of a sheet,
        but which says nothing of the material.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        failure : torch.Tensor, shape () or (n_envs,)
            The union of the failure flags of SHELL_SOLVE_STATUS the solves reported, zero if none failed:
            MAX_ITERATIONS (1) when the iteration limit stopped a linear or contact solve short of its tolerance,
            BREAKDOWN (2) when a search direction lost positive curvature, NON_FINITE (4) when a solve produced
            non-finite values. A solve stalling at the floating-point precision floor above its tolerance (STAGNATION)
            keeps an iterate as accurate as that precision allows, which counts as no failure.
        """
        tensor = qd_to_torch(self._shell_state.envs_solver_failure, envs_idx, copy=True)
        return tensor[0] if self._scene.n_envs == 0 else tensor

    @gs.assert_built
    def get_envs_pcg_iterations(self, envs_idx=None) -> torch.Tensor:
        """
        Get the number of system products the linear solves of the last substep ran in every environment.

        The count sums the solves of every iteration of the contact solve, the products checking the true residual of
        the warm start and of the converged solution included.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        n_iterations : torch.Tensor, shape () or (n_envs,)
        """
        tensor = qd_to_torch(self._shell_scratch.envs_n_iterations, envs_idx, copy=True)
        return tensor[0] if self._scene.n_envs == 0 else tensor

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_entities > 0

    @property
    def n_verts(self):
        if self.is_built:
            return self._n_verts
        return sum(entity.n_verts_max for entity in self._entities)

    @property
    def n_faces(self):
        if self.is_built:
            return self._n_faces
        return sum(entity.n_faces for entity in self._entities)

    @property
    def n_hinges(self):
        if self.is_built:
            return self._n_hinges
        return sum(entity.n_hinges for entity in self._entities)

    @property
    def fracture_capacity(self) -> float:
        return self._fracture_capacity

    @property
    def n_coarse_patches(self) -> int:
        return self._n_coarse_patches

    @property
    def shell_state(self) -> array_class.ShellState:
        return self._shell_state

    @property
    def shell_info(self) -> array_class.ShellInfo:
        return self._shell_info

    @property
    def shell_scratch(self) -> array_class.ShellScratch:
        return self._shell_scratch

    @property
    def shell_contact(self) -> array_class.ShellContactScratch | None:
        return self._shell_contact


# ------------------------------------------------------------------------------------
# ------------------------------------- helpers --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_vert_offset(i_v: int, i_u: int, i_b: int, shell_state: array_class.ShellState):
    """Position of vertex i_v relative to vertex i_u, exact up to the precision of the small anchored offsets.

    The difference of the integer grid cells is exact, and any reordering of the floating-point sum stays at the scale
    of the edge rather than of the absolute position.
    """
    cell_v = shell_state.verts_pos_cell[i_v, i_b]
    cell_u = shell_state.verts_pos_cell[i_u, i_b]
    return (cell_v - cell_u).cast(gs.qd_float) * POS_GRID + (
        shell_state.verts_pos_offset[i_v, i_b] - shell_state.verts_pos_offset[i_u, i_b]
    )


@qd.func
def func_sym2_eigen(S: qd.types.matrix(2, 2)):
    """Eigen-decompose a symmetric 2x2 matrix, returning the larger eigenvalue, the smaller one, and the unit
    eigenvector of the larger one (the other one being its counter-clockwise perpendicular)."""
    mean = 0.5 * (S[0, 0] + S[1, 1])
    half_diff = 0.5 * (S[0, 0] - S[1, 1])
    radius = qd.sqrt(half_diff * half_diff + S[0, 1] * S[0, 1])
    angle = 0.5 * qd.atan2(S[0, 1], half_diff)
    return mean + radius, mean - radius, qd.Vector([qd.cos(angle), qd.sin(angle)], dt=gs.qd_float)


@qd.func
def func_sym2_compose(lambda_0: float, lambda_1: float, eigvec_0: qd.types.vector(2)):
    """Compose the symmetric 2x2 matrix of eigenvalues lambda_0 along eigvec_0 and lambda_1 perpendicular to it."""
    eigvec_1 = qd.Vector([-eigvec_0[1], eigvec_0[0]], dt=gs.qd_float)
    return lambda_0 * eigvec_0.outer_product(eigvec_0) + lambda_1 * eigvec_1.outer_product(eigvec_1)


@qd.func
def func_membrane_stress(F: qd.types.matrix(3, 2), modulus: float, nu: float):
    """Plane-stress Saint Venant-Kirchhoff stress of a membrane, in N/m, for the deformation gradient of its face."""
    G = 0.5 * (F.transpose() @ F - qd.Matrix.identity(gs.qd_float, 2))
    return modulus * ((1.0 - nu) * G + nu * G.trace() * qd.Matrix.identity(gs.qd_float, 2))


@qd.func
def func_face_deformation(i_f: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Return the elastic deformation gradient of a face, from its rest frame to the world, and the matrix Y mapping
    the edge vectors of the face to it (F = [x1 - x0, x2 - x0] @ Y)."""
    i_v0 = shell_state.corners_vert[3 * i_f, i_b]
    i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
    i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
    Ds = qd.Matrix.cols(
        [func_vert_offset(i_v1, i_v0, i_b, shell_state), func_vert_offset(i_v2, i_v0, i_b, shell_state)]
    )
    Y = shell_info.faces_Dm_inv[i_f] @ shell_state.faces_plastic[i_f, i_b]
    return Ds @ Y, Y


@qd.func
def func_membrane_stiffness_product(
    dx0: qd.types.vector(3),
    dx1: qd.types.vector(3),
    dx2: qd.types.vector(3),
    F: qd.types.matrix(3, 2),
    stress_pos: qd.types.matrix(2, 2),
    Y: qd.types.matrix(2, 2),
    modulus: float,
    nu: float,
):
    """Product of the membrane stiffness of a face, per unit area, with a displacement of its three vertices.

    The stiffness keeps the geometric term of the positive part of the stress only, which makes it positive
    semi-definite: the material term is dG : C : dG and the geometric one stress_pos : dF^T dF, both non-negative.
    """
    dF = (dx1 - dx0).outer_product(qd.Vector([Y[0, 0], Y[0, 1]])) + (dx2 - dx0).outer_product(
        qd.Vector([Y[1, 0], Y[1, 1]])
    )
    dG = 0.5 * (dF.transpose() @ F + F.transpose() @ dF)
    dS = modulus * ((1.0 - nu) * dG + nu * dG.trace() * qd.Matrix.identity(gs.qd_float, 2))
    Q = (F @ dS + dF @ stress_pos) @ Y.transpose()
    q1 = qd.Vector([Q[0, 0], Q[1, 0], Q[2, 0]])
    q2 = qd.Vector([Q[0, 1], Q[1, 1], Q[2, 1]])
    return -(q1 + q2), q1, q2


@qd.func
def func_hinge_angle(edge_b: qd.types.vector(3), edge_c: qd.types.vector(3), edge_d: qd.types.vector(3)):
    """Signed dihedral angle of a hinge about its edge from a to b, the first face holding (a, b, c) and the second
    (b, a, d), zero when flat, from the positions of b, c and d relative to a."""
    cross_0 = edge_b.cross(edge_c)
    cross_1 = -edge_b.cross(edge_d - edge_b)
    normal_0 = cross_0 / qd.max(cross_0.norm(), NORM_FLOOR)
    normal_1 = cross_1 / qd.max(cross_1.norm(), NORM_FLOOR)
    edge = edge_b / qd.max(edge_b.norm(), NORM_FLOOR)
    return qd.atan2(edge.dot(normal_0.cross(normal_1)), normal_0.dot(normal_1))


@qd.func
def func_hinge_angle_gradient(edge_b: qd.types.vector(3), edge_c: qd.types.vector(3), edge_d: qd.types.vector(3)):
    """Gradient of the dihedral angle of a hinge (see func_hinge_angle) with respect to a, b, c and d, as columns."""
    edge_len = qd.max(edge_b.norm(), NORM_FLOOR)
    cross_0 = edge_b.cross(edge_c)
    cross_1 = -edge_b.cross(edge_d - edge_b)
    double_area_0 = qd.max(cross_0.norm(), NORM_FLOOR)
    double_area_1 = qd.max(cross_1.norm(), NORM_FLOOR)
    # The gradient at an opposite vertex is the normal of its face over its height above the edge.
    grad_c = -cross_0 / double_area_0 * (edge_len / double_area_0)
    grad_d = -cross_1 / double_area_1 * (edge_len / double_area_1)
    s_c = edge_c.dot(edge_b) / (edge_len * edge_len)
    s_d = edge_d.dot(edge_b) / (edge_len * edge_len)
    grad_a = -((1.0 - s_c) * grad_c + (1.0 - s_d) * grad_d)
    grad_b = -(s_c * grad_c + s_d * grad_d)
    return qd.Matrix.cols([grad_a, grad_b, grad_c, grad_d])


@qd.func
def func_hinge_edges(i_va: int, i_vb: int, i_vc: int, i_vd: int, i_b: int, shell_state: array_class.ShellState):
    """Positions of the vertices b, c and d of a hinge relative to its vertex a (see func_vert_offset)."""
    return (
        func_vert_offset(i_vb, i_va, i_b, shell_state),
        func_vert_offset(i_vc, i_va, i_b, shell_state),
        func_vert_offset(i_vd, i_va, i_b, shell_state),
    )


@qd.func
def func_hinge_verts(i_h: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Return the vertices a, b, c, d of a hinge (see func_hinge_angle), and whether its two faces still share their
    edge."""
    corners = shell_info.hinges_corner[i_h]
    opposite_corners = shell_info.hinges_opposite_corner[i_h]
    i_va = shell_state.corners_vert[corners[0], i_b]
    i_vb = shell_state.corners_vert[corners[1], i_b]
    i_va_1 = shell_state.corners_vert[corners[2], i_b]
    i_vb_1 = shell_state.corners_vert[corners[3], i_b]
    i_vc = shell_state.corners_vert[opposite_corners[0], i_b]
    i_vd = shell_state.corners_vert[opposite_corners[1], i_b]
    return i_va, i_vb, i_vc, i_vd, i_va == i_va_1 and i_vb == i_vb_1


@qd.func
def func_hinge_stiffness(i_h: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Bending stiffness of a hinge, in N*m per rad^2, such that its energy is 0.5 * k * (angle - rest angle)^2.

    The factor rest_len^2 / rest_area makes the hinges of a regular mesh sum up to the bending energy of an isotropic
    plate, D / 2 * curvature^2 per unit area.
    """
    i_e = shell_info.hinges_entity[i_h]
    corners = shell_info.hinges_corner[i_h]
    thickness = 0.5 * (
        shell_state.faces_thickness[corners[0] // 3, i_b] + shell_state.faces_thickness[corners[2] // 3, i_b]
    )
    rest_len = shell_info.hinges_rest_len[i_h]
    return (
        shell_info.entities_bending_modulus[i_e] * thickness**3 * rest_len * rest_len / shell_info.hinges_rest_area[i_h]
    )


@qd.func
def func_face_curvature(i_f: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Elastic curvature tensor of a face, in 1/m, in the basis of its rest frame (see faces_basis).

    Every intact hinge of the face bends it by its elastic dihedral angle about its edge, which spreads over the face
    as sum(angle * len * n n^T) / (2 * area) for the in-plane unit normal n of the edge, the shape operator a cylinder
    of radius R tessellated into strips recovers as 1 / R across its axis.
    """
    Dm = shell_info.faces_Dm[i_f]
    curvature = qd.Matrix.zero(gs.qd_float, 2, 2)
    faces_hinge = shell_info.faces_hinge[i_f]
    for k in range(3):
        i_h = faces_hinge[0]
        if k == 1:
            i_h = faces_hinge[1]
        elif k == 2:
            i_h = faces_hinge[2]
        if i_h >= 0:
            i_va, i_vb, i_vc, i_vd, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
            if is_intact:
                edge_b, edge_c, edge_d = func_hinge_edges(i_va, i_vb, i_vc, i_vd, i_b, shell_state)
                angle = func_hinge_angle(edge_b, edge_c, edge_d)
                angle_elastic = angle - shell_info.hinges_rest_angle[i_h] - shell_state.hinges_plastic_angle[i_h, i_b]
                edge = qd.Vector([Dm[0, 0], Dm[1, 0]])
                if k == 1:
                    edge = qd.Vector([Dm[0, 1] - Dm[0, 0], Dm[1, 1] - Dm[1, 0]])
                elif k == 2:
                    edge = -qd.Vector([Dm[0, 1], Dm[1, 1]])
                edge_len = qd.max(edge.norm(), NORM_FLOOR)
                normal = qd.Vector([-edge[1], edge[0]]) / edge_len
                curvature += angle_elastic * edge_len * normal.outer_product(normal)
    return curvature / (2.0 * shell_info.faces_rest_area[i_f])


@qd.func
def func_is_vert_free(i_v: int, i_b: int, shell_state: array_class.ShellState):
    """Whether a pool slot holds a vertex that the forces move."""
    return shell_state.verts_origin[i_v, i_b] >= 0 and not shell_state.verts_is_fixed[i_v, i_b]


# ------------------------------------------------------------------------------------
# -------------------------------- implicit dynamics ---------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_shell_compute_forces(
    dt: float,
    gravity: qd.Tensor,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Assemble the right-hand side dt * (f + M g) - K v of the velocity update, the diagonal blocks of M + K, and the
    per-element data the stiffness products read (see ShellScratch).

    The diagonal blocks stay assembled, the contacts adding to them before the linear solve inverts them. The warm start
    of a vertex the forces do not move is cleared, its velocity change being zero.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_scratch.hinges_stiffness.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        shell_scratch.verts_mass[i_v, i_b] = 0.0
        shell_scratch.verts_rhs[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        shell_scratch.verts_prec[i_v, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)

    for i_c, i_b in qd.ndrange(3 * n_faces, B):
        i_v = shell_state.corners_vert[i_c, i_b]
        shell_scratch.verts_mass[i_v, i_b] += shell_info.faces_mass[i_c // 3] / 3.0

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_e = shell_info.faces_entity[i_f]
        modulus = shell_info.entities_stretching_modulus[i_e] * shell_state.faces_thickness[i_f, i_b]
        nu = shell_info.entities_nu[i_e]
        rest_area = shell_info.faces_rest_area[i_f]
        F, Y = func_face_deformation(i_f, i_b, shell_state, shell_info)
        stress = func_membrane_stress(F, modulus, nu)
        lambda_0, lambda_1, eigvec_0 = func_sym2_eigen(stress)
        stress_pos = func_sym2_compose(qd.max(lambda_0, 0.0), qd.max(lambda_1, 0.0), eigvec_0)
        stiffness = dt * (dt + shell_info.entities_damping[i_e]) * rest_area
        shell_scratch.faces_F[i_f, i_b] = F
        shell_scratch.faces_stress[i_f, i_b] = stress_pos
        shell_scratch.faces_stiffness[i_f, i_b] = stiffness

        i_v0 = shell_state.corners_vert[3 * i_f, i_b]
        i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
        i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
        # Elastic force: minus the gradient of rest_area / 2 * G : stress, rest_area * F @ stress being its gradient
        # with respect to F.
        Q = rest_area * (F @ stress) @ Y.transpose()
        force_1 = -qd.Vector([Q[0, 0], Q[1, 0], Q[2, 0]])
        force_2 = -qd.Vector([Q[0, 1], Q[1, 1], Q[2, 1]])
        Kv0, Kv1, Kv2 = func_membrane_stiffness_product(
            shell_state.verts_vel[i_v0, i_b],
            shell_state.verts_vel[i_v1, i_b],
            shell_state.verts_vel[i_v2, i_b],
            F,
            stress_pos,
            Y,
            modulus,
            nu,
        )
        shell_scratch.verts_rhs[i_v0, i_b] += -dt * (force_1 + force_2) - stiffness * Kv0
        shell_scratch.verts_rhs[i_v1, i_b] += dt * force_1 - stiffness * Kv1
        shell_scratch.verts_rhs[i_v2, i_b] += dt * force_2 - stiffness * Kv2

        # Diagonal blocks of the stiffness: displacing vertex k alone changes F by dx * w_k^T.
        for k in qd.static(range(3)):
            w = -qd.Vector([Y[0, 0] + Y[1, 0], Y[0, 1] + Y[1, 1]])
            if qd.static(k > 0):
                w = qd.Vector([Y[k - 1, 0], Y[k - 1, 1]])
            M = modulus * (
                0.5 * (1.0 - nu) * w.dot(w) * qd.Matrix.identity(gs.qd_float, 2)
                + (0.5 * (1.0 - nu) + nu) * w.outer_product(w)
            )
            H = F @ M @ F.transpose() + w.dot(stress_pos @ w) * qd.Matrix.identity(gs.qd_float, 3)
            i_v = shell_state.corners_vert[3 * i_f + k, i_b]
            shell_scratch.verts_prec[i_v, i_b] += stiffness * H

    for i_h, i_b in qd.ndrange(n_hinges, B):
        shell_scratch.hinges_stiffness[i_h, i_b] = 0.0
        i_va, i_vb, i_vc, i_vd, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
        if is_intact:
            edge_b, edge_c, edge_d = func_hinge_edges(i_va, i_vb, i_vc, i_vd, i_b, shell_state)
            angle = func_hinge_angle(edge_b, edge_c, edge_d)
            grad = func_hinge_angle_gradient(edge_b, edge_c, edge_d)
            i_e = shell_info.hinges_entity[i_h]
            k_bend = func_hinge_stiffness(i_h, i_b, shell_state, shell_info)
            rest_angle = shell_info.hinges_rest_angle[i_h] + shell_state.hinges_plastic_angle[i_h, i_b]
            stiffness = dt * (dt + shell_info.entities_damping[i_e]) * k_bend
            shell_scratch.hinges_grad[i_h, i_b] = grad
            shell_scratch.hinges_stiffness[i_h, i_b] = stiffness

            grad_dot_vel = (
                grad[:, 0].dot(shell_state.verts_vel[i_va, i_b])
                + grad[:, 1].dot(shell_state.verts_vel[i_vb, i_b])
                + grad[:, 2].dot(shell_state.verts_vel[i_vc, i_b])
                + grad[:, 3].dot(shell_state.verts_vel[i_vd, i_b])
            )
            coeff = -dt * k_bend * (angle - rest_angle) - stiffness * grad_dot_vel
            shell_scratch.verts_rhs[i_va, i_b] += coeff * grad[:, 0]
            shell_scratch.verts_rhs[i_vb, i_b] += coeff * grad[:, 1]
            shell_scratch.verts_rhs[i_vc, i_b] += coeff * grad[:, 2]
            shell_scratch.verts_rhs[i_vd, i_b] += coeff * grad[:, 3]
            shell_scratch.verts_prec[i_va, i_b] += stiffness * grad[:, 0].outer_product(grad[:, 0])
            shell_scratch.verts_prec[i_vb, i_b] += stiffness * grad[:, 1].outer_product(grad[:, 1])
            shell_scratch.verts_prec[i_vc, i_b] += stiffness * grad[:, 2].outer_product(grad[:, 2])
            shell_scratch.verts_prec[i_vd, i_b] += stiffness * grad[:, 3].outer_product(grad[:, 3])

    for i_b in range(B):
        shell_scratch.envs_free_mass[i_b] = 0.0
        shell_scratch.envs_needs_solve[i_b] = True
        shell_scratch.envs_n_iterations[i_b] = 0
        shell_scratch.envs_solve_status[i_b] = SHELL_SOLVE_STATUS.CONVERGED

    for i_v, i_b in qd.ndrange(n_verts, B):
        mass = shell_scratch.verts_mass[i_v, i_b]
        if func_is_vert_free(i_v, i_b, shell_state) and mass > 0.0:
            shell_scratch.verts_rhs[i_v, i_b] += dt * mass * gravity[i_b]
            shell_scratch.verts_prec[i_v, i_b] += mass * qd.Matrix.identity(gs.qd_float, 3)
            shell_scratch.envs_free_mass[i_b] += mass
        else:
            shell_scratch.verts_rhs[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            shell_scratch.verts_prec[i_v, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            shell_state.verts_dv[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_pcg_direction(
    i_v: int, i_b: int, shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch
):
    """The vector of a vertex the next system product applies to: the search direction, or the solution while its true
    residual is checked (see PCG_MODE)."""
    x = shell_scratch.verts_p[i_v, i_b]
    if shell_scratch.envs_pcg_mode[i_b] == PCG_MODE.CHECK:
        x = shell_state.verts_dv[i_v, i_b]
    return x


@qd.func
def func_system_product(
    shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch, shell_info: array_class.ShellInfo
):
    """Add the stiffness K x of the faces and hinges to verts_Ap in every environment still solving, matrix-free, x
    being the vector func_pcg_direction selects, and x^T K x to envs_pAp.

    verts_Ap holds M x beforehand (see func_pcg_advance). The faces and the hinges share one parallel loop, so that the
    product costs a single launch. The rows of the vertices the forces do not move are left for func_pcg_residual to
    ignore.
    """
    B = shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_scratch.hinges_stiffness.shape[0]

    for i_e_, i_b in qd.ndrange(n_faces + n_hinges, B):
        if shell_scratch.envs_is_solving[i_b]:
            if i_e_ < n_faces:
                i_f = i_e_
                i_e = shell_info.faces_entity[i_f]
                i_v0 = shell_state.corners_vert[3 * i_f, i_b]
                i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
                i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
                x0 = func_pcg_direction(i_v0, i_b, shell_state, shell_scratch)
                x1 = func_pcg_direction(i_v1, i_b, shell_state, shell_scratch)
                x2 = func_pcg_direction(i_v2, i_b, shell_state, shell_scratch)
                Y = shell_info.faces_Dm_inv[i_f] @ shell_state.faces_plastic[i_f, i_b]
                stiffness = shell_scratch.faces_stiffness[i_f, i_b]
                Kp0, Kp1, Kp2 = func_membrane_stiffness_product(
                    x0,
                    x1,
                    x2,
                    shell_scratch.faces_F[i_f, i_b],
                    shell_scratch.faces_stress[i_f, i_b],
                    Y,
                    shell_info.entities_stretching_modulus[i_e] * shell_state.faces_thickness[i_f, i_b],
                    shell_info.entities_nu[i_e],
                )
                shell_scratch.verts_Ap[i_v0, i_b] += stiffness * Kp0
                shell_scratch.verts_Ap[i_v1, i_b] += stiffness * Kp1
                shell_scratch.verts_Ap[i_v2, i_b] += stiffness * Kp2
                shell_scratch.envs_pAp[i_b] += stiffness * (x0.dot(Kp0) + x1.dot(Kp1) + x2.dot(Kp2))
            else:
                i_h = i_e_ - n_faces
                stiffness = shell_scratch.hinges_stiffness[i_h, i_b]
                if stiffness > 0.0:
                    i_va, i_vb, i_vc, i_vd, _ = func_hinge_verts(i_h, i_b, shell_state, shell_info)
                    grad = shell_scratch.hinges_grad[i_h, i_b]
                    rate = (
                        grad[:, 0].dot(func_pcg_direction(i_va, i_b, shell_state, shell_scratch))
                        + grad[:, 1].dot(func_pcg_direction(i_vb, i_b, shell_state, shell_scratch))
                        + grad[:, 2].dot(func_pcg_direction(i_vc, i_b, shell_state, shell_scratch))
                        + grad[:, 3].dot(func_pcg_direction(i_vd, i_b, shell_state, shell_scratch))
                    )
                    coeff = stiffness * rate
                    shell_scratch.verts_Ap[i_va, i_b] += coeff * grad[:, 0]
                    shell_scratch.verts_Ap[i_vb, i_b] += coeff * grad[:, 1]
                    shell_scratch.verts_Ap[i_vc, i_b] += coeff * grad[:, 2]
                    shell_scratch.verts_Ap[i_vd, i_b] += coeff * grad[:, 3]
                    shell_scratch.envs_pAp[i_b] += coeff * rate


@qd.func
def func_precondition(
    i_v: int,
    i_b: int,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
):
    """Preconditioned residual of a vertex: its block-Jacobi part plus the prolongation of the coarse correction."""
    z = shell_scratch.verts_prec[i_v, i_b] @ shell_scratch.verts_r[i_v, i_b]
    if qd.static(static_config.has_coarse_space):
        i_o = shell_state.verts_origin[i_v, i_b]
        i_d = shell_info.verts_coarse_dof[i_o]
        if i_d >= 0:
            phi = shell_info.verts_coarse_phi[i_o]
            for a, j in qd.static(qd.ndrange(3, 3)):
                z[j] += phi[a] * shell_scratch.coarse_sol[i_d + 3 * a + j, i_b]
    return z


@qd.func
def func_pcg_prepare(
    pcg_flag: qd.types.ndarray(qd.i32, ndim=0),
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    static_config: qd.template(),
    tolerance: float,
    velocity_tolerance: float,
):
    """Set up the linear solve of every environment from its warm start, whose true residual it checks first.

    The block-Jacobi preconditioner is inverted, the convergence thresholds set and the product of the warm start
    prepared (see PCG_MODE).

    An environment converges once either its residual r in the norm of the block-Jacobi preconditioner D^-1 falls below
    the relative tolerance, r^T D^-1 r <= tolerance^2 * b^T D^-1 b for its right-hand side b, or the mass-weighted
    squared error of its velocity change falls below the velocity tolerance, e^T M e <= velocity_tolerance^2 * m for the
    mass m of its free vertices. The system A being at least as large as the lumped mass matrix M, r^T M^-1 r bounds
    e^T M e from above.

    The first norm weights the error by the stiffness of the sheet. It leaves out the coarse correction, which weights
    the smooth motions of a sheet orders of magnitude above its deformation, so that the coarse space changes the cost
    of a solve but not the accuracy of the stress it resolves. A stiff sheet at rest cannot resolve the first norm in
    floating point, hence the second, which needs a true bound: block-Jacobi on a stiff membrane underestimates the
    error of its smooth motions by orders of magnitude, so that z^T M z for z = D^-1 r would stop a falling sheet at
    rest.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]

    for i_b in range(B):
        shell_scratch.envs_residual_threshold[i_b] = 0.0
        shell_scratch.envs_residual_checked[i_b] = -1.0
        shell_scratch.envs_n_solve_iterations[i_b] = 0
        shell_scratch.envs_is_solving[i_b] = shell_scratch.envs_needs_solve[i_b]
        shell_scratch.envs_pcg_mode[i_b] = PCG_MODE.CHECK
        func_pcg_reset(i_b, shell_scratch, static_config)

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_needs_solve[i_b]:
            shell_scratch.verts_Ap[i_v, i_b] = shell_scratch.verts_mass[i_v, i_b] * shell_state.verts_dv[i_v, i_b]
            if func_is_vert_free(i_v, i_b, shell_state) and shell_scratch.verts_mass[i_v, i_b] > 0.0:
                prec = shell_scratch.verts_prec[i_v, i_b].inverse()
                rhs = shell_scratch.verts_rhs[i_v, i_b]
                shell_scratch.verts_prec[i_v, i_b] = prec
                shell_scratch.envs_residual_threshold[i_b] += rhs.dot(prec @ rhs)

    for i_b in range(B):
        shell_scratch.envs_residual_threshold[i_b] = tolerance * tolerance * shell_scratch.envs_residual_threshold[i_b]
        shell_scratch.envs_vel_error_threshold[i_b] = (
            velocity_tolerance * velocity_tolerance * shell_scratch.envs_free_mass[i_b]
        )
        if i_b == 0:
            pcg_flag[()] = 1


@qd.func
def func_pcg_reset(i_b: int, shell_scratch: array_class.ShellScratch, static_config: qd.template()):
    """Clear the reductions the next iteration of the solve of an environment accumulates."""
    shell_scratch.envs_pAp[i_b] = 0.0
    shell_scratch.envs_rz_new[i_b] = 0.0
    shell_scratch.envs_residual[i_b] = 0.0
    shell_scratch.envs_vel_error[i_b] = 0.0
    if qd.static(static_config.has_coarse_space):
        for i_d in range(shell_scratch.coarse_vec.shape[0]):
            shell_scratch.coarse_vec[i_d, i_b] = 0.0


@qd.func
def func_pcg_residual(
    pcg_flag: qd.types.ndarray(qd.i32, ndim=0),
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
):
    """Update the residual of every environment still solving from the system product verts_Ap, and accumulate the
    measures of the iteration.

    An environment in SOLVE mode takes the conjugate gradient step dv += alpha p, r -= alpha A p, alpha = r^T z / p^T A
    p, while CHECK mode recomputes the true residual r = b - A dv. Every free vertex then adds r^T D^-1 r to
    envs_residual, r^T M^-1 r to envs_vel_error and the restriction of r to coarse_vec, which func_pcg_coarse_solve
    completes. Clearing the device loop flag here lets func_pcg_decide raise it for the environments still solving.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]

    for i_v, i_b in qd.ndrange(n_verts, B):
        if i_v == 0 and i_b == 0:
            pcg_flag[()] = 0
        if shell_scratch.envs_is_solving[i_b]:
            if func_is_vert_free(i_v, i_b, shell_state) and shell_scratch.verts_mass[i_v, i_b] > 0.0:
                r = shell_scratch.verts_r[i_v, i_b]
                if shell_scratch.envs_pcg_mode[i_b] == PCG_MODE.SOLVE:
                    p_A_p = shell_scratch.envs_pAp[i_b]
                    if p_A_p > 0.0:
                        alpha = shell_scratch.envs_rz[i_b] / p_A_p
                        shell_state.verts_dv[i_v, i_b] += alpha * shell_scratch.verts_p[i_v, i_b]
                        r = r - alpha * shell_scratch.verts_Ap[i_v, i_b]
                else:
                    r = shell_scratch.verts_rhs[i_v, i_b] - shell_scratch.verts_Ap[i_v, i_b]
                shell_scratch.verts_r[i_v, i_b] = r
                shell_scratch.envs_residual[i_b] += r.dot(shell_scratch.verts_prec[i_v, i_b] @ r)
                shell_scratch.envs_vel_error[i_b] += r.norm_sqr() / shell_scratch.verts_mass[i_v, i_b]
                if qd.static(static_config.has_coarse_space):
                    i_o = shell_state.verts_origin[i_v, i_b]
                    i_d = shell_info.verts_coarse_dof[i_o]
                    if i_d >= 0:
                        phi = shell_info.verts_coarse_phi[i_o]
                        for a, j in qd.static(qd.ndrange(3, 3)):
                            shell_scratch.coarse_vec[i_d + 3 * a + j, i_b] += phi[a] * r[j]
            else:
                shell_scratch.verts_r[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_pcg_coarse_solve(shell_scratch: array_class.ShellScratch, shell_info: array_class.ShellInfo):
    """Solve the coarse system of every environment still solving for its restricted residual.

    The coarse correction goes to coarse_sol and its share of r^T z to envs_rz_new (see the coarse space in
    kernel_shell_coarse_factorize).
    """
    B = shell_scratch.coarse_vec.shape[1]

    for i_d, i_b in qd.ndrange(shell_scratch.coarse_vec.shape[0], B):
        if shell_scratch.envs_is_solving[i_b]:
            i_e = shell_info.coarse_dofs_entity[i_d]
            dof_start = shell_info.entities_coarse_dof_start[i_e]
            dim = shell_info.entities_coarse_dim[i_e]
            row_start = shell_info.entities_coarse_matrix_start[i_e] + (i_d - dof_start) * dim
            value = gs.qd_float(0.0)
            for j in range(dim):
                value += shell_scratch.coarse_matrix[i_b, row_start + j] * shell_scratch.coarse_vec[dof_start + j, i_b]
            shell_scratch.coarse_sol[i_d, i_b] = value
            shell_scratch.envs_rz_new[i_b] += shell_scratch.coarse_vec[i_d, i_b] * value


@qd.func
def func_pcg_decide(
    pcg_flag: qd.types.ndarray(qd.i32, ndim=0),
    shell_scratch: array_class.ShellScratch,
    static_config: qd.template(),
    max_iterations: int,
    errno: qd.Tensor,
):
    """Decide the next iteration of every environment still solving, from the measures of the current one.

    An environment in SOLVE mode whose recursive residual falls below its threshold switches to CHECK mode, whose
    product recomputes the true residual b - A dv: the solve stops if that one converged too, and restarts the conjugate
    gradient from it otherwise, which also covers the warm start. The solve of an environment stops with a failure
    status when it reaches max_iterations products, when its search direction loses positive curvature, or when its
    residual is not finite, in which case errno flags it. It also stops when a restart leaves its true residual above
    STAGNATION_RATIO times the one of the previous restart: the recursive residual then keeps converging while the true
    one sits at the floor of the floating-point precision, so that further restarts would only spend the iteration
    limit. The step envs_step of a continuing solve holds the conjugate gradient coefficient beta, zero on a restart.
    """
    B = shell_scratch.envs_is_solving.shape[0]

    for i_b in range(B):
        if shell_scratch.envs_is_solving[i_b]:
            residual = shell_scratch.envs_residual[i_b]
            rz_new = residual + shell_scratch.envs_rz_new[i_b]
            shell_scratch.envs_n_iterations[i_b] += 1
            shell_scratch.envs_n_solve_iterations[i_b] += 1
            shell_scratch.envs_step[i_b] = 0.0
            if shell_scratch.envs_pcg_mode[i_b] == PCG_MODE.SOLVE and shell_scratch.envs_pAp[i_b] <= 0.0:
                shell_scratch.envs_is_solving[i_b] = False
                shell_scratch.envs_solve_status[i_b] |= SHELL_SOLVE_STATUS.BREAKDOWN
            elif qd.math.isnan(rz_new) or qd.math.isinf(rz_new):
                shell_scratch.envs_is_solving[i_b] = False
                shell_scratch.envs_solve_status[i_b] |= SHELL_SOLVE_STATUS.NON_FINITE
                errno[i_b] = errno[i_b] | array_class.ErrorCode.INVALID_SHELL_SOLVE_NAN
            elif (
                residual <= shell_scratch.envs_residual_threshold[i_b]
                or shell_scratch.envs_vel_error[i_b] <= shell_scratch.envs_vel_error_threshold[i_b]
            ):
                if shell_scratch.envs_pcg_mode[i_b] == PCG_MODE.CHECK:
                    shell_scratch.envs_is_solving[i_b] = False
                else:
                    shell_scratch.envs_pcg_mode[i_b] = PCG_MODE.CHECK
            elif shell_scratch.envs_pcg_mode[i_b] == PCG_MODE.CHECK:
                residual_checked = shell_scratch.envs_residual_checked[i_b]
                if residual_checked >= 0.0 and residual > STAGNATION_RATIO * residual_checked:
                    shell_scratch.envs_is_solving[i_b] = False
                    shell_scratch.envs_solve_status[i_b] |= SHELL_SOLVE_STATUS.STAGNATION
                else:
                    # Restart from the true residual, the search direction being the preconditioned residual
                    shell_scratch.envs_residual_checked[i_b] = residual
                    shell_scratch.envs_pcg_mode[i_b] = PCG_MODE.SOLVE
            else:
                shell_scratch.envs_step[i_b] = rz_new / shell_scratch.envs_rz[i_b]
            shell_scratch.envs_rz[i_b] = rz_new
            if shell_scratch.envs_is_solving[i_b] and shell_scratch.envs_n_solve_iterations[i_b] >= max_iterations:
                shell_scratch.envs_is_solving[i_b] = False
                shell_scratch.envs_solve_status[i_b] |= SHELL_SOLVE_STATUS.MAX_ITERATIONS
            if shell_scratch.envs_is_solving[i_b]:
                pcg_flag[()] = 1
            func_pcg_reset(i_b, shell_scratch, static_config)


@qd.func
def func_pcg_advance(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
):
    """Set the vector the next system product of every environment still solving applies to, with its mass part.

    The vector is the search direction p = z + beta p for z the preconditioned residual, or the solution while its true
    residual is checked. Its mass part M x goes to verts_Ap, and the mass part of p^T A p to envs_pAp.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_is_solving[i_b]:
            mass = shell_scratch.verts_mass[i_v, i_b]
            if shell_scratch.envs_pcg_mode[i_b] == PCG_MODE.SOLVE:
                p = qd.Vector.zero(gs.qd_float, 3)
                if func_is_vert_free(i_v, i_b, shell_state) and mass > 0.0:
                    p = (
                        func_precondition(i_v, i_b, shell_state, shell_scratch, shell_info, static_config)
                        + shell_scratch.envs_step[i_b] * shell_scratch.verts_p[i_v, i_b]
                    )
                shell_scratch.verts_p[i_v, i_b] = p
                shell_scratch.verts_Ap[i_v, i_b] = mass * p
                shell_scratch.envs_pAp[i_b] += mass * p.norm_sqr()
            else:
                shell_scratch.verts_Ap[i_v, i_b] = mass * shell_state.verts_dv[i_v, i_b]


@qd.kernel(graph=True)
def kernel_shell_pcg_solve(
    pcg_flag: qd.types.ndarray(qd.i32, ndim=0),
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
    max_iterations: int,
    tolerance: float,
    velocity_tolerance: float,
    errno: qd.Tensor,
):
    """Solve the velocity update of every environment by preconditioned conjugate gradient (PCG), warm-started from
    verts_dv, until each one converges or fails (see func_pcg_prepare and func_pcg_decide).

    The iterations loop on the device while any environment iterates, so that the solve costs the iterations of its
    slowest environment, with no host synchronization. An iteration fuses its reductions into the passes that produce
    their terms, a graph node costing more than the work of a pass over a few hundred vertices.
    """
    func_pcg_prepare(pcg_flag, shell_state, shell_scratch, static_config, tolerance, velocity_tolerance)
    while qd.graph.do_while(pcg_flag):
        func_system_product(shell_state, shell_scratch, shell_info)
        func_pcg_residual(pcg_flag, shell_state, shell_scratch, shell_info, static_config)
        if qd.static(static_config.has_coarse_space):
            func_pcg_coarse_solve(shell_scratch, shell_info)
        func_pcg_decide(pcg_flag, shell_scratch, static_config, max_iterations, errno)
        func_pcg_advance(shell_state, shell_scratch, shell_info, static_config)


@qd.func
def func_select3(i: int, x0, x1, x2):
    """Return x0, x1 or x2 by index, for indexing local values at runtime."""
    x = x0
    if i == 1:
        x = x1
    elif i == 2:
        x = x2
    return x


@qd.func
def func_coarse_add_block(
    i_e: int,
    i_d_row: int,
    i_d_col: int,
    block: qd.types.matrix(3, 3),
    i_b: int,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Add a 3x3 block to the lower triangle of the coarse matrix of an entity, at global coarse rows and columns."""
    dof_start = shell_info.entities_coarse_dof_start[i_e]
    dim = shell_info.entities_coarse_dim[i_e]
    matrix_start = shell_info.entities_coarse_matrix_start[i_e]
    for j, k in qd.static(qd.ndrange(3, 3)):
        row = i_d_row - dof_start + j
        col = i_d_col - dof_start + k
        if row >= col:
            shell_scratch.coarse_assembly[matrix_start + row * dim + col, i_b] += block[j, k]


@qd.kernel
def kernel_shell_coarse_assemble(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    coarse_update_interval: int,
):
    """Assemble the coarse matrix Z^T (M + K) Z of every entity, Z spanning the displacements affine in
    the rest coordinates of each patch of vertices.

    Every patch of vertices moves by a displacement affine in the in-plane rest coordinates of its vertices, which
    captures the smooth stretching and bending that block-Jacobi preconditioning resolves slowly in stiff sheets. The
    stiffness of an element being bilinear in the displacement of its vertices, its coarse block for a pair of shape
    functions is its stiffness evaluated on the sum of the vertex weights times those shape functions, per patch.
    The Cholesky factorization drops the pivots that vanish (fixed or degenerate patches), solving the coarse system
    on the remaining unknowns. Only the environments whose coarse matrices are due (see envs_coarse_age) are assembled.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_scratch.hinges_stiffness.shape[0]
    n_entities = shell_state.entities_n_verts.shape[0]

    for i_m, i_b in qd.ndrange(shell_scratch.coarse_assembly.shape[0], B):
        if shell_scratch.envs_coarse_age[i_b] >= coarse_update_interval:
            shell_scratch.coarse_assembly[i_m, i_b] = 0.0

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_coarse_age[i_b] >= coarse_update_interval and func_is_vert_free(i_v, i_b, shell_state):
            i_o = shell_state.verts_origin[i_v, i_b]
            i_d = shell_info.verts_coarse_dof[i_o]
            if i_d >= 0:
                i_e = func_vert_entity(i_v, i_b, shell_state, shell_info)
                phi = shell_info.verts_coarse_phi[i_o]
                mass = shell_scratch.verts_mass[i_v, i_b]
                for a in range(3):
                    for b in range(a + 1):
                        coeff = mass * func_select3(a, phi[0], phi[1], phi[2]) * func_select3(b, phi[0], phi[1], phi[2])
                        func_coarse_add_block(
                            i_e,
                            i_d + 3 * a,
                            i_d + 3 * b,
                            coeff * qd.Matrix.identity(gs.qd_float, 3),
                            i_b,
                            shell_scratch,
                            shell_info,
                        )

    # Faces: the coarse block of shape functions (a of patch P, b of patch Q) evaluates the membrane stiffness on the
    # weights W_Pa = sum over the free face vertices k of patch P of phi_k[a] * w_k, w_k mapping a displacement of
    # vertex k to the change of deformation gradient (see func_membrane_stiffness_product).
    for i_f, i_b in qd.ndrange(n_faces, B):
        i_e = shell_info.faces_entity[i_f]
        if shell_scratch.envs_coarse_age[i_b] >= coarse_update_interval and shell_info.entities_coarse_dim[i_e] > 0:
            nu = shell_info.entities_nu[i_e]
            modulus = shell_info.entities_stretching_modulus[i_e] * shell_state.faces_thickness[i_f, i_b]
            stiffness = shell_scratch.faces_stiffness[i_f, i_b]
            F = shell_scratch.faces_F[i_f, i_b]
            stress_pos = shell_scratch.faces_stress[i_f, i_b]
            Y = shell_info.faces_Dm_inv[i_f] @ shell_state.faces_plastic[i_f, i_b]
            w_1 = qd.Vector([Y[0, 0], Y[0, 1]])
            w_2 = qd.Vector([Y[1, 0], Y[1, 1]])
            w_0 = -(w_1 + w_2)
            i_v0 = shell_state.corners_vert[3 * i_f, i_b]
            i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
            i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
            is_free_0 = func_is_vert_free(i_v0, i_b, shell_state)
            is_free_1 = func_is_vert_free(i_v1, i_b, shell_state)
            is_free_2 = func_is_vert_free(i_v2, i_b, shell_state)
            i_o0 = shell_state.verts_origin[i_v0, i_b]
            i_o1 = shell_state.verts_origin[i_v1, i_b]
            i_o2 = shell_state.verts_origin[i_v2, i_b]
            dof_0 = shell_info.verts_coarse_dof[i_o0]
            dof_1 = shell_info.verts_coarse_dof[i_o1]
            dof_2 = shell_info.verts_coarse_dof[i_o2]
            phi_0 = shell_info.verts_coarse_phi[i_o0]
            phi_1 = shell_info.verts_coarse_phi[i_o1]
            phi_2 = shell_info.verts_coarse_phi[i_o2]
            for k, l, a, b in qd.ndrange(3, 3, 3, 3):
                dof_k = func_select3(k, dof_0, dof_1, dof_2)
                dof_l = func_select3(l, dof_0, dof_1, dof_2)
                # Each patch of the face is handled by its first vertex, and the pair of patches once, in the lower
                # triangle.
                is_leader_k = k == 0 or (k == 1 and dof_1 != dof_0) or (k == 2 and dof_2 != dof_0 and dof_2 != dof_1)
                is_leader_l = l == 0 or (l == 1 and dof_1 != dof_0) or (l == 2 and dof_2 != dof_0 and dof_2 != dof_1)
                if is_leader_k and is_leader_l and dof_k >= dof_l:
                    W_k = qd.Vector.zero(gs.qd_float, 2)
                    W_l = qd.Vector.zero(gs.qd_float, 2)
                    if is_free_0:
                        W_k += (dof_0 == dof_k) * func_select3(a, phi_0[0], phi_0[1], phi_0[2]) * w_0
                        W_l += (dof_0 == dof_l) * func_select3(b, phi_0[0], phi_0[1], phi_0[2]) * w_0
                    if is_free_1:
                        W_k += (dof_1 == dof_k) * func_select3(a, phi_1[0], phi_1[1], phi_1[2]) * w_1
                        W_l += (dof_1 == dof_l) * func_select3(b, phi_1[0], phi_1[1], phi_1[2]) * w_1
                    if is_free_2:
                        W_k += (dof_2 == dof_k) * func_select3(a, phi_2[0], phi_2[1], phi_2[2]) * w_2
                        W_l += (dof_2 == dof_l) * func_select3(b, phi_2[0], phi_2[1], phi_2[2]) * w_2
                    M = modulus * (
                        0.5 * (1.0 - nu) * (W_k.dot(W_l) * qd.Matrix.identity(gs.qd_float, 2) + W_l.outer_product(W_k))
                        + nu * W_k.outer_product(W_l)
                    )
                    block = stiffness * (
                        F @ M @ F.transpose() + W_k.dot(stress_pos @ W_l) * qd.Matrix.identity(gs.qd_float, 3)
                    )
                    if dof_k >= 0 and dof_l >= 0:
                        func_coarse_add_block(i_e, dof_k + 3 * a, dof_l + 3 * b, block, i_b, shell_scratch, shell_info)

    # Hinges: the same with the gradient of the dihedral angle, the stiffness being k * grad grad^T
    for i_h, i_b in qd.ndrange(n_hinges, B):
        stiffness = shell_scratch.hinges_stiffness[i_h, i_b]
        if shell_scratch.envs_coarse_age[i_b] >= coarse_update_interval and stiffness > 0.0:
            i_e = shell_info.hinges_entity[i_h]
            if shell_info.entities_coarse_dim[i_e] > 0:
                i_va, i_vb, i_vc, i_vd, _ = func_hinge_verts(i_h, i_b, shell_state, shell_info)
                grad = shell_scratch.hinges_grad[i_h, i_b]
                for k, l, a, b in qd.ndrange(4, 4, 3, 3):
                    i_vk = i_va
                    i_vl = i_va
                    G_k = qd.Vector.zero(gs.qd_float, 3)
                    G_l = qd.Vector.zero(gs.qd_float, 3)
                    is_leader_k = True
                    is_leader_l = True
                    dof_k = -1
                    dof_l = -1
                    for m in qd.static(range(4)):
                        i_vm = i_va
                        if qd.static(m == 1):
                            i_vm = i_vb
                        elif qd.static(m == 2):
                            i_vm = i_vc
                        elif qd.static(m == 3):
                            i_vm = i_vd
                        if m == k:
                            i_vk = i_vm
                        if m == l:
                            i_vl = i_vm
                    i_ok = shell_state.verts_origin[i_vk, i_b]
                    i_ol = shell_state.verts_origin[i_vl, i_b]
                    dof_k = shell_info.verts_coarse_dof[i_ok]
                    dof_l = shell_info.verts_coarse_dof[i_ol]
                    for m in qd.static(range(4)):
                        i_vm = i_va
                        if qd.static(m == 1):
                            i_vm = i_vb
                        elif qd.static(m == 2):
                            i_vm = i_vc
                        elif qd.static(m == 3):
                            i_vm = i_vd
                        i_om = shell_state.verts_origin[i_vm, i_b]
                        dof_m = shell_info.verts_coarse_dof[i_om]
                        if m < k and dof_m == dof_k:
                            is_leader_k = False
                        if m < l and dof_m == dof_l:
                            is_leader_l = False
                        if func_is_vert_free(i_vm, i_b, shell_state):
                            phi_m = shell_info.verts_coarse_phi[i_om]
                            grad_m = qd.Vector([grad[0, m], grad[1, m], grad[2, m]])
                            if dof_m == dof_k:
                                G_k += func_select3(a, phi_m[0], phi_m[1], phi_m[2]) * grad_m
                            if dof_m == dof_l:
                                G_l += func_select3(b, phi_m[0], phi_m[1], phi_m[2]) * grad_m
                    if is_leader_k and is_leader_l and dof_k >= dof_l and dof_l >= 0:
                        func_coarse_add_block(
                            i_e,
                            dof_k + 3 * a,
                            dof_l + 3 * b,
                            stiffness * G_k.outer_product(G_l),
                            i_b,
                            shell_scratch,
                            shell_info,
                        )


@qd.kernel
def kernel_shell_coarse_factorize(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
    static_config: qd.template(),
    coarse_update_interval: int,
):
    """Factorize and invert the coarse matrix of every entity in the environments it is due in (see
    kernel_shell_coarse_assemble), whose age then restarts."""
    B = shell_state.verts_pos.shape[1]
    n_entities = shell_state.entities_n_verts.shape[0]

    # Cholesky factorization, dropping the pivots that vanish (fixed or degenerate patches) so that the coarse system is
    # solved on the remaining unknowns, then explicit inversion, which turns the coarse solve of every iteration into a
    # parallel matrix-vector product. The lanes of a block share the rows of one entity in one environment.
    _K = qd.static(static_config.coarse_block_dim)
    if qd.static(_K > 1):
        qd.loop_config(block_dim=_K)
    for i_flat in range(n_entities * B * _K):
        tid = i_flat % _K
        i_b = (i_flat // _K) % B
        i_e = i_flat // (_K * B)
        # The lanes of a block share their environment, so that they all skip it together
        dim = shell_info.entities_coarse_dim[i_e]
        if shell_scratch.envs_coarse_age[i_b] < coarse_update_interval:
            dim = 0
        matrix_start = shell_info.entities_coarse_matrix_start[i_e]
        for i_chunk in range((dim * dim + _K - 1) // _K):
            i_entry = i_chunk * _K + tid
            if i_entry < dim * dim:
                shell_scratch.coarse_matrix[i_b, matrix_start + i_entry] = shell_scratch.coarse_assembly[
                    matrix_start + i_entry, i_b
                ]
        if qd.static(_K > 1):
            qd.simt.block.sync()
        for j in range(dim):
            if tid == 0:
                diag = shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + j]
                pivot_sq = diag
                for k in range(j):
                    pivot_sq -= shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + k] ** 2
                pivot = gs.qd_float(0.0)
                if pivot_sq > 1e-6 * diag:
                    pivot = qd.sqrt(pivot_sq)
                shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + j] = pivot
            if qd.static(_K > 1):
                qd.simt.block.sync()
            pivot = shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + j]
            for i_chunk in range((dim - j - 1 + _K - 1) // _K):
                i = j + 1 + i_chunk * _K + tid
                if i < dim:
                    value = gs.qd_float(0.0)
                    if pivot > 0.0:
                        value = shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + j]
                        for k in range(j):
                            value -= (
                                shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + k]
                                * shell_scratch.coarse_matrix[i_b, matrix_start + j * dim + k]
                            )
                        value = value / pivot
                    shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + j] = value
            if qd.static(_K > 1):
                qd.simt.block.sync()

        # Inverse of the lower triangular factor, one column per lane by forward substitution
        for k_chunk in range((dim + _K - 1) // _K):
            k = k_chunk * _K + tid
            if k < dim:
                for i in range(k, dim):
                    value = gs.qd_float(1.0) if i == k else gs.qd_float(0.0)
                    for m in range(k, i):
                        value -= (
                            shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + m]
                            * shell_scratch.coarse_factor_inv[i_b, matrix_start + m * dim + k]
                        )
                    pivot = shell_scratch.coarse_matrix[i_b, matrix_start + i * dim + i]
                    shell_scratch.coarse_factor_inv[i_b, matrix_start + i * dim + k] = (
                        value / pivot if pivot > 0.0 else 0.0
                    )
        if qd.static(_K > 1):
            qd.simt.block.sync()

        # Inverse of the coarse matrix, L^-T L^-1, written over the factor that is no longer needed
        for i_chunk in range((dim * dim + _K - 1) // _K):
            i_entry = i_chunk * _K + tid
            if i_entry < dim * dim:
                i = i_entry // dim
                j = i_entry % dim
                value = gs.qd_float(0.0)
                for k in range(qd.max(i, j), dim):
                    value += (
                        shell_scratch.coarse_factor_inv[i_b, matrix_start + k * dim + i]
                        * shell_scratch.coarse_factor_inv[i_b, matrix_start + k * dim + j]
                    )
                shell_scratch.coarse_matrix[i_b, matrix_start + i_entry] = value
        if qd.static(_K > 1):
            qd.simt.block.sync()

    for i_b in range(B):
        if shell_scratch.envs_coarse_age[i_b] >= coarse_update_interval:
            shell_scratch.envs_coarse_age[i_b] = 0


@qd.kernel
def kernel_shell_apply_dv(shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch):
    """Add the solved velocity change to the free vertices, and record the failures of the solve of every environment.

    A non-finite solution is dropped and its warm start cleared, errno halting the simulation at the next check.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_solve_status[i_b] & SHELL_SOLVE_STATUS.NON_FINITE:
            shell_state.verts_dv[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        elif func_is_vert_free(i_v, i_b, shell_state):
            shell_state.verts_vel[i_v, i_b] += shell_state.verts_dv[i_v, i_b]

    for i_b in range(B):
        shell_state.envs_solver_failure[i_b] = shell_state.envs_solver_failure[i_b] | (
            shell_scratch.envs_solve_status[i_b] & SHELL_SOLVE_FAILURE
        )


@qd.kernel
def kernel_shell_integrate(dt: float, shell_state: array_class.ShellState, shell_scratch: array_class.ShellScratch):
    """Advance the position of every vertex by its velocity, moving whole grid cells from its offset to its cell.

    The offset staying below a cell, its increments keep their precision, and moving whole cells out of it is exact.
    The coarse matrices of every environment age by one substep.
    """
    for i_v, i_b in qd.ndrange(shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]):
        if shell_state.verts_origin[i_v, i_b] >= 0:
            offset = shell_state.verts_pos_offset[i_v, i_b] + dt * shell_state.verts_vel[i_v, i_b]
            cell_shift = qd.floor(offset / POS_GRID + 0.5).cast(gs.qd_int)
            cell = shell_state.verts_pos_cell[i_v, i_b] + cell_shift
            offset = offset - cell_shift.cast(gs.qd_float) * POS_GRID
            shell_state.verts_pos_cell[i_v, i_b] = cell
            shell_state.verts_pos_offset[i_v, i_b] = offset
            shell_state.verts_pos[i_v, i_b] = cell.cast(gs.qd_float) * POS_GRID + offset

    for i_b in range(shell_scratch.envs_coarse_age.shape[0]):
        shell_scratch.envs_coarse_age[i_b] += 1


# ------------------------------------------------------------------------------------
# ----------------------------------- plasticity -------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_shell_plastic_flow(
    dt: float,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Yield the faces whose von Mises stress exceeds the yield stress, and the hinges bent past the yield curvature.

    The plastic stretching flows multiplicatively, F_el <- F_el @ V Sigma^-gamma V^T up to a dilation that thins the
    face so as to conserve its volume, gamma growing with the relative overstress. The plastic bending moves the rest
    angle of a hinge until its elastic curvature, 3 * len * angle / (2 * area), drops to the yield curvature.
    """
    B = shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_scratch.hinges_stiffness.shape[0]

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_e = shell_info.faces_entity[i_f]
        yield_stress = shell_info.entities_yield_stress[i_e]
        if yield_stress > 0.0:
            thickness = shell_state.faces_thickness[i_f, i_b]
            F, _ = func_face_deformation(i_f, i_b, shell_state, shell_info)
            stress = func_membrane_stress(F, shell_info.entities_stretching_modulus[i_e], shell_info.entities_nu[i_e])
            von_mises = qd.sqrt(
                qd.max(
                    stress[0, 0] ** 2 + stress[1, 1] ** 2 - stress[0, 0] * stress[1, 1] + 3.0 * stress[0, 1] ** 2,
                    0.0,
                )
            )
            gamma = qd.math.clamp(
                dt * shell_info.entities_plastic_flow_rate[i_e] * (von_mises - yield_stress) / yield_stress, 0.0, 1.0
            )
            if gamma > 0.0:
                lambda_0, lambda_1, eigvec_0 = func_sym2_eigen(F.transpose() @ F)
                sigma_0 = qd.sqrt(qd.max(lambda_0, gs.EPS))
                sigma_1 = qd.sqrt(qd.max(lambda_1, gs.EPS))
                det_sigma = sigma_0 * sigma_1
                flow = func_sym2_compose(sigma_0 ** (-gamma), sigma_1 ** (-gamma), eigvec_0) * det_sigma ** (
                    gamma / 3.0
                )
                shell_state.faces_plastic[i_f, i_b] = shell_state.faces_plastic[i_f, i_b] @ flow
                shell_state.faces_thickness[i_f, i_b] = thickness * det_sigma ** (-gamma / 3.0)

    for i_h, i_b in qd.ndrange(n_hinges, B):
        i_e = shell_info.hinges_entity[i_h]
        yield_curvature = shell_info.entities_yield_curvature[i_e]
        if yield_curvature > 0.0:
            i_va, i_vb, i_vc, i_vd, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
            if is_intact:
                edge_b, edge_c, edge_d = func_hinge_edges(i_va, i_vb, i_vc, i_vd, i_b, shell_state)
                angle = func_hinge_angle(edge_b, edge_c, edge_d)
                angle_scale = 2.0 * shell_info.hinges_rest_area[i_h] / (3.0 * shell_info.hinges_rest_len[i_h])
                angle_elastic = angle - shell_info.hinges_rest_angle[i_h] - shell_state.hinges_plastic_angle[i_h, i_b]
                curvature = angle_elastic / angle_scale
                if qd.abs(curvature) > yield_curvature:
                    shell_state.hinges_plastic_angle[i_h, i_b] += (
                        qd.math.sign(curvature) * (qd.abs(curvature) - yield_curvature) * angle_scale
                    )


# ------------------------------------------------------------------------------------
# ------------------------------------- damage ---------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_face_damage(i_f: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Damage index of a face: the largest principal stress of the two outer surfaces of the sheet over its tensile
    strength, zero for an entity without one (see kernel_shell_damage)."""
    i_e = shell_info.faces_entity[i_f]
    tensile_strength = shell_info.entities_tensile_strength[i_e]
    damage = gs.qd_float(0.0)
    if tensile_strength > 0.0:
        nu = shell_info.entities_nu[i_e]
        thickness = shell_state.faces_thickness[i_f, i_b]
        F = func_face_deformation(i_f, i_b, shell_state, shell_info)[0]
        membrane = func_membrane_stress(F, shell_info.entities_stretching_modulus[i_e], nu)
        curvature = func_face_curvature(i_f, i_b, shell_state, shell_info)
        # The outer-fiber bending stress (see kernel_shell_damage), D being bending_modulus * h^3
        bending = (
            6.0
            * shell_info.entities_bending_modulus[i_e]
            * thickness
            * ((1.0 - nu) * curvature + nu * curvature.trace() * qd.Matrix.identity(gs.qd_float, 2))
        )
        stress_top = func_sym2_eigen(membrane + bending)[0]
        stress_bottom = func_sym2_eigen(membrane - bending)[0]
        damage = qd.max(qd.max(stress_top, stress_bottom), 0.0) / tensile_strength
    return damage


@qd.kernel
def kernel_shell_damage(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Evaluate the damage index of every face, and record the peak and the first failure of every entity.

    The damage index is the Rankine (maximum principal stress) criterion on the two outer surfaces of the sheet, over
    the tensile strength: the stress of a surface is the membrane stress plus or minus the outer-fiber bending stress.
    The membrane stress is the plane-stress Saint Venant-Kirchhoff second Piola-Kirchhoff stress of the elastic Green
    strain, in Pa. The bending stress is 6 M / h^2 for the Kirchhoff plate moment M = D ((1 - nu) S + nu tr(S) I) of
    the elastic curvature S of the face (see func_face_curvature), D being the bending stiffness of the material. An
    entity fails in the first substep its peak reaches one, at the face of largest damage index.
    """
    B = shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_entities = shell_state.entities_n_verts.shape[0]

    for i_f, i_b in qd.ndrange(n_faces, B):
        shell_scratch.faces_damage[i_f, i_b] = func_face_damage(i_f, i_b, shell_state, shell_info)

    # One lane per entity and environment scans its faces in order, so that the face of a tie is deterministic
    for i_e, i_b in qd.ndrange(n_entities, B):
        damage_max = gs.qd_float(0.0)
        i_f_max = -1
        for i_f in range(shell_info.entities_face_start[i_e], shell_info.entities_face_end[i_e]):
            damage = shell_scratch.faces_damage[i_f, i_b]
            if damage > damage_max:
                damage_max = damage
                i_f_max = i_f
        if damage_max >= 1.0 and shell_state.entities_failure_face[i_e, i_b] < 0:
            shell_state.entities_failure_face[i_e, i_b] = i_f_max - shell_info.entities_face_start[i_e]
        shell_state.entities_peak_damage[i_e, i_b] = qd.max(shell_state.entities_peak_damage[i_e, i_b], damage_max)


# ------------------------------------------------------------------------------------
# ------------------------------------ fracture --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_fan_entry_is_owned(
    i_v: int, i_j: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo
):
    """Whether the corner of fan entry i_j still belongs to vertex i_v."""
    i_c = shell_info.fans_corner[i_j]
    return shell_state.corners_vert[i_c, i_b] == i_v


@qd.func
def func_fan_link_is_intact(
    i_v: int, i_j: int, i_j_next: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo
):
    """Whether vertex i_v holds the corners of consecutive fan entries i_j and i_j_next, joined by an intact hinge."""
    is_intact = False
    i_h = shell_info.fans_next_hinge[i_j]
    if i_h >= 0:
        if func_fan_entry_is_owned(i_v, i_j, i_b, shell_state, shell_info) and func_fan_entry_is_owned(
            i_v, i_j_next, i_b, shell_state, shell_info
        ):
            _, _, _, _, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
    return is_intact


@qd.func
def func_vert_arc(
    i_v: int, i_arc: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo
):
    """Locate the corners of a vertex within the fan of the original vertex it descends from.

    The corners a vertex holds form arcs of consecutive fan entries joined by intact hinges. Returns the fan start and
    length of its original vertex, the number of arcs (zero for a vertex holding its whole closed fan, which forms a
    ring), and the first entry (relative to the fan start) and length of arc i_arc, or of the ring.
    """
    i_o = shell_state.verts_origin[i_v, i_b]
    fan_start = shell_info.verts_fan_start[i_o]
    fan_len = shell_info.verts_fan_len[i_o]
    is_closed = shell_info.verts_is_fan_closed[i_o]
    n_arcs = 0
    n_owned = 0
    arc_start = -1
    for k in range(fan_len):
        is_owned = func_fan_entry_is_owned(i_v, fan_start + k, i_b, shell_state, shell_info)
        has_link_prev = False
        if k > 0:
            has_link_prev = func_fan_link_is_intact(i_v, fan_start + k - 1, fan_start + k, i_b, shell_state, shell_info)
        elif is_closed:
            has_link_prev = func_fan_link_is_intact(
                i_v, fan_start + fan_len - 1, fan_start, i_b, shell_state, shell_info
            )
        if is_owned:
            n_owned += 1
            if not has_link_prev:
                if n_arcs == i_arc:
                    arc_start = k
                n_arcs += 1
    arc_len = 0
    if n_arcs == 0 and n_owned > 0:
        arc_start = 0
        arc_len = fan_len
    elif arc_start >= 0:
        arc_len = 1
        for k in range(1, fan_len):
            i_j = fan_start + (arc_start + k - 1) % fan_len
            i_j_next = fan_start + (arc_start + k) % fan_len
            if not func_fan_link_is_intact(i_v, i_j, i_j_next, i_b, shell_state, shell_info):
                break
            arc_len += 1
    return fan_start, fan_len, n_arcs, arc_start, arc_len


@qd.func
def func_corner_traction(
    i_c: int, i_b: int, shell_scratch: array_class.ShellScratch, shell_info: array_class.ShellInfo
):
    """Return the traction a corner sector of a vertex fan transmits across a small disc around the vertex, in the
    material space, and the rest angle of the corner.

    The traction across the arc of the disc inside the face integrates to stress @ (t_end - t_start), t being the
    unit edge directions turned a quarter counter-clockwise, which is what the fracture criterion sums over a side of
    a candidate split.
    """
    i_f = i_c // 3
    k = i_c % 3
    Dm = shell_info.faces_Dm[i_f]
    p1 = qd.Vector([Dm[0, 0], Dm[1, 0]])
    p2 = qd.Vector([Dm[0, 1], Dm[1, 1]])
    edge_start = p1
    edge_end = p2
    if k == 1:
        edge_start = p2 - p1
        edge_end = -p1
    elif k == 2:
        edge_start = -p2
        edge_end = p1 - p2
    edge_start = edge_start / qd.max(edge_start.norm(), NORM_FLOOR)
    edge_end = edge_end / qd.max(edge_end.norm(), NORM_FLOOR)
    stress = shell_scratch.faces_fracture_stress[i_f, i_b]
    traction = stress @ (qd.Vector([-edge_end[1], edge_end[0]]) - qd.Vector([-edge_start[1], edge_start[0]]))
    angle = qd.acos(qd.math.clamp(edge_start.dot(edge_end), -1.0, 1.0))
    return shell_info.faces_basis[i_f] @ traction, angle


@qd.func
def func_split_score(traction_0: qd.types.vector(3), traction_1: qd.types.vector(3), is_open: bool):
    """Stress a split relieves, in N/m: the smaller of the opposing tractions its two sides pull apart with."""
    score = gs.qd_float(0.0)
    if traction_0.dot(traction_1) < 0.0:
        traction_diff = traction_0 - traction_1
        mid = traction_diff / qd.max(traction_diff.norm(), NORM_FLOOR)
        score = 0.5 * qd.min(qd.abs(traction_0.dot(mid)), qd.abs(traction_1.dot(mid)))
        if is_open:
            score = 2.0 * score
    return score


@qd.func
def func_alloc_vert(
    i_v: int, i_e: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo
):
    """Fill a free slot of the pool of an entity with a copy of vertex i_v, returning it, or -1 if none is left."""
    i_new = shell_info.entities_vert_start[i_e] + qd.atomic_add(shell_state.entities_n_verts[i_e, i_b], 1)
    if i_new >= shell_info.entities_vert_end[i_e]:
        qd.atomic_sub(shell_state.entities_n_verts[i_e, i_b], 1)
        i_new = -1
    else:
        shell_state.verts_pos[i_new, i_b] = shell_state.verts_pos[i_v, i_b]
        shell_state.verts_pos_cell[i_new, i_b] = shell_state.verts_pos_cell[i_v, i_b]
        shell_state.verts_pos_offset[i_new, i_b] = shell_state.verts_pos_offset[i_v, i_b]
        shell_state.verts_vel[i_new, i_b] = shell_state.verts_vel[i_v, i_b]
        shell_state.verts_origin[i_new, i_b] = shell_state.verts_origin[i_v, i_b]
        shell_state.verts_is_fixed[i_new, i_b] = shell_state.verts_is_fixed[i_v, i_b]
    return i_new


@qd.func
def func_vert_entity(i_v: int, i_b: int, shell_state: array_class.ShellState, shell_info: array_class.ShellInfo):
    """Entity of a filled pool slot, read from the face of the first corner of its original vertex."""
    i_o = shell_state.verts_origin[i_v, i_b]
    i_j = shell_info.verts_fan_start[i_o]
    i_c = shell_info.fans_corner[i_j]
    return shell_info.faces_entity[i_c // 3]


@qd.kernel
def kernel_shell_fracture(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Split the vertices whose surrounding stress exceeds the tensile strength of their material.

    The stress of a face adds the bending strain of its outer layer to the membrane one, F_bend = F @ (I + h / 2 * |S|)
    for the curvature S of its hinges. A vertex then evaluates every split of its fan along the mesh edges: for a
    vertex inside the sheet, two edges roughly opposite each other, and for a vertex on a boundary or a crack, one
    edge, the boundary acting as the other side. The best split scores the tractions its sides pull apart with, over
    the tensile strength times the thickness, and breaks above one. Splitting moves the corners of one side to a new
    vertex, which the hinges along the cut lose, so their far vertices become crack tips. Two vertices sharing a face
    never split in the same pass, the higher score winning, which keeps the splits independent. A vertex whose corners
    end up in several arcs (a crack reaching a boundary or another crack) is split into one vertex per arc, separating
    the pieces.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_e = shell_info.faces_entity[i_f]
        if shell_info.entities_is_fracturable[i_e]:
            thickness = shell_state.faces_thickness[i_f, i_b]
            F, _ = func_face_deformation(i_f, i_b, shell_state, shell_info)
            curvature = func_face_curvature(i_f, i_b, shell_state, shell_info)
            lambda_0, lambda_1, eigvec_0 = func_sym2_eigen(curvature)
            bending = func_sym2_compose(qd.abs(lambda_0), qd.abs(lambda_1), eigvec_0)
            bending_scale = 0.5 * shell_info.entities_bending_fracture_scale[i_e] * thickness
            F_bend = F @ (qd.Matrix.identity(gs.qd_float, 2) + bending_scale * bending)
            stress = func_membrane_stress(
                F_bend, shell_info.entities_stretching_modulus[i_e] * thickness, shell_info.entities_nu[i_e]
            )
            lambda_0, lambda_1, eigvec_0 = func_sym2_eigen(stress)
            shell_scratch.faces_fracture_stress[i_f, i_b] = func_sym2_compose(
                qd.max(lambda_0, 0.0), qd.max(lambda_1, 0.0), eigvec_0
            )

    for i_v, i_b in qd.ndrange(n_verts, B):
        separation = gs.qd_float(0.0)
        split = qd.Vector([-1, -1], dt=gs.qd_int)
        if shell_state.verts_origin[i_v, i_b] >= 0:
            i_e = func_vert_entity(i_v, i_b, shell_state, shell_info)
            tensile_strength = shell_info.entities_tensile_strength[i_e]
            if shell_info.entities_is_fracturable[i_e]:
                fan_start, fan_len, n_arcs, arc_start, arc_len = func_vert_arc(i_v, 0, i_b, shell_state, shell_info)
                is_open = n_arcs == 1
                if n_arcs <= 1 and arc_len >= 2:
                    # The stress a split relieves is at most twice the largest principal stress around the vertex.
                    traction_total = qd.Vector.zero(gs.qd_float, 3)
                    angle_total = gs.qd_float(0.0)
                    stress_max = gs.qd_float(0.0)
                    thickness_sum = gs.qd_float(0.0)
                    for p in range(arc_len):
                        i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                        traction, angle = func_corner_traction(i_c, i_b, shell_scratch, shell_info)
                        traction_total += traction
                        angle_total += angle
                        lambda_0, _, _ = func_sym2_eigen(shell_scratch.faces_fracture_stress[i_c // 3, i_b])
                        stress_max = qd.max(stress_max, lambda_0)
                        thickness_sum += shell_state.faces_thickness[i_c // 3, i_b]
                    toughness = tensile_strength * thickness_sum / arc_len
                    stress_bound = 2.0 * stress_max
                    if is_open:
                        stress_bound = 2.0 * stress_bound
                    if stress_bound >= toughness:
                        if is_open:
                            traction_0 = qd.Vector.zero(gs.qd_float, 3)
                            for p in range(arc_len - 1):
                                i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                                traction, _ = func_corner_traction(i_c, i_b, shell_scratch, shell_info)
                                traction_0 += traction
                                score = func_split_score(traction_0, traction_total - traction_0, True) / toughness
                                if score > separation:
                                    separation = score
                                    split = qd.Vector([p, -1], dt=gs.qd_int)
                        else:
                            for p in range(arc_len - 1):
                                traction_1 = qd.Vector.zero(gs.qd_float, 3)
                                angle_1 = gs.qd_float(0.0)
                                for q in range(p + 1, arc_len):
                                    i_c = shell_info.fans_corner[fan_start + (arc_start + q) % fan_len]
                                    traction, angle = func_corner_traction(i_c, i_b, shell_scratch, shell_info)
                                    traction_1 += traction
                                    angle_1 += angle
                                    # The two edges of the split run roughly opposite each other across the vertex.
                                    if 0.25 * angle_total <= angle_1 and angle_1 <= 0.75 * angle_total:
                                        score = func_split_score(traction_total - traction_1, traction_1, False)
                                        score = score / toughness
                                        if score > separation:
                                            separation = score
                                            split = qd.Vector([p, q], dt=gs.qd_int)
        shell_scratch.verts_separation[i_v, i_b] = separation
        shell_scratch.verts_split[i_v, i_b] = split

    # A vertex splits only if its score beats the score of every vertex it shares a face with. The verdict goes to
    # the split of the vertex, which no other vertex reads, keeping the comparison free of races.
    for i_v, i_b in qd.ndrange(n_verts, B):
        separation = shell_scratch.verts_separation[i_v, i_b]
        if separation > 1.0:
            fan_start, fan_len, _, arc_start, arc_len = func_vert_arc(i_v, 0, i_b, shell_state, shell_info)
            is_winner = True
            for p in range(arc_len):
                i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                for k in range(1, 3):
                    i_u = shell_state.corners_vert[3 * (i_c // 3) + (i_c + k) % 3, i_b]
                    separation_u = shell_scratch.verts_separation[i_u, i_b]
                    if separation_u > separation or (separation_u == separation and i_u < i_v):
                        is_winner = False
            if not is_winner:
                shell_scratch.verts_split[i_v, i_b] = qd.Vector([-1, -1], dt=gs.qd_int)

    for i_v, i_b in qd.ndrange(n_verts, B):
        split = shell_scratch.verts_split[i_v, i_b]
        if shell_scratch.verts_separation[i_v, i_b] > 1.0 and split[0] >= 0:
            fan_start, fan_len, n_arcs, arc_start, arc_len = func_vert_arc(i_v, 0, i_b, shell_state, shell_info)
            i_new = func_alloc_vert(
                i_v, func_vert_entity(i_v, i_b, shell_state, shell_info), i_b, shell_state, shell_info
            )
            if i_new >= 0:
                p_end = split[1]
                if n_arcs == 1:
                    p_end = arc_len - 1
                for p in range(split[0] + 1, p_end + 1):
                    i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                    shell_state.corners_vert[i_c, i_b] = i_new

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_state.verts_origin[i_v, i_b] >= 0:
            _, _, n_arcs, _, _ = func_vert_arc(i_v, 0, i_b, shell_state, shell_info)
            i_e = func_vert_entity(i_v, i_b, shell_state, shell_info)
            # The arcs past the first one move out one at a time, the next one becoming arc 1 in turn.
            for i_arc_ in range(n_arcs - 1):
                fan_start, fan_len, _, arc_start, arc_len = func_vert_arc(i_v, 1, i_b, shell_state, shell_info)
                i_new = func_alloc_vert(i_v, i_e, i_b, shell_state, shell_info)
                if i_new >= 0:
                    for p in range(arc_len):
                        i_c = shell_info.fans_corner[fan_start + (arc_start + p) % fan_len]
                        shell_state.corners_vert[i_c, i_b] = i_new


# ------------------------------------------------------------------------------------
# ------------------------------- rigid contact geometry -----------------------------
# ------------------------------------------------------------------------------------

# Every face of a sheet holds at most one contact per side of its mid-surface, at its point deepest into a rigid geom,
# so that a contact inside a face, along an edge or at a vertex is found alike, and a contact point shared by adjacent
# faces counts once per face, as a one-point quadrature of the contact over each face. The contact is compliant: a
# normal impulse proportional to the penetration of the surface of the sheet, half its thickness off its mid-surface,
# and a regularized Coulomb friction, both linearized over the substep and added to the implicit system of the sheets.
# The degrees of freedom of the movable rigid links the contacts touch join that system through their mass matrix, then
# are eliminated by their Schur complement, so that a contact moves the articulated rigid links as much as the sheet in
# the same substep, and the rigid link receives the exact opposite of the impulse the sheet receives.


@qd.func
def func_geom_distance(
    i_g: int,
    i_b: int,
    pos: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_config: qd.template(),
):
    """Return the signed distance from a point to the surface of a geom, in m, and the outward unit normal there.

    Boxes and capsules are exact, the other geoms reading the signed distance of the rigid collider, exact for spheres
    and planes, sampled from a grid for meshes.
    """
    geom_pos = dyn_state.geoms.pos[i_g, i_b]
    geom_quat = dyn_state.geoms.quat[i_g, i_b]
    geom_type = dyn_info.geoms.type[i_g]
    data = dyn_info.geoms.data[i_g]
    dist = gs.qd_float(0.0)
    normal = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    if geom_type == gs.GEOM_TYPE.BOX:
        pos_local = gu.qd_inv_transform_by_trans_quat(pos, geom_pos, geom_quat)
        signs = qd.select(pos_local >= 0.0, 1.0, -1.0)
        excess = qd.abs(pos_local) - 0.5 * qd.Vector([data[0], data[1], data[2]], dt=gs.qd_float)
        excess_out = qd.max(excess, 0.0)
        normal_local = qd.Vector.zero(gs.qd_float, 3)
        if excess_out.norm() > 0.0:
            dist = excess_out.norm()
            normal_local = signs * excess_out / dist
        else:
            # Inside, the closest face is the one of largest excess
            dist = excess.max()
            for k in qd.static(range(3)):
                if excess[k] >= dist:
                    normal_local = qd.Vector.zero(gs.qd_float, 3)
                    normal_local[k] = signs[k]
        normal = gu.qd_transform_by_quat(normal_local, geom_quat)
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        pos_local = gu.qd_inv_transform_by_trans_quat(pos, geom_pos, geom_quat)
        half_length = 0.5 * data[1]
        offset = pos_local - qd.Vector([0.0, 0.0, qd.math.clamp(pos_local[2], -half_length, half_length)])
        dist = offset.norm() - data[0]
        normal = gu.qd_transform_by_quat(offset / qd.max(offset.norm(), NORM_FLOOR), geom_quat)
    else:
        dist = sdf.sdf_func_world_local(i_g, pos, geom_pos, geom_quat, dyn_info.geoms, sdf_info)
        normal = gu.qd_normalize(
            sdf.sdf_func_grad_world_local(
                i_g, pos, geom_pos, geom_quat, dyn_info.geoms, rigid_info, sdf_info, collider_config
            ),
            NORM_FLOOR,
        )
    return dist, normal


@qd.func
def func_face_deepest_point(
    i_g: int,
    i_b: int,
    x0: qd.types.vector(3),
    x1: qd.types.vector(3),
    x2: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_config: qd.template(),
):
    """Return the barycentric coordinates of the point of a triangle deepest into a geom, the signed distance from that
    point to the geom, and the outward normal of the geom there.

    The signed distance of a convex geom is convex, so that its minimum over the triangle of vertices x0, x1 and x2 is
    found by projected gradient descent, exact for a sphere (the closest point to its center) and a plane (the deepest
    vertex, or a blend of the vertices lying level with it). The descent starts from a soft minimum of the vertices,
    which keeps the point of a face lying flat against a geom at its centroid rather than at an arbitrary vertex.
    """
    edge_len = qd.max(qd.max((x1 - x0).norm(), (x2 - x1).norm()), (x0 - x2).norm())
    width = qd.max(DEEPEST_POINT_SOFTMIN_WIDTH * edge_len, NORM_FLOOR)

    geom_type = dyn_info.geoms.type[i_g]
    n_vertices = 3
    n_descent_iterations = N_DEEPEST_POINT_ITERATIONS
    bary = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float) / 3.0
    if geom_type == gs.GEOM_TYPE.SPHERE:
        bary = gu.qd_closest_point_barycentric(dyn_state.geoms.pos[i_g, i_b], x0, x1, x2)
        n_vertices = 0
        n_descent_iterations = 0
    elif geom_type == gs.GEOM_TYPE.PLANE:
        n_descent_iterations = 0

    # Every iteration evaluates the distance at one point, first at the vertices, whose soft minimum it accumulates
    # relative to the smallest distance so far, then at the iterates of the descent, the last one being returned. All of
    # them share one call site, which keeps a single inlined instance of the distance query.
    dist_min = gs.qd_float(0.0)
    weight_sum = gs.qd_float(0.0)
    bary_sum = qd.Vector.zero(gs.qd_float, 3)
    step = 0.5 * edge_len
    dist = gs.qd_float(0.0)
    normal = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    for i_iter_ in range(n_vertices + n_descent_iterations + 1):
        bary_point = bary
        if i_iter_ < n_vertices:
            bary_point = func_select3(
                i_iter_,
                qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float),
                qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float),
                qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float),
            )
        pos = bary_point[0] * x0 + bary_point[1] * x1 + bary_point[2] * x2
        dist, normal = func_geom_distance(i_g, i_b, pos, dyn_state, dyn_info, rigid_info, sdf_info, collider_config)
        if i_iter_ < n_vertices:
            weight = gs.qd_float(1.0)
            if i_iter_ == 0:
                dist_min = dist
            elif dist < dist_min:
                scale = qd.exp((dist - dist_min) / width)
                weight_sum = scale * weight_sum
                bary_sum = scale * bary_sum
                dist_min = dist
            else:
                weight = qd.exp((dist_min - dist) / width)
            weight_sum += weight
            bary_sum += weight * bary_point
            bary = bary_sum / weight_sum
        elif i_iter_ < n_vertices + n_descent_iterations:
            bary = gu.qd_closest_point_barycentric(pos - step * normal, x0, x1, x2)
            step = 0.5 * step
    return bary, dist, normal


@qd.func
def func_contact_is_in_normal_cone(
    i_f: int,
    i_b: int,
    bary: qd.types.vector(3),
    normal: qd.types.vector(3),
    shell_state: array_class.ShellState,
    shell_info: array_class.ShellInfo,
):
    """Whether a geom pushing a face at a point of its edges pushes within the normal cone of the surface there.

    The point of a face closest to a geom lies on an edge when the geom lies beyond it, over the neighbor face. The
    push of the geom, along its outward normal, may then lean into the face by the dihedral angle of a convex edge at
    most, as on a ridge it touches both faces. A larger lean, or any lean across a flat or concave edge, puts the
    neighbor face between the geom and the face. A boundary edge, or the edge of a hinge fracture broke, bounds the
    surface, so that any push on it holds.
    """
    is_in_cone = True
    faces_hinge = shell_info.faces_hinge[i_f]
    for k in range(3):
        # Edge k joins corners k and k + 1 of the face, opposite corner k + 2
        i_h = faces_hinge[0]
        bary_opposite = bary[2]
        if k == 1:
            i_h = faces_hinge[1]
            bary_opposite = bary[0]
        elif k == 2:
            i_h = faces_hinge[2]
            bary_opposite = bary[1]
        if i_h >= 0 and bary_opposite < EDGE_BARY_TOLERANCE:
            i_va, i_vb, i_vc, i_vd, is_intact = func_hinge_verts(i_h, i_b, shell_state, shell_info)
            if is_intact:
                x_a = shell_state.verts_pos[i_va, i_b]
                edge = shell_state.verts_pos[i_vb, i_b] - x_a
                edge = edge / qd.max(edge.norm(), NORM_FLOOR)
                # The opposite vertices of the face and of its neighbor, as the hinge lists its faces in either order
                i_c = shell_info.hinges_opposite_corner[i_h][0]
                i_v_face = i_vc
                i_v_neighbor = i_vd
                if i_c // 3 != i_f:
                    i_v_face = i_vd
                    i_v_neighbor = i_vc
                dir_face = shell_state.verts_pos[i_v_face, i_b] - x_a
                dir_face = dir_face - dir_face.dot(edge) * edge
                dir_face = dir_face / qd.max(dir_face.norm(), NORM_FLOOR)
                dir_neighbor = shell_state.verts_pos[i_v_neighbor, i_b] - x_a
                dir_neighbor = dir_neighbor - dir_neighbor.dot(edge) * edge
                dir_neighbor = dir_neighbor / qd.max(dir_neighbor.norm(), NORM_FLOOR)
                # The normal of the face on the side of the geom, against its push
                normal_side = edge.cross(dir_face)
                if normal_side.dot(normal) > 0.0:
                    normal_side = -normal_side
                # Across the edge, in the basis of the direction into the face and of the normal on the side of the
                # geom, the push of the geom on the neighbor face bounds the cone of a convex edge
                neighbor_x = dir_neighbor.dot(dir_face)
                neighbor_y = dir_neighbor.dot(normal_side)
                bound_x = gs.qd_float(0.0)
                bound_y = gs.qd_float(-1.0)
                if neighbor_y < 0.0:
                    bound_x = -neighbor_y
                    bound_y = neighbor_x
                push_x = normal.dot(dir_face)
                push_y = normal.dot(normal_side)
                lean = bound_x * push_y - bound_y * push_x
                if lean > NORMAL_CONE_TOLERANCE * qd.sqrt(push_x * push_x + push_y * push_y):
                    is_in_cone = False
    return is_in_cone


# ------------------------------------------------------------------------------------
# ------------------------------ rigid contact chains --------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_contact_tree(
    i_c: int,
    i_b: int,
    shell_contact: array_class.ShellContactScratch,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    """The kinematic tree a contact moves, -1 for a contact with a static link."""
    i_l = func_contact_link(i_c, i_b, shell_contact, dyn_info)
    return rigid_info.links_tree_idx[i_l]


@qd.func
def func_contact_jacobian(
    i_c: int,
    i_b: int,
    shell_state: array_class.ShellState,
    shell_contact: array_class.ShellContactScratch,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    """Write the Jacobian of the point of a contact with a movable link over the degrees of freedom of its kinematic
    tree in contacts_jac, zero for the ones outside the chain of the link."""
    i_l = func_contact_link(i_c, i_b, shell_contact, dyn_info)
    i_t = rigid_info.links_tree_idx[i_l]
    i_d_start = rigid_info.trees_dof_start[i_t]
    for k in range(rigid_info.trees_n_dofs[i_t]):
        shell_contact.contacts_jac[i_c, k, i_b] = qd.Vector.zero(gs.qd_float, 3)
    offset = func_contact_point(i_c, i_b, shell_state, shell_contact) - dyn_state.links.root_COM[i_l, i_b]
    i_l_ = i_l
    for i_depth_ in range(dyn_info.links.parent_idx.shape[0]):
        if i_l_ >= 0:
            I_l = [i_l_, i_b] if qd.static(rigid_config.batch_links_info) else i_l_
            for i_d in range(dyn_info.links.dof_start[I_l], dyn_info.links.dof_end[I_l]):
                shell_contact.contacts_jac[i_c, i_d - i_d_start, i_b] = dyn_state.dofs.cdof_vel[
                    i_d, i_b
                ] + dyn_state.dofs.cdof_ang[i_d, i_b].cross(offset)
            i_l_ = dyn_info.links.parent_idx[I_l]


@qd.func
def func_contact_rigid_velocity(
    i_c: int,
    i_b: int,
    i_t: int,
    dofs_vec: qd.template(),
    shell_contact: array_class.ShellContactScratch,
    rigid_info: array_class.RigidInfo,
):
    """Velocity of the rigid point of a contact for given velocities of the degrees of freedom of its tree."""
    i_d_start = rigid_info.trees_dof_start[i_t]
    vel = qd.Vector.zero(gs.qd_float, 3)
    for k in range(rigid_info.trees_n_dofs[i_t]):
        vel += shell_contact.contacts_jac[i_c, k, i_b] * dofs_vec[i_d_start + k, i_b]
    return vel


@qd.func
def func_contact_rigid_response(
    i_c: int,
    i_b: int,
    i_t: int,
    i_d_offset: int,
    dofs_vec: qd.template(),
    shell_contact: array_class.ShellContactScratch,
    rigid_info: array_class.RigidInfo,
):
    """Velocity of the rigid point of a contact for the rigid velocities given generalized forces impose on its tree.

    The forces f are read from dofs_vec from index i_d_offset on, and the rigid velocities are S^-1 f, S being the
    Schur matrix of the tree (see func_contact_assemble).
    """
    i_d_start = rigid_info.trees_dof_start[i_t]
    n_tree_dofs = rigid_info.trees_n_dofs[i_t]
    vel = qd.Vector.zero(gs.qd_float, 3)
    for k in range(n_tree_dofs):
        response = gs.qd_float(0.0)
        for l in range(n_tree_dofs):
            response += (
                shell_contact.dofs_schur_inv[i_d_start + k, i_d_start + l, i_b]
                * dofs_vec[i_d_offset + i_d_start + l, i_b]
            )
        vel += shell_contact.contacts_jac[i_c, k, i_b] * response
    return vel


@qd.func
def func_contact_link(
    i_c: int, i_b: int, shell_contact: array_class.ShellContactScratch, dyn_info: array_class.DynInfo
):
    """The rigid link a contact touches."""
    i_g = shell_contact.contacts_geom[i_c, i_b]
    return dyn_info.geoms.link_idx[i_g]


@qd.func
def func_contact_point(
    i_c: int,
    i_b: int,
    shell_state: array_class.ShellState,
    shell_contact: array_class.ShellContactScratch,
):
    """World position of the point of the mid-surface of a face a contact lies at, at the start of the substep."""
    i_f = i_c // 2
    bary = shell_contact.contacts_bary[i_c, i_b]
    pos = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        i_v = shell_state.corners_vert[3 * i_f + k, i_b]
        pos += bary[k] * shell_state.verts_pos[i_v, i_b]
    return pos


@qd.func
def func_contact_shell_velocity(
    i_c: int,
    i_b: int,
    vel: qd.template(),
    shell_state: array_class.ShellState,
    shell_contact: array_class.ShellContactScratch,
):
    """Velocity of the face point of a contact, interpolated from given velocities of the vertices of its face."""
    i_f = i_c // 2
    bary = shell_contact.contacts_bary[i_c, i_b]
    vel_point = qd.Vector.zero(gs.qd_float, 3)
    for k in qd.static(range(3)):
        i_v = shell_state.corners_vert[3 * i_f + k, i_b]
        vel_point += bary[k] * vel[i_v, i_b]
    return vel_point


@qd.func
def func_contact_add_to_verts(
    i_c: int,
    i_b: int,
    impulse: qd.types.vector(3),
    dst: qd.template(),
    shell_state: array_class.ShellState,
    shell_contact: array_class.ShellContactScratch,
):
    """Spread a vector applied at the face point of a contact on the free vertices of its face, by its barycentric
    coordinates."""
    i_f = i_c // 2
    bary = shell_contact.contacts_bary[i_c, i_b]
    for k in qd.static(range(3)):
        i_v = shell_state.corners_vert[3 * i_f + k, i_b]
        if func_is_vert_free(i_v, i_b, shell_state):
            dst[i_v, i_b] += bary[k] * impulse


@qd.func
def func_contact_stick_velocity(stiffness: float, friction_bound: float):
    """Slip velocity below which a contact sticks, at least FRICTION_STICK_VELOCITY.

    It widens to 2 * friction_bound / stiffness, so that the stuck friction is no stiffer than the normal penalty,
    which keeps the linear system of the contacts as well conditioned as their normal stiffness alone, and the creep of
    a held contact proportional to its penetration.
    """
    return qd.max(FRICTION_STICK_VELOCITY, 2.0 * friction_bound / qd.max(stiffness, NORM_FLOOR))


@qd.func
def func_contact_potential(
    dt: float,
    vel: qd.types.vector(3),
    gap: float,
    normal: qd.types.vector(3),
    stiffness: float,
    friction_bound: float,
):
    """Contact potential of a relative velocity over the substep, in J, and the impulse it applies to the sheet.

    The normal potential is the penalty stiffness / (2 * dt^2) * max(0, -g)^2 of the penetration -g at the end of the
    substep, g = gap + dt * n.u. The friction potential is friction_bound * F0(|u_t|), friction_bound being mu times a
    normal impulse (see kernel_shell_rigid_contact_detect), and F0 the smoothed Coulomb potential whose slope
    f1(y) = 2 y / eps - y^2 / eps^2 grows from zero to one as the slip velocity y reaches the stick velocity eps (see
    func_contact_stick_velocity), so that the friction impulse never exceeds friction_bound and sticks below eps.
    """
    vel_normal = vel.dot(normal)
    vel_tangent = vel - vel_normal * normal
    slip = vel_tangent.norm()
    penetration = qd.max(-(gap + dt * vel_normal), 0.0)
    potential = 0.5 * stiffness / (dt * dt) * penetration * penetration
    impulse = (stiffness / dt) * penetration * normal
    eps = func_contact_stick_velocity(stiffness, friction_bound)
    if slip < eps:
        potential += friction_bound * slip * slip * (1.0 / eps - slip / (3.0 * eps * eps))
        impulse -= friction_bound * (2.0 / eps - slip / (eps * eps)) * vel_tangent
    else:
        potential += friction_bound * (slip - eps / 3.0)
        impulse -= friction_bound / slip * vel_tangent
    return potential, impulse


@qd.func
def func_contact_hessian(
    dt: float,
    vel: qd.types.vector(3),
    gap: float,
    normal: qd.types.vector(3),
    stiffness: float,
    friction_bound: float,
):
    """Hessian of the contact potential of func_contact_potential with respect to the relative velocity, in kg.

    The penalty adds stiffness * n n^T while the surface penetrates. The friction adds friction_bound * f1(y) / y
    across the slip and friction_bound * f1'(y) along it, both non-negative, so that the Hessian is positive
    semi-definite, the sliding direction of a contact past the stick velocity carrying no stiffness.
    """
    vel_normal = vel.dot(normal)
    vel_tangent = vel - vel_normal * normal
    slip = vel_tangent.norm()
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if gap + dt * vel_normal < 0.0:
        hessian += stiffness * normal.outer_product(normal)
    eps = func_contact_stick_velocity(stiffness, friction_bound)
    projector = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
    direction = vel_tangent / qd.max(slip, NORM_FLOOR)
    if slip < eps:
        hessian += friction_bound * (
            (2.0 / eps - slip / (eps * eps)) * projector - slip / (eps * eps) * direction.outer_product(direction)
        )
    else:
        hessian += friction_bound / slip * (projector - direction.outer_product(direction))
    return hessian


@qd.func
def func_contact_vel(i_c: int, i_b: int, shell_contact: array_class.ShellContactScratch):
    """Relative velocity of a contact at the current iterate: its value at the start of the substep plus its change."""
    return shell_contact.contacts_vel[i_c, i_b] + shell_contact.contacts_vel_change[i_c, i_b]


# ------------------------------------------------------------------------------------
# ------------------------------ rigid contact solve ---------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_shell_rigid_contact_detect(
    dt: float,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    dyn_state: array_class.DynState,
    shell_info: array_class.ShellInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_config: qd.template(),
    contact_stiffness: float,
):
    """Detect the contacts of every face with the rigid geoms at the start of the substep, and start the contact solve
    from the warm start of the sheet and the free motion of the rigid links.

    A face keeps, on either side of its mid-surface, the geom its surface penetrates deepest (see
    func_face_deepest_point), at the signed distance gap from it, unless the geom has not reached that point yet and it
    lies on an edge the geom pushes past the normal cone of the surface, leaving the contact to the neighbor face (see
    func_contact_is_in_normal_cone). The contact is linearized about that point: its normal is the outward normal of the
    geom there, and its relative velocity the velocity of the face point minus the velocity of the rigid point at the
    same position, starting from the end velocity the rigid solver found before the contacts of the sheets. Its penalty
    stiffness is contact_stiffness times the mass of the contact point over dt^2, from the diagonal blocks of the system
    of the sheet and the inverse weight of the rigid link, so that it scales with the local stiffness and mass of either
    side. Its friction bound lags: a contact slot that touched the same geom over the previous substep keeps mu times
    the normal impulse it received then, as the Coulomb friction of a steady contact, so that a contact starts to rub
    one substep after it starts to push. A bound taken from an iterate of the solve instead would overestimate friction,
    an early iterate pushing far more than the solution. The system of the sheet without contacts is kept for every
    iteration of the solve.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]
    n_geoms = dyn_info.geoms.type.shape[0]
    n_dofs = shell_contact.dofs_schur_vec.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        shell_contact.verts_rhs_base[i_v, i_b] = shell_scratch.verts_rhs[i_v, i_b]
        shell_contact.verts_diag_base[i_v, i_b] = shell_scratch.verts_prec[i_v, i_b]
        shell_contact.verts_dv_prev[i_v, i_b] = shell_state.verts_dv[i_v, i_b]

    # The rigid velocities the substep ends at before the contacts, from which the solve starts
    for i_d, i_b in qd.ndrange(n_dofs, B):
        shell_contact.dofs_dv[i_d, i_b] = gs.qd_float(0.0)
        shell_contact.dofs_schur_vec[i_d, i_b] = gs.qd_float(0.0)
        if i_d < dyn_state.dofs.vel.shape[0]:
            shell_contact.dofs_schur_vec[i_d, i_b] = dyn_state.dofs.vel[i_d, i_b] + dt * dyn_state.dofs.acc[i_d, i_b]

    for i_b in range(B):
        shell_contact.envs_step[i_b] = 1.0
        shell_contact.envs_n_newton_iterations[i_b] = 0

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_v0 = shell_state.corners_vert[3 * i_f, i_b]
        i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
        i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
        x0 = shell_state.verts_pos[i_v0, i_b]
        x1 = shell_state.verts_pos[i_v1, i_b]
        x2 = shell_state.verts_pos[i_v2, i_b]
        normal_face = (x1 - x0).cross(x2 - x0)
        center = (x0 + x1 + x2) / 3.0
        radius = qd.max(qd.max((x0 - center).norm(), (x1 - center).norm()), (x2 - center).norm())
        half_thickness = 0.5 * shell_state.faces_thickness[i_f, i_b]
        speed = qd.max(
            qd.max(shell_state.verts_vel[i_v0, i_b].norm(), shell_state.verts_vel[i_v1, i_b].norm()),
            shell_state.verts_vel[i_v2, i_b].norm(),
        )
        geom_front = -1
        geom_back = -1
        gap_front = gs.qd_float(0.0)
        gap_back = gs.qd_float(0.0)
        bary_front = qd.Vector.zero(gs.qd_float, 3)
        bary_back = qd.Vector.zero(gs.qd_float, 3)
        normal_front = qd.Vector.zero(gs.qd_float, 3)
        normal_back = qd.Vector.zero(gs.qd_float, 3)
        for i_g in range(n_geoms):
            if dyn_info.geoms.needs_coup[i_g]:
                bound_radius = shell_contact.geoms_bound_radius[i_g]
                bound_center = gu.qd_transform_by_trans_quat(
                    shell_contact.geoms_bound_center[i_g], dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]
                )
                # Contacts are speculative within the distance the face and the geom can close over the substep, so
                # that the end gap of the solve stops a fast approach before the surfaces pass through each other
                i_l = dyn_info.geoms.link_idx[i_g]
                margin = (
                    CONTACT_MARGIN_RATIO
                    * dt
                    * (
                        speed
                        + func_vel_at_point(i_l, i_b, bound_center, dyn_state.links).norm()
                        + dyn_state.links.cd_ang[i_l, i_b].norm() * qd.max(bound_radius, 0.0)
                    )
                )
                if (
                    bound_radius < 0.0
                    or (center - bound_center).norm() <= radius + bound_radius + half_thickness + margin
                ):
                    bary, dist, normal = func_face_deepest_point(
                        i_g, i_b, x0, x1, x2, dyn_state, dyn_info, rigid_info, sdf_info, collider_config
                    )
                    gap = dist - half_thickness
                    # A contact the geom has not reached yet leaves an edge it pushes past the normal cone of the
                    # surface to the neighbor face: linearized about the edge, its gap would close on a corner the
                    # neighbor face covers, which brakes a geom sliding across the edge
                    if gap < margin and (
                        gap <= 0.0 or func_contact_is_in_normal_cone(i_f, i_b, bary, normal, shell_state, shell_info)
                    ):
                        if normal.dot(normal_face) >= 0.0:
                            if geom_front < 0 or gap < gap_front:
                                geom_front = i_g
                                gap_front = gap
                                bary_front = bary
                                normal_front = normal
                        elif geom_back < 0 or gap < gap_back:
                            geom_back = i_g
                            gap_back = gap
                            bary_back = bary
                            normal_back = normal
        shell_contact.contacts_geom[2 * i_f, i_b] = geom_front
        shell_contact.contacts_gap[2 * i_f, i_b] = gap_front
        shell_contact.contacts_bary[2 * i_f, i_b] = bary_front
        shell_contact.contacts_normal[2 * i_f, i_b] = normal_front
        shell_contact.contacts_geom[2 * i_f + 1, i_b] = geom_back
        shell_contact.contacts_gap[2 * i_f + 1, i_b] = gap_back
        shell_contact.contacts_bary[2 * i_f + 1, i_b] = bary_back
        shell_contact.contacts_normal[2 * i_f + 1, i_b] = normal_back

    # Stiffness and start velocity of every contact, reading the diagonal blocks of the sheet without contacts
    for i_c, i_b in qd.ndrange(2 * n_faces, B):
        i_g = shell_contact.contacts_geom[i_c, i_b]
        if i_g >= 0:
            i_f = i_c // 2
            normal = shell_contact.contacts_normal[i_c, i_b]
            bary = shell_contact.contacts_bary[i_c, i_b]
            compliance = gs.qd_float(0.0)
            for k in qd.static(range(3)):
                i_v = shell_state.corners_vert[3 * i_f + k, i_b]
                if func_is_vert_free(i_v, i_b, shell_state) and shell_scratch.verts_mass[i_v, i_b] > 0.0:
                    compliance += bary[k] ** 2 * normal.dot(shell_scratch.verts_prec[i_v, i_b].inverse() @ normal)
            i_l = dyn_info.geoms.link_idx[i_g]
            I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
            compliance += dyn_info.links.invweight[I_l][0]
            if compliance > 0.0:
                shell_contact.contacts_stiffness[i_c, i_b] = contact_stiffness / compliance
                vel = func_contact_shell_velocity(i_c, i_b, shell_state.verts_vel, shell_state, shell_contact)
                # The rigid point moves with the kinematic tree of its link, whose Jacobian holds over the substep
                i_t = rigid_info.links_tree_idx[i_l]
                if i_t >= 0:
                    func_contact_jacobian(
                        i_c, i_b, shell_state, shell_contact, dyn_state, dyn_info, rigid_info, rigid_config
                    )
                    vel = vel - func_contact_rigid_velocity(
                        i_c, i_b, i_t, shell_contact.dofs_schur_vec, shell_contact, rigid_info
                    )
                shell_contact.contacts_vel[i_c, i_b] = vel
                shell_contact.contacts_vel_change[i_c, i_b] = func_contact_shell_velocity(
                    i_c, i_b, shell_state.verts_dv, shell_state, shell_contact
                )
                friction_bound = gs.qd_float(0.0)
                if shell_state.contacts_geom_prev[i_c, i_b] == i_g:
                    friction_bound = shell_state.contacts_friction_bound_prev[i_c, i_b]
                shell_contact.contacts_friction_bound[i_c, i_b] = friction_bound
            else:
                shell_contact.contacts_geom[i_c, i_b] = -1


@qd.func
def func_contact_linearize(
    dt: float,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    contact_stiffness: float,
    velocity_tolerance: float,
    max_iterations: int,
):
    """Evaluate the impulse and the Hessian of every contact at the current iterate of the contact solve, and decide
    which environments iterate again.

    The friction bound of a contact stays fixed over the iterations, so that they minimize one convex potential (see
    kernel_shell_rigid_contact_detect). An environment stops iterating once its last step was full and the impulse of
    every contact matches the linear model of the previous iteration within the impulse that would change the velocity
    of the contact point by velocity_tolerance, the contact potential then being quadratic along the step. It also stops
    once it ran max_iterations iterations, which its solve status reports.
    """
    B = shell_contact.envs_step.shape[0]
    n_slots = shell_contact.contacts_geom.shape[0]

    for i_b in range(B):
        shell_contact.envs_is_nonlinear[i_b] = shell_contact.envs_n_newton_iterations[i_b] == 0

    for i_c, i_b in qd.ndrange(n_slots, B):
        i_g = shell_contact.contacts_geom[i_c, i_b]
        if shell_scratch.envs_needs_solve[i_b] and i_g >= 0:
            vel = func_contact_vel(i_c, i_b, shell_contact)
            gap = shell_contact.contacts_gap[i_c, i_b]
            normal = shell_contact.contacts_normal[i_c, i_b]
            stiffness = shell_contact.contacts_stiffness[i_c, i_b]
            friction_bound = shell_contact.contacts_friction_bound[i_c, i_b]
            impulse = func_contact_potential(dt, vel, gap, normal, stiffness, friction_bound)[1]
            # Mismatch of the impulse against the linear model of the previous iterate, along the step it took
            model = shell_contact.contacts_impulse[i_c, i_b] - shell_contact.envs_step[i_b] * (
                shell_contact.contacts_hessian[i_c, i_b] @ shell_contact.contacts_dir[i_c, i_b]
            )
            # The mass of the contact point, from its stiffness (see kernel_shell_rigid_contact_detect)
            impulse_tolerance = velocity_tolerance * stiffness / contact_stiffness
            if (impulse - model).norm() > impulse_tolerance:
                shell_contact.envs_is_nonlinear[i_b] = True
            shell_contact.contacts_impulse[i_c, i_b] = impulse
            shell_contact.contacts_hessian[i_c, i_b] = func_contact_hessian(
                dt, vel, gap, normal, stiffness, friction_bound
            )

    for i_b in range(B):
        if shell_scratch.envs_needs_solve[i_b]:
            is_converged = shell_contact.envs_step[i_b] >= 1.0 and not shell_contact.envs_is_nonlinear[i_b]
            n_iterations = shell_contact.envs_n_newton_iterations[i_b]
            if not is_converged and n_iterations >= max_iterations:
                shell_scratch.envs_solve_status[i_b] |= SHELL_SOLVE_STATUS.MAX_ITERATIONS
            if is_converged or n_iterations >= max_iterations:
                shell_scratch.envs_needs_solve[i_b] = False
            else:
                shell_contact.envs_n_newton_iterations[i_b] = n_iterations + 1


@qd.func
def func_contact_assemble(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    """Add the contacts, linearized about the current iterate, to the linear system of the sheet in every environment
    still solving, then eliminate the movable rigid degrees of freedom.

    About the iterate of contact impulse p and Hessian H, the impulse of a contact is p_lin - H (J_s dv_s - J_r dv_r),
    p_lin = p + H u_change for the relative velocity change u_change of the iterate. The rigid system of every
    kinematic tree the contacts touch is its mass matrix plus the contact stiffness, S = M_r + sum J_r^T H J_r, whose
    inverse corrects the right-hand side of the sheet by C S^-1 b_r, C = sum J_s^T H J_r, and whose Schur complement
    func_contact_schur_product applies. Every contact reads the Jacobian of its rigid point over the degrees of freedom
    of its tree from the detection (see func_contact_jacobian), so that the products of the solve run in parallel over
    the contacts.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_slots = shell_contact.contacts_geom.shape[0]
    n_trees = shell_contact.trees_is_coupled.shape[0]
    n_dofs = shell_contact.dofs_schur_rhs.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_needs_solve[i_b]:
            shell_scratch.verts_rhs[i_v, i_b] = shell_contact.verts_rhs_base[i_v, i_b]
            shell_scratch.verts_prec[i_v, i_b] = shell_contact.verts_diag_base[i_v, i_b]

    # The mass matrix couples nothing across its blocks, whose entries it leaves unwritten
    for i_t, i_b in qd.ndrange(n_trees, B):
        if shell_scratch.envs_needs_solve[i_b]:
            shell_contact.trees_is_coupled[i_t, i_b] = False
            i_d_start = rigid_info.trees_dof_start[i_t]
            i_d_end = i_d_start + rigid_info.trees_n_dofs[i_t]
            for i_d in range(i_d_start, i_d_end):
                shell_contact.dofs_schur_rhs[i_d, i_b] = gs.qd_float(0.0)
                shell_contact.dofs_schur_product[i_d, i_b] = gs.qd_float(0.0)
                shell_contact.dofs_schur_product[n_dofs + i_d, i_b] = gs.qd_float(0.0)
                for j_d in range(i_d_start, i_d_end):
                    mass = gs.qd_float(0.0)
                    if rigid_info.dofs_mass_block_start[j_d] == rigid_info.dofs_mass_block_start[i_d]:
                        mass = rigid_info.mass_mat[i_d, j_d, i_b]
                    shell_contact.dofs_schur_inv[i_d, j_d, i_b] = mass

    for i_c, i_b in qd.ndrange(n_slots, B):
        if shell_scratch.envs_needs_solve[i_b] and shell_contact.contacts_geom[i_c, i_b] >= 0:
            i_f = i_c // 2
            bary = shell_contact.contacts_bary[i_c, i_b]
            hessian = shell_contact.contacts_hessian[i_c, i_b]
            impulse = shell_contact.contacts_impulse[i_c, i_b] + hessian @ shell_contact.contacts_vel_change[i_c, i_b]
            func_contact_add_to_verts(i_c, i_b, impulse, shell_scratch.verts_rhs, shell_state, shell_contact)
            for k in qd.static(range(3)):
                i_v = shell_state.corners_vert[3 * i_f + k, i_b]
                if func_is_vert_free(i_v, i_b, shell_state):
                    shell_scratch.verts_prec[i_v, i_b] += bary[k] ** 2 * hessian
            i_t = func_contact_tree(i_c, i_b, shell_contact, dyn_info, rigid_info)
            if i_t >= 0:
                shell_contact.trees_is_coupled[i_t, i_b] = True
                i_d_start = rigid_info.trees_dof_start[i_t]
                n_tree_dofs = rigid_info.trees_n_dofs[i_t]
                for k in range(n_tree_dofs):
                    jac_k = shell_contact.contacts_jac[i_c, k, i_b]
                    # The rigid link receives the opposite of the impulse of the sheet
                    shell_contact.dofs_schur_rhs[i_d_start + k, i_b] -= jac_k.dot(impulse)
                    row = hessian @ jac_k
                    for l in range(n_tree_dofs):
                        shell_contact.dofs_schur_inv[i_d_start + k, i_d_start + l, i_b] += row.dot(
                            shell_contact.contacts_jac[i_c, l, i_b]
                        )

    # Inverse of the Schur matrix of every coupled tree, in place by Gauss-Jordan elimination, which needs no pivoting
    # on a positive definite matrix
    for i_t, i_b in qd.ndrange(n_trees, B):
        if shell_scratch.envs_needs_solve[i_b] and shell_contact.trees_is_coupled[i_t, i_b]:
            i_d_start = rigid_info.trees_dof_start[i_t]
            i_d_end = i_d_start + rigid_info.trees_n_dofs[i_t]
            for k_d in range(i_d_start, i_d_end):
                pivot = 1.0 / shell_contact.dofs_schur_inv[k_d, k_d, i_b]
                shell_contact.dofs_schur_inv[k_d, k_d, i_b] = gs.qd_float(1.0)
                for j_d in range(i_d_start, i_d_end):
                    shell_contact.dofs_schur_inv[k_d, j_d, i_b] *= pivot
                for i_d in range(i_d_start, i_d_end):
                    if i_d != k_d:
                        factor = shell_contact.dofs_schur_inv[i_d, k_d, i_b]
                        shell_contact.dofs_schur_inv[i_d, k_d, i_b] = gs.qd_float(0.0)
                        for j_d in range(i_d_start, i_d_end):
                            shell_contact.dofs_schur_inv[i_d, j_d, i_b] -= (
                                factor * shell_contact.dofs_schur_inv[k_d, j_d, i_b]
                            )

    # Right-hand side of the sheet corrected by the coupling: + C S^-1 b_r
    for i_c, i_b in qd.ndrange(n_slots, B):
        if shell_scratch.envs_needs_solve[i_b] and shell_contact.contacts_geom[i_c, i_b] >= 0:
            i_t = func_contact_tree(i_c, i_b, shell_contact, dyn_info, rigid_info)
            if i_t >= 0:
                vel_rigid = func_contact_rigid_response(
                    i_c, i_b, i_t, 0, shell_contact.dofs_schur_rhs, shell_contact, rigid_info
                )
                func_contact_add_to_verts(
                    i_c,
                    i_b,
                    shell_contact.contacts_hessian[i_c, i_b] @ vel_rigid,
                    shell_scratch.verts_rhs,
                    shell_state,
                    shell_contact,
                )


@qd.func
def func_contact_product(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    """Add the contacts to the system product of every environment still solving, their rigid part excepted.

    The contact stiffness of the sheet J_s^T H J_s x goes to verts_Ap and its share x^T J_s^T H J_s x to envs_pAp,
    while the generalized forces C^T x of the rigid contacts gather in the Schur accumulator of the iteration (see
    func_contact_schur_product).
    """
    B = shell_state.verts_pos.shape[1]
    n_slots = shell_contact.contacts_geom.shape[0]
    n_dofs = shell_contact.dofs_schur_rhs.shape[0]

    for i_c, i_b in qd.ndrange(n_slots, B):
        if shell_scratch.envs_is_solving[i_b] and shell_contact.contacts_geom[i_c, i_b] >= 0:
            i_f = i_c // 2
            bary = shell_contact.contacts_bary[i_c, i_b]
            vel = qd.Vector.zero(gs.qd_float, 3)
            for k in qd.static(range(3)):
                i_v = shell_state.corners_vert[3 * i_f + k, i_b]
                if func_is_vert_free(i_v, i_b, shell_state):
                    vel += bary[k] * func_pcg_direction(i_v, i_b, shell_state, shell_scratch)
            impulse = shell_contact.contacts_hessian[i_c, i_b] @ vel
            func_contact_add_to_verts(i_c, i_b, impulse, shell_scratch.verts_Ap, shell_state, shell_contact)
            shell_scratch.envs_pAp[i_b] += vel.dot(impulse)
            i_t = func_contact_tree(i_c, i_b, shell_contact, dyn_info, rigid_info)
            if i_t >= 0:
                i_d_offset = (shell_scratch.envs_n_solve_iterations[i_b] % 2) * n_dofs
                i_d_start = rigid_info.trees_dof_start[i_t]
                for k in range(rigid_info.trees_n_dofs[i_t]):
                    shell_contact.dofs_schur_product[i_d_offset + i_d_start + k, i_b] += shell_contact.contacts_jac[
                        i_c, k, i_b
                    ].dot(impulse)


@qd.func
def func_contact_schur_product(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    """Subtract the Schur complement of the rigid degrees of freedom, C S^-1 C^T x, from the system product of every
    environment still solving, and its share x^T C S^-1 C^T x from envs_pAp.

    The generalized forces C^T x of an iteration accumulate in one of the two halves of dofs_schur_product, alternating
    with the parity of the iteration, so that every tree clears the half of the previous iteration while its contacts
    read the current one, with no pass of its own.
    """
    B = shell_state.verts_pos.shape[1]
    n_slots = shell_contact.contacts_geom.shape[0]
    n_trees = shell_contact.trees_is_coupled.shape[0]
    n_dofs = shell_contact.dofs_schur_rhs.shape[0]

    for i_c, i_b in qd.ndrange(n_slots + n_trees, B):
        if shell_scratch.envs_is_solving[i_b]:
            i_d_offset = (shell_scratch.envs_n_solve_iterations[i_b] % 2) * n_dofs
            if i_c < n_slots:
                if shell_contact.contacts_geom[i_c, i_b] >= 0:
                    i_t = func_contact_tree(i_c, i_b, shell_contact, dyn_info, rigid_info)
                    if i_t >= 0:
                        vel_rigid = func_contact_rigid_response(
                            i_c, i_b, i_t, i_d_offset, shell_contact.dofs_schur_product, shell_contact, rigid_info
                        )
                        func_contact_add_to_verts(
                            i_c,
                            i_b,
                            -(shell_contact.contacts_hessian[i_c, i_b] @ vel_rigid),
                            shell_scratch.verts_Ap,
                            shell_state,
                            shell_contact,
                        )
            else:
                i_t = i_c - n_slots
                if shell_contact.trees_is_coupled[i_t, i_b]:
                    i_d_start = rigid_info.trees_dof_start[i_t]
                    i_d_end = i_d_start + rigid_info.trees_n_dofs[i_t]
                    energy = gs.qd_float(0.0)
                    for i_d in range(i_d_start, i_d_end):
                        force = shell_contact.dofs_schur_product[i_d_offset + i_d, i_b]
                        for j_d in range(i_d_start, i_d_end):
                            energy += (
                                force
                                * shell_contact.dofs_schur_inv[i_d, j_d, i_b]
                                * shell_contact.dofs_schur_product[i_d_offset + j_d, i_b]
                            )
                    shell_scratch.envs_pAp[i_b] -= energy
                    for i_d in range(i_d_start, i_d_end):
                        shell_contact.dofs_schur_product[n_dofs - i_d_offset + i_d, i_b] = gs.qd_float(0.0)


@qd.func
def func_contact_direction(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    shell_info: array_class.ShellInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
):
    """Turn the solution of the linearized contacts into the step of the contact solve from its current iterate.

    The rigid velocity change of the solution follows from the Schur system, dv_r = S^-1 (b_r + C^T dv_s). The step is
    the difference between the solution and the iterate, for the sheet, the rigid degrees of freedom and the relative
    velocity of every contact, and the elastic system product of the step of the sheet, verts_Ap = (M + K) d_s, follows
    for the line search.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_slots = shell_contact.contacts_geom.shape[0]
    n_trees = shell_contact.trees_is_coupled.shape[0]

    for i_t, i_b in qd.ndrange(n_trees, B):
        if shell_scratch.envs_needs_solve[i_b]:
            i_d_start = rigid_info.trees_dof_start[i_t]
            for i_d in range(i_d_start, i_d_start + rigid_info.trees_n_dofs[i_t]):
                shell_contact.dofs_schur_vec[i_d, i_b] = shell_contact.dofs_schur_rhs[i_d, i_b]
                shell_contact.dofs_dir[i_d, i_b] = gs.qd_float(0.0)

    for i_c, i_b in qd.ndrange(n_slots, B):
        if shell_scratch.envs_needs_solve[i_b] and shell_contact.contacts_geom[i_c, i_b] >= 0:
            i_t = func_contact_tree(i_c, i_b, shell_contact, dyn_info, rigid_info)
            if i_t >= 0:
                impulse = shell_contact.contacts_hessian[i_c, i_b] @ func_contact_shell_velocity(
                    i_c, i_b, shell_state.verts_dv, shell_state, shell_contact
                )
                i_d_start = rigid_info.trees_dof_start[i_t]
                for k in range(rigid_info.trees_n_dofs[i_t]):
                    shell_contact.dofs_schur_vec[i_d_start + k, i_b] += shell_contact.contacts_jac[i_c, k, i_b].dot(
                        impulse
                    )

    for i_t, i_b in qd.ndrange(n_trees, B):
        if shell_scratch.envs_needs_solve[i_b] and shell_contact.trees_is_coupled[i_t, i_b]:
            i_d_start = rigid_info.trees_dof_start[i_t]
            i_d_end = i_d_start + rigid_info.trees_n_dofs[i_t]
            for i_d in range(i_d_start, i_d_end):
                dv = gs.qd_float(0.0)
                for j_d in range(i_d_start, i_d_end):
                    dv += shell_contact.dofs_schur_inv[i_d, j_d, i_b] * shell_contact.dofs_schur_vec[j_d, i_b]
                shell_contact.dofs_dir[i_d, i_b] = dv - shell_contact.dofs_dv[i_d, i_b]

    # The product of the step of the sheet applies to verts_p, read by func_system_product in SOLVE mode
    for i_b in range(B):
        shell_scratch.envs_is_solving[i_b] = shell_scratch.envs_needs_solve[i_b]
        shell_scratch.envs_pcg_mode[i_b] = PCG_MODE.SOLVE

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_needs_solve[i_b]:
            step = shell_state.verts_dv[i_v, i_b] - shell_contact.verts_dv_prev[i_v, i_b]
            shell_scratch.verts_p[i_v, i_b] = step
            shell_scratch.verts_Ap[i_v, i_b] = shell_scratch.verts_mass[i_v, i_b] * step

    for i_c, i_b in qd.ndrange(n_slots, B):
        if shell_scratch.envs_needs_solve[i_b] and shell_contact.contacts_geom[i_c, i_b] >= 0:
            direction = func_contact_shell_velocity(i_c, i_b, shell_scratch.verts_p, shell_state, shell_contact)
            i_t = func_contact_tree(i_c, i_b, shell_contact, dyn_info, rigid_info)
            if i_t >= 0:
                direction = direction - func_contact_rigid_velocity(
                    i_c, i_b, i_t, shell_contact.dofs_dir, shell_contact, rigid_info
                )
            shell_contact.contacts_dir[i_c, i_b] = direction

    func_system_product(shell_state, shell_scratch, shell_info)


@qd.func
def func_contact_line_search_start(
    line_flag: qd.types.ndarray(qd.i32, ndim=0),
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    rigid_info: array_class.RigidInfo,
):
    """Accumulate the quadratic part of the incremental potential along the step of the contact solve, and open the
    bracket [0, 1] of the minimum along the step in every environment still solving.

    The potential is the quadratic of the sheet, 1/2 dv^T (M + K) dv - b^T dv, plus the kinetic energy the contacts give
    the rigid degrees of freedom, 1/2 dv_r^T M_r dv_r, plus the contact potentials (see func_contact_potential), which
    the solution of the linearized contacts minimizes jointly. It is convex along the step, its slope increasing with
    the step length. The quadratic part is known in closed form from the elastic product of the step, and the line
    search passes (see func_contact_line_search_update) add the contact potentials at the step lengths they evaluate.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_slots = shell_contact.contacts_geom.shape[0]

    for i_b in range(B):
        for k in qd.static(range(N_LINE_SEARCH_TERMS)):
            shell_contact.envs_line_terms[i_b, k] = gs.qd_float(0.0)

    # Quadratic of the sheet along the step: dv^T A d, b^T d and d^T A d
    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_needs_solve[i_b] and func_is_vert_free(i_v, i_b, shell_state):
            step = shell_scratch.verts_p[i_v, i_b]
            product = shell_scratch.verts_Ap[i_v, i_b]
            shell_contact.envs_line_terms[i_b, 0] += shell_contact.verts_dv_prev[i_v, i_b].dot(product)
            shell_contact.envs_line_terms[i_b, 1] += shell_contact.verts_rhs_base[i_v, i_b].dot(step)
            shell_contact.envs_line_terms[i_b, 2] += step.dot(product)

    # Slope -p^T d of the contact potentials at the iterate
    for i_c, i_b in qd.ndrange(n_slots, B):
        if shell_scratch.envs_needs_solve[i_b] and shell_contact.contacts_geom[i_c, i_b] >= 0:
            shell_contact.envs_line_terms[i_b, 3] -= shell_contact.contacts_impulse[i_c, i_b].dot(
                shell_contact.contacts_dir[i_c, i_b]
            )

    # Kinetic energy of the rigid step, then the bracket [0, 1] if the step descends, its full length otherwise
    for i_b in range(B):
        shell_contact.envs_line_pass[i_b] = -1
        if shell_scratch.envs_needs_solve[i_b]:
            for i_t in range(shell_contact.trees_is_coupled.shape[0]):
                if shell_contact.trees_is_coupled[i_t, i_b]:
                    i_d_start = rigid_info.trees_dof_start[i_t]
                    i_d_end = i_d_start + rigid_info.trees_n_dofs[i_t]
                    for i_d in range(i_d_start, i_d_end):
                        product = gs.qd_float(0.0)
                        for j_d in range(i_d_start, i_d_end):
                            if rigid_info.dofs_mass_block_start[j_d] == rigid_info.dofs_mass_block_start[i_d]:
                                product += rigid_info.mass_mat[i_d, j_d, i_b] * shell_contact.dofs_dir[j_d, i_b]
                        shell_contact.envs_line_terms[i_b, 4] += shell_contact.dofs_dv[i_d, i_b] * product
                        shell_contact.envs_line_terms[i_b, 5] += shell_contact.dofs_dir[i_d, i_b] * product
            slope_start = (
                shell_contact.envs_line_terms[i_b, 0]
                - shell_contact.envs_line_terms[i_b, 1]
                + shell_contact.envs_line_terms[i_b, 3]
                + shell_contact.envs_line_terms[i_b, 4]
            )
            shell_contact.envs_step[i_b] = 1.0
            # A direction without descent is the vanishing step of a converged iterate, taken in full
            if slope_start < 0.0:
                shell_contact.envs_line_pass[i_b] = 0
                shell_contact.envs_line_bracket[i_b, 0] = gs.qd_float(0.0)
                shell_contact.envs_line_bracket[i_b, 1] = slope_start
                shell_contact.envs_line_bracket[i_b, 2] = gs.qd_float(1.0)

    for _ in range(1):
        line_flag[()] = 1


@qd.func
def func_contact_line_search_points(
    dt: float,
    shell_contact: array_class.ShellContactScratch,
):
    """Add the slope and curvature of every contact potential along the step at the step lengths of the current pass.

    A bracketing pass evaluates N_LINE_SEARCH_POINTS evenly spaced points of the bracket, a refining pass its candidate
    step length (see func_contact_line_search_update).
    """
    n_slots, B = shell_contact.contacts_geom.shape[0], shell_contact.contacts_geom.shape[1]
    n_points = shell_contact.envs_line_slope.shape[1]

    for i_b, j in qd.ndrange(B, n_points):
        shell_contact.envs_line_slope[i_b, j] = gs.qd_float(0.0)
        shell_contact.envs_line_curvature[i_b, j] = gs.qd_float(0.0)

    for i_c, i_b in qd.ndrange(n_slots, B):
        i_pass = shell_contact.envs_line_pass[i_b]
        if i_pass >= 0 and shell_contact.contacts_geom[i_c, i_b] >= 0:
            vel = func_contact_vel(i_c, i_b, shell_contact)
            direction = shell_contact.contacts_dir[i_c, i_b]
            gap = shell_contact.contacts_gap[i_c, i_b]
            normal = shell_contact.contacts_normal[i_c, i_b]
            stiffness = shell_contact.contacts_stiffness[i_c, i_b]
            friction_bound = shell_contact.contacts_friction_bound[i_c, i_b]
            step_lower = shell_contact.envs_line_bracket[i_b, 0]
            width = shell_contact.envs_line_bracket[i_b, 2] - step_lower
            n_evaluated = n_points
            if i_pass >= N_LINE_SEARCH_LEVELS:
                n_evaluated = 1
            for j in range(n_evaluated):
                step_length = step_lower + width * (j + 1.0) / n_points
                if i_pass >= N_LINE_SEARCH_LEVELS:
                    step_length = shell_contact.envs_line_bracket[i_b, 5]
                vel_step = vel + step_length * direction
                impulse = func_contact_potential(dt, vel_step, gap, normal, stiffness, friction_bound)[1]
                hessian = func_contact_hessian(dt, vel_step, gap, normal, stiffness, friction_bound)
                shell_contact.envs_line_slope[i_b, j] -= impulse.dot(direction)
                shell_contact.envs_line_curvature[i_b, j] += direction.dot(hessian @ direction)


@qd.func
def func_contact_line_search_update(
    line_flag: qd.types.ndarray(qd.i32, ndim=0),
    shell_contact: array_class.ShellContactScratch,
):
    """Narrow the bracket of the minimum of the potential along the step from the slopes of the current pass.

    The step length is chosen once the bracket is narrow enough, and the device loop keeps running while any
    environment still searches.

    The bracket holds its lower end, the slope there, its upper end, the slope and curvature there, and the candidate
    step length of the next refining pass. The full step is taken when the slope at its end stays below
    FULL_STEP_SLOPE_RATIO times the initial descent rate. Otherwise every bracketing pass keeps the first of its points
    of non-negative slope and the point before it, narrowing the bracket N_LINE_SEARCH_POINTS-fold. The refining passes
    then move the candidate by Newton steps on the slope, or bisect the bracket when the Newton step leaves it, until
    the slope at the candidate falls below LINE_SEARCH_SLOPE_TOLERANCE times the initial one. The resolution matters: a
    sticking contact holds within a band of slip velocities narrower than the bracketing resolution of the step.

    The step goes to the larger of the roots of the slope by secant over the last bracket and by tangent at its upper
    end, which lies at or past the minimum whether the slope bends up there (a contact starting to push) or down (a
    contact starting to slide). Landing past the minimum rather than short of it matters: a contact the step starts to
    push carries no stiffness in the Newton model until an iterate penetrates, so that steps stopping short of it would
    approach it without end.
    """
    B = shell_contact.envs_step.shape[0]
    n_points = shell_contact.envs_line_slope.shape[1]

    for _ in range(1):
        line_flag[()] = 0

    for i_b in range(B):
        i_pass = shell_contact.envs_line_pass[i_b]
        if i_pass >= 0:
            slope_linear = (
                shell_contact.envs_line_terms[i_b, 0]
                - shell_contact.envs_line_terms[i_b, 1]
                + shell_contact.envs_line_terms[i_b, 4]
            )
            curvature_linear = shell_contact.envs_line_terms[i_b, 2] + shell_contact.envs_line_terms[i_b, 5]
            slope_start = slope_linear + shell_contact.envs_line_terms[i_b, 3]
            step_lower = shell_contact.envs_line_bracket[i_b, 0]
            slope_lower = shell_contact.envs_line_bracket[i_b, 1]
            step_upper = shell_contact.envs_line_bracket[i_b, 2]
            slope_upper = shell_contact.envs_line_bracket[i_b, 3]
            curvature_upper = shell_contact.envs_line_bracket[i_b, 4]
            candidate = shell_contact.envs_line_bracket[i_b, 5]
            is_full = False
            is_found = False
            if i_pass < N_LINE_SEARCH_LEVELS:
                width = step_upper - step_lower
                slope_end = (
                    slope_linear + step_upper * curvature_linear + shell_contact.envs_line_slope[i_b, n_points - 1]
                )
                if i_pass == 0 and slope_end <= -FULL_STEP_SLOPE_RATIO * slope_start:
                    is_full = True
                else:
                    # The first point of non-negative slope, the last one when rounding leaves them all negative
                    j_upper = n_points - 1
                    for j_ in range(n_points - 1):
                        j = n_points - 2 - j_
                        step_length = step_lower + width * (j + 1.0) / n_points
                        if slope_linear + step_length * curvature_linear + shell_contact.envs_line_slope[i_b, j] >= 0.0:
                            j_upper = j
                    if j_upper > 0:
                        step_lower_new = step_lower + width * j_upper / n_points
                        slope_lower = (
                            slope_linear
                            + step_lower_new * curvature_linear
                            + shell_contact.envs_line_slope[i_b, j_upper - 1]
                        )
                        step_lower = step_lower_new
                    step_upper = step_lower + width / n_points
                    slope_upper = (
                        slope_linear + step_upper * curvature_linear + shell_contact.envs_line_slope[i_b, j_upper]
                    )
                    curvature_upper = curvature_linear + shell_contact.envs_line_curvature[i_b, j_upper]
                    candidate = 0.5 * (step_lower + step_upper)
                    if slope_upper > slope_lower:
                        candidate = step_lower - (step_upper - step_lower) * slope_lower / (slope_upper - slope_lower)
            else:
                slope_candidate = slope_linear + candidate * curvature_linear + shell_contact.envs_line_slope[i_b, 0]
                curvature_candidate = curvature_linear + shell_contact.envs_line_curvature[i_b, 0]
                if slope_candidate >= 0.0:
                    step_upper = candidate
                    slope_upper = slope_candidate
                    curvature_upper = curvature_candidate
                else:
                    step_lower = candidate
                    slope_lower = slope_candidate
                is_found = qd.abs(slope_candidate) <= -LINE_SEARCH_SLOPE_TOLERANCE * slope_start
                step_newton = gs.qd_float(-1.0)
                if curvature_candidate > 0.0:
                    step_newton = candidate - slope_candidate / curvature_candidate
                candidate = 0.5 * (step_lower + step_upper)
                if step_lower < step_newton and step_newton < step_upper:
                    candidate = step_newton

            if is_full:
                shell_contact.envs_line_pass[i_b] = -1
            elif is_found or i_pass == N_LINE_SEARCH_LEVELS + N_LINE_SEARCH_REFINEMENTS - 1:
                step_chosen = step_upper
                if slope_upper > slope_lower:
                    step_chosen = step_lower - (step_upper - step_lower) * slope_lower / (slope_upper - slope_lower)
                if curvature_upper > 0.0:
                    step_chosen = qd.max(step_chosen, step_upper - slope_upper / curvature_upper)
                shell_contact.envs_step[i_b] = qd.min(qd.max(step_chosen, step_lower), step_upper)
                shell_contact.envs_line_pass[i_b] = -1
            else:
                shell_contact.envs_line_bracket[i_b, 0] = step_lower
                shell_contact.envs_line_bracket[i_b, 1] = slope_lower
                shell_contact.envs_line_bracket[i_b, 2] = step_upper
                shell_contact.envs_line_bracket[i_b, 3] = slope_upper
                shell_contact.envs_line_bracket[i_b, 4] = curvature_upper
                shell_contact.envs_line_bracket[i_b, 5] = candidate
                shell_contact.envs_line_pass[i_b] = i_pass + 1
                line_flag[()] = 1


@qd.func
def func_contact_line_search_apply(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
):
    """Move the iterate of every environment still solving by the step length its line search chose."""
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_slots = shell_contact.contacts_geom.shape[0]
    n_dofs = shell_contact.dofs_schur_vec.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        if shell_scratch.envs_needs_solve[i_b]:
            dv = shell_contact.verts_dv_prev[i_v, i_b] + shell_contact.envs_step[i_b] * shell_scratch.verts_p[i_v, i_b]
            shell_state.verts_dv[i_v, i_b] = dv
            shell_contact.verts_dv_prev[i_v, i_b] = dv

    for i_d, i_b in qd.ndrange(n_dofs, B):
        if shell_scratch.envs_needs_solve[i_b]:
            shell_contact.dofs_dv[i_d, i_b] += shell_contact.envs_step[i_b] * shell_contact.dofs_dir[i_d, i_b]

    for i_c, i_b in qd.ndrange(n_slots, B):
        if shell_scratch.envs_needs_solve[i_b] and shell_contact.contacts_geom[i_c, i_b] >= 0:
            shell_contact.contacts_vel_change[i_c, i_b] += (
                shell_contact.envs_step[i_b] * shell_contact.contacts_dir[i_c, i_b]
            )


@qd.func
def func_contact_continue(newton_flag: qd.types.ndarray(qd.i32, ndim=0), shell_scratch: array_class.ShellScratch):
    """Keep the device loop of the contact solve running while any environment still iterates."""
    for _ in range(1):
        newton_flag[()] = 0

    for i_b in range(shell_scratch.envs_needs_solve.shape[0]):
        if shell_scratch.envs_needs_solve[i_b]:
            newton_flag[()] = 1


@qd.kernel(graph=True)
def kernel_shell_rigid_contact_solve(
    dt: float,
    newton_flag: qd.types.ndarray(qd.i32, ndim=0),
    pcg_flag: qd.types.ndarray(qd.i32, ndim=0),
    line_flag: qd.types.ndarray(qd.i32, ndim=0),
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    dyn_state: array_class.DynState,
    shell_info: array_class.ShellInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    shell_static_config: qd.template(),
    rigid_config: qd.template(),
    max_newton_iterations: int,
    max_iterations: int,
    contact_stiffness: float,
    tolerance: float,
    velocity_tolerance: float,
    errno: qd.Tensor,
):
    """Solve the velocity update of the sheets jointly with their rigid contacts, by Newton iterations on the
    incremental potential of the substep, until every environment converges or fails.

    Every iteration linearizes the contacts about the current iterate (func_contact_linearize), solves the linearized
    system by the preconditioned conjugate gradient (PCG) of kernel_shell_pcg_solve on the Schur complement of the rigid
    degrees of freedom, and steps to the minimum of the potential towards its solution
    (func_contact_line_search_update). The three loops run on the device while any environment iterates.
    """
    for _ in range(1):
        newton_flag[()] = 1
    while qd.graph.do_while(newton_flag):
        func_contact_linearize(
            dt, shell_scratch, shell_contact, contact_stiffness, velocity_tolerance, max_newton_iterations
        )
        func_contact_assemble(shell_state, shell_scratch, shell_contact, dyn_info, rigid_info)
        func_pcg_prepare(pcg_flag, shell_state, shell_scratch, shell_static_config, tolerance, velocity_tolerance)
        while qd.graph.do_while(pcg_flag):
            func_system_product(shell_state, shell_scratch, shell_info)
            func_contact_product(shell_state, shell_scratch, shell_contact, dyn_info, rigid_info)
            func_contact_schur_product(shell_state, shell_scratch, shell_contact, dyn_info, rigid_info)
            func_pcg_residual(pcg_flag, shell_state, shell_scratch, shell_info, shell_static_config)
            if qd.static(shell_static_config.has_coarse_space):
                func_pcg_coarse_solve(shell_scratch, shell_info)
            func_pcg_decide(pcg_flag, shell_scratch, shell_static_config, max_iterations, errno)
            func_pcg_advance(shell_state, shell_scratch, shell_info, shell_static_config)
        func_contact_direction(shell_state, shell_scratch, shell_contact, shell_info, dyn_info, rigid_info)
        func_contact_line_search_start(line_flag, shell_state, shell_scratch, shell_contact, rigid_info)
        while qd.graph.do_while(line_flag):
            func_contact_line_search_points(dt, shell_contact)
            func_contact_line_search_update(line_flag, shell_contact)
        func_contact_line_search_apply(shell_state, shell_scratch, shell_contact)
        func_contact_continue(newton_flag, shell_scratch)


@qd.kernel
def kernel_shell_rigid_contact_finalize(
    dt: float,
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_contact: array_class.ShellContactScratch,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
):
    """Apply the solved contacts at the last iterate to the sheets and the rigid links.

    The rigid degrees of freedom accelerate by their velocity change over the substep, which the rigid solver then
    integrates, and every contact spreads its force on the vertices of its face and adds the opposite force to the
    contact force of its rigid link. Every contact slot records its geom and mu times its normal impulse, the friction
    bound of the next substep (see kernel_shell_rigid_contact_detect). An environment whose solve produced non-finite
    values applies nothing, errno halting the simulation at its next check.
    """
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_slots = shell_contact.contacts_geom.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        shell_contact.verts_contact_force[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_d, i_b in qd.ndrange(dyn_state.dofs.acc.shape[0], B):
        if not shell_scratch.envs_solve_status[i_b] & SHELL_SOLVE_STATUS.NON_FINITE:
            dyn_state.dofs.acc[i_d, i_b] += shell_contact.dofs_dv[i_d, i_b] / dt

    for i_c, i_b in qd.ndrange(n_slots, B):
        i_g = shell_contact.contacts_geom[i_c, i_b]
        if shell_scratch.envs_solve_status[i_b] & SHELL_SOLVE_STATUS.NON_FINITE:
            i_g = -1
        shell_state.contacts_geom_prev[i_c, i_b] = i_g
        shell_state.contacts_friction_bound_prev[i_c, i_b] = gs.qd_float(0.0)
        if i_g >= 0:
            impulse = func_contact_potential(
                dt,
                func_contact_vel(i_c, i_b, shell_contact),
                shell_contact.contacts_gap[i_c, i_b],
                shell_contact.contacts_normal[i_c, i_b],
                shell_contact.contacts_stiffness[i_c, i_b],
                shell_contact.contacts_friction_bound[i_c, i_b],
            )[1]
            shell_contact.contacts_impulse[i_c, i_b] = impulse
            shell_state.contacts_friction_bound_prev[i_c, i_b] = dyn_info.geoms.coup_friction[i_g] * qd.max(
                impulse.dot(shell_contact.contacts_normal[i_c, i_b]), 0.0
            )
            force = impulse / dt
            i_f = i_c // 2
            bary = shell_contact.contacts_bary[i_c, i_b]
            for k in qd.static(range(3)):
                i_v = shell_state.corners_vert[3 * i_f + k, i_b]
                shell_contact.verts_contact_force[i_v, i_b] += bary[k] * force
            i_l = dyn_info.geoms.link_idx[i_g]
            dyn_state.links.contact_force[i_l, i_b] -= force


# ------------------------------------------------------------------------------------
# ------------------------------------ accessors -------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel
def kernel_shell_update_render(
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    shell_info: array_class.ShellInfo,
):
    """Write the position and the smooth normal of every face corner, from the vertex holding it."""
    n_verts, B = shell_state.verts_pos.shape[0], shell_state.verts_pos.shape[1]
    n_faces = shell_state.faces_thickness.shape[0]

    for i_v, i_b in qd.ndrange(n_verts, B):
        shell_scratch.verts_normal[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_f, i_b in qd.ndrange(n_faces, B):
        i_v0 = shell_state.corners_vert[3 * i_f, i_b]
        i_v1 = shell_state.corners_vert[3 * i_f + 1, i_b]
        i_v2 = shell_state.corners_vert[3 * i_f + 2, i_b]
        x0 = shell_state.verts_pos[i_v0, i_b]
        normal = (shell_state.verts_pos[i_v1, i_b] - x0).cross(shell_state.verts_pos[i_v2, i_b] - x0)
        shell_scratch.verts_normal[i_v0, i_b] += normal
        shell_scratch.verts_normal[i_v1, i_b] += normal
        shell_scratch.verts_normal[i_v2, i_b] += normal

    for i_c, i_b in qd.ndrange(3 * n_faces, B):
        i_v = shell_state.corners_vert[i_c, i_b]
        shell_scratch.corners_render_pos[i_c, i_b] = shell_state.verts_pos[i_v, i_b]
        normal = shell_scratch.verts_normal[i_v, i_b]
        shell_scratch.corners_render_normal[i_c, i_b] = normal / qd.max(normal.norm(), NORM_FLOOR)


@qd.kernel
def kernel_shell_set_verts_vel(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    shell_state: array_class.ShellState,
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            shell_state.verts_vel[i_v, i_b][j] = values[i_b_, i_v_, j]


@qd.kernel
def kernel_shell_set_verts_pos(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    shell_state: array_class.ShellState,
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        pos = qd.Vector([values[i_b_, i_v_, 0], values[i_b_, i_v_, 1], values[i_b_, i_v_, 2]], dt=gs.qd_float)
        cell = qd.floor(pos / POS_GRID + 0.5).cast(gs.qd_int)
        shell_state.verts_pos[i_v, i_b] = pos
        shell_state.verts_pos_cell[i_v, i_b] = cell
        shell_state.verts_pos_offset[i_v, i_b] = pos - cell.cast(gs.qd_float) * POS_GRID


@qd.kernel
def kernel_shell_set_verts_fixed(
    verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), is_fixed: bool, shell_state: array_class.ShellState
):
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        shell_state.verts_is_fixed[i_v, i_b] = is_fixed


@qd.kernel
def kernel_shell_set_state(
    envs_idx: qd.types.ndarray(),
    verts_pos: qd.types.ndarray(),
    verts_pos_cell: qd.types.ndarray(),
    verts_pos_offset: qd.types.ndarray(),
    verts_vel: qd.types.ndarray(),
    verts_dv: qd.types.ndarray(),
    verts_origin: qd.types.ndarray(),
    verts_is_fixed: qd.types.ndarray(),
    corners_vert: qd.types.ndarray(),
    contacts_geom_prev: qd.types.ndarray(),
    contacts_friction_bound_prev: qd.types.ndarray(),
    entities_n_verts: qd.types.ndarray(),
    entities_peak_damage: qd.types.ndarray(),
    entities_failure_face: qd.types.ndarray(),
    envs_solver_failure: qd.types.ndarray(),
    faces_plastic: qd.types.ndarray(),
    faces_thickness: qd.types.ndarray(),
    hinges_plastic_angle: qd.types.ndarray(),
    shell_state: array_class.ShellState,
    shell_scratch: array_class.ShellScratch,
    coarse_update_interval: int,
    errno: qd.Tensor,
):
    """Write the state of some environments, whose coarse matrices become due and whose errors clear."""
    n_verts = shell_state.verts_pos.shape[0]
    n_faces = shell_state.faces_thickness.shape[0]
    n_hinges = shell_state.hinges_plastic_angle.shape[0]
    n_entities = shell_state.entities_n_verts.shape[0]
    for i_v, i_b_ in qd.ndrange(n_verts, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            shell_state.verts_pos[i_v, i_b][j] = verts_pos[i_b_, i_v, j]
            shell_state.verts_pos_cell[i_v, i_b][j] = verts_pos_cell[i_b_, i_v, j]
            shell_state.verts_pos_offset[i_v, i_b][j] = verts_pos_offset[i_b_, i_v, j]
            shell_state.verts_vel[i_v, i_b][j] = verts_vel[i_b_, i_v, j]
            shell_state.verts_dv[i_v, i_b][j] = verts_dv[i_b_, i_v, j]
        shell_state.verts_origin[i_v, i_b] = verts_origin[i_b_, i_v]
        shell_state.verts_is_fixed[i_v, i_b] = verts_is_fixed[i_b_, i_v]
    for i_c, i_b_ in qd.ndrange(3 * n_faces, envs_idx.shape[0]):
        shell_state.corners_vert[i_c, envs_idx[i_b_]] = corners_vert[i_b_, i_c]
    for i_c, i_b_ in qd.ndrange(2 * n_faces, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        shell_state.contacts_geom_prev[i_c, i_b] = contacts_geom_prev[i_b_, i_c]
        shell_state.contacts_friction_bound_prev[i_c, i_b] = contacts_friction_bound_prev[i_b_, i_c]
    for i_f, i_b_ in qd.ndrange(n_faces, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for j, k in qd.static(qd.ndrange(2, 2)):
            shell_state.faces_plastic[i_f, i_b][j, k] = faces_plastic[i_b_, i_f, j, k]
        shell_state.faces_thickness[i_f, i_b] = faces_thickness[i_b_, i_f]
    for i_h, i_b_ in qd.ndrange(n_hinges, envs_idx.shape[0]):
        shell_state.hinges_plastic_angle[i_h, envs_idx[i_b_]] = hinges_plastic_angle[i_b_, i_h]
    for i_e, i_b_ in qd.ndrange(n_entities, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        shell_state.entities_n_verts[i_e, i_b] = entities_n_verts[i_b_, i_e]
        shell_state.entities_peak_damage[i_e, i_b] = entities_peak_damage[i_b_, i_e]
        shell_state.entities_failure_face[i_e, i_b] = entities_failure_face[i_b_, i_e]
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        shell_state.envs_solver_failure[i_b] = envs_solver_failure[i_b_]
        shell_scratch.envs_coarse_age[i_b] = coarse_update_interval
        errno[i_b] = 0
