from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti

import genesis as gs
from genesis.options.solvers import IPCCouplerOptions
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


@ti.data_oriented
class IPCCoupler(RBC):
    """
    Coupler class for handling Incremental Potential Contact (IPC) simulation coupling.

    This coupler manages the communication between Genesis solvers and the IPC system,
    including rigid bodies (as ABD objects) and FEM bodies in a unified contact framework.
    """

    def __init__(self, simulator: "Simulator", options: "IPCCouplerOptions") -> None:
        """
        Initialize IPC Coupler.

        Parameters
        ----------
        simulator : Simulator
            The simulator containing all solvers
        options : IPCCouplerOptions
            IPC configuration options
        """
        self.sim = simulator
        self.options = options

        # Store solver references
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver

        # IPC system components (will be initialized in build)
        self._ipc_engine = None
        self._ipc_world = None
        self._ipc_scene = None
        self._ipc_abd = None
        self._ipc_stk = None
        self._ipc_abd_contact = None
        self._ipc_fem_contact = None
        self._ipc_scene_contacts = {}

    def build(self) -> None:
        """Build IPC system and auto-configure solver options"""
        # Auto-enable IPC for active solvers when using IPCCoupler
        if self.fem_solver.is_active():
            self.fem_solver._options.use_IPC = True
            gs.logger.info("Auto-enabled use_IPC for FEM solver due to IPCCoupler")

        if self.rigid_solver.is_active():
            self.rigid_solver._options.use_IPC = True
            gs.logger.info("Auto-enabled use_IPC for Rigid solver due to IPCCoupler")

        # Initialize IPC system since we're using IPCCoupler
        self._init_ipc()
        self._add_objects_to_ipc()
        self._finalize_ipc()

    def _init_ipc(self):
        """Initialize IPC system components"""
        from uipc.core import Engine, World, Scene
        from uipc.constitution import AffineBodyConstitution, StableNeoHookean

        # Disable IPC logging if requested
        if self.options.disable_ipc_logging:
            from uipc import Logger, Timer
            Logger.set_level(Logger.Level.Error)
            Timer.disable_all()

        # Create IPC engine and world
        from asset_dir import AssetDir
        workspace = AssetDir.output_path(__file__)
        self._ipc_engine = Engine('cuda', workspace)
        self._ipc_world = World(self._ipc_engine)

        # Create IPC scene with configuration
        config = Scene.default_config()
        config['dt'] = self.options.dt
        config['gravity'] = [[self.options.gravity[0]], [self.options.gravity[1]], [self.options.gravity[2]]]
        config['contact']['d_hat'] = self.options.contact_d_hat
        config['contact']['friction']['enable'] = self.options.contact_friction_enable
        config['newton']['velocity_tol'] = self.options.newton_velocity_tol
        config['line_search']['max_iter'] = self.options.line_search_max_iter
        config['linear_system']['tol_rate'] = self.options.linear_system_tol_rate
        config['sanity_check']['enable'] = self.options.sanity_check_enable

        self._ipc_scene = Scene(config)

        # Create constitutions
        self._ipc_abd = AffineBodyConstitution()
        self._ipc_stk = StableNeoHookean()

        # Add constitutions to scene
        self._ipc_scene.constitution_tabular().insert(self._ipc_abd)
        self._ipc_scene.constitution_tabular().insert(self._ipc_stk)

        # Set up contact model
        self._ipc_scene.contact_tabular().default_model(self.options.contact_friction_mu, self.options.contact_resistance)

        # Set up contact subscenes for multi-environment
        B = self.sim._B
        for i in range(B):
            self._ipc_scene_contacts[i] = self._ipc_scene.contact_tabular().create_subscene(f"contact_model{i}")
        for i in range(B):
            for j in range(B):
                if i != j:
                    self._ipc_scene.contact_tabular().subscene_insert(self._ipc_scene_contacts[i], self._ipc_scene_contacts[j], False)

        # Set up separate contact elements for ABD and FEM
        self._ipc_abd_contact = self._ipc_scene.contact_tabular().create("abd_contact")
        self._ipc_fem_contact = self._ipc_scene.contact_tabular().create("fem_contact")

        # Configure contact interactions based on IPC coupler options
        self._ipc_scene.contact_tabular().insert(self._ipc_fem_contact, self._ipc_fem_contact, self.options.contact_friction_mu, self.options.contact_resistance, True)
        self._ipc_scene.contact_tabular().insert(self._ipc_fem_contact, self._ipc_abd_contact, self.options.contact_friction_mu, self.options.contact_resistance, True)
        self._ipc_scene.contact_tabular().insert(self._ipc_abd_contact, self._ipc_abd_contact, self.options.contact_friction_mu, self.options.contact_resistance, self.options.IPC_self_contact)

    def _add_objects_to_ipc(self):
        """Add solvers to IPC system"""
        # Add FEM solver to IPC
        if self.fem_solver.is_active() and self.fem_solver._options.use_IPC:
            # Pass IPC components to FEM solver
            self.fem_solver._ipc_engine = self._ipc_engine
            self.fem_solver._ipc_world = self._ipc_world
            self.fem_solver._ipc_scene = self._ipc_scene
            self.fem_solver._ipc_stk = self._ipc_stk
            self.fem_solver._ipc_fem_contact = self._ipc_fem_contact
            self.fem_solver._ipc_scene_contacts = self._ipc_scene_contacts
            self.fem_solver.add_fem_entities_to_ipc()

        # Add rigid solver to IPC
        if self.rigid_solver.is_active() and self.rigid_solver._options.use_IPC:
            # Pass IPC components to rigid solver
            self.rigid_solver._ipc_scene = self._ipc_scene
            self.rigid_solver._ipc_abd = self._ipc_abd
            self.rigid_solver._ipc_abd_contact = self._ipc_abd_contact
            self.rigid_solver._ipc_scene_contacts = self._ipc_scene_contacts
            self.rigid_solver._abd_transforms = {}
            self.rigid_solver.add_rigid_geoms_to_ipc(self.options.ipc_constraint_strength)

    def _finalize_ipc(self):
        """Finalize IPC setup"""
        self._ipc_world.init(self._ipc_scene)
        gs.logger.info("IPC world initialized successfully")

    def is_active(self) -> bool:
        """Check if IPC coupling is active"""
        return self._ipc_world is not None

    def preprocess(self, f):
        """Preprocessing step before coupling"""
        pass

    def couple(self, f):
        """Execute IPC coupling step"""
        if not self.is_active():
            return

        # Advance IPC simulation
        self._ipc_world.advance()
        self._ipc_world.retrieve()

        # Call retrieve methods on solvers to get updated states
        if self.rigid_solver.is_active() and self.rigid_solver._options.use_IPC:
            if hasattr(self.rigid_solver, 'retrieve_ipc'):
                self.rigid_solver.retrieve_ipc(f)

        if self.fem_solver.is_active() and self.fem_solver._options.use_IPC:
            if hasattr(self.fem_solver, 'retrieve_ipc'):
                self.fem_solver.retrieve_ipc(f)

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        # IPC doesn't support gradients yet
        pass

    def reset(self, envs_idx=None):
        """Reset coupling state"""
        # IPC doesn't need special reset logic currently
        pass