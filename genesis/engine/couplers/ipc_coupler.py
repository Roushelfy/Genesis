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
        """Add objects from solvers to IPC system"""
        # Add FEM entities to IPC
        if self.fem_solver.is_active() and self.fem_solver._options.use_IPC:
            self._add_fem_entities_to_ipc()

        # Add rigid geoms to IPC
        if self.rigid_solver.is_active() and self.rigid_solver._options.use_IPC:
            self._add_rigid_geoms_to_ipc()

    def _add_fem_entities_to_ipc(self):
        """Add FEM entities to the existing IPC scene"""
        from uipc.constitution import ElasticModuli
        from uipc.geometry import label_surface, tetmesh

        fem_solver = self.fem_solver
        scene = self._ipc_scene
        stk = self._ipc_stk
        scene_contacts = self._ipc_scene_contacts

        fem_solver._mesh_handles = {}
        fem_solver.list_env_obj = []
        fem_solver.list_env_mesh = []

        for i_b in range(self.sim._B):
            fem_solver.list_env_obj.append([])
            fem_solver.list_env_mesh.append([])
            for i_e, entity in enumerate(fem_solver._entities):
                # Create FEM object in IPC
                fem_solver.list_env_obj[i_b].append(scene.objects().create(f"fem_obj_{i_b}_{i_e}"))

                # Create tetrahedral mesh for FEM entity
                fem_solver.list_env_mesh[i_b].append(tetmesh(entity.init_positions.cpu().numpy(), entity.elems))

                # Add to contact subscene
                scene_contacts[i_b].subscene_append(fem_solver.list_env_mesh[i_b][i_e])
                # Apply FEM contact element for selective collision control
                self._ipc_fem_contact.apply_to(fem_solver.list_env_mesh[i_b][i_e])
                label_surface(fem_solver.list_env_mesh[i_b][i_e])

                # Apply material properties
                moduli_box = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                stk.apply_to(fem_solver.list_env_mesh[i_b][i_e], moduli_box,mass_density=entity.material.rho)

                # Add metadata to identify this as FEM geometry
                meta_attrs = fem_solver.list_env_mesh[i_b][i_e].meta()
                meta_attrs.create("solver_type", "fem")
                meta_attrs.create("env_idx", str(i_b))
                meta_attrs.create("entity_idx", str(i_e))

                # Create geometry in IPC scene
                fem_solver.list_env_obj[i_b][i_e].geometries().create(fem_solver.list_env_mesh[i_b][i_e])
                fem_solver._mesh_handles[f"gs_ipc_{i_b}_{i_e}"] = fem_solver.list_env_mesh[i_b][i_e]

    def _add_rigid_geoms_to_ipc(self):
        """Add rigid geoms to the existing IPC scene as ABD objects, merging geoms by link_idx"""
        from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles, merge, ground
        from genesis.utils import mesh as mu
        import numpy as np
        import trimesh

        rigid_solver = self.rigid_solver
        scene = self._ipc_scene
        abd = self._ipc_abd
        scene_contacts = self._ipc_scene_contacts

        # Initialize lists following FEM solver pattern
        rigid_solver.list_env_obj = []
        rigid_solver.list_env_mesh = []
        rigid_solver._mesh_handles = {}
        rigid_solver._abd_transforms = {}

        for i_b in range(self.sim._B):
            rigid_solver.list_env_obj.append([])
            rigid_solver.list_env_mesh.append([])

            # Group geoms by link_idx for merging
            link_geoms = {}  # link_idx -> dict with 'meshes', 'link_world_pos', 'link_world_quat', 'entity_idx'
            link_planes = {}  # link_idx -> list of plane geoms (handle separately)

            # First pass: collect and group geoms by link_idx
            for i_g in range(rigid_solver.n_geoms_):
                geom_type = rigid_solver.geoms_info.type[i_g]
                link_idx = rigid_solver.geoms_info.link_idx[i_g]
                entity_idx = rigid_solver.links_info.entity_idx[link_idx]
                entity = rigid_solver._entities[entity_idx]

                # Check if this link should be included in IPC based on entity's filter
                if hasattr(entity, '_ipc_link_filter') and entity._ipc_link_filter is not None:
                    if link_idx not in entity._ipc_link_filter:
                        continue  # Skip this geom/link

                # Initialize link group if not exists
                if link_idx not in link_geoms:
                    link_geoms[link_idx] = {
                        'meshes': [],
                        'link_world_pos': None,
                        'link_world_quat': None,
                        'entity_idx': entity_idx
                    }
                    link_planes[link_idx] = []

                try:
                    if geom_type == gs.GEOM_TYPE.PLANE:
                        # Handle planes separately (they can't be merged with SimplicialComplex)
                        pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                        normal = np.array([0.0, 0.0, 1.0])  # Z-up
                        height = np.dot(pos, normal)
                        plane_geom = ground(height, normal)
                        self._ipc_abd_contact.apply_to(plane_geom)
                        link_planes[link_idx].append((i_g, plane_geom))

                    else:
                        # For all non-plane geoms, create tetmesh
                        vert_num = rigid_solver.geoms_info.vert_num[i_g]
                        if vert_num == 0:
                            continue  # Skip geoms without vertices

                        # Extract vertex and face data
                        vert_start = rigid_solver.geoms_info.vert_start[i_g]
                        vert_end = rigid_solver.geoms_info.vert_end[i_g]
                        face_start = rigid_solver.geoms_info.face_start[i_g]
                        face_end = rigid_solver.geoms_info.face_end[i_g]

                        # Get vertices and faces
                        geom_verts = rigid_solver.verts_info.init_pos.to_numpy()[vert_start:vert_end]
                        geom_faces = rigid_solver.faces_info.verts_idx.to_numpy()[face_start:face_end]
                        geom_faces = geom_faces - vert_start  # Adjust indices

                        # Apply geom-relative transform to vertices (needed for merging)
                        geom_rel_pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                        geom_rel_quat = rigid_solver.geoms_info.quat[i_g].to_numpy()

                        # Transform vertices by geom relative transform
                        import genesis.utils.geom as gu
                        geom_rot_mat = gu.quat_to_R(geom_rel_quat)
                        transformed_verts = geom_verts @ geom_rot_mat.T + geom_rel_pos

                        # Convert trimesh to tetmesh
                        try:
                            tri_mesh = trimesh.Trimesh(vertices=transformed_verts, faces=geom_faces)
                            verts, elems = mu.tetrahedralize_mesh(tri_mesh, tet_cfg=dict())
                            rigid_mesh = tetmesh(verts.astype(np.float64), elems.astype(np.int32))

                            # Store mesh and geom info
                            link_geoms[link_idx]['meshes'].append((i_g, rigid_mesh))

                        except Exception as e:
                            gs.logger.warning(f"Failed to convert trimesh to tetmesh for geom {i_g}: {e}")
                            continue

                    # Store link transform info (same for all geoms in link)
                    if link_geoms[link_idx]['link_world_pos'] is None:
                        link_geoms[link_idx]['link_world_pos'] = rigid_solver.links_state.pos[link_idx, i_b]
                        link_geoms[link_idx]['link_world_quat'] = rigid_solver.links_state.quat[link_idx, i_b]

                except Exception as e:
                    gs.logger.warning(f"Failed to process geom {i_g}: {e}")
                    continue

            # Second pass: merge geoms per link and create IPC objects
            link_obj_counter = 0
            for link_idx, link_data in link_geoms.items():
                try:
                    # Handle regular meshes (merge if multiple)
                    if link_data['meshes']:
                        if len(link_data['meshes']) == 1:
                            # Single mesh in link
                            geom_idx, merged_mesh = link_data['meshes'][0]
                        else:
                            # Multiple meshes in link - merge them
                            meshes_to_merge = [mesh for geom_idx, mesh in link_data['meshes']]
                            merged_mesh = merge(meshes_to_merge)
                            geom_idx = link_data['meshes'][0][0]  # Use first geom's index for metadata

                        # Apply link world transform
                        from uipc import view, Transform, Vector3, Quaternion
                        trans_view = view(merged_mesh.transforms())
                        t = Transform.Identity()

                        link_world_pos = link_data['link_world_pos']
                        link_world_quat = link_data['link_world_quat']

                        # Ensure numpy format
                        link_world_pos = link_world_pos.to_numpy()
                        link_world_quat = link_world_quat.to_numpy()

                        t.translate(Vector3.Values((link_world_pos[0], link_world_pos[1], link_world_pos[2])))
                        uipc_link_quat = Quaternion(link_world_quat)
                        t.rotate(uipc_link_quat)
                        trans_view[0] = t.matrix()

                        # Process surface for contact
                        label_surface(merged_mesh)
                        label_triangle_orient(merged_mesh)
                        merged_mesh = flip_inward_triangles(merged_mesh)

                        # Create rigid object
                        rigid_obj = scene.objects().create(f"rigid_link_{i_b}_{link_idx}")
                        rigid_solver.list_env_obj[i_b].append(rigid_obj)
                        rigid_solver.list_env_mesh[i_b].append(merged_mesh)

                        # Add to contact subscene and apply ABD constitution
                        scene_contacts[i_b].subscene_append(merged_mesh)
                        self._ipc_abd_contact.apply_to(merged_mesh)
                        from uipc.unit import MPa
                        abd.apply_to(merged_mesh, 10.0 * MPa)

                        # Apply soft transform constraints
                        from uipc.constitution import SoftTransformConstraint
                        if not hasattr(self, '_ipc_stc'):
                            self._ipc_stc = SoftTransformConstraint()
                            scene.constitution_tabular().insert(self._ipc_stc)

                        strength_tuple = self.options.ipc_constraint_strength
                        constraint_strength = np.array([
                            strength_tuple[0],  # translation strength
                            strength_tuple[1],  # rotation strength
                        ])
                        self._ipc_stc.apply_to(merged_mesh, constraint_strength)

                        # Add metadata
                        meta_attrs = merged_mesh.meta()
                        meta_attrs.create("solver_type", "rigid")
                        meta_attrs.create("env_idx", str(i_b))
                        meta_attrs.create("link_idx", str(link_idx))  # Use link_idx instead of geom_idx

                        rigid_obj.geometries().create(merged_mesh)

                        # Set up animator for this link
                        if not hasattr(self, '_ipc_animator'):
                            self._ipc_animator = scene.animator()

                        def create_animate_function(env_idx, link_idx):
                            def animate_rigid_link(info):
                                from uipc import view, builtin, Transform, Vector3, Quaternion

                                geo_slots = info.geo_slots()
                                if len(geo_slots) == 0:
                                    return
                                geo = geo_slots[0].geometry()

                                try:
                                    # Get current link state instead of geom state
                                    link_pos = rigid_solver.get_links_pos(links_idx=link_idx, envs_idx=env_idx)
                                    link_quat = rigid_solver.get_links_quat(links_idx=link_idx, envs_idx=env_idx)

                                    link_pos = link_pos.detach().cpu().numpy()
                                    link_quat = link_quat.detach().cpu().numpy()

                                    # Handle array shapes
                                    while len(link_pos.shape) > 1 and link_pos.shape[0] == 1:
                                        link_pos = link_pos[0]
                                    while len(link_quat.shape) > 1 and link_quat.shape[0] == 1:
                                        link_quat = link_quat[0]

                                    pos_1d = link_pos.flatten()[:3]
                                    quat_1d = link_quat.flatten()[:4]

                                    # Create transform
                                    t = Transform.Identity()
                                    t.translate(Vector3.Values((pos_1d[0], pos_1d[1], pos_1d[2])))
                                    uipc_quat = Quaternion(quat_1d)
                                    t.rotate(uipc_quat)

                                    # Enable constraint and set target transform
                                    is_constrained = geo.instances().find(builtin.is_constrained)
                                    aim_transform = geo.instances().find(builtin.aim_transform)

                                    if is_constrained and aim_transform:
                                        view(is_constrained)[0] = 1
                                        view(aim_transform)[:] = t.matrix()

                                except Exception as e:
                                    gs.logger.warning(f"Error retrieving Genesis state for IPC animation: {e}")

                            return animate_rigid_link

                        animate_func = create_animate_function(i_b, link_idx)
                        self._ipc_animator.insert(rigid_obj, animate_func)

                        rigid_solver._mesh_handles[f"rigid_link_{i_b}_{link_idx}"] = merged_mesh
                        link_obj_counter += 1

                    # Handle planes for this link separately
                    for geom_idx, plane_geom in link_planes[link_idx]:
                        plane_obj = scene.objects().create(f"rigid_plane_{i_b}_{geom_idx}")
                        rigid_solver.list_env_obj[i_b].append(plane_obj)
                        rigid_solver.list_env_mesh[i_b].append(None)  # Planes are ImplicitGeometry

                        plane_obj.geometries().create(plane_geom)
                        rigid_solver._mesh_handles[f"rigid_plane_{i_b}_{geom_idx}"] = plane_geom
                        link_obj_counter += 1

                except Exception as e:
                    gs.logger.warning(f"Failed to create IPC object for link {link_idx}: {e}")
                    continue


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

        # Retrieve updated states from IPC for all solvers
        self._retrieve_fem_states(f)
        self._retrieve_rigid_states(f)

        # Update IPC GUI if enabled
        # scene = self.sim._scene
        # if hasattr(scene, '_ipc_gui_enabled') and scene._ipc_gui_enabled and hasattr(scene, '_ipc_scene_gui'):
        #     scene._ipc_scene_gui.update()

    def _retrieve_fem_states(self, f):
        # IPC world advance/retrieve is handled at Scene level
        # This method only handles FEM-specific post-processing

        if not (self.fem_solver.is_active() and self.fem_solver._options.use_IPC):
            return

        # Gather FEM volumetric tet states using metadata filtering
        from uipc import builtin
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot, apply_transform, merge
        import numpy as np

        visitor = SceneVisitor(self._ipc_scene)

        # Collect only FEM geometries using metadata
        fem_geo_by_entity = {}
        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() == 3:
                    try:
                        # Check if this is a FEM geometry using metadata
                        meta_attrs = geo.meta()
                        solver_type_attr = meta_attrs.find("solver_type")

                        if solver_type_attr and solver_type_attr.name() == "solver_type":
                            # Actually read solver type from metadata
                            try:
                                # Try to read the solver type value
                                solver_type_view = solver_type_attr.view()
                                if len(solver_type_view) > 0:
                                    solver_type = str(solver_type_view[0])
                                else:
                                    continue
                            except:
                                continue

                            if solver_type == "fem":
                                env_idx_attr = meta_attrs.find("env_idx")
                                entity_idx_attr = meta_attrs.find("entity_idx")

                                if env_idx_attr and entity_idx_attr:
                                    # Read string values and convert to int
                                    env_idx_str = str(env_idx_attr.view()[0])
                                    entity_idx_str = str(entity_idx_attr.view()[0])
                                    env_idx = int(env_idx_str)
                                    entity_idx = int(entity_idx_str)

                                    if entity_idx not in fem_geo_by_entity:
                                        fem_geo_by_entity[entity_idx] = {}

                                    proc_geo = geo
                                    if geo.instances().size() >= 1:
                                        proc_geo = merge(apply_transform(geo))
                                    pos = proc_geo.positions().view().reshape(-1, 3)
                                    fem_geo_by_entity[entity_idx][env_idx] = pos

                    except Exception as e:
                        # Skip this geometry if metadata reading fails
                        continue

        # Update FEM entities using filtered geometries
        for entity_idx, env_positions in fem_geo_by_entity.items():
            if entity_idx < len(self.fem_solver._entities):
                entity = self.fem_solver._entities[entity_idx]
                env_pos_list = []

                for env_idx in range(self.sim._B):
                    if env_idx in env_positions:
                        env_pos_list.append(env_positions[env_idx])
                    else:
                        # Fallback for missing environment
                        env_pos_list.append(np.zeros((0, 3)))

                if env_pos_list:
                    all_env_pos = np.stack(env_pos_list, axis=0)
                    entity.set_pos(0, all_env_pos)

    def _retrieve_rigid_states(self, f):
        """
        Handle rigid body IPC: Retrieve ABD transforms/affine matrices after IPC step
        """
        # IPC world advance/retrieve is handled at Scene level
        # Retrieve ABD transform matrices after IPC simulation

        if not hasattr(self, '_ipc_scene') or not hasattr(self.rigid_solver, 'list_env_mesh'):
            return

        from uipc import builtin, view
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot
        import numpy as np

        rigid_solver = self.rigid_solver
        visitor = SceneVisitor(self._ipc_scene)

        # Collect ABD geometries and their transforms using metadata
        abd_affine_by_geom = {}
        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() == 3:
                    try:
                        # Check if this is an ABD geometry using metadata
                        meta_attrs = geo.meta()
                        solver_type_attr = meta_attrs.find("solver_type")

                        if solver_type_attr and solver_type_attr.name() == "solver_type":
                            # Actually read solver type from metadata
                            try:
                                solver_type_view = solver_type_attr.view()
                                if len(solver_type_view) > 0:
                                    solver_type = str(solver_type_view[0])
                                else:
                                    continue
                            except:
                                continue

                            if solver_type == "rigid":
                                env_idx_attr = meta_attrs.find("env_idx")
                                link_idx_attr = meta_attrs.find("link_idx")

                                if env_idx_attr and link_idx_attr:
                                    # Read metadata values
                                    env_idx_str = str(env_idx_attr.view()[0])
                                    link_idx_str = str(link_idx_attr.view()[0])
                                    env_idx = int(env_idx_str)
                                    link_idx = int(link_idx_str)

                                    # Get current transform matrix from ABD object
                                    transforms = geo.transforms()
                                    if transforms.size() > 0:
                                        transform_matrix = view(transforms)[0]  # 4x4 affine matrix

                                        if link_idx not in abd_affine_by_geom:
                                            abd_affine_by_geom[link_idx] = {}
                                        abd_affine_by_geom[link_idx][env_idx] = transform_matrix.copy()

                    except Exception as e:
                        gs.logger.warning(f"Failed to retrieve ABD geometry transform: {e}")
                        continue

        # Store transforms for later access
        rigid_solver._abd_affines = abd_affine_by_geom

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        # IPC doesn't support gradients yet
        pass

    def reset(self, envs_idx=None):
        """Reset coupling state"""
        # IPC doesn't need special reset logic currently
        pass