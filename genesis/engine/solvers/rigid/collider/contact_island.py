from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.collider import Collider
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


# ---- Standalone kernels ----
# These must be standalone (not class methods) because the Quadrants ndarray-mode compiler
# cannot resolve nested frozen-dataclass attribute access (e.g. collider_state.contact_data.link_a)
# inside @qd.data_oriented class methods.
#
# The ContactIsland object (qd.template()) provides qd.field() arrays that the standalone kernels
# read/write. Frozen dataclass structs (ColliderState, LinksInfo, etc.) use their typed annotations.


@qd.func
def _func_add_edge(
    link_a,
    link_b,
    i_b,
    ci: qd.template(),
    links_info: array_class.LinksInfo,
    static_cfg: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    link_a_maybe_batch = [link_a, i_b] if qd.static(static_cfg.batch_links_info) else link_a
    link_b_maybe_batch = [link_b, i_b] if qd.static(static_cfg.batch_links_info) else link_b

    ea = links_info.entity_idx[link_a_maybe_batch]
    eb = links_info.entity_idx[link_b_maybe_batch]

    # fill in collider-info edges with indices to connected entities.
    n_edge = ci.n_edges[i_b]
    max_edges = ci.ci_edges.shape[0]
    if n_edge < max_edges:
        ci.ci_edges[n_edge, 0, i_b] = ea
        ci.ci_edges[n_edge, 1, i_b] = eb
        ci.n_edges[i_b] = n_edge + 1
        # update num edges per entity - only when edge is actually stored
        ci.entity_edge_n[ea, i_b] = ci.entity_edge_n[ea, i_b] + 1
        ci.entity_edge_n[eb, i_b] = ci.entity_edge_n[eb, i_b] + 1
    else:
        # Signal buffer overflow via errno bit 4 (0b00010000)
        errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_HIBERNATION_ISLANDS


@qd.kernel
def _kernel_add_contact_edges(
    collider_state: array_class.ColliderState,
    ci: qd.template(),
    links_info: array_class.LinksInfo,
    static_cfg: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    _B = ci.n_edges.shape[0]
    qd.loop_config(serialize=static_cfg.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_col in range(collider_state.n_contacts[i_b]):
            # get links indices of the impact
            link_a = collider_state.contact_data.link_a[i_col, i_b]
            link_b = collider_state.contact_data.link_b[i_col, i_b]
            _func_add_edge(link_a, link_b, i_b, ci, links_info, static_cfg, errno)


@qd.kernel
def _kernel_add_hibernated_edges(
    ci: qd.template(),
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    static_cfg: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    _B = ci.n_edges.shape[0]
    n_entities = ci.entity_edge_n.shape[0]
    qd.loop_config(serialize=static_cfg.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_e in range(n_entities):
            next_entity_idx = ci.entity_idx_to_next_entity_idx_in_hibernated_island[i_e, i_b]
            # Guard: validate next_entity_idx is within valid bounds
            if 0 <= next_entity_idx < n_entities and next_entity_idx != i_e:
                any_link_a = entities_info.link_start[i_e]
                any_link_b = entities_info.link_start[next_entity_idx]
                _func_add_edge(any_link_a, any_link_b, i_b, ci, links_info, static_cfg, errno)


@qd.kernel
def _kernel_preprocess_and_map_edges(
    ci: qd.template(),
    static_cfg: qd.template(),
    errno: array_class.V_ANNOTATION,
    n_entities: int,
):
    _B = ci.n_edges.shape[0]
    qd.loop_config(serialize=static_cfg.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        entity_list_start = 0
        for i in range(n_entities):
            ci.entity_edge_start[i, i_b] = entity_list_start
            ci.entity_edge_curr[i, i_b] = entity_list_start
            entity_list_start = entity_list_start + ci.entity_edge_n[i, i_b]

        # Invariant check: ensure total half-edges don't exceed edge_id buffer
        if entity_list_start > ci.edge_id.shape[0]:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_HIBERNATION_ISLANDS

        # process added collider-info edges
        for i in range(ci.n_edges[i_b]):
            ea = ci.ci_edges[i, 0, i_b]
            eb = ci.ci_edges[i, 1, i_b]

            # map entity's half-edge index to edge index.
            ci.edge_id[ci.entity_edge_curr[ea, i_b], i_b] = i
            ci.edge_id[ci.entity_edge_curr[eb, i_b], i_b] = i

            ci.entity_edge_curr[ea, i_b] = ci.entity_edge_curr[ea, i_b] + 1
            ci.entity_edge_curr[eb, i_b] = ci.entity_edge_curr[eb, i_b] + 1


@qd.kernel
def _kernel_construct_islands(
    ci: qd.template(),
    entities_info: array_class.EntitiesInfo,
    static_cfg: qd.template(),
):
    """
    This assigns entities to islands, by setting their entity_island[entity_idx, batch_idx] = island_idx.
    """
    _B = ci.n_edges.shape[0]
    n_entities = ci.entity_edge_n.shape[0]
    qd.loop_config(serialize=static_cfg.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_v in range(n_entities):
            # only create islands for entities with collisions and with dofs
            if ci.entity_edge_n[i_v, i_b] > 0 and entities_info.n_dofs[i_v] > 0:
                if ci.entity_island[i_v, i_b] != -1:
                    continue
                ci.n_stack[i_b] = 0
                ci.stack[ci.n_stack[i_b], i_b] = i_v
                ci.n_stack[i_b] = ci.n_stack[i_b] + 1
                ci.entity_island[i_v, i_b] = ci.n_islands[i_b]
                # FIXME: Add proper mechanism to detection overflow in Quadrants-scope
                # but raise exception in Python-scope

                while ci.n_stack[i_b] > 0:
                    ci.n_stack[i_b] = ci.n_stack[i_b] - 1
                    v = ci.stack[ci.n_stack[i_b], i_b]

                    for i_edge in range(ci.entity_edge_n[v, i_b]):
                        # half-edge index
                        _id = ci.entity_edge_start[v, i_b] + i_edge
                        # edge index
                        edge = ci.edge_id[_id, i_b]
                        # other entity index, connected by edge
                        next_v = ci.ci_edges[edge, 0, i_b]
                        if next_v == v:
                            next_v = ci.ci_edges[edge, 1, i_b]

                        if (
                            entities_info.n_dofs[next_v] > 0 and next_v != v and ci.entity_island[next_v, i_b] == -1
                        ):  # 2nd condition must not happen ?
                            ci.stack[ci.n_stack[i_b], i_b] = next_v
                            ci.n_stack[i_b] = ci.n_stack[i_b] + 1
                            ci.entity_island[next_v, i_b] = ci.n_islands[i_b]
                            # FIXME: Add proper mechanism to detection overflow in Quadrants-scope
                            # but raise exception in Python-scope

                ci.n_islands[i_b] = ci.n_islands[i_b] + 1

    # create single-entity islands for entities without collisions
    if qd.static(static_cfg.enable_joint_limit):
        qd.loop_config(serialize=static_cfg.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_v in range(n_entities):
                if entities_info.n_dofs[i_v] > 0 and ci.entity_island[i_v, i_b] == -1:
                    ci.entity_island[i_v, i_b] = ci.n_islands[i_b]
                    ci.n_islands[i_b] = ci.n_islands[i_b] + 1


@qd.kernel
def _kernel_postprocess_islands(
    collider_state: array_class.ColliderState,
    ci: qd.template(),
    links_info: array_class.LinksInfo,
    entities_state: array_class.EntitiesState,
    static_cfg: qd.template(),
    n_entities: int,
):
    _B = ci.n_edges.shape[0]
    qd.loop_config(serialize=static_cfg.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_col in range(collider_state.n_contacts[i_b]):
            # get links indices of the impact
            link_a = collider_state.contact_data.link_a[i_col, i_b]
            link_b = collider_state.contact_data.link_b[i_col, i_b]
            link_a_maybe_batch = [link_a, i_b] if qd.static(static_cfg.batch_links_info) else link_a
            link_b_maybe_batch = [link_b, i_b] if qd.static(static_cfg.batch_links_info) else link_b

            ea = links_info.entity_idx[link_a_maybe_batch]
            eb = links_info.entity_idx[link_b_maybe_batch]

            island_a = ci.entity_island[ea, i_b]
            island_b = ci.entity_island[eb, i_b]

            # handle collisions between dynamic and fixed entities (island_idx == -1)
            island = island_a
            if island_a == -1:
                island = island_b

            ci.island_col_n[island, i_b] = ci.island_col_n[island, i_b] + 1
            ci.constraint_list[i_col, i_b] = island

        constraint_list_start = 0
        for i in range(ci.n_islands[i_b]):
            ci.island_col_start[i, i_b] = constraint_list_start
            constraint_list_start = constraint_list_start + ci.island_col_n[i, i_b]
            ci.island_col_curr[i, i_b] = ci.island_col_start[i, i_b]

            ci.island_hibernated[i, i_b] = 1

        for i_col in range(collider_state.n_contacts[i_b]):
            island = ci.constraint_list[i_col, i_b]
            ci.constraint_id[ci.island_col_curr[island, i_b], i_b] = i_col
            ci.island_col_curr[island, i_b] = ci.island_col_curr[island, i_b] + 1

        # island_entity
        for i in range(n_entities):
            if ci.entity_island[i, i_b] >= 0:
                ci.island_entity_n[ci.entity_island[i, i_b], i_b] = (
                    ci.island_entity_n[ci.entity_island[i, i_b], i_b] + 1
                )
                if entities_state.hibernated[i, i_b] == 0:
                    ci.island_hibernated[ci.entity_island[i, i_b], i_b] = 0

        entity_list_start = 0
        for i in range(ci.n_islands[i_b]):
            ci.island_entity_start[i, i_b] = entity_list_start
            ci.island_entity_curr[i, i_b] = ci.island_entity_start[i, i_b]
            entity_list_start = entity_list_start + ci.island_entity_n[i, i_b]

        for i in range(n_entities):
            island = ci.entity_island[i, i_b]
            if island >= 0:
                ci.entity_id[ci.island_entity_curr[island, i_b], i_b] = i
                ci.island_entity_curr[island, i_b] = ci.island_entity_curr[island, i_b] + 1


# ---- ContactIsland class ----
# @qd.data_oriented so that solver_island.py (also @qd.data_oriented) can access
# the qd.field() attributes in its own class-method kernels via self.contact_island.<field>[...].


@qd.data_oriented
class ContactIsland:
    def __init__(self, collider: "Collider"):
        self.solver: "RigidSolver" = collider._solver
        self.collider: "Collider" = collider

        _B = self.solver._B
        max_contact_pairs = max(collider._collider_info.max_contact_pairs[None], 1)
        n_entities = max(self.solver.n_entities, 1)
        max_hibernation_edges = n_entities if self.solver._use_hibernation else 0
        max_edges = max_contact_pairs + max_hibernation_edges

        # All fields as qd.field() — accessible from both:
        # 1. Standalone kernels (via qd.template() parameter pointing to this object)
        # 2. solver_island.py class-method kernels (via self.contact_island.<field>)
        self.ci_edges = qd.field(dtype=gs.qd_int, shape=(max_edges, 2, _B))
        self.edge_id = qd.field(dtype=gs.qd_int, shape=(max_edges * 2, _B))
        self.constraint_list = qd.field(dtype=gs.qd_int, shape=(max_contact_pairs, _B))
        self.constraint_id = qd.field(dtype=gs.qd_int, shape=(max_contact_pairs * 2, _B))
        self.island_hibernated = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.entity_id = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.n_edges = qd.field(dtype=gs.qd_int, shape=(_B,))
        self.n_islands = qd.field(dtype=gs.qd_int, shape=(_B,))
        self.n_stack = qd.field(dtype=gs.qd_int, shape=(_B,))
        self.entity_island = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.stack = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.entity_idx_to_next_entity_idx_in_hibernated_island = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))

        # per-entity range of half-edges
        self.entity_edge_n = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.entity_edge_start = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.entity_edge_curr = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))

        # per-island collision range
        self.island_col_n = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.island_col_start = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.island_col_curr = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))

        # per-island entity range
        self.island_entity_n = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.island_entity_start = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))
        self.island_entity_curr = qd.field(dtype=gs.qd_int, shape=(n_entities, _B))

        # Also create V()-based struct for external standalone kernels (ABD, forward_kinematics)
        # that take StructContactIslandState as a typed parameter.
        self.contact_island_state = array_class.get_contact_island_state(self.solver, collider)

        self.entity_idx_to_next_entity_idx_in_hibernated_island.fill(-1)
        self.contact_island_state.entity_idx_to_next_entity_idx_in_hibernated_island.fill(-1)

    def construct(self):
        solver = self.solver
        collider = self.collider
        cfg = solver._static_rigid_sim_config
        errno = solver._errno

        _kernel_clear(self, cfg, solver.n_entities)
        _kernel_add_contact_edges(collider._collider_state, self, solver.links_info, cfg, errno)
        _kernel_add_hibernated_edges(self, solver.entities_info, solver.links_info, cfg, errno)
        _kernel_preprocess_and_map_edges(self, cfg, errno, solver.n_entities)
        _kernel_construct_islands(self, solver.entities_info, cfg)
        _kernel_postprocess_islands(
            collider._collider_state, self, solver.links_info, solver.entities_state, cfg, solver.n_entities
        )

        # Sync qd.field() data to V()-based struct for external standalone kernels
        # (ABD, forward_kinematics) that take StructContactIslandState.
        self._sync_to_struct()

    def _sync_to_struct(self):
        """Copy qd.field() data to V()-based struct via CPU numpy roundtrip."""
        cis = self.contact_island_state
        for name in (
            "n_edges",
            "n_islands",
            "entity_edge_n",
            "entity_edge_start",
            "entity_edge_curr",
            "island_col_n",
            "island_col_start",
            "island_col_curr",
            "island_entity_n",
            "island_entity_start",
            "island_entity_curr",
            "island_hibernated",
            "entity_island",
            "entity_id",
            "ci_edges",
            "edge_id",
            "constraint_list",
            "constraint_id",
            "n_stack",
            "stack",
            "entity_idx_to_next_entity_idx_in_hibernated_island",
        ):
            getattr(cis, name).from_numpy(getattr(self, name).to_numpy())


@qd.kernel
def _kernel_clear(
    ci: qd.template(),
    static_cfg: qd.template(),
    n_entities: int,
):
    _B = ci.n_edges.shape[0]
    qd.loop_config(serialize=static_cfg.para_level < gs.PARA_LEVEL.ALL)
    for i_e, i_b in qd.ndrange(n_entities, _B):
        ci.entity_edge_n[i_e, i_b] = 0
        ci.island_col_n[i_e, i_b] = 0
        ci.island_entity_n[i_e, i_b] = 0
        ci.entity_island[i_e, i_b] = -1

    qd.loop_config(serialize=static_cfg.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        ci.n_edges[i_b] = 0
        ci.n_islands[i_b] = 0
