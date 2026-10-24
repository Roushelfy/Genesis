from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V, maybe_shape, of_kind


@dataclass(frozen=True)
class StressInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    vertices: qd.Tensor
    elements: qd.Tensor
    surface_nodes: qd.Tensor
    surface_owners: qd.Tensor
    surface_opposite_vertices: qd.Tensor
    surface_neighbors: qd.Tensor
    edges: qd.Tensor
    gradients: qd.Tensor
    corner_gradients: qd.Tensor
    corner_nodes: qd.Tensor
    volumes: qd.Tensor
    row_start: qd.Tensor
    columns: qd.Tensor
    assembly_start: qd.Tensor
    assembly_idx: qd.Tensor
    stiffness: qd.Tensor
    mass: qd.Tensor
    modes: qd.Tensor
    mass_modes: qd.Tensor
    relief: qd.Tensor
    centrifugal: qd.Tensor
    centrifugal_wrench: qd.Tensor
    gram: qd.Tensor
    gram_inverse: qd.Tensor
    pins: qd.Tensor
    is_free: qd.Tensor
    diagonal_inverse: qd.Tensor
    scalar_diagonal_inverse: qd.Tensor
    mass_properties: qd.Tensor
    load_nodes: qd.Tensor
    load_offsets: qd.Tensor


@dataclass(frozen=True)
class StressState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    force: qd.Tensor
    rhs: qd.Tensor
    displacement: qd.Tensor = of_kind(DataKind.WARMSTART)  # noqa: RUF009 - typed array metadata
    residual: qd.Tensor
    direction: qd.Tensor
    product: qd.Tensor
    preconditioned: qd.Tensor
    boundary_columns: qd.Tensor
    boundary_count: qd.Tensor
    wrench: qd.Tensor
    rhs_norm_squared: qd.Tensor
    residual_norm_squared: qd.Tensor
    residual_preconditioned: qd.Tensor
    next_residual_preconditioned: qd.Tensor
    direction_product: qd.Tensor
    active: qd.Tensor
    iterations: qd.Tensor
    peak: qd.Tensor
    step_peak: qd.Tensor = of_kind(DataKind.DERIVED)  # noqa: RUF009 - typed array metadata
    step_valid: qd.Tensor = of_kind(DataKind.DERIVED)  # noqa: RUF009 - typed array metadata
    stress_tensor: qd.Tensor = of_kind(DataKind.DERIVED)  # noqa: RUF009 - typed array metadata
    von_mises: qd.Tensor = of_kind(DataKind.DERIVED)  # noqa: RUF009 - typed array metadata
    membrane_resultant: qd.Tensor = of_kind(DataKind.DERIVED)  # noqa: RUF009 - typed array metadata
    bending_resultant: qd.Tensor = of_kind(DataKind.DERIVED)  # noqa: RUF009 - typed array metadata
    transverse_resultant: qd.Tensor = of_kind(DataKind.DERIVED)  # noqa: RUF009 - typed array metadata
    valid: qd.Tensor
    corrections: qd.Tensor
    fallbacks: qd.Tensor
    invalid_steps: qd.Tensor
    invalid_reported: qd.Tensor


def allocate_stress_info(
    n_nodes: int, n_elements: int, n_faces: int, n_entries: int, n_assembly: int, n_load_nodes: int = 0
) -> StressInfo:
    """Allocate the shared three-component block operator storage."""
    return StressInfo(
        vertices=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes,)),
        elements=V(dtype=gs.qd_int, shape=maybe_shape((n_elements, 10), n_elements > 0)),
        surface_nodes=V(dtype=gs.qd_int, shape=maybe_shape((n_faces, 6), n_faces > 0)),
        surface_owners=V(dtype=gs.qd_int, shape=maybe_shape((n_faces,), n_faces > 0)),
        surface_opposite_vertices=V(dtype=gs.qd_int, shape=maybe_shape((n_faces,), n_faces > 0)),
        surface_neighbors=V(dtype=gs.qd_int, shape=maybe_shape((n_faces, 3), n_faces > 0)),
        edges=V(dtype=gs.qd_int, shape=(6, 2)),
        gradients=V_MAT(4, 3, dtype=gs.qd_float, shape=(n_elements,)),
        corner_gradients=V_VEC(3, dtype=gs.qd_float, shape=(n_elements, 4, 7)),
        corner_nodes=V(dtype=gs.qd_int, shape=maybe_shape((n_elements, 4, 7), n_elements > 0)),
        volumes=V(dtype=gs.qd_float, shape=maybe_shape((n_elements,), n_elements > 0)),
        row_start=V(dtype=gs.qd_int, shape=(n_nodes + 1,)),
        columns=V(dtype=gs.qd_int, shape=maybe_shape((n_entries,), n_entries > 0)),
        assembly_start=V(dtype=gs.qd_int, shape=(n_entries + 1,)),
        assembly_idx=V(dtype=gs.qd_int, shape=maybe_shape((n_assembly,), n_assembly > 0)),
        stiffness=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_entries,)),
        mass=V(dtype=gs.qd_float, shape=maybe_shape((n_entries,), n_entries > 0)),
        modes=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
        mass_modes=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
        relief=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
        centrifugal=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
        centrifugal_wrench=V_MAT(6, 6, dtype=gs.qd_float, shape=()),
        gram=V_MAT(6, 6, dtype=gs.qd_float, shape=()),
        gram_inverse=V_MAT(6, 6, dtype=gs.qd_float, shape=()),
        pins=V(dtype=gs.qd_int, shape=(6,)),
        is_free=V_VEC(3, dtype=gs.qd_int, shape=(n_nodes,)),
        diagonal_inverse=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_nodes,)),
        scalar_diagonal_inverse=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes,)),
        mass_properties=V(dtype=gs.qd_float, shape=(4,)),
        load_nodes=V_VEC(2, dtype=gs.qd_int, shape=(n_load_nodes,)),
        load_offsets=V_MAT(2, 3, dtype=gs.qd_float, shape=(n_load_nodes,)),
    )


@qd.func
def func_scatter_load(
    i_n: int, i_b: int, force: qd.types.vector(3), stress_state: StressState, stress_info: StressInfo
):
    if stress_info.load_nodes.shape[0] > 0:
        # Consistent virtual work on the physical surface retains both surface eccentricity and supplied wrench.
        for i_parent in qd.static(range(2)):
            j_n = stress_info.load_nodes[i_n][i_parent]
            moment = 0.5 * stress_info.load_offsets[i_n][i_parent, :].cross(force)
            for a in qd.static(range(3)):
                qd.atomic_add(stress_state.force[2 * j_n, i_b][a], 0.5 * force[a])
                qd.atomic_add(stress_state.force[2 * j_n + 1, i_b][a], moment[a])
    else:
        for a in qd.static(range(3)):
            qd.atomic_add(stress_state.force[i_n, i_b][a], force[a])
