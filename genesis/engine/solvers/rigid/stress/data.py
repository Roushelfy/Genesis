from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V, of_kind


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
    valid: qd.Tensor
    corrections: qd.Tensor
    fallbacks: qd.Tensor
    invalid_steps: qd.Tensor
    invalid_reported: qd.Tensor


def allocate_stress_info(n_nodes: int, n_elements: int, n_faces: int, n_entries: int, n_assembly: int) -> StressInfo:
    """Allocate the shared three-component block operator storage."""
    return StressInfo(
        vertices=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes,)),
        elements=V(dtype=gs.qd_int, shape=(n_elements, 10)),
        surface_nodes=V(dtype=gs.qd_int, shape=(n_faces, 6)),
        surface_owners=V(dtype=gs.qd_int, shape=(n_faces,)),
        surface_opposite_vertices=V(dtype=gs.qd_int, shape=(n_faces,)),
        surface_neighbors=V(dtype=gs.qd_int, shape=(n_faces, 3)),
        edges=V(dtype=gs.qd_int, shape=(6, 2)),
        gradients=V_MAT(4, 3, dtype=gs.qd_float, shape=(n_elements,)),
        corner_gradients=V_VEC(3, dtype=gs.qd_float, shape=(n_elements, 4, 7)),
        corner_nodes=V(dtype=gs.qd_int, shape=(n_elements, 4, 7)),
        volumes=V(dtype=gs.qd_float, shape=(n_elements,)),
        row_start=V(dtype=gs.qd_int, shape=(n_nodes + 1,)),
        columns=V(dtype=gs.qd_int, shape=(n_entries,)),
        assembly_start=V(dtype=gs.qd_int, shape=(n_entries + 1,)),
        assembly_idx=V(dtype=gs.qd_int, shape=(n_assembly,)),
        stiffness=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_entries,)),
        mass=V(dtype=gs.qd_float, shape=(n_entries,)),
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
    )
