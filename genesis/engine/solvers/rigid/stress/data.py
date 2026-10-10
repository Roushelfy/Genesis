from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

from genesis.utils.array_class import DataKind


@dataclass(frozen=True)
class StressInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    vertices: qd.Tensor
    elements: qd.Tensor
    surface_nodes: qd.Tensor
    edges: qd.Tensor
    gradients: qd.Tensor
    volumes: qd.Tensor
    row_start: qd.Tensor
    columns: qd.Tensor
    element_entries: qd.Tensor
    stiffness: qd.Tensor
    mass: qd.Tensor
    modes: qd.Tensor
    mass_modes: qd.Tensor
    centrifugal: qd.Tensor
    gram: qd.Tensor
    gram_inverse: qd.Tensor
    pins: qd.Tensor
    is_free: qd.Tensor
    diagonal_inverse: qd.Tensor
    mass_properties: qd.Tensor


@dataclass(frozen=True)
class StressState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    force: qd.Tensor
    rhs: qd.Tensor
    displacement: qd.Tensor
    residual: qd.Tensor
    direction: qd.Tensor
    product: qd.Tensor
    preconditioned: qd.Tensor
    wrench: qd.Tensor
    rhs_norm_squared: qd.Tensor
    residual_norm_squared: qd.Tensor
    residual_preconditioned: qd.Tensor
    next_residual_preconditioned: qd.Tensor
    direction_product: qd.Tensor
    active: qd.Tensor
    iterations: qd.Tensor
    peak: qd.Tensor
    step_peak: qd.Tensor
    valid: qd.Tensor
