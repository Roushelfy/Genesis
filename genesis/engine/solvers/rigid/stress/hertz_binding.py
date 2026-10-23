"""Bind radial friction memory to continuous material contact regions."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind, ErrorCode

from .contact import StressContactState
from .data import StressInfo
from .hertz import StressHertzState
from .hertz_friction import StressFrictionInput, StressFrictionState
from .hertz_recovery import StressHertzFrames
from .hertz_tangential import StressHertzShear
from .provenance import StressSourceState
from .surface import StressSurfaceInfo
from .surface_walk import func_projected_surface_walk


@dataclass(frozen=True)
class StressFrictionRegions:
    """Carry grouped material forces and pressure profiles into tangential recovery.

    Profile1 uses radial Hertz memory. Profile2 uses proportional edge shear.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    force: qd.Tensor
    source_count: qd.Tensor
    leader: qd.Tensor
    profile: qd.Tensor


@dataclass(frozen=True)
class StressFrictionBindingState:
    kind: ClassVar[DataKind] = DataKind.STATE

    geom: qd.Tensor
    pair: qd.Tensor
    center: qd.Tensor
    basis: qd.Tensor
    anchor: qd.Tensor
    radius: qd.Tensor
    is_lost: qd.Tensor


@qd.kernel
def kernel_friction_bind(
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressFrictionRegions,
    law: StressHertzState,
    frames: StressHertzFrames,
    binding: StressFrictionBindingState,
    inputs: StressFrictionInput,
    shear: StressHertzShear,
    input_status: qd.Tensor,
    input_is_valid: qd.Tensor,
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i_b in range(binding.is_lost.shape[0]):
        has_load = False
        for i_c in range(contacts.valid.shape[0]):
            has_load = has_load or (contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0)
        if not has_load:
            binding.is_lost[i_b] = False
        elif input_status[i_b] != 0 or not input_is_valid[i_b]:
            binding.is_lost[i_b] = True
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        i_p = i_c * contacts.valid.shape[1] + i_b
        inputs.radius[i_p], inputs.normal[i_p], inputs.friction[i_p], inputs.angle[i_p] = 0.0, 0.0, 0.0, 0.0
        inputs.force[i_p], shear.force[i_p] = qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2)
        inputs.previous_index[i_p], inputs.is_reset[i_p], inputs.is_valid[i_p] = -1, True, True
        shear.friction[i_p] = 0.0
        if regions.source_count[i_c, i_b] > 0:
            direction = law.direction[i_p]
            basis = frames.basis[i_p] @ qd.Matrix(
                [[direction[0], -direction[1], 0.0], [direction[1], direction[0], 0.0], [0.0, 0.0, 1.0]]
            )
            tangent = basis.transpose() @ regions.force[i_c, i_b]
            friction = contacts.friction[i_c, i_b]
            is_compatible = True
            for i_s in range(contacts.valid.shape[0]):
                if contacts.valid[i_s, i_b] and regions.leader[i_s, i_b] == i_c:
                    is_compatible = is_compatible and qd.abs(contacts.friction[i_s, i_b] - friction) <= (
                        64 * epsilon * qd.max(1.0, friction)
                    )
            shear.force[i_p] = qd.Vector([tangent[0], tangent[1]])
            shear.friction[i_p] = friction
            inputs.is_valid[i_p] = is_compatible and not binding.is_lost[i_b] and law.status[i_p] == 0
            if regions.profile[i_c, i_b] == 1:
                inputs.radius[i_p], inputs.normal[i_p], inputs.friction[i_p] = (
                    law.axes[i_p][0],
                    law.force[i_p],
                    friction,
                )
                inputs.force[i_p] = shear.force[i_p]
                i_previous, n_candidates = -1, 0
                for i_s in range(contacts.valid.shape[0]):
                    i_old = i_s * contacts.valid.shape[1] + i_b
                    if (
                        binding.radius[i_old] > 0
                        and binding.geom[i_old] == sources.geom[i_c, i_b]
                        and binding.pair[i_old] == sources.pair[i_c, i_b]
                        and binding.basis[i_old][:, 2].dot(basis[:, 2]) > 0
                        and (binding.center[i_old] - frames.center[i_p]).norm()
                        <= binding.radius[i_old] + law.axes[i_p][0]
                    ):
                        i_previous, n_candidates = i_old, n_candidates + 1
                if n_candidates == 1:
                    anchor, barycentric, position = func_projected_surface_walk(
                        binding.anchor[i_previous],
                        binding.center[i_previous],
                        frames.center[i_p] - binding.center[i_previous],
                        -basis[:, 2],
                        info,
                        surface,
                        epsilon,
                    )
                    is_continuous = anchor >= 0 and (position - frames.center[i_p]).norm() <= (
                        1e-10 * qd.max(binding.radius[i_previous], law.axes[i_p][0])
                    )
                    inputs.is_valid[i_p] = inputs.is_valid[i_p] and is_continuous
                    inputs.previous_index[i_p], inputs.is_reset[i_p] = i_previous, False
                    # Minimal normal rotation transports the old material tangent.
                    previous_normal, normal = binding.basis[i_previous][:, 2], basis[:, 2]
                    cross = previous_normal.cross(normal)
                    previous_tangent = binding.basis[i_previous][:, 0]
                    transported = (
                        previous_tangent
                        + cross.cross(previous_tangent)
                        + cross.cross(cross.cross(previous_tangent)) / (1 + previous_normal.dot(normal))
                    )
                    inputs.angle[i_p] = qd.atan2(transported.dot(basis[:, 1]), transported.dot(basis[:, 0]))
                elif n_candidates > 1:
                    inputs.is_valid[i_p] = False
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        i_p = i_c * contacts.valid.shape[1] + i_b
        i_previous = inputs.previous_index[i_p]
        if i_previous >= 0:
            n_matches = 0
            for i_s in range(contacts.valid.shape[0]):
                i_other = i_s * contacts.valid.shape[1] + i_b
                n_matches += gs.qd_int(inputs.previous_index[i_other] == i_previous)
            if n_matches != 1:
                inputs.is_valid[i_p] = False


@qd.kernel
def kernel_friction_commit(
    contacts: StressContactState,
    sources: StressSourceState,
    regions: StressFrictionRegions,
    law: StressHertzState,
    frames: StressHertzFrames,
    binding: StressFrictionBindingState,
    inputs: StressFrictionInput,
    friction: StressFrictionState,
    input_status: qd.Tensor,
    input_is_valid: qd.Tensor,
    errno: qd.Tensor,
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        i_p = i_c * contacts.valid.shape[1] + i_b
        if regions.source_count[i_c, i_b] > 0 and friction.status[i_p] > 1:
            contacts.status[i_c, i_b] = 55
            qd.atomic_or(input_status[i_b], 16)
    for i_b in range(binding.is_lost.shape[0]):
        if input_status[i_b] != 0:
            input_is_valid[i_b], binding.is_lost[i_b] = False, True
            qd.atomic_or(errno[i_b], ErrorCode.INVALID_STRESS_LOAD)
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        i_p = i_c * contacts.valid.shape[1] + i_b
        binding.radius[i_p], binding.anchor[i_p] = 0.0, -1
        binding.geom[i_p], binding.pair[i_p] = -1, -1
        if regions.profile[i_c, i_b] == 1 and inputs.is_valid[i_p] and friction.status[i_p] == 0:
            binding.radius[i_p], binding.anchor[i_p] = law.axes[i_p][0], frames.anchor[i_p]
            binding.geom[i_p], binding.pair[i_p] = sources.geom[i_c, i_b], sources.pair[i_c, i_b]
            binding.center[i_p] = frames.center[i_p]
            direction = law.direction[i_p]
            binding.basis[i_p] = frames.basis[i_p] @ qd.Matrix(
                [[direction[0], -direction[1], 0.0], [direction[1], direction[0], 0.0], [0.0, 0.0, 1.0]]
            )
