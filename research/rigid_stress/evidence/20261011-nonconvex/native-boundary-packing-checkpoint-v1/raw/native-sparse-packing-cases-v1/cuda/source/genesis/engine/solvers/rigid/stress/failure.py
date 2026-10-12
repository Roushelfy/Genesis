"""Device-side first-contact diagnostics retained until the rigid load error is cleared."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, ErrorCode, V

from .contact import StressContactState
from .provenance import StressSourceState


@dataclass(frozen=True)
class StressContactFailure:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    contact: qd.Tensor
    status: qd.Tensor
    radius: qd.Tensor
    friction: qd.Tensor
    position: qd.Tensor
    anchor: qd.Tensor
    force: qd.Tensor
    free_moment: qd.Tensor


def create_contact_failure(n_envs: int) -> StressContactFailure:
    failure = StressContactFailure(
        contact=V(dtype=gs.qd_int, shape=(n_envs,)),
        status=V(dtype=gs.qd_int, shape=(n_envs,)),
        radius=V(dtype=gs.qd_float, shape=(n_envs,)),
        friction=V(dtype=gs.qd_float, shape=(n_envs,)),
        position=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        anchor=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        force=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        free_moment=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
    )
    failure.contact.fill(-1)
    failure.status.fill(0)
    return failure


@qd.kernel
def kernel_capture_contact_failure(
    contacts: StressContactState, sources: StressSourceState, failure: StressContactFailure, errno: qd.Tensor
):
    """Capture one deterministic raw contact per failed environment, without host synchronization.

    Ordinary successful environments do not scan contact capacity. A subsequent recovery cannot overwrite the
    first record while INVALID_STRESS_LOAD remains set. Clearing that error, including a selected-environment
    reset, invalidates the record on the next recovery. The diagnostic does not change loads or error flags.
    """
    for i_b in range(failure.status.shape[0]):
        if errno[i_b] & ErrorCode.INVALID_STRESS_LOAD:
            if failure.status[i_b] == 0:
                for i_c in range(contacts.valid.shape[0]):
                    if failure.status[i_b] == 0 and contacts.valid[i_c, i_b] and contacts.status[i_c, i_b] != 0:
                        failure.contact[i_b] = i_c
                        failure.status[i_b] = contacts.status[i_c, i_b]
                        failure.radius[i_b] = contacts.radius[i_c, i_b]
                        failure.friction[i_b] = contacts.friction[i_c, i_b]
                        failure.position[i_b] = contacts.position[i_c, i_b]
                        failure.anchor[i_b] = contacts.center[i_c, i_b]
                        failure.force[i_b] = contacts.force[i_c, i_b]
                        failure.free_moment[i_b] = sources.free_moment[i_c, i_b]
        else:
            failure.contact[i_b] = -1
            failure.status[i_b] = 0
            failure.radius[i_b] = 0.0
            failure.friction[i_b] = 0.0
            failure.position[i_b] = qd.Vector.zero(gs.qd_float, 3)
            failure.anchor[i_b] = qd.Vector.zero(gs.qd_float, 3)
            failure.force[i_b] = qd.Vector.zero(gs.qd_float, 3)
            failure.free_moment[i_b] = qd.Vector.zero(gs.qd_float, 3)
