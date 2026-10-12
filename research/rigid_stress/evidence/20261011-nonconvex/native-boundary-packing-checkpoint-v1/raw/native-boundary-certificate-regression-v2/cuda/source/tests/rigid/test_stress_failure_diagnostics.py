from types import SimpleNamespace

import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.failure import create_contact_failure, kernel_capture_contact_failure
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import ErrorCode, V, iter_data
from genesis.utils.misc import qd_to_numpy
from tests.utils.assertions import assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
def test_first_failed_contact_survives_working_buffers_and_selected_error_reset():
    """A delayed scene exception must retain its first load, even after contact death and other environment resets."""
    contacts, sources, failure = create_contacts(3, 4, 0.006), create_sources(3, 4), create_contact_failure(4)
    errno = V(dtype=gs.qd_int, shape=(4,))
    error = int(ErrorCode.INVALID_STRESS_LOAD)
    errno.from_numpy(np.array([error, error, 0, error], dtype=np.int32))
    contacts.valid.from_numpy(
        np.array([[False, False, False, False], [True, True, True, False], [True, True, True, False]])
    )
    contacts.status.from_numpy(np.array([[0, 0, 0, 34], [0, 36, 0, 34], [34, 34, 0, 34]], dtype=np.int32))
    contacts.radius.from_numpy(np.array([[0.006] * 4, [0.005] * 4, [0.004] * 4]))
    contacts.friction.from_numpy(np.array([[0.5] * 4, [0.85] * 4, [1.2] * 4]))
    for field, scale in (
        (contacts.position, 1.0),
        (contacts.center, 2.0),
        (contacts.force, 3.0),
        (sources.free_moment, 4.0),
    ):
        field.from_numpy(scale * np.arange(36).reshape(3, 4, 3))
    kernel_capture_contact_failure(contacts, sources, failure, errno)
    assert_equal(qd_to_numpy(failure.contact), [2, 1, -1, -1])
    assert_equal(qd_to_numpy(failure.status), [34, 36, 0, 0])
    snapshot = {item.name: qd_to_numpy(item.value, transpose=True, copy=True) for item in iter_data(failure, "failure")}
    # Simulate later recovery/contact death and reuse of the same scratch slots.
    contacts.valid.fill(False)
    contacts.status.fill(0)
    contacts.radius.fill(0.1)
    contacts.friction.fill(0.01)
    contacts.force.fill(99.0)
    sources.free_moment.fill(99.0)
    kernel_capture_contact_failure(contacts, sources, failure, errno)
    for item in iter_data(failure, "failure"):
        assert_equal(qd_to_numpy(item.value, transpose=True), snapshot[item.name])
    recovery = RigidStressRecovery.__new__(RigidStressRecovery)
    recovery.links = [
        SimpleNamespace(link=SimpleNamespace(idx=12), intrinsic=SimpleNamespace(failure=failure), contacts=contacts)
    ]
    message = recovery.describe_load_failures()
    assert "env=0, contact=2, status=34" in message and "mu=1.2" in message
    assert "env=1, contact=1, status=36" in message and "mu=0.85" in message
    assert "No failed contact remains" not in message
    assert "env=2" not in message and "env=3" not in message
    # Clearing one environment's rigid error leaves the other original diagnostic byte-identical.
    errno.from_numpy(np.array([0, error, 0, error], dtype=np.int32))
    kernel_capture_contact_failure(contacts, sources, failure, errno)
    for item in iter_data(failure, "failure"):
        actual = qd_to_numpy(item.value, transpose=True)
        assert_equal(actual[1], snapshot[item.name][1])
        assert_equal(actual[0], -1 if item.name == "failure.contact" else 0)
    # A different new failure can now populate env0; the original env1 remains isolated.
    valid, status = np.zeros((3, 4), dtype=bool), np.zeros((3, 4), dtype=np.int32)
    valid[0, 0], status[0, 0] = True, 35
    contacts.valid.from_numpy(valid)
    contacts.status.from_numpy(status)
    errno.from_numpy(np.array([error, error, 0, error], dtype=np.int32))
    kernel_capture_contact_failure(contacts, sources, failure, errno)
    assert_equal(qd_to_numpy(failure.contact), [0, 1, -1, -1])
    assert_equal(qd_to_numpy(failure.status), [35, 36, 0, 0])
    assert_equal(qd_to_numpy(failure.friction)[:2], [0.01, 0.85])
    assert_equal(qd_to_numpy(errno), [error, error, 0, error])
