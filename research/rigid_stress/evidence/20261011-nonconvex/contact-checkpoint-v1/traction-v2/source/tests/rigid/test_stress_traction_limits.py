import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionStatus,
    create_tractions,
    kernel_fit_tractions,
)
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from tests.rigid.test_stress_traction import kernel_traction_queries
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_planar_traction_limits_and_release():
    radius = 0.01
    xy = np.stack(np.meshgrid([-0.004, 0.0, 0.004], [-0.004, 0.0, 0.004]), axis=-1).reshape(-1, 2)
    positions = np.column_stack((xy, np.zeros(len(xy))))
    normals = np.broadcast_to([0.0, 0.0, 1.0], positions.shape).copy()
    distance = np.linalg.norm(positions, axis=1) / radius
    profile = np.exp(-0.5 * (distance / 0.45) ** 2) * (1.0 - distance**2) ** 2
    weights = profile / profile.sum()
    friction = np.array([0.0, 0.42, 0.42, 0.84, 0.6])
    directions = np.array([[0.0, 0.0, 1.0], [0.21, 0.0, 1.0], [0.42, 0.0, 1.0], [0.0, 0.0, -1.0], [0.6, 0.0, 1.0]])
    bias = np.array([0.5, 0.5, 0.0, 0.0, 0.5])
    reference = 1e-3 * weights[:, None, None] * (1.0 + positions[:, None, 0:1] / radius * bias[None, :, None])
    reference = reference * directions[None]
    force = reference.sum(axis=0)
    moment = np.cross(positions[:, None], reference).sum(axis=0)
    n_envs = len(friction)
    contacts = create_contacts(1, n_envs, radius)
    sources = create_sources(1, n_envs)
    tractions = create_tractions(len(positions), 1, n_envs)
    info = StressTractionInfo(
        positions=V_VEC(3, dtype=gs.qd_float, shape=(len(positions),)),
        inward_normals=V_VEC(3, dtype=gs.qd_float, shape=(len(positions),)),
    )
    info.positions.from_numpy(positions)
    info.inward_normals.from_numpy(normals)
    contacts.center.fill(0.0)
    contacts.position.fill(0.0)
    contacts.force.from_numpy(force[None])
    contacts.friction.from_numpy(friction[None])
    contacts.valid.fill(True)
    sources.anchor_moment.from_numpy(moment[None])
    tractions.profile.from_numpy(np.broadcast_to(profile[:, None, None], (len(positions), 1, n_envs)).copy())
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 1e-9, 128)
    status = qd_to_numpy(tractions.status, transpose=True)
    assert_equal(status[[0, 1, 2, 4]], StressTractionStatus.SUCCESS)
    assert_equal(status[3], StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE)
    actual_field = V_VEC(3, dtype=gs.qd_float, shape=(len(positions), 1, n_envs))
    kernel_traction_queries(actual_field, contacts, tractions, info)
    actual = qd_to_numpy(actual_field, transpose=True)[:, :, 0].transpose((1, 0, 2)).copy()
    legal = np.array([0, 1, 2, 4])
    assert_allclose(actual[:, legal].sum(axis=0), force[legal], rtol=0.0, atol=1e-11)
    actual_moment = np.cross(positions[:, None], actual).sum(axis=0)
    error = np.linalg.norm(actual_moment[legal] - moment[legal], axis=1) / (
        radius * np.linalg.norm(force[legal], axis=1)
    )
    assert (error <= 1e-8).all()
    compression = actual[:, legal, 2]
    tangent = np.linalg.norm(actual[:, legal, :2], axis=2)
    assert (compression >= 0).all()
    assert (tangent <= friction[None, legal] * compression + 1e-15).all()
    certificate = qd_to_numpy(tractions.certificate, transpose=True)[3, 0]
    values = certificate[:3] + np.cross(certificate[3:], positions / radius)
    assert (values[:, 2] + friction[3] * np.linalg.norm(values[:, :2], axis=1) < 0.0).all()
    assert np.dot(force[3], certificate[:3]) + np.dot(moment[3] / radius, certificate[3:]) > 0.0
    # Release and rebirth clear the released environment while preserving the
    # independently recomputed legal environments exactly.
    validity = np.ones((1, n_envs), dtype=gs.np_bool)
    validity[0, 1] = False
    contacts.valid.from_numpy(validity)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 1e-9, 128)
    kernel_traction_queries(actual_field, contacts, tractions, info)
    released = qd_to_numpy(actual_field, transpose=True)[:, :, 0].transpose((1, 0, 2)).copy()
    assert_equal(released[:, 1], 0.0)
    assert_equal(released[:, [0, 2, 4]], actual[:, [0, 2, 4]])
    contacts.valid.fill(True)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 1e-9, 128)
    kernel_traction_queries(actual_field, contacts, tractions, info)
    restored = qd_to_numpy(actual_field, transpose=True)[:, :, 0].transpose((1, 0, 2))
    assert_equal(restored, actual)
