"""Parallelize raw contact metadata clearing while preserving stable packing and all original input slots."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.engine.solvers.rigid.stress.intrinsic import StressContactPacking, StressWrenchCertificates


@qd.kernel
def kernel_pack_intrinsic_contacts(
    raw_contacts: StressContactState,
    packing: StressContactPacking,
    certificates: StressWrenchCertificates,
    state: StressState,
):
    for _ in range(1):
        packing.maximum[None] = 0
    for i_c, i_b in qd.ndrange(raw_contacts.valid.shape[0], raw_contacts.valid.shape[1]):
        certificates.ray[i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
        certificates.margin[i_c, i_b] = 0.0
        raw_contacts.status[i_c, i_b] = 0
        raw_contacts.force_error[i_c, i_b] = 0.0
        raw_contacts.moment_error[i_c, i_b] = 0.0
        raw_contacts.evaluations[i_c, i_b] = 0
        raw_contacts.candidate_count[i_c, i_b] = 0
    for i_b in range(packing.count.shape[0]):
        n_contacts = 0
        for i_c in range(raw_contacts.valid.shape[0]):
            if raw_contacts.valid[i_c, i_b]:
                packing.indices[n_contacts, i_b] = i_c
                n_contacts += 1
        packing.count[i_b] = n_contacts
        qd.atomic_max(packing.maximum[None], n_contacts)
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.force.shape[1]):
        state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
