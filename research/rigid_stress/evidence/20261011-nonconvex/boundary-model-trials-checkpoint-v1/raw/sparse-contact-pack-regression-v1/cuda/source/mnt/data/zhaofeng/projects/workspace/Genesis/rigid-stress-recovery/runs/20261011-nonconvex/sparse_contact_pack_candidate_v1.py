"""Compact valid raw slots in parallel, then sort the short active list into the original stable order."""

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
    for i_b in range(packing.count.shape[0]):
        packing.count[i_b] = 0
    for i_c, i_b in qd.ndrange(raw_contacts.valid.shape[0], raw_contacts.valid.shape[1]):
        certificates.ray[i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
        certificates.margin[i_c, i_b] = 0.0
        raw_contacts.status[i_c, i_b] = 0
        raw_contacts.force_error[i_c, i_b] = 0.0
        raw_contacts.moment_error[i_c, i_b] = 0.0
        raw_contacts.evaluations[i_c, i_b] = 0
        raw_contacts.candidate_count[i_c, i_b] = 0
        if raw_contacts.valid[i_c, i_b]:
            destination = qd.atomic_add(packing.count[i_b], 1)
            packing.indices[destination, i_b] = i_c
    for i_b in range(packing.count.shape[0]):
        n_contacts = packing.count[i_b]
        for i_c in range(1, n_contacts):
            value = packing.indices[i_c, i_b]
            destination = i_c
            for i_j in range(i_c):
                if packing.indices[i_j, i_b] > value:
                    destination = qd.min(destination, i_j)
            for offset in range(i_c - destination):
                source = i_c - 1 - offset
                packing.indices[source + 1, i_b] = packing.indices[source, i_b]
            packing.indices[destination, i_b] = value
        qd.atomic_max(packing.maximum[None], n_contacts)
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.force.shape[1]):
        state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
