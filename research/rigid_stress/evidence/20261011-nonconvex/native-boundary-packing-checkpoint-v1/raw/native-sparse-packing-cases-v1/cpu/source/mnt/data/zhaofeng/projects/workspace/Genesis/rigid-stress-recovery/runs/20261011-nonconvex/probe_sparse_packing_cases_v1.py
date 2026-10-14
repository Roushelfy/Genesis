"""Check stable sparse/dense packing and disclosed worst-case cost across complete raw arrays."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import quadrants as qd
from sparse_contact_pack_candidate_v1 import kernel_pack_intrinsic_contacts as quadratic_pack
from sparse_contact_pack_candidate_v2 import kernel_pack_intrinsic_contacts as hybrid_pack

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.intrinsic import (
    StressContactPacking, StressWrenchCertificates, kernel_pack_intrinsic_contacts as native_pack,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy

backend, output = sys.argv[1], Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
gs.init(backend={"cpu": gs.cpu, "cuda": gs.cuda}[backend], precision="64", logging_level="warning")
model = StressModel(gs.options.RigidStressOptions(mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz", method="direct"))
n_slots, n_envs = 1200, 6
contacts = create_contacts(n_slots, n_envs, 0.006)
packing = StressContactPacking(indices=V(dtype=gs.qd_int, shape=(n_slots, n_envs)), count=V(dtype=gs.qd_int, shape=(n_envs,)), maximum=V(dtype=gs.qd_int, shape=()))
certificates = StressWrenchCertificates(ray=V_VEC(6, dtype=gs.qd_float, shape=(n_slots, n_envs)), margin=V(dtype=gs.qd_float, shape=(n_slots, n_envs)))
state = model.create_state(n_envs, "max")
rng = np.random.default_rng(624011)
records = []
for label, counts, repeats in [
    ("sparse", [0, 1, 2, 4, 6, 8], 64),
    ("mixed", [0, 1, 8, 32, 64, 1200], 8),
    ("dense", [1200] * n_envs, 8),
    ("released", [0] * n_envs, 64),
    ("restored", [31, 32, 33, 1, 0, 1200], 8),
]:
    valid = np.zeros((n_slots, n_envs), dtype=bool)
    for i_b, count in enumerate(counts):
        valid[rng.choice(n_slots, count, replace=False), i_b] = True
    contacts.valid.from_numpy(valid)
    row = {"case": label, "active_counts": counts, "repeats": repeats, "kernel_wall_ms": {}}
    for name, function in [("native", native_pack), ("quadratic", quadratic_pack), ("hybrid", hybrid_pack)]:
        contacts.status.fill(7)
        contacts.force_error.fill(7)
        contacts.moment_error.fill(7)
        contacts.evaluations.fill(7)
        contacts.candidate_count.fill(7)
        certificates.ray.fill(7)
        certificates.margin.fill(7)
        state.force.fill(7)
        packing.indices.fill(-1)
        packing.count.fill(7)
        packing.maximum.fill(7)
        function(contacts, packing, certificates, state)
        actual = qd_to_numpy(packing.indices, transpose=True, copy=True)
        np.testing.assert_array_equal(qd_to_numpy(packing.count, transpose=True), counts)
        assert int(qd_to_numpy(packing.maximum)) == max(counts)
        for i_b, count in enumerate(counts):
            np.testing.assert_array_equal(actual[i_b, :count], np.flatnonzero(valid[:, i_b]))
        np.testing.assert_array_equal(qd_to_numpy(contacts.valid, transpose=True), valid.T)
        for buffer in (contacts.status, contacts.force_error, contacts.moment_error, contacts.evaluations, contacts.candidate_count, certificates.ray, certificates.margin, state.force):
            assert not qd_to_numpy(buffer, transpose=True).any()
        function(contacts, packing, certificates, state)
        qd.sync()
        started = time.perf_counter()
        for _ in range(repeats):
            function(contacts, packing, certificates, state)
        qd.sync()
        row["kernel_wall_ms"][name] = 1000 * (time.perf_counter() - started) / repeats
    records.append(row)
    (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(row), flush=True)
