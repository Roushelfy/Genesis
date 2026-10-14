"""Generate and independently audit native separating rays for the actual policy component failure."""

import json
import sys
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.compatibility import create_wrench_search, kernel_certify_continuous_wrench, kernel_search_wrench_certificate
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import StressTractionStatus, create_traction_info, create_tractions
from genesis.utils.misc import qd_to_numpy

from continuum_boundary_certificate_candidate_v1 import kernel_boundary_certificate_support

root = Path(__file__).parent
backend, output = sys.argv[1], Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
gs.init(backend={"cpu": gs.cpu, "cuda": gs.cuda}[backend], precision="64", logging_level="warning")
with np.load(root / "audit-policy-failure-inputs-v1/results/original-environment-24.npz") as saved:
    original = {name: saved[name].copy() for name in saved.files}
model = StressModel(gs.options.RigidStressOptions(mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz", method="pcg", contact_model="intrinsic", intrinsic_support="pair_union"))
surface = StressSurface(10, model.info)
info = create_traction_info(surface.info)
contacts, sources = create_contacts(2, 1, 0.006), create_sources(2, 1)
for name in ("position", "center", "anchor_face", "force", "normal", "friction", "radius", "valid"):
    contacts.__dict__[name].from_numpy(original["contacts." + name][:2, None].copy())
for name in ("geom", "pair", "status", "free_moment", "anchor_moment", "barycentric"):
    sources.__dict__[name].from_numpy(original["raw_sources." + name][:2, None].copy())
angular_samples = 64
samples = surface.info.face_origin.shape[0] * (10 + angular_samples)
tractions = create_tractions(samples, 2, 1, local_samples=True, shared_support=True)
search = create_wrench_search(2, 1, 12)
epsilon = float(np.finfo(gs.np_float).eps)
kernel_boundary_certificate_support(contacts, tractions, surface.info, epsilon, angular_samples)
tractions.status.from_numpy(np.full((2, 1), StressTractionStatus.NUMERICAL_FAILURE, dtype=gs.np_int))
before = {name: qd_to_numpy(contacts.__dict__[name], transpose=True, copy=True) for name in ("position", "force", "center", "radius", "friction")}
records = []
for label, function in (
    ("native_boundary_search", lambda: kernel_search_wrench_certificate(contacts, sources, tractions, search, info, epsilon, 1e-8)),
    ("continuous_certificate", lambda: kernel_certify_continuous_wrench(contacts, sources, tractions, search, surface.info, epsilon, 12, 65536, 1e-8)),
):
    function()
    row = {"stage": label}
    for name in ("status", "certificate", "certificate_margin"):
        row[name] = qd_to_numpy(tractions.__dict__[name], transpose=True, copy=True)[0].tolist()
    for name in ("evaluations", "visited", "bound"):
        row["search_" + name] = qd_to_numpy(search.__dict__[name], transpose=True, copy=True)[0].tolist()
    records.append(row)
    print(json.dumps(row), flush=True)
    (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
for name, expected in before.items():
    np.testing.assert_array_equal(qd_to_numpy(contacts.__dict__[name], transpose=True, copy=True), expected)
profile = qd_to_numpy(tractions.profile, transpose=True, copy=True)[0]
positions = qd_to_numpy(tractions.sample_positions, transpose=True, copy=True)[0]
normals = qd_to_numpy(tractions.sample_normals, transpose=True, copy=True)[0]
ray = qd_to_numpy(tractions.certificate, transpose=True, copy=True)[0, 1]
np.savez(output / "native-ray-and-support.npz", ray=ray, profile=profile, positions=positions, normals=normals)
assert records[-1]["status"][0] != StressTractionStatus.CONTINUOUS_WRENCH_INFEASIBLE
assert records[-1]["status"][1] == StressTractionStatus.CONTINUOUS_WRENCH_INFEASIBLE, records[-1]
