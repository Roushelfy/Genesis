"""Native certificate search on the independently audited clipped physical support."""

import json
import sys
from pathlib import Path

import numpy as np
from boundary_wrench_penalty_candidate_v1 import func_boundary_wrench_penalty
from genesis.engine.solvers.rigid.stress import compatibility
from genesis.engine.solvers.rigid.stress.compatibility import (
    create_wrench_search,
    kernel_certify_continuous_wrench,
    kernel_search_wrench_certificate,
)
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
)
from genesis.engine.solvers.rigid.stress.intrinsic import kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import create_traction_info, kernel_fit_tractions
from triangle_disk_clip_candidate_v2 import install, kernel_clipped_profiles

import genesis as gs
from genesis.utils.misc import qd_to_numpy

root = Path(__file__).parent
backend, output = sys.argv[1], Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
install()
compatibility.func_wrench_polar_penalty = func_boundary_wrench_penalty
gs.init(backend={"cpu": gs.cpu, "cuda": gs.cuda}[backend], precision="64", logging_level="warning")
with np.load(root / "audit-policy-failure-inputs-v1/results/original-environment-24.npz") as saved:
    original = {key: saved[key].copy() for key in saved.files}
model = StressModel(
    gs.options.RigidStressOptions(
        mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz",
        method="pcg",
        contact_model="intrinsic",
        intrinsic_support="pair_union",
    )
)
surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
info = create_traction_info(surface.info)
contacts, sources = create_contacts(2, 1, 0.006), create_sources(2, 1)
for name in ("position", "center", "anchor_face", "force", "normal", "friction", "radius", "valid"):
    contacts.__dict__[name].from_numpy(original["contacts." + name][:2, None].copy())
for name in ("geom", "pair", "status", "free_moment", "anchor_moment", "barycentric"):
    sources.__dict__[name].from_numpy(original["raw_sources." + name][:2, None].copy())
footprints = create_footprints(paths.info.positions.shape[0], 2, 1)
tractions = create_surface_fan_tractions(surface.info, 2, 1, shared_support=True)
search = create_wrench_search(2, 1, 12)
epsilon = float(np.finfo(gs.np_float).eps)
kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
kernel_clipped_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
kernel_bind_traction_support(contacts, sources, footprints, tractions)
kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
records = []
for label, function in (
    ("initial_fit", None),
    (
        "native_search",
        lambda: kernel_search_wrench_certificate(contacts, sources, tractions, search, info, epsilon, 1e-8),
    ),
    (
        "continuous_certificate",
        lambda: kernel_certify_continuous_wrench(
            contacts, sources, tractions, search, surface.info, epsilon, 12, 65536, 1e-8
        ),
    ),
):
    if function is not None:
        function()
    row = {"stage": label}
    for name in ("status", "certificate", "certificate_margin", "force_error", "moment_error"):
        row[name] = qd_to_numpy(tractions.__dict__[name], transpose=True, copy=True)[0].tolist()
    for name in ("evaluations", "visited", "bound"):
        row["search_" + name] = qd_to_numpy(search.__dict__[name], transpose=True, copy=True)[0].tolist()
    records.append(row)
    (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(row), flush=True)
np.savez(output / "native-ray.npz", ray=qd_to_numpy(tractions.certificate, transpose=True, copy=True)[0, 1])
assert records[-1]["status"] == [0, 6], records[-1]
