"""Measure native dual stopping errors on captured contacts without changing acceptance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    create_contacts,
    func_contact_cell,
    func_contact_index,
    func_contact_sample,
    func_force_frame,
    kernel_anchor,
    kernel_pressure,
    kernel_scatter,
)
from genesis.engine.solvers.rigid.stress.grid import func_patch_grid
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_gradient(contacts: StressContactState, surface: StressSurfaceInfo, output: qd.Tensor):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        first, second = func_force_frame(contacts.force[i_c, i_b])
        low, dimensions = func_patch_grid(contacts.center[i_c, i_b], contacts.radius[i_c, i_b], surface.grid)
        gradient = qd.Vector.zero(gs.qd_float, 3)
        coefficient = contacts.coefficient[i_c, i_b] * contacts.weight_sum[i_c, i_b]
        for cell in range(dimensions[0] * dimensions[1] * dimensions[2] + 1):
            start, end = func_contact_cell(cell, i_c, i_b, low, dimensions, contacts, surface)
            for entry in range(start, end):
                sample = func_contact_index(entry, cell, dimensions, surface)
                coordinates, weight, _position, _shape, _face = func_contact_sample(
                    sample, i_c, i_b, first, second, contacts, surface
                )
                profile = qd.max(0.0, coordinates.dot(coefficient))
                gradient += weight / contacts.weight_sum[i_c, i_b] * profile * coordinates
        gradient[0] -= 1.0
        output[i_c, i_b] = gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    cases = json.loads(args.fixture.read_text())["cases"]
    gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning")
    model = StressModel(gs.options.RigidStressOptions(mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz"))
    surface = StressSurface(10, model.info)
    rows = []
    for cooperative in (False, True):
        contacts = create_contacts(1, len(cases), 0.006, cooperative)
        state = model.create_state(len(cases))
        for field in ("position", "force", "normal", "friction", "radius"):
            getattr(contacts, field).from_numpy(np.asarray([[case[field] for case in cases]]))
        contacts.valid.fill(True)
        kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
        kernel_pressure(contacts, surface.info, cooperative)
        output = V_VEC(3, dtype=gs.qd_float, shape=contacts.valid.shape)
        kernel_gradient(contacts, surface.info, output)
        row = {"cooperative": cooperative}
        for field in ("status", "evaluations", "coefficient", "center", "weight_sum", "candidate_count"):
            row[field] = qd_to_numpy(getattr(contacts, field), copy=True).tolist()
        row["dual_gradient"] = qd_to_numpy(output, copy=True).tolist()
        # Diagnostic only: inspect the same coefficients against the final unchanged
        # scatter wrench checks. This cannot publish a valid recovery or bypass errno.
        contacts.status.fill(0)
        kernel_scatter(contacts, state, model.info, surface.info)
        row["scatter_status"] = qd_to_numpy(contacts.status, copy=True).tolist()
        row["force_error"] = qd_to_numpy(contacts.force_error, copy=True).tolist()
        row["moment_error"] = qd_to_numpy(contacts.moment_error, copy=True).tolist()
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "cases": cases,
                "rows": rows,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path("genesis/engine/solvers/rigid/stress/contact.py"))
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
