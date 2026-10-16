"""Check byte-equivalent mass entries and active force/tangent under sparse assembly."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.sparse import _lil

from .dynamic_franka_reference import InertialRigidFingerContactSystem
from .franka_contact_reference import build_from_capture
from .sparse_franka_inertia import SparseInertialRigidFingerContactSystem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.reference / "results/configuration.json").read_text())
    closure = json.loads((Path(config["closure_run"]) / "results/configuration.json").read_text())
    static, _ = build_from_capture(Path(closure["capture"]), closure["dhat"])
    old = InertialRigidFingerContactSystem(static, config["dt"])
    new = SparseInertialRigidFingerContactSystem(static, config["dt"])
    delta = new.inertial_mass - old.inertial_mass
    assert delta.nnz == 0 or np.max(np.abs(delta.data)) == 0
    last = config["steps"]
    with np.load(args.reference / f"results/step{last:04d}.npz") as saved:
        u = saved["displacement"].copy()
    with np.load(args.reference / f"results/step{last - 1:04d}.npz") as saved:
        previous = saved["displacement"].copy()
        v = saved["velocity"].copy()
    external = static.gravity_force([0, 0, -9.81])
    for system in (old, new):
        system.q_target[:] = 0.018
        system.predicted_displacement = previous + config["dt"] * v
    a = old.evaluate_updated_friction(u, previous, external, True)
    b = new.evaluate_updated_friction(u, previous, external, True)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[4], b[4])
    h = b[2] - a[2]
    assert h.nnz == 0 or np.max(np.abs(h.data)) == 0
    np.testing.assert_allclose(a[0], b[0], rtol=0, atol=0)
    source = inspect.getsource(_lil._lil_base._set_arrayXarray_sparse)
    (args.output / "actual_scipy_lil_assignment.py").write_text(source)
    assert "toarray" in source
    report = {
        "scope": __doc__,
        "mass_shape": new.inertial_mass.shape,
        "sparse_mass_nonzeros": new.inertial_mass.nnz,
        "max_mass_entry_difference": 0.0,
        "max_active_force_difference_N": float(np.max(np.abs(a[1] - b[1]))),
        "max_active_tangent_entry_difference": 0.0,
        "actual_scipy_lil_source": str(_lil.__file__),
        "goal_completion": False,
        "limitations": "Exact matrix/active-equation equivalence qualifies the constructor replacement, not finer-mesh physical accuracy. The original source and long jobs remain unchanged; the isolated runner records actual executable snapshots. Runtime memory/cost on larger refinements requires its own measured evidence.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
