"""Refine quadrature on saved heterogeneous inputs without changing their motion or support."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .audit_gpu_candidates import per_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.scene / f"field-{args.frame:04d}.npz"
    saved = np.load(path)
    n_envs, n_contacts = saved["valid"].shape
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    mesh = Path(gs.__file__).parent.parent / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    options = gs.options.RigidStressOptions.dpr(mesh=mesh, output_mode="full", quadrature=32)
    model = StressModel(options)
    state = model.create_state(n_envs, "full")
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.from_numpy(saved["angular_velocity"])
    contacts = create_contacts(n_contacts, n_envs, 0.001, hertz_profile=True)
    inputs = (
        ("position", contacts.position),
        ("force", contacts.force),
        ("normal", contacts.normal),
        ("friction", contacts.friction),
        ("radius", contacts.radius),
        ("valid", contacts.valid),
    )
    result = dict(identity=runtime_identity(), source=file_id(path), complete=False, rows=[])
    write_json(args.output / "result.json", result)
    fields = {}
    for degree in (32, 64, 128):
        surface = StressSurface(degree, model.load_info, angular=True)
        for name, array in inputs:
            array.from_numpy(np.ascontiguousarray(np.swapaxes(saved[name], 0, 1)))
        kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
        kernel_pressure(contacts, surface.info)
        kernel_scatter(contacts, state, model.load_info, surface.info)
        model.recover(omega, state, options, surface_load=True)
        status = qd_to_numpy(contacts.status, transpose=True)
        row = dict(
            quadrature=degree,
            failed_contacts=int(np.count_nonzero(saved["valid"] & (status != 0))),
            valid=bool(qd_to_numpy(state.valid).all()),
            force_error=float(qd_to_numpy(contacts.force_error).max()),
            moment_error=float(qd_to_numpy(contacts.moment_error).max()),
        )
        fields[degree] = {
            "nodal_force": qd_to_numpy(state.force, transpose=True),
            "displacement": qd_to_numpy(state.displacement, transpose=True),
            "stress_tensor": qd_to_numpy(state.stress_tensor, transpose=True),
            "peak": qd_to_numpy(state.peak),
        }
        np.savez_compressed(args.output / f"angle{degree}.npz", **fields[degree])
        if degree == 32:
            row["saved_replay_errors"] = {
                key: float(per_environment(value, saved[key]).max())
                for key, value in fields[degree].items()
                if key != "peak"
            }
        result["rows"].append(row)
        write_json(args.output / "result.json", result)
        assert row["failed_contacts"] == 0 and row["valid"]
    result["comparisons"] = []
    for degree in (32, 64):
        errors = {key: per_environment(value, fields[128][key]) for key, value in fields[degree].items()}
        passed = all(float(values.max()) < (1e-6 if key == "nodal_force" else 1e-4) for key, values in errors.items())
        result["comparisons"].append(
            dict(quadrature=degree, passed=passed, errors={key: values.tolist() for key, values in errors.items()})
        )
    result["saved_replay_passed"] = max(result["rows"][0]["saved_replay_errors"].values()) < 1e-8
    result["complete"] = True
    write_json(args.output / "result.json", result)
    assert result["saved_replay_passed"]
    assert all(row["passed"] for row in result["comparisons"])


if __name__ == "__main__":
    main()
