"""Run independent native operator and eccentric two-sided contact controls on automatically generated shell meshes."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.stress_mesh_asset import prepare_stress_mesh
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.qipc_comparison.artifacts import runtime_identity, write_json
from research.rigid_stress.qipc_comparison.check_native_shell_contacts import (
    control as contact_control,
    field_components,
)
from research.rigid_stress.qipc_comparison.check_native_shell_operators import control as operator_control, relative
from research.rigid_stress.shell_comparison.automatic_shell_geometry import surface_recipe
from research.rigid_stress.shell_cpu import core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--shape", choices=("egg", "grasp", "neck", "u", "plate"), required=True)
    parser.add_argument("--operators-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "runtime.json", runtime_identity(require_gpu_inventory=args.backend == "gpu"))
    surface = surface_recipe(args.shape, 1 if args.shape == "neck" else 0)
    source = args.output / "midsurface.obj"
    surface.export(source)
    prepared = prepare_stress_mesh(source, None, "shell", 1.0, None, args.output / "cache")
    with np.load(prepared.elastic) as asset:
        vertices, quads = asset["vertices"], asset["quads"]
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, precision="64")
    operator_control(args.output, "operators", vertices, quads, "inverse", "translation", spinning=True)
    if not args.operators_only:
        contact_control(args.output, "contacts", vertices, quads, 16, "faces")
    reloaded = prepare_stress_mesh(prepared.elastic, None, "shell", 1.0, None, args.output / "cache")
    force = np.random.default_rng(137).normal(size=(len(vertices), 6, 4)) * 0.001
    force[:, 3:] *= 0.0005
    force[:, :, -1] = 0
    omega = V_VEC(3, dtype=gs.qd_float, shape=(4,))
    omega.fill(0)
    fields, peaks = [], []
    for path in (prepared.elastic, reloaded.elastic):
        options = gs.options.RigidStressOptions.dpr(mesh=path, backend="shell", output_mode="full", method="inverse")
        model = ShellModel(options)
        state = model.create_state(4, "full")
        state.force.from_numpy(force.reshape((-1, 3, 4)).transpose(0, 2, 1).copy())
        model.recover(omega, state)
        assert qd_to_numpy(state.valid).all()
        fields.append(qd_to_numpy(state.stress_tensor, transpose=True).reshape((4, len(quads), 2, 4, 6)))
        peaks.append(qd_to_numpy(state.step_peak))
    with np.load(prepared.elastic) as asset:
        oracle = core.ShellModel(
            vertices, quads, density=options.density, directors=asset["directors"], gauge="translation"
        )
    reference = field_components(oracle.stress(oracle.solve(force.reshape((-1, 4)))))
    errors = {
        "reload_full_tensor": relative(fields[1], fields[0]),
        "reload_peak": relative(peaks[1], peaks[0]),
        "independent_cpu_full_tensor": relative(fields[0], reference),
    }
    write_json(args.output / "reload.json", errors)
    np.savez_compressed(args.output / "reload-fields.npz", automatic=fields[0], explicit=fields[1], cpu=reference)
    assert max(errors.values()) < 1e-8, errors


if __name__ == "__main__":
    main()
