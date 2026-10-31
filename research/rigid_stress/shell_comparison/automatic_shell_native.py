"""Run independent native operator and eccentric two-sided contact controls on automatically generated shell meshes."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.stress_mesh_asset import prepare_stress_mesh
from research.rigid_stress.qipc_comparison.artifacts import runtime_identity, write_json
from research.rigid_stress.qipc_comparison.check_native_shell_contacts import control as contact_control
from research.rigid_stress.qipc_comparison.check_native_shell_operators import control as operator_control
from research.rigid_stress.shell_comparison.automatic_shell_geometry import surface_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--shape", choices=("egg", "neck", "u", "plate"), required=True)
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


if __name__ == "__main__":
    main()
