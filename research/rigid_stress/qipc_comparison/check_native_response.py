"""Check integrated H/precision/screening against every-step native displacement recovery."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo
from genesis.utils.array_class import V_MAT
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import runtime_identity, write_json
from .audit_gpu_candidates import per_environment
from .check_native_atlas import cpu_control


@qd.kernel
def kernel_perturb(source: StressSurfaceInverseInfo, target: StressSurfaceInverseInfo):
    for i, j in qd.ndrange(source.force_blocks.shape[0], source.force_blocks.shape[1]):
        target.force_blocks[i, j] = 1.0001 * source.force_blocks[i, j]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--stress-response", choices=("sparse", "dense", "hybrid", "screen"), default="sparse")
    parser.add_argument("--response-precision", choices=("64", "32"), default="64")
    parser.add_argument("--response-accumulation", choices=("64", "32"), default="64")
    parser.add_argument("--correction-probe", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget = 1e-8 if args.response_precision == args.response_accumulation == "64" else 1e-4
    result = dict(
        identity=runtime_identity(),
        budget=budget,
        complete=False,
        rows=[],
        settings=vars(args) | {"conditions": str(args.conditions), "output": str(args.output)},
    )
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(
        args.envs,
        conditions=args.conditions,
        output_mode="max" if args.stress_response == "screen" else "full",
        radius_model="hertz",
        pressure_profile="hertz",
        quadrature=16,
        box_edge_policy="planar",
        stress_backend=args.stress_backend,
        stress_response=args.stress_response,
        response_precision=args.response_precision,
        response_accumulation=args.response_accumulation,
    )
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    state = entry.state
    reference_options = entry.options.model_copy(update={"stress_response": "none"})
    fields = (("displacement", state.displacement), ("peak", state.peak))
    if args.stress_response != "screen":
        fields += (("stress_tensor", state.stress_tensor),)
        if args.stress_backend == "shell":
            fields += (
                ("membrane_resultant", state.membrane_resultant),
                ("bending_resultant", state.bending_resultant),
                ("transverse_resultant", state.transverse_resultant),
            )
    original = entry.model.surface_inverse.info
    perturbed = None
    if args.correction_probe:
        perturbed = StressSurfaceInverseInfo(
            nodes=original.nodes,
            force_blocks=V_MAT(3, 3, dtype=gs.qd_float, shape=original.force_blocks.shape),
            centrifugal=original.centrifugal,
        )
        kernel_perturb(original, perturbed)
    for step in range(args.steps):
        if perturbed is not None and step == 130:
            entry.model.surface_inverse.info = perturbed
        workload.step()
        candidate = {name: qd_to_numpy(array, transpose=True) for name, array in fields}
        public_peak = tensor_to_array(workload.link.get_max_stress())
        np.testing.assert_allclose(public_peak, candidate["peak"], rtol=1e-12, atol=1e-8)
        assert qd_to_numpy(state.valid).all()
        row = dict(
            step=step,
            corrections=qd_to_numpy(state.corrections).tolist(),
            fallbacks=qd_to_numpy(state.fallbacks).tolist(),
            displacement_recovery=qd_to_numpy(entry.response.state.recover_displacement).tolist(),
            evaluated_blocks=qd_to_numpy(entry.response.state.evaluated).tolist(),
        )
        entry.model.surface_inverse.info = original
        entry.model.recover(entry.omega, state, reference_options, surface_load=True)
        row["errors"] = {
            name: float(per_environment(candidate[name], qd_to_numpy(array, transpose=True)).max())
            for name, array in fields
        }
        result["rows"].append(row)
        write_json(args.output / "result.json", result)
        assert qd_to_numpy(state.valid).all()
        assert max(row["errors"].values()) < budget
        if perturbed is not None and step == 130:
            assert max(row["corrections"]) > 0
            assert all(row["displacement_recovery"])
        if step == min(129, args.steps - 1):
            result["cpu_control"] = cpu_control(entry)
            np.savez_compressed(args.output / "fields.npz", **candidate)
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
