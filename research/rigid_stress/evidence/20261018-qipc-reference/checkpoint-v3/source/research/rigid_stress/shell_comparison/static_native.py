"""Time the native core under the common prescribed volume load, independently of contact mapping."""

from time import perf_counter

import numpy as np
import torch

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

from .common import synchronize
from .static_cost import make_parser, measure


class NativeStatic:
    def __init__(self, args):
        self.options = RigidStressOptions(
            mesh=args.inputs / f"{args.geometry}-level{args.level}.npz", output_mode="full"
        )
        if args.inverse_max_bytes is not None:
            self.options = self.options.model_copy(update={"inverse_max_bytes": args.inverse_max_bytes})
        self.model = StressModel(self.options)
        self.state = self.model.create_state(1, "full")
        self.omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
        self.omega.fill(0.0)
        with np.load(args.loads / f"level{args.level}.npz") as saved:
            force, coordinates = saved["native_force_n"], saved["native_coordinates_m"]
        np.testing.assert_allclose(qd_to_numpy(self.model.info.vertices), coordinates, rtol=0, atol=1e-17)
        assert gs.use_zerocopy
        self.force_view = qd_to_torch(self.state.force, copy=False)
        self.base_force = torch.as_tensor(force[:, None], dtype=gs.tc_float, device=gs.device)
        self.description = {
            "method": self.model.selected_method,
            "options": self.options.model_dump(mode="json"),
            "nodes": len(coordinates),
            "mesh_fixed": True,
            "logical_full_inverse_payload_bytes": len(coordinates) ** 2 * 9 * 8
            if self.model.inverse is not None
            else 0,
        }

    def solve(self, amplitude):
        synchronize()
        started = perf_counter()
        kernel_begin_step(self.state)
        self.force_view.copy_(self.base_force * amplitude)
        self.model.recover(self.omega, self.state)
        synchronize()
        seconds = perf_counter() - started
        valid = bool(qd_to_numpy(self.state.valid, transpose=True)[0])
        rhs = qd_to_numpy(self.state.rhs, transpose=True)[0]
        residual = qd_to_numpy(self.state.residual, transpose=True)[0]
        relative = float(np.linalg.norm(residual) / np.linalg.norm(rhs))
        assert valid and relative <= 1e-8
        return {
            "amplitude": amplitude,
            "seconds": seconds,
            "numerically_valid": valid,
            "complete_relative_force_residual": relative,
        }

    def export(self, output):
        np.savez_compressed(
            output / "fields.npz",
            tensor_pa=qd_to_numpy(self.state.stress_tensor, transpose=True, copy=True)[0],
            vm_pa=qd_to_numpy(self.state.von_mises, transpose=True, copy=True)[0, ..., 0],
            force_n=tensor_to_array(self.force_view)[:, 0],
        )


def main():
    args = make_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    started = perf_counter()
    solver = NativeStatic(args)
    synchronize()
    solver.description["setup_seconds_in_this_process"] = perf_counter() - started
    measure(args, solver)


if __name__ == "__main__":
    main()
