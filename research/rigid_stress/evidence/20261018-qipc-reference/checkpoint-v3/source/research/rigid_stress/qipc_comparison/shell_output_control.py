"""Verify Shell max-only interval output against full Cauchy fields at every physical step."""

import numpy as np
import torch

import genesis as gs

from ..shell_comparison import diagnostics
from ..shell_comparison.application import ShellEgg
from ..shell_comparison.observation import FullField
from ..shell_comparison.rollout import controller_step, make_parser
from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = make_parser()
    parser.add_argument("--prefix-steps", type=int, default=420)
    args = parser.parse_args()
    assert args.model == "shell" and args.output_mode == "full" and not args.instrumented
    assert args.precision == "64" and args.steps > 0 and args.prefix_steps >= 0
    assert args.envs >= 2, "The control also checks isolation of a deliberately invalid environment"
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    workload = ShellEgg(args)
    # Reach loaded contact through the actual controller/physics history.
    # This is not a static-pose or snapshot restart.
    for _ in range(args.prefix_steps):
        controller_step(workload, True)
    full = workload.observation
    maximum = FullField.allocate(workload.scene.sim.shell_solver, "max")
    assert maximum.nbytes < full.nbytes
    expected = torch.zeros_like(maximum.peaks)
    observed = 0
    expected_history, actual_history = [], []

    def observe(solver):
        nonlocal observed
        first = observed % args.substeps == 0
        if first:
            expected.zero_()
        full.update(solver)
        maximum.update(solver, reset_peak=first)
        tensor = full.tensors
        trace = tensor.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        deviator = tensor - trace[..., None, None] * torch.eye(3, dtype=tensor.dtype, device=tensor.device) / 3
        # Derive VM from exported Cauchy with Torch, independently of the
        # observer's VM buffer/reduction and of its full_output specialization.
        independent_vm = (1.5**0.5) * torch.linalg.vector_norm(deviator, dim=(-2, -1))
        assert torch.isfinite(independent_vm).all().item()
        expected.copy_(torch.maximum(expected, independent_vm.amax(dim=0)))
        observed += 1

    diagnostics.SUBSTEP_OBSERVER = observe
    try:
        for _ in range(args.steps):
            controller_step(workload, True)
            torch.testing.assert_close(maximum.peaks, expected, rtol=2e-11, atol=1e-5)
            expected_history.append(expected.cpu().numpy().copy())
            actual_history.append(maximum.peaks.cpu().numpy().copy())
    finally:
        diagnostics.SUBSTEP_OBSERVER = None
    assert observed == args.steps * args.substeps
    assert np.max(expected_history) > 1.0, "An unloaded zero-field control cannot qualify loaded max output"
    # A separate, deliberately invalid read-only observer input. No physical
    # step is advanced after this public position setter.
    invalid = workload.egg.get_verts_pos().clone()
    invalid[0, 0, :] = float("nan")
    workload.egg.set_verts_pos(invalid)
    full.update(workload.scene.sim.shell_solver)
    maximum.update(workload.scene.sim.shell_solver)
    assert not torch.isfinite(full.tensors[:, 0]).all().item()
    assert torch.isposinf(maximum.peaks[0]).item(), "A max reduction must not hide nonfinite stress"
    assert torch.isfinite(full.tensors[:, 1:]).all().item() and torch.isfinite(maximum.peaks[1:]).all().item()
    torch.testing.assert_close(maximum.peaks[1:], full.peaks[1:], rtol=2e-11, atol=1e-5)
    np.savez_compressed(args.output / "peaks.npz", expected_pa=expected_history, actual_pa=actual_history)
    write_json(
        args.output / "result.json",
        {
            "kind": "same-state-full-and-max-interval-observer-control",
            "identity": runtime_identity(),
            "conditions": file_id(args.conditions),
            "input_files": [file_id(path) for path in sorted(args.inputs.iterdir()) if path.is_file()],
            "prefix_control_ticks": args.prefix_steps,
            "control_ticks": args.steps,
            "environments": args.envs,
            "physical_observations": observed,
            "maximum_peak_difference_pa": np.abs(np.array(expected_history) - actual_history).max().item(),
            "controlled_nonfinite_Cauchy_detected_in_full_and_max": True,
            "controlled_nonfinite_isolated_to_selected_environment": True,
            "expected_VM_source": "Torch deviator norm from complete Cauchy tensors at every physical step, not the observer's VM/peak buffers",
            "full_observer_bytes": full.nbytes,
            "max_only_observer_bytes": maximum.nbytes,
            "scope": "Same actual state, unmodified PR22 physics and constitutive observer at every substep. Max-only stores one VM peak per environment over the full scene interval without writing full field arrays. The deliberately invalid final observer probe is not an accepted physical trajectory or a claim that Shell supplies an invalid-step counter.",
        },
    )


if __name__ == "__main__":
    main()
