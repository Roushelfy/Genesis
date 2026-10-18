"""Isolate omitted numerical/force history in the original public state restore.

These are diagnostic private-buffer restores, not a repaired baseline or a
replacement public checkpoint implementation. The original source is untouched.
"""

import math
from dataclasses import dataclass

import torch

import genesis as gs
from genesis.utils.array_class import iter_data
from genesis.utils.misc import qd_to_torch

from .application import ShellEgg
from .common import identity, vars_json, write_json
from .lifecycle import capture, difference
from .rollout import controller_step, make_parser


@dataclass
class SavedBuffer:
    name: str
    destination: object
    values: torch.Tensor

    def restore(self):
        view = qd_to_torch(self.destination, copy=False)
        view.copy_(self.values)


def save(name, field):
    return SavedBuffer(name, field, qd_to_torch(field, copy=True))


def main():
    parser = make_parser()
    parser.add_argument("--modes", nargs="+", default=["public", "shell-dv-only", "shell-scratch"])
    args = parser.parse_args()
    assert args.model == "shell" and not args.instrumented
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    assert gs.use_zerocopy
    workload = ShellEgg(args)
    for _ in range(40):
        controller_step(workload, True)
    state = workload.scene.get_state()
    shell = workload.scene.sim.shell_solver
    rigid = workload.scene.rigid_solver
    shell_scratch = [
        save(item.name, item.value)
        for item in iter_data(shell.shell_scratch, "shell_scratch")
        if math.prod(item.value.shape) > 0
    ]
    age = shell._coarse_age
    shell_dv = [item for item in shell_scratch if item.name.endswith(".verts_dv")]
    assert len(shell_dv) == 1
    rigid_checkpoint = rigid.__getstate__()
    constraint = rigid.constraint_solver.constraint_state
    warmstart = [
        save("constraint.qacc_ws", constraint.qacc_ws),
        save("constraint.is_warmstart", constraint.is_warmstart),
    ]
    coupling = [
        save("links.cfrc_coupling_vel", rigid.dyn_state.links.cfrc_coupling_vel),
        save("links.cfrc_coupling_ang", rigid.dyn_state.links.cfrc_coupling_ang),
    ]
    controller_step(workload, True)
    reference = capture(workload)
    rows = []
    for mode in args.modes:
        for repeat in range(3):
            workload.scene.reset(state=state)
            workload.tick = 40
            if mode == "shell-dv-only":
                shell_dv[0].restore()
            if mode in ("shell-scratch", "shell-and-rigid"):
                for item in shell_scratch:
                    item.restore()
                shell._coarse_age = age
            if mode == "rigid-warmstart":
                for item in warmstart:
                    item.restore()
            if mode == "pending-coupling":
                for item in coupling:
                    item.restore()
            if mode in ("rigid-checkpoint", "shell-and-rigid"):
                rigid.__setstate__(rigid_checkpoint)
            controller_step(workload, True)
            rows.append({"mode": mode, "repeat": repeat, **difference(reference, capture(workload), slice(None))})
    write_json(
        args.output / "result.json",
        {
            "kind": "private-cache-restore-ablation",
            "args": vars_json(args),
            "identity": identity(),
            "checks": rows,
            "shell_scratch_fields": [item.name for item in shell_scratch],
            "note": "A successful private-buffer ablation does not claim a fixed public SceneState API or bitwise GPU determinism.",
        },
    )


if __name__ == "__main__":
    main()
