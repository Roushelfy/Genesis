"""Compare all active sorted contact inputs and dynamics around first divergence."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def difference(name, first, second):
    assert first.shape == second.shape
    delta = np.abs(first.astype(float) - second.astype(float))
    return {
        "name": name,
        "different_values": int((first != second).sum()),
        "maximum_absolute_difference": float(delta.max(initial=0)),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, receipts = [], []
    for step in (108, 109, 110, 120, 1200):
        paths = [args.campaign / "group0" / name / f"step{step}.npz" for name in ("rigid", "dpr")]
        receipts.extend(file_id(p) for p in paths)
        with np.load(paths[0]) as a, np.load(paths[1]) as b:
            counts = a["collider_state.n_contacts"]
            assert np.array_equal(counts, b["collider_state.n_contacts"])
            maximum = int(counts.max(initial=0))
            env = np.arange(len(counts))[None]
            valid = np.arange(maximum)[:, None] < counts[None]
            ids = [np.where(valid, saved["collider_state.contact_sort_idx"][:maximum], 0) for saved in (a, b)]
            contact = []
            for name in ("geom_a", "geom_b", "pos", "normal", "penetration", "friction", "sol_params", "force"):
                key = "collider_state.contact_data." + name
                first, second = [saved[key][index, env][valid] for saved, index in zip((a, b), ids)]
                assert np.isfinite(first).all() and np.isfinite(second).all()
                contact.append(difference(name, first, second))
            dynamics = []
            for name in a.files:
                if name.startswith(("dyn_state.dofs.", "dyn_state.links.", "rigid_info.qpos")):
                    first, second = a[name], b[name]
                    if np.isfinite(first).all() and np.isfinite(second).all():
                        d = difference(name, first, second)
                        if d["different_values"]:
                            dynamics.append(d)
            rows.append(
                {
                    "step": step,
                    "active_contacts": int(counts.sum()),
                    "contact_differences": contact,
                    "different_dynamics": dynamics,
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "comparison_completed": True,
            "steps": rows,
            "receipts": receipts,
            "scope": "Complete active contacts in the solver's sorted order, excluding inactive buffer contents. Position/normal/penetration precede the same step's constraint response; force and dynamics follow it. This locates observed differences without declaring a global causal proof or relaxing motion criteria.",
        },
    )


if __name__ == "__main__":
    main()
