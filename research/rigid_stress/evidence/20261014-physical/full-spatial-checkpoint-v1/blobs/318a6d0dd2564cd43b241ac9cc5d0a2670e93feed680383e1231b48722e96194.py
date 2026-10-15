"""Export audited complete precision-aware states and derived final fields for spatial comparisons."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .skfem_reference import SkfemElasticReference
from .stable_neohookean_reference import StableNeoHookeanBulk
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.audit_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    config = json.loads((args.run / "results/configuration.json").read_text())
    producer = json.loads((args.run / "results/summary.json").read_text())
    audit = json.loads((args.audit_run / "results/summary.json").read_text())
    assert Path(audit["run"]).resolve() == args.run.resolve()
    assert audit["states_checked"] == config["steps"] == len(producer["steps"])
    for row in audit["rows"]:
        assert row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = []
    for source in sorted((args.run / "results").glob("*.npz")):
        if source.name in ("failure.npz", "body_fields.npz"):
            continue
        target = args.output / source.name
        os.link(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        provenance.append(
            {"source": str(source), "export": str(target), "sha256": digest, "storage": "immutable hard link"}
        )
    with np.load(args.output / "reference_geometry.npz") as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    with np.load(args.output / f"step{config['steps']:04d}.npz") as saved:
        vector = np.empty(body.basis.N)
        vector[saved["body_component_permutation"]] = saved["displacement"][: body.basis.N]
    bulk = StableNeoHookeanBulk(body)
    tensor, vm, upper = bulk.stress_bounds(vector)
    assert not (args.output / "body_fields.npz").exists()
    np.savez_compressed(args.output / "body_fields.npz", tensor_Pa=tensor, vm_Pa=vm, upper_peak_Pa=upper)
    config["output"] = str(args.output)
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    result = {
        **producer,
        "scope": __doc__,
        "source_producer": str(args.run),
        "independent_audit": str(args.audit_run),
        "body_peak_lower_Pa": float(vm.max()),
        "body_peak_upper_Pa": float(upper.max()),
        "wide_coordinate_final_peak_lower_Pa": audit["rows"][-1]["raw_global_peak_lower_Pa"],
        "wide_coordinate_final_peak_upper_Pa": audit["rows"][-1]["raw_global_peak_upper_Pa"],
        "derived_final_field_convention": "Original FP64 reference stress-bound convention reconstructed from saved final positions; independent wide-coordinate intervals retained separately.",
        "new_dynamics_solved": False,
        "goal_completion": False,
        "limitations": "This exports an already completed and independently audited physical history; it is not a new simulation. All original state bytes remain unchanged, with explicit source hashes. Derived final fields support the established spatial-comparison format and preserve the independent wide-coordinate raw peak interval. Full physical contact/geometry/time/model/feedback and verified accuracy-cost acceptance remain incomplete.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "export_provenance.json").write_text(
        json.dumps({"producer": str(args.run), "audit_run": str(args.audit_run), "files": provenance}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "exported_states": len(producer["steps"]),
                "derived_FP64_peak_Pa": float(vm.max()),
                "independent_wide_peak_Pa": result["wide_coordinate_final_peak_lower_Pa"],
            }
        )
    )


if __name__ == "__main__":
    main()
