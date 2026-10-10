"""Summarize every environment's numerical, contact-tail and hold-proxy audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        content = path.read_bytes()
        source = json.loads(content)
        counters = np.array(source["per_environment_counters"], dtype=np.int64)
        errors = np.array(source["per_environment_errors"])
        failures = np.array(source["every_step_failures"])
        nonzero_contacts = int(counters[:, 0].sum())
        observed = counters[:, 0] > 0
        rows.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "source_revision": source["source_revision"],
                "source_sha256": source["source_sha256"],
                "gpu": source["gpu"],
                "gpu_uuid": source["gpu_uuid"],
                "scope": source["scope"],
                "envs": source["envs"],
                "seed": source["seed"],
                "output_mode": source["output_mode"],
                "audited_env_steps": source["envs"] * (source["warmup_steps"] + source["trajectory_steps"]),
                "invalid_environment_steps": int(failures.sum()),
                "environments_with_invalid_steps": int(np.count_nonzero(failures)),
                "nonzero_contact_observations": nonzero_contacts,
                "local_retry_contact_observations": int(counters[:, 1].sum()),
                "local_retry_fraction": float(counters[:, 1].sum() / max(1, nonzero_contacts)),
                "fit_evaluations_total": int(counters[:, 2].sum()),
                "fit_evaluations_mean_per_nonzero_contact": float(counters[:, 2].sum() / max(1, nonzero_contacts)),
                "fit_evaluations_max_per_contact": int(counters[:, 3].max()),
                "corrections": int(counters[:, 4].sum()),
                "factor_fallbacks": int(counters[:, 5].sum()),
                "scatter_overflow_calls": source["scatter_overflow_calls"],
                "hold_check_count": source["hold_check_count"],
                "failed_hold_check_count": source["failed_hold_check_count"],
                "hold_check_success_fraction": source["hold_check_success_fraction"],
                "environments_without_bilateral_lift": source["environments_without_bilateral_lift"],
                "environments_with_failed_hold_checks": source["environments_with_failed_hold_checks"],
                "hold_check_note": source["hold_check_note"],
                "full_residual_budget_ratio_max": float(errors[:, 0].max()),
                "wrench_force_relative_max": float(errors[:, 2].max()),
                "wrench_moment_relative_max": float(errors[:, 3].max()),
                "actual_mu_range": [float(errors[observed, 5].min()), float(errors[observed, 4].max())]
                if observed.any()
                else None,
                "environments_with_no_nonzero_contact": int((~observed).sum()),
                "per_environment_work_quantiles": source["per_environment_work_quantiles"],
                "note": "All environments remain in failures, work quantiles and hold outcomes. Actual mu range uses environments that observed a nonzero contact; no throughput rate is inferred from intrusive audits.",
            }
        )
        print(
            path.name,
            "invalid",
            rows[-1]["invalid_environment_steps"],
            "retry fraction",
            rows[-1]["local_retry_fraction"],
            "fit evaluations/contact",
            rows[-1]["fit_evaluations_mean_per_nonzero_contact"],
            "max",
            rows[-1]["fit_evaluations_max_per_contact"],
            flush=True,
        )
    args.output.write_text(
        json.dumps({"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "audits": rows}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
