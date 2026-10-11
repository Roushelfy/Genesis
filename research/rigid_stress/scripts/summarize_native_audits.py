"""Summarize all-environment work and acceptance without removing failed rows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def columns(values, names):
    array = np.asarray(values)
    return {
        name: {
            "sum": float(array[:, i].sum()),
            "mean": float(array[:, i].mean()),
            "p50_p90_p99_max": np.quantile(array[:, i], [0.5, 0.9, 0.99, 1.0]).tolist(),
            "nonzero_environments": int(np.count_nonzero(array[:, i])),
        }
        for i, name in enumerate(names)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    result = []
    for path in args.inputs:
        content = path.read_bytes()
        source = json.loads(content)
        row = {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "source_revision": source.get("source_revision"),
            "source_sha256": source.get("source_sha256"),
            "gpu_uuid": source.get("gpu_uuid"),
        }
        if "per_environment_work" in source:
            row.update({key: source[key] for key in ("scope", "envs", "seed", "warmup_steps", "trajectory_steps")})
            row["outcomes"] = dict(zip(source["outcome_columns"], source["trajectory_only_totals"]))
            row["failed_environment_steps"] = source["failed_environment_steps"]
            row["per_environment_work"] = columns(source["per_environment_work"], source["counter_columns"])
            row["additional_work"] = columns(
                source["additional_per_environment_work"], source["additional_work_columns"]
            )
            row["errors"] = columns(source["per_environment_error_maxima"], source["error_columns"])
            row["actual_mu_range"] = source["actual_mu_range"]
            row["scatter_overflow_calls"] = source["scatter_overflow_calls"]
            row["scope_notes"] = [source["tail_cost_scope"], source["additional_work_scope"], source["criteria"]]
        elif "oracle_rows" in source:
            observations = source["oracle_rows"]
            row["accepted_samples"] = len(observations)
            row["actual_mu_range"] = source["actual_contact_mu_range"]
            row["max_full_residual_budget_ratio"] = max(
                item["full_gauge_residual_N"] / item["residual_budget_N"] for item in observations
            )
            for key in (
                "global_peak_error_Pa",
                "tensor_max_absolute_error_Pa",
                "von_mises_field_max_absolute_error_Pa",
            ):
                row[key] = max(item[key] for item in observations if item[key] is not None)
        elif "records" in source:
            row["arguments"] = source["arguments"]
            row["invalid_environment_steps"] = source["invalid_environment_steps"]
            row["stages"] = source["summary"]
            row["scope_note"] = source["scope_note"]
        else:
            raise ValueError(f"Unsupported audit schema: {path}")
        result.append(row)
        print(
            path.name,
            json.dumps({key: value for key, value in row.items() if key not in ("source_sha256",)}),
            flush=True,
        )
    args.output.write_text(
        json.dumps(
            {"summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "audits": result}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
