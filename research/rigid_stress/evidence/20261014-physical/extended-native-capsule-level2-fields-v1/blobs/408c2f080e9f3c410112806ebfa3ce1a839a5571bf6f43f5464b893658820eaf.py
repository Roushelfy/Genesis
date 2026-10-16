"""Exercise report completeness on real completed/failed prefixes without changing their original bytes."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from .original_history_status import history_status


def requested_steps(run):
    config = json.loads((run / "results/configuration.json").read_text())
    protocol = json.loads((Path(config["protocol_run"]) / "results/original_protocol.json").read_text())
    return config["macro_steps"] * round(protocol["macro_dt_s"] / config["dt"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-run", type=Path, required=True)
    parser.add_argument("--partial-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert int((args.complete_run / "returncode.txt").read_text()) == 0
    partial_code = int((args.partial_run / "returncode.txt").read_text())
    assert partial_code != 0 and not (args.partial_run / "results/summary.json").exists()
    complete = history_status(args.complete_run, 0, requested_steps(args.complete_run))
    failed = history_status(args.partial_run, partial_code, requested_steps(args.partial_run), True)
    zero_code_partial = history_status(args.partial_run, 0, requested_steps(args.partial_run), True)
    assert complete[1] and not failed[1] and not zero_code_partial[1]
    assert failed[0] == zero_code_partial[0]
    try:
        history_status(args.partial_run, 0, requested_steps(args.partial_run))
    except AssertionError:
        strict_partial_rejected = True
    else:
        raise AssertionError("Zero-code incomplete source was accepted without the opt-in flag")
    with tempfile.TemporaryDirectory(dir=args.output, prefix="metadata-") as folder:
        run = Path(folder)
        (run / "results").mkdir()
        for step in [1, 2]:
            (run / f"results/step{step:05d}.npz").touch()
        (run / "results/summary.json").write_text("{")
        assert history_status(run, 0, 2, True) == (2, False, "invalid_json")
        (run / "results/summary.json").write_text(json.dumps({"steps": [1]}))
        assert history_status(run, 0, 2, True) == (2, False, "saved_state_count_mismatch")
        (run / "results/step00002.npz").rename(run / "results/step00003.npz")
        try:
            history_status(run, 0, 3, True)
        except AssertionError:
            gap_rejected = True
        else:
            raise AssertionError("A nonconsecutive prefix was accepted")
    sources = [
        args.complete_run / "returncode.txt",
        args.complete_run / "results/summary.json",
        args.partial_run / "returncode.txt",
    ]
    result = {
        "scope": __doc__,
        "complete_case": complete,
        "failed_case": failed,
        "counterfactual_zero_wrapper_code_on_same_real_prefix": zero_code_partial,
        "actual_partial_producer_code_unchanged": partial_code,
        "strict_partial_rejected": strict_partial_rejected,
        "invalid_summary_and_mismatched_count_rejected": True,
        "nonconsecutive_prefix_rejected": gap_rejected,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "goal_completion": False,
        "limitations": "Metadata-only classification regression. The zero-code case is a supplied counterfactual parameter, not a changed original returncode file or a claimed actual scheduler status. Tiny metadata fixtures deliberately contain no physical arrays. Source closure, authoritative Slurm state and complete physical state/force audits remain separate requirements.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
