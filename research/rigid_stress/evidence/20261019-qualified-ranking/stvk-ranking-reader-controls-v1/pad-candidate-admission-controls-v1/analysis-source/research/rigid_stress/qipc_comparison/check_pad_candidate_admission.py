"""Recheck original candidate mechanics and bind all six requested stress fields."""

import argparse
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json
from .pad_candidate_admission import admit_candidates


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, receipts = {}, {}

    def verify(expected=None, *, path=None):
        path = Path(path if expected is None else expected["path"])
        if str(path) not in receipts:
            receipts[str(path)] = file_id(path)
        if expected is not None:
            assert receipts[str(path)] == expected, path
        return receipts[str(path)]

    def read(path, *, journal=False):
        verify(path=path)
        data = Path(path).read_text()
        return [json.loads(line) for line in data.splitlines()] if journal else json.loads(data)

    def check(name, passed):
        checks[name] = bool(passed)
        assert passed, name

    candidates, failures = admit_candidates(args.data_root, read, verify, check)
    assert len(candidates) == 3 and sum(len(c["frames"]) for c in candidates.values()) == 6 and len(failures) == 2
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": all(checks.values()),
            "checks": checks,
            "candidates": candidates,
            "retained_every_step_failures": failures,
            "receipts": list(receipts.values()),
            "fine_reference_qualified": False,
            "scope": "Rechecks three original400-step candidate mechanical histories and six own-admitted requested endpoint fields. Exact journal/audit frame hashes, original equations/contact laws, declared2mm full-source settings and distinct failed every-step costs are retained. No fine-reference error, rank or physical guarantee at unobserved native steps is inferred.",
        },
    )


if __name__ == "__main__":
    main()
