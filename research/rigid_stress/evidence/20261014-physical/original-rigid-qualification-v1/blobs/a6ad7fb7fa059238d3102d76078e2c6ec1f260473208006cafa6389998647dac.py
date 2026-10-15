"""Verify an explicit post-termination observation without fabricating a producer EXIT record."""

import hashlib
import json
from pathlib import Path


def verify_interrupted_source(run, observation_path):
    observation = json.loads(observation_path.read_text())
    assert Path(observation["source_run"]).resolve() == run.resolve()
    assert observation["scheduler_state"] in {
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
    }
    assert observation["natural_producer_closure"] is False
    assert not (run / "source-after.sha256").exists()
    assert not (run / "returncode.txt").exists()
    identities = observation["identities"]
    assert identities and {row["identity"] for row in identities} >= {"source"}
    expected = {}
    for kind in {row["identity"] for row in identities}:
        assert kind in {"source", "input-source"}
        for line in (run / (kind + ".sha256")).read_text().splitlines():
            sha, relative = line.split("  ", 1)
            expected[kind, relative] = sha
    assert {(row["identity"], row["file"]): row["sha256"] for row in identities} == expected
    for (kind, relative), sha in expected.items():
        assert hashlib.sha256((run / kind / relative).read_bytes()).hexdigest() == sha
    return observation
