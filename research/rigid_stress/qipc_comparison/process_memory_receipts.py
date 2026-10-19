"""Independently check collector lifecycle, complete poll receipts and observed gaps."""

import math
import statistics


def audit_polling(memory, target_pid, gpu_uuid):
    assert memory["enabled"] and memory["method"] == "independent_process_nvidia_smi"
    assert memory["collector_returncode"] == 0
    assert memory["target_pid"] == target_pid > 0
    assert memory["collector_pid"] != target_pid and memory["collector_pid"] > 0
    ready = memory["readiness"]
    assert ready["event"] == "ready" and ready["target_pid"] == target_pid
    assert ready["collector_pid"] == memory["collector_pid"] and ready["scientific_modules_loaded"] == []
    assert memory["requested_interval_seconds"] == 0.1
    polls, samples = memory["polls"], memory["samples"]
    assert len(polls) > 1 and len(samples) > 1
    assert [s for p in polls for s in p["matches"]] == samples
    for poll in polls:
        assert poll["event"] == "poll"
        assert math.isfinite(poll["started_seconds"]) and math.isfinite(poll["seconds"])
        assert 0 <= poll["started_seconds"] <= poll["seconds"]
        for sample in poll["matches"]:
            assert sample["seconds"] == poll["seconds"]
            assert sample["gpu_uuid"].removeprefix("GPU-") == gpu_uuid.removeprefix("GPU-")
            assert isinstance(sample["process_mib"], int) and sample["process_mib"] >= 0
    assert all(b["started_seconds"] >= a["seconds"] for a, b in zip(polls[:-1], polls[1:]))
    sample_gaps = [b["seconds"] - a["seconds"] for a, b in zip(samples[:-1], samples[1:])]
    query_gaps = [b["started_seconds"] - a["started_seconds"] for a, b in zip(polls[:-1], polls[1:])]
    assert all(g > 0 for g in sample_gaps) and all(g > 0 for g in query_gaps)
    durations = [p["seconds"] - p["started_seconds"] for p in polls]
    assert memory["peak_mib"] == max(s["process_mib"] for s in samples)
    assert memory["maximum_observed_sample_gap_seconds"] == max(sample_gaps)
    assert memory["maximum_observed_query_gap_seconds"] == max(query_gaps)
    assert memory["maximum_query_duration_seconds"] == max(durations)
    assert memory["first_poll_started_seconds"] == polls[0]["started_seconds"]
    assert 0 < memory["requested_stop_seconds"] <= memory["collector_stopped_seconds"]
    assert polls[-1]["seconds"] <= memory["collector_stopped_seconds"]
    return {
        "method": memory["method"], "target_pid": target_pid, "collector_pid": memory["collector_pid"],
        "collector_returncode": memory["collector_returncode"],
        "poll_count": len(polls), "visible_poll_count": sum(bool(p["matches"]) for p in polls),
        "sample_count": len(samples), "first_query_seconds": polls[0]["started_seconds"],
        "first_process_sample_seconds": samples[0]["seconds"],
        "last_query_seconds": polls[-1]["seconds"],
        "requested_stop_seconds": memory["requested_stop_seconds"],
        "maximum_query_duration_seconds": max(durations),
        "maximum_observed_query_gap_seconds": max(query_gaps),
        "median_observed_query_gap_seconds": statistics.median(query_gaps),
        "maximum_observed_sample_gap_seconds": max(sample_gaps),
        "median_observed_sample_gap_seconds": statistics.median(sample_gaps),
        "query_gap_target_seconds": 1.0,
        "observed_query_gap_within_target": max(query_gaps) <= 1.0,
        "scope": "Complete retained NVIDIA polling records, including absent-parent queries before GPU visibility. The1s observed-query-gap target describes instrumentation coverage, not mechanical admission or a bound on missed memory peaks.",
    }
