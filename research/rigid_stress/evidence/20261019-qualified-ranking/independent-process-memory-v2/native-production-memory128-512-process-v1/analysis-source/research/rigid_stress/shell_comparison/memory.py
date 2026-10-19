"""Intrusive NVIDIA process-memory sampling outside the workload's Python GIL."""

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from time import perf_counter, sleep


INTERVAL_SECONDS = 0.1


def collect(pid, started):
    """Run in a separate interpreter without importing a simulator or CUDA."""
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps({
        "event": "ready", "collector_pid": os.getpid(), "target_pid": pid,
        "scientific_modules_loaded": [name for name in ("torch", "genesis", "numpy", "quadrants") if name in sys.modules],
    }), flush=True)
    while not stopping and os.getppid() == pid:
        before = perf_counter() - started
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        after = perf_counter() - started
        matches = [
            {"seconds": after, "gpu_uuid": row[1].strip(), "process_mib": int(row[2].strip())}
            for row in csv.reader(result.stdout.splitlines())
            if row and row[0].strip() == str(pid)
        ]
        print(json.dumps({"event": "poll", "started_seconds": before, "seconds": after, "matches": matches}), flush=True)
        if not stopping:
            sleep(INTERVAL_SECONDS)


class ProcessMemory:
    def __init__(self, enabled):
        self.enabled = enabled
        self.samples = []
        self.process = None
        self.started = None
        self.collector_pid = None
        self.ready = None
        self._records = None
        self._errors = None
        self._finished = False

    def start(self):
        if not self.enabled:
            return
        if self.process is not None or self._finished:
            raise RuntimeError("Memory sampler cannot be started twice.")
        self.started = perf_counter()
        # TemporaryFile follows the job's explicit TMPDIR. Files avoid a full
        # stdout pipe blocking the child while the parent holds the GIL.
        self._records = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self._errors = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--pid", str(os.getpid()), "--started", repr(self.started)],
                stdout=self._records,
                stderr=self._errors,
            )
            deadline = perf_counter() + 5
            while True:
                # pread does not move the file offset shared with child stdout.
                first = os.pread(self._records.fileno(), 4096, 0).split(b"\n", 1)
                if len(first) == 2:
                    ready = json.loads(first[0])
                    if ready["event"] != "ready" or ready["target_pid"] != os.getpid():
                        raise RuntimeError("Invalid memory-collector readiness receipt.")
                    self.collector_pid = ready["collector_pid"]
                    if self.collector_pid != self.process.pid or self.collector_pid == os.getpid():
                        raise RuntimeError("Memory collector is not a separate owned process.")
                    if ready["scientific_modules_loaded"]:
                        raise RuntimeError("Memory collector unexpectedly imported a scientific runtime.")
                    self.ready = ready
                    break
                if self.process.poll() is not None or perf_counter() >= deadline:
                    raise RuntimeError("Memory collector exited or failed to become ready within5s.")
                sleep(0.01)
        except BaseException:
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self._records.close()
            self._errors.close()
            self._finished = True
            raise

    def finish(self):
        if not self.enabled:
            return {"enabled": False, "scope": "process-memory sampling disabled", "peak_mib": None, "samples": []}
        if self.process is None or self._finished:
            raise RuntimeError("Memory sampler must be started and finished exactly once.")
        requested_stop = perf_counter() - self.started
        try:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                code = self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("Memory collector did not stop within15s.") from None
            stopped = perf_counter() - self.started
            self._errors.seek(0)
            errors = self._errors.read()
            if code:
                raise RuntimeError(f"Memory collector exited with{code}: {errors}")
            self._records.seek(0)
            records = [json.loads(line) for line in self._records]
        finally:
            self._records.close()
            self._errors.close()
            self._finished = True
        polls = [r for r in records if r["event"] == "poll"]
        self.samples = [sample for poll in polls for sample in poll["matches"]]
        if not self.samples:
            raise RuntimeError("No GPU process-memory samples were recorded for this process.")
        sample_gaps = [b["seconds"] - a["seconds"] for a, b in zip(self.samples[:-1], self.samples[1:])]
        query_gaps = [b["started_seconds"] - a["started_seconds"] for a, b in zip(polls[:-1], polls[1:])]
        return {
            "enabled": True,
            "scope": "intrusive build/warm/rollout parent-process sampling by an independent stdlib-only interpreter",
            "method": "independent_process_nvidia_smi",
            "target_pid": os.getpid(),
            "collector_pid": self.collector_pid,
            "collector_returncode": code,
            "readiness": self.ready,
            "requested_interval_seconds": INTERVAL_SECONDS,
            "requested_stop_seconds": requested_stop,
            "collector_stopped_seconds": stopped,
            "peak_mib": max(row["process_mib"] for row in self.samples),
            "samples": self.samples,
            "polls": polls,
            "first_poll_started_seconds": polls[0]["started_seconds"],
            "maximum_query_duration_seconds": max(r["seconds"] - r["started_seconds"] for r in polls),
            "maximum_observed_query_gap_seconds": max(query_gaps, default=None),
            "maximum_observed_sample_gap_seconds": max(sample_gaps, default=None),
            "sampling_limit": "Independent polling avoids the parent's GIL stalls. NVIDIA visibility, query latency, process scheduling and the requested interval can still miss peaks. Query receipts include polls before GPU visibility; this is not a continuous peak or hardware-capacity bound.",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--started", type=float, required=True)
    arguments = parser.parse_args()
    collect(arguments.pid, arguments.started)
