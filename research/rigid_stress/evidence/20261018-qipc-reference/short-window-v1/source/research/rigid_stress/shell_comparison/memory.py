"""Intrusive process-memory sampling, kept separate from normal throughput."""

import csv
import os
import subprocess
import threading
from time import perf_counter


class ProcessMemory:
    def __init__(self, enabled):
        self.enabled = enabled
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def start(self):
        if self.enabled:
            self.thread.start()

    def sample(self):
        pid = str(os.getpid())
        started = perf_counter()
        while not self.stop_event.is_set():
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                self.error = result.stderr
                self.stop_event.set()
                return
            for row in csv.reader(result.stdout.splitlines()):
                if row[0].strip() == pid:
                    self.samples.append(
                        {
                            "seconds": perf_counter() - started,
                            "gpu_uuid": row[1].strip(),
                            "process_mib": int(row[2].strip()),
                        }
                    )
            self.stop_event.wait(0.1)

    def finish(self):
        if self.enabled:
            self.stop_event.set()
            self.thread.join()
            if self.error is not None:
                raise RuntimeError(self.error)
            if not self.samples:
                raise RuntimeError("No GPU process-memory samples were recorded for this process.")
        return {
            "enabled": self.enabled,
            "scope": "intrusive build/warm/rollout process sampling",
            "peak_mib": max((row["process_mib"] for row in self.samples), default=None),
            "samples": self.samples,
        }
