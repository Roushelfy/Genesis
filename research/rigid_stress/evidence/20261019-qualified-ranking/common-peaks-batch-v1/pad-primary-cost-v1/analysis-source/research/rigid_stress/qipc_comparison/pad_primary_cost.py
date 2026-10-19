"""Whole-level wall time for the complete admitted live-pad workflow."""

from statistics import median
from time import perf_counter

from .artifacts import write_json


class PrimaryCost:
    def __init__(self, args):
        self.args = args
        self.enabled = args.primary_cost
        self.active = None
        self.samples = []
        if self.enabled:
            assert args.cost_warm_levels >= 1 and args.stress_stride > 0

    def begin(self, level, step):
        if not self.enabled or self.active is not None:
            return
        import torch

        torch.cuda.synchronize()
        self.active = (level, step, perf_counter())

    def end(self, level, step, endpoint):
        if not self.enabled or not endpoint:
            return
        import torch

        torch.cuda.synchronize()
        elapsed = perf_counter() - self.active[2]
        assert level == self.active[0]
        steps = step - self.active[1] + 1
        envs = 1
        self.samples.append(
            {
                "level": level,
                "warmup": level < self.args.cost_warm_levels,
                "steps": steps,
                "wall_seconds": elapsed,
                "step_latency_ms": 1000 * elapsed / steps,
                "aggregate_env_steps_per_second": envs * steps / elapsed,
                "torch_allocated_bytes_after_level": torch.cuda.memory_allocated(),
                "torch_reserved_bytes_after_level": torch.cuda.memory_reserved(),
            }
        )
        self.active = None

    def write(self, rows, completed, admitted):
        if not self.enabled:
            return
        measured = [s for s in self.samples if not s["warmup"]]
        for sample in self.samples:
            selected = [row["stress"] for row in rows if row["level"] == sample["level"] and "stress" in row]
            sample.update(
                stress_observations=len(selected),
                all_stress_observations_valid=all(r.get("native_valid", r.get("valid", False)) for r in selected),
                field_host_bytes=sum(r["field_host_bytes"] for r in selected),
            )
        latency = [s["step_latency_ms"] for s in measured]
        write_json(
            self.args.output / "cost.json",
            {
                "completed": completed,
                "mechanically_admitted": admitted,
                "method": getattr(self.args, "method", "shell"),
                "envs": 1,
                "output_mode": self.args.stress_output,
                "stress_stride": self.args.stress_stride,
                "primary_timing": True,
                "samples": self.samples,
                "measured_repeats": len(measured),
                "median_latency_ms": median(latency) if latency else None,
                "minimum_latency_ms": min(latency) if latency else None,
                "maximum_latency_ms": max(latency) if latency else None,
                "incomplete_level": self.active[0] if self.active is not None else None,
                "scope": "One synchronized wall interval per complete continuous load level, including actual native/pad or Shell integration, requested recovery, host output, all per-step mechanical admission calculations, endpoint field/certificate output and step-journal I/O. Per-phase timers are disabled. Initial complete levels are warmup; measured levels are consecutive warm repeats, not independent fresh-process replicates. This is end-to-end timing of the declared admitted research workflow, not pure production-solver latency. PyTorch endpoint allocations exclude Quadrants and are not whole-device peak memory. Use this exact run's mechanical audit; separate preflights cannot admit a failed primary run.",
            },
        )
