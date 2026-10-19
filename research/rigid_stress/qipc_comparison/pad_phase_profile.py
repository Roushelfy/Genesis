"""Instrument actual solver phases in a separate fully audited live trajectory."""

import argparse
import csv
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .artifacts import file_id, runtime_identity, write_json
from .audit_pad_batch import peak

_MISSING = object()


class PhaseProfile:
    def __init__(self):
        self.level = None
        self.stack = []
        self.records = {}
        self.restore = []
        self.build_timings = None

    def timed(self, label, function):
        def call(*args, **kwargs):
            torch.cuda.synchronize()
            started = perf_counter()
            self.stack.append(0.0)
            try:
                return function(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                elapsed = perf_counter() - started
                children = self.stack.pop()
                if self.stack:
                    self.stack[-1] += elapsed
                record = self.records.setdefault(
                    (self.level, label),
                    {
                        "level": self.level,
                        "phase": label,
                        "calls": 0,
                        "inclusive_seconds": 0.0,
                        "exclusive_seconds": 0.0,
                    },
                )
                record["calls"] += 1
                record["inclusive_seconds"] += elapsed
                record["exclusive_seconds"] += elapsed - children

        return call

    def replace(self, owner, name, value):
        original = owner.__dict__.get(name, _MISSING)
        self.restore.append((owner, name, original))
        setattr(owner, name, value)

    def wrap(self, owner, name, label):
        self.replace(owner, name, self.timed(label, getattr(owner, name)))

    def install(self, driver, native):
        original = driver.force_history

        def history(args):
            for record in original(args):
                self.level = record[0]
                yield record

        self.replace(driver, "force_history", history)
        self.wrap(driver.gs.Scene, "build", "scene_build_inclusive")
        self.wrap(driver.gs.Scene, "step", "scene_step_inclusive")
        self.wrap(driver, "tensor_to_array", "motion_host_readback")
        self.wrap(driver, "qd_to_numpy", "driver_host_readback")
        if native:
            self.wrap(driver, "prepare_geometry", "physical_pad_geometry")
            self.wrap(driver, "solve", "physical_pad_contact_solve")
            observer = importlib.import_module("research.rigid_stress.qipc_comparison.native_pad_recovery")
            original_allocate = observer.NativePadRecovery.allocate

            def allocate(cls, *args, **kwargs):
                result = original_allocate(*args, **kwargs)
                self.build_timings = dataclasses.asdict(result.entry.model.build_timings)
                self.wrap(result.entry.model, "recover", "elastic_reconstruction")
                if result.entry.intrinsic is not None:
                    self.wrap(result.entry.intrinsic, "scatter", "intrinsic_contact_reconstruction")
                return result

            self.replace(
                observer.NativePadRecovery,
                "allocate",
                classmethod(self.timed("recovery_preprocessing_inclusive", allocate)),
            )
            self.wrap(observer.NativePadRecovery, "sample", "stress_observation_inclusive")
            self.wrap(observer.ContactViews, "load", "contact_packet_upload")
            self.wrap(observer, "qd_to_numpy", "stress_host_readback")
            for name in ("kernel_anchor", "kernel_pressure", "kernel_scatter", "kernel_accept"):
                self.wrap(observer, name, name)
        else:
            self.wrap(driver.FullField, "update", "stress_observer_compute")
            self.wrap(driver, "observe_elastic_force", "mechanical_audit_elastic_force")
            for name in (
                "kernel_shell_compute_forces",
                "kernel_shell_coarse_assemble",
                "kernel_shell_coarse_factorize",
                "kernel_shell_rigid_contact_detect",
                "kernel_shell_rigid_contact_solve",
                "kernel_shell_rigid_contact_finalize",
                "kernel_shell_apply_dv",
                "kernel_shell_integrate",
            ):
                self.wrap(driver.shell, name, name)

    def close(self):
        for owner, name, original in reversed(self.restore):
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", type=Path, required=True)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    task = json.loads(args.task.read_text())
    assert task["module"] in (
        "research.rigid_stress.qipc_comparison.plate_native_pad",
        "research.rigid_stress.qipc_comparison.plate_shell_coupled",
    )
    assert "--output" not in task["args"] and "--primary-cost" in task["args"]
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.output / "run"
    write_json(args.output / "start.json", {"task": file_id(args.task), "identity": runtime_identity()})
    driver = importlib.import_module(task["module"])
    native = task["module"].endswith("plate_native_pad")
    profile = PhaseProfile()
    profile.install(driver, native)
    previous = sys.argv
    try:
        sys.argv = [task["module"], *task["args"], "--intrusive-phase-profile", "--output", str(run)]
        driver.main()
    finally:
        sys.argv = previous
        profile.close()
        records = list(profile.records.values())
        write_json(
            args.output / "phases.json",
            {
                "records": records,
                "native_build_timings": profile.build_timings,
                "primary_timing": False,
            },
        )
        if records:
            with (args.output / "phases.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
    audit_module = importlib.import_module(
        "research.rigid_stress.qipc_comparison." + ("audit_native_pad" if native else "audit_coupled_shell")
    )
    try:
        sys.argv = [audit_module.__name__, "--runs", str(run), "--output", str(args.output / "audit")]
        audit_module.main()
    finally:
        sys.argv = previous
    audit = json.loads((args.output / "audit/result.json").read_text())
    result, start, cost = [json.loads((run / name).read_text()) for name in ("result.json", "start.json", "cost.json")]
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    config = start["args"]
    method = config.get("method", "shell")
    admitted = (
        audit["passed"]
        if native
        else all(
            r["completed"] and all(w["mechanical_window_passed"] for w in r["equilibrium_windows"])
            for r in audit["runs"]
        )
    )
    if not native:
        admitted &= all(
            not any(r["solver_failure_flags"])
            and r["equation_residual_relative"] <= 1e-3
            and r["physical_pad"]["normal_law_and_transfer_passed"]
            for r in rows
        )
    baseline = args.data_root / (
        f"native-pad-full-source-stability-v1/group{0 if method == 'dpr' else 1}/native/frames/stress-level0.npz"
        if native
        else "compliant-pad-fine-coarse-history-v1/group0/cycle/frames/level0.npz"
    )
    field = run / "frames" / ("stress-level1.npz" if native else "level1.npz")
    key = "stress_pa" if native else "cauchy_world_pa"
    with np.load(baseline) as saved:
        target = saved[key]
    with np.load(field) as saved:
        tensor = saved[key]
    field_error = float(np.linalg.norm(tensor - target) / np.linalg.norm(target))
    peak_error = abs(peak(tensor) / peak(target) - 1)
    checks = {
        "own_complete_history_admitted": bool(result["completed"] and admitted and len(rows) == 400),
        "field_conformance": field_error <= 0.005 and peak_error <= 0.01,
        "explicitly_non_primary": cost["primary_timing"] is False and cost["intrusive_phase_profile"],
        "warm_phase_records": any(r["level"] == 1 and r["calls"] > 0 for r in records),
    }
    write_json(
        args.output / "result.json",
        {
            "passed": all(checks.values()),
            "checks": checks,
            "primary_timing": False,
            "method": method,
            "field_relative_l2": field_error,
            "peak_relative_difference": peak_error,
            "receipts": [
                file_id(p)
                for p in (
                    run / "result.json",
                    run / "start.json",
                    run / "cost.json",
                    run / "steps.jsonl",
                    args.output / "audit/result.json",
                    field,
                    baseline,
                )
            ],
            "scope": "Intrusive CUDA-synchronized Python-call phase profile on its own full400-step admitted trajectory. Level0 is warmup and level1 is measured; None denotes preprocessing. Inclusive times overlap when calls nest; exclusive times subtract nested measured calls, not all uninstrumented work. Source files and solver arithmetic are unchanged; process-local wrappers are restored. Shell contact and elastic correction are one coupled nonlinear solve and are reported jointly, not split into invented costs. All timing from this process is diagnostic and cannot replace a primary wall-time run.",
        },
    )
    assert all(checks.values()), "Phase profile must pass its own history and field checks"


if __name__ == "__main__":
    main()
