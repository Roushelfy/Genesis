"""Audit complete resident-history module measurements and their retained source inputs."""

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def source_receipt(campaign, receipts):
    """Rehash the pinned engine/assets and submitted research workers."""
    path = campaign / "source.json"
    source = json.loads(path.read_text())
    receipts.append(file_id(path))
    assert len(source["files"]) >= 858 and len(source["revision"]) == 40
    assert len({r["path"] for r in source["files"]}) == len(source["files"])
    for row in source["files"]:
        actual = file_id(Path(source["source"]) / row["path"])
        assert all(actual[key] == row[key] for key in ("bytes", "sha256")), row["path"]
    patch = file_id(campaign / "source.patch")
    assert patch["sha256"] == source["patch_sha256"] and patch["bytes"] == 0
    receipts.append(patch)
    path = campaign / "analysis-files.json"
    workers = json.loads(path.read_text())
    receipts.append(file_id(path))
    assert len({r["path"] for r in workers}) == len(workers)
    for row in workers:
        actual = file_id(campaign / "analysis-source" / row["path"])
        assert all(actual[key] == row[key] for key in ("bytes", "sha256")), row["path"]
    return source, workers


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    folder = args.campaign / "group0"
    assert (folder / "returncode.txt").read_text().strip() == "0"
    rows, receipts, cases = [], [], []
    source, workers = source_receipt(args.campaign, receipts)
    receipts.append(file_id(folder / "returncode.txt"))
    for mode in ("dpr-max", "dpr-full", "scr-max", "scr-full"):
        code = folder / (mode + ".returncode.txt")
        assert code.read_text().strip() == "0"
        receipts.append(file_id(code))
        root = folder / mode
        path = root / "result.json"
        result = json.loads(path.read_text())
        receipts.append(file_id(path))
        assert result["passed"] and all(result["checks"].values())
        assert result["envs"] == 8 and result["steps"] == 1200
        assert result["warmup"] == 900 and result["dt_s"] == 0.01 and result["substeps"] == 1
        assert result["stress_stride"] == 1
        assert (result["method"], result["output_mode"]) == tuple(mode.split("-"))
        assert file_id(result["conditions"]["path"]) == result["conditions"]
        receipts.append(result["conditions"])
        receipts.append(file_id(root / "start.json"))
        replay = result["replay"]
        assert replay["complete_history_check_passed"] and replay["partial_resets_replayed"]
        assert len(replay["buffers"]) == 24
        for key in ("trajectory",):
            ident = file_id(result[key]["path"])
            assert ident == result[key]
            receipts.append(ident)
        ident = file_id(replay["input_history"]["path"])
        assert ident == replay["input_history"]
        receipts.append(ident)
        with np.load(ident["path"]) as saved:
            assert set(saved.files) == {b["name"] for b in replay["buffers"]}
            for b in replay["buffers"]:
                assert list(saved[b["name"]].shape) == b["shape"]
                assert "torch." + str(saved[b["name"]].dtype) == b["dtype"]
            assert sum(saved[b["name"]].nbytes for b in replay["buffers"]) == replay["recorded_device_bytes"]
            count, order = saved["n_contacts"], saved["contact_sort_idx"]
            assert count.shape == (1200, 8) and (count >= 0).all() and (count <= order.shape[-1]).all()
            valid = np.arange(order.shape[-1]) < count[..., None]
            indices = np.where(valid, order, 0)
            assert (indices >= 0).all() and (indices < saved["force"].shape[2]).all()
            force = np.take_along_axis(saved["force"], indices[..., None], axis=2)
            normal = np.take_along_axis(saved["normal"], indices[..., None], axis=2)
            force = np.where(valid[..., None], force, 0)
            normal = np.where(valid[..., None], normal, 0)
            assert np.isfinite(force).all() and np.isfinite(normal).all()
            magnitude = np.linalg.norm(force, axis=-1)
            tangent = force - (force * normal).sum(axis=-1, keepdims=True) * normal
            assert (magnitude > 0).any() and (np.diff(force, axis=0) != 0).any()
            assert (np.linalg.norm(tangent, axis=-1) > 0).any()
            # These are all recorded collider slots, not a claimed egg-only
            # footprint or physical-pad traction decomposition.
            source_summary = {
                "loaded_collider_slots": int((magnitude > 0).sum()),
                "maximum_collider_force_n": float(magnitude.max()),
                "maximum_collider_tangent_n": float(np.linalg.norm(tangent, axis=-1).max()),
                "changing_inputs": True,
                "scope": "All valid collider slots; no force threshold, no egg-only attribution.",
            }
        with np.load(result["trajectory"]["path"]) as saved:
            assert saved["valid"].all() and (saved["errno"] == 0).all() and np.isfinite(saved["peak_pa"]).all()
        repeats = replay["repeats"]
        assert len(repeats) == 3
        for r in repeats:
            assert r["updates"] == 1200 and r["invalid_environment_updates"] == 0
            assert math.isfinite(r["seconds"]) and r["seconds"] > 0
            np.testing.assert_allclose(r["env_updates_per_second"], 9600 / r["seconds"], rtol=1e-12, atol=0)
        rates = [r["env_updates_per_second"] for r in repeats]
        rows.append(
            {
                "mode": mode,
                "batch": 8,
                "median_env_updates_per_s": statistics.median(rates),
                "min_env_updates_per_s": min(rates),
                "max_env_updates_per_s": max(rates),
                "median_update_batch_ms": statistics.median([1000 * r["seconds"] / 1200 for r in repeats]),
                "recorded_device_bytes": replay["recorded_device_bytes"],
                "replay_peak_max_absolute_difference_pa": replay["peak_max_absolute_difference_pa"],
                "source_summary": source_summary,
                "gpu": result["gpu_name"],
                "gpu_uuid": result["gpu_uuid"],
                "result": file_id(path),
                "stress_configuration": result["stress_configuration"],
                "physical_error": None,
            }
        )
        cases.append(result)
    assert all(c["replay"]["stress_reset_schedule"] == cases[0]["replay"]["stress_reset_schedule"] for c in cases)
    assert len({c["gpu_uuid"] for c in cases}) == 1
    assert len({c["conditions"]["sha256"] for c in cases}) == 1
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": True,
            "rows": rows,
            "receipts": receipts,
            "engine_revision": source["revision"],
            "source_files_rehashed": len(source["files"]),
            "captured_workers_rehashed": len(workers),
            "physical_reference_qualified": False,
            "production_motion_conformance_inferred": False,
            "scope": "Four B8 module modes, same GPU allocation, complete1200-frame resident histories and actual stress reset schedule. Replay includes24-buffer device restoration and unchanged native recovery/acceptance. Source/archive/field validation is outside timing; no dynamics or policy transition rate. Warm repeats and this native collider load configuration do not establish physical-pad accuracy or final production admission.",
        },
    )


if __name__ == "__main__":
    main()
