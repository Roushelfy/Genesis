"""Prepare terminal saved-state receipts and restore the complete physical P2 state.

Preparation is stdlib-only scheduler/file inspection. NumPy state restoration
is called only by the reference driver inside its compute allocation.
"""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from .artifacts import file_id, write_json
from .plates import equilibrium, force_history

PHYSICS_FILES = (
    "solid_pad_step.py",
    "modal_pad_dynamics.py",
    "condensed_pad_dynamics.py",
    "matrix_free_pad_dynamics.py",
    "p2_elastic.py",
    "p2_stvk.py",
    "compliant_plate_reference.py",
    "cvxopt_factor.py",
    "plates.py",
    "../skfem_reference.py",
)
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "PREEMPTED",
    "CANCELLED",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def semantic_signature(path):
    """Ignore documentation only; preserve every executable AST node."""
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def physical_arguments(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("output", "resume_checkpoint", "checkpoint_stride")
    }


def selected_state_file(proof):
    """Original endpoint receipts predate the explicit state-frame member."""
    if "state_frame" in proof:
        return proof["state_frame"]
    return proof["frames"][-1]["file"]


def complete_state_archive(path):
    """A killed compressed write must never become an admitted recovery state."""
    try:
        with zipfile.ZipFile(path) as archive:
            required = {
                key + ".npy" for key in (
                    "vertices_m", "tetrahedra", "load_n", "displacement_m", "velocity_m_per_s",
                    "source_points_m", "source_forces_n", "projected_area_m2", "forces_n",
                    "body_forces_n", "inertial_forces_n", "young_pa", "poisson",
                )
            }
            return required.issubset(archive.namelist()) and archive.testzip() is None
    except (OSError, ValueError, zipfile.BadZipFile, EOFError):
        return False


def select_saved_state(run, configuration, lines, *, level=None, step=None):
    """Select by declared global step; latest skips unfinished compressed writes."""
    count = round((configuration["ramp"] + configuration["hold"]) / configuration["dt"])
    total = len(configuration["loads"]) * count
    if level is not None:
        assert 0 <= level < len(configuration["loads"]) - 1
        prefix_steps, kind = (level + 1) * count, "endpoint"
        state_path = run / "frames" / f"level{level}.npz"
    elif step is not None:
        prefix_steps, kind = step, "step"
        state_path = run / "checkpoints" / f"step{prefix_steps}.npz"
    else:
        candidates = [
            ((index + 1) * count, "endpoint", run / "frames" / f"level{index}.npz")
            for index in range(len(configuration["loads"]) - 1)
        ]
        for path in (run / "checkpoints").glob("step*.npz"):
            number = path.stem.removeprefix("step")
            if number.isdigit():
                candidates.append((int(number), "step", path))
        candidates.sort(key=lambda row: row[0], reverse=True)
        selected = next(
            (row for row in candidates if 0 < row[0] < total and row[0] <= len(lines)
             and complete_state_archive(row[2])), None
        )
        assert selected is not None, "No complete saved state with remaining steps"
        prefix_steps, kind, state_path = selected
    assert 0 < prefix_steps < total and complete_state_archive(state_path)
    assert len(lines) >= prefix_steps and all(line.endswith(b"\n") for line in lines[:prefix_steps])
    return prefix_steps, kind, state_path


def validate_checkpoint(args, step_checks):
    proof = json.loads(args.resume_checkpoint.read_text())
    assert proof["passed"] and proof["terminal_scheduler_state"] in TERMINAL
    for receipt in proof["receipts"]:
        assert file_id(receipt["path"]) == receipt, "Checkpoint input changed: " + receipt["path"]
    for receipt in (
        proof["start"],
        proof["journal"],
        selected_state_file(proof),
        *(row["file"] for row in proof["frames"]),
        *(row["file"] for row in proof["physics_modules"]),
    ):
        assert receipt in proof["receipts"], "Unbound named checkpoint input"
    parent = Path(proof["run"])
    assert proof["start"]["path"] == str(parent / "start.json")
    assert proof["journal"]["path"] == str(parent / "steps.jsonl")
    count = round((args.ramp + args.hold) / args.dt)
    prefix_steps = proof["prefix_steps"]
    assert isinstance(prefix_steps, int) and 0 < prefix_steps < len(args.loads) * count
    kind = proof.get("checkpoint_kind", "endpoint")
    assert kind in ("endpoint", "step")
    completed_levels = prefix_steps // count
    assert proof["level"] == completed_levels - 1
    if kind == "endpoint":
        assert prefix_steps == (proof["level"] + 1) * count
        assert selected_state_file(proof) == proof["frames"][-1]["file"]
    else:
        assert selected_state_file(proof)["path"] == str(parent / "checkpoints" / f"step{prefix_steps}.npz")
    assert [row["level"] for row in proof["frames"]] == list(range(completed_levels))
    assert all(row["file"]["path"] == str(parent / "frames" / f"level{row['level']}.npz") for row in proof["frames"])
    assert tuple(row["relative"] for row in proof["physics_modules"]) == PHYSICS_FILES, (
        "Incomplete numerical module coverage"
    )
    start = json.loads(Path(proof["start"]["path"]).read_text())
    expected = {
        key: value for key, value in start["args"].items()
        if key not in ("output", "resume_checkpoint", "checkpoint_stride")
    }
    assert physical_arguments(args) == expected, "Checkpoint physical/solver arguments differ"
    assert start["mesh"] == file_id(args.mesh)
    assert args.contact_solver == "modal" and args.warm_contact and args.elasticity == "stvk"
    assert args.constitutive_correction == "direct" and args.reuse_contact_velocity
    for row in proof["physics_modules"]:
        captured = (
            parent.parent.parent / "analysis-source/research/rigid_stress/qipc_comparison" / row["relative"]
        ).resolve()
        assert row["file"]["path"] == str(captured), "Numerical source receipt belongs to another module"
        assert semantic_signature(captured) == row["semantic_sha256"], (
            "Recorded numerical signature does not match its captured source"
        )
        assert semantic_signature(Path(__file__).parent / row["relative"]) == row["semantic_sha256"], (
            "Checkpoint numerical implementation differs: " + row["relative"]
        )
    lines = Path(proof["journal"]["path"]).read_bytes().splitlines(keepends=True)[:prefix_steps]
    assert len(lines) == prefix_steps and all(line.endswith(b"\n") for line in lines)
    prefix = b"".join(lines)
    assert hashlib.sha256(prefix).hexdigest() == proof["prefix_sha256"]
    rows = [json.loads(line) for line in lines]
    expected_history = list(force_history(args))[:prefix_steps]
    assert [row["step"] for row in rows] == list(range(1, prefix_steps + 1))
    for index, (row, (level, _, command, _)) in enumerate(zip(rows, expected_history)):
        assert row["level"] == level and abs(row["time_s"] - (index + 1) * args.dt) <= 1e-12
        assert row["command_force_n"] == command and all(step_checks(row).values())
        assert row["contact_force_evaluation"] == "full_velocity"
    for level in range(completed_levels):
        endpoint = rows[(level + 1) * count - 1]
        window = [row for row in rows if row["level"] == level and row["time_s"] >= endpoint["time_s"] - 0.05 - 1e-10]
        assert len(window) >= 51 and window[-1]["time_s"] - window[0]["time_s"] >= 0.05 - 1e-10
        assert all(all(equilibrium(row, endpoint["command_force_n"]).values()) for row in window)
    return proof, rows, prefix


def restore_checkpoint(args, context, model, tets, points, area, sign):
    import numpy as np

    proof, rows, prefix = context
    endpoint = rows[-1]
    selected = selected_state_file(proof)
    with np.load(selected["path"]) as saved:
        if proof.get("checkpoint_kind", "endpoint") == "step":
            assert int(saved["step"]) == endpoint["step"] == proof["prefix_steps"]
            assert int(saved["level"]) == endpoint["level"]
            for key in ("time_s", "compression_coordinate_m", "plate_speed_m_per_s"):
                assert float(saved[key]) == endpoint[key], "Saved state/journal disagreement: " + key
        np.testing.assert_array_equal(saved["vertices_m"], model.vertices)
        np.testing.assert_array_equal(saved["tetrahedra"], tets)
        np.testing.assert_array_equal(saved["source_points_m"], points)
        np.testing.assert_array_equal(saved["projected_area_m2"], area)
        assert float(saved["young_pa"]) == model.reference.young and float(saved["poisson"]) == model.reference.poisson
        displacement, velocity, force = (
            saved[key] for key in ("displacement_m", "velocity_m_per_s", "source_forces_n")
        )
        assert displacement.size == velocity.size == model.reference.basis.N
        assert force.shape == points.shape and all(
            np.isfinite(value).all() for value in (displacement, velocity, force)
        )
        assert all(value.dtype == np.float64 for value in (displacement, velocity, force))
        assert float(saved["load_n"]) == endpoint["command_force_n"]
        np.testing.assert_allclose(
            [-force[sign < 0, 2].sum(), force[sign > 0, 2].sum()],
            [endpoint["upper_reaction_n"], endpoint["lower_reaction_n"]],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(force[:, :2].sum(axis=0), endpoint["tangential_resultant_n"], rtol=0, atol=1e-12)
        current = model.vertices + displacement
        net = saved["forces_n"] + saved["body_forces_n"] - saved["inertial_forces_n"]
        scale = max(endpoint["command_force_n"], 1e-3)
        assert np.linalg.norm(net.sum(axis=0)) <= 1e-7 * scale
        assert np.linalg.norm(np.cross(current, net).sum(axis=0)) <= 1e-7 * scale * 0.03
        assert np.all(sign * force[:, 2] >= 0)
    restored_displacement = np.r_[displacement.reshape(-1), endpoint["compression_coordinate_m"]]
    restored_velocity = np.r_[velocity.reshape(-1), endpoint["plate_speed_m_per_s"]]
    for row in proof["frames"]:
        target = args.output / "frames" / f"level{row['level']}.npz"
        assert not target.exists()
        shutil.copy2(row["file"]["path"], target)
        actual = file_id(target)
        assert all(actual[key] == row["file"][key] for key in ("bytes", "sha256"))
    provenance = {
        "receipt": file_id(args.resume_checkpoint),
        "terminal_parent_job": proof["job"],
        "parent_state": proof["terminal_scheduler_state"],
        "prefix_steps": len(rows),
        "prefix_sha256": proof["prefix_sha256"],
        "selected_frame": selected,
        "physical_state": "Exact saved full body displacement/velocity, journal plate displacement/velocity and lagged normal point force. Global load/time indices continue unchanged.",
        "algorithmic_warm_start": "Physical point force times dt; solver caches are rebuilt and original convergence gates remain mandatory.",
        "new_segment_elapsed_is_whole_trajectory_cost": False,
    }
    return restored_displacement, restored_velocity, sign * force[:, 2], (args.dt * force).reshape(-1), provenance


def main():
    cli = argparse.ArgumentParser(
        description="Prepare an immutable saved-state receipt only after the parent job is terminal"
    )
    cli.add_argument("--run", type=Path, required=True)
    selection = cli.add_mutually_exclusive_group(required=True)
    selection.add_argument("--level", type=int, help="Use a completed load-level endpoint")
    selection.add_argument("--step", type=int, help="Use an exact periodic full-state checkpoint")
    selection.add_argument("--latest", action="store_true", help="Use the newest complete saved state before the final step")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    proof = {"passed": False, "error": None, "run": str(args.run), "level": args.level, "receipts": []}
    try:
        campaign = args.run.parent.parent
        submission = campaign / "submission.json"
        job = (
            str(json.loads(submission.read_text())["job"]).split(";")[0]
            + "_"
            + args.run.parent.name.removeprefix("group")
        )
        output = subprocess.check_output(
            ["sacct", "-j", job, "-P", "--noheader", "--format=JobID,State,ExitCode"], text=True
        )
        matches = [line.split("|") for line in output.splitlines() if line.split("|")[0] == job]
        assert len(matches) == 1, "Parent scheduler state is missing or ambiguous"
        state = matches[0][1].split()[0].removesuffix("+")
        proof.update({"job": job, "scheduler_output": output, "terminal_scheduler_state": state})
        assert state in TERMINAL, "Parent scheduler job is still live; no continuation receipt admitted"
        start, journal = args.run / "start.json", args.run / "steps.jsonl"
        configuration = json.loads(start.read_text())["args"]
        count = round((configuration["ramp"] + configuration["hold"]) / configuration["dt"])
        lines = journal.read_bytes().splitlines(keepends=True)
        prefix_steps, kind, state_path = select_saved_state(
            args.run, configuration, lines, level=args.level, step=args.step
        )
        prefix = b"".join(lines[:prefix_steps])
        frames = [
            {"level": level, "file": file_id(args.run / "frames" / f"level{level}.npz")}
            for level in range(prefix_steps // count)
        ]
        physics_root = campaign / "analysis-source/research/rigid_stress/qipc_comparison"
        physics = [
            {
                "relative": name,
                "file": file_id((physics_root / name).resolve()),
                "semantic_sha256": semantic_signature(physics_root / name),
            }
            for name in PHYSICS_FILES
        ]
        proof.update(
            {
                "passed": True,
                "checkpoint_kind": kind,
                "state_frame": file_id(state_path),
                "level": prefix_steps // count - 1,
                "prefix_steps": prefix_steps,
                "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "start": file_id(start),
                "journal": file_id(journal),
                "frames": frames,
                "physics_modules": physics,
            }
        )
        proof["receipts"] = [
            file_id(submission),
            proof["start"],
            proof["journal"],
            proof["state_frame"],
            *[row["file"] for row in frames],
            *[row["file"] for row in physics],
        ]
    except (OSError, KeyError, ValueError, AssertionError, subprocess.CalledProcessError) as failure:
        proof["passed"], proof["error"] = False, str(failure)
    write_json(args.output / "result.json", proof)
    if not proof["passed"]:
        raise SystemExit("Terminal saved-state preparation refused; no continuation admitted")


if __name__ == "__main__":
    main()
