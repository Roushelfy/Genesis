"""Save and restore the rigid inputs to an isolated pressure-recovery benchmark.

Only active collider slots, the observed link and geometry poses/bounds travel
between processes. Every consumer still runs capture, association and Hertz
sizing on its own elastic surface. Fitted stress contacts are never restored.
"""

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

SCHEMA = "genesis.rigid-stress-module-raw-inputs.v1"


class RawField(NamedTuple):
    name: str
    value: qd.Tensor
    selection: str


class ContactSelection(NamedTuple):
    counts: np.ndarray
    environment: np.ndarray
    rank: np.ndarray
    slot: np.ndarray


def array_identity(value: np.ndarray) -> dict:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "bytes": value.nbytes,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _contact_fields(state, prefix: str):
    for member in dataclasses.fields(state):
        value = vars(state)[member.name]
        name = prefix + "." + member.name
        if dataclasses.is_dataclass(value):
            yield from _contact_fields(value, name)
        else:
            yield RawField(name, value, "contact_slot")


def raw_fields(workload: FrankaEgg) -> tuple[RawField, ...]:
    solver = workload.scene.rigid_solver
    collider = solver.collider.collider_state
    links, geoms = solver.dyn_state.links, solver.dyn_state.geoms
    return (
        RawField("collider.n_contacts", collider.n_contacts, "complete"),
        RawField("collider.contact_sort_idx", collider.contact_sort_idx, "contact_rank"),
        *_contact_fields(collider.contact_data, "collider.contact_data"),
        *(
            RawField("links." + name, vars(links)[name], "observed_link")
            for name in (
                "pos",
                "quat",
                "cd_ang",
                "cfrc_applied_ang",
                "cfrc_applied_vel",
                "cfrc_coupling_ang",
                "cfrc_coupling_vel",
            )
        ),
        *(RawField("geoms." + name, vars(geoms)[name], "complete") for name in ("pos", "quat", "aabb_min", "aabb_max")),
    )


def contact_selection(workload: FrankaEgg) -> ContactSelection:
    collider = workload.scene.rigid_solver.collider.collider_state
    counts = qd_to_numpy(collider.n_contacts, transpose=True, copy=True)
    capacity = collider.contact_sort_idx.shape[0]
    if counts.ndim != 1 or (counts < 0).any() or (counts > capacity).any():
        raise ValueError("Raw contact counts exceed the collider capacity.")
    environment, rank = np.nonzero(np.arange(capacity)[None, :] < counts[:, None])
    order = qd_to_torch(collider.contact_sort_idx, transpose=True, copy=False)
    slot = tensor_to_array(order[environment, rank]).copy()
    if (slot < 0).any() or (slot >= capacity).any():
        raise ValueError("An active sorted contact refers outside the raw contact buffer.")
    if len(np.unique(environment * capacity + slot)) != len(slot):
        raise ValueError("Active sorted contacts must refer to distinct raw slots per environment.")
    return ContactSelection(counts, environment, rank, slot)


def read_raw(workload: FrankaEgg, selection: ContactSelection) -> tuple[list[dict], dict[str, np.ndarray]]:
    records, values = [], {}
    link = workload.scene.rigid_solver.stress_recovery.links[0].link.idx
    for field in raw_fields(workload):
        view = qd_to_torch(field.value, transpose=True, copy=False)
        shape = list(view.shape)
        if field.selection == "contact_slot":
            view = view[selection.environment, selection.slot]
        elif field.selection == "contact_rank":
            view = view[selection.environment, selection.rank]
        elif field.selection == "observed_link":
            view = view[:, link : link + 1]
        value = np.ascontiguousarray(tensor_to_array(view))
        records.append(
            {"name": field.name, "selection": field.selection, "native_shape": shape, **array_identity(value)}
        )
        values[field.name] = value
    return records, values


def raw_identity(records: list[dict]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot_bindings(workload: FrankaEgg, recipe: dict) -> dict:
    """Bind the actual immutable inputs of association and original-geometry Hertz sizing."""
    solver = workload.scene.rigid_solver
    if not workload.stress or len(solver.stress_recovery.links) != 1:
        raise ValueError("Raw module snapshots require one observed stress link.")
    entry = solver.stress_recovery.links[0]
    options, radius = entry.options, entry.automatic_radius
    if options.contact_model != "pressure" or options.load_transfer != "patch" or options.radius_model != "hertz":
        raise ValueError("Raw module snapshots currently support pressure/patch/Hertz recovery only.")
    if options.pressure_profile != "hertz" or radius is None or not gs.use_zerocopy or gs.backend != gs.cuda:
        raise ValueError("Raw module snapshots require native CUDA interop and Hertz pressure sizing.")
    geometry = radius.geometry
    tensors = (
        *(
            RawField("geometry.geoms." + name, vars(geometry.geoms)[name], "complete")
            for name in ("type", "data", "face_start", "face_end", "link_idx", "contype", "conaffinity")
        ),
        *(
            RawField("geometry.verts." + name, vars(geometry.verts)[name], "complete")
            for name in ("init_pos", "init_normal", "geom_idx")
        ),
        RawField("geometry.faces.verts_idx", geometry.faces.verts_idx, "complete"),
        *(
            RawField("curvature." + member.name, vars(radius.curvature)[member.name], "complete")
            for member in dataclasses.fields(radius.curvature)
        ),
    )
    immutable = [
        {"name": field.name, **array_identity(qd_to_numpy(field.value, transpose=True, copy=True))} for field in tensors
    ]
    bank = (
        ("initial_joints", workload.initial_joints),
        ("position", workload.initial_position),
        ("quaternion", workload.initial_quaternion),
        ("radius", workload.base_radius),
        ("friction", workload.friction),
        ("pick_joints", workload.pick_joints),
        ("lift_joints", workload.lift_joints),
        ("slide_joints", workload.slide_joints),
    )
    root = Path(gs.__file__).resolve().parent.parent
    assets = Path("examples/rigid/assets/hollow_egg") / ("level" + str(recipe["level"]))
    return {
        "recipe": recipe,
        "observed_link_index": entry.link.idx,
        "environment_count": workload.n_envs,
        "precision": np.dtype(gs.np_float).str,
        "quadrants_version": qd.__version__,
        "constraint_contacts_enabled": not solver._disable_constraint,
        "source_epsilon": solver.stress_recovery.source_epsilon,
        "batch_offsets": solver._links_offset_quat.ndim == 3,
        "offset_position": array_identity(tensor_to_array(solver._links_offset_pos)),
        "offset_quaternion": array_identity(tensor_to_array(solver._links_offset_quat)),
        "Hertz": {
            "modulus": radius.modulus,
            "unloaded_radius": radius.unloaded_radius,
            "epsilon": radius.epsilon,
            "box_edges_as_planes": radius.box_edges_as_planes,
        },
        "physical_geometry_inputs": immutable,
        "actual_conditions": [{"name": name, **array_identity(tensor_to_array(value))} for name, value in bank]
        + [{"name": "delays", **array_identity(workload.delays)}],
        "rigid_assets": [
            {"path": str(assets / name), **file_identity(root / assets / name)}
            for name in ("egg_outer.obj", "egg_shell.urdf")
        ],
        "capture_source_files": [
            {"path": name, **file_identity(root / name)}
            for name in (
                "genesis/engine/solvers/rigid/stress/association.py",
                "genesis/engine/solvers/rigid/stress/automatic_radius.py",
                "genesis/engine/solvers/rigid/stress/curvature.py",
                "genesis/engine/solvers/rigid/stress/pair_geometry.py",
                "genesis/engine/solvers/rigid/stress/provenance.py",
                "genesis/utils/geom.py",
            )
        ],
    }


def save_module_snapshot(workload: FrankaEgg, directory: Path, bindings: dict, preparation: dict) -> dict:
    """Save active native rigid inputs after the validated frozen-frame preparation."""
    directory.mkdir(parents=True, exist_ok=False)
    qd.sync()
    selection = contact_selection(workload)
    records, values = read_raw(workload, selection)
    payload = directory / "raw-inputs.npz"
    np.savez_compressed(
        payload,
        **values,
        selection_environment=selection.environment,
        selection_rank=selection.rank,
        selection_slot=selection.slot,
    )
    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "bindings": bindings,
        "fields": records,
        "raw_input_sha256": raw_identity(records),
        "payload": {"path": payload.name, **file_identity(payload)},
        "selection": {
            name: array_identity(value)
            for name, value in (
                ("environment", selection.environment),
                ("rank", selection.rank),
                ("slot", selection.slot),
            )
        },
        "producer": {
            "command": sys.argv,
            "tick": workload.tick,
            "hostname": platform.node(),
            "pid": os.getpid(),
            "source_revision": subprocess.check_output(
                ["git", "-C", str(Path(gs.__file__).resolve().parent.parent), "rev-parse", "HEAD"], text=True
            ).strip(),
            "Slurm": {
                name: os.environ.get(name)
                for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "GENESIS_IMAGE_VER")
            },
            "stress_options": workload.link.stress_options.model_dump(mode="json"),
            "benchmark_helper": file_identity(Path(__file__)),
            "preparation": preparation,
        },
        "scope": "Actual raw collider slots referenced by the active sort permutation, observed link pose/angular velocity/applied and coupling wrenches, and all geom poses/AABBs. Unread inactive capacity is excluded. No fitted stress contacts, pressure coefficients, elastic K/solution, dynamic verts or BVH are stored. The producer boundary uses the validated existing frozen-frame restoration; it is not claimed to be a byte copy of the historical pre-integration pose.",
    }
    with np.load(payload, allow_pickle=False) as saved:
        for name, value in values.items():
            if array_identity(saved[name]) != array_identity(value):
                raise ValueError("Saved raw snapshot failed lossless roundtrip: " + name)
        for name, value in (("environment", selection.environment), ("rank", selection.rank), ("slot", selection.slot)):
            if array_identity(saved["selection_" + name]) != array_identity(value):
                raise ValueError("Saved raw selection failed lossless roundtrip: " + name)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {
        "directory": str(directory),
        "manifest": file_identity(path),
        "raw_input_sha256": manifest["raw_input_sha256"],
        "active_raw_contacts": len(selection.slot),
        "mode": "producer",
        "manifest_complete": True,
        "producer_tick": workload.tick,
    }


def restore_module_snapshot(workload: FrankaEgg, directory: Path, bindings: dict) -> dict:
    """Restore typed raw inputs, without calling a rigid setter or fitting a stress source."""
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["schema"] != SCHEMA or manifest["complete"] is not True or manifest["bindings"] != bindings:
        raise ValueError("Raw module snapshot schema or physical/controller/geometry bindings do not match.")
    payload = directory / "raw-inputs.npz"
    if manifest["payload"] != {"path": payload.name, **file_identity(payload)}:
        raise ValueError("Raw snapshot payload identity mismatch.")
    fields = raw_fields(workload)
    if [field.name for field in fields] != [record["name"] for record in manifest["fields"]]:
        raise ValueError("Raw snapshot field inventory mismatch.")
    link = workload.scene.rigid_solver.stress_recovery.links[0].link.idx
    with np.load(payload, allow_pickle=False) as saved:
        selection = ContactSelection(
            saved["collider.n_contacts"],
            saved["selection_environment"],
            saved["selection_rank"],
            saved["selection_slot"],
        )
        for name, value in (("environment", selection.environment), ("rank", selection.rank), ("slot", selection.slot)):
            if array_identity(value) != manifest["selection"][name]:
                raise ValueError("Raw contact selection identity mismatch: " + name)
        for field, record in zip(fields, manifest["fields"]):
            value = saved[field.name]
            view = qd_to_torch(field.value, transpose=True, copy=False)
            expected = {
                "name": field.name,
                "selection": field.selection,
                "native_shape": list(view.shape),
                **array_identity(value),
            }
            if record != expected or np.dtype(str(view.dtype).removeprefix("torch.")) != value.dtype:
                raise ValueError("Raw snapshot native type, shape or payload mismatch: " + field.name)
            original = torch.from_numpy(value).to(device=view.device)
            if field.selection == "contact_slot":
                view[selection.environment, selection.slot] = original
            elif field.selection == "contact_rank":
                view[selection.environment, selection.rank] = original
            elif field.selection == "observed_link":
                view[:, link : link + 1].copy_(original)
            else:
                view.copy_(original)
    qd.sync()
    actual = verify_module_snapshot(workload, directory)
    return {
        "directory": str(directory),
        "manifest": file_identity(path),
        "raw_input_sha256": actual,
        "active_raw_contacts": len(selection.slot),
        "mode": "consumer",
        "restore_exact": True,
        "producer_tick": manifest["producer"]["tick"],
        "consumer_scene_tick": workload.tick,
    }


def verify_module_snapshot(workload: FrankaEgg, directory: Path) -> str:
    """Require the same active raw inputs before and after native recovery."""
    manifest = json.loads((directory / "manifest.json").read_text())
    selection = contact_selection(workload)
    records, _values = read_raw(workload, selection)
    if records != manifest["fields"] or raw_identity(records) != manifest["raw_input_sha256"]:
        raise ValueError("Native recovery changed a frozen raw collider/link/geometry input.")
    for name, value in (("environment", selection.environment), ("rank", selection.rank), ("slot", selection.slot)):
        if array_identity(value) != manifest["selection"][name]:
            raise ValueError("Native recovery changed the active raw contact selection.")
    return manifest["raw_input_sha256"]
