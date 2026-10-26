"""Compare scalar and cooperative feature selection on retained native geometry inputs.

Run under Slurm rtx-mid, image 1_26. The worker reconstructs the recorded
mother geometry and feature coverage, anchors and fits the original finite
pressure contacts, and compares both selector schedules. Timings isolate the
selector, including its force-buffer clear, and do not measure full recovery.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure
from genesis.engine.solvers.rigid.stress.feature_basis import FeatureBasis
from genesis.engine.solvers.rigid.stress.feature_regions import FeatureRegions, FeatureRegionState, kernel_select
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json


@dataclass(frozen=True)
class SelectionInputs:
    position: np.ndarray
    force: np.ndarray
    normal: np.ndarray
    valid: np.ndarray
    radius: np.ndarray


def read_inputs(path: Path) -> SelectionInputs:
    with np.load(path, allow_pickle=False) as frame:
        inputs = SelectionInputs(
            frame["contact_position"],
            frame["contact_force"],
            frame["contact_normal"],
            frame["contact_valid"],
            frame["contact_radius"],
        )
    if (
        inputs.position.ndim != 3
        or inputs.position.shape[-1] != 3
        or inputs.force.shape != inputs.position.shape
        or inputs.normal.shape != inputs.position.shape
        or inputs.valid.shape != inputs.position.shape[:2]
        or inputs.radius.shape != inputs.valid.shape
    ):
        raise ValueError(f"Frame {path} does not contain matching native contact arrays.")
    return inputs


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--native", type=Path, required=True, help="Geometry-control variant containing result.json.")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--envs", type=int, help="Tile all original environments to this exact multiple of their count.")
    cli.add_argument("--frames", nargs="+", help="Recorded frame names; defaults cover loads, release and boundaries.")
    args = cli.parse_args()
    if args.envs is not None and args.envs < 1:
        cli.error("--envs must be positive.")
    assert os.environ.get("SLURM_JOB_ID"), "Scientific execution requires a Slurm job."
    assert os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid", "Use rtx-mid."
    assert os.environ.get("GENESIS_IMAGE_VER") == "1_26", "Use image 1_26."
    native_path = args.native / "result.json"
    native = json.loads(native_path.read_text())
    if not native["complete"]:
        cli.error("The source geometry variant must have a complete result.")
    default_names = (
        "moving_asymmetric_0",
        "friction_reversal",
        "partial_footprint",
        "partial_environment_release",
        "release_all",
        "restore_after_release",
        "continuity_junction_1e-06_-1",
        "continuity_junction_1e-06_+1",
        "continuity_coverage_1e-06_-1",
        "continuity_coverage_1e-06_+1",
    )
    requested = set(default_names if args.frames is None else args.frames)
    frames = [row for row in native["rows"] if row["name"] in requested]
    missing = requested - {row["name"] for row in frames}
    if missing:
        cli.error(f"Source geometry variant lacks requested frame names: {sorted(missing)}")
    first = read_inputs(Path(frames[0]["frame"]["path"]))
    n_contacts, n_source_envs = first.valid.shape
    n_envs = n_source_envs if args.envs is None else args.envs
    if n_envs % n_source_envs:
        cli.error("--envs must be an exact multiple of the original environment count.")
    env_selection = np.arange(n_envs) % n_source_envs
    args.output.mkdir(parents=True, exist_ok=False)
    options = gs.options.RigidStressOptions(**native["options"])
    if options.contact_model != "pressure" or options.load_transfer != "patch" or options.atlas_regions < 1:
        cli.error("The source variant must use native finite-pressure patches and a nonempty feature library.")
    options = options.model_copy(update={"method": "direct", "surface_inverse": False})
    result = {
        "complete": False,
        "passed": False,
        "identity": runtime_identity(),
        "native": file_id(native_path),
        "source_variant_spatial_passed": native["passed"],
        "source_options": native["options"],
        "reconstruction_options": options.model_dump(mode="json"),
        "budget": 1e-12,
        "source_environments": n_source_envs,
        "environments": n_envs,
        "input_layout": "contact, environment, component; environments tile in original order",
        "pressure_friction": 0.5,
        "timing_scope": "Selector only, including the original group force-buffer clear; no elastic solve.",
        "synchronization": "qd.sync before and after each 100-call repeat; no readbacks inside timed intervals.",
        "rows": [],
    }
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    model = ShellModel(options) if options.backend == "shell" else StressModel(options)
    parents, fixed_nodes = None, None
    if options.atlas_basis is not None:
        with np.load(options.atlas_basis, allow_pickle=False) as basis_asset:
            if "parent_nodes" in basis_asset:
                parents = basis_asset["parent_nodes"]
            if "fixed_nodes" in basis_asset:
                fixed_nodes = basis_asset["fixed_nodes"]
    basis = FeatureBasis(model, parents, fixed_nodes, shell_bending=options.atlas_shell_bending)
    contacts = create_contacts(
        n_contacts, n_envs, options.contact_radius, hertz_profile=options.pressure_profile == "hertz"
    )
    surface = StressSurface(options.quadrature, model.load_info, angular=options.pressure_quadrature == "angle")
    regions = FeatureRegions(
        model,
        basis,
        contacts,
        options.atlas_regions,
        options.atlas_transition,
        options.atlas_core_scale,
        options.atlas_halo,
        center_mode=options.atlas_center_mode,
        selection=options.atlas_selection,
        narrow_fraction=options.atlas_narrow_fraction,
    )
    assert model.info.vertices.shape[0] == native["atlas"]["mother_blocks"]
    assert regions.info.positions.shape[0] == native["atlas"]["physical_nodes"]
    assert len(basis.coarse_nodes) == native["atlas"]["coarse_nodes"]
    scalar_state = FeatureRegionState(
        weights=V(dtype=gs.qd_float, shape=regions.state.weights.shape),
        fallback=V(dtype=gs.qd_bool, shape=regions.state.fallback.shape),
        force=regions.state.force,
    )
    scalar_state.weights.fill(0)
    scalar_state.fallback.fill(False)
    regions.state.weights.fill(0)
    regions.state.fallback.fill(False)
    widths = [options.atlas_transition]
    if options.atlas_transition != 0.0:
        widths.append(0.0)
    stress_source = Path(gs.__file__).resolve().parent / "engine/solvers/rigid/stress"
    result.update(
        quadrants_version=qd.__version__,
        quadrants_reductions=file_id(Path(qd.__file__).resolve().parent / "lang/simt/reductions.py"),
        numerical_sources=[
            file_id(stress_source / name)
            for name in ("feature_regions.py", "feature_basis.py", "contact.py", "surface.py", "quadrilateral.py")
        ],
        worker=file_id(__file__),
        mesh=file_id(options.mesh),
        basis=None if options.atlas_basis is None else file_id(options.atlas_basis),
        mother_blocks=model.info.vertices.shape[0],
        physical_nodes=regions.info.positions.shape[0],
        coarse_nodes=len(basis.coarse_nodes),
        features=options.atlas_regions,
        physical_faces=surface.info.face_origin.shape[0],
        contact_capacity=n_contacts,
        force_buffer_shape=regions.state.force.shape,
        widths=widths,
    )
    for row in frames:
        frame_path = Path(row["frame"]["path"])
        frame_identity = file_id(frame_path)
        assert frame_identity["sha256"] == row["frame"]["sha256"]
        inputs = read_inputs(frame_path)
        assert inputs.valid.shape == (n_contacts, n_source_envs)
        contacts.position.from_numpy(np.ascontiguousarray(inputs.position[:, env_selection]))
        contacts.force.from_numpy(np.ascontiguousarray(inputs.force[:, env_selection]))
        contacts.normal.from_numpy(np.ascontiguousarray(inputs.normal[:, env_selection]))
        contacts.valid.from_numpy(np.ascontiguousarray(inputs.valid[:, env_selection]))
        contacts.radius.from_numpy(np.ascontiguousarray(inputs.radius[:, env_selection]))
        contacts.friction.fill(0.5)
        kernel_anchor(float(np.finfo(gs.np_float).eps), contacts, surface.info)
        kernel_pressure(contacts, surface.info, options.cooperative_pressure)
        valid = inputs.valid[:, env_selection].T
        loaded = valid & (np.linalg.norm(inputs.force[:, env_selection], axis=-1).T > 1e-30)
        status = qd_to_numpy(contacts.status, transpose=True, copy=True)
        assert (status[loaded] == 0).all(), "Original native pressure fitting must admit every loaded contact."
        admitted = valid & (status == 0)
        for width in widths:
            for cooperative, state in ((False, scalar_state), (True, regions.state)):
                for _ in range(20):
                    kernel_select(
                        width,
                        contacts,
                        model.load_info,
                        surface.info,
                        regions.info,
                        state,
                        options.atlas_regions,
                        cooperative,
                        regions.selection_code,
                        regions.narrow_fraction,
                    )
            qd.sync()
            scalar_seconds, cooperative_seconds = [], []
            for _ in range(5):
                for cooperative, state in ((False, scalar_state), (True, regions.state)):
                    qd.sync()
                    started = time.perf_counter()
                    for _call in range(100):
                        kernel_select(
                            width,
                            contacts,
                            model.load_info,
                            surface.info,
                            regions.info,
                            state,
                            options.atlas_regions,
                            cooperative,
                            regions.selection_code,
                            regions.narrow_fraction,
                        )
                    qd.sync()
                    seconds = time.perf_counter() - started
                    if cooperative:
                        cooperative_seconds.append(seconds)
                    else:
                        scalar_seconds.append(seconds)
            scalar = qd_to_numpy(scalar_state.weights, transpose=True, copy=True)
            cooperative_weights = qd_to_numpy(regions.state.weights, transpose=True, copy=True)
            scalar_fallback = qd_to_numpy(scalar_state.fallback, transpose=True, copy=True)
            cooperative_fallback = qd_to_numpy(regions.state.fallback, transpose=True, copy=True)
            assert np.isfinite(scalar).all() and np.isfinite(cooperative_weights).all()
            np.testing.assert_allclose(cooperative_weights, scalar, rtol=0.0, atol=1e-12)
            np.testing.assert_array_equal(cooperative_fallback, scalar_fallback)
            np.testing.assert_allclose(scalar.sum(axis=1)[admitted], 1.0, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(cooperative_weights.sum(axis=1)[admitted], 1.0, rtol=0.0, atol=1e-12)
            assert not scalar_fallback[~valid].any() and not cooperative_fallback[~valid].any()
            np.testing.assert_allclose(scalar, scalar[:n_source_envs][env_selection], rtol=0.0, atol=1e-12)
            np.testing.assert_array_equal(scalar_fallback, scalar_fallback[:n_source_envs][env_selection])
            fine_contributors = np.count_nonzero(scalar[:, 1:, :] > 1e-14, axis=1)
            result["rows"].append(
                {
                    "name": row["name"],
                    "category": row["category"],
                    "frame": frame_identity,
                    "width": width,
                    "loaded": int(loaded.sum()),
                    "valid": int(valid.sum()),
                    "weight_shape": scalar.shape,
                    "maximum_weight_difference": float(np.max(np.abs(cooperative_weights - scalar))),
                    "maximum_admitted_weight_sum_error": float(
                        np.max(np.abs(scalar.sum(axis=1)[admitted] - 1.0), initial=0.0)
                    ),
                    "maximum_fine_contributors": int(np.max(fine_contributors[loaded], initial=0)),
                    "coverage_fallbacks": int(scalar_fallback[loaded].sum()),
                    "mother_weight_sum": float(scalar[:, 0, :][loaded].sum()),
                    "environment_isolation_passed": True,
                    "fallback_equal": True,
                    "repeats": 5,
                    "calls_per_repeat": 100,
                    "warmup_calls_per_schedule": 20,
                    "scalar_seconds": scalar_seconds,
                    "cooperative_seconds": cooperative_seconds,
                    "passed": True,
                }
            )
            write_json(args.output / "result.json", result)
    result.update(complete=True, passed=True)
    write_json(args.output / "result.json", result)
    print(json.dumps({"complete": True, "passed": True, "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
