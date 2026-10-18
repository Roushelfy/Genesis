"""Compare recorded physical per-link reactions without cancelling opposing finger loads."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import complete_control_grid, requested_duration_s, time_integrated_error
from .reference_uncertainty import frame_index


def reactions(path, owners, local, body_count):
    with np.load(path) as saved:
        active = saved["active_global_ids"]
        selected = active < len(owners)
        body = owners[active[selected]]
        points = saved["active_global_positions_m"][selected]
        loads = saved["active_global_forces_n"][selected]
        rotation = Rotation.from_quat(saved["actual_robot_material_quaternion_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
        com = saved["actual_robot_com_world_m"]
        expected = np.einsum("vij,vj->vi", rotation[body], local[active[selected]]) + com[body]
        defect = np.linalg.norm(expected - points, axis=1).max().item() if len(points) else 0.0
        assert defect < 2e-11, "Source does not share the verified physical rigid-vertex layout"
        force = np.zeros((body_count, 3))
        moment = np.zeros_like(force)
        np.add.at(force, body, loads)
        np.add.at(moment, body, np.cross(points - com[body], loads))
        missing = saved["active_global_forces_n"][~selected].sum(axis=0) - saved["contact_forces_n"].sum(axis=0)
        assert np.linalg.norm(missing) < 1e-8, "Unexpected non-rigid contact owners in prescribed scene"
    return force, moment, defect


def relative(actual, expected, floor):
    norm = np.linalg.norm(expected)
    return (np.linalg.norm(actual - expected) / norm).item() if norm > floor else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--finger-names", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.layout) as saved:
        owners, local, names = (
            saved["rigid_vertex_public_owner"],
            saved["rigid_local_points_m"],
            saved["body_names"].tolist(),
        )
    fingers = np.array([names.index(name) for name in args.finger_names])
    reference = frame_index(args.reference)
    reference_receipt = json.loads((args.reference / "result.json").read_text())
    reference_duration = requested_duration_s(reference_receipt)
    expected = {tick: reactions(path, owners, local, len(names)) for tick, path in reference.items()}
    records, arrays = [], {}
    times = sorted(reference)
    arrays["reference_time_s"] = np.array(times) * 0.01
    arrays["reference_link_force_world_n"] = np.stack([expected[tick][0] for tick in times])
    arrays["reference_link_moment_about_com_nm"] = np.stack([expected[tick][1] for tick in times])
    for index, folder in enumerate(args.candidates):
        frames = frame_index(folder)
        rows, candidate_force, candidate_moment, candidate_time = [], [], [], []
        for tick in sorted(reference.keys() & frames.keys()):
            force, moment, defect = reactions(frames[tick], owners, local, len(names))
            ref_force, ref_moment, ref_defect = expected[tick]
            rows.append(
                {
                    "time_s": tick * 0.01,
                    "maximum_physical_point_layout_defect_m": max(defect, ref_defect),
                    "finger_force_vector_relative_difference": relative(force[fingers], ref_force[fingers], 1e-3),
                    "finger_force_vector_absolute_difference_n": np.linalg.norm(
                        force[fingers] - ref_force[fingers]
                    ).item(),
                    "finger_moment_vector_relative_difference": relative(moment[fingers], ref_moment[fingers], 1e-7),
                    "per_body_force_relative_difference": [relative(a, b, 1e-3) for a, b in zip(force, ref_force)],
                    "per_body_moment_relative_difference": [relative(a, b, 1e-7) for a, b in zip(moment, ref_moment)],
                }
            )
            candidate_time.append(tick * 0.01)
            candidate_force.append(force)
            candidate_moment.append(moment)
        arrays[f"candidate{index}_time_s"] = np.array(candidate_time)
        arrays[f"candidate{index}_link_force_world_n"] = np.array(candidate_force)
        arrays[f"candidate{index}_link_moment_about_com_nm"] = np.array(candidate_moment)
        temporal = {}
        complete = (
            reference_receipt["completed_requested_history"]
            and json.loads((folder / "result.json").read_text())["completed_requested_history"]
            and len(rows) == len(reference)
            and complete_control_grid(reference, reference_duration)
        )
        if complete:
            for name, actual, ref_value, floor in (
                (
                    "finger_force",
                    np.array(candidate_force)[:, fingers],
                    np.array([expected[tick][0][fingers] for tick in times]),
                    1e-3,
                ),
                (
                    "finger_moment",
                    np.array(candidate_moment)[:, fingers],
                    np.array([expected[tick][1][fingers] for tick in times]),
                    1e-7,
                ),
            ):
                temporal[name] = time_integrated_error(
                    candidate_time,
                    np.linalg.norm(ref_value, axis=(1, 2)),
                    np.linalg.norm(actual - ref_value, axis=(1, 2)),
                    floor,
                )
        records.append(
            {
                "source": file_id(folder / "result.json"),
                "history": rows,
                "missing_candidate_ticks": sorted(reference.keys() - frames.keys()),
                "expected_reference_duration_s": reference_duration,
                "complete_reference_observation_grid": complete_control_grid(reference, reference_duration),
                "complete_time_integrated_finger_metrics": temporal,
            }
        )
    np.savez_compressed(args.output / "reactions.npz", **arrays)
    write_json(
        args.output / "result.json",
        {
            "kind": "actual-per-public-link-reaction-uncertainty",
            "identity": runtime_identity(),
            "layout": file_id(args.layout),
            "reference": file_id(args.reference / "result.json"),
            "body_names": names,
            "finger_names": args.finger_names,
            "finger_body_ids": fingers.tolist(),
            "comparisons": records,
            "reference_qualified": False,
            "scope": "Actual recorded last-assembled contact gradients, force=-gradient/dt^2, and actual physical points. Rigid ownership is verified at every source frame against its saved accepted COM/material quaternion; non-rigid gradients must sum to the saved egg force. All body reactions, including any self-contact, are retained. Finger vector norms keep opposing loads separate; net egg wrench is a different observable. Moment is about each actual body COM. Absolute floors1e-3N/1e-7Nm suppress near-zero relative denominators. Accepted-state gradient closure and continuum/contact uncertainty remain separate qualification gates.",
        },
    )


if __name__ == "__main__":
    main()
