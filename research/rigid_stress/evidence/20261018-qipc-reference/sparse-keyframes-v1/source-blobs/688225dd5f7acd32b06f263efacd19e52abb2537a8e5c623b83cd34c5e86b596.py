"""Locate mechanical reaction events before declaring a capped-control contact window."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .reaction_history import reactions
from .reference_uncertainty import frame_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.layout) as saved:
        owners, local, names = (
            saved["rigid_vertex_public_owner"],
            saved["rigid_local_points_m"],
            saved["body_names"].tolist(),
        )
    fingers = [names.index(name) for name in ("left_finger", "right_finger")]
    rows = []
    for tick, path in sorted(frame_index(args.reference).items()):
        if tick * 0.01 > args.time_limit + 1e-12:
            continue
        force, moment, defect = reactions(path, owners, local, len(names))
        with np.load(path) as saved:
            rows.append(
                {
                    "time_s": saved["time_s"].item(),
                    "finger_force_world_n": force[fingers].tolist(),
                    "finger_force_norm_n": np.linalg.norm(force[fingers], axis=1).tolist(),
                    "finger_moment_about_com_nm": moment[fingers].tolist(),
                    "minimum_egg_height_m": saved["positions_m"][:, 2].min().item(),
                    "q": saved["q"].tolist(),
                    "verified_rigid_layout_error_m": defect,
                }
            )
    assert rows
    first_events = []
    for threshold in (0.001, 0.1, 1.0):
        first_events.append(
            {
                "threshold_n": threshold,
                "first_observation_above_threshold": next(
                    (row for row in rows if max(row["finger_force_norm_n"]) > threshold), None
                ),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "mechanical-finger-reaction-event-window-planning",
            "identity": runtime_identity(),
            "reference": file_id(args.reference / "result.json"),
            "layout": file_id(args.layout),
            "first_events": first_events,
            "history": rows,
            "scope": "Retained mechanical reactions at original10ms geometry observations. Finger loads include every retained physical source, including any self-contact. They are not an egg-only traction classification. Layout/position closure is verified on each frame. No stress-error value is used to select events and no temporal contact-onset certificate follows from sparse saved times.",
        },
    )


if __name__ == "__main__":
    main()
