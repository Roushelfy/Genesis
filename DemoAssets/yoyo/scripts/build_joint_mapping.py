"""One-time script: extract Genesis DOF-index-to-joint-name mapping and save to JSON.

Run this once to generate ``genesis_joint_order.json`` so that the replay
controller can map trajectory qpos values to the correct joints without
importing Genesis at runtime.

Usage:
    python build_joint_mapping.py
    python build_joint_mapping.py --output ../v5_init/genesis_joint_order.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def get_sharpa_urdf() -> str:
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id="Genesis-Intelligence/internal_assets",
        repo_type="dataset",
        allow_patterns="marvin_sharpa_description/**",
    )
    return os.path.join(local_dir, "marvin_sharpa_description", "marvin_sharpa.urdf")


def build_mapping(urdf_path: str) -> list[str]:
    import genesis as gs

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(
        morph=gs.morphs.URDF(file=urdf_path, fixed=True),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    scene.build()

    n_qs = robot.n_qs
    idx_to_name: list[str] = [""] * n_qs
    for j in range(robot.n_joints):
        joint = robot.joints[j]
        for qi in joint.qs_idx_local:
            qi = int(qi)
            if 0 <= qi < n_qs:
                idx_to_name[qi] = joint.name

    print(f"Robot: {robot.n_joints} joints, {n_qs} DOFs")
    for i, name in enumerate(idx_to_name):
        print(f"  qpos[{i:2d}] -> {name}")

    gs.destroy()
    return idx_to_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate genesis_joint_order.json for the replay controller"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON path (default: <traj_dir>/genesis_joint_order.json)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="URDF path (default: download marvin_sharpa from HuggingFace)",
    )
    args = parser.parse_args()

    if args.urdf:
        urdf_path = args.urdf
    else:
        print("Downloading/locating marvin_sharpa URDF ...")
        urdf_path = get_sharpa_urdf()
    print(f"URDF: {urdf_path}")

    mapping = build_mapping(urdf_path)

    if args.output:
        out_path = Path(args.output)
    else:
        script_dir = Path(__file__).resolve().parent
        default_dir = script_dir.parent / "v5_init"
        default_dir.mkdir(parents=True, exist_ok=True)
        out_path = default_dir / "genesis_joint_order.json"

    out_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"\nSaved {len(mapping)} DOF mapping to {out_path}")


if __name__ == "__main__":
    main()
