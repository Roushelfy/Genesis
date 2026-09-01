"""Shared Marvin Wuji construction for QIPC worlds."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

import genesis as gs

ARM_JOINTS = {
    "right": tuple(f"Joint{i}_R" for i in range(1, 8)),
    "left": tuple(f"Joint{i}_L" for i in range(1, 8)),
}
HAND_JOINTS = {
    side: tuple(f"{side}_hand_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))
    for side in ("right", "left")
}
PALM_LINK = {
    "right": "right_hand_palm_link",
    "left": "left_hand_palm_link",
}
DEFAULT_INIT_ARM_QPOS = {
    "right": tuple(np.deg2rad((-110.0, -75.0, 90.0, -110.0, -75.0, 0.0, 0.0))),
    "left": tuple(np.deg2rad((110.0, -75.0, -90.0, -110.0, 75.0, 0.0, 0.0))),
}
ARM_KP = (7200.0, 7200.0, 7200.0, 3600.0, 3600.0, 3600.0, 3600.0)
ARM_KV = (600.0, 600.0, 600.0, 400.0, 200.0, 200.0, 200.0)
HAND_KP = 50.0
HAND_KV = 0.5
HAND_EFFORT_SCALE = 8.0
ARM_FORCE_LIMIT = 2000.0


@dataclass(frozen=True)
class BuiltMarvinWuji:
    robot: object
    dofs: dict[tuple[str, str], list[int]]
    home_qpos: np.ndarray
    hand_effort: dict[str, np.ndarray]


def resolve_marvin_wuji_urdf(explicit_path: str | None) -> str:
    """Resolve the Marvin Wuji URDF from an explicit path, environment, or HF cache."""
    if explicit_path:
        path = os.path.abspath(os.path.expanduser(explicit_path))
        if not os.path.isfile(path):
            gs.raise_exception(f"Marvin/Wuji URDF does not exist: '{path}'.")
        return path

    environment_path = os.environ.get("QIPC_MARVIN_URDF")
    if environment_path:
        return resolve_marvin_wuji_urdf(environment_path)

    roots = [
        os.environ.get("HF_HOME", ""),
        (os.path.join(os.environ["XDG_CACHE_HOME"], "huggingface") if os.environ.get("XDG_CACHE_HOME") else ""),
        os.path.expanduser("~/.cache/huggingface"),
    ]
    for root in filter(None, roots):
        matches = sorted(
            glob.glob(
                os.path.join(
                    root,
                    "hub/datasets--Genesis-Intelligence--internal_assets/snapshots/*/"
                    "*/marvin_robots/assemble/marvin_wuji_capsule_scaled.urdf",
                )
            )
        )
        if matches:
            return matches[-1]

    gs.raise_exception(
        "Marvin/Wuji URDF not found. Set an explicit URDF path, set "
        "QIPC_MARVIN_URDF, or download Genesis-Intelligence/internal_assets "
        f"from HuggingFace (searched: {[root for root in roots if root]})."
    )


def add_marvin_wuji(
    scene,
    *,
    urdf_path: str | None,
    robot_position: tuple[float, float, float],
    initial_arm_qpos: dict[str, tuple[float, ...]],
    initial_hand_qpos: dict[str, tuple[float, ...]] | None = None,
    kappa_pivot: float,
    kappa_axis: float,
) -> BuiltMarvinWuji:
    """Add the shared Marvin Wuji entity before a QIPC scene is built.

    `initial_hand_qpos` poses the fingers, ordered as `HAND_JOINTS[side]` -- five fingers
    thumb to pinky, four joints each. Omitted, they start flat at the URDF zero pose.
    """
    for side in ("right", "left"):
        if len(initial_arm_qpos[side]) != 7:
            raise ValueError(f"Expected 7 initial {side} arm joints.")
        if initial_hand_qpos is not None and len(initial_hand_qpos[side]) != len(HAND_JOINTS[side]):
            raise ValueError(
                f"Expected {len(HAND_JOINTS[side])} initial {side} hand joints, "
                f"got {len(initial_hand_qpos[side])}."
            )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=resolve_marvin_wuji_urdf(urdf_path),
            pos=robot_position,
            euler=(0.0, 0.0, 0.0),
            fixed=True,
            default_armature=None,
            merge_fixed_links=False,
            requires_jac_and_IK=True,
            convexify=True,
            collision=True,
            links_to_keep=list(PALM_LINK.values()),
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
            coup_friction=1.0,
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=kappa_pivot,
            qipc_kappa_axis=kappa_axis,
            qipc_d_hat=1e-3,
            qipc_self_contact=False,
        ),
    )
    dofs = {
        (kind, side): [robot.get_joint(name).dofs_idx_local[0] for name in names[side]]
        for kind, names in (("arm", ARM_JOINTS), ("hand", HAND_JOINTS))
        for side in ("right", "left")
    }
    home_qpos = np.zeros(robot.n_qs, dtype=np.float64)
    for side in ("right", "left"):
        home_qpos[dofs[("arm", side)]] = initial_arm_qpos[side]
        if initial_hand_qpos is not None:
            home_qpos[dofs[("hand", side)]] = initial_hand_qpos[side]
    robot.material.qipc_home_qpos = home_qpos.tolist()

    hand_effort = {
        side: np.asarray(
            [
                max(abs(float(value)) for value in robot.get_joint(name).dofs_force_range[0])
                for name in HAND_JOINTS[side]
            ],
            dtype=np.float64,
        )
        for side in ("right", "left")
    }
    return BuiltMarvinWuji(
        robot=robot,
        dofs=dofs,
        home_qpos=home_qpos,
        hand_effort=hand_effort,
    )


def configure_marvin_wuji(scene, built: BuiltMarvinWuji) -> None:
    """Configure QIPC joint drives after the scene is built."""
    for side in ("right", "left"):
        scene.sim.coupler.configure_dofs(
            built.robot,
            built.dofs[("arm", side)],
            kp=np.asarray(ARM_KP),
            kv=np.asarray(ARM_KV),
            force_lower=-ARM_FORCE_LIMIT,
            force_upper=ARM_FORCE_LIMIT,
        )
        effort = HAND_EFFORT_SCALE * built.hand_effort[side]
        scene.sim.coupler.configure_dofs(
            built.robot,
            built.dofs[("hand", side)],
            kp=HAND_KP,
            kv=HAND_KV,
            force_lower=-effort,
            force_upper=effort,
        )


def initialize_marvin_wuji(built: BuiltMarvinWuji) -> None:
    """Apply the configured QIPC home pose and matching drive targets."""
    built.robot.set_qpos(built.home_qpos)
    for indices in built.dofs.values():
        built.robot.control_dofs_position(built.home_qpos[indices], dofs_idx_local=indices)
