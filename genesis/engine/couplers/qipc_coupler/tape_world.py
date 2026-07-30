"""Reusable Marvin Wuji + QIPC adhesive-tape world.

Scene construction, QIPC-specific robot setup, control, stepping, and reset live
here. Interactive applications own only input binding and presentation.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.couplers.qipc_coupler.coupler import QIPCSolverStatistics
from genesis.engine.couplers.qipc_coupler.tape import (
    TapeAsset,
    add_tape_roll,
    recommended_coupler_options,
)
from genesis.utils.misc import get_assets_dir

TapeAdhesionMode = Literal["bond", "soft"]
TapeWorldSide = Literal["right", "left"]

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
INIT_ARM_DEG = {
    "right": (-110.0, -75.0, 90.0, -110.0, -75.0, 0.0, 0.0),
    "left": (110.0, -75.0, -90.0, -110.0, 75.0, 0.0, 0.0),
}
ARM_KP = (7200.0, 7200.0, 7200.0, 3600.0, 3600.0, 3600.0, 3600.0)
ARM_KV = (600.0, 600.0, 600.0, 400.0, 200.0, 200.0, 200.0)
HAND_KP = 50.0
HAND_KV = 0.5
HAND_EFFORT_SCALE = 8.0
ARM_FORCE_LIMIT = 2000.0


@dataclass(frozen=True)
class TapeWorldConfig:
    """Configuration for a reusable Marvin Wuji QIPC tape world."""

    mode: TapeAdhesionMode = "bond"
    sticky_hands: bool = True
    newton_velocity_tol: float = 0.01
    linear_tol_rate: float | None = None
    linear_max_iter: int | None = None
    dt: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.8)
    table_position: tuple[float, float, float] = (0.597, 0.0, 0.38)
    table_size: tuple[float, float, float] = (0.85, 1.5, 0.76)
    robot_position: tuple[float, float, float] = (0.0, 0.0, 1.08)
    roll_xy: tuple[float, float] = (0.379, -0.25)
    show_viewer: bool = True
    viewer_camera_position: tuple[float, float, float] = (1.1, -0.95, 1.35)
    viewer_camera_lookat: tuple[float, float, float] = (0.45, 0.0, 0.85)
    viewer_camera_fov: float = 45.0
    urdf_path: str | None = None
    tape_asset_path: str | None = None
    grip_speed: float = 2.0
    max_arm_command_step: float = 0.03

    def __post_init__(self) -> None:
        if self.mode not in ("bond", "soft"):
            raise ValueError(f"Unsupported tape adhesion mode: {self.mode!r}.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.newton_velocity_tol <= 0.0:
            raise ValueError("newton_velocity_tol must be positive.")
        if self.linear_tol_rate is not None and self.linear_tol_rate <= 0.0:
            raise ValueError("linear_tol_rate must be positive.")
        if self.linear_max_iter is not None and self.linear_max_iter <= 0:
            raise ValueError("linear_max_iter must be positive.")
        if self.grip_speed <= 0.0:
            raise ValueError("grip_speed must be positive.")
        if self.max_arm_command_step <= 0.0:
            raise ValueError("max_arm_command_step must be positive.")


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
        "Marvin/Wuji URDF not found. Set TapeWorldConfig.urdf_path, set "
        "QIPC_MARVIN_URDF, or download Genesis-Intelligence/internal_assets "
        f"from HuggingFace (searched: {[root for root in roots if root]})."
    )


def resolve_tape_asset_path(config: TapeWorldConfig) -> str:
    """Resolve the wound-tape asset selected by the configuration."""
    if config.tape_asset_path is not None:
        path = os.path.abspath(os.path.expanduser(config.tape_asset_path))
    else:
        suffix = {"bond": "lock", "soft": "soft"}[config.mode]
        path = os.path.join(get_assets_dir(), "qipc", f"tape_roll_{suffix}.npz")
    if not os.path.isfile(path):
        gs.raise_exception(f"QIPC tape asset does not exist: '{path}'.")
    return path


def tape_world_coupler_options(config: TapeWorldConfig, asset: TapeAsset) -> dict[str, object]:
    """Map tape-world configuration onto QIPCCouplerOptions keyword arguments."""
    options: dict[str, object] = recommended_coupler_options(asset)
    options["solver_newton_velocity_tol"] = config.newton_velocity_tol
    if config.mode == "soft":
        options["adhesion_bond_distance_lock"] = False
        options["adhesion_bond_max_bonds"] = 0
    if config.linear_tol_rate is not None:
        options["solver_linear_tol_rate"] = config.linear_tol_rate
    if config.linear_max_iter is not None:
        options["solver_linear_max_iter"] = config.linear_max_iter
    return options


class QIPCTapeWorld:
    """Built tape world with a device-independent arm/hand control surface."""

    def __init__(
        self,
        *,
        config: TapeWorldConfig,
        scene,
        robot,
        table,
        tape,
        hub,
        asset: TapeAsset,
        coupler_options: dict[str, object],
        dofs: dict[tuple[str, str], list[int]],
        home_qpos: np.ndarray,
        hand_postures: dict[tuple[str, bool], np.ndarray],
    ) -> None:
        self.config = config
        self.scene = scene
        self.robot = robot
        self.table = table
        self.tape = tape
        self.hub = hub
        self.asset = asset
        self.coupler_options = coupler_options
        self.dofs = dofs
        self.home_qpos = home_qpos
        self.hand_postures = hand_postures
        self.palms = {side: robot.get_link(PALM_LINK[side]) for side in ("right", "left")}
        self._initial_tape_positions = self.tape_positions()
        self._qpos_command = home_qpos.copy()
        self._closed = {"right": False, "left": False}
        self._hand_command = {side: hand_postures[(side, False)].copy() for side in ("right", "left")}
        self._target_position: dict[str, np.ndarray] = {}
        self._target_quaternion: dict[str, np.ndarray] = {}
        self.reset_targets()

    @property
    def max_grip_step(self) -> float:
        return self.config.grip_speed * self.config.dt

    def palm_position(self, side: TapeWorldSide) -> np.ndarray:
        return self.palms[side].get_pos().reshape(-1)[:3].cpu().numpy().copy()

    def palm_quaternion(self, side: TapeWorldSide) -> np.ndarray:
        return self.palms[side].get_quat().reshape(-1)[:4].cpu().numpy().copy()

    def palm_target(self, side: TapeWorldSide) -> tuple[np.ndarray, np.ndarray]:
        return self._target_position[side].copy(), self._target_quaternion[side].copy()

    def set_palm_target(
        self,
        side: TapeWorldSide,
        position: np.ndarray | tuple[float, float, float],
        quaternion: np.ndarray | tuple[float, float, float, float],
    ) -> None:
        position_array = np.asarray(position, dtype=gs.np_float)
        quaternion_array = np.asarray(quaternion, dtype=gs.np_float)
        if position_array.shape != (3,):
            raise ValueError(f"Palm position must have shape (3,), got {position_array.shape}.")
        if quaternion_array.shape != (4,):
            raise ValueError(f"Palm quaternion must have shape (4,), got {quaternion_array.shape}.")
        self._target_position[side] = position_array.copy()
        self._target_quaternion[side] = quaternion_array.copy()

    def move_palm_target(
        self,
        side: TapeWorldSide,
        delta_position: np.ndarray | tuple[float, float, float],
    ) -> None:
        delta = np.asarray(delta_position, dtype=gs.np_float)
        if delta.shape != (3,):
            raise ValueError(f"Palm position delta must have shape (3,), got {delta.shape}.")
        self._target_position[side] += delta

    def rotate_palm_target(self, side: TapeWorldSide, delta_euler: np.ndarray) -> None:
        delta = np.asarray(delta_euler, dtype=gs.np_float)
        if delta.shape != (3,):
            raise ValueError(f"Palm Euler delta must have shape (3,), got {delta.shape}.")
        self._target_quaternion[side] = gu.transform_quat_by_quat(
            self._target_quaternion[side],
            gu.xyz_to_quat(delta),
        )

    def reset_targets(self) -> None:
        for side in ("right", "left"):
            self._target_position[side] = self.palm_position(side)
            self._target_quaternion[side] = self.palm_quaternion(side)

    def set_grip(self, side: TapeWorldSide, closed: bool) -> None:
        self._closed[side] = closed

    def grip_is_closed(self, side: TapeWorldSide) -> bool:
        return self._closed[side]

    def hand_dofs_position(self, side: TapeWorldSide) -> np.ndarray:
        values = self.robot.get_dofs_position(dofs_idx_local=self.dofs[("hand", side)])
        return values.reshape(-1).cpu().numpy().copy()

    def hand_dofs_applied_force(self, side: TapeWorldSide) -> np.ndarray:
        values = self.scene.sim.coupler.get_dofs_applied_force(
            self.robot,
            self.dofs[("hand", side)],
        )
        return values.cpu().numpy().copy()

    def tape_positions(self) -> np.ndarray:
        return self.tape.get_state().pos[0].cpu().numpy().copy()

    def get_solver_statistics(self) -> QIPCSolverStatistics:
        return self.scene.sim.coupler.get_solver_statistics()

    def step(self) -> QIPCSolverStatistics:
        for side in ("right", "left"):
            arm_dofs = self.dofs[("arm", side)]
            qpos = self.robot.inverse_kinematics(
                link=self.palms[side],
                pos=self._target_position[side],
                quat=self._target_quaternion[side],
                init_qpos=self._qpos_command,
                max_samples=1,
                max_solver_iters=30,
                dofs_idx_local=arm_dofs,
            )
            qpos_array = qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos)
            delta = np.clip(
                qpos_array[arm_dofs] - self._qpos_command[arm_dofs],
                -self.config.max_arm_command_step,
                self.config.max_arm_command_step,
            )
            self._qpos_command[arm_dofs] += delta
            self.robot.control_dofs_position(
                self._qpos_command[arm_dofs],
                dofs_idx_local=arm_dofs,
            )

            goal = self.hand_postures[(side, self._closed[side])]
            self._hand_command[side] += np.clip(
                goal - self._hand_command[side],
                -self.max_grip_step,
                self.max_grip_step,
            )
            self.robot.control_dofs_position(
                self._hand_command[side],
                dofs_idx_local=self.dofs[("hand", side)],
            )

        self.scene.step()
        return self.get_solver_statistics()

    def reset(self) -> None:
        """Restore robot, tape, hub, controller commands, and palm targets."""
        self.scene.reset()
        self.robot.set_qpos(self.home_qpos)
        self._qpos_command[:] = self.home_qpos
        for side in ("right", "left"):
            self._closed[side] = False
            self._hand_command[side][:] = self.hand_postures[(side, False)]
            self.robot.control_dofs_position(
                self.home_qpos[self.dofs[("arm", side)]],
                dofs_idx_local=self.dofs[("arm", side)],
            )
            self.robot.control_dofs_position(
                self._hand_command[side],
                dofs_idx_local=self.dofs[("hand", side)],
            )
        self.reset_targets()

    def reset_error(self) -> float:
        """Maximum absolute tape-position error from the built initial state."""
        return float(np.abs(self.tape_positions() - self._initial_tape_positions).max())


def _hand_postures(robot) -> dict[tuple[str, bool], np.ndarray]:
    postures: dict[tuple[str, bool], np.ndarray] = {}
    for side in ("right", "left"):
        for closed in (False, True):
            values = []
            for name in HAND_JOINTS[side]:
                finger = int(name.split("finger", 1)[1].split("_", 1)[0])
                joint = int(name.rsplit("joint", 1)[1])
                lower, upper = (float(value) for value in robot.get_joint(name).dofs_limit[0])
                if closed and finger == 1:
                    desired = (0.9, 0.35, 0.9, 0.9)[joint - 1]
                elif closed and finger == 2:
                    desired = (0.95, 0.0, 0.95, 0.95)[joint - 1]
                else:
                    desired = 0.08 if finger == 1 and joint == 1 else 0.0
                values.append(min(max(desired, lower), upper))
            postures[(side, closed)] = np.asarray(values, dtype=np.float64)
    return postures


def build_qipc_tape_world(config: TapeWorldConfig) -> QIPCTapeWorld:
    """Build and initialize the reusable Marvin Wuji QIPC tape world."""
    asset = TapeAsset.from_npz(resolve_tape_asset_path(config))
    coupler_options = tape_world_coupler_options(config, asset)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.dt, gravity=config.gravity),
        coupler_options=gs.options.QIPCCouplerOptions(**coupler_options),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=config.viewer_camera_position,
            camera_lookat=config.viewer_camera_lookat,
            camera_fov=config.viewer_camera_fov,
        ),
        show_viewer=config.show_viewer,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=resolve_marvin_wuji_urdf(config.urdf_path),
            pos=config.robot_position,
            euler=(0.0, 0.0, 0.0),
            fixed=True,
            merge_fixed_links=False,
            requires_jac_and_IK=True,
            convexify=True,
            collision=True,
            links_to_keep=list(PALM_LINK.values()),
        ),
        material=gs.materials.Rigid(
            coup_friction=1.0,
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=1e7,
            qipc_kappa_axis=1e7,
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
        home_qpos[dofs[("arm", side)]] = np.deg2rad(INIT_ARM_DEG[side])
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

    table = scene.add_entity(
        morph=gs.morphs.Box(
            pos=config.table_position,
            size=config.table_size,
            fixed=True,
            collision=True,
        ),
        material=gs.materials.Rigid(coup_friction=0.8),
        surface=gs.surfaces.Default(color=(0.62, 0.6, 0.58)),
    )

    table_top = config.table_position[2] + 0.5 * config.table_size[2]
    lowest = min(float(asset.tape_positions[:, 2].min()), -0.5 * asset.hub_height)
    roll_z = table_top - lowest + asset.thick + 0.5 * asset.d_hat
    tape, hub = add_tape_roll(
        scene,
        asset,
        pos=(config.roll_xy[0], config.roll_xy[1], roll_z),
        euler=(0.0, 0.0, 0.0),
        with_hub=True,
        hub_fixed=False,
        tape_surface=gs.surfaces.Plastic(color=(0.85, 0.75, 0.3, 1.0)),
        hub_surface=gs.surfaces.Plastic(color=(0.4, 0.25, 0.15, 1.0)),
    )
    if config.sticky_hands:
        scene.sim.coupler.add_adhesion(
            tape,
            robot,
            Cn=float(asset.params.get("CN", 1.0)),
            Ct=float(asset.params.get("CT", 1.0)),
            W=float(asset.params.get("ADH_W", 1.0)),
            eta=float(asset.params.get("ETA", 100.0)),
            bonding_rate=1.0,
            beta0=1.0,
            friction=1.0,
        )

    scene.build()

    for side in ("right", "left"):
        scene.sim.coupler.configure_dofs(
            robot,
            dofs[("arm", side)],
            kp=np.asarray(ARM_KP),
            kv=np.asarray(ARM_KV),
            force_lower=-ARM_FORCE_LIMIT,
            force_upper=ARM_FORCE_LIMIT,
        )
        effort = HAND_EFFORT_SCALE * hand_effort[side]
        scene.sim.coupler.configure_dofs(
            robot,
            dofs[("hand", side)],
            kp=HAND_KP,
            kv=HAND_KV,
            force_lower=-effort,
            force_upper=effort,
        )

    robot.set_qpos(home_qpos)
    for indices in dofs.values():
        robot.control_dofs_position(home_qpos[indices], dofs_idx_local=indices)

    world = QIPCTapeWorld(
        config=config,
        scene=scene,
        robot=robot,
        table=table,
        tape=tape,
        hub=hub,
        asset=asset,
        coupler_options=coupler_options,
        dofs=dofs,
        home_qpos=home_qpos,
        hand_postures=_hand_postures(robot),
    )

    palm = world.palm_position("right")
    if np.linalg.norm(palm[:2] - np.asarray(config.roll_xy)) > 0.05:
        gs.logger.warning(
            f"Right palm rests at {np.round(palm[:2], 3)} but the roll was placed at "
            f"{np.round(config.roll_xy, 3)}; update TapeWorldConfig.roll_xy for this home pose."
        )
    gs.logger.info(
        f"QIPC tape world mode={config.mode} roll={np.round(config.roll_xy, 3)} "
        f"z={roll_z:.4f} right_palm={np.round(palm, 3)}"
    )
    return world
