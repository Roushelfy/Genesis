"""Reusable free-base Wuji hands QIPC world."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

import genesis as gs
from genesis.engine.couplers.qipc_coupler.coupler import QIPCSolverStatistics

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity

HandSide = Literal["right", "left"]
HAND_SIDES: tuple[HandSide, ...] = ("right", "left")
HAND_DOF = 20
PALM_LINK = {side: f"{side}_palm_link" for side in HAND_SIDES}
FINGERTIP_LINKS = {side: tuple(f"{side}_finger{finger}_tip_link" for finger in range(1, 6)) for side in HAND_SIDES}
HAND_JOINTS = {
    side: tuple(f"{side}_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))
    for side in HAND_SIDES
}


@dataclass(frozen=True)
class WujiHandsWorldConfig:
    """Configuration for two independently STC-driven Wuji hands."""

    right_urdf_path: str
    left_urdf_path: str
    right_wrist_position: tuple[float, float, float]
    right_wrist_quaternion: tuple[float, float, float, float]
    left_wrist_position: tuple[float, float, float]
    left_wrist_quaternion: tuple[float, float, float, float]
    newton_velocity_tol: float = 0.01
    linear_tol_rate: float | None = None
    linear_max_iter: int | None = None
    kappa_pivot: float = 1e7
    kappa_axis: float = 1e7
    finger_kp: float = 50.0
    finger_kv: float = 0.5
    finger_effort_scale: float = 8.0
    stc_translation_strength: float = 1e4
    stc_rotation_strength: float = 1e4
    dt: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.8)
    show_viewer: bool = True
    viewer_camera_position: tuple[float, float, float] = (1.1, -0.95, 1.35)
    viewer_camera_lookat: tuple[float, float, float] = (0.4, 0.0, 0.9)
    viewer_camera_fov: float = 45.0

    def __post_init__(self) -> None:
        for path in (self.right_urdf_path, self.left_urdf_path):
            if not os.path.isfile(path):
                raise ValueError(f"Wuji hand URDF does not exist: '{path}'.")
        positive_values = {
            "dt": self.dt,
            "newton_velocity_tol": self.newton_velocity_tol,
            "kappa_pivot": self.kappa_pivot,
            "kappa_axis": self.kappa_axis,
            "finger_kp": self.finger_kp,
            "finger_effort_scale": self.finger_effort_scale,
            "stc_translation_strength": self.stc_translation_strength,
            "stc_rotation_strength": self.stc_rotation_strength,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.finger_kv < 0.0:
            raise ValueError("finger_kv must be non-negative.")
        if self.linear_tol_rate is not None and self.linear_tol_rate <= 0.0:
            raise ValueError("linear_tol_rate must be positive.")
        if self.linear_max_iter is not None and self.linear_max_iter <= 0:
            raise ValueError("linear_max_iter must be positive.")
        for side, quaternion in (
            ("right", self.right_wrist_quaternion),
            ("left", self.left_wrist_quaternion),
        ):
            if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-5):
                raise ValueError(f"Initial {side} wrist quaternion must be normalized.")


@dataclass(frozen=True)
class BuiltWujiHand:
    entity: RigidEntity
    finger_dofs: tuple[int, ...]
    home_qpos: np.ndarray
    home_finger_qpos: np.ndarray


class QIPCWujiHandsWorld:
    """Built two-hand world with explicit STC and finger-controller state."""

    def __init__(self, *, config: WujiHandsWorldConfig, scene, hands: dict[HandSide, BuiltWujiHand]) -> None:
        self.config = config
        self.scene = scene
        self.hands = hands
        self.entities = {side: hand.entity for side, hand in hands.items()}
        self.palms = {side: hand.entity.get_link(PALM_LINK[side]) for side, hand in hands.items()}
        self.fingertips = {
            side: tuple(hand.entity.get_link(name) for name in FINGERTIP_LINKS[side]) for side, hand in hands.items()
        }
        self._wrist_targets = {
            "right": (
                np.asarray(config.right_wrist_position, dtype=np.float64),
                np.asarray(config.right_wrist_quaternion, dtype=np.float64),
            ),
            "left": (
                np.asarray(config.left_wrist_position, dtype=np.float64),
                np.asarray(config.left_wrist_quaternion, dtype=np.float64),
            ),
        }
        self._finger_targets = {side: hand.home_finger_qpos.copy() for side, hand in hands.items()}

    def set_wrist_target(self, side: HandSide, position, quaternion) -> None:
        position_array = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion_array = np.asarray(quaternion, dtype=np.float64).reshape(4)
        self._wrist_targets[side] = (position_array.copy(), quaternion_array.copy())
        self.scene.sim.coupler.set_soft_transform_target(
            self.entities[side],
            position_array,
            quaternion_array,
            enabled=True,
            link=PALM_LINK[side],
        )

    def set_finger_target(self, side: HandSide, joint_positions) -> None:
        target = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if target.shape != (HAND_DOF,):
            raise ValueError(f"Expected {HAND_DOF} {side} finger targets, got {target.shape}.")
        self._finger_targets[side] = target.copy()
        self.entities[side].control_dofs_position(target, dofs_idx_local=self.hands[side].finger_dofs)

    def palm_position(self, side: HandSide) -> np.ndarray:
        return self.palms[side].get_pos().reshape(-1)[:3].cpu().numpy().copy()

    def palm_quaternion(self, side: HandSide) -> np.ndarray:
        return self.palms[side].get_quat().reshape(-1)[:4].cpu().numpy().copy()

    def finger_dofs_position(self, side: HandSide) -> np.ndarray:
        values = self.entities[side].get_dofs_position(dofs_idx_local=self.hands[side].finger_dofs)
        return values.reshape(-1).cpu().numpy().copy()

    def get_solver_statistics(self) -> QIPCSolverStatistics:
        return self.scene.sim.coupler.get_solver_statistics()

    def step(self) -> QIPCSolverStatistics:
        self.scene.step()
        return self.get_solver_statistics()

    def reset(self) -> None:
        self.scene.reset()
        for side in HAND_SIDES:
            position, quaternion = self._initial_wrist_target(side)
            self.set_wrist_target(side, position, quaternion)
            self.set_finger_target(side, self.hands[side].home_finger_qpos)

    def _initial_wrist_target(self, side: HandSide) -> tuple[np.ndarray, np.ndarray]:
        if side == "right":
            position = self.config.right_wrist_position
            quaternion = self.config.right_wrist_quaternion
        else:
            position = self.config.left_wrist_position
            quaternion = self.config.left_wrist_quaternion
        return np.asarray(position, dtype=np.float64), np.asarray(quaternion, dtype=np.float64)


def wuji_hands_coupler_options(config: WujiHandsWorldConfig) -> dict[str, object]:
    """Map Wuji-hands configuration onto QIPCCouplerOptions arguments."""
    options: dict[str, object] = {
        "contact_enable": False,
        "solver_newton_velocity_tol": config.newton_velocity_tol,
    }
    if config.linear_tol_rate is not None:
        options["solver_linear_tol_rate"] = config.linear_tol_rate
    if config.linear_max_iter is not None:
        options["solver_linear_max_iter"] = config.linear_max_iter
    return options


def _add_hand(scene, config: WujiHandsWorldConfig, side: HandSide) -> BuiltWujiHand:
    if side == "right":
        urdf_path = config.right_urdf_path
        position = config.right_wrist_position
        quaternion = config.right_wrist_quaternion
    else:
        urdf_path = config.left_urdf_path
        position = config.left_wrist_position
        quaternion = config.left_wrist_quaternion

    entity = scene.add_entity(
        morph=gs.morphs.URDF(
            file=urdf_path,
            pos=position,
            quat=quaternion,
            fixed=False,
            merge_fixed_links=False,
            requires_jac_and_IK=False,
            convexify=True,
            collision=True,
            links_to_keep=[PALM_LINK[side]],
        ),
        material=gs.materials.Rigid(
            coup_friction=1.0,
            qipc_abd_kappa=1e8,
            qipc_kappa_pivot=config.kappa_pivot,
            qipc_kappa_axis=config.kappa_axis,
            qipc_d_hat=1e-3,
            qipc_self_contact=False,
        ),
    )
    finger_dofs = tuple(entity.get_joint(name).dofs_idx_local[0] for name in HAND_JOINTS[side])
    home_qpos = np.asarray(entity.init_qpos, dtype=np.float64).copy()
    home_finger_qpos = np.asarray(
        [np.clip(0.0, *entity.get_joint(name).dofs_limit[0]) for name in HAND_JOINTS[side]],
        dtype=np.float64,
    )
    for name, value in zip(HAND_JOINTS[side], home_finger_qpos, strict=True):
        joint = entity.get_joint(name)
        home_qpos[joint.q_start - entity.q_start] = value
    entity.material.qipc_home_qpos = home_qpos.tolist()
    scene.sim.coupler.enable_soft_transform(
        entity,
        link=PALM_LINK[side],
        strength=(config.stc_translation_strength, config.stc_rotation_strength),
    )
    return BuiltWujiHand(
        entity=entity,
        finger_dofs=finger_dofs,
        home_qpos=home_qpos,
        home_finger_qpos=home_finger_qpos,
    )


def _configure_hand(scene, config: WujiHandsWorldConfig, side: HandSide, hand: BuiltWujiHand) -> None:
    effort = config.finger_effort_scale * np.asarray(
        [
            max(abs(float(value)) for value in hand.entity.get_joint(name).dofs_force_range[0])
            for name in HAND_JOINTS[side]
        ],
        dtype=np.float64,
    )
    scene.sim.coupler.configure_dofs(
        hand.entity,
        hand.finger_dofs,
        kp=config.finger_kp,
        kv=config.finger_kv,
        force_lower=-effort,
        force_upper=effort,
    )


def build_qipc_wuji_hands_world(config: WujiHandsWorldConfig) -> QIPCWujiHandsWorld:
    """Build and initialize two independent free-base Wuji hands."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.dt, gravity=config.gravity),
        coupler_options=gs.options.QIPCCouplerOptions(**wuji_hands_coupler_options(config)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=config.viewer_camera_position,
            camera_lookat=config.viewer_camera_lookat,
            camera_fov=config.viewer_camera_fov,
        ),
        show_viewer=config.show_viewer,
    )
    hands = {side: _add_hand(scene, config, side) for side in HAND_SIDES}
    scene.build()
    for side, hand in hands.items():
        _configure_hand(scene, config, side, hand)

    world = QIPCWujiHandsWorld(config=config, scene=scene, hands=hands)
    for side in HAND_SIDES:
        position, quaternion = world._initial_wrist_target(side)
        world.set_wrist_target(side, position, quaternion)
        world.set_finger_target(side, hands[side].home_finger_qpos)
    return world
