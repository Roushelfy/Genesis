"""Reusable robot-only Marvin Wuji QIPC world."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import genesis as gs
from genesis.engine.couplers.qipc_coupler.coupler import QIPCSolverStatistics
from genesis.engine.couplers.qipc_coupler.marvin_wuji import (
    DEFAULT_INIT_ARM_QPOS,
    PALM_LINK,
    add_marvin_wuji,
    configure_marvin_wuji,
    initialize_marvin_wuji,
)


@dataclass(frozen=True)
class RobotWorldConfig:
    """Configuration for a robot-only Marvin Wuji QIPC world."""

    newton_velocity_tol: float = 0.01
    linear_tol_rate: float | None = None
    linear_max_iter: int | None = None
    kappa_pivot: float = 1e7
    kappa_axis: float = 1e7
    dt: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.8)
    robot_position: tuple[float, float, float] = (0.0, 0.0, 1.08)
    initial_right_arm_qpos: tuple[float, ...] = DEFAULT_INIT_ARM_QPOS["right"]
    initial_left_arm_qpos: tuple[float, ...] = DEFAULT_INIT_ARM_QPOS["left"]
    show_viewer: bool = True
    viewer_camera_position: tuple[float, float, float] = (1.1, -0.95, 1.35)
    viewer_camera_lookat: tuple[float, float, float] = (0.45, 0.0, 0.85)
    viewer_camera_fov: float = 45.0
    urdf_path: str | None = None

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.newton_velocity_tol <= 0.0:
            raise ValueError("newton_velocity_tol must be positive.")
        if self.linear_tol_rate is not None and self.linear_tol_rate <= 0.0:
            raise ValueError("linear_tol_rate must be positive.")
        if self.linear_max_iter is not None and self.linear_max_iter <= 0:
            raise ValueError("linear_max_iter must be positive.")
        if self.kappa_pivot <= 0.0:
            raise ValueError("kappa_pivot must be positive.")
        if self.kappa_axis <= 0.0:
            raise ValueError("kappa_axis must be positive.")
        if len(self.initial_right_arm_qpos) != 7 or len(self.initial_left_arm_qpos) != 7:
            raise ValueError("Initial arm configurations must contain 7 joints per side.")


class QIPCRobotWorld:
    """Built robot-only world with QIPC physics lifecycle and state access."""

    def __init__(self, *, config: RobotWorldConfig, scene, built_robot) -> None:
        self.config = config
        self.scene = scene
        self.robot = built_robot.robot
        self.dofs = built_robot.dofs
        self.home_qpos = built_robot.home_qpos
        self.palms = {side: self.robot.get_link(PALM_LINK[side]) for side in ("right", "left")}

    def palm_position(self, side: str) -> np.ndarray:
        return self.palms[side].get_pos().reshape(-1)[:3].cpu().numpy().copy()

    def hand_dofs_position(self, side: str) -> np.ndarray:
        values = self.robot.get_dofs_position(dofs_idx_local=self.dofs[("hand", side)])
        return values.reshape(-1).cpu().numpy().copy()

    def get_solver_statistics(self) -> QIPCSolverStatistics:
        return self.scene.sim.coupler.get_solver_statistics()

    def step(self) -> QIPCSolverStatistics:
        self.scene.step()
        return self.get_solver_statistics()

    def reset(self) -> None:
        self.scene.reset()


def robot_world_coupler_options(config: RobotWorldConfig) -> dict[str, object]:
    """Map robot-world configuration onto QIPCCouplerOptions keyword arguments."""
    options: dict[str, object] = {
        "contact_enable": False,
        "solver_newton_velocity_tol": config.newton_velocity_tol,
    }
    if config.linear_tol_rate is not None:
        options["solver_linear_tol_rate"] = config.linear_tol_rate
    if config.linear_max_iter is not None:
        options["solver_linear_max_iter"] = config.linear_max_iter
    return options


def build_qipc_robot_world(config: RobotWorldConfig) -> QIPCRobotWorld:
    """Build and initialize a robot-only Marvin Wuji QIPC world."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.dt, gravity=config.gravity),
        coupler_options=gs.options.QIPCCouplerOptions(**robot_world_coupler_options(config)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=config.viewer_camera_position,
            camera_lookat=config.viewer_camera_lookat,
            camera_fov=config.viewer_camera_fov,
        ),
        show_viewer=config.show_viewer,
    )
    built_robot = add_marvin_wuji(
        scene,
        urdf_path=config.urdf_path,
        robot_position=config.robot_position,
        initial_arm_qpos={
            "right": config.initial_right_arm_qpos,
            "left": config.initial_left_arm_qpos,
        },
        kappa_pivot=config.kappa_pivot,
        kappa_axis=config.kappa_axis,
    )
    scene.build()
    configure_marvin_wuji(scene, built_robot)
    initialize_marvin_wuji(built_robot)
    return QIPCRobotWorld(config=config, scene=scene, built_robot=built_robot)
