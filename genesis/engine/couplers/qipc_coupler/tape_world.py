"""Reusable Marvin Wuji + QIPC adhesive-tape world.

Scene construction, QIPC-specific robot setup, physics stepping, and reset live
here. Applications own robot command generation and rate limiting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

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
from genesis.engine.couplers.qipc_coupler.tape import (
    TapeAsset,
    add_tape_roll,
    recommended_coupler_options,
)
from genesis.utils.misc import get_assets_dir

TapeAdhesionMode = Literal["bond", "soft"]
TapeWorldSide = Literal["right", "left"]


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
    """Built tape world with QIPC physics lifecycle and state access."""

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
        self.palms = {side: robot.get_link(PALM_LINK[side]) for side in ("right", "left")}
        self._initial_tape_positions = self.tape_positions()

    def palm_position(self, side: TapeWorldSide) -> np.ndarray:
        return self.palms[side].get_pos().reshape(-1)[:3].cpu().numpy().copy()

    def palm_quaternion(self, side: TapeWorldSide) -> np.ndarray:
        return self.palms[side].get_quat().reshape(-1)[:4].cpu().numpy().copy()

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

    def tape_velocities(self) -> np.ndarray:
        return self.tape.get_state().vel[0].cpu().numpy().copy()

    def get_solver_statistics(self) -> QIPCSolverStatistics:
        return self.scene.sim.coupler.get_solver_statistics()

    def step(self) -> QIPCSolverStatistics:
        self.scene.step()
        return self.get_solver_statistics()

    def reset(self) -> None:
        """Restore the built QIPC scene in place."""
        self.scene.reset()

    def reset_error(self) -> float:
        """Maximum absolute tape-position error from the built initial state."""
        return float(np.abs(self.tape_positions() - self._initial_tape_positions).max())


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

    built_robot = add_marvin_wuji(
        scene,
        urdf_path=config.urdf_path,
        robot_position=config.robot_position,
        initial_arm_qpos=DEFAULT_INIT_ARM_QPOS,
        kappa_pivot=1e7,
        kappa_axis=1e7,
    )
    robot = built_robot.robot
    dofs = built_robot.dofs
    home_qpos = built_robot.home_qpos

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

    configure_marvin_wuji(scene, built_robot)
    initialize_marvin_wuji(built_robot)

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
