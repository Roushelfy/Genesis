"""
Shared robot teleop module for IPC examples.

Provides keyboard-driven end-effector control (position + orientation)
with gripper open/close via the Genesis viewer keybind system.
Works with any robot that has an end-effector link and gripper DOFs.

Keyboard Controls:
    Arrow keys  - Move in XY plane
    n / m       - Move Up / Down
    j / k       - Yaw Left / Right
    u / o       - Pitch Up / Down
    l / ;       - Roll Left / Right
    g           - Reset to initial pose
    space       - Hold to close gripper, release to open
    esc         - Quit

Usage:
    from robot_teleop import RobotTeleop

    # Franka preset (all Franka-specific defaults built in):
    teleop = RobotTeleop.franka(scene, franka, init_pos=(0.5, 0.0, 0.6))
    teleop.setup()
    teleop.run()

    # Generic robot:
    teleop = RobotTeleop(
        scene=scene,
        robot=my_robot,
        ee_link_name="tool0",
        motor_dofs_idx=range(6),
        gripper_dofs_idx=range(6, 8),
    )
    teleop.setup()
    teleop.run()
"""

import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind


class RobotTeleop:
    """Keyboard teleop controller for any robot with an end-effector and gripper."""

    def __init__(
        self,
        scene,
        robot,
        ee_link_name,
        motor_dofs_idx,
        gripper_dofs_idx,
        init_pos=(0.5, 0.0, 0.6),
        init_euler=(0.0, 180.0, 0.0),
        gripper_kp=500.0,
        gripper_kv=50.0,
        gripper_init_pos=0.04,
        gripper_open_vel=0.1,
        gripper_close_vel=-0.1,
        delta_pos=0.003,
        delta_rot=0.02,
        on_step=None,
    ):
        """
        Args:
            scene: Genesis scene.
            robot: Robot entity (any articulated entity with IK support).
            ee_link_name: Name of the end-effector link for IK.
            motor_dofs_idx: Indices (or slice) for the arm motor DOFs.
            gripper_dofs_idx: Indices (or slice) for the gripper DOFs.
            init_pos: Initial end-effector position (x, y, z).
            init_euler: Initial end-effector orientation in degrees (rx, ry, rz).
            gripper_kp: Proportional gain for gripper DOFs.
            gripper_kv: Derivative gain for gripper DOFs.
            gripper_open_pos: Target position for open gripper.
            gripper_close_force: Force applied when closing gripper.
            delta_pos: Position increment per key frame.
            delta_rot: Rotation increment per key frame (radians).
            on_step: Optional callback called each step with (teleop_instance,).
        """
        self.scene = scene
        self.robot = robot
        self.ee_link = robot.get_link(ee_link_name)
        self.motor_dofs_idx = motor_dofs_idx
        self.gripper_dofs_idx = gripper_dofs_idx
        self.gripper_kp = gripper_kp
        self.gripper_kv = gripper_kv
        self.gripper_init_pos = gripper_init_pos
        self.gripper_open_vel = gripper_open_vel
        self.gripper_close_vel = gripper_close_vel
        self.delta_pos = delta_pos
        self.delta_rot = delta_rot
        self.on_step = on_step

        self._init_pos = np.array(init_pos, dtype=gs.np_float)
        self._init_quat = gu.xyz_to_quat(np.array(init_euler, dtype=gs.np_float), degrees=True)
        self.target_pos = self._init_pos.copy()
        self.target_quat = self._init_quat.copy()

        self._gripper_close = np.array(False, dtype=gs.np_bool)
        self._is_running = True
        self._target_ik = None

    def setup(self, set_initial_qpos=True):
        """Configure gripper PD gains, set initial qpos, draw debug frame, and register keybinds.

        Call this after scene.build().
        """
        self.robot.set_dofs_kp(self.gripper_kp, dofs_idx_local=self.gripper_dofs_idx)
        self.robot.set_dofs_kv(self.gripper_kv, dofs_idx_local=self.gripper_dofs_idx)

        if set_initial_qpos:
            qpos = self.robot.inverse_kinematics(
                link=self.ee_link,
                pos=self.target_pos,
                quat=self.target_quat,
                dofs_idx_local=self.motor_dofs_idx,
            )
            qpos[self.gripper_dofs_idx] = self.gripper_init_pos
            self.robot.set_qpos(qpos)
            self.robot.control_dofs_position(qpos)

        self._target_ik = self.scene.draw_debug_frame(
            T=gu.trans_quat_to_T(self.target_pos, self.target_quat),
            axis_length=0.15,
            origin_size=0.01,
            axis_radius=0.007,
        )
        self.scene.viewer.update(force=True)

        self._register_keybinds()

    def _register_keybinds(self):
        dp = self.delta_pos
        dr = self.delta_rot

        def move(dpos_xyz):
            self.target_pos[:] += dpos_xyz

        def rotate(axis_idx, delta):
            delta_xyz = np.zeros(3, dtype=gs.np_float)
            delta_xyz[axis_idx] = delta
            delta_quat = gu.xyz_to_quat(delta_xyz)
            self.target_quat[:] = gu.transform_quat_by_quat(self.target_quat, delta_quat)

        def reset():
            self.target_pos[:] = self._init_pos
            self.target_quat[:] = self._init_quat
            pose = gu.trans_quat_to_T(self.target_pos, self.target_quat)
            self.scene.visualizer.context.update_debug_objects((self._target_ik,), (pose,))
            qpos = self.robot.inverse_kinematics(
                link=self.ee_link,
                pos=self.target_pos,
                quat=self.target_quat,
                dofs_idx_local=self.motor_dofs_idx,
            )
            self.robot.control_dofs_position(qpos[self.motor_dofs_idx], self.motor_dofs_idx)

        def set_gripper(close):
            self._gripper_close[()] = close

        def stop():
            self._is_running = False

        self.scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((dp, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((-dp, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, dp, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, -dp, 0),)),
            Keybind("move_up", Key.N, KeyAction.HOLD, callback=move, args=((0, 0, dp),)),
            Keybind("move_down", Key.M, KeyAction.HOLD, callback=move, args=((0, 0, -dp),)),
            Keybind("yaw_left", Key.J, KeyAction.HOLD, callback=rotate, args=(2, dr)),
            Keybind("yaw_right", Key.K, KeyAction.HOLD, callback=rotate, args=(2, -dr)),
            Keybind("pitch_up", Key.U, KeyAction.HOLD, callback=rotate, args=(1, dr)),
            Keybind("pitch_down", Key.O, KeyAction.HOLD, callback=rotate, args=(1, -dr)),
            Keybind("roll_left", Key.L, KeyAction.HOLD, callback=rotate, args=(0, dr)),
            Keybind("roll_right", Key.SEMICOLON, KeyAction.HOLD, callback=rotate, args=(0, -dr)),
            Keybind("reset_scene", Key.G, KeyAction.PRESS, callback=reset),
            Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=set_gripper, args=(True,)),
            Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=set_gripper, args=(False,)),
            Keybind("quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
            overwrite=True,
        )

    def step(self):
        """Execute one teleop step: update IK target, control gripper, step scene."""
        # Update debug frame visualization
        pose = gu.trans_quat_to_T(self.target_pos, self.target_quat)
        self.scene.visualizer.context.update_debug_objects((self._target_ik,), (pose,))

        # Solve IK and send motor commands
        qpos = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=self.target_pos,
            quat=self.target_quat,
            dofs_idx_local=self.motor_dofs_idx,
        )
        self.robot.control_dofs_position(qpos[self.motor_dofs_idx], self.motor_dofs_idx)

        # Gripper control
        if self._gripper_close[()]:
            self.robot.control_dofs_velocity(self.gripper_close_vel, dofs_idx_local=self.gripper_dofs_idx)
        else:
            self.robot.control_dofs_velocity(self.gripper_open_vel, dofs_idx_local=self.gripper_dofs_idx)

        # Optional user callback
        if self.on_step is not None:
            self.on_step(self)

        self.scene.step()

    def run(self):
        """Run the teleop loop until quit or viewer closes."""
        if self.scene.viewer is None:
            gs.logger.warning("Viewer is not active. Keyboard input requires the Genesis viewer.")
            return

        try:
            while self._is_running and self.scene.viewer.is_alive():
                self.step()

                if "PYTEST_VERSION" in os.environ:
                    break
        except KeyboardInterrupt:
            gs.logger.info("Simulation interrupted, exiting.")
        finally:
            gs.logger.info("Simulation finished.")

    @classmethod
    def franka(cls, scene, robot, **kwargs):
        """Preset for Franka Emika Panda."""
        defaults = dict(
            ee_link_name="hand",
            motor_dofs_idx=slice(0, 7),
            gripper_dofs_idx=slice(7, 9),
            gripper_kp=500.0,
            gripper_kv=50.0,
            gripper_init_pos=0.04,
            gripper_open_vel=0.1,
            gripper_close_vel=-0.1,
        )
        defaults.update(kwargs)
        return cls(scene=scene, robot=robot, **defaults)

    @classmethod
    def marvin_right(cls, scene, robot, **kwargs):
        """Preset for Marvin single right arm with Pika gripper."""
        defaults = dict(
            ee_link_name="Gripper_Tip_R",
            motor_dofs_idx=slice(0, 7),
            gripper_dofs_idx=slice(7, 9),
            gripper_kp=500.0,
            gripper_kv=50.0,
            gripper_init_pos=0.0475,
            gripper_open_vel=0.1,
            gripper_close_vel=-0.1,
            init_euler=(0.0, 0.0, 0.0),
        )
        defaults.update(kwargs)
        return cls(scene=scene, robot=robot, **defaults)

    @classmethod
    def marvin_left(cls, scene, robot, **kwargs):
        """Preset for Marvin single left arm with Pika gripper."""
        defaults = dict(
            ee_link_name="Gripper_Tip_L",
            motor_dofs_idx=slice(0, 7),
            gripper_dofs_idx=slice(7, 9),
            gripper_kp=500.0,
            gripper_kv=50.0,
            gripper_init_pos=0.0475,
            gripper_open_vel=0.1,
            gripper_close_vel=-0.1,
            init_euler=(0.0, 0.0, 0.0),
        )
        defaults.update(kwargs)
        return cls(scene=scene, robot=robot, **defaults)
