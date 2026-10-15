"""Use the authored Panda inertias/joints in MuJoCo, with external contact loads."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def skeleton_xml(path):
    root = ET.fromstring(Path(path).read_bytes())
    # Explicit authored inertias remain. Collision surfaces are supplied by the
    # independently qualified contact reference; no mesh assets are required here.
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"geom", "asset", "light", "camera"}:
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")


class PandaSkeleton:
    def __init__(self, path, timestep):
        self.xml = skeleton_xml(path)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.model.opt.timestep = timestep
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco.MjData(self.model)
        assert self.model.nq == self.model.nv == 9 and self.model.neq == 1
        self.body_ids = {name: self.model.body(name).id for name in ["hand", "left_finger", "right_finger"]}

    def set_state(self, q, velocity):
        self.data.qpos[:] = q
        self.data.qvel[:] = velocity
        self.data.qfrc_applied[:] = 0
        mujoco.mj_forward(self.model, self.data)
        assert not np.any(self.data.qfrc_actuator)

    def mass(self):
        matrix = np.empty((9, 9))
        for column in range(9):
            direction = np.zeros(9)
            direction[column] = 1
            product = np.empty(9)
            mujoco.mj_mulM(self.model, self.data, product, direction)
            matrix[:, column] = product
        return matrix

    def jacobian(self, name):
        linear, angular = np.empty((3, 9)), np.empty((3, 9))
        mujoco.mj_jacBody(self.model, self.data, linear, angular, self.body_ids[name])
        return np.vstack((linear, angular))

    def point_jacobian(self, name, world_point):
        linear, angular = np.empty((3, 9)), np.empty((3, 9))
        mujoco.mj_jac(self.model, self.data, linear, angular, np.asarray(world_point), self.body_ids[name])
        return np.vstack((linear, angular))

    def step_velocity(self, q, velocity, effort):
        data = mujoco.MjData(self.model)
        data.qpos[:] = q
        data.qvel[:] = velocity
        data.qfrc_applied[:] = effort
        mujoco.mj_step(self.model, data)
        assert data.ncon == 0
        return data.qvel.copy()

    def velocity_response(self, q, velocity):
        """Exact affine effort response for the current active constraint set.

        This is measured from independent MuJoCo steps, not from the Genesis
        mass readout. A coupled solve must verify the predicted velocity with
        a fresh complete step; constraint-set changes invalidate the response.
        """
        baseline = self.step_velocity(q, velocity, np.zeros(9))
        response = np.column_stack([self.step_velocity(q, velocity, direction) - baseline for direction in np.eye(9)])
        assert np.linalg.matrix_rank(response) == 9
        return baseline, response
