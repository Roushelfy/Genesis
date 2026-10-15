"""Evaluate the same compiled joint tree in centered extended precision."""

import mujoco
import numpy as np


def quaternion_rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.longdouble)
    w, x, y, z = q / np.sqrt(q @ q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.longdouble,
    )


class PrecisePandaKinematics:
    def __init__(self, model, origin):
        self.model = model
        self.origin = np.asarray(origin, dtype=np.longdouble)
        self.rotations = [quaternion_rotation(q) for q in model.body_quat]

    def poses(self, q):
        model = self.model
        q = np.asarray(q, dtype=np.longdouble)
        rotations = np.zeros((model.nbody, 3, 3), dtype=np.longdouble)
        positions = np.zeros((model.nbody, 3), dtype=np.longdouble)
        rotations[0] = np.eye(3, dtype=np.longdouble)
        positions[0] = -self.origin
        for body in range(1, model.nbody):
            local_rotation = self.rotations[body]
            local_position = np.asarray(model.body_pos[body], dtype=np.longdouble).copy()
            assert model.body_jntnum[body] in (0, 1)
            if model.body_jntnum[body]:
                joint = model.body_jntadr[body]
                address = model.jnt_qposadr[joint]
                displacement = q[address] - np.longdouble(model.qpos0[address])
                axis = np.asarray(model.jnt_axis[joint], dtype=np.longdouble)
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE:
                    local_position += local_rotation @ axis * displacement
                else:
                    assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
                    x, y, z = axis
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.longdouble)
                    rotation = (
                        np.eye(3, dtype=np.longdouble)
                        + np.sin(displacement) * skew
                        + (1 - np.cos(displacement)) * (skew @ skew)
                    )
                    anchor = np.asarray(model.jnt_pos[joint], dtype=np.longdouble)
                    local_position += local_rotation @ (anchor - rotation @ anchor)
                    local_rotation = local_rotation @ rotation
            parent = model.body_parentid[body]
            positions[body] = positions[parent] + rotations[parent] @ local_position
            rotations[body] = rotations[parent] @ local_rotation
        return positions, rotations
