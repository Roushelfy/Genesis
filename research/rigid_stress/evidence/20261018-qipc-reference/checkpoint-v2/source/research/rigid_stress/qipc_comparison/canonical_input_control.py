"""Check original versus explicit-attribute Franka input using MuJoCo's actual MJCF compiler."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from .artifacts import file_id, write_json
from .robot_inputs import prepare_mjcf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.conditions) as bank:
        qpos = bank["initial_joints"][0]
    generated = prepare_mjcf(args.mjcf, qpos, args.output)
    original = mujoco.MjModel.from_xml_path(str(args.mjcf))
    explicit = mujoco.MjModel.from_xml_path(str(generated))
    for left, right in (
        (original.geom_type, explicit.geom_type),
        (original.geom_size, explicit.geom_size),
        (original.geom_pos, explicit.geom_pos),
        (original.geom_quat, explicit.geom_quat),
        (original.mesh_vert, explicit.mesh_vert),
        (original.mesh_face, explicit.mesh_face),
        (original.body_mass, explicit.body_mass),
        (original.body_inertia, explicit.body_inertia),
        (original.body_ipos, explicit.body_ipos),
        (original.body_iquat, explicit.body_iquat),
        (original.dof_armature, explicit.dof_armature),
        (original.dof_damping, explicit.dof_damping),
        (original.eq_data, explicit.eq_data),
        (original.eq_solref, explicit.eq_solref),
        (original.eq_solimp, explicit.eq_solimp),
        (original.actuator_gainprm, explicit.actuator_gainprm),
        (original.actuator_biasprm, explicit.actuator_biasprm),
    ):
        np.testing.assert_array_equal(left, right)
    data, explicit_data = mujoco.MjData(original), mujoco.MjData(explicit)
    data.qpos[:], explicit_data.qpos[:] = qpos, qpos
    mujoco.mj_forward(original, data)
    mujoco.mj_forward(explicit, explicit_data)
    np.testing.assert_array_equal(data.geom_xpos, explicit_data.geom_xpos)
    np.testing.assert_array_equal(data.geom_xmat, explicit_data.geom_xmat)
    write_json(
        args.output / "result.json",
        {
            "kind": "original-MJCF-semantics-preserved",
            "mujoco_version": mujoco.__version__,
            "original": file_id(args.mjcf),
            "generated": file_id(generated),
            "geometries": original.ngeom,
            "all_compared_geometry_inertia_joint_equality_actuator_arrays_exact": True,
            "maximum_world_geom_position_difference_m": 0.0,
        },
    )


if __name__ == "__main__":
    main()
