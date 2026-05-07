"""Convert Robowits MCAP recordings to NPZ trajectories for Genesis replay.

Extracts robot joint angles (18 DOF bimanual MARVIN_PIKA) and rigid object
poses from MCAP files recorded via the gs-core Robowits toolbench.

Usage
-----
    # Convert a single file
    python convert_mcap_to_npz.py /path/to/02.mcap -o trajectories/

    # Convert all MCAPs in a directory
    python convert_mcap_to_npz.py /path/to/mcap_render/ -o trajectories/

Output
------
For each MCAP, produces ``{task_id}.npz`` containing:
    - ``sim_time``      : (N,) float64 — seconds from start
    - ``robot_qpos``    : (N, 18) float32 — bimanual interleaved qpos
    - ``rigid_{name}``  : (N, 6) float32 — pos(3) + euler(3, RPY radians) per entity

When ``rt/arm_state_{left,right}`` has no messages (e.g. task 14), the file
still contains ``sim_time`` and rigid entity arrays but no ``robot_qpos``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool
from google.protobuf import reflection as pb_reflection
from mcap.reader import make_reader


# ---------------------------------------------------------------------------
# Protobuf helpers
# ---------------------------------------------------------------------------


def _build_msg_classes(summary):
    """Build protobuf message classes from MCAP-embedded schemas.

    Supports two MCAP formats:
      - lattice.v1: ArmStateMessage (combined joints+gripper), ObjsInfoMessage
      - we.v2: JointAngles7DMessage + GripperWidthMessage (separate),
               ObjsInfoMessage

    Returned dict keys are stable: callers see "ArmState"/"ObjsInfo" for
    lattice.v1, or "JointAngles"/"GripperWidth"/"ObjsInfo" for we.v2.
    """
    pool = descriptor_pool.DescriptorPool()
    added = set()
    schema_names: set[str] = set()
    for _sid, schema in summary.schemas.items():
        if schema.encoding != "protobuf":
            continue
        schema_names.add(schema.name)
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(schema.data)
        for fd in fds.file:
            if fd.name not in added:
                try:
                    pool.Add(fd)
                    added.add(fd.name)
                except Exception:
                    pass

    def _make(full_name):
        desc = pool.FindMessageTypeByName(full_name)
        return pb_reflection.GeneratedProtocolMessageType(desc.name, (), {"DESCRIPTOR": desc, "__module__": None})

    if "lattice.v1.ArmStateMessage" in schema_names:
        return {
            "_format": "lattice.v1",
            "ArmState": _make("lattice.v1.ArmStateMessage"),
            "ObjsInfo": _make("lattice.v1.ObjsInfoMessage"),
        }
    if "we.v2.JointAngles7DMessage" in schema_names:
        return {
            "_format": "we.v2",
            "JointAngles": _make("we.v2.JointAngles7DMessage"),
            "GripperWidth": _make("we.v2.GripperWidthMessage"),
            "ObjsInfo": _make("we.v2.ObjsInfoMessage"),
        }
    raise ValueError(f"Unrecognized MCAP schemas: {sorted(schema_names)}")


# ---------------------------------------------------------------------------
# Bimanual qpos construction
# ---------------------------------------------------------------------------


def _build_bimanual_qpos(
    left_angles7: list[float],
    right_angles7: list[float],
    left_gripper: float,
    right_gripper: float,
) -> np.ndarray:
    """Build 18-DOF bimanual qpos from per-arm joint angles + gripper widths.

    Interleaved arm joints: [R0, L0, R1, L1, ..., R6, L6] (14)
    Finger joints: [Joint8_R, Joint9_R, Joint8_L, Joint9_L] = [R_f0, R_f1, L_f0, L_f1] (4)

    Finger order matches gs-core build_bimanual_qpos() and Genesis URDF parse order
    for marvin_pika.urdf (R fingers come before L because URDF declares them first).
    """
    qpos = np.zeros(18, dtype=np.float32)
    for i in range(7):
        qpos[2 * i] = right_angles7[i]
        qpos[2 * i + 1] = left_angles7[i]
    qpos[14] = right_gripper / 2.0
    qpos[15] = right_gripper / 2.0
    qpos[16] = left_gripper / 2.0
    qpos[17] = left_gripper / 2.0
    return qpos


# ---------------------------------------------------------------------------
# MCAP extraction
# ---------------------------------------------------------------------------


def _extract_arm_state(msg) -> tuple[list[float], float]:
    """Extract (joint_angles_7d, gripper_width) from an ArmStateMessage."""
    d = msg.data
    angles = [d.angles7d.j0, d.angles7d.j1, d.angles7d.j2, d.angles7d.j3, d.angles7d.j4, d.angles7d.j5, d.angles7d.j6]
    return angles, float(d.gripper_width)


def _extract_objs_info(msg) -> dict[str, np.ndarray]:
    """Extract per-entity pos(3) + euler(3, RPY radians) from ObjsInfoMessage."""
    result = {}
    for key in msg.data.entities:
        ent = msg.data.entities[key]
        pos = list(ent.pos)  # 3 floats
        euler = list(ent.euler)  # 3 floats (RPY radians)
        result[key] = np.array(pos + euler, dtype=np.float32)
    return result


_TASK_DIR_RE = re.compile(r"^\d{2}(_(v\d+|add\d+|trap\d+|pivot\d+))?$")


def _resolve_task_id(mcap_path: Path) -> str:
    """Derive task id from MCAP path.

    Layouts supported:
      - .../{NN}_SUC_raw/<file>.mcap        -> "NN"
      - .../suc_teleop/{NN[_variant]}/<file>.mcap -> "NN" or "NN_variant"
        where variant ∈ {v\\d+, add\\d+, trap\\d+, pivot\\d+}
      - otherwise: file stem
    """
    parent = mcap_path.parent.name
    if "_SUC_raw" in parent:
        return parent.split("_", 1)[0]
    if _TASK_DIR_RE.match(parent):
        return parent
    return mcap_path.stem


def _nearest_idx(msgs, target_ts, start):
    """Advance `start` to the largest index whose log_time <= target_ts."""
    while start < len(msgs) - 1 and msgs[start + 1][0] <= target_ts:
        start += 1
    return start


def _resample(arrays: dict[str, np.ndarray], rate_hz: float) -> dict[str, np.ndarray]:
    """Linearly interpolate to a fixed rate; SLERP via euler for `rigid_*` entries.

    `rigid_*` arrays are (N, 6) with pos(3) + euler XYZ radians (3); positions
    and everything else use `np.interp`, rotations go euler→Rotation→Slerp→euler.
    """
    from scipy.spatial.transform import Rotation, Slerp

    src_times = arrays["sim_time"].astype(np.float64)
    target_times = np.arange(src_times[0], src_times[-1], 1.0 / rate_hz, dtype=np.float64)
    out: dict[str, np.ndarray] = {"sim_time": target_times.astype(arrays["sim_time"].dtype)}
    for key, val in arrays.items():
        if key == "sim_time":
            continue
        if key.startswith("rigid_") and val.ndim == 2 and val.shape[1] == 6:
            pos_out = np.stack([np.interp(target_times, src_times, val[:, k]) for k in range(3)], axis=1)
            rot = Rotation.from_euler("xyz", val[:, 3:6])
            euler_out = Slerp(src_times, rot)(target_times).as_euler("xyz")
            out[key] = np.concatenate([pos_out, euler_out], axis=1).astype(val.dtype)
        else:
            flat = val.reshape(val.shape[0], -1)
            interp = np.stack(
                [np.interp(target_times, src_times, flat[:, k]) for k in range(flat.shape[1])],
                axis=1,
            )
            out[key] = interp.reshape((len(target_times),) + val.shape[1:]).astype(val.dtype)
    return out


def convert_mcap(mcap_path: Path, output_dir: Path, rate_hz: float) -> Path:
    """Convert one MCAP to NPZ."""
    task_id = _resolve_task_id(mcap_path)
    out_path = output_dir / f"{task_id}.npz"

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        msg_classes = _build_msg_classes(summary)
        fmt = msg_classes["_format"]

        # Collect all messages by topic, sorted by log_time
        arm_left_msgs = []  # lattice.v1 only: combined ArmStateMessage
        arm_right_msgs = []
        joints_left_msgs = []  # we.v2: separate JointAngles7DMessage
        joints_right_msgs = []
        grip_left_msgs = []  # we.v2: separate GripperWidthMessage
        grip_right_msgs = []
        objs_msgs = []

        f.seek(0)
        reader2 = make_reader(f)
        for _schema, channel, message in reader2.iter_messages():
            topic = channel.topic
            if fmt == "lattice.v1":
                if topic == "rt/arm_state_left":
                    m = msg_classes["ArmState"]()
                    m.ParseFromString(message.data)
                    arm_left_msgs.append((message.log_time, m))
                elif topic == "rt/arm_state_right":
                    m = msg_classes["ArmState"]()
                    m.ParseFromString(message.data)
                    arm_right_msgs.append((message.log_time, m))
                elif topic == "rt/objs_info":
                    m = msg_classes["ObjsInfo"]()
                    m.ParseFromString(message.data)
                    objs_msgs.append((message.log_time, m))
            else:  # we.v2
                if topic == "MARVIN_PIKA.state.left_arm.joint_angles":
                    m = msg_classes["JointAngles"]()
                    m.ParseFromString(message.data)
                    joints_left_msgs.append((message.log_time, m))
                elif topic == "MARVIN_PIKA.state.right_arm.joint_angles":
                    m = msg_classes["JointAngles"]()
                    m.ParseFromString(message.data)
                    joints_right_msgs.append((message.log_time, m))
                elif topic == "MARVIN_PIKA.state.left_eef.gripper_width":
                    m = msg_classes["GripperWidth"]()
                    m.ParseFromString(message.data)
                    grip_left_msgs.append((message.log_time, m))
                elif topic == "MARVIN_PIKA.state.right_eef.gripper_width":
                    m = msg_classes["GripperWidth"]()
                    m.ParseFromString(message.data)
                    grip_right_msgs.append((message.log_time, m))
                elif topic == "MARVIN_PIKA.objs_info":
                    m = msg_classes["ObjsInfo"]()
                    m.ParseFromString(message.data)
                    objs_msgs.append((message.log_time, m))

    if not objs_msgs:
        raise ValueError(f"No objs_info messages in {mcap_path}")

    n_frames = len(objs_msgs)
    t0 = objs_msgs[0][0]
    sim_time = np.array([(ts - t0) / 1e9 for ts, _ in objs_msgs], dtype=np.float64)

    # Extract rigid entity data
    entity_names = list(_extract_objs_info(objs_msgs[0][1]).keys())
    rigid_data = {name: np.zeros((n_frames, 6), dtype=np.float32) for name in entity_names}

    for i, (ts, msg) in enumerate(objs_msgs):
        info = _extract_objs_info(msg)
        for name in entity_names:
            if name in info:
                rigid_data[name][i] = info[name]

    # Extract robot qpos
    if fmt == "lattice.v1":
        has_robot = len(arm_left_msgs) > 0 and len(arm_right_msgs) > 0
    else:
        has_robot = (
            len(joints_left_msgs) > 0
            and len(joints_right_msgs) > 0
            and len(grip_left_msgs) > 0
            and len(grip_right_msgs) > 0
        )
    robot_qpos = None
    if has_robot:
        robot_qpos = np.zeros((n_frames, 18), dtype=np.float32)
        if fmt == "lattice.v1":
            left_idx = right_idx = 0
            for i, (objs_ts, _) in enumerate(objs_msgs):
                left_idx = _nearest_idx(arm_left_msgs, objs_ts, left_idx)
                right_idx = _nearest_idx(arm_right_msgs, objs_ts, right_idx)
                la, lg = _extract_arm_state(arm_left_msgs[left_idx][1])
                ra, rg = _extract_arm_state(arm_right_msgs[right_idx][1])
                robot_qpos[i] = _build_bimanual_qpos(la, ra, lg, rg)
        else:
            jl = jr = gl = gr = 0
            for i, (objs_ts, _) in enumerate(objs_msgs):
                jl = _nearest_idx(joints_left_msgs, objs_ts, jl)
                jr = _nearest_idx(joints_right_msgs, objs_ts, jr)
                gl = _nearest_idx(grip_left_msgs, objs_ts, gl)
                gr = _nearest_idx(grip_right_msgs, objs_ts, gr)
                ja_l = joints_left_msgs[jl][1].data
                ja_r = joints_right_msgs[jr][1].data
                la = [ja_l.j0, ja_l.j1, ja_l.j2, ja_l.j3, ja_l.j4, ja_l.j5, ja_l.j6]
                ra = [ja_r.j0, ja_r.j1, ja_r.j2, ja_r.j3, ja_r.j4, ja_r.j5, ja_r.j6]
                lg = float(grip_left_msgs[gl][1].data.value)
                rg = float(grip_right_msgs[gr][1].data.value)
                robot_qpos[i] = _build_bimanual_qpos(la, ra, lg, rg)

    # Build NPZ arrays
    arrays: dict[str, np.ndarray] = {"sim_time": sim_time}
    if robot_qpos is not None:
        arrays["robot_qpos"] = robot_qpos
    for name in entity_names:
        arrays[f"rigid_{name}"] = rigid_data[name]

    n_in = n_frames
    arrays = _resample(arrays, rate_hz)
    print(f"[{task_id}] resampled {n_in} → {len(arrays['sim_time'])} frames at {rate_hz} Hz")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    out_n = len(arrays["sim_time"])
    print(
        f"[{task_id}] {out_n} frames, {len(entity_names)} entities"
        f"{', 18-DOF robot' if has_robot else ', NO robot data'}"
        f" → {out_path}"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert Robowits MCAP to NPZ")
    parser.add_argument("input", type=Path, help="MCAP file or directory of .mcap files")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("trajectories"), help="Output directory (default: trajectories/)"
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=60.0,
        help="Resample to this fixed rate (Hz). Linear interp for pos/qpos, SLERP for rotations.",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        mcaps = sorted(args.input.glob("*.mcap"))
        if not mcaps:
            mcaps = sorted(args.input.rglob("*.mcap"))
        if not mcaps:
            print(f"No .mcap files found in {args.input}")
            return
        for mcap_path in mcaps:
            try:
                convert_mcap(mcap_path, args.output, rate_hz=args.rate_hz)
            except Exception as e:
                print(f"[{mcap_path.stem}] ERROR: {e}")
    else:
        convert_mcap(args.input, args.output, rate_hz=args.rate_hz)


if __name__ == "__main__":
    main()
