"""Keyframe editor for right-hand finger/wrist poses in the Phase 1 trajectory.

Provides a Polyscope GUI to:
  - Scrub through trajectory frames with gear + robot visualization
  - Drag gizmos to adjust right wrist (6-DOF) and fingertip (translate-only) poses via IK
  - Set keyframes at any frame
  - Cubic-spline interpolate between keyframes
  - Export modified trajectory as a new .npz

Usage:
  python edit_phase1_fingers.py [--input trajectory.npz] [--output phase1_edited.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui
from scipy.interpolate import CubicSpline

_HERE = Path(__file__).resolve().parent
_DEMO_ASSETS = _HERE.parent
_GEAR = _DEMO_ASSETS / "planetary_gear"
_GEAR_ASSETS = _GEAR / "assets"
_SHARPA_URDF = _DEMO_ASSETS / "marvin_sharpa_description" / "marvin_sharpa.urdf"
_YOYO = _DEMO_ASSETS / "yoyo" / "scripts"
if str(_YOYO) not in sys.path:
    sys.path.insert(0, str(_YOYO))

MESH_SCALE = 0.0012
ROBOT_BASE_Z = 1.08

_RIGHT_ARM_JOINTS = [f"Joint{i}_R" for i in range(1, 8)]
_IK_TARGET_LINK = "Link7_R"

_R_FINGER_CHAINS: dict[str, dict] = {
    "thumb":  {"tip": "right_thumb_DP",
               "joints": ["right_thumb_CMC_FE", "right_thumb_CMC_AA",
                          "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP"]},
    "index":  {"tip": "right_index_DP",
               "joints": ["right_index_MCP_FE",  "right_index_MCP_AA",
                          "right_index_PIP",  "right_index_DIP"]},
    "middle": {"tip": "right_middle_DP",
               "joints": ["right_middle_MCP_FE", "right_middle_MCP_AA",
                          "right_middle_PIP", "right_middle_DIP"]},
    "ring":   {"tip": "right_ring_DP",
               "joints": ["right_ring_MCP_FE",   "right_ring_MCP_AA",
                          "right_ring_PIP",   "right_ring_DIP"]},
    "pinky":  {"tip": "right_pinky_DP",
               "joints": ["right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA",
                          "right_pinky_PIP", "right_pinky_DIP"]},
}

_R_ALL_FINGER_JOINTS: list[str] = []
for _ch in _R_FINGER_CHAINS.values():
    _R_ALL_FINGER_JOINTS.extend(_ch["joints"])
_R_SETUP_JOINTS = _RIGHT_ARM_JOINTS + _R_ALL_FINGER_JOINTS

_GEAR_OBJ = {
    "rigid_sun_gear":      "sun_gear_handle_v2.obj",
    "rigid_carrier":       "carrier.obj",
    "rigid_ring_gear":     "ring_gear.obj",
    "rigid_planet_gear_0": "planet_gear_v2.obj",
    "rigid_planet_gear_1": "planet_gear_v2.obj",
    "rigid_planet_gear_2": "planet_gear_v2.obj",
    "rigid_support_pin":   "support_pin.obj",
}
_GEAR_COLOR = {
    "rigid_sun_gear":      (0.45, 0.45, 0.50),
    "rigid_carrier":       (0.55, 0.50, 0.45),
    "rigid_ring_gear":     (0.60, 0.60, 0.60),
    "rigid_planet_gear_0": (0.50, 0.55, 0.60),
    "rigid_planet_gear_1": (0.50, 0.55, 0.60),
    "rigid_planet_gear_2": (0.50, 0.55, 0.60),
    "rigid_support_pin":   (0.70, 0.65, 0.55),
}


def _load_obj(path):
    verts, faces = [], []
    for line in open(path, errors="replace"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) == 4:
                faces.append([idx[0], idx[1], idx[2]])
                faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, np.float64), np.array(faces, np.int32)


def _quat_wxyz_to_mat3(qw, qx, qy, qz):
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _apply_gear_transform(V_scaled, pose7):
    px, py, pz = pose7[:3].astype(np.float64)
    R = _quat_wxyz_to_mat3(*pose7[3:].astype(np.float64))
    return (R @ V_scaled.T).T + np.array([px, py, pz])


def _cube_mesh(half: float):
    h = float(half)
    V = np.array([[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
                  [-h,-h, h],[h,-h, h],[h,h, h],[-h,h, h]], dtype=np.float64)
    F = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],
                  [0,4,5],[0,5,1],[2,6,7],[2,7,3],
                  [1,5,6],[1,6,2],[0,3,7],[0,7,4]], dtype=np.int32)
    return V, F


def _load_joint_order():
    for c in (_GEAR / "genesis_joint_order.json",
              _DEMO_ASSETS / "marvin_sharpa_description" / "genesis_joint_order.json"):
        if c.exists():
            return json.loads(c.read_text())
    return []


def main():
    from urdf_controller import URDFController

    default_in  = str(_GEAR / "trajectory_gear_sharpa_objorigin.npz")
    default_out = str(_HERE / "phase1_edited.npz")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  default=default_in)
    parser.add_argument("--output", default=default_out)
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    data = {k: v.copy() for k, v in np.load(args.input, allow_pickle=True).items()}
    robot_qpos = data["robot_qpos"].copy()
    n_frames = len(robot_qpos)
    print(f"  {n_frames} frames, {robot_qpos.shape[1]} DOF")

    joint_order = _load_joint_order()
    n_joints = len(joint_order)

    # Map joint names to qpos column indices
    joint_to_col = {name: i for i, name in enumerate(joint_order)}
    rh_joint_names = _R_SETUP_JOINTS
    rh_cols = [joint_to_col[j] for j in rh_joint_names if j in joint_to_col]

    # URDF controller
    vis_ctrl = URDFController(str(_SHARPA_URDF), mesh_source="visual")
    root_tf = np.eye(4); root_tf[2, 3] = ROBOT_BASE_Z
    vis_ctrl.set_root_transform(root_tf)

    # Gear meshes
    gear_V_local: dict[str, np.ndarray] = {}
    gear_F: dict[str, np.ndarray] = {}
    for k, obj in _GEAR_OBJ.items():
        p = _GEAR_ASSETS / obj
        if not p.exists():
            continue
        V, F = _load_obj(str(p))
        gear_V_local[k] = (V * MESH_SCALE).astype(np.float64)
        gear_F[k] = F

    # --- Polyscope init ---
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1600, 1000)
    ps.set_automatically_compute_scene_extents(False)

    # Register gear meshes
    ps_gears: dict[str, ps.SurfaceMesh] = {}
    for k in gear_V_local:
        sm = ps.register_surface_mesh(f"gear/{k}", gear_V_local[k], gear_F[k])
        sm.set_color(_GEAR_COLOR.get(k, (0.7, 0.7, 0.7)))
        sm.set_smooth_shade(True)
        ps_gears[k] = sm

    # Register robot meshes
    ps_robot: dict[str, ps.SurfaceMesh] = {}
    tfs = vis_ctrl.get_mesh_transforms()
    for node in vis_ctrl.mesh_nodes:
        tf = tfs.get(node.node_name, np.eye(4))
        wv = node.local_vertices @ tf[:3, :3].T + tf[:3, 3]
        m = ps.register_surface_mesh(f"robot/{node.node_name}", wv, node.faces)
        m.set_transparency(0.35)
        m.set_smooth_shade(True)
        ps_robot[node.node_name] = m

    # --- Gizmo setup ---
    _cube_V, _cube_F = _cube_mesh(0.018)
    _tip_V, _tip_F = _cube_mesh(0.007)

    wrist_gizmo = ps.register_surface_mesh("edit/wrist", _cube_V, _cube_F, smooth_shade=False)
    wrist_gizmo.set_color((1.0, 0.85, 0.15))
    wrist_gizmo.set_transparency(0.55)
    wrist_gizmo.set_transform_gizmo_enabled(False)
    _wg = wrist_gizmo.get_transformation_gizmo()
    _wg.set_allow_translation(True); _wg.set_allow_rotation(True)
    _wg.set_allow_scaling(False); _wg.set_interact_in_local_space(False)

    tip_gizmos: dict[str, ps.SurfaceMesh] = {}
    for fname, chain in _R_FINGER_CHAINS.items():
        sm = ps.register_surface_mesh(f"edit/tip_{fname}", _tip_V, _tip_F, smooth_shade=False)
        sm.set_color((1.0, 0.55, 0.15))
        sm.set_transparency(0.65)
        sm.set_transform_gizmo_enabled(False)
        g = sm.get_transformation_gizmo()
        g.set_allow_translation(True); g.set_allow_rotation(False)
        g.set_allow_scaling(False); g.set_interact_in_local_space(False)
        tip_gizmos[fname] = sm

    _last_wrist_T = [np.eye(4)]
    _last_tip_T = {fn: np.eye(4) for fn in _R_FINGER_CHAINS}
    gizmos_enabled = [False]

    def _set_robot_from_qpos(frame: int):
        q = robot_qpos[frame]
        n_use = min(n_joints, len(q))
        vis_ctrl.set_joint_positions({joint_order[i]: float(q[i]) for i in range(n_use)})

    def _sync_gizmos_to_fk():
        T = vis_ctrl.get_link_transform(_IK_TARGET_LINK)
        wrist_gizmo.set_transform(T)
        _last_wrist_T[0] = T.copy()
        for fn, chain in _R_FINGER_CHAINS.items():
            pos = vis_ctrl.get_link_position(chain["tip"])
            Tt = np.eye(4); Tt[:3, 3] = pos
            tip_gizmos[fn].set_transform(Tt)
            _last_tip_T[fn] = Tt.copy()

    def _update_robot_meshes():
        tfs = vis_ctrl.get_mesh_transforms()
        for node in vis_ctrl.mesh_nodes:
            tf = tfs.get(node.node_name)
            if tf is not None and node.node_name in ps_robot:
                ps_robot[node.node_name].update_vertex_positions(
                    node.local_vertices @ tf[:3, :3].T + tf[:3, 3])

    def _update_gears(frame: int):
        for k, sm in ps_gears.items():
            poses = data.get(k)
            if poses is not None and frame < len(poses):
                sm.update_vertex_positions(
                    _apply_gear_transform(gear_V_local[k], poses[frame]))

    def _update_display(frame: int):
        _set_robot_from_qpos(frame)
        _update_robot_meshes()
        _update_gears(frame)
        _sync_gizmos_to_fk()

    def _get_rh_joints() -> dict[str, float]:
        """Get current right-hand joint values from vis_ctrl."""
        pos = vis_ctrl.get_joint_positions()
        return {j: float(pos.get(j, 0.0)) for j in rh_joint_names}

    def _write_rh_joints_to_qpos(frame: int, joints: dict[str, float]):
        """Write right-hand joint values into robot_qpos array."""
        for jname, val in joints.items():
            col = joint_to_col.get(jname)
            if col is not None and col < robot_qpos.shape[1]:
                robot_qpos[frame, col] = val

    # --- Keyframe storage ---
    keyframes: dict[int, dict[str, float]] = {}
    _KF_FILE = _HERE / "finger_keyframes.json"

    def _save_keyframes():
        out = {str(f): joints for f, joints in keyframes.items()}
        _KF_FILE.write_text(json.dumps(out, indent=2))
        print(f"[keyframes] saved {len(keyframes)} keyframes -> {_KF_FILE.name}")

    def _load_keyframes():
        if not _KF_FILE.exists():
            print(f"[keyframes] {_KF_FILE.name} not found")
            return
        raw = json.loads(_KF_FILE.read_text())
        keyframes.clear()
        for f_str, joints in raw.items():
            keyframes[int(f_str)] = joints
        print(f"[keyframes] loaded {len(keyframes)} keyframes <- {_KF_FILE.name}")

    # Auto-load if exists
    if _KF_FILE.exists():
        _load_keyframes()

    def _bake_keyframes():
        """Cubic spline interpolate keyframes into robot_qpos."""
        if len(keyframes) < 2:
            print("[bake] Need at least 2 keyframes")
            return False
        kf_frames = sorted(keyframes.keys())
        for jname in rh_joint_names:
            col = joint_to_col.get(jname)
            if col is None:
                continue
            vals = [keyframes[f].get(jname, float(robot_qpos[f, col])) for f in kf_frames]
            cs = CubicSpline(kf_frames, vals, bc_type="clamped")
            # Only interpolate within keyframe range
            f_start, f_end = kf_frames[0], kf_frames[-1]
            frames_range = np.arange(f_start, f_end + 1)
            robot_qpos[f_start:f_end + 1, col] = cs(frames_range).astype(np.float32)
        print(f"[bake] Interpolated {len(rh_joint_names)} joints across "
              f"frames {kf_frames[0]}-{kf_frames[-1]} ({len(kf_frames)} keyframes)")
        return True

    # --- State ---
    st = {"frame": 0, "play": False, "_acc": 0.0, "_last": time.perf_counter(),
          "saved": False, "baked": False, "auto_bake": True}

    # Auto-bake on startup if keyframes were loaded
    if len(keyframes) >= 2:
        _bake_keyframes()
        st["baked"] = True

    _update_display(0)
    fps = 60.0

    def gui_callback():
        now = time.perf_counter()
        dt_w = now - st["_last"]; st["_last"] = now

        imgui.Text("=== Phase 1 Finger Keyframe Editor ===")
        imgui.Text(f"{n_frames} frames, {len(rh_joint_names)} editable joints (right hand)")
        imgui.Separator()

        # --- Playback ---
        if imgui.Button("Pause" if st["play"] else "Play"):
            st["play"] = not st["play"]; st["_acc"] = 0.0
        imgui.SameLine()
        if imgui.Button("|<"):
            st["frame"] = 0; st["play"] = False
        imgui.SameLine()
        if imgui.Button(">|"):
            st["frame"] = n_frames - 1; st["play"] = False

        changed, val = imgui.SliderInt("Frame", st["frame"], 0, n_frames - 1)
        if changed:
            st["frame"] = val; st["play"] = False
            _update_display(val)

        if st["play"]:
            st["_acc"] += dt_w * fps
            adv = int(st["_acc"]); st["_acc"] -= adv
            old_frame = st["frame"]
            st["frame"] = (st["frame"] + adv) % n_frames
            if st["frame"] != old_frame:
                _update_display(st["frame"])

        imgui.Separator()

        # --- Gizmo toggle ---
        if imgui.Button("Enable Gizmos" if not gizmos_enabled[0] else "Disable Gizmos"):
            gizmos_enabled[0] = not gizmos_enabled[0]
            wrist_gizmo.set_transform_gizmo_enabled(gizmos_enabled[0])
            for sm in tip_gizmos.values():
                sm.set_transform_gizmo_enabled(gizmos_enabled[0])
            if gizmos_enabled[0]:
                _sync_gizmos_to_fk()

        # --- Poll gizmos ---
        if gizmos_enabled[0]:
            robot_dirty = False

            T_wrist_now = wrist_gizmo.get_transform()
            if not np.allclose(T_wrist_now, _last_wrist_T[0], atol=1e-6):
                _last_wrist_T[0] = T_wrist_now.copy()
                vis_ctrl.solve_ik(
                    _IK_TARGET_LINK,
                    T_wrist_now[:3, 3],
                    target_orientation=T_wrist_now[:3, :3],
                    orientation_mode="all",
                    arm_joints=_RIGHT_ARM_JOINTS,
                )
                _sync_gizmos_to_fk()
                robot_dirty = True

            for fname, chain in _R_FINGER_CHAINS.items():
                T_tip_now = tip_gizmos[fname].get_transform()
                if not np.allclose(T_tip_now, _last_tip_T[fname], atol=1e-6):
                    _last_tip_T[fname] = T_tip_now.copy()
                    vis_ctrl.solve_ik(
                        chain["tip"],
                        T_tip_now[:3, 3],
                        arm_joints=chain["joints"],
                    )
                    pos = vis_ctrl.get_link_position(chain["tip"])
                    Tt = np.eye(4); Tt[:3, 3] = pos
                    tip_gizmos[fname].set_transform(Tt)
                    _last_tip_T[fname] = Tt.copy()
                    robot_dirty = True

            if robot_dirty:
                _update_robot_meshes()

        imgui.Separator()

        # --- Keyframe controls ---
        imgui.Text("--- Keyframes ---")

        if imgui.Button("Set Keyframe"):
            f = st["frame"]
            joints = _get_rh_joints()
            keyframes[f] = joints
            _write_rh_joints_to_qpos(f, joints)
            if len(keyframes) >= 2:
                _bake_keyframes()
                st["baked"] = True
            else:
                st["baked"] = False
            print(f"[keyframe] Set at frame {f}")

        imgui.SameLine()
        if imgui.Button("Delete Keyframe"):
            f = st["frame"]
            if f in keyframes:
                del keyframes[f]
                if len(keyframes) >= 2:
                    _bake_keyframes()
                    st["baked"] = True
                else:
                    st["baked"] = False
                print(f"[keyframe] Deleted frame {f}")

        imgui.SameLine()
        if imgui.Button("Save KF"):
            _save_keyframes()
        imgui.SameLine()
        if imgui.Button("Load KF"):
            _load_keyframes()
            if len(keyframes) >= 2:
                _bake_keyframes()
                st["baked"] = True
                _update_display(st["frame"])
            else:
                st["baked"] = False

        imgui.Text(f"Keyframes: {len(keyframes)}")
        if keyframes:
            kf_list = sorted(keyframes.keys())
            imgui.Text(f"  Frames: {kf_list}")
            for kf in kf_list:
                if imgui.Button(f"Go {kf}"):
                    st["frame"] = kf; st["play"] = False
                    _update_display(kf)
                imgui.SameLine()
            imgui.NewLine()

        imgui.Separator()

        # --- Bake & Save ---
        if len(keyframes) >= 2:
            if imgui.Button("Bake (spline interpolate)"):
                if _bake_keyframes():
                    st["baked"] = True
                    _update_display(st["frame"])
            if st["baked"]:
                imgui.SameLine()
                imgui.TextColored((0.2, 1.0, 0.2, 1.0), "Baked!")
        else:
            imgui.TextColored((0.7, 0.7, 0.3, 1.0), "Need >= 2 keyframes to bake")

        if imgui.Button("Save"):
            out_data = dict(data)
            out_data["robot_qpos"] = robot_qpos.astype(np.float32)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(out_path), **out_data)
            st["saved"] = True
            print(f"[save] {n_frames} frames -> {out_path}")
        if st["saved"]:
            imgui.SameLine()
            imgui.TextColored((0.2, 1.0, 0.2, 1.0), f"Saved -> {Path(args.output).name}")

    ps.set_user_callback(gui_callback)
    _update_display(0)
    ps.show()


if __name__ == "__main__":
    main()
