"""Standalone Polyscope GUI tool for blending teleop trajectories with manual IK corrections.

Workflow:
  1. Robot loads automatically on startup (marvin_sharpa, ikpy ready).
  2. Load a teleop trajectory via the Setup panel (default: ../v5_init).
  3. Scrub to an imperfect frame, adjust the pose via IK keyboard controls,
     then record a blend keyframe. Only the joints that differ from the
     original teleop are stored, together with a per-keyframe blend weight.
  4. Preview the blended result by playing the timeline with blend preview ON.
  5. Bake to a new trajectory.npz for use with robot_replay_controller_main.py.

Usage:
    python tool_create_blending.py
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import polyscope as ps
from polyscope import imgui

from blend_utils import (
    BlendKeyframe,
    BlendSchedule,
    bake_trajectory,
    load_blend_keyframes,
    save_blend_keyframes,
)
from replay_utils import (
    TrajectoryData,
    load_genesis_joint_order,
    load_trajectory_npz,
    qpos_to_joint_dict,
)
from urdf_controller import URDFController
from urdf_gui import (
    ORIENTATION_PRESETS,
    PRESET_NAMES,
    make_rotation,
    parse_obj,
    try_key_pressed,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_URDF_PATH = str(_REPO_ROOT / "DemoAssets" / "marvin_sharpa_description" / "marvin_sharpa.urdf")
_ROBOT_BASE_POS = np.array([0.0, 0.0, 1.08], dtype=np.float64)
_DEFAULT_TRAJ_DIR = str(_SCRIPT_DIR.parent / "v5_init")
_DEFAULT_BLEND_FILE = str(_SCRIPT_DIR / "blend_keyframes_v1.json")
_DEFAULT_OUTPUT_DIR = str(_SCRIPT_DIR.parent / "v5_init_blended_v1")


def _get_orientation(idx: int):
    R = ORIENTATION_PRESETS[PRESET_NAMES[idx]]
    return None if np.allclose(R, 0) else R


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load robot
    # ------------------------------------------------------------------
    controller = URDFController(_URDF_PATH, mesh_source="visual")
    root_tf = np.eye(4, dtype=np.float64)
    root_tf[:3, 3] = _ROBOT_BASE_POS
    controller.set_root_transform(root_tf)

    joint_names = controller.joint_names
    joint_limits = controller.joint_limits
    end_effectors = controller.find_end_effectors()
    left_ee = next((e for e in end_effectors if "_L" in e or "left" in e.lower()), end_effectors[0])
    right_ee = next((e for e in end_effectors if "_R" in e or "right" in e.lower()), end_effectors[-1])
    print(f"[blend-tool] {len(joint_names)} joints, EE: L={left_ee}  R={right_ee}")

    # Ghost controller for original teleop comparison
    ghost_controller = URDFController(_URDF_PATH, mesh_source="visual")
    ghost_controller.set_root_transform(root_tf)

    # ------------------------------------------------------------------
    # 2. Polyscope init + register visual meshes
    # ------------------------------------------------------------------
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    # Blended robot: solid
    vis_meshes: dict[str, tuple] = {}
    transforms = controller.get_mesh_transforms()
    for node in controller.mesh_nodes:
        tf = transforms.get(node.node_name)
        if tf is None:
            continue
        world_v = node.local_vertices @ tf[:3, :3].T + tf[:3, 3]
        name = f"vis_{node.node_name}"
        m = ps.register_surface_mesh(name, world_v, node.faces)
        m.set_transparency(1.0)
        m.set_color((1.0, 1.0, 1.0))
        vis_meshes[node.node_name] = (m, node)

    # Original teleop ghost: solid red
    ghost_meshes: dict[str, tuple] = {}
    ghost_transforms = ghost_controller.get_mesh_transforms()
    for node in ghost_controller.mesh_nodes:
        tf = ghost_transforms.get(node.node_name)
        if tf is None:
            continue
        world_v = node.local_vertices @ tf[:3, :3].T + tf[:3, 3]
        name = f"ghost_{node.node_name}"
        m = ps.register_surface_mesh(name, world_v, node.faces)
        m.set_transparency(1.0)
        m.set_color((1.0, 0.3, 0.3))
        ghost_meshes[node.node_name] = (m, node)

    ghost_visible = {"enabled": True}

    def _update_vis():
        tfs = controller.get_mesh_transforms()
        for nname, (mesh, node) in vis_meshes.items():
            tf = tfs.get(nname)
            if tf is None:
                continue
            mesh.update_vertex_positions(node.local_vertices @ tf[:3, :3].T + tf[:3, 3])

    def _update_ghost():
        if not ghost_visible["enabled"]:
            return
        tfs = ghost_controller.get_mesh_transforms()
        for nname, (mesh, node) in ghost_meshes.items():
            tf = tfs.get(nname)
            if tf is None:
                continue
            mesh.update_vertex_positions(node.local_vertices @ tf[:3, :3].T + tf[:3, 3])

    # ------------------------------------------------------------------
    # 3. Mutable state
    # ------------------------------------------------------------------

    # Setup paths
    setup = {
        "traj_dir": _DEFAULT_TRAJ_DIR,
        "blend_file": _DEFAULT_BLEND_FILE,
        "output_dir": _DEFAULT_OUTPUT_DIR,
        "status": "",
    }

    # Trajectory (loaded on demand)
    traj_holder: dict[str, TrajectoryData | None] = {"traj": None}
    genesis_names_holder: dict[str, list[str]] = {"names": []}

    # Original teleop trajectory (kept for ghost comparison when playing baked)
    original_traj_holder: dict[str, TrajectoryData | None] = {"traj": None}
    original_gnames_holder: dict[str, list[str]] = {"names": []}

    # Playback
    play = {
        "frame": 0,
        "playing": False,
        "speed": 1.0,
        "loop": True,
        "last_time": time.monotonic(),
        "accum": 0.0,
    }

    # Blend
    blend_schedule = BlendSchedule()
    blend_state = {
        "weight": 1.0,
        "sel_kf": 0,
        "preview": True,
        "bake_status": "",
    }

    # IK
    ik_state = {
        "step": 0.01,
        "left_orient_idx": 0,
        "right_orient_idx": 0,
    }

    # Reference OBJ overlay
    ref_state: dict[str, Any] = {"entries": [], "input_buf": "", "sel_ref": 0}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_trajectory() -> bool:
        traj_dir = setup["traj_dir"].strip()
        if not traj_dir:
            setup["status"] = "No traj dir specified"
            return False
        try:
            traj = load_trajectory_npz(traj_dir)
        except FileNotFoundError as e:
            setup["status"] = str(e)
            print(f"[blend-tool] {e}")
            return False

        search_dirs = [traj.traj_dir, Path(_URDF_PATH).parent, _SCRIPT_DIR]
        try:
            gnames = load_genesis_joint_order(search_dirs)
        except FileNotFoundError as e:
            setup["status"] = str(e)
            print(f"[blend-tool] {e}")
            return False

        traj_holder["traj"] = traj
        genesis_names_holder["names"] = gnames
        play["frame"] = 0
        play["playing"] = False
        setup["status"] = f"Loaded: {traj.n_frames} frames, {traj.robot_qpos.shape[1]} DOFs, dt={traj.dt:.6f}s"
        print(f"[blend-tool] {setup['status']}")

        # Remember the first loaded trajectory as the original for ghost comparison
        if original_traj_holder["traj"] is None:
            original_traj_holder["traj"] = traj
            original_gnames_holder["names"] = list(gnames)

        init_joints = qpos_to_joint_dict(traj.robot_qpos[0], gnames)
        controller.set_joint_positions(init_joints)
        _update_vis()
        ghost_controller.set_joint_positions(init_joints)
        _update_ghost()
        return True

    def _load_blend_file() -> None:
        bf = setup["blend_file"].strip()
        if not bf:
            return
        kfs = load_blend_keyframes(bf)
        blend_schedule.clear()
        for kf in kfs:
            blend_schedule.add_keyframe(kf)
        blend_state["sel_kf"] = 0
        print(f"[blend-tool] Loaded {len(kfs)} blend keyframes from {bf}")

    def _save_blend_file() -> None:
        bf = setup["blend_file"].strip()
        if not bf:
            return
        save_blend_keyframes(bf, blend_schedule.keyframes)
        print(f"[blend-tool] Saved {len(blend_schedule)} blend keyframes to {bf}")

    def _teleop_joints_at(frame: int) -> dict[str, float]:
        traj = traj_holder["traj"]
        gnames = genesis_names_holder["names"]
        if traj is None:
            return {}
        frame = max(0, min(frame, traj.n_frames - 1))
        return qpos_to_joint_dict(traj.robot_qpos[frame], gnames)

    def _original_joints_at(frame: int) -> dict[str, float]:
        """Get joint dict from the *original* teleop trajectory (for ghost)."""
        otraj = original_traj_holder["traj"]
        ognames = original_gnames_holder["names"]
        if otraj is None:
            return {}
        frame = max(0, min(frame, otraj.n_frames - 1))
        return qpos_to_joint_dict(otraj.robot_qpos[frame], ognames)

    def _apply_frame(idx: int) -> None:
        traj = traj_holder["traj"]
        if traj is None:
            return
        idx = max(0, min(idx, traj.n_frames - 1))
        play["frame"] = idx
        teleop_joints = _teleop_joints_at(idx)
        if blend_state["preview"]:
            final = blend_schedule.blend_joints(teleop_joints, float(idx))
        else:
            final = teleop_joints
        controller.set_joint_positions(final)
        _update_vis()
        # Ghost always shows the original teleop for comparison
        orig_joints = _original_joints_at(idx)
        if orig_joints:
            ghost_controller.set_joint_positions(orig_joints)
        else:
            ghost_controller.set_joint_positions(teleop_joints)
        _update_ghost()

    def _do_ik(ee_name: str, delta: np.ndarray, orient_idx: int) -> None:
        cur_tf = controller.get_link_transform(ee_name)
        cur_pos = cur_tf[:3, 3].copy()
        arm_joints = controller.find_arm_joints(ee_name)
        if not arm_joints:
            return
        target_orient = _get_orientation(orient_idx)
        orient_mode = "all" if target_orient is not None else None
        controller.solve_ik(
            ee_name,
            cur_pos + delta,
            target_orientation=target_orient,
            orientation_mode=orient_mode,
            arm_joints=arm_joints,
        )
        _update_vis()

    # -- Reference OBJ helpers --

    def _load_ref_obj(path_str: str, transform: list[float] | None = None) -> bool:
        p = Path(path_str)
        if not p.exists():
            print(f"[ref] File not found: {p}")
            return False
        try:
            verts, faces, edges = parse_obj(p)
        except Exception as e:
            print(f"[ref] Failed to parse {p}: {e}")
            return False
        if len(verts) == 0:
            return False
        idx = len(ref_state["entries"])
        tf = list(transform) if transform else [0.0] * 6
        entry: dict[str, Any] = {
            "path": path_str, "transform": tf, "verts": verts,
            "surf": None, "curve": None,
        }
        R = make_rotation(tf[3], tf[4], tf[5])
        offset = np.array(tf[:3])
        transformed = verts @ R.T + offset
        if faces is not None:
            sname = f"ref_surf_{p.stem}_{idx}"
            m = ps.register_surface_mesh(sname, transformed, faces)
            m.set_transparency(0.4)
            m.set_color((0.6, 0.8, 1.0))
            entry["surf"] = sname
        if edges is not None:
            cname = f"ref_line_{p.stem}_{idx}"
            ps.register_curve_network(cname, transformed, edges,
                                      color=(1.0, 0.4, 0.2), radius=0.001)
            entry["curve"] = cname
        ref_state["entries"].append(entry)
        print(f"[ref] Loaded {p.name} ({len(verts)} verts)")
        return True

    def _apply_ref_transform(entry: dict) -> None:
        t = entry["transform"]
        R = make_rotation(t[3], t[4], t[5])
        offset = np.array(t[:3])
        transformed = entry["verts"] @ R.T + offset
        if entry["surf"]:
            ps.get_surface_mesh(entry["surf"]).update_vertex_positions(transformed)
        if entry["curve"]:
            ps.get_curve_network(entry["curve"]).update_node_positions(transformed)

    def _unload_ref(idx: int) -> None:
        e = ref_state["entries"][idx]
        if e["surf"]:
            ps.remove_surface_mesh(e["surf"], error_if_absent=False)
        if e["curve"]:
            ps.remove_curve_network(e["curve"])
        ref_state["entries"].pop(idx)

    def _unload_all_refs() -> None:
        while ref_state["entries"]:
            _unload_ref(0)

    def _detect_modified() -> dict[str, float]:
        teleop = _teleop_joints_at(play["frame"])
        current = controller.get_joint_positions()
        return {
            name: current[name]
            for name in current
            if abs(current[name] - teleop.get(name, 0.0)) > 1e-5
        }

    # ------------------------------------------------------------------
    # Try loading defaults on startup
    # ------------------------------------------------------------------
    _load_trajectory()
    _load_blend_file()

    # ------------------------------------------------------------------
    # GUI panels
    # ------------------------------------------------------------------

    def _draw_setup_panel() -> None:
        if not imgui.TreeNodeEx("Setup", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return

        c, v = imgui.InputText("Traj Dir##setup", setup["traj_dir"])
        if c:
            setup["traj_dir"] = v
        if imgui.Button("Load Trajectory"):
            _load_trajectory()

        imgui.Separator()
        c, v = imgui.InputText("Blend File##setup", setup["blend_file"])
        if c:
            setup["blend_file"] = v
        if imgui.Button("Load Blend##setup_bl"):
            _load_blend_file()
        imgui.SameLine()
        if imgui.Button("Save Blend##setup_bs"):
            _save_blend_file()

        imgui.Separator()
        c, v = imgui.InputText("Output Dir##setup", setup["output_dir"])
        if c:
            setup["output_dir"] = v

        if setup["status"]:
            imgui.TextWrapped(setup["status"])
        imgui.TreePop()

    def _draw_playback_panel() -> None:
        traj = traj_holder["traj"]
        if traj is None:
            imgui.Text("No trajectory loaded.")
            return
        if not imgui.TreeNodeEx("Trajectory Playback", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return

        n = traj.n_frames
        t_val = traj.sim_times[play["frame"]] if play["frame"] < n else 0.0
        total_t = float(traj.sim_times[-1]) if n > 0 else 0.0
        imgui.Text(f"Time: {t_val:.3f}s / {total_t:.3f}s   Frame: {play['frame']} / {n - 1}")

        mode_label = "BLENDED" if blend_state["preview"] else "ORIGINAL"
        imgui.TextColored(
            (0.2, 1.0, 0.4, 1.0) if blend_state["preview"] else (1.0, 0.6, 0.2, 1.0),
            f"Playing: {mode_label}"
        )

        if imgui.Button("Play Blended"):
            blend_state["preview"] = True
            play["playing"] = True
            play["last_time"] = time.monotonic()
            play["accum"] = 0.0
        imgui.SameLine()
        if imgui.Button("Play Original"):
            blend_state["preview"] = False
            play["playing"] = True
            play["last_time"] = time.monotonic()
            play["accum"] = 0.0
        imgui.SameLine()
        if imgui.Button("Pause" if play["playing"] else "Resume"):
            if play["playing"]:
                play["playing"] = False
            else:
                play["playing"] = True
                play["last_time"] = time.monotonic()
                play["accum"] = 0.0

        if imgui.Button("Reset##traj"):
            play["playing"] = False
            _apply_frame(0)

        changed, val = imgui.SliderInt("Frame##scrub", play["frame"], 0, n - 1)
        if changed:
            play["playing"] = False
            _apply_frame(val)

        changed, spd = imgui.SliderFloat("Speed##traj", play["speed"], 0.1, 5.0)
        if changed:
            play["speed"] = spd

        changed, lp = imgui.Checkbox("Loop##traj", play["loop"])
        if changed:
            play["loop"] = lp

        imgui.TreePop()

    def _draw_ik_panel() -> None:
        if not imgui.TreeNode("IK Control"):
            return
        c, step = imgui.SliderFloat("IK Step (m)", ik_state["step"], 0.001, 0.05)
        if c:
            ik_state["step"] = step
        s = ik_state["step"]
        imgui.Separator()

        # Left gripper
        if imgui.TreeNode(f"Left: {left_ee}  [WASDQE]"):
            ltf = controller.get_link_transform(left_ee)
            lp = ltf[:3, 3]
            imgui.Text(f"Pos: ({lp[0]:.4f}, {lp[1]:.4f}, {lp[2]:.4f})")
            c, oidx = imgui.Combo("##orient_L", ik_state["left_orient_idx"], PRESET_NAMES)
            if c:
                ik_state["left_orient_idx"] = oidx
            for label, delta in [
                ("+X##Lik", [s, 0, 0]), ("-X##Lik", [-s, 0, 0]),
                ("+Y##Lik", [0, s, 0]), ("-Y##Lik", [0, -s, 0]),
                ("+Z##Lik", [0, 0, s]), ("-Z##Lik", [0, 0, -s]),
            ]:
                if label != "+X##Lik":
                    imgui.SameLine()
                if imgui.Button(label):
                    _do_ik(left_ee, np.array(delta), ik_state["left_orient_idx"])
            imgui.TreePop()
        imgui.Separator()

        # Right gripper
        if imgui.TreeNode(f"Right: {right_ee}  [IJKLUO]"):
            rtf = controller.get_link_transform(right_ee)
            rp = rtf[:3, 3]
            imgui.Text(f"Pos: ({rp[0]:.4f}, {rp[1]:.4f}, {rp[2]:.4f})")
            c, oidx = imgui.Combo("##orient_R", ik_state["right_orient_idx"], PRESET_NAMES)
            if c:
                ik_state["right_orient_idx"] = oidx
            for label, delta in [
                ("+X##Rik", [s, 0, 0]), ("-X##Rik", [-s, 0, 0]),
                ("+Y##Rik", [0, s, 0]), ("-Y##Rik", [0, -s, 0]),
                ("+Z##Rik", [0, 0, s]), ("-Z##Rik", [0, 0, -s]),
            ]:
                if label != "+X##Rik":
                    imgui.SameLine()
                if imgui.Button(label):
                    _do_ik(right_ee, np.array(delta), ik_state["right_orient_idx"])
            imgui.TreePop()
        imgui.Separator()

        # Keyboard polling
        left_delta = np.zeros(3, dtype=np.float64)
        if try_key_pressed("W"):
            left_delta[0] += s
        if try_key_pressed("S"):
            left_delta[0] -= s
        if try_key_pressed("A"):
            left_delta[1] += s
        if try_key_pressed("D"):
            left_delta[1] -= s
        if try_key_pressed("E"):
            left_delta[2] += s
        if try_key_pressed("Q"):
            left_delta[2] -= s
        if np.any(left_delta != 0):
            _do_ik(left_ee, left_delta, ik_state["left_orient_idx"])

        right_delta = np.zeros(3, dtype=np.float64)
        if try_key_pressed("I"):
            right_delta[0] += s
        if try_key_pressed("K"):
            right_delta[0] -= s
        if try_key_pressed("J"):
            right_delta[1] += s
        if try_key_pressed("L"):
            right_delta[1] -= s
        if try_key_pressed("O"):
            right_delta[2] += s
        if try_key_pressed("U"):
            right_delta[2] -= s
        if np.any(right_delta != 0):
            _do_ik(right_ee, right_delta, ik_state["right_orient_idx"])

        imgui.Text("Left:  W/S=X  A/D=Y  Q/E=Z")
        imgui.Text("Right: I/K=X  J/L=Y  U/O=Z")
        imgui.TreePop()

    joint_panel_state = {
        "filter": "",
        "show_modified_only": False,
    }

    def _draw_joint_values_panel() -> None:
        if not imgui.TreeNode("Joint Values"):
            return
        traj = traj_holder["traj"]
        current = controller.get_joint_positions()

        c, filt = imgui.InputText("Filter##jfilt", joint_panel_state["filter"])
        if c:
            joint_panel_state["filter"] = filt
        filt_lower = joint_panel_state["filter"].strip().lower()

        teleop: dict[str, float] = {}
        modified: dict[str, float] = {}
        if traj is not None:
            teleop = _teleop_joints_at(play["frame"])
            modified = _detect_modified()
            c, sm = imgui.Checkbox("Show modified only", joint_panel_state["show_modified_only"])
            if c:
                joint_panel_state["show_modified_only"] = sm
            n_mod = len(modified)
            imgui.Text(f"Modified: {n_mod} / {len(joint_names)}")

        if imgui.Button("Reset to Teleop##jtReset") and traj is not None:
            controller.set_joint_positions(teleop)
            _update_vis()

        imgui.Separator()

        names_to_show = []
        for name in joint_names:
            if filt_lower and filt_lower not in name.lower():
                continue
            if joint_panel_state["show_modified_only"] and name not in modified:
                continue
            names_to_show.append(name)

        imgui.BeginChild("##joint_scroll", (0, min(len(names_to_show) * 24 + 4, 500)))
        any_changed = False
        new_positions: dict[str, float] = {}
        for name in names_to_show:
            lo, hi = joint_limits.get(name, (-3.14159, 3.14159))
            cv = current.get(name, 0.0)
            is_mod = name in modified
            if is_mod:
                imgui.PushStyleColor(imgui.ImGuiCol_Text, (0.2, 1.0, 0.4, 1.0))
            changed, new_val = imgui.DragFloat(
                f"##{name}", cv, 0.005, lo, hi, f"{name}  %.4f"
            )
            if is_mod:
                imgui.PopStyleColor()
            if changed:
                any_changed = True
                new_positions[name] = new_val
        imgui.EndChild()

        if any_changed:
            cur = controller.get_joint_positions()
            cur.update(new_positions)
            controller.set_joint_positions(cur)
            _update_vis()

        imgui.TreePop()

    def _draw_blend_panel() -> None:
        if not imgui.TreeNodeEx("Blend Keyframes", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return

        c, w = imgui.SliderFloat("Blend Weight", blend_state["weight"], 0.0, 1.0)
        if c:
            blend_state["weight"] = w

        c, prev = imgui.Checkbox("Blend Preview", blend_state["preview"])
        if c:
            blend_state["preview"] = prev
            if traj_holder["traj"] is not None:
                _apply_frame(play["frame"])

        imgui.Separator()

        if imgui.Button("Record Modified KF"):
            modified = _detect_modified()
            if modified:
                kf = BlendKeyframe(
                    traj_frame=play["frame"],
                    blend_weight=blend_state["weight"],
                    joints=modified,
                )
                blend_schedule.add_keyframe(kf)
                blend_state["sel_kf"] = next(
                    i for i, k in enumerate(blend_schedule.keyframes)
                    if k.traj_frame == kf.traj_frame
                )
                print(f"[blend-tool] Recorded KF at frame {kf.traj_frame}: "
                      f"w={kf.blend_weight:.2f}, {len(modified)} joints")
            else:
                print("[blend-tool] No joints modified -- nothing to record")

        imgui.SameLine()
        if imgui.Button("Insert Full KF"):
            current = controller.get_joint_positions()
            kf = BlendKeyframe(
                traj_frame=play["frame"],
                blend_weight=blend_state["weight"],
                joints=dict(current),
            )
            blend_schedule.add_keyframe(kf)
            blend_state["sel_kf"] = next(
                i for i, k in enumerate(blend_schedule.keyframes)
                if k.traj_frame == kf.traj_frame
            )
            print(f"[blend-tool] Inserted full KF at frame {kf.traj_frame}: "
                  f"w={kf.blend_weight:.2f}, {len(current)} joints")

        if imgui.Button("Reset to Teleop##blend"):
            if traj_holder["traj"] is not None:
                teleop = _teleop_joints_at(play["frame"])
                controller.set_joint_positions(teleop)
                _update_vis()

        kfs = blend_schedule.keyframes
        n_kf = len(kfs)
        imgui.Text(f"Keyframes: {n_kf}")

        if n_kf > 0:
            labels = [
                f"KF {i}: frame={kf.traj_frame} w={kf.blend_weight:.2f} ({len(kf.joints)}j)"
                for i, kf in enumerate(kfs)
            ]
            c, sel = imgui.Combo("##kf_list", blend_state["sel_kf"], labels)
            if c:
                blend_state["sel_kf"] = sel

            si = min(blend_state["sel_kf"], n_kf - 1)
            sel_kf = kfs[si]

            if imgui.Button("Go To##kf"):
                teleop = _teleop_joints_at(sel_kf.traj_frame)
                merged = dict(teleop)
                for k, v in sel_kf.joints.items():
                    merged[k] = teleop.get(k, 0.0) + sel_kf.blend_weight * (v - teleop.get(k, 0.0))
                controller.set_joint_positions(merged)
                play["frame"] = sel_kf.traj_frame
                play["playing"] = False
                _update_vis()

            imgui.SameLine()
            if imgui.Button("Update##kf"):
                modified = _detect_modified()
                if modified:
                    new_kf = BlendKeyframe(
                        traj_frame=play["frame"],
                        blend_weight=blend_state["weight"],
                        joints=modified,
                    )
                    blend_schedule.update_keyframe(si, new_kf)
                    print(f"[blend-tool] Updated KF {si}")

            imgui.SameLine()
            if imgui.Button("Delete##kf"):
                blend_schedule.remove_keyframe(si)
                blend_state["sel_kf"] = max(0, min(blend_state["sel_kf"], len(blend_schedule) - 1))
                print(f"[blend-tool] Deleted KF {si}")

            # Show selected keyframe joints
            if imgui.TreeNode(f"KF {si} joints ({len(sel_kf.joints)})##kf_detail"):
                for jn, jv in sorted(sel_kf.joints.items()):
                    imgui.Text(f"  {jn}: {jv:+.4f}")
                imgui.TreePop()

        imgui.Separator()
        if imgui.Button("Save Blend KFs##bp"):
            _save_blend_file()
        imgui.SameLine()
        if imgui.Button("Load Blend KFs##bp"):
            _load_blend_file()
        imgui.SameLine()
        if imgui.Button("Clear All KFs##bp"):
            blend_schedule.clear()
            blend_state["sel_kf"] = 0
            print("[blend-tool] All blend keyframes cleared")

        imgui.TreePop()

    def _draw_bake_panel() -> None:
        if not imgui.TreeNode("Bake"):
            return
        traj = traj_holder["traj"]
        gnames = genesis_names_holder["names"]

        c, v = imgui.InputText("Output Dir##bake", setup["output_dir"])
        if c:
            setup["output_dir"] = v

        can_bake = traj is not None and len(blend_schedule) > 0
        if not can_bake:
            imgui.TextColored(
                (1.0, 0.8, 0.2, 1.0),
                "Need trajectory + at least 1 blend keyframe to bake."
            )

        if imgui.Button("Bake##do") and can_bake:
            out_dir = Path(setup["output_dir"].strip())
            out_dir.mkdir(parents=True, exist_ok=True)

            baked_qpos = bake_trajectory(traj.robot_qpos, gnames, blend_schedule)

            save_dict: dict[str, np.ndarray] = {
                "sim_times": traj.sim_times,
                "robot_qpos": baked_qpos,
            }
            if traj.yoyo_pos is not None:
                save_dict["yoyo_pos"] = traj.yoyo_pos
            if traj.yoyo_quat is not None:
                save_dict["yoyo_quat"] = traj.yoyo_quat
            if traj.string_particles is not None:
                save_dict["yoyo_string_particles"] = traj.string_particles

            np.savez(str(out_dir / "trajectory.npz"), **save_dict)

            # Copy genesis_joint_order.json so the replay controller finds it
            for src_dir in [traj.traj_dir, Path(_URDF_PATH).parent, _SCRIPT_DIR]:
                src = src_dir / "genesis_joint_order.json"
                if src.exists():
                    dst = out_dir / "genesis_joint_order.json"
                    if not dst.exists():
                        shutil.copy2(str(src), str(dst))
                    break

            # Copy mesh files if present
            for mesh_name in ["yoyo_ball.obj", "yoyo_string.obj"]:
                src = traj.traj_dir / mesh_name
                if src.exists():
                    dst = out_dir / mesh_name
                    if not dst.exists():
                        shutil.copy2(str(src), str(dst))

            # Save blend keyframes alongside for reference
            save_blend_keyframes(out_dir / "blend_keyframes.json", blend_schedule.keyframes)

            blend_state["bake_status"] = (
                f"Baked {traj.n_frames} frames -> {out_dir}"
            )
            print(f"[blend-tool] {blend_state['bake_status']}")

        if blend_state["bake_status"]:
            imgui.TextColored((0.2, 1.0, 0.4, 1.0), blend_state["bake_status"])

        imgui.Separator()
        baked_dir = Path(setup["output_dir"].strip()) if setup["output_dir"].strip() else None
        baked_exists = baked_dir is not None and (baked_dir / "trajectory.npz").exists()

        if baked_exists:
            if imgui.Button("Load & Play Baked##bake_run"):
                setup["traj_dir"] = str(baked_dir)
                if _load_trajectory():
                    blend_state["preview"] = False
                    play["playing"] = True
                    play["last_time"] = time.monotonic()
                    play["accum"] = 0.0
                    # Enable ghost to show original teleop for comparison
                    ghost_visible["enabled"] = True
                    for _, (mesh, _) in ghost_meshes.items():
                        mesh.set_enabled(True)
                    print(f"[blend-tool] Playing baked clip from {baked_dir}")
        else:
            imgui.TextColored(
                (1.0, 0.8, 0.2, 1.0),
                "No baked trajectory.npz found in Output Dir."
            )

        imgui.TreePop()

    def _draw_ref_panel() -> None:
        if not imgui.TreeNode("Reference OBJ"):
            return
        c, buf = imgui.InputText("OBJ Path##ref", ref_state["input_buf"])
        if c:
            ref_state["input_buf"] = buf
        if imgui.Button("Load##ref_load"):
            p = ref_state["input_buf"].strip()
            if p:
                _load_ref_obj(p)
        imgui.SameLine()
        if imgui.Button("Clear All##ref_clear"):
            _unload_all_refs()
            print("[ref] All reference meshes removed")

        entries = ref_state["entries"]
        if entries:
            labels = [f"{i}: {Path(e['path']).name}" for i, e in enumerate(entries)]
            c, sel = imgui.Combo("##ref_sel", ref_state["sel_ref"], labels)
            if c:
                ref_state["sel_ref"] = sel
            si = min(ref_state["sel_ref"], len(entries) - 1)
            entry = entries[si]
            if imgui.Button(f"Remove##{si}"):
                _unload_ref(si)
                ref_state["sel_ref"] = max(0, min(ref_state["sel_ref"], len(entries) - 1))
            else:
                t = entry["transform"]
                tf_changed = False
                for axis, idx_val in [("Tx", 0), ("Ty", 1), ("Tz", 2)]:
                    c, v = imgui.SliderFloat(f"{axis}##ref{si}", t[idx_val], -2.0, 2.0)
                    if c:
                        t[idx_val] = v
                        tf_changed = True
                for axis, idx_val in [("Rx", 3), ("Ry", 4), ("Rz", 5)]:
                    c, v = imgui.SliderFloat(f"{axis}##ref{si}", t[idx_val], -180.0, 180.0)
                    if c:
                        t[idx_val] = v
                        tf_changed = True
                if tf_changed:
                    _apply_ref_transform(entry)
                if imgui.Button(f"Reset Transform##ref{si}"):
                    entry["transform"] = [0.0] * 6
                    _apply_ref_transform(entry)
        imgui.TreePop()

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def on_update() -> None:
        imgui.Text("=== Blend Baking Tool ===")

        changed, gv = imgui.Checkbox("Show Original (ghost)", ghost_visible["enabled"])
        if changed:
            ghost_visible["enabled"] = gv
            for _, (mesh, _) in ghost_meshes.items():
                mesh.set_enabled(gv)

        imgui.Separator()
        _draw_setup_panel()
        imgui.Separator()
        _draw_playback_panel()
        imgui.Separator()
        _draw_ik_panel()
        imgui.Separator()
        _draw_joint_values_panel()
        imgui.Separator()
        _draw_blend_panel()
        imgui.Separator()
        _draw_ref_panel()
        imgui.Separator()
        _draw_bake_panel()

        # FK playback tick
        traj = traj_holder["traj"]
        if play["playing"] and traj is not None:
            now = time.monotonic()
            dt_wall = now - play["last_time"]
            play["last_time"] = now
            play["accum"] += dt_wall * play["speed"]

            n = traj.n_frames
            adv = int(play["accum"] / traj.dt) if traj.dt > 0 else 0
            if adv > 0:
                play["accum"] -= adv * traj.dt
                new_frame = play["frame"] + adv
                if new_frame >= n:
                    if play["loop"]:
                        new_frame = new_frame % n
                    else:
                        new_frame = n - 1
                        play["playing"] = False
                _apply_frame(new_frame)

    ps.set_user_callback(on_update)
    ps.show()


if __name__ == "__main__":
    main()
