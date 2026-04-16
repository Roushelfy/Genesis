"""Standalone Polyscope GUI tool for blending cooking teleop trajectories.

Controls only the pan and spatula (rigid-body pose: position + quaternion).
No robot, no joints, no IK -- just direct slider control of 6-DOF poses.

Workflow:
  1. Pan / spatula USD meshes load automatically on startup.
  2. Load a cooking teleop trajectory (JSON) via the Setup panel.
  3. Scrub to a frame, adjust pan/spatula pose via sliders, record a blend
     keyframe.  Only modified channels are stored.
  4. Preview the blended result by playing the timeline with blend preview ON.
  5. Bake to a new trajectory JSON for use with replay_cook.py.

Usage:
    python tool_create_blending.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

from cook_blend_utils import (
    CHANNEL_NAMES,
    ENTITIES,
    BlendKeyframe,
    BlendSchedule,
    bake_cook_trajectory,
    channels_to_frame,
    euler_deg_to_quat,
    frame_to_channels,
    load_blend_keyframes,
    load_cook_trajectory,
    quat_to_euler_deg,
    renormalize_quat_channels,
    save_blend_keyframes,
    save_cook_trajectory,
)
from usd_mesh_loader import load_usd_mesh

_SCRIPT_DIR = Path(__file__).resolve().parent
_ASSET_ROOT = _SCRIPT_DIR.parent
_COOK_ROOT = _ASSET_ROOT.parent / "cook"

PAN_USD = _COOK_ROOT / "Pan025" / "Pan025.usd"
SPATULA_USD = _COOK_ROOT / "Spatula018" / "Spatula018.usd"
_DEFAULT_TRAJ = str(_COOK_ROOT / "trajectories" / "cooking_demo.json")
_DEFAULT_BLEND_FILE = str(_SCRIPT_DIR / "blend_keyframes.json")
_DEFAULT_OUTPUT = str(_COOK_ROOT / "trajectories" / "cooking_demo_blended.json")
_DEFAULT_PLACEMENT = str(_ASSET_ROOT / "placement.json")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def quat_pos_to_4x4(quat, pos):
    """[w, x, y, z] quaternion + [x, y, z] position -> 4x4 matrix."""
    w, x, y, z = quat
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = 1 - 2 * (y * y + z * z)
    M[0, 1] = 2 * (x * y - w * z)
    M[0, 2] = 2 * (x * z + w * y)
    M[1, 0] = 2 * (x * y + w * z)
    M[1, 1] = 1 - 2 * (x * x + z * z)
    M[1, 2] = 2 * (y * z - w * x)
    M[2, 0] = 2 * (x * z - w * y)
    M[2, 1] = 2 * (y * z + w * x)
    M[2, 2] = 1 - 2 * (x * x + y * y)
    M[0, 3] = pos[0]
    M[1, 3] = pos[1]
    M[2, 3] = pos[2]
    return M


def _apply_4x4(verts, M):
    """Apply 4x4 homogeneous transform to Nx3 vertices."""
    V4 = np.ones((len(verts), 4), dtype=np.float64)
    V4[:, :3] = verts
    return (V4 @ M.T)[:, :3]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load USD meshes for pan and spatula
    # ------------------------------------------------------------------
    print("[blend-tool] Loading USD meshes ...")
    pan_verts_raw, pan_faces = load_usd_mesh(PAN_USD)
    spatula_verts_raw, spatula_faces = load_usd_mesh(SPATULA_USD)

    # Load pan scale from placement.json if available
    pan_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    if Path(_DEFAULT_PLACEMENT).exists():
        try:
            with open(_DEFAULT_PLACEMENT) as f:
                placement = json.load(f)
            raw_scale = placement.get("pan", {}).get("scale", [1.0, 1.0, 1.0])
            if isinstance(raw_scale, (int, float)):
                pan_scale[:] = [raw_scale, raw_scale, raw_scale]
            else:
                pan_scale[:] = raw_scale
            print(f"[blend-tool] Pan scale from placement: {pan_scale.tolist()}")
        except Exception as e:
            print(f"[blend-tool] Could not load placement: {e}")

    pan_verts_scaled = pan_verts_raw * pan_scale

    # ------------------------------------------------------------------
    # 2. Polyscope init
    # ------------------------------------------------------------------
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    # Register blended meshes (solid white)
    pan_mesh = ps.register_surface_mesh("pan_blend", pan_verts_scaled, pan_faces)
    pan_mesh.set_transparency(1.0)
    pan_mesh.set_color((1.0, 1.0, 1.0))

    spat_mesh = ps.register_surface_mesh("spatula_blend", spatula_verts_raw, spatula_faces)
    spat_mesh.set_transparency(1.0)
    spat_mesh.set_color((1.0, 1.0, 1.0))

    # Register ghost meshes (solid red, for original teleop)
    pan_ghost = ps.register_surface_mesh("pan_ghost", pan_verts_scaled, pan_faces)
    pan_ghost.set_transparency(1.0)
    pan_ghost.set_color((1.0, 0.3, 0.3))

    spat_ghost = ps.register_surface_mesh("spatula_ghost", spatula_verts_raw, spatula_faces)
    spat_ghost.set_transparency(1.0)
    spat_ghost.set_color((1.0, 0.3, 0.3))

    ghost_visible = {"enabled": True}

    def _update_mesh(entity: str, pos, quat, is_ghost=False):
        M = quat_pos_to_4x4(quat, pos)
        if entity == "pan":
            verts = _apply_4x4(pan_verts_scaled, M)
            mesh = pan_ghost if is_ghost else pan_mesh
        else:
            verts = _apply_4x4(spatula_verts_raw, M)
            mesh = spat_ghost if is_ghost else spat_mesh
        mesh.update_vertex_positions(verts)

    # ------------------------------------------------------------------
    # 3. Mutable state
    # ------------------------------------------------------------------

    # Setup paths
    setup = {
        "traj_file": _DEFAULT_TRAJ,
        "blend_file": _DEFAULT_BLEND_FILE,
        "output_file": _DEFAULT_OUTPUT,
        "status": "",
    }

    # Trajectory (loaded on demand)
    traj_holder: dict = {"data": None, "frames": [], "times": []}

    # Playback
    play = {
        "frame": 0,
        "playing": False,
        "speed": 1.0,
        "loop": True,
        "last_time": time.monotonic(),
        "accum": 0.0,
    }

    # Current pose state (edited by sliders)
    cur_state: dict = {
        "pan_pos": [0.0, 0.0, 0.0],
        "pan_euler": [0.0, 0.0, 0.0],
        "pan_quat": [1.0, 0.0, 0.0, 0.0],
        "spatula_pos": [0.0, 0.0, 0.0],
        "spatula_euler": [0.0, 0.0, 0.0],
        "spatula_quat": [1.0, 0.0, 0.0, 0.0],
    }

    # Blend
    blend_schedule = BlendSchedule()
    blend_state = {
        "weight": 1.0,
        "sel_kf": 0,
        "preview": True,
        "bake_status": "",
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _n_frames() -> int:
        return len(traj_holder["frames"])

    def _teleop_channels_at(frame: int) -> dict[str, float]:
        frames = traj_holder["frames"]
        if not frames:
            return {}
        frame = max(0, min(frame, len(frames) - 1))
        return frame_to_channels(frames[frame])

    def _current_channels() -> dict[str, float]:
        channels: dict[str, float] = {}
        for ent in ENTITIES:
            pos = cur_state[f"{ent}_pos"]
            quat = cur_state[f"{ent}_quat"]
            channels[f"{ent}.pos.x"] = pos[0]
            channels[f"{ent}.pos.y"] = pos[1]
            channels[f"{ent}.pos.z"] = pos[2]
            channels[f"{ent}.quat.w"] = quat[0]
            channels[f"{ent}.quat.x"] = quat[1]
            channels[f"{ent}.quat.y"] = quat[2]
            channels[f"{ent}.quat.z"] = quat[3]
        return channels

    def _set_current_from_channels(channels: dict[str, float]) -> None:
        for ent in ENTITIES:
            pos = cur_state[f"{ent}_pos"]
            quat = cur_state[f"{ent}_quat"]
            pos[0] = channels.get(f"{ent}.pos.x", pos[0])
            pos[1] = channels.get(f"{ent}.pos.y", pos[1])
            pos[2] = channels.get(f"{ent}.pos.z", pos[2])
            quat[0] = channels.get(f"{ent}.quat.w", quat[0])
            quat[1] = channels.get(f"{ent}.quat.x", quat[1])
            quat[2] = channels.get(f"{ent}.quat.y", quat[2])
            quat[3] = channels.get(f"{ent}.quat.z", quat[3])
            cur_state[f"{ent}_euler"] = quat_to_euler_deg(quat)

    def _update_all_meshes() -> None:
        for ent in ENTITIES:
            _update_mesh(ent, cur_state[f"{ent}_pos"], cur_state[f"{ent}_quat"])

    def _update_ghost_meshes() -> None:
        if not ghost_visible["enabled"]:
            return
        teleop = _teleop_channels_at(play["frame"])
        if not teleop:
            return
        for ent in ENTITIES:
            pos = [teleop.get(f"{ent}.pos.{a}", 0.0) for a in ("x", "y", "z")]
            quat = [teleop.get(f"{ent}.quat.{a}", 0.0) for a in ("w", "x", "y", "z")]
            _update_mesh(ent, pos, quat, is_ghost=True)

    def _apply_frame(idx: int) -> None:
        frames = traj_holder["frames"]
        if not frames:
            return
        idx = max(0, min(idx, len(frames) - 1))
        play["frame"] = idx

        teleop = _teleop_channels_at(idx)
        if blend_state["preview"]:
            blended = blend_schedule.blend_joints(teleop, float(idx))
            for ent in ENTITIES:
                renormalize_quat_channels(blended, ent)
        else:
            blended = teleop

        _set_current_from_channels(blended)
        _update_all_meshes()
        _update_ghost_meshes()

    def _detect_modified() -> dict[str, float]:
        teleop = _teleop_channels_at(play["frame"])
        current = _current_channels()
        return {
            name: current[name]
            for name in CHANNEL_NAMES
            if name in current and abs(current[name] - teleop.get(name, 0.0)) > 1e-5
        }

    # ------------------------------------------------------------------
    # Trajectory load/save
    # ------------------------------------------------------------------

    def _load_trajectory() -> bool:
        traj_file = setup["traj_file"].strip()
        if not traj_file:
            setup["status"] = "No trajectory file specified"
            return False
        p = Path(traj_file)
        if not p.exists():
            setup["status"] = f"Not found: {p}"
            print(f"[blend-tool] {setup['status']}")
            return False
        try:
            data = load_cook_trajectory(p)
        except Exception as e:
            setup["status"] = f"Load error: {e}"
            print(f"[blend-tool] {setup['status']}")
            return False

        frames = data.get("frames", [])
        times = [f.get("sim_time", i / 60.0) for i, f in enumerate(frames)]
        traj_holder["data"] = data
        traj_holder["frames"] = frames
        traj_holder["times"] = times
        play["frame"] = 0
        play["playing"] = False

        n = len(frames)
        t0 = times[0] if times else 0.0
        t1 = times[-1] if times else 0.0
        setup["status"] = f"Loaded: {n} frames, t={t0:.3f}..{t1:.3f}s"
        print(f"[blend-tool] {setup['status']}")

        _apply_frame(0)
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

    # ------------------------------------------------------------------
    # GUI Panels
    # ------------------------------------------------------------------

    def _draw_setup_panel() -> None:
        if not imgui.TreeNodeEx("Setup", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return

        c, v = imgui.InputText("Traj File##setup", setup["traj_file"])
        if c:
            setup["traj_file"] = v
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
        c, v = imgui.InputText("Output File##setup", setup["output_file"])
        if c:
            setup["output_file"] = v

        if setup["status"]:
            imgui.TextWrapped(setup["status"])
        imgui.TreePop()

    def _draw_playback_panel() -> None:
        n = _n_frames()
        if n == 0:
            imgui.Text("No trajectory loaded.")
            return
        if not imgui.TreeNodeEx("Trajectory Playback", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            return

        times = traj_holder["times"]
        t_val = times[play["frame"]] if play["frame"] < n else 0.0
        total_t = times[-1] if n > 0 else 0.0
        imgui.Text(f"Time: {t_val:.3f}s / {total_t:.3f}s   Frame: {play['frame']} / {n - 1}")

        if imgui.Button("Play" if not play["playing"] else "Pause"):
            play["playing"] = not play["playing"]
            play["last_time"] = time.monotonic()
            play["accum"] = 0.0
        imgui.SameLine()
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

    def _draw_entity_control(entity: str) -> None:
        label = entity.capitalize()
        if not imgui.TreeNode(f"{label} Control"):
            return

        pos = cur_state[f"{entity}_pos"]
        euler = cur_state[f"{entity}_euler"]
        quat = cur_state[f"{entity}_quat"]

        # Position sliders
        pose_changed = False
        c, pos[0] = imgui.SliderFloat(f"X##{entity}", pos[0], -1.0, 2.0)
        pose_changed |= c
        c, pos[1] = imgui.SliderFloat(f"Y##{entity}", pos[1], -1.0, 2.0)
        pose_changed |= c
        c, pos[2] = imgui.SliderFloat(f"Z##{entity}", pos[2], -1.0, 2.0)
        pose_changed |= c

        imgui.Separator()

        # Euler angle sliders (editing)
        euler_changed = False
        c, euler[0] = imgui.SliderFloat(f"Rx (deg)##{entity}", euler[0], -180.0, 180.0)
        euler_changed |= c
        c, euler[1] = imgui.SliderFloat(f"Ry (deg)##{entity}", euler[1], -180.0, 180.0)
        euler_changed |= c
        c, euler[2] = imgui.SliderFloat(f"Rz (deg)##{entity}", euler[2], -180.0, 180.0)
        euler_changed |= c

        if euler_changed:
            new_quat = euler_deg_to_quat(euler)
            quat[:] = new_quat

        # Read-only quaternion display
        imgui.Text(f"Quat: w={quat[0]:+.5f} x={quat[1]:+.5f} y={quat[2]:+.5f} z={quat[3]:+.5f}")

        if pose_changed or euler_changed:
            _update_mesh(entity, pos, quat)

        imgui.Separator()
        if imgui.Button(f"Reset to Teleop##{entity}"):
            teleop = _teleop_channels_at(play["frame"])
            pos[0] = teleop.get(f"{entity}.pos.x", pos[0])
            pos[1] = teleop.get(f"{entity}.pos.y", pos[1])
            pos[2] = teleop.get(f"{entity}.pos.z", pos[2])
            quat[0] = teleop.get(f"{entity}.quat.w", quat[0])
            quat[1] = teleop.get(f"{entity}.quat.x", quat[1])
            quat[2] = teleop.get(f"{entity}.quat.y", quat[2])
            quat[3] = teleop.get(f"{entity}.quat.z", quat[3])
            euler[:] = quat_to_euler_deg(quat)
            _update_mesh(entity, pos, quat)

        imgui.TreePop()

    def _draw_values_panel() -> None:
        if not imgui.TreeNode("Values (Teleop vs Current)"):
            return

        n = _n_frames()
        if n == 0:
            imgui.Text("No trajectory loaded.")
            imgui.TreePop()
            return

        teleop = _teleop_channels_at(play["frame"])
        current = _current_channels()
        modified = _detect_modified()
        n_mod = len(modified)
        imgui.Text(f"Modified channels: {n_mod} / {len(CHANNEL_NAMES)}")
        imgui.Separator()

        imgui.Columns(4, "val_cols", True)
        imgui.Text("Channel")
        imgui.NextColumn()
        imgui.Text("Teleop")
        imgui.NextColumn()
        imgui.Text("Current")
        imgui.NextColumn()
        imgui.Text("Delta")
        imgui.NextColumn()
        imgui.Separator()

        for name in CHANNEL_NAMES:
            tv = teleop.get(name, 0.0)
            cv = current.get(name, 0.0)
            delta = cv - tv
            is_mod = name in modified
            if is_mod:
                imgui.PushStyleColor(imgui.ImGuiCol_Text, (0.2, 1.0, 0.4, 1.0))
            imgui.Text(name)
            imgui.NextColumn()
            imgui.Text(f"{tv:+.5f}")
            imgui.NextColumn()
            imgui.Text(f"{cv:+.5f}")
            imgui.NextColumn()
            imgui.Text(f"{delta:+.5f}")
            imgui.NextColumn()
            if is_mod:
                imgui.PopStyleColor()
        imgui.Columns(1)
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
            if _n_frames() > 0:
                _apply_frame(play["frame"])

        imgui.Separator()

        if imgui.Button("Record Blend KF"):
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
                      f"w={kf.blend_weight:.2f}, {len(modified)} channels")
            else:
                print("[blend-tool] No channels modified -- nothing to record")

        imgui.SameLine()
        if imgui.Button("Reset to Teleop##blend"):
            if _n_frames() > 0:
                teleop = _teleop_channels_at(play["frame"])
                _set_current_from_channels(teleop)
                _update_all_meshes()

        kfs = blend_schedule.keyframes
        n_kf = len(kfs)
        imgui.Text(f"Keyframes: {n_kf}")

        if n_kf > 0:
            labels = [
                f"KF {i}: frame={kf.traj_frame} w={kf.blend_weight:.2f} ({len(kf.joints)}ch)"
                for i, kf in enumerate(kfs)
            ]
            c, sel = imgui.Combo("##kf_list", blend_state["sel_kf"], labels)
            if c:
                blend_state["sel_kf"] = sel

            si = min(blend_state["sel_kf"], n_kf - 1)
            sel_kf = kfs[si]

            if imgui.Button("Go To##kf"):
                teleop = _teleop_channels_at(sel_kf.traj_frame)
                merged = dict(teleop)
                for k, v in sel_kf.joints.items():
                    tv = teleop.get(k, 0.0)
                    merged[k] = tv + sel_kf.blend_weight * (v - tv)
                for ent in ENTITIES:
                    renormalize_quat_channels(merged, ent)
                _set_current_from_channels(merged)
                play["frame"] = sel_kf.traj_frame
                play["playing"] = False
                _update_all_meshes()
                _update_ghost_meshes()

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

            if imgui.TreeNode(f"KF {si} channels ({len(sel_kf.joints)})##kf_detail"):
                for jn, jv in sorted(sel_kf.joints.items()):
                    imgui.Text(f"  {jn}: {jv:+.5f}")
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

        c, v = imgui.InputText("Output File##bake", setup["output_file"])
        if c:
            setup["output_file"] = v

        frames = traj_holder["frames"]
        can_bake = len(frames) > 0 and len(blend_schedule) > 0
        if not can_bake:
            imgui.TextColored(
                (1.0, 0.8, 0.2, 1.0),
                "Need trajectory + at least 1 blend keyframe to bake."
            )

        if imgui.Button("Bake##do") and can_bake:
            out_path = Path(setup["output_file"].strip())
            out_path.parent.mkdir(parents=True, exist_ok=True)

            baked_frames = bake_cook_trajectory(frames, blend_schedule)

            out_data = dict(traj_holder["data"])
            out_data["frames"] = baked_frames

            save_cook_trajectory(out_path, out_data)

            blend_state["bake_status"] = (
                f"Baked {len(baked_frames)} frames -> {out_path.name}"
            )
            print(f"[blend-tool] {blend_state['bake_status']}")

        if blend_state["bake_status"]:
            imgui.TextColored((0.2, 1.0, 0.4, 1.0), blend_state["bake_status"])

        imgui.TreePop()

    # ------------------------------------------------------------------
    # Try loading defaults on startup
    # ------------------------------------------------------------------
    _load_trajectory()
    _load_blend_file()

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def on_update() -> None:
        imgui.Text("=== Cook Blend Baking Tool ===")

        changed, gv = imgui.Checkbox("Show Original (ghost)", ghost_visible["enabled"])
        if changed:
            ghost_visible["enabled"] = gv
            pan_ghost.set_enabled(gv)
            spat_ghost.set_enabled(gv)

        imgui.Separator()
        _draw_setup_panel()
        imgui.Separator()
        _draw_playback_panel()
        imgui.Separator()
        _draw_entity_control("pan")
        imgui.Separator()
        _draw_entity_control("spatula")
        imgui.Separator()
        _draw_values_panel()
        imgui.Separator()
        _draw_blend_panel()
        imgui.Separator()
        _draw_bake_panel()

        # FK playback tick
        n = _n_frames()
        if play["playing"] and n > 0:
            now = time.monotonic()
            dt_wall = now - play["last_time"]
            play["last_time"] = now
            play["accum"] += dt_wall * play["speed"]

            times = traj_holder["times"]
            if len(times) >= 2:
                avg_dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
            else:
                avg_dt = 1.0 / 60.0

            adv = int(play["accum"] / avg_dt) if avg_dt > 0 else 0
            if adv > 0:
                play["accum"] -= adv * avg_dt
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
