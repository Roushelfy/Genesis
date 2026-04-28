"""stitch_gear_sequences.py — GUI editor for combining teleop + IK sequences.

Phase 1  :  teleop trajectory  (trajectory_gear_sharpa.npz)
Transition:  cubic-spline robot + slerp gears
Phase 2  :  IK sequence  (sim_gear_with_robot_setup --headless output)

GUI controls (polyscope + imgui):
  - Trim Phase 1: start / end frame sliders
  - Trim Phase 2: start / end frame sliders
  - Transition: blend-frames slider
  - Scrubber + Play/Pause to preview the combined result
  - Save / Load clip config (JSON)
  - Save combined sequence (.npz)
  - Replay a previously saved sequence

Headless (no GUI):
    python stitch_gear_sequences.py --no-gui \\
        --phase1 ... --phase2 ... --blend-frames 60 --output full.npz

Replay saved sequence:
    python stitch_gear_sequences.py --replay full_sequence.npz

Usage (GUI):
    python stitch_gear_sequences.py --phase1 ... --phase2 ... --output full.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

# ---------------------------------------------------------------------------
# Paths shared with sim_gear_with_robot_setup
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent
_DEMO_ASSETS = _HERE.parent
_GEAR        = _DEMO_ASSETS / "planetary_gear"
_GEAR_ASSETS = _GEAR / "assets"
_SHARPA_URDF = _DEMO_ASSETS / "marvin_sharpa_description" / "marvin_sharpa.urdf"
_YOYO        = _DEMO_ASSETS / "yoyo" / "scripts"
if str(_YOYO) not in sys.path:
    sys.path.insert(0, str(_YOYO))

MESH_SCALE        = 0.0012

# Rendering formula: V_world = R(q_vis) @ V_scaled + pos

_GEAR_KEYS = [
    "rigid_sun_gear", "rigid_carrier", "rigid_ring_gear",
    "rigid_planet_gear_0", "rigid_planet_gear_1", "rigid_planet_gear_2",
]
_ALL_RIGID_KEYS = _GEAR_KEYS + ["rigid_support_pin"]

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

_CLIP_CONFIG_FILE = _HERE / "stitch_clip_config.json"

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _pose_to_pos_quat(pose7: np.ndarray):
    pos = pose7[:3].astype(np.float64)
    qw, qx, qy, qz = pose7[3:].astype(np.float64)
    return pos, Rotation.from_quat([qx, qy, qz, qw])   # scipy scalar-last


def _pos_rot_to_pose(pos: np.ndarray, rot: Rotation) -> np.ndarray:
    qx, qy, qz, qw = rot.as_quat()
    return np.array([*pos, qw, qx, qy, qz], dtype=np.float32)


def _slerp_poses(pose_a, pose_b, t_vals):
    pos_a, rot_a = _pose_to_pos_quat(pose_a)
    pos_b, rot_b = _pose_to_pos_quat(pose_b)
    qa = rot_a.as_quat()
    qb = rot_b.as_quat()
    if np.dot(qa, qb) < 0:
        rot_b = Rotation.from_quat(-qb)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot_a, rot_b]))
    out = np.zeros((len(t_vals), 7), dtype=np.float32)
    for i, (t, r) in enumerate(zip(t_vals, slerp(t_vals))):
        out[i] = _pos_rot_to_pose(pos_a + t * (pos_b - pos_a), r)
    return out


def _spline_qpos(q0, q1, n):
    cs = CubicSpline([0.0, 1.0], np.stack([q0.astype(np.float64),
                                            q1.astype(np.float64)]),
                     bc_type="clamped")
    return cs(np.linspace(0.0, 1.0, n + 1)[:-1]).astype(np.float32)


def build_combined(p1: dict, p2: dict,
                   p1_s: int, p1_e: int,
                   p2_s: int, p2_e: int,
                   blend: int) -> dict:
    """Return dict with same keys as trajectory npz, including support_pin."""
    q1 = p1["robot_qpos"][p1_s:p1_e + 1]
    q2 = p2["robot_qpos"][p2_s:p2_e + 1]
    t1 = p1["sim_time"][p1_s:p1_e + 1].astype(np.float32)
    dt = float(t1[1] - t1[0]) if len(t1) > 1 else 0.01667
    t2 = (t1[-1] + dt * (1 + np.arange(blend))).astype(np.float32)
    t3 = (t2[-1] + dt * (1 + np.arange(len(q2)))).astype(np.float32)
    tv = np.linspace(0.0, 1.0, blend + 1)[1:].astype(np.float64)

    result: dict[str, np.ndarray] = {
        "sim_time":   np.concatenate([t1, t2, t3]),
        "robot_qpos": np.concatenate([q1, _spline_qpos(q1[-1], q2[0], blend), q2]),
    }
    for k in _ALL_RIGID_KEYS:
        a = p1.get(k); b = p2.get(k)
        if a is None and b is None:
            continue
        seg1 = a[p1_s:p1_e + 1] if a is not None else np.tile(b[p2_s:p2_s+1], (len(q1), 1))
        seg2 = b[p2_s:p2_e + 1] if b is not None else np.tile(seg1[-1:], (len(q2), 1))
        bl   = _slerp_poses(seg1[-1], seg2[0], tv) if (a is not None and b is not None) \
               else (np.tile(seg1[-1], (blend, 1)) if b is None else np.tile(seg2[0], (blend, 1)))
        result[k] = np.concatenate([seg1, bl, seg2]).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# OBJ / gear visualization helpers
# ---------------------------------------------------------------------------

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


def _build_gear_body_local(V_obj_mm: np.ndarray) -> np.ndarray:
    return (V_obj_mm * MESH_SCALE).astype(np.float64)


# ---------------------------------------------------------------------------
# Robot helper
# ---------------------------------------------------------------------------

def _load_joint_order():
    for c in (_GEAR / "genesis_joint_order.json",
              _DEMO_ASSETS / "marvin_sharpa_description" / "genesis_joint_order.json"):
        if c.exists():
            return json.loads(c.read_text())
    return []


# ---------------------------------------------------------------------------
# Clip config save / load
# ---------------------------------------------------------------------------

def _save_clip_config(st: dict, path: Path = _CLIP_CONFIG_FILE) -> None:
    cfg = {
        "p1_s": st["p1_s"], "p1_e": st["p1_e"],
        "p2_s": st["p2_s"], "p2_e": st["p2_e"],
        "blend": st["blend"],
    }
    path.write_text(json.dumps(cfg, indent=2))
    print(f"[clip] saved config -> {path}")


def _load_clip_config(st: dict, path: Path = _CLIP_CONFIG_FILE) -> bool:
    if not path.exists():
        return False
    cfg = json.loads(path.read_text())
    for k in ("p1_s", "p1_e", "p2_s", "p2_e", "blend"):
        if k in cfg:
            st[k] = cfg[k]
    st["dirty"] = True
    print(f"[clip] loaded config <- {path}")
    return True


# ---------------------------------------------------------------------------
# Headless build + save
# ---------------------------------------------------------------------------

def _headless_save(p1, p2, p1_s, p1_e, p2_s, p2_e, blend, out_path):
    from convert_to_genesis import convert_to_genesis

    combo = build_combined(p1, p2, p1_s, p1_e, p2_s, p2_e, blend)
    n = len(combo["sim_time"])
    has_pin = "rigid_support_pin" in combo
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save UIPC convention (for Polyscope replay)
    np.savez_compressed(str(out_path), **combo)
    print(f"[stitch] saved {n} frames -> {out_path}  "
          f"({out_path.stat().st_size//1024} KB)  pin={'yes' if has_pin else 'NO'}")

    # Save Genesis convention (for Genesis GLB replay)
    gs_path = out_path.parent / f"gs_{out_path.name}"
    gs_combo = convert_to_genesis(combo)
    np.savez_compressed(str(gs_path), **gs_combo)
    print(f"[stitch] saved {n} frames -> {gs_path}  "
          f"({gs_path.stat().st_size//1024} KB)  (genesis convention)")


# ---------------------------------------------------------------------------
# Replay mode: load and play a previously saved sequence
# ---------------------------------------------------------------------------

def run_replay(seq_path: Path) -> None:
    import polyscope as ps
    from polyscope import imgui
    from urdf_controller import URDFController  # type: ignore

    data = dict(np.load(str(seq_path)))
    n_frames = len(data["sim_time"])
    print(f"[replay] {seq_path.name}: {n_frames} frames, keys={sorted(data.keys())}")
    has_pin = "rigid_support_pin" in data

    gear_V_local: dict[str, np.ndarray] = {}
    gear_F: dict[str, np.ndarray] = {}
    for k, obj in _GEAR_OBJ.items():
        path = _GEAR_ASSETS / obj
        if not path.exists():
            continue
        V, F = _load_obj(str(path))
        gear_V_local[k] = _build_gear_body_local(V)
        gear_F[k] = F

    joint_order = _load_joint_order()
    n_use = len(joint_order)
    vis_ctrl = URDFController(str(_SHARPA_URDF), mesh_source="visual")
    root_tf = np.eye(4); root_tf[2, 3] = 1.08
    vis_ctrl.set_root_transform(root_tf)

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1600, 1000)
    ps.set_automatically_compute_scene_extents(False)

    ps_gears: dict[str, ps.SurfaceMesh] = {}
    for k in gear_V_local:
        if k not in data and k != "rigid_support_pin":
            continue
        V0 = gear_V_local[k]
        sm = ps.register_surface_mesh(f"gear/{k}", V0, gear_F[k])
        sm.set_color(_GEAR_COLOR.get(k, (0.7, 0.7, 0.7)))
        sm.set_smooth_shade(True)
        ps_gears[k] = sm

    tfs = vis_ctrl.get_mesh_transforms()
    ps_robot: dict[str, ps.SurfaceMesh] = {}
    for node in vis_ctrl.mesh_nodes:
        tf = tfs.get(node.node_name, np.eye(4))
        wv = node.local_vertices @ tf[:3, :3].T + tf[:3, 3]
        m = ps.register_surface_mesh(f"robot/{node.node_name}", wv, node.faces)
        m.set_transparency(0.35)
        m.set_smooth_shade(True)
        ps_robot[node.node_name] = m

    st = {"frame": 0, "play": False, "_acc": 0.0, "_last": time.perf_counter()}

    def _update(f):
        q = data["robot_qpos"][f]
        if n_use and len(q) >= n_use:
            vis_ctrl.set_joint_positions({joint_order[i]: float(q[i]) for i in range(n_use)})
            tfs = vis_ctrl.get_mesh_transforms()
            for nd in vis_ctrl.mesh_nodes:
                tf = tfs.get(nd.node_name)
                if tf is not None and nd.node_name in ps_robot:
                    ps_robot[nd.node_name].update_vertex_positions(
                        nd.local_vertices @ tf[:3, :3].T + tf[:3, 3])
        for k, sm in ps_gears.items():
            poses = data.get(k)
            if poses is not None and f < len(poses):
                sm.update_vertex_positions(_apply_gear_transform(gear_V_local[k], poses[f]))

    _update(0)
    fps = 60.0

    def gui_cb():
        now = time.perf_counter()
        dt_w = now - st["_last"]; st["_last"] = now

        imgui.Text(f"=== Replay: {seq_path.name} ===")
        imgui.Text(f"{n_frames} frames   pin={'yes' if has_pin else 'NO'}")
        imgui.Separator()

        if imgui.Button("Pause" if st["play"] else "Play"):
            st["play"] = not st["play"]; st["_acc"] = 0.0
        imgui.SameLine()
        if imgui.Button("Reset"):
            st["frame"] = 0; st["play"] = False

        c, v = imgui.SliderInt("Frame##replay", st["frame"], 0, n_frames - 1)
        if c:
            st["frame"] = v; st["play"] = False

        if st["play"]:
            st["_acc"] += dt_w * fps
            adv = int(st["_acc"]); st["_acc"] -= adv
            st["frame"] = (st["frame"] + adv) % n_frames

        _update(st["frame"])

    ps.set_user_callback(gui_cb)
    ps.show()


# ---------------------------------------------------------------------------
# GUI (stitch editor)
# ---------------------------------------------------------------------------

def run_gui(p1: dict, p2: dict, out_path: Path) -> None:
    import polyscope as ps
    from polyscope import imgui
    from urdf_controller import URDFController  # type: ignore

    n1 = int(p1["robot_qpos"].shape[0])
    n2 = int(p2["robot_qpos"].shape[0])

    _P1_DEFAULT_START = min(100, n1 - 1)
    _P1_DEFAULT_END   = min(488, n1 - 1)
    st = {
        "p1_s": _P1_DEFAULT_START, "p1_e": _P1_DEFAULT_END,
        "p2_s": 0,   "p2_e": n2 - 1,
        "blend": 60,
        "frame": 0,
        "play": False,
        "dirty": True,
        "saved": False,
        "combo": None,
        "_acc": 0.0, "_last": time.perf_counter(),
    }

    # Try loading saved clip config
    if _CLIP_CONFIG_FILE.exists():
        _load_clip_config(st)

    def _rebuild():
        st["combo"] = build_combined(
            p1, p2,
            st["p1_s"], st["p1_e"],
            st["p2_s"], st["p2_e"],
            st["blend"])
        st["frame"] = min(st["frame"], len(st["combo"]["sim_time"]) - 1)
        st["dirty"] = False

    # --- gear visuals ---
    gear_V_local: dict[str, np.ndarray] = {}
    gear_F: dict[str, np.ndarray] = {}
    for k, obj in _GEAR_OBJ.items():
        path = _GEAR_ASSETS / obj
        if not path.exists():
            continue
        V, F = _load_obj(str(path))
        gear_V_local[k] = _build_gear_body_local(V)
        gear_F[k] = F

    # --- robot visual ---
    joint_order = _load_joint_order()
    n_use = len(joint_order)
    vis_ctrl = URDFController(str(_SHARPA_URDF), mesh_source="visual")
    root_tf = np.eye(4); root_tf[2, 3] = 1.08
    vis_ctrl.set_root_transform(root_tf)

    # --- polyscope init ---
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_window_size(1600, 1000)
    ps.set_automatically_compute_scene_extents(False)

    # Register gear meshes
    ps_gears: dict[str, ps.SurfaceMesh] = {}
    for k in gear_V_local:
        V0 = gear_V_local[k]
        sm = ps.register_surface_mesh(f"gear/{k}", V0, gear_F[k])
        sm.set_color(_GEAR_COLOR.get(k, (0.7, 0.7, 0.7)))
        sm.set_smooth_shade(True)
        ps_gears[k] = sm

    # Register robot meshes
    tfs = vis_ctrl.get_mesh_transforms()
    ps_robot: dict[str, ps.SurfaceMesh] = {}
    for node in vis_ctrl.mesh_nodes:
        tf = tfs.get(node.node_name, np.eye(4))
        wv = node.local_vertices @ tf[:3, :3].T + tf[:3, 3]
        m = ps.register_surface_mesh(f"robot/{node.node_name}", wv, node.faces)
        m.set_transparency(0.35)
        m.set_smooth_shade(True)
        ps_robot[node.node_name] = m

    def _update_display(frame: int) -> None:
        if st["combo"] is None:
            return
        q = st["combo"]["robot_qpos"][frame]
        if n_use and len(q) >= n_use:
            vis_ctrl.set_joint_positions({joint_order[i]: float(q[i])
                                          for i in range(n_use)})
            tfs = vis_ctrl.get_mesh_transforms()
            for nd in vis_ctrl.mesh_nodes:
                tf = tfs.get(nd.node_name)
                if tf is not None and nd.node_name in ps_robot:
                    ps_robot[nd.node_name].update_vertex_positions(
                        nd.local_vertices @ tf[:3, :3].T + tf[:3, 3])
        for k, sm in ps_gears.items():
            poses = st["combo"].get(k)
            if poses is not None and frame < len(poses):
                V_w = _apply_gear_transform(gear_V_local[k], poses[frame])
                sm.update_vertex_positions(V_w)

    _rebuild()
    _update_display(0)

    fps = 60.0

    def gui_callback() -> None:
        now = time.perf_counter()
        dt_w = now - st["_last"]; st["_last"] = now

        combo = st["combo"]
        n_total = len(combo["sim_time"]) if combo is not None else 0
        n_blend = st["blend"]
        n_p1    = st["p1_e"] - st["p1_s"] + 1
        n_p2    = st["p2_e"] - st["p2_s"] + 1

        imgui.Text("=== Sequence Editor ===")
        imgui.Separator()

        # ---- Phase 1 trim ----
        if imgui.CollapsingHeader("Phase 1  (teleop)"):
            imgui.Text(f"Total frames in file: {n1}  (clip: {n_p1})")
            c, v = imgui.SliderInt("P1 start##p1s", st["p1_s"], 0, n1 - 1)
            if c: st["p1_s"] = min(v, st["p1_e"]); st["dirty"] = True
            c, v = imgui.SliderInt("P1 end  ##p1e", st["p1_e"], 0, n1 - 1)
            if c: st["p1_e"] = max(v, st["p1_s"]); st["dirty"] = True
            if imgui.Button("Reset P1"): st["p1_s"]=0; st["p1_e"]=n1-1; st["dirty"]=True

        imgui.Separator()

        # ---- Transition ----
        if imgui.CollapsingHeader("Transition  (spline blend)"):
            imgui.Text(f"Blend frames: {n_blend}")
            c, v = imgui.SliderInt("Blend frames", st["blend"], 1, 300)
            if c: st["blend"] = v; st["dirty"] = True

        imgui.Separator()

        # ---- Phase 2 trim ----
        if imgui.CollapsingHeader("Phase 2  (IK)"):
            imgui.Text(f"Total frames in file: {n2}  (clip: {n_p2})")
            c, v = imgui.SliderInt("P2 start##p2s", st["p2_s"], 0, n2 - 1)
            if c: st["p2_s"] = min(v, st["p2_e"]); st["dirty"] = True
            c, v = imgui.SliderInt("P2 end  ##p2e", st["p2_e"], 0, n2 - 1)
            if c: st["p2_e"] = max(v, st["p2_s"]); st["dirty"] = True
            if imgui.Button("Reset P2"): st["p2_s"]=0; st["p2_e"]=n2-1; st["dirty"]=True

        imgui.Separator()

        # ---- Rebuild if dirty ----
        if st["dirty"]:
            _rebuild()
            combo = st["combo"]
            n_total = len(combo["sim_time"])

        # ---- Timeline info ----
        imgui.Text(f"Combined: P1={n_p1}  +  Blend={n_blend}  +  P2={n_p2}"
                   f"  =  {n_total} frames")
        frac = st["frame"] / max(1, n_total - 1)
        p1_frac = n_p1 / max(1, n_total)
        bl_frac = n_blend / max(1, n_total)
        imgui.Text(f"[{'='*int(p1_frac*30):30s}|{'-'*int(bl_frac*30):30s}|{'~'*max(0,30-int(p1_frac*30)-int(bl_frac*30)):30s}]")

        imgui.Separator()

        # ---- Playback ----
        if imgui.Button("Pause" if st["play"] else "Play"):
            st["play"] = not st["play"]; st["_acc"] = 0.0
        imgui.SameLine()
        if imgui.Button("|<"):
            st["frame"] = 0; st["play"] = False
        imgui.SameLine()
        if imgui.Button(">|"):
            st["frame"] = n_total - 1; st["play"] = False

        c, v = imgui.SliderInt("Frame", st["frame"], 0, max(0, n_total - 1))
        if c:
            st["frame"] = v; st["play"] = False

        t_now = float(combo["sim_time"][st["frame"]]) if n_total else 0.0
        imgui.Text(f"t = {t_now:.3f} s")

        if st["play"] and n_total > 0:
            st["_acc"] += dt_w * fps
            adv = int(st["_acc"]); st["_acc"] -= adv
            st["frame"] = (st["frame"] + adv) % n_total

        _update_display(st["frame"])

        imgui.Separator()

        # ---- Clip config save / load ----
        if imgui.Button("Save clip config"):
            _save_clip_config(st)
        imgui.SameLine()
        if imgui.Button("Load clip config"):
            if _load_clip_config(st):
                _rebuild()
                combo = st["combo"]
                n_total = len(combo["sim_time"])

        imgui.Separator()

        # ---- Save sequence ----
        has_pin = "rigid_support_pin" in (combo or {})
        if imgui.Button("Save combined sequence"):
            _headless_save(p1, p2,
                           st["p1_s"], st["p1_e"],
                           st["p2_s"], st["p2_e"],
                           st["blend"], out_path)
            st["saved"] = True
        if st["saved"]:
            imgui.SameLine()
            imgui.TextColored((0.2, 1.0, 0.2, 1.0), f"  Saved -> {out_path.name}")
        imgui.Text(f"Pin in output: {'yes' if has_pin else 'NO (missing from source data)'}")

        imgui.Separator()

        # ---- Load & replay saved sequence ----
        if out_path.exists():
            if imgui.Button(f"Replay: {out_path.name}"):
                saved_data = dict(np.load(str(out_path)))
                st["combo"] = saved_data
                st["frame"] = 0
                st["play"] = False
                st["dirty"] = False
                n_total = len(saved_data["sim_time"])
                _update_display(0)
                print(f"[replay] loaded {out_path.name}: {n_total} frames, "
                      f"pin={'yes' if 'rigid_support_pin' in saved_data else 'no'}")

    ps.set_user_callback(gui_callback)
    ps.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GUI editor: stitch teleop + IK gear sequences"
    )
    _default_p1  = str(_GEAR  / "trajectory_gear_sharpa_objorigin.npz")
    _default_p2  = str(_HERE  / "ik_sequence.npz")
    _default_out = str(_HERE  / "full_sequence.npz")

    parser.add_argument("--phase1", default=_default_p1)
    parser.add_argument("--phase2", default=_default_p2)
    parser.add_argument("--blend-frames", type=int, default=60)
    parser.add_argument("--output", default=_default_out)
    parser.add_argument("--no-gui", action="store_true",
                        help="Skip GUI; build and save immediately")
    parser.add_argument("--replay", type=str, default=None,
                        help="Replay a previously saved .npz sequence")
    # headless-only trim args
    parser.add_argument("--p1-start", type=int, default=0)
    parser.add_argument("--p1-end",   type=int, default=-1)
    parser.add_argument("--p2-start", type=int, default=0)
    parser.add_argument("--p2-end",   type=int, default=-1)
    args = parser.parse_args()

    # Replay mode
    if args.replay:
        rp = Path(args.replay)
        if not rp.exists():
            print(f"[replay] file not found: {rp}")
            sys.exit(1)
        run_replay(rp)
        return

    print(f"[stitch] loading phase 1: {args.phase1}")
    p1 = {k: v for k, v in np.load(args.phase1).items()}
    print(f"  {p1['robot_qpos'].shape[0]} frames")

    print(f"[stitch] loading phase 2: {args.phase2}")
    p2 = {k: v for k, v in np.load(args.phase2).items()}
    print(f"  {p2['robot_qpos'].shape[0]} frames")

    out_path = Path(args.output)

    if args.no_gui:
        n1 = p1["robot_qpos"].shape[0]
        n2 = p2["robot_qpos"].shape[0]
        p1e = args.p1_end if args.p1_end >= 0 else n1 - 1
        p2e = args.p2_end if args.p2_end >= 0 else n2 - 1
        _headless_save(p1, p2, args.p1_start, p1e,
                       args.p2_start, p2e, args.blend_frames, out_path)
    else:
        run_gui(p1, p2, out_path)


if __name__ == "__main__":
    main()
