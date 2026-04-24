"""Polyscope replay of planetary gear teleop trajectory (MARVIN_SHARPA).

Loads ``trajectory_gear_sharpa.npz`` (628 frames, 60 Hz, ~10.5 s) and shows:
  - Ring gear (fixed housing)
  - Sun gear with handle  (driven by teleop)
  - Carrier
  - Three planet gears
  - MARVIN_SHARPA robot (58 DOF)  driven by recorded joint angles

The trajectory already starts with all gears assembled and running —
no planet installation phase is shown.

Coordinate / unit convention
-----------------------------
  OBJ meshes are in millimetres.  The Genesis simulation that captured this
  trajectory used ``scale = 0.0012`` (mm → m).  Rigid-body poses in the npz
  are absolute world transforms stored as ``[px, py, pz, qx, qy, qz, qw]``
  (scalar-last quaternion).

Usage
-----
    python view_teleop.py [--traj PATH] [--loop]

Controls (ImGui panel)
----------------------
    Run / Pause   — toggle playback
    Speed slider  — 0.1 × … 4 ×
    Frame slider  — scrub manually
    Loop          — toggle loop-at-end
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polyscope as ps
from polyscope import imgui

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DEMO_ASSETS = _HERE.parent            # Genesis_IPC_demo/DemoAssets
_GEAR = _DEMO_ASSETS / "planetary_gear"
_GEAR_ASSETS = _GEAR / "assets"
_SHARPA_URDF = _DEMO_ASSETS / "marvin_sharpa_description" / "marvin_sharpa.urdf"
_DEFAULT_TRAJ = _GEAR / "trajectory_gear_sharpa.npz"
_YOYO_SCRIPTS = _DEMO_ASSETS / "yoyo" / "scripts"

# ---------------------------------------------------------------------------
# Simulation constants  (must match the Genesis sim that captured the npz)
# ---------------------------------------------------------------------------

MESH_SCALE = 0.0012    # OBJ files are in mm; Genesis sim used 0.0012
ROBOT_BASE_Z = 1.08    # fixed robot base height in world space (metres)

# ---------------------------------------------------------------------------
# Gear part definitions
# ---------------------------------------------------------------------------

_GEAR_FILES: dict[str, str] = {
    "ring_gear":     "ring_gear.obj",
    "sun_gear":      "sun_gear_handle.obj",
    "carrier":       "carrier.obj",
    "planet_gear_0": "planet_gear.obj",
    "planet_gear_1": "planet_gear.obj",
    "planet_gear_2": "planet_gear.obj",
}

_GEAR_COLORS: dict[str, tuple] = {
    "ring_gear":     (0.60, 0.60, 0.60),
    "sun_gear":      (0.45, 0.45, 0.50),
    "carrier":       (0.55, 0.50, 0.45),
    "planet_gear_0": (0.50, 0.55, 0.60),
    "planet_gear_1": (0.50, 0.55, 0.60),
    "planet_gear_2": (0.50, 0.55, 0.60),
}

# ---------------------------------------------------------------------------
# Quaternion / transform utilities
# ---------------------------------------------------------------------------


def quat_wxyz_to_mat3(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Genesis scalar-first [qw, qx, qy, qz] quaternion → 3×3 rotation matrix."""
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def apply_world_transform(verts_local: np.ndarray, pos_quat: np.ndarray) -> np.ndarray:
    """Transform local mesh vertices to world space using a Genesis 7-D pose.

    Parameters
    ----------
    verts_local : (N, 3) float64  — mesh vertices in local/rest space
    pos_quat    : (7,)  float32   — Genesis format [px, py, pz, qw, qx, qy, qz]

    Returns
    -------
    (N, 3) float64 world-space vertices
    """
    px, py, pz       = pos_quat[:3].astype(np.float64)
    qw, qx, qy, qz   = pos_quat[3:].astype(np.float64)
    R = quat_wxyz_to_mat3(qw, qx, qy, qz)
    return (R @ verts_local.T).T + np.array([px, py, pz])


# ---------------------------------------------------------------------------
# Minimal OBJ loader
# ---------------------------------------------------------------------------


def load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and triangulated faces from a Wavefront OBJ file."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                idxs = [int(t.split("/")[0]) - 1 for t in tokens]
                if len(idxs) == 3:
                    faces.append(idxs)
                elif len(idxs) == 4:
                    # Fan-triangulate quads
                    faces.append([idxs[0], idxs[1], idxs[2]])
                    faces.append([idxs[0], idxs[2], idxs[3]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# URDFController helpers
# ---------------------------------------------------------------------------


def _load_urdf_controller(urdf_path: str):
    """Load URDFController from the yoyo/scripts helper."""
    if str(_YOYO_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_YOYO_SCRIPTS))
    from urdf_controller import URDFController  # type: ignore

    ctrl = URDFController(urdf_path, mesh_source="visual")
    return ctrl


def _build_qpos_joint_order(ctrl) -> list[str]:
    """Return joint names in the Genesis qpos column order for MARVIN_SHARPA.

    Genesis serialises marvin-family robots with arm joints interleaved
    (Joint1_R, Joint1_L, Joint2_R, Joint2_L, …, Joint7_R, Joint7_L) followed
    by all remaining (finger/wrist) joints in URDF declaration order.
    """
    all_joints = set(ctrl.joint_names)

    ordered: list[str] = []
    # Interleaved arm joints
    for k in range(1, 8):
        for side in ("R", "L"):
            jname = f"Joint{k}_{side}"
            if jname in all_joints:
                ordered.append(jname)

    # Remaining joints (fingers, etc.) in original URDF order
    arm_set = set(ordered)
    for jname in ctrl.joint_names:
        if jname not in arm_set:
            ordered.append(jname)

    return ordered


def _update_robot(ctrl, ps_robot: dict[str, "ps.SurfaceMesh"]) -> None:
    """Push FK results from URDFController into the polyscope surface meshes."""
    transforms = ctrl.get_mesh_transforms()
    for node in ctrl.mesh_nodes:
        label = f"robot/{node.node_name}"
        if label not in ps_robot:
            continue
        tf = transforms.get(node.node_name, np.eye(4))
        v_h = np.hstack(
            [node.local_vertices, np.ones((len(node.local_vertices), 1))]
        )
        world_v = (tf @ v_h.T).T[:, :3]
        ps_robot[label].update_vertex_positions(world_v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polyscope viewer: planetary gear teleop trajectory"
    )
    parser.add_argument(
        "--traj",
        type=str,
        default=str(_DEFAULT_TRAJ),
        help="Path to trajectory .npz (default: trajectory_gear_sharpa.npz)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback when the last frame is reached",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load trajectory
    # ------------------------------------------------------------------
    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    print(f"Loading trajectory: {traj_path.name}")
    traj = np.load(str(traj_path))
    sim_time: np.ndarray = traj["sim_time"]       # (N,)
    robot_qpos: np.ndarray = traj["robot_qpos"]   # (N, 58)

    n_frames = len(sim_time)
    dt = float(sim_time[1] - sim_time[0]) if n_frames > 1 else 1 / 60.0
    fps = min(max(1, int(round(1.0 / dt))), 120)
    duration = float(sim_time[-1] - sim_time[0])

    rigid_poses: dict[str, np.ndarray] = {
        "ring_gear":     traj["rigid_ring_gear"],
        "sun_gear":      traj["rigid_sun_gear"],
        "carrier":       traj["rigid_carrier"],
        "planet_gear_0": traj["rigid_planet_gear_0"],
        "planet_gear_1": traj["rigid_planet_gear_1"],
        "planet_gear_2": traj["rigid_planet_gear_2"],
    }

    print(
        f"  {n_frames} frames  {fps} fps  {duration:.2f} s  "
        f"robot_qpos shape = {robot_qpos.shape}"
    )

    # ------------------------------------------------------------------
    # Load gear OBJ meshes  (mm → m via MESH_SCALE)
    # ------------------------------------------------------------------
    gear_verts_local: dict[str, np.ndarray] = {}
    gear_faces: dict[str, np.ndarray] = {}

    for name, obj_file in _GEAR_FILES.items():
        obj_path = _GEAR_ASSETS / obj_file
        if not obj_path.exists():
            print(f"  [WARN] {obj_path} not found — skipping {name}")
            continue
        V, F = load_obj(str(obj_path))
        gear_verts_local[name] = V * MESH_SCALE  # mm → m
        gear_faces[name] = F
        print(f"  {name:18s}  {len(V):5d} verts  {len(F):6d} faces")

    # ------------------------------------------------------------------
    # Load robot (MARVIN_SHARPA) via URDFController
    # ------------------------------------------------------------------
    robot_ctrl = None
    joint_order: list[str] = []

    if _SHARPA_URDF.exists():
        print(f"\nLoading robot URDF: {_SHARPA_URDF.name}")
        robot_ctrl = _load_urdf_controller(str(_SHARPA_URDF))

        # Fixed base at (0, 0, ROBOT_BASE_Z)
        root_tf = np.eye(4, dtype=np.float64)
        root_tf[2, 3] = ROBOT_BASE_Z
        robot_ctrl.set_root_transform(root_tf)

        joint_order = _build_qpos_joint_order(robot_ctrl)
        n_urdf = len(joint_order)
        n_npz = robot_qpos.shape[1]
        status = "OK" if n_urdf == n_npz else f"MISMATCH (using first {min(n_urdf, n_npz)})"
        print(f"  URDF joints: {n_urdf}   npz qpos cols: {n_npz}   [{status}]")
        print(f"  First 14 joints: {joint_order[:14]}")
    else:
        print(f"\n[WARN] URDF not found: {_SHARPA_URDF}")

    # ------------------------------------------------------------------
    # Initialise Polyscope
    # ------------------------------------------------------------------
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_window_size(1600, 1000)
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height(0.0)
    ps.set_automatically_compute_scene_extents(False)

    # ------------------------------------------------------------------
    # Register gear meshes at frame 0
    # ------------------------------------------------------------------
    ps_gears: dict[str, ps.SurfaceMesh] = {}

    for name in gear_verts_local:
        V0 = apply_world_transform(gear_verts_local[name], rigid_poses[name][0])
        mesh = ps.register_surface_mesh(f"gear/{name}", V0, gear_faces[name])
        mesh.set_color(_GEAR_COLORS.get(name, (0.7, 0.7, 0.7)))
        mesh.set_smooth_shade(True)
        mesh.set_edge_width(0.5)
        ps_gears[name] = mesh

    # ------------------------------------------------------------------
    # Register robot meshes at frame 0
    # ------------------------------------------------------------------
    ps_robot: dict[str, ps.SurfaceMesh] = {}

    if robot_ctrl is not None:
        # Apply frame-0 joint angles
        q0 = robot_qpos[0]
        n_use = min(len(joint_order), len(q0))
        robot_ctrl.set_joint_positions(
            {joint_order[i]: float(q0[i]) for i in range(n_use)}
        )

        transforms = robot_ctrl.get_mesh_transforms()
        n_nodes = 0
        for node in robot_ctrl.mesh_nodes:
            label = f"robot/{node.node_name}"
            tf = transforms.get(node.node_name, np.eye(4))
            v_h = np.hstack(
                [node.local_vertices, np.ones((len(node.local_vertices), 1))]
            )
            V_w = (tf @ v_h.T).T[:, :3]
            m = ps.register_surface_mesh(label, V_w, node.faces)
            m.set_color((0.40, 0.52, 0.65))
            m.set_transparency(0.25)
            m.set_smooth_shade(True)
            ps_robot[label] = m
            n_nodes += 1
        print(f"  Registered {n_nodes} robot mesh nodes")

    # ------------------------------------------------------------------
    # Playback state
    # ------------------------------------------------------------------
    frame = [0]
    run = [False]
    speed = [1.0]
    loop = [args.loop]
    accum = [0.0]
    last_wall = [time.perf_counter()]

    def update_to_frame(f: int) -> None:
        f = max(0, min(f, n_frames - 1))
        frame[0] = f

        # Gear meshes
        for name, ps_mesh in ps_gears.items():
            V_w = apply_world_transform(gear_verts_local[name], rigid_poses[name][f])
            ps_mesh.update_vertex_positions(V_w)

        # Robot
        if robot_ctrl is not None:
            q = robot_qpos[f]
            n_use = min(len(joint_order), len(q))
            robot_ctrl.set_joint_positions(
                {joint_order[i]: float(q[i]) for i in range(n_use)}
            )
            _update_robot(robot_ctrl, ps_robot)

    # Ensure frame 0 is displayed
    update_to_frame(0)

    # ------------------------------------------------------------------
    # ImGui callback
    # ------------------------------------------------------------------
    def gui_callback() -> None:
        now = time.perf_counter()
        wall_dt = now - last_wall[0]
        last_wall[0] = now

        # --- Controls row 1 ---
        if imgui.Button("Pause" if run[0] else "Run "):
            run[0] = not run[0]
        imgui.SameLine()
        changed, loop[0] = imgui.Checkbox("Loop", loop[0])
        imgui.SameLine()
        if imgui.Button("Reset"):
            run[0] = False
            update_to_frame(0)

        # --- Speed ---
        _, speed[0] = imgui.SliderFloat("Speed", speed[0], 0.1, 4.0)

        # --- Info ---
        f = frame[0]
        t_now = float(sim_time[f])
        imgui.Text(
            f"Frame {f + 1:4d} / {n_frames}    "
            f"t = {t_now:.3f} s    "
            f"(duration {duration:.2f} s,  {fps} Hz)"
        )

        # --- Frame scrubber ---
        c, new_f = imgui.SliderInt("Frame", f, 0, n_frames - 1)
        if c:
            run[0] = False
            update_to_frame(new_f)
            return  # skip advance logic this tick

        # --- Advance playback ---
        if run[0]:
            accum[0] += wall_dt * speed[0] * fps
            steps = int(accum[0])
            accum[0] -= steps
            new_frame = f + steps
            if new_frame >= n_frames:
                if loop[0]:
                    new_frame = new_frame % n_frames
                else:
                    new_frame = n_frames - 1
                    run[0] = False
            if new_frame != f:
                update_to_frame(new_frame)

        # --- Gear pose info (collapsible) ---
        imgui.Separator()
        opened = imgui.CollapsingHeader("Gear poses (current frame)")
        if opened:
            for name in ("sun_gear", "carrier", "planet_gear_0"):
                p = rigid_poses[name][frame[0]]
                imgui.Text(
                    f"  {name:18s}  pos = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                )

    ps.set_user_callback(gui_callback)
    print("\nOpening Polyscope window …  (close the window to exit)")
    ps.show()


if __name__ == "__main__":
    main()
