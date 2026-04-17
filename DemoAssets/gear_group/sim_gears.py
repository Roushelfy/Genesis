"""
IPC simulation of the gear mechanism using libuipc.

- gear_ring_A, gear_body, gear_ring_B: free bodies (AffineBodyShell)
- base_plate, lever_arm, axle_pin_1/2/3, lock_ball: fixed bodies
- The handle is part of gear_body; a keyboard-controlled particle is
  stitched to gear_body to drag it
- Fixed-to-fixed collisions are disabled (they already self-intersect)

Controls: W/S = +/-Y (fwd/back),  A/D = +/-X,  Q/E = +/-Z (up/down)
          Ctrl+Left-drag = mouse drag the particle

Usage:  python sim_gears.py
"""

from pathlib import Path
import json
import numpy as np
import keyboard

import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Animation
from uipc import Vector3, Transform, AngleAxis
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import (
    GeometrySlot, SimplicialComplex, SimplicialComplexIO,
    pointcloud, label_surface, ground,
)
from uipc.constitution import (
    AffineBodyShell,
    Particle,
    SoftPositionConstraint,
    SoftVertexTriangleStitch,
    ElasticModuli,
)
from uipc.gui import SceneGUI
from uipc.unit import MPa

ASSETS_DIR = Path(__file__).parent / "assets"
FIXED_DIR = ASSETS_DIR / "fixed"
JSON_PATH = ASSETS_DIR / "transforms.json"
WORKSPACE = Path(__file__).parent / "sim_output"

REPO_ROOT = Path(__file__).resolve().parents[2]  # Genesis_IPC_demo
URDF_PATH = REPO_ROOT / "DemoAssets" / "marvin_robot" / "urdf" / "marvin_pika.urdf"

FREE_PARTS = {"gear_ring_A", "gear_body", "gear_ring_B"}
FIXED_PARTS = {"base_plate", "lever_arm", "axle_pin_1", "axle_pin_2",
               "axle_pin_3", "lock_ball"}

SHELL_THICKNESS = 0.00001
ABD_KAPPA = 80.0  # MPa
SCENE_SCALE = 0.3

Logger.set_level(Logger.Level.Warn)


JOINT_ANGLES_PATH = Path(__file__).parent / "robot_joints.json"


def _get_urdf_controller():
    """Return a URDFController instance (imports lazily)."""
    import sys
    scripts_dir = REPO_ROOT / "DemoAssets" / "yoyo" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from urdf_controller import URDFController
    return URDFController(str(URDF_PATH), mesh_source="visual")


def _build_root_transform(pos, rot_deg):
    """Build a 4x4 root transform from position and Euler angles (degrees)."""
    rx, ry, rz = np.radians(rot_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = Rz @ Ry @ Rx
    tf[:3, 3] = pos
    return tf


def load_robot_reference(ctrl, root_pos, root_rot_deg) -> None:
    """Register URDF visual meshes into polyscope and apply saved state."""
    saved_joints, saved_pos, saved_rot = _load_joint_angles()
    if saved_joints:
        ctrl.set_joint_positions(saved_joints)
    root_pos[:] = saved_pos
    root_rot_deg[:] = saved_rot
    ctrl.set_root_transform(_build_root_transform(root_pos, root_rot_deg))
    _update_robot_meshes(ctrl)
    print(f"  [REF] Loaded {len(ctrl.mesh_nodes)} robot meshes from {URDF_PATH.name}")


def _update_robot_meshes(ctrl) -> None:
    """Re-register all robot meshes in polyscope with current joint state."""
    transforms = ctrl.get_mesh_transforms()
    for node in ctrl.mesh_nodes:
        tf = transforms.get(node.node_name, np.eye(4))
        verts_h = np.hstack([node.local_vertices,
                             np.ones((len(node.local_vertices), 1))])
        world_verts = (tf @ verts_h.T).T[:, :3]
        label = f"robot/{node.node_name}"
        surf = ps.register_surface_mesh(label, world_verts, node.faces)
        surf.set_color((0.4, 0.5, 0.6))
        surf.set_transparency(0.5)


def _load_joint_angles() -> tuple[dict[str, float], list[float], list[float]]:
    """Load joint angles + root transform from disk."""
    joints: dict[str, float] = {}
    root_pos = [0.0, 0.0, 0.0]
    root_rot = [0.0, 0.0, 0.0]
    if JOINT_ANGLES_PATH.exists():
        try:
            with open(JOINT_ANGLES_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "_root_pos" in data:
                root_pos = data.pop("_root_pos")
                root_rot = data.pop("_root_rot", root_rot)
                joints = data
            elif isinstance(data, dict):
                joints = data
            print(f"  [JOINTS] Loaded saved state from {JOINT_ANGLES_PATH.name}")
        except Exception:
            pass
    return joints, root_pos, root_rot


def _save_joint_angles(ctrl, root_pos, root_rot_deg) -> None:
    """Persist current joint angles + root transform to disk."""
    data = ctrl.get_joint_positions()
    data["_root_pos"] = list(root_pos)
    data["_root_rot"] = list(root_rot_deg)
    with open(JOINT_ANGLES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [JOINTS] Saved state to {JOINT_ANGLES_PATH.name}")


def resolve_obj(obj_name: str) -> Path:
    fixed = FIXED_DIR / obj_name
    if fixed.exists():
        return fixed
    return ASSETS_DIR / obj_name


def find_nearest_triangle(verts: np.ndarray, faces: np.ndarray,
                          point: np.ndarray) -> int:
    """Return index of the triangle whose centroid is closest to point."""
    centroids = verts[faces].mean(axis=1)
    dists = np.linalg.norm(centroids - point, axis=1)
    return int(np.argmin(dists))


class KeyboardIO:
    SPEED = 0.1 * SCENE_SCALE

    @staticmethod
    def movement() -> np.ndarray:
        # Z-up: W/S = +/-Y (forward/back), A/D = +/-X (left/right), Q/E = -/+Z (down/up)
        dx = (1.0 if keyboard.is_pressed("d") else 0.0) + \
             (-1.0 if keyboard.is_pressed("a") else 0.0)
        dy = (1.0 if keyboard.is_pressed("w") else 0.0) + \
             (-1.0 if keyboard.is_pressed("s") else 0.0)
        dz = (1.0 if keyboard.is_pressed("e") else 0.0) + \
             (-1.0 if keyboard.is_pressed("q") else 0.0)
        return np.array([dx, dy, dz]) * KeyboardIO.SPEED


def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    part_names = [k for k in meta if k != "_meta"]

    # ── Engine / World / Scene ──
    engine = Engine("cuda", str(WORKSPACE))
    world = World(engine)

    config = Scene.default_config()
    dt = 0.01
    config["dt"] = dt
    config["contact"]["d_hat"] = 0.001 * SCENE_SCALE
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = True
    config["newton"]["velocity_tol"] = 1
    config["newton"]["max_iter"] = 16
    config["newton"]["transrate_tol"] = 10
    config["sanity_check"]["enable"] = True
    config["gravity"] = [[0.0], [0.0], [-9.8]]  # Z-up
    scene = Scene(config)

    # ── Constitutions ──
    abd_shell = AffineBodyShell()
    particle_const = Particle()
    spc = SoftPositionConstraint()
    svts = SoftVertexTriangleStitch()

    # ── Contact groups ──
    ct = scene.contact_tabular()
    ct.default_model(0.0, 1e9)
    default_elem = ct.default_element()

    fixed_elem = ct.create("fixed_group")
    ct.insert(fixed_elem, fixed_elem, 0, 0, False)

    particle_elem = ct.create("particle_group")
    ct.insert(particle_elem, fixed_elem, 0, 0, False)
    ct.insert(particle_elem, particle_elem, 0, 0, False)
    ct.insert(particle_elem, default_elem, 0, 0, False)

    # Y-up -> Z-up: rotate +90° around X, then uniform scale
    # (x,y,z) -> s*(x, -z, y)
    s = SCENE_SCALE
    pre_tf = Transform.Identity()
    pre_tf.scale(s)
    pre_tf.rotate(AngleAxis(np.pi / 2, Vector3.UnitX()))
    io = SimplicialComplexIO(pre_tf)

    # ── Load mesh parts ──
    geo_slots = {}
    rest_geo_slots = {}
    objects = {}

    gear_body_slot = None
    gear_body_rest_slot = None
    gear_body_verts = None
    gear_body_faces = None

    for name in part_names:
        info = meta[name]
        obj_path = resolve_obj(info["obj"])

        mesh = io.read(str(obj_path))
        label_surface(mesh)

        abd_shell.apply_to(mesh, ABD_KAPPA * MPa, thickness=SHELL_THICKNESS)

        if name in FIXED_PARTS:
            is_fixed = mesh.instances().find(builtin.is_fixed)
            view(is_fixed)[:] = 1
            fixed_elem.apply_to(mesh)
        else:
            default_elem.apply_to(mesh)

        if name == "gear_body":
            gear_body_verts = np.array(
                view(mesh.positions()), copy=True
            ).reshape(-1, 3)
            gear_body_faces = np.array(
                view(mesh.triangles().topo()), copy=True
            ).reshape(-1, 3)

        obj = scene.objects().create(name)
        gs, rgs = obj.geometries().create(mesh)
        geo_slots[name] = gs
        rest_geo_slots[name] = rgs
        objects[name] = obj

        if name == "gear_body":
            gear_body_slot = gs
            gear_body_rest_slot = rgs

        tag = "FIXED" if name in FIXED_PARTS else "FREE"
        src = "fixed" if (FIXED_DIR / info["obj"]).exists() else "orig"
        print(f"  [{tag:5s}] {name:<20s}  ({src})  {info['vertices']} verts")

    # ── Particle near gear_body handle tip ──
    # Original handle extends along +Z; after Y-up->Z-up rotation +Z_orig -> -Y_new
    # So the handle tip = min-Y vertex of gear_body
    y_vals = gear_body_verts[:, 1]
    y_thresh = y_vals.min() + 0.02 * (y_vals.max() - y_vals.min())
    tip_mask = y_vals <= y_thresh
    tip_centroid = gear_body_verts[tip_mask].mean(axis=0)
    particle_pos = tip_centroid + np.array([0.0, -0.005, 0.0])

    print(f"\n  handle tip ({tip_mask.sum()} verts): {tip_centroid.tolist()}")
    print(f"  Particle at ({particle_pos[0]:.3f}, "
          f"{particle_pos[1]:.3f}, {particle_pos[2]:.3f})")

    particle_mesh = pointcloud(particle_pos.reshape(1, 3))
    label_surface(particle_mesh)
    particle_const.apply_to(particle_mesh, thickness=0.01, mass_density=300)
    spc.apply_to(particle_mesh, 10000.0)
    particle_elem.apply_to(particle_mesh)

    particle_obj = scene.objects().create("control_particle")
    p_slot, p_rest_slot = particle_obj.geometries().create(particle_mesh)

    # ── Stitch particle → gear_body (handle) ──
    if gear_body_faces is not None and len(gear_body_faces) > 0:
        best_tri = find_nearest_triangle(
            gear_body_verts, gear_body_faces, particle_pos
        )
        tri_verts = gear_body_verts[gear_body_faces[best_tri]]
        tri_center = tri_verts.mean(axis=0)
        print(f"  Stitching particle v0 -> gear_body tri {best_tri}")
        print(f"    tri verts: {tri_verts.tolist()}")
        print(f"    tri center: {tri_center.tolist()}")
        print(f"    particle:   {particle_pos.tolist()}")
        print(f"    distance:   {np.linalg.norm(particle_pos - tri_center):.6f}")

        pairs = np.array([[0, best_tri]], dtype=np.int32)
        stitch_geo = svts.create_geometry(
            (p_slot, gear_body_slot),
            (p_rest_slot, gear_body_rest_slot),
            pairs,
            ElasticModuli.youngs_poisson(1e4, 0.49),
        )
        stitch_obj = scene.objects().create("particle_gear_stitch")
        stitch_obj.geometries().create(stitch_geo)
    else:
        print("  [WARN] gear_body has no triangles — stitch skipped")

    # ── Ground ──
    ground_height = -1.0 * SCENE_SCALE
    ground_obj = scene.objects().create("ground")
    ground_normal = np.array([[0.0], [0.0], [1.0]])  # Z-up
    ground_obj.geometries().create(ground(ground_height, ground_normal))

    # ── Mouse drag state ──
    class DragState:
        active = False
        target = particle_pos.copy()
        has_target = False
        start_mouse = np.zeros(2)        # screen coords at drag start
        start_pos = np.zeros(3)          # particle world pos at drag start
        cam_right = np.zeros(3)          # camera right axis at drag start
        cam_up = np.zeros(3)             # camera up axis at drag start
        world_per_pixel = 1.0            # world units per screen pixel

    # ── Animation ──
    animator = scene.animator()

    def particle_animation(info: Animation.UpdateInfo):
        geo: SimplicialComplex = info.geo_slots()[0].geometry()
        is_constrained = geo.vertices().find(builtin.is_constrained)
        aim_pos = geo.vertices().find(builtin.aim_position)
        pos = np.array(geo.positions().view(), copy=True).reshape(-1, 3)[0]
        ap = view(aim_pos)

        V = KeyboardIO.movement()
        kbd = np.linalg.norm(V) > 0

        if kbd or DragState.active:
            view(is_constrained).fill(1)
            target = pos.copy()
            if kbd:
                target = target + V * info.dt()
            if DragState.active:
                target = target + (DragState.target - pos)
            ap[:] = target.reshape(ap.shape)
        else:
            view(is_constrained).fill(0)

    animator.insert(particle_obj, particle_animation)

    # ── Init ──
    world.init(scene)

    # ── Polyscope ──
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_window_size(1600, 1000)

    # ── Sanity check ──
    if not world.is_valid():
        print("\n  [SANITY] World is NOT valid -- visualizing error meshes")
        checker = world.sanity_checker()
        checker.report()

        sanity_dir = WORKSPACE / "sanity_check"
        if sanity_dir.exists():
            error_io = SimplicialComplexIO()
            for obj_file in sorted(sanity_dir.rglob("*.obj")):
                try:
                    err_mesh = error_io.read(str(obj_file))
                    verts = np.array(view(err_mesh.positions()), copy=True).reshape(-1, 3)
                    faces_attr = err_mesh.triangles()
                    if faces_attr is not None:
                        faces = np.array(
                            view(faces_attr.topo()), copy=True
                        ).reshape(-1, 3)
                    else:
                        faces = np.zeros((0, 3), dtype=np.int32)

                    rel = obj_file.relative_to(sanity_dir)
                    label = f"ERR/{rel}"
                    print(f"    {label}  ({verts.shape[0]} verts, {faces.shape[0]} tris)")

                    if faces.shape[0] > 0:
                        surf = ps.register_surface_mesh(label, verts, faces)
                        surf.set_color((1.0, 0.15, 0.15))
                        surf.set_edge_width(2)
                        surf.set_transparency(0.7)
                    elif verts.shape[0] > 0:
                        pc = ps.register_point_cloud(label, verts)
                        pc.set_color((1.0, 0.15, 0.15))
                        pc.set_point_radius(0.005)
                except Exception as exc:
                    print(f"    [WARN] could not load {obj_file}: {exc}")

        # also register the scene meshes so user can see context
        for name in part_names:
            info = meta[name]
            obj_path = resolve_obj(info["obj"])
            try:
                raw = SimplicialComplexIO().read(str(obj_path))
                v = np.array(view(raw.positions()), copy=True).reshape(-1, 3)
                f = np.array(view(raw.triangles().topo()), copy=True).reshape(-1, 3)
                surf = ps.register_surface_mesh(name, v, f)
                surf.set_color((0.7, 0.7, 0.7))
                surf.set_transparency(0.3)
            except Exception:
                pass

        imgui_msg = "[SANITY FAILED] See red meshes for close/intersected regions"

        def on_update_err():
            imgui.TextColored((1.0, 0.2, 0.2, 1.0), imgui_msg)

        ps.set_user_callback(on_update_err)
        ps.show()
        return

    # ── Normal simulation mode ──
    print("\n  [SANITY] World is valid -- starting simulation")

    sgui = SceneGUI(scene, "split")
    sgui.register()
    sgui.set_edge_width(1)

    # ── Load robot reference (visual only, not simulated) ──
    robot_ctrl = None
    root_pos = [0.0, 0.0, 0.0]
    root_rot_deg = [0.0, 0.0, 0.0]
    if URDF_PATH.exists():
        robot_ctrl = _get_urdf_controller()
        load_robot_reference(robot_ctrl, root_pos, root_rot_deg)

    run = False
    joint_panel_open = [True]

    def on_update():
        nonlocal run

        io = imgui.GetIO()
        ctrl_held = io.KeyCtrl

        # ── Mouse drag logic (Ctrl + left-click) ──
        if ctrl_held and imgui.IsMouseClicked(0):
            view_mat = np.array(ps.get_camera_view_matrix())
            R = view_mat[:3, :3]
            t_vec = view_mat[:3, 3]
            DragState.cam_right = R[0, :]
            DragState.cam_up = R[1, :]
            cam_pos = -R.T @ t_vec

            p_geo: SimplicialComplex = p_slot.geometry()
            cur_pos = np.array(p_geo.positions().view(), copy=True).reshape(-1, 3)[0]
            DragState.start_pos = cur_pos.copy()
            DragState.start_mouse = np.array(io.MousePos)

            dist = np.linalg.norm(cur_pos - cam_pos)
            fov_rad = np.radians(ps.get_vertical_fov_degrees())
            win_h = imgui.GetIO().DisplaySize[1]
            DragState.world_per_pixel = 2.0 * dist * np.tan(fov_rad / 2.0) / win_h

            DragState.active = True
            DragState.target = cur_pos.copy()
            ps.set_do_default_mouse_interaction(False)

        if DragState.active:
            if imgui.IsMouseDown(0):
                cur_mouse = np.array(io.MousePos)
                delta_px = cur_mouse - DragState.start_mouse
                world_delta = (DragState.cam_right * delta_px[0]
                               - DragState.cam_up * delta_px[1]
                               ) * DragState.world_per_pixel
                DragState.target = DragState.start_pos + world_delta
            else:
                DragState.active = False
                ps.set_do_default_mouse_interaction(True)

        # ── GUI ──
        if imgui.Button("Run / Stop"):
            run = not run
        imgui.Separator()
        imgui.TextUnformatted("Keyboard:  W/S = +/-Y   A/D = +/-X   Q/E = +/-Z")
        imgui.TextUnformatted("Mouse:     Ctrl + Left-drag")
        imgui.TextUnformatted(f"Frame: {world.frame()}   "
                              f"Time: {world.frame()*dt:.3f}s")

        # ── Robot joint panel ──
        if robot_ctrl is not None:
            imgui.Separator()
            _, joint_panel_open[0] = imgui.CollapsingHeader(
                "Robot Joints", joint_panel_open[0])
            if joint_panel_open[0]:
                root_changed = False
                for axis, label in enumerate(["Root X", "Root Y", "Root Z"]):
                    changed, new_val = imgui.SliderFloat(
                        label, root_pos[axis], -1.0, 1.0)
                    if changed:
                        root_pos[axis] = new_val
                        root_changed = True
                for axis, label in enumerate(["Rot X", "Rot Y", "Rot Z"]):
                    changed, new_val = imgui.SliderFloat(
                        label, root_rot_deg[axis], -180.0, 180.0)
                    if changed:
                        root_rot_deg[axis] = new_val
                        root_changed = True
                if root_changed:
                    robot_ctrl.set_root_transform(
                        _build_root_transform(root_pos, root_rot_deg))
                    _update_robot_meshes(robot_ctrl)

                imgui.Separator()
                joints_changed = False
                limits = robot_ctrl.joint_limits
                positions = robot_ctrl.get_joint_positions()
                for name in robot_ctrl.joint_names:
                    lo, hi = limits.get(name, (-np.pi, np.pi))
                    changed, new_val = imgui.SliderFloat(
                        name, positions[name], lo, hi)
                    if changed:
                        positions[name] = new_val
                        joints_changed = True
                if joints_changed:
                    robot_ctrl.set_joint_positions(positions)
                    _update_robot_meshes(robot_ctrl)
                if imgui.Button("Reset Joints"):
                    robot_ctrl.set_joint_positions(
                        {n: 0.0 for n in robot_ctrl.joint_names})
                    _update_robot_meshes(robot_ctrl)
                imgui.SameLine()
                if imgui.Button("Save Joints"):
                    _save_joint_angles(robot_ctrl, root_pos, root_rot_deg)

        if run:
            world.advance()
            world.retrieve()
            sgui.update()

    ps.set_user_callback(on_update)
    ps.show()

    # Save joint angles on exit
    if robot_ctrl is not None:
        _save_joint_angles(robot_ctrl, root_pos, root_rot_deg)


if __name__ == "__main__":
    main()
