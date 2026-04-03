"""Yoyo scenario: FK/IK inspector + UIPC simulation with ball + string + bearing.

Thin entry script that configures URDFGuiApp with the yoyo scene
(previously loaded via ``load_user_scene.py``).

Modes:
  --sim       Launch UIPC physics simulation directly with SceneGUI.
  (default)   FK/IK inspector with fully initialized UIPC scene.

Controls (inspector mode):
  - Joint Angles panel: sliders for each actuated joint
  - End Effectors panel: live position + orientation readout
  - IK Control panel: dual-gripper keyboard control
      Left  gripper: W/S = X,  A/D = Y,  Q/E = Z
      Right gripper: I/K = X,  J/L = Y,  U/O = Z
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from urdf_gui import SceneState, URDFGuiApp, closest_point_on_triangle

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]

_BALL_OBJ = _REPO_ROOT / "DemoAssets" / "yoyo" / "v3" / "yoyo-ball.obj"
_STRING_OBJ = _REPO_ROOT / "DemoAssets" / "yoyo" / "v3" / "yoyo_string_6153.obj"
_BEARING_OUTER_OBJ = _REPO_ROOT / "DemoAssets" / "yoyo" / "v1" / "bearing_outer.obj"
_BEARING_SPHERES_OBJ = _REPO_ROOT / "DemoAssets" / "yoyo" / "v1" / "bearing_spheres.obj"

_URDF_PATH = _REPO_ROOT / "DemoAssets" / "marvin_robot" / "urdf" / "marvin_pika.urdf"
_SIM_LINK_PATTERNS = ["*Link8*", "*Link9*"]
_STC_STRENGTH = np.array([120.0, 120.0], dtype=np.float64)
_STITCH_BINDING_PATTERN = "*Link9_R*"


# ---------------------------------------------------------------------------
# Yoyo scene loading (inlined from load_user_scene.py)
# ---------------------------------------------------------------------------


def load_yoyo_scene(scene, world, ss: SceneState) -> None:
    """Add the full yoyo (ball + string + bearings) to *scene*.

    Populates *ss* (SceneState) with references to all created objects.
    Also overrides scene config to match the yoyo simulation parameters.
    """
    from uipc import Animation, builtin, view
    from uipc.constitution import (
        AffineBodyConstitution,
        HookeanSpring,
        KirchhoffRodBending,
        RotatingMotor,
    )
    from uipc.geometry import (
        SimplicialComplexIO,
        apply_region,
        label_region,
        label_surface,
        mesh_partition,
    )
    from uipc.unit import GPa, MPa

    ss.clear()
    ss.fem_objects = {"yoyo_string"}
    io = SimplicialComplexIO()

    # ---- Simulation parameters ----
    cfg = scene.config()
    view(cfg.find("dt"))[0] = 0.001
    view(cfg.find("contact/enable"))[0] = 1
    view(cfg.find("contact/friction/enable"))[0] = 1
    view(cfg.find("contact/d_hat"))[0] = 0.0001
    view(cfg.find("newton/velocity_tol"))[0] = 0.1
    view(cfg.find("newton/transrate_tol"))[0] = 10
    view(cfg.find("sanity_check/enable"))[0] = 1

    tabular = scene.contact_tabular()
    tabular.default_model(0.3, 1e9, True)

    # ---- Contacts ----
    ball_contact = tabular.create("ball")
    string_contact = tabular.create("string")
    bearing_contact = tabular.create("bearing")
    scene.contact_tabular().insert(ball_contact, string_contact, 0.7, 800.0 * MPa, enable=True)
    scene.contact_tabular().insert(string_contact, string_contact, 0.2, 800.0 * MPa, enable=True)
    scene.contact_tabular().insert(bearing_contact, bearing_contact, 0.01, 800.0 * MPa, enable=True)
    scene.contact_tabular().insert(bearing_contact, ball_contact, 0.05, 800.0 * MPa, enable=True)
    scene.contact_tabular().insert(bearing_contact, string_contact, 0.8, 800.0 * MPa, enable=True)

    robot_contact = tabular.default_element()
    tabular.insert(ball_contact, robot_contact, 0.5, 800.0 * MPa, True)
    tabular.insert(string_contact, robot_contact, 0.7, 800.0 * MPa, True)
    tabular.insert(bearing_contact, robot_contact, 0.3, 800.0 * MPa, True)

    # ---- Yoyo Ball (free, spinning) ----
    ball_mesh = io.read(str(_BALL_OBJ))
    label_surface(ball_mesh)
    AffineBodyConstitution().apply_to(ball_mesh, 200.0 * MPa)
    ball_contact.apply_to(ball_mesh)
    view(ball_mesh.instances().find(builtin.is_dynamic))[0] = 1

    motor = RotatingMotor()
    motor.apply_to(
        ball_mesh,
        strength=1,
        motor_axis=np.array([[0.0], [1.0], [0.0]], dtype=np.float64),
        motor_rot_vel=-2.0 * np.pi * 10,
    )

    ball_obj = scene.objects().create("yoyo_ball")
    ball_gs, _ = ball_obj.geometries().create(ball_mesh)
    ss.objects["yoyo_ball"] = ball_obj
    ss.geo_slots["yoyo_ball"] = ball_gs
    ss.obj_sources["yoyo_ball"] = _BALL_OBJ
    ss.original_transforms["yoyo_ball"] = np.array(view(ball_mesh.transforms())[0], copy=True)

    def rotate_yoyo_ball(info: Animation.UpdateInfo) -> None:
        geo = info.geo_slots()[0].geometry()
        view(geo.instances().find(builtin.is_constrained))[0] = 1
        RotatingMotor.animate(geo, info.dt())

    scene.animator().insert(ball_obj, rotate_yoyo_ball)

    # ---- String (top vertex fixed) ----
    string_mesh = io.read(str(_STRING_OBJ))
    label_surface(string_mesh)
    HookeanSpring().apply_to(string_mesh, 1.0 * GPa, thickness=0.00038, mass_density=100.0)
    KirchhoffRodBending().apply_to(string_mesh, 1.0e4)
    string_contact.apply_to(string_mesh)
    mesh_partition(string_mesh, 16)

    string_obj = scene.objects().create("yoyo_string")
    string_gs, string_rest_gs = string_obj.geometries().create(string_mesh)
    ss.objects["yoyo_string"] = string_obj
    ss.geo_slots["yoyo_string"] = string_gs
    ss.rest_geo_slots["yoyo_string"] = string_rest_gs
    ss.obj_sources["yoyo_string"] = _STRING_OBJ
    ss.original_transforms["yoyo_string"] = np.array(view(string_mesh.transforms())[0], copy=True)
    ss.original_positions["yoyo_string"] = np.array(view(string_gs.geometry().positions()), copy=True).reshape(-1, 3)

    # ---- Bearing Outer ----
    bearing_outer_mesh = io.read(str(_BEARING_OUTER_OBJ))
    label_surface(bearing_outer_mesh)
    AffineBodyConstitution().apply_to(bearing_outer_mesh, 200.0 * MPa)
    bearing_contact.apply_to(bearing_outer_mesh)
    bearing_outer_obj = scene.objects().create("bearing_outer")
    bo_gs, _ = bearing_outer_obj.geometries().create(bearing_outer_mesh)
    ss.objects["bearing_outer"] = bearing_outer_obj
    ss.geo_slots["bearing_outer"] = bo_gs
    ss.obj_sources["bearing_outer"] = _BEARING_OUTER_OBJ
    ss.original_transforms["bearing_outer"] = np.array(view(bearing_outer_mesh.transforms())[0], copy=True)

    # ---- Bearing Spheres (split by region) ----
    bearing_spheres_mesh = io.read(str(_BEARING_SPHERES_OBJ))
    label_surface(bearing_spheres_mesh)
    label_region(bearing_spheres_mesh)
    sphere_regions = apply_region(bearing_spheres_mesh)
    for i, sphere_mesh in enumerate(sphere_regions):
        label_surface(sphere_mesh)
        AffineBodyConstitution().apply_to(sphere_mesh, 200.0 * MPa)
        bearing_contact.apply_to(sphere_mesh)
        sp_obj = scene.objects().create(f"bearing_sphere_{i}")
        sp_gs, _ = sp_obj.geometries().create(sphere_mesh)
        ss.objects[f"bearing_sphere_{i}"] = sp_obj
        ss.geo_slots[f"bearing_sphere_{i}"] = sp_gs
        ss.original_transforms[f"bearing_sphere_{i}"] = np.array(view(sphere_mesh.transforms())[0], copy=True)
        split_obj_path = _REPO_ROOT / "DemoAssets" / "yoyo" / "v3" / f"bearing_sphere_{i}.obj"
        if not split_obj_path.exists():
            io.write(str(split_obj_path), sphere_mesh)
        ss.obj_sources[f"bearing_sphere_{i}"] = split_obj_path

    print(f"[user_scene] Yoyo loaded: ball + string + bearing_outer + {len(sphere_regions)} bearing spheres")


# ---------------------------------------------------------------------------
# Stitch setup (inlined from load_user_scene.py)
# ---------------------------------------------------------------------------


def setup_yoyo_stitch(scene, controller, ss: SceneState) -> None:
    """Stitch yoyo string vertex 0 to the nearest triangle on the right gripper."""
    from uipc import view
    from uipc.constitution import ElasticModuli, SoftVertexTriangleStitch
    from uipc.geometry import apply_transform

    binding_names = [(b.node_name, b.parent_link, b.object_name) for b in controller.bindings]
    print(f"[stitch-debug] Available bindings: {binding_names}")

    binding = controller.find_binding(_STITCH_BINDING_PATTERN)
    if binding is None:
        print(f"[stitch-debug] No binding matched '{_STITCH_BINDING_PATTERN}'")
        return
    print(f"[stitch-debug] Matched binding: {binding.node_name}")

    string_gs = ss.geo_slots.get("yoyo_string")
    string_rest_gs = ss.rest_geo_slots.get("yoyo_string")
    if string_gs is None or string_rest_gs is None:
        print("[stitch] yoyo_string not loaded, skipping")
        return

    gripper_geo_slot = binding.geo_slot
    gripper_rest_geo_slot = binding.rest_geo_slot

    string_geo = string_gs.geometry()
    gripper_geo = gripper_geo_slot.geometry()
    gripper_rest_geo = gripper_rest_geo_slot.geometry()

    gripper_tf = np.array(view(gripper_geo.transforms()), copy=False).reshape(-1, 4, 4)[0]
    view(gripper_rest_geo.transforms())[0] = gripper_tf

    string_pos = np.array(view(string_geo.positions()), copy=False).reshape(-1, 3)
    gripper_world = apply_transform(gripper_geo)[0]
    gripper_world_pos = np.array(view(gripper_world.positions()), copy=False).reshape(-1, 3)
    gripper_world_tris = np.array(view(gripper_world.triangles().topo()), copy=False).reshape(-1, 3)

    v0 = string_pos[0]
    best_tri = 0
    best_dist = float("inf")
    for ti in range(gripper_world_tris.shape[0]):
        tri_verts = gripper_world_pos[gripper_world_tris[ti]]
        cp = closest_point_on_triangle(v0, tri_verts[0], tri_verts[1], tri_verts[2])
        d = float(np.linalg.norm(v0 - cp))
        if d < best_dist:
            best_dist = d
            best_tri = ti

    pairs = np.array([[0, best_tri]], dtype=np.int32)
    svts = SoftVertexTriangleStitch()
    stitch_geo = svts.create_geometry(
        (string_gs, gripper_geo_slot),
        (string_rest_gs, gripper_rest_geo_slot),
        pairs,
        ElasticModuli.youngs_poisson(1.0e5, 0.49),
    )
    stitch_obj = scene.objects().create("string_gripper_stitch")
    stitch_obj.geometries().create(stitch_geo)

    ss.stitch_vis["string_gs"] = string_gs
    ss.stitch_vis["gripper_gs"] = gripper_geo_slot
    ss.stitch_vis["pairs"] = pairs
    ss.stitch_vis["gripper_tris"] = np.array(view(gripper_geo.triangles().topo()), copy=True).reshape(-1, 3)

    print(f"[stitch] string vertex 0 -> gripper triangle {best_tri} (dist={best_dist:.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def create_yoyo_app(entry_file: str = __file__) -> URDFGuiApp:
    """Create a URDFGuiApp configured for the yoyo scenario."""
    return URDFGuiApp(
        urdf_path=_URDF_PATH,
        sim_link_patterns=_SIM_LINK_PATTERNS,
        stc_strength=_STC_STRENGTH,
        load_scene_fn=load_yoyo_scene,
        setup_stitch_fn=setup_yoyo_stitch,
        record_file=_SCRIPT_DIR / "motion_keyframes.json",
        joint_file=_SCRIPT_DIR / "joint_angles.json",
        config_file=_SCRIPT_DIR / "urdf_gui_config.json",
        seq_dir=_REPO_ROOT / "DemoAssets" / "yoyo" / "v3" / "seq",
        entry_file=entry_file,
    )


if __name__ == "__main__":
    create_yoyo_app(__file__).main_cli()
