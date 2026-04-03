"""
Stage-3 yoyo throw-act-revolve:
- Load yoyo ball + string + bearing outer + bearing spheres (no tuber).
- Yoyo ball is free (not fixed) with a spinning motor.
- String start vertex (vertex 0) is fixed.
- bearing_spheres.obj is split by regions into independent ABD bodies.
"""

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Animation, Engine, Logger, Scene, SceneIO, Timer, World, builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    HookeanSpring,
    KirchhoffRodBending,
    RotatingMotor,
    SoftPositionConstraint,
)
from uipc.geometry import SimplicialComplexIO, apply_region, label_region, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

from asset_dir import AssetDir

Logger.set_level(Logger.Level.Info)
Timer.enable_all()

# ---- Tunables ----
DT = 0.001
MOTOR_ROT_VEL = 2.0 * np.pi * 10.0

workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)
_YOYO_DIR = folder.parents[2] / "DemoAssets" / "yoyo"

ball_obj = _YOYO_DIR / "v1" / "yoyo-ball.obj"
string_obj = _YOYO_DIR / "v1" / "yoyo_string_1007_with_bearing.obj"
bearing_outer_obj = _YOYO_DIR / "v1" / "bearing_outer.obj"
bearing_spheres_obj = _YOYO_DIR / "v1" / "bearing_spheres.obj"

engine = Engine("cuda", str(workspace))
world = World(engine)

config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.0001
config["newton"]["velocity_tol"] = 0.05
config["newton"]["transrate_tol"] = 10
config["collision_detection"]["method"] = "info_stackless_bvh"
scene = Scene(config)

scene.contact_tabular().default_model(0.3, 800.0 * MPa)
ball_contact = scene.contact_tabular().create("ball")
string_contact = scene.contact_tabular().create("string")
bearing_contact = scene.contact_tabular().create("bearing")
scene.contact_tabular().insert(ball_contact, string_contact, 0.5, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(string_contact, string_contact, 0.10, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(bearing_contact, bearing_contact, 0.01, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(bearing_contact, ball_contact, 0.01, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(bearing_contact, string_contact, 0.5, 800.0 * MPa, enable=True)

io = SimplicialComplexIO()

# ---- Yoyo ball: free + spinning motor ----
ball_mesh = io.read(str(ball_obj))
label_surface(ball_mesh)
AffineBodyConstitution().apply_to(ball_mesh, 200.0 * MPa)
ball_contact.apply_to(ball_mesh)
view(ball_mesh.instances().find(builtin.is_dynamic))[0] = 1

motor = RotatingMotor()
motor.apply_to(
    ball_mesh,
    strength=1,
    motor_axis=np.array([[0.0], [0.0], [1.0]], dtype=np.float64),
    motor_rot_vel=MOTOR_ROT_VEL,
)

ball_obj_node = scene.objects().create("yoyo_ball")
ball_obj_node.geometries().create(ball_mesh)


def rotate_yoyo_ball(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    RotatingMotor.animate(geo, info.dt())


scene.animator().insert(ball_obj_node, rotate_yoyo_ball)

# ---- String: vertex 0 fixed ----
string_mesh = io.read(str(string_obj))
label_surface(string_mesh)
HookeanSpring().apply_to(string_mesh, 1.0 * GPa, thickness=0.0003, mass_density=100.0)
KirchhoffRodBending().apply_to(string_mesh, 1.0e3)
SoftPositionConstraint().apply_to(string_mesh, 100.0)
string_contact.apply_to(string_mesh)
mesh_partition(string_mesh, 16)

is_fixed = string_mesh.vertices().find(builtin.is_fixed)
view(is_fixed)[0] = 1

string_obj_node = scene.objects().create("yoyo_string")
string_obj_node.geometries().create(string_mesh)

# ---- Bearing outer: fixed ABD ----
bearing_outer_mesh = io.read(str(bearing_outer_obj))
label_surface(bearing_outer_mesh)
AffineBodyConstitution().apply_to(bearing_outer_mesh, 200.0 * MPa)
bearing_contact.apply_to(bearing_outer_mesh)
scene.objects().create("bearing_outer").geometries().create(bearing_outer_mesh)

# ---- Bearing spheres: split into multiple ABD bodies by region ----
bearing_spheres_mesh = io.read(str(bearing_spheres_obj))
label_surface(bearing_spheres_mesh)
label_region(bearing_spheres_mesh)
sphere_regions = apply_region(bearing_spheres_mesh)
for i, sphere_mesh in enumerate(sphere_regions):
    label_surface(sphere_mesh)
    AffineBodyConstitution().apply_to(sphere_mesh, 200.0 * MPa)
    bearing_contact.apply_to(sphere_mesh)
    scene.objects().create(f"bearing_sphere_{i}").geometries().create(sphere_mesh)

# ---- Init & GUI ----
world.init(scene)
world.retrieve()

ps.init()
gui = SceneGUI(scene)
scene_io = SceneIO(scene)
scene_io.write_surface(f"{workspace}/yoyo_throw_act_revolve_{world.frame()}.obj")
gui.register()

running = False


def on_update():
    global running

    if imgui.Button("Run / Pause"):
        running = not running

    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")
    imgui.SameLine()
    imgui.Text(f"Time: {world.frame() * DT:.3f} s")

    if running:
        world.advance()
        world.dump()
        world.retrieve()
        scene_io.write_surface(f"{workspace}/yoyo_throw_act_revolve_{world.frame()}.obj")
        gui.update()


ps.set_user_callback(on_update)
ps.show()
