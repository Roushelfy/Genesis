"""
Stage-1 yoyo throw (scene from 0_tight_yoyo_string_2):
- Load yoyo ball + string + bearing outer + bearing spheres.
- Yoyo ball is free (not fixed).
- bearing_spheres.obj is split by regions into independent ABD bodies.
"""

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Animation, Engine, Logger, Scene, SceneIO, Timer, Transform, World, builtin, view, AngleAxis
from uipc.constitution import (
    AffineBodyConstitution,
    HookeanSpring,
    KirchhoffRodBending,
    RotatingMotor,
    SoftPositionConstraint,
)
from uipc.geometry import (
    SimplicialComplexIO,
    apply_region,
    label_region,
    label_surface,
    mesh_partition,
)
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

from asset_dir import AssetDir

Logger.set_level(Logger.Level.Info)
Timer.enable_all()

STRING_TOP_UP_SPEED = 5.0

workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)

ball_obj = folder / "results" / "v1" /"yoyo-ball.obj"
string_obj = folder / "results" / "v1" /"yoyo_string_1669.obj"

engine = Engine("cuda", str(workspace))
world = World(engine)

dt = 0.001
config = Scene.default_config()
config["dt"] = dt
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
# config["integrator"]["type"] = "bdf2" # BDF2 integrator for better kinetic energy conservation
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

# Free yoyo ball (not fixed)
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
    motor_rot_vel=2.0 * np.pi * 10,
)

# # Optional initial downward velocity
# velocity = ball_mesh.instances().find(builtin.velocity)
# t = Transform.Identity()
# t.translate(np.array([0.0, -0.01, 0.0], dtype=np.float64))
# aa = AngleAxis(0.5 * np.pi, np.array([0.0, 0.0, 1], dtype=np.float64))
# t.rotate(aa)
# view(velocity)[0] = (t.matrix() - Transform.Identity().matrix()) * 10

ball_obj_node = scene.objects().create("yoyo_ball")
ball_obj_node.geometries().create(ball_mesh)


def rotate_yoyo_ball(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    RotatingMotor.animate(geo, info.dt())


scene.animator().insert(ball_obj_node, rotate_yoyo_ball)

# String: top endpoint fixed
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


def pull_string_segment_up(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    is_constrained = view(geo.vertices().find(builtin.is_constrained))
    aim_position = view(geo.vertices().find(builtin.aim_position))
    cur_positions = np.array(view(geo.positions()), copy=False).reshape(-1, 3)

    is_constrained[:] = 0
    t_now = info.frame() * info.dt()
    pull_start = 1.2
    pull_end = 1.25
    if pull_start <= t_now < pull_end:
        max_vid = min(51, cur_positions.shape[0])
        for vid in range(1, max_vid):
            is_constrained[vid] = 1
            target = cur_positions[vid].copy()
            target[1] += STRING_TOP_UP_SPEED * info.dt()
            aim_position[vid] = target.reshape(3, 1)


scene.animator().insert(string_obj_node, pull_string_segment_up)

world.init(scene)
world.retrieve()

ps.init()
gui = SceneGUI(scene)
scene_io = SceneIO(scene)
scene_io.write_surface(f"{workspace}/yoyo_throw_{world.frame()}.obj")
gui.register()

running = False

world.recover(1199)
world.retrieve()

def on_update():
    global running

    if imgui.Button("Run / Pause"):
        running = not running

    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")
    imgui.SameLine()
    imgui.Text(f"Time: {world.frame() * dt:.3f} s")

    if running:
        world.advance()
        world.dump()
        world.retrieve()
        scene_io.write_surface(f"{workspace}/yoyo_throw_{world.frame()}.obj")
        gui.update()


ps.set_user_callback(on_update)
ps.show()
