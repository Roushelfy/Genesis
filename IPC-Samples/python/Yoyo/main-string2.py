"""
This script is used to run the Yoyo simulation with the yoyo-string2.obj, which is better than the yoyo-string.obj.
"""

from pathlib import Path

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
from uipc.geometry import SimplicialComplexIO, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc.unit import MPa
from asset_dir import AssetDir


Logger.set_level(Logger.Level.Warn)
Timer.enable_all()

repo = Path(__file__).resolve().parents[3]
workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)

ball_obj = folder / "results" / "yoyo-ball.obj"
string_obj = folder / "results" / "yoyo-string2.obj"

engine = Engine("cuda", str(workspace))
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.002
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.0002
config["newton"]["velocity_tol"] = 0.1
config["newton"]["transrate_tol"] = 10
scene = Scene(config)

scene.contact_tabular().default_model(0.5, 800.0 * MPa)
ball_contact = scene.contact_tabular().create("ball")
string_contact = scene.contact_tabular().create("string")
scene.contact_tabular().insert(ball_contact, string_contact, 0.25, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(string_contact, string_contact, 0.1, 800.0 * MPa, enable=True)

io = SimplicialComplexIO()

ball_mesh = io.read(str(ball_obj))
label_surface(ball_mesh)
AffineBodyConstitution().apply_to(ball_mesh, 200.0 * MPa)
ball_contact.apply_to(ball_mesh)
motor = RotatingMotor()
motor.apply_to(
    ball_mesh,
    strength=100.0,
    motor_axis=np.array([[0.0], [0.0], [1.0]], dtype=np.float64),
    motor_rot_vel=2.0 * np.pi * 10,
)

string_mesh = io.read(str(string_obj))
label_surface(string_mesh)
HookeanSpring().apply_to(string_mesh, 500.0 * MPa, thickness=0.0004, mass_density=300.0)
KirchhoffRodBending().apply_to(string_mesh, 1.0e4)
SoftPositionConstraint().apply_to(string_mesh, 20.0)
string_contact.apply_to(string_mesh)
mesh_partition(string_mesh, 8)
is_fixed = string_mesh.vertices().find(builtin.is_fixed)
view(is_fixed)[0] = 1

ball_obj_node = scene.objects().create("yoyo_ball")
ball_obj_node.geometries().create(ball_mesh)

string_obj_node = scene.objects().create("yoyo_string")
string_obj_node.geometries().create(string_mesh)


def rotate_yoyo_ball(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    RotatingMotor.animate(geo, info.dt())


scene.animator().insert(ball_obj_node, rotate_yoyo_ball)


def pull_string_segment_up(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    rest_geo = info.rest_geo_slots()[0].geometry()

    is_constrained = view(geo.vertices().find(builtin.is_constrained))
    aim_position = view(geo.vertices().find(builtin.aim_position))
    rest_positions = np.array(view(rest_geo.positions()), copy=False).reshape(-1, 3)

    is_constrained[:] = 0
    if info.frame()*info.dt() >= 0.5:
        jump_up = 0.05
        # Keep vertex 0 pinned by is_fixed; pull vertices 1..50 upward suddenly.
        for vid in range(1, 51):
            is_constrained[vid] = 1
            target = rest_positions[vid].copy()
            target[1] += jump_up
            aim_position[vid] = target.reshape(3, 1)
    if info.frame()*info.dt() >= 0.55:
        is_constrained[:] = 0


scene.animator().insert(string_obj_node, pull_string_segment_up)

world.init(scene)

ps.init()
gui = SceneGUI(scene)
scene_io = SceneIO(scene)
scene_io.write_surface(f"{workspace}/yoyo_{world.frame()}.obj")
gui.register()

running = False


def on_update():
    global running
    if imgui.Button("Run / Pause"):
        running = not running

    if running:
        world.advance()
        world.retrieve()
        world.dump()
        scene_io.write_surface(f"{workspace}/yoyo_{world.frame()}.obj")
        Timer.report()
        gui.update()


ps.set_user_callback(on_update)
ps.show()
