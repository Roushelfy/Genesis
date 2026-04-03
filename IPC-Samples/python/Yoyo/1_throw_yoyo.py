"""
Stage-1 yoyo throw:
Load the wrapped yoyo (ball + string from stage-0 output).
Fix the string top end (vertex 0), give the ball an initial downward velocity,
and release under gravity.
"""

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Animation, Engine, Logger, Scene, SceneIO, Timer, Transform, Vector3, World, builtin, view, AngleAxis
from uipc.constitution import (
    AffineBodyConstitution,
    HookeanSpring,
    KirchhoffRodBending,
    RotatingMotor,
    SoftPositionConstraint,
    SoftTransformConstraint,
    AffineBodyShell,
)
from uipc.geometry import SimplicialComplexIO, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc.unit import MPa

from asset_dir import AssetDir

# ── Tunable parameters (overridable via GUI) ────────────────────────
THROW_SPEED = 0.0  # initial downward speed (m/s)
THROW_DURATION = 0.00  # how long to apply the push (seconds)
STRING_TOP_UP_SPEED = 0.02  # top control point upward speed (m/s)
# ────────────────────────────────────────────────────────────────────

Logger.set_level(Logger.Level.Warn)
Timer.enable_all()

workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)
_YOYO_DIR = folder.parents[2] / "DemoAssets" / "yoyo"

ball_obj = _YOYO_DIR / "yoyo-ball.obj"
string_obj = _YOYO_DIR / "yoyo_string_2750.obj"

engine = Engine("cuda", str(workspace))
world = World(engine)

dt = 0.001
config = Scene.default_config()
config["dt"] = dt
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.0002
config["newton"]["velocity_tol"] = 0.05
config["newton"]["transrate_tol"] = 0.1
scene = Scene(config)

scene.contact_tabular().default_model(0.05, 800.0 * MPa)
ball_contact = scene.contact_tabular().create("ball")
string_contact = scene.contact_tabular().create("string")
scene.contact_tabular().insert(ball_contact, string_contact, 0.5, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(string_contact, string_contact, 0.0, 800.0 * MPa, enable=True)

io = SimplicialComplexIO()

ball_mesh = io.read(str(ball_obj))
label_surface(ball_mesh)
AffineBodyConstitution().apply_to(ball_mesh, 100.0 * MPa, mass_density=400.0)
ball_contact.apply_to(ball_mesh)
motor = RotatingMotor()
# motor.apply_to(
#     ball_mesh,
#     strength=100.0,
#     motor_axis=np.array([[0.0], [0.0], [1.0]], dtype=np.float64),
#     motor_rot_vel=-2.0 * np.pi * 10,
# )
velocity = ball_mesh.instances().find(builtin.velocity)
t = Transform.Identity()
t.translate(np.array([0.0, -0.01, 0.0], dtype=np.float64))
aa = AngleAxis(0.5 * np.pi, np.array([0.0, 0.0, 1], dtype=np.float64))
t.rotate(aa)
view(velocity)[0] = (t.matrix() - Transform.Identity().matrix()) * 50

string_mesh = io.read(str(string_obj))
label_surface(string_mesh)
HookeanSpring().apply_to(string_mesh, 5000.0 * MPa, thickness=0.0004, mass_density=300.0)
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


# def rotate_yoyo_ball(info: Animation.UpdateInfo):
#     geo = info.geo_slots()[0].geometry()
#     view(geo.instances().find(builtin.is_constrained))[0] = 1
#     RotatingMotor.animate(geo, info.dt())


# scene.animator().insert(ball_obj_node, rotate_yoyo_ball)

world.init(scene)
world.retrieve()

ps.init()
gui = SceneGUI(scene)
scene_io = SceneIO(scene)
scene_io.write_surface(f"{workspace}/yoyo_throw_{world.frame()}.obj")
gui.register()

running = False


def on_update():
    global running

    if imgui.Button("Run / Pause"):
        running = not running

    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")
    imgui.Text(f"Time: {world.frame() * dt:.3f} s")

    if running:
        world.advance()
        world.retrieve()
        world.dump()
        scene_io.write_surface(f"{workspace}/yoyo_throw_{world.frame()}.obj")
        gui.update()


ps.set_user_callback(on_update)
ps.show()
