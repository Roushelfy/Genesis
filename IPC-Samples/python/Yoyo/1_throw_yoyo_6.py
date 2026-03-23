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
THROW_SPEED = 0.0        # initial downward speed (m/s)
THROW_DURATION = 0.00    # how long to apply the push (seconds)
STRING_TOP_UP_SPEED = 1.0  # top control segment upward speed (m/s)
# ────────────────────────────────────────────────────────────────────

Logger.set_level(Logger.Level.Info)
Timer.enable_all()

workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)

ball_obj = folder / "results" / "yoyo-ball.obj"
string_obj = folder / "results" / "yoyo_string_1408.obj"
bearing_obj = folder / "results" / "yoyo-bear.obj"
finger_obj = folder / "results" / "finger.obj"

engine = Engine("cuda", str(workspace))
world = World(engine)

dt = 0.002
config = Scene.default_config()
config["dt"] = dt
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["integrator"]["type"] = "bdf1" # BDF2 integrator for better kinetic energy conservation
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.0001
config["newton"]["velocity_tol"] = 0.5
config["newton"]["transrate_tol"] = 0.1
# config["collision_detection"]["method"] = "stackless_bvh"
scene = Scene(config)

scene.contact_tabular().default_model(0.3, 800.0 * MPa)
bearing_contact = scene.contact_tabular().create("bearing")
ball_contact = scene.contact_tabular().create("ball")
string_contact = scene.contact_tabular().create("string")
finger_contact = scene.contact_tabular().create("finger")
scene.contact_tabular().insert(ball_contact, string_contact, 0.15, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(string_contact, string_contact, 0.10, 100.0 * MPa, enable=True)
scene.contact_tabular().insert(bearing_contact, bearing_contact, 0.0, 800.0 * MPa, enable=False)
scene.contact_tabular().insert(bearing_contact, ball_contact, 0.05, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(bearing_contact, string_contact, 0.5, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(finger_contact, ball_contact, 0.3, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(finger_contact, string_contact, 0.3, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(finger_contact, bearing_contact, 0.0, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(finger_contact, finger_contact, 0.0, 800.0 * MPa, enable=False)

io = SimplicialComplexIO()

ball_mesh = io.read(str(ball_obj))
label_surface(ball_mesh)
AffineBodyConstitution().apply_to(ball_mesh, 100.0 * MPa, mass_density=100.0)
ball_contact.apply_to(ball_mesh)
motor = RotatingMotor()
motor.apply_to(
    ball_mesh,
    strength=100.0,
    motor_axis=np.array([[0.0], [0.0], [1.0]], dtype=np.float64),
    motor_rot_vel= 2.0 * np.pi * 10,
)
velocity = ball_mesh.instances().find(builtin.velocity)
t = Transform.Identity()
t.translate(np.array([0.0, -0.01, 0.0], dtype=np.float64))
aa = AngleAxis(0.5 * np.pi, np.array([0.0, 0.0, 1], dtype=np.float64))
t.rotate(aa)
view(velocity)[0] = (t.matrix() - Transform.Identity().matrix()) * 10

ball_obj_node = scene.objects().create("yoyo_ball")
ball_obj_node.geometries().create(ball_mesh)

def rotate_yoyo_ball(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    RotatingMotor.animate(geo, info.dt())

scene.animator().insert(ball_obj_node, rotate_yoyo_ball)

bearing_mesh = io.read(str(bearing_obj))
label_surface(bearing_mesh)
AffineBodyConstitution().apply_to(bearing_mesh, 100.0 * MPa, mass_density=400.0)
bearing_contact.apply_to(bearing_mesh)

finger_mesh = io.read(str(finger_obj))
label_surface(finger_mesh)
AffineBodyShell().apply_to(finger_mesh, 100.0 * MPa, thickness=0.001)
finger_contact.apply_to(finger_mesh)
finger_stc = SoftTransformConstraint()
finger_stc.apply_to(finger_mesh, np.array([100.0, 100.0], dtype=np.float64))


string_mesh = io.read(str(string_obj))
label_surface(string_mesh)
HookeanSpring().apply_to(string_mesh, 5000.0 * MPa, thickness=0.0002, mass_density=300.0)
KirchhoffRodBending().apply_to(string_mesh, 1.0e4)
SoftPositionConstraint().apply_to(string_mesh, 200.0)
string_contact.apply_to(string_mesh)
mesh_partition(string_mesh, 8)

is_fixed = string_mesh.vertices().find(builtin.is_fixed)
view(is_fixed)[0] = 1

string_obj_node = scene.objects().create("yoyo_string")
string_obj_node.geometries().create(string_mesh)

bearing_obj_node = scene.objects().create("yoyo_bearing")
bearing_obj_node.geometries().create(bearing_mesh)

finger_obj_node = scene.objects().create("finger_tube")
finger_obj_node.geometries().create(finger_mesh)


def move_finger_tube(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    rest_geo = info.rest_geo_slots()[0].geometry()
    is_constrained = view(geo.instances().find(builtin.is_constrained))
    aim_transform = view(geo.instances().find(builtin.aim_transform))

    t_now = info.frame() * info.dt()
    move1_start = 0.5
    move1_end = 0.8
    move2_end = 1.0
    if t_now <= move1_start:
        alpha1 = 0.0
    elif t_now >= move1_end:
        alpha1 = 1.0
    else:
        alpha1 = (t_now - move1_start) / (move1_end - move1_start)

    if t_now <= move1_end:
        alpha2 = 0.0
    elif t_now >= move2_end:
        alpha2 = 1.0
    else:
        alpha2 = (t_now - move1_end) / (move2_end - move1_end)

    direction1 = np.array([1.0, -0.7, 0.0], dtype=np.float64)
    direction1 /= np.linalg.norm(direction1)
    offset1 = direction1 * 0.10 * alpha1
    offset2 = np.array([-0.08 * alpha2, 0.0, 0.0], dtype=np.float64)
    offset = offset1 + offset2

    rest_tf = Transform(rest_geo.transforms().view()[0])
    rest_pos = rest_tf.translation()
    aim = Transform.Identity()
    aim.translate(
        Vector3.Values(
            [
                rest_pos[0][0] + offset[0],
                rest_pos[1][0] + offset[1],
                rest_pos[2][0] + offset[2],
            ]
        )
    )
    is_constrained[0] = 1
    aim_transform[0] = aim.matrix()


scene.animator().insert(finger_obj_node, move_finger_tube)

# def pull_string_segment_up(info: Animation.UpdateInfo):
#     geo = info.geo_slots()[0].geometry()

#     is_constrained = view(geo.vertices().find(builtin.is_constrained))
#     aim_position = view(geo.vertices().find(builtin.aim_position))
#     cur_positions = np.array(view(geo.positions()), copy=False).reshape(-1, 3)

#     is_constrained[:] = 0
#     t_now = info.frame() * info.dt()
#     pull_start = 0.6
#     pull_end = 0.7
#     if pull_start <= t_now < pull_end:
#         max_vid = min(51, cur_positions.shape[0])
#         for vid in range(1, max_vid):
#             is_constrained[vid] = 1
#             target = cur_positions[vid].copy()
#             target[1] += STRING_TOP_UP_SPEED * info.dt()
#             aim_position[vid] = target.reshape(3, 1)


# scene.animator().insert(string_obj_node, pull_string_segment_up)

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


    if running:
        world.advance()
        world.retrieve()
        # world.dump()
        scene_io.write_surface(f"{workspace}/yoyo_throw_{world.frame()}.obj")
        gui.update()
        imgui.Text(f"Frame: {world.frame()}")
        imgui.SameLine()
        imgui.Text(f"Time: {world.frame() * dt:.3f} s")
        Timer.report()
    
    if world.frame() >= 4000:
        running = False


ps.set_user_callback(on_update)
ps.show()
