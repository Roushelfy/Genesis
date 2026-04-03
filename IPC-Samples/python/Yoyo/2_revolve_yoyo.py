"""
Stage-2 yoyo revolve / retract:
- Keep yoyo ball spinning.
- Keep string start fixed (vertex 0).
- No string pull animation.
- Animate tube (tuber.obj) retreating away.
Assets are pre-positioned; no pre-transform is applied.
"""

import numpy as np
import polyscope as ps
from polyscope import imgui
import argparse

from uipc import Animation, Engine, Logger, Scene, SceneIO, Timer, Transform, Vector3, World, builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyShell,
    HookeanSpring,
    KirchhoffRodBending,
    RotatingMotor,
    SoftPositionConstraint,
    SoftTransformConstraint,
)
from uipc.geometry import SimplicialComplexIO, apply_region, label_region, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc.unit import GPa, MPa

from asset_dir import AssetDir
from mesh_sequence_tool import MeshSequenceTool

# ---- Tunables ----
DT = 0.001
MOTOR_ROT_VEL = 2.0 * np.pi * 10.0
ENABLE_COLLISION = True
ENABLE_GRAVITY = True
STRING_END_PULL_SPEED = 5

STRING_END_PULL_START = 0.7
STRING_END_PULL_END = 1.1

TUBER_RETREAT_START_0 = 0.4
TUBER_RETREAT_END_0 = 0.6
TUBER_RETREAT_START_1 = 0.6
TUBER_RETREAT_END_1 = 1.1
TUBER_RETREAT_DISTANCE_X_Y = (0.15, 0.1)  # meters, step-1 along +X and +Y
TUBER_RETREAT_DISTANCE_Z = 0.3  # meters, step-2 along +Z
EXPORT_JSON = True

Logger.set_level(Logger.Level.Info)
Timer.enable_all()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yoyo revolve stage.")
    parser.add_argument("--recover-no-gui", action="store_true", help="Run recover-only export mode without GUI.")
    parser.add_argument("--recover-start-frame", type=int, default=0, help="Frame to start recover from.")
    parser.add_argument(
        "--recover-max-frames",
        type=int,
        default=-1,
        help="Max recovered frames to export (-1 means until recover fails).",
    )
    return parser.parse_args()


workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)
_YOYO_DIR = folder.parents[2] / "DemoAssets" / "yoyo"
args = parse_args()

asset_root = _YOYO_DIR / "v2"
ball_obj = asset_root / "yoyo-ball.obj"
string_obj = asset_root / "yoyo-string-revolve.obj"
bearing_outer_obj = asset_root / "bearing_outer.obj"
bearing_spheres_obj = asset_root / "bearing_spheres.obj"
tuber_obj = asset_root / "tuber.obj"
SPLIT_SPHERE_OBJ_PREFIX = "bearing_spheres_"

engine = Engine("cuda", str(workspace))
world = World(engine)

config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [-9.8], [0.0]] if ENABLE_GRAVITY else [[0.0], [0.0], [0.0]]
config["contact"]["enable"] = ENABLE_COLLISION
config["contact"]["friction"]["enable"] = ENABLE_COLLISION
config["contact"]["d_hat"] = 0.0001
config["newton"]["velocity_tol"] = 0.05
config["newton"]["transrate_tol"] = 10
config["collision_detection"]["method"] = "info_stackless_bvh"
scene = Scene(config)

scene.contact_tabular().default_model(0.3, 800.0 * MPa)
ball_contact = scene.contact_tabular().create("ball")
string_contact = scene.contact_tabular().create("string")
bearing_contact = scene.contact_tabular().create("bearing")
tuber_contact = scene.contact_tabular().create("tuber")

scene.contact_tabular().insert(ball_contact, string_contact, 0.7, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(string_contact, string_contact, 0.10, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(bearing_contact, bearing_contact, 0.01, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(bearing_contact, ball_contact, 0.1, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(bearing_contact, string_contact, 0.5, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(tuber_contact, tuber_contact, 0.0, 800.0 * MPa, enable=False)
scene.contact_tabular().insert(tuber_contact, ball_contact, 0.3, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(tuber_contact, string_contact, 0.3, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(tuber_contact, bearing_contact, 0.0, 800.0 * MPa, enable=True)

io = SimplicialComplexIO()

# Yoyo ball: free + spinning motor
ball_mesh = io.read(str(ball_obj))
label_surface(ball_mesh)
AffineBodyConstitution().apply_to(ball_mesh, 200.0 * MPa)
ball_contact.apply_to(ball_mesh)
view(ball_mesh.instances().find(builtin.is_dynamic))[0] = 1

motor = RotatingMotor()
motor.apply_to(
    ball_mesh,
    strength=1.0,
    motor_axis=np.array([[0.0], [0.0], [1.0]], dtype=np.float64),
    motor_rot_vel=MOTOR_ROT_VEL,
)

ball_obj_node = scene.objects().create("yoyo_ball")
ball_geo_slot = ball_obj_node.geometries().create(ball_mesh)[0]


def rotate_yoyo_ball(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    view(geo.instances().find(builtin.is_constrained))[0] = 1
    RotatingMotor.animate(geo, info.dt())


scene.animator().insert(ball_obj_node, rotate_yoyo_ball)

# String: fixed start only, no pull-up animation
string_mesh = io.read(str(string_obj))
label_surface(string_mesh)
HookeanSpring().apply_to(string_mesh, 1.0 * GPa, thickness=0.0004, mass_density=100.0)
KirchhoffRodBending().apply_to(string_mesh, 1.0e3)
SoftPositionConstraint().apply_to(string_mesh, 100.0)
string_contact.apply_to(string_mesh)
mesh_partition(string_mesh, 16)

is_fixed = string_mesh.vertices().find(builtin.is_fixed)
view(is_fixed)[0] = 1

string_obj_node = scene.objects().create("yoyo_string")
string_geo_slot = string_obj_node.geometries().create(string_mesh)[0]


def pull_string_end_diag(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    is_constrained = view(geo.vertices().find(builtin.is_constrained))
    aim_position = view(geo.vertices().find(builtin.aim_position))
    cur_positions = np.array(view(geo.positions()), copy=False).reshape(-1, 3)

    is_constrained[:] = 0
    t_now = info.frame() * info.dt()
    if t_now < STRING_END_PULL_START:
        return

    direction = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    max_vid = min(21, cur_positions.shape[0])
    for vid in range(1, max_vid):
        target = cur_positions[vid].copy() + direction * (STRING_END_PULL_SPEED * info.dt())
        is_constrained[vid] = 1
        aim_position[vid] = target.reshape(3, 1)

    if t_now >= STRING_END_PULL_END:
        is_constrained[:] = 0


scene.animator().insert(string_obj_node, pull_string_end_diag)

# Bearing outer (fixed)
bearing_outer_mesh = io.read(str(bearing_outer_obj))
label_surface(bearing_outer_mesh)
AffineBodyConstitution().apply_to(bearing_outer_mesh, 200.0 * MPa)
bearing_contact.apply_to(bearing_outer_mesh)
bearing_outer_slot = scene.objects().create("bearing_outer").geometries().create(bearing_outer_mesh)[0]

# Bearing spheres split into separate fixed ABD bodies
bearing_spheres_mesh = io.read(str(bearing_spheres_obj))
label_surface(bearing_spheres_mesh)
label_region(bearing_spheres_mesh)
sphere_regions = apply_region(bearing_spheres_mesh)
sphere_slot_map = {}
for i, sphere_mesh in enumerate(sphere_regions):
    io.write(str(asset_root / f"{SPLIT_SPHERE_OBJ_PREFIX}{i}.obj"), sphere_mesh)
    label_surface(sphere_mesh)
    AffineBodyConstitution().apply_to(sphere_mesh, 200.0 * MPa)
    bearing_contact.apply_to(sphere_mesh)
    sphere_slot_map[f"bearing_sphere_{i}"] = (
        scene.objects().create(f"bearing_sphere_{i}").geometries().create(sphere_mesh)[0]
    )

# Tuber: ABD tube with retreat animation
tuber_mesh = io.read(str(tuber_obj))
label_surface(tuber_mesh)
AffineBodyShell().apply_to(tuber_mesh, 100.0 * MPa, thickness=0.001)
tuber_contact.apply_to(tuber_mesh)

stc = SoftTransformConstraint()
stc.apply_to(tuber_mesh, np.array([100.0, 100.0], dtype=np.float64))

tuber_obj_node = scene.objects().create("tuber")
tuber_obj_node.geometries().create(tuber_mesh)


def move_tuber_away(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    rest_geo = info.rest_geo_slots()[0].geometry()
    is_constrained = view(geo.instances().find(builtin.is_constrained))
    aim_transform = view(geo.instances().find(builtin.aim_transform))

    t_now = info.frame() * info.dt()
    if t_now <= TUBER_RETREAT_START_0:
        alpha_x = 0.0
    elif t_now >= TUBER_RETREAT_END_0:
        alpha_x = 1.0
    else:
        alpha_x = (t_now - TUBER_RETREAT_START_0) / (TUBER_RETREAT_END_0 - TUBER_RETREAT_START_0)

    if t_now <= TUBER_RETREAT_START_1:
        alpha_z = 0.0
    elif t_now >= TUBER_RETREAT_END_1:
        alpha_z = 1.0
    else:
        alpha_z = (t_now - TUBER_RETREAT_START_1) / (TUBER_RETREAT_END_1 - TUBER_RETREAT_START_1)

    offset = np.array(
        [
            TUBER_RETREAT_DISTANCE_X_Y[0] * alpha_x,
            TUBER_RETREAT_DISTANCE_X_Y[1] * alpha_x,
            TUBER_RETREAT_DISTANCE_Z * alpha_z,
        ],
        dtype=np.float64,
    )

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


scene.animator().insert(tuber_obj_node, move_tuber_away)

exporter = None
if EXPORT_JSON:
    export_dir = asset_root / "sequence" if args.recover_no_gui else f"{workspace}/frame_json"
    exporter = MeshSequenceTool(
        output_dir=export_dir,
        file_prefix="yoyo_revolve",
    )
    exporter.transform_output_slot("yoyo_ball", ball_geo_slot)
    exporter.transform_output_slot("bearing_outer", bearing_outer_slot)
    for name, slot in sphere_slot_map.items():
        exporter.transform_output_slot(name, slot)
    exporter.position_output_slot(string_geo_slot, key="string_positions", name="yoyo_string")

world.init(scene)
world.retrieve()
if exporter is not None:
    exporter.save(world.frame())

if args.recover_no_gui:
    start_frame = max(int(args.recover_start_frame), world.frame())
    if start_frame > world.frame():
        if not world.recover(start_frame):
            raise SystemExit(f"Recover failed at start frame {start_frame}.")
        if exporter is not None:
            exporter.save(world.frame())

    recovered_count = 0
    while True:
        if args.recover_max_frames >= 0 and recovered_count >= args.recover_max_frames:
            break
        next_frame = world.frame() + 1
        if not world.recover(next_frame):
            break
        if exporter is not None:
            exporter.save(world.frame())
        recovered_count += 1

    print(f"[recover-no-gui] exported {recovered_count} recovered frames to {asset_root / 'sequence'}")
    raise SystemExit(0)

ps.init()
gui = SceneGUI(scene)
scene_io = SceneIO(scene)
scene_io.write_surface(f"{workspace}/yoyo_revolve_{world.frame()}.obj")
gui.register()

running = False


def on_update():
    global running

    if imgui.Button("Run / Pause"):
        running = not running

    imgui.SameLine()
    if imgui.Button("Screenshot"):
        ps.screenshot(f"{workspace}/revolve_shot_{world.frame()}.png")

    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")
    imgui.SameLine()
    imgui.Text(f"Time: {world.frame() * DT:.3f} s")
    imgui.Text(f"Collision: {'ON' if ENABLE_COLLISION else 'OFF'}")
    imgui.Text(f"Gravity: {'ON' if ENABLE_GRAVITY else 'OFF'}")

    if running:
        world.advance()
        world.dump()
        world.retrieve()
        if exporter is not None:
            exporter.save(world.frame())
        scene_io.write_surface(f"{workspace}/yoyo_revolve_{world.frame()}.obj")
        gui.update()


ps.set_user_callback(on_update)
ps.show()
