"""
Stage-0 yoyo preparation:
Wrap the string end around the yoyo ball while pulling outward to keep tension.
Only vertex 0 (far end) is constrained; the rest follow freely.
"""

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Animation, Engine, Logger, Scene, SceneIO, Timer, World, builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    HookeanSpring,
    KirchhoffRodBending,
    SoftPositionConstraint,
)
from uipc.geometry import SimplicialComplexIO, label_surface, mesh_partition
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa

from asset_dir import AssetDir

# ── Default tunable parameters (overridable via GUI) ────────────────
WRAP_SPEED = 1  # tangential wrapping speed (m/s)
OUTWARD_SPEED = 0.8  # radial outward pull speed for tension (m/s)
INWARD_SPEED = 0.02  # radial inward speed (m/s)
# ────────────────────────────────────────────────────────────────────

Logger.set_level(Logger.Level.Warn)
Timer.enable_all()

workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)
_YOYO_DIR = folder.parents[2] / "DemoAssets" / "yoyo"

ball_obj = _YOYO_DIR / "v1" / "yoyo-ball.obj"
string_obj = _YOYO_DIR / "v1" / "yoyo-string.obj"

engine = Engine("cuda", str(workspace))
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.002
config["gravity"] = [[0.0], [0.0], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.0002
config["newton"]["velocity_tol"] = 0.1
config["newton"]["transrate_tol"] = 10
config["collision_detection"]["method"] = "stackless_bvh"
scene = Scene(config)

scene.contact_tabular().default_model(0.5, 800.0 * MPa)
ball_contact = scene.contact_tabular().create("ball")
string_contact = scene.contact_tabular().create("string")
bearing_contact = scene.contact_tabular().create("bearing")
scene.contact_tabular().insert(ball_contact, string_contact, 0.15, 800.0 * MPa, enable=True)
scene.contact_tabular().insert(string_contact, string_contact, 0.05, 800.0 * MPa, enable=True)

io = SimplicialComplexIO()

ball_mesh = io.read(str(ball_obj))
label_surface(ball_mesh)
AffineBodyConstitution().apply_to(ball_mesh, 200.0 * MPa)
ball_contact.apply_to(ball_mesh)
view(ball_mesh.instances().find(builtin.is_fixed))[0] = 1
view(ball_mesh.instances().find(builtin.is_dynamic))[0] = 0
ball_center = np.array(view(ball_mesh.positions()), copy=False).reshape(-1, 3).mean(axis=0)

string_mesh = io.read(str(string_obj))
label_surface(string_mesh)
HookeanSpring().apply_to(string_mesh, 1 * GPa, thickness=0.0003, mass_density=100.0)
KirchhoffRodBending().apply_to(string_mesh, 1.0e3)
SoftPositionConstraint().apply_to(string_mesh, 100.0)
string_contact.apply_to(string_mesh)
mesh_partition(string_mesh, 16)
view(string_mesh.vertices().find(builtin.is_dynamic))[:] = 1

ball_obj_node = scene.objects().create("yoyo_ball_fixed")
ball_obj_node.geometries().create(ball_mesh)

string_obj_node = scene.objects().create("yoyo_string")
string_geo_slot = string_obj_node.geometries().create(string_mesh)

# ── GUI state ───────────────────────────────────────────────────────
ui = {
    "wrap_speed": WRAP_SPEED,
    "outward_speed": OUTWARD_SPEED,
    "inward_speed": INWARD_SPEED,
    "stopped": False,
    "save_path": str(_YOYO_DIR / ""),
}
# ────────────────────────────────────────────────────────────────────


def wrap_string_end(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()

    is_constrained = view(geo.vertices().find(builtin.is_constrained))
    aim_position = view(geo.vertices().find(builtin.aim_position))
    cur_positions = np.array(view(geo.positions()), copy=False).reshape(-1, 3)

    vid = 0
    is_constrained[:] = 0

    if ui["stopped"]:
        return

    dt = info.dt()
    axle = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cur = cur_positions[vid].copy()

    rel = cur - ball_center
    rel_perp = rel - axle * np.dot(rel, axle)
    rel_norm = float(np.linalg.norm(rel_perp))
    if rel_norm < 1e-9:
        radial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        radial = rel_perp / rel_norm

    tangent_vec = np.cross(axle, radial)
    outward_vec = radial
    inward_vec = -radial

    target = (
        cur
        + tangent_vec * (ui["wrap_speed"] * dt)
        + outward_vec * (ui["outward_speed"] * dt)
        + inward_vec * (ui["inward_speed"] * dt)
    )

    is_constrained[vid] = 1
    aim_position[vid] = target.reshape(3, 1)


scene.animator().insert(string_obj_node, wrap_string_end)

world.init(scene)
world.retrieve()

ps.init()
gui = SceneGUI(scene)
scene_io = SceneIO(scene)
scene_io.write_surface(f"{workspace}/yoyo_tight_{world.frame()}.obj")
gui.register()

running = False


def on_update():
    global running

    if imgui.Button("Run / Pause"):
        running = not running
    imgui.SameLine()
    if not ui["stopped"]:
        if imgui.Button("Stop"):
            ui["stopped"] = True
    else:
        imgui.Text("[Stopped]")

    imgui.Separator()
    changed, val = imgui.SliderFloat("Wrap Speed", ui["wrap_speed"], 0.0, 5.0)
    if changed:
        ui["wrap_speed"] = val
    changed, val = imgui.SliderFloat("Outward Speed", ui["outward_speed"], 0.0, 2.0)
    if changed:
        ui["outward_speed"] = val
    changed, val = imgui.SliderFloat("Inward Speed", ui["inward_speed"], 0.0, 1.0)
    if changed:
        ui["inward_speed"] = val

    imgui.Separator()
    changed, new_path = imgui.InputText("Save Path", ui["save_path"])
    if changed:
        ui["save_path"] = new_path
    if imgui.Button("Save String"):
        save_dir = _YOYO_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        path = ui["save_path"]
        out_file = f"{path}/yoyo_string_{world.frame()}.obj"
        sio = SimplicialComplexIO()
        sio.write(out_file, string_geo_slot[0].geometry())
        print(f"[save] wrote {out_file}")

    if running:
        world.advance()
        world.retrieve()
        # world.dump()
        gui.update()


ps.set_user_callback(on_update)
ps.show()
