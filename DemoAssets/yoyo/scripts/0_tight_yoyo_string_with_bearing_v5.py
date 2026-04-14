"""
Stage-0 yoyo preparation:
Wrap the string around the yoyo ball with constant angular velocity.
All vertices farther than a threshold from the world origin are constrained
and rotated at uniform angular velocity around the Z-axis through ball_center.
Vertices close to the origin (< threshold) are left free.
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
ANGULAR_SPEED = 4.0  # angular velocity around Z-axis (rad/s)
OUTWARD_SPEED = 3.2  # radial outward pull speed for tension (m/s)
INWARD_SPEED = 0.02  # radial inward speed (m/s)
FREE_THRESHOLD = 0.05  # vertices closer than this to world origin are free
# ────────────────────────────────────────────────────────────────────

Logger.set_level(Logger.Level.Warn)
Timer.enable_all()

workspace = AssetDir.output_path(__file__)
folder = AssetDir.folder(__file__)
_YOYO_DIR = folder.parent

ball_obj = _YOYO_DIR / "v3" / "yoyo-ball.obj"
string_obj = _YOYO_DIR / "v5" / "yoyo-string.obj"
bearing_outer_obj = _YOYO_DIR / "v3" / "bearing_outer.obj"
bearing_spheres_obj = _YOYO_DIR / "v3" / "bearing_spheres.obj"

engine = Engine("cuda", str(workspace))
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.002
config["gravity"] = [[0.0], [0.0], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.0001
config["newton"]["velocity_tol"] = 0.1
config["newton"]["transrate_tol"] = 10
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

bearing_outer_mesh = io.read(str(bearing_outer_obj))
label_surface(bearing_outer_mesh)
AffineBodyConstitution().apply_to(bearing_outer_mesh, 200.0 * MPa)
bearing_contact.apply_to(bearing_outer_mesh)
view(bearing_outer_mesh.instances().find(builtin.is_fixed))[0] = 1
scene.objects().create("bearing_outer").geometries().create(bearing_outer_mesh)

bearing_spheres_mesh = io.read(str(bearing_spheres_obj))
label_surface(bearing_spheres_mesh)
AffineBodyConstitution().apply_to(bearing_spheres_mesh, 200.0 * MPa)
bearing_contact.apply_to(bearing_spheres_mesh)
view(bearing_spheres_mesh.instances().find(builtin.is_fixed))[0] = 1
scene.objects().create("bearing_spheres").geometries().create(bearing_spheres_mesh)


string_mesh = io.read(str(string_obj))
label_surface(string_mesh)
HookeanSpring().apply_to(string_mesh, 1 * GPa, thickness=0.0004, mass_density=100.0)
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
    "angular_speed": ANGULAR_SPEED,
    "outward_speed": OUTWARD_SPEED,
    "inward_speed": INWARD_SPEED,
    "free_threshold": FREE_THRESHOLD,
    "stopped": False,
    "save_path": str(_YOYO_DIR / ""),
    "constrained_mask": None,
}
# ────────────────────────────────────────────────────────────────────


def wrap_string_end(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()

    is_constrained = view(geo.vertices().find(builtin.is_constrained))
    aim_position = view(geo.vertices().find(builtin.aim_position))
    cur_positions = np.array(view(geo.positions()), copy=False).reshape(-1, 3)

    n_verts = cur_positions.shape[0]
    is_constrained[:] = 0

    if ui["stopped"]:
        return

    dt = info.dt()
    axle = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    threshold = ui["free_threshold"]

    omega = ui["angular_speed"]
    theta = omega * dt
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array(
        [[cos_t, -sin_t, 0.0],
         [sin_t,  cos_t, 0.0],
         [0.0,    0.0,   1.0]],
        dtype=np.float64,
    )

    mask = np.zeros(n_verts, dtype=np.float64)

    for vid in range(n_verts):
        cur = cur_positions[vid].copy()

        if np.linalg.norm(cur) < threshold:
            continue

        rel = cur - ball_center
        target = ball_center + rot @ rel

        rel_perp = rel - axle * np.dot(rel, axle)
        rel_norm = float(np.linalg.norm(rel_perp))
        if rel_norm > 1e-9:
            radial = rel_perp / rel_norm
            target += radial * (ui["outward_speed"] * dt)
            target -= radial * (ui["inward_speed"] * dt)

        is_constrained[vid] = 1
        aim_position[vid] = target.reshape(3, 1)
        mask[vid] = 1.0

    ui["constrained_mask"] = mask

    if mask.sum() == 0:
        ui["stopped"] = True
        print(f"[auto-stop] All vertices unconstrained at frame {info.frame()}, stopping.")


scene.animator().insert(string_obj_node, wrap_string_end)

world.init(scene)
world.retrieve()

ps.init()
sgui = SceneGUI(scene)
scene_io = SceneIO(scene)
scene_io.write_surface(f"{workspace}/yoyo_tight_{world.frame()}.obj")
sgui.register()

running = False
ui["recover_frame"] = 0
ui["skip_start_verts"] = 0
ui["stop_at_frame"] = 0

_string_curve_name = ""
for name in ["linemesh", "yoyo_string", "yoyo_string_0"]:
    try:
        ps.get_curve_network(name)
        _string_curve_name = name
        break
    except Exception:
        pass
if _string_curve_name:
    print(f"[gui] String curve network: '{_string_curve_name}'")
else:
    print("[gui] WARNING: string curve network not found for skip highlighting")


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

    imgui.Text(f"Frame: {world.frame()}")

    changed, val = imgui.InputInt("Stop At Frame", ui["stop_at_frame"])
    if changed:
        ui["stop_at_frame"] = max(0, val)

    imgui.Separator()
    changed, val = imgui.SliderFloat("Angular Speed (rad/s)", ui["angular_speed"], -20.0, 20.0)
    if changed:
        ui["angular_speed"] = val
    changed, val = imgui.SliderFloat("Outward Speed", ui["outward_speed"], 0.0, 5.0)
    if changed:
        ui["outward_speed"] = val
    changed, val = imgui.SliderFloat("Inward Speed", ui["inward_speed"], 0.0, 1.0)
    if changed:
        ui["inward_speed"] = val
    changed, val = imgui.SliderFloat("Free Threshold", ui["free_threshold"], 0.0, 0.5)
    if changed:
        ui["free_threshold"] = val

    # ── Recover ──
    imgui.Separator()
    imgui.Text("Recover")
    changed, val = imgui.InputInt("Recover Frame", ui["recover_frame"])
    if changed:
        ui["recover_frame"] = max(0, val)
    if imgui.Button("Recover"):
        target = ui["recover_frame"]
        if world.recover(target):
            world.retrieve()
            sgui.update()
            print(f"[recover] recovered to frame {target}")
        else:
            print(f"[recover] failed to recover to frame {target}")

    # ── Export ──
    imgui.Separator()
    changed, val = imgui.InputInt("Skip Start Vertices", ui["skip_start_verts"])
    if changed:
        ui["skip_start_verts"] = max(0, val)

    changed, new_path = imgui.InputText("Save Path", ui["save_path"])
    if changed:
        ui["save_path"] = new_path
    if imgui.Button("Save String"):
        save_dir = _YOYO_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        path = ui["save_path"]
        out_file = f"{path}/yoyo_string_{world.frame()}.obj"

        geo = string_geo_slot[0].geometry()
        skip = ui["skip_start_verts"]
        if skip > 0:
            positions = np.array(view(geo.positions()), copy=False).reshape(-1, 3)
            edges = np.array(view(geo.edges().topo()), copy=False).reshape(-1, 2)
            keep_mask = np.ones(positions.shape[0], dtype=bool)
            keep_mask[: min(skip, positions.shape[0])] = False
            old_to_new = np.full(positions.shape[0], -1, dtype=np.int32)
            old_to_new[keep_mask] = np.arange(int(keep_mask.sum()), dtype=np.int32)
            new_positions = positions[keep_mask]
            valid_edges = np.all(np.isin(edges, np.nonzero(keep_mask)[0]), axis=1)
            new_edges = old_to_new[edges[valid_edges]]

            from uipc.geometry import linemesh

            trimmed = linemesh(new_positions, new_edges)
            sio = SimplicialComplexIO()
            sio.write(out_file, trimmed)
            print(f"[save] wrote {out_file} (skipped {skip} start verts, {new_positions.shape[0]} remaining)")
        else:
            sio = SimplicialComplexIO()
            sio.write(out_file, geo)
            print(f"[save] wrote {out_file}")

    if running:
        stop_frame = ui["stop_at_frame"]
        if stop_frame > 0 and world.frame() >= stop_frame:
            running = False
            ui["stopped"] = True
            print(f"[auto-stop] Reached stop frame {stop_frame}")
            return

        world.advance()
        world.retrieve()
        world.dump()
        sgui.update()

        mask = ui["constrained_mask"]
        if _string_curve_name and mask is not None:
            cn = ps.get_curve_network(_string_curve_name)
            colors = np.zeros((len(mask), 3), dtype=np.float64)
            colors[mask > 0.5] = [1.0, 0.2, 0.2]  # red = constrained
            colors[mask < 0.5] = [0.2, 0.5, 1.0]   # blue = free
            skip = ui["skip_start_verts"]
            if skip > 0:
                colors[: min(skip, len(mask))] = [0.9, 0.9, 0.1]  # yellow = will be skipped on save
            cn.add_color_quantity("constraint", colors, defined_on="nodes", enabled=True)


ps.set_user_callback(on_update)
ps.show()
