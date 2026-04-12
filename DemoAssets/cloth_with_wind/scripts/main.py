import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import Logger, Timer, Scene, World, Engine, Animation, view
import uipc.builtin as builtin
from uipc.geometry import trimesh, label_surface, mesh_partition, ground
from uipc.constitution import (
    StrainLimitingBaraffWitkinShell,
    DiscreteShellBending,
    ElasticModuli2D,
    FiniteElementExternalForce,
)
from uipc.gui import SceneGUI
from uipc.unit import kPa

from asset_dir import AssetDir

Logger.set_level(Logger.Level.Info)
Timer.enable_all()

output_path = AssetDir.output_path(__file__)

engine = Engine("cuda", output_path)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.01
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = False
config["newton"]["velocity_tol"] = 0.1
config["contact"]["d_hat"] = 0.002
config["linear_system"]["tol_rate"] = 1e-3
scene = Scene(config)

scene.contact_tabular().default_model(0.5, 1e9)
default_contact = scene.contact_tabular().default_element()

# Build a 20x20 grid cloth mesh
N = 20
cloth_size = 1.0
spacing = cloth_size / N

Vs = []
for i in range(N + 1):
    for j in range(N + 1):
        Vs.append([i * spacing, 0.5, j * spacing])

Fs = []
for i in range(N):
    for j in range(N):
        v00 = i * (N + 1) + j
        v10 = (i + 1) * (N + 1) + j
        v01 = i * (N + 1) + (j + 1)
        v11 = (i + 1) * (N + 1) + (j + 1)
        Fs.append([v00, v10, v11])
        Fs.append([v00, v11, v01])

Vs = np.array(Vs, dtype=np.float64)
Fs = np.array(Fs, dtype=np.int32)

cloth_mesh = trimesh(Vs, Fs)
label_surface(cloth_mesh)
mesh_partition(cloth_mesh)

# Apply shell constitution
slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()
cloth_moduli = ElasticModuli2D.youngs_poisson(60 * kPa, 0.49)
slbws.apply_to(cloth_mesh, moduli=cloth_moduli, mass_density=200, thickness=0.001)
dsb.apply_to(cloth_mesh, bending_stiffness=10.0)

# Apply external force constitution (initially zero)
ext_force = FiniteElementExternalForce()
ext_force.apply_to(cloth_mesh, np.array([0.0, 0.0, 0.0]))

default_contact.apply_to(cloth_mesh)

# Fix one side: all vertices with i == 0 (the x == 0 edge)
is_fixed = cloth_mesh.vertices().find(builtin.is_fixed)
is_fixed_view = view(is_fixed)
for j in range(N + 1):
    vid = 0 * (N + 1) + j  # i == 0
    is_fixed_view[vid] = 1

cloth_obj = scene.objects().create("cloth")
cloth_obj.geometries().create(cloth_mesh)

# Ground plane below the cloth
ground_obj = scene.objects().create("ground")
g = ground(-0.1)
default_contact.apply_to(g)
ground_obj.geometries().create(g)

# Wind animator: apply random forces to vertices each frame
rng = np.random.default_rng(42)

def animate_wind(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    n_verts = (N + 1) * (N + 1)

    is_constrained = geo.vertices().find(builtin.is_constrained)
    is_constrained_view = view(is_constrained)
    for k in range(n_verts):
        is_constrained_view[k] = 1

    force_attr = geo.vertices().find("external_force")
    force_view = view(force_attr)

    # Base wind direction (+x with slight lift) plus per-vertex noise
    base_wind = np.array([0.01, 0.01, 0.0])
    noise = rng.normal(0, 0.008, size=(n_verts, 3))
    forces = base_wind + noise

    for k in range(n_verts):
        force_view[k] = forces[k].reshape(3, 1)

scene.animator().insert(cloth_obj, animate_wind)

world.init(scene)

ps.init()
ps.set_ground_plane_height(-0.1)
sgui = SceneGUI(scene, 'split')
sgui.register()
sgui.set_edge_width(1.0)

run = False

def on_update():
    global run
    if imgui.Button('run & stop'):
        run = not run

    if run:
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
