import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger
from uipc import Engine, World, Scene, Animation
from uipc.geometry import linemesh, label_surface, ground, mesh_partition
from uipc.constitution import HookeanSpring, KirchhoffRodBending
from uipc.gui import SceneGUI

from asset_dir import AssetDir

Logger.set_level(Logger.Level.Info)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.005
config["contact"]["d_hat"] = 0.001
config["contact"]["friction"]["enable"] = True
config["newton"]["semi_implicit"] = True
config["newton"]["velocity_tol"] = 0.1
config["linear_system"]["tol_rate"] = 1e-4
print(config)
scene = Scene(config)

hs = HookeanSpring()
krb = KirchhoffRodBending()
scene.contact_tabular().default_model(0.05, -1)
default_element = scene.contact_tabular().default_element()

bending_stiffness = 1e4

# Noodle: line along Y axis, 30 edges => 31 vertices
thickness = 0.001
n_edges = 50
n_verts = n_edges + 1
y_length = 0.2  # length in Y

noodles_obj = scene.objects().create("noodles")
grid_nx, grid_nz = 1, 1
spacing = 0.005

for ix in range(grid_nx):
    for iz in range(grid_nz):
        x = ix * spacing
        z = iz * spacing
        Vs = np.zeros((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            Vs[i] = [x, i * (y_length / n_edges), z]
        Es = np.array([[i, i + 1] for i in range(n_edges)], dtype=np.int32)
        mesh = linemesh(Vs, Es)
        label_surface(mesh)
        hs.apply_to(mesh, thickness=thickness)
        krb.apply_to(mesh, bending_stiffness)
        default_element.apply_to(mesh)
        mesh_partition(mesh)
        noodles_obj.geometries().create(mesh)

# Ground
ground_height = - y_length / 1.5
ground_obj = scene.objects().create("ground")
g = ground(ground_height)
ground_obj.geometries().create(g)

world.init(scene)

ps.init()
sgui = SceneGUI(scene)
tri_surf, edge_surf, _ = sgui.register()
edge_surf.set_radius(thickness, False)
ps.set_ground_plane_height(ground_height)

gui = {}
gui['run'] = False
gui['recover_frame'] = 0
def on_update():
    imgui.Text(f"Frame: {world.frame()}")
    if imgui.Button("Run & Stop"):
        gui['run'] = not gui['run']
    
    changed, value = imgui.InputInt("Recover Frame", gui['recover_frame'])
    if changed:
        gui['recover_frame'] = value

    if imgui.Button("Recover"):
        world.recover(gui['recover_frame'])
        world.retrieve()
        sgui.update()
    
    if gui['run']:
        world.advance()
        world.retrieve()
        world.dump()
        sgui.update()
    
    if imgui.Button("ONCE_ADVANCE"):
        world.advance()
        world.retrieve()
        world.dump()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()
