import os
import pathlib

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, builtin
from uipc import Vector3, Transform
from uipc import Engine, World, Scene, Animation, Timer, SceneIO
from uipc.geometry import linemesh, label_surface, ground, mesh_partition, SimplicialComplexIO, SimplicialComplexSlot
from uipc.constitution import HookeanSpring, KirchhoffRodBending, AffineBodyConstitution, AffineBodyShell, SoftTransformConstraint
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa

from asset_dir import AssetDir

Logger.set_level(Logger.Level.Warn)
Timer.enable_all()

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.005
config["contact"]["d_hat"] = 0.001
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = False
config["newton"]["semi_implicit"] = True         
config["newton"]["velocity_tol"] = 1
config["newton"]["transrate_tol"] = 10
config["linear_system"]["tol_rate"] = 1e-4
print(config)
scene = Scene(config)

hs = HookeanSpring()
krb = KirchhoffRodBending()
abd = AffineBodyConstitution()
stc = SoftTransformConstraint()
scene.contact_tabular().default_model(0.01, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()
noodle_elem = scene.contact_tabular().create("noodle")

bending_stiffness = 1e4

# Noodle: line along Y axis, 30 edges => 31 vertices
thickness = 0.001
n_edges = 50
n_verts = n_edges + 1
y_length = 0.2  # length in Y

noodles_obj = scene.objects().create("noodles")
grid_nx, grid_nz = 30, 30
spacing = 0.006
xz_perturbation = 0.001  # small random offset along x and z per vertex
# Center of noodles grid at world origin in XZ
x_center = (grid_nx - 1) * spacing / 2.0
z_center = (grid_nz - 1) * spacing / 2.0

for ix in range(grid_nx):
    for iz in range(grid_nz):
        x = ix * spacing - x_center
        z = iz * spacing - z_center
        Vs = np.zeros((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            dx = np.random.uniform(-xz_perturbation, xz_perturbation)
            dz = np.random.uniform(-xz_perturbation, xz_perturbation)
            Vs[i] = [x + dx, i * (y_length / n_edges), z + dz]
        Es = np.array([[i, i + 1] for i in range(n_edges)], dtype=np.int32)
        mesh = linemesh(Vs, Es)
        label_surface(mesh)
        hs.apply_to(mesh, thickness=thickness)
        krb.apply_to(mesh, bending_stiffness)
        default_element.apply_to(mesh)
        mesh_partition(mesh, 16)
        noodles_obj.geometries().create(mesh)

# Ground
ground_height = - y_length / 1.0
ground_obj = scene.objects().create("ground")
g = ground(ground_height)
ground_obj.geometries().create(g)

world.init(scene)

ps.init()
sgui = SceneGUI(scene, 'merge')
sio = SceneIO(scene)
sio.write_surface(f"{workspace}/noodles_{world.frame()}.obj")
sgui.register()

run = False
recover_run = False

def on_update():
    global run
    global recover_run

    if imgui.Button("Run & Stop"):
        run = not run
    
    if imgui.Button("Recover Run"):
        recover_run = True
    
    if run:
        world.advance()
        world.retrieve()
        Timer.report()
        sgui.update()
    
    if recover_run:
        if world.recover(world.frame() + 1):
            world.retrieve()
            sgui.update()
        else:
            recover_run = False

ps.set_user_callback(on_update)
ps.show()
