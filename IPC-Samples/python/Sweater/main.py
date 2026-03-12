from pathlib import Path

import uipc
import polyscope as ps
from polyscope import imgui
from uipc import Engine, Logger, Scene, Timer, Transform, Vector3, World
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from uipc.geometry import SimplicialComplexIO, ground, label_surface
from uipc.gui import SceneGUI
from uipc.unit import MPa

Logger.set_level(Logger.Level.Warn)
Timer.enable_all()

repo = Path(__file__).resolve().parents[3]
workspace = repo / "IPC-Samples" / "output" / "SweaterIPCOnly"
workspace.mkdir(parents=True, exist_ok=True)
sweater_obj = repo / "DemoAssets" / "Sweater-Out.obj"

engine = Engine("cuda", str(workspace))
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.005
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = 0.001
config["newton"]["semi_implicit"] = True
config["newton"]["velocity_tol"] = 1
config["newton"]["transrate_tol"] = 10
config["sanity_check"]["enable"] = True
scene = Scene(config)

scene.contact_tabular().default_model(0.5, 1000.0 * MPa)
cloth_contact = scene.contact_tabular().create("cloth")
ground_contact = scene.contact_tabular().create("ground")
scene.contact_tabular().insert(cloth_contact, cloth_contact, 0.05, 10.0 * MPa, enable=True)
scene.contact_tabular().insert(cloth_contact, ground_contact, 0.5, 1000.0 * MPa, enable=True)

pre = Transform.Identity()
pre.translate(Vector3.Values([0.0, -0.5, 0.0]))
io = SimplicialComplexIO(pre)
cloth_mesh = io.read(str(sweater_obj))
label_surface(cloth_mesh)
uipc.geometry.mesh_partition(cloth_mesh)

shell = StrainLimitingBaraffWitkinShell()
bending = DiscreteShellBending()
shell.apply_to(
    cloth_mesh,
    moduli=ElasticModuli2D.youngs_poisson(8e3, 0.45),
    mass_density=200.0,
    thickness=0.0005,
)
bending.apply_to(cloth_mesh, bending_stiffness=37.0)
cloth_contact.apply_to(cloth_mesh)

cloth_obj = scene.objects().create("sweater_cloth")
cloth_obj.geometries().create(cloth_mesh)

ground_obj = scene.objects().create("ground")
ground_mesh = ground(-1.0)
ground_contact.apply_to(ground_mesh)
ground_obj.geometries().create(ground_mesh)

world.init(scene)

ps.init()
gui = SceneGUI(scene)
gui.register()

running = False


def on_update():
    global running
    if imgui.Button("Run / Pause"):
        running = not running
    imgui.Text(f"Frame: {world.frame()}")
    if running:
        world.advance()
        world.retrieve()
        world.dump()
        Timer.report()
        gui.update()


ps.set_user_callback(on_update)
ps.show()
